"""
Neural network probabilistic forecasting model for flu hospitalizations.

Phase A: Feedforward quantile regression NN on the same engineered features
as XGBoost, providing an apples-to-apples comparison.

Key classes:
- QuantileRegressionNet: PyTorch feedforward network with multi-quantile output
- PinballLoss: Multi-quantile pinball loss function
- NNQuantileDirectForecastEnsemble: Same API as QuantileDirectForecastEnsemble
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src import config


class QuantileRegressionNet(nn.Module):
    """
    Feedforward neural network with multi-quantile output.

    Architecture: Input(n_features) -> 256 -> 128 -> 64 -> n_quantiles
    With BatchNorm + ReLU + Dropout between layers.
    """

    def __init__(self, n_features: int, n_quantiles: int = 5,
                 hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.n_quantiles = n_quantiles
        hidden_dims = hidden_dims or [256, 128, 64]

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_quantiles))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (batch_size, n_quantiles)."""
        return self.network(x)


class PinballLoss(nn.Module):
    """
    Multi-quantile pinball (quantile) loss.

    For quantile q:
        L(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)

    Trains all quantiles simultaneously, averaged across quantiles and samples.
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, n_quantiles)
            targets: (batch_size,) or (batch_size, 1)
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)  # (batch_size, 1)

        diff = targets - predictions  # (batch_size, n_quantiles)
        loss = torch.where(
            diff >= 0,
            self.quantiles * diff,
            (self.quantiles - 1) * diff
        )
        return loss.mean()


class NNQuantileDirectForecastEnsemble:
    """
    Neural network quantile forecast ensemble with the same API as
    QuantileDirectForecastEnsemble.

    4 networks total (one per horizon), each outputting 5 quantiles.
    Uses StandardScaler on features and log1p target transformation.
    """

    def __init__(self, forecast_horizon: int = 4,
                 target_mode: Optional[str] = None,
                 quantiles: Optional[List[float]] = None,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 batch_size: int = 256,
                 max_epochs: int = 300,
                 patience: int = 20,
                 device: Optional[str] = None):
        """
        Args:
            forecast_horizon: Number of weeks ahead (default 4)
            target_mode: Target transformation ("log", "raw", "ratio")
            quantiles: Quantile levels to predict
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Adam learning rate
            batch_size: Training batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
        """
        self.forecast_horizon = forecast_horizon
        self.target_mode = target_mode or getattr(config, 'TARGET_MODE', 'raw')
        self.quantiles = quantiles or getattr(config, 'QUANTILES', [0.05, 0.25, 0.5, 0.75, 0.95])
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Models and scalers (one per horizon)
        self.models: Dict[int, QuantileRegressionNet] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: Optional[List[str]] = None
        self.is_trained = False
        self.training_info = {}

        # Target transformation settings (same as XGBoost ensemble)
        self.ratio_clip_min = getattr(config, 'RATIO_CLIP_MIN', 0.05)
        self.ratio_clip_max = getattr(config, 'RATIO_CLIP_MAX', 10.0)
        self.ratio_min_denominator = getattr(config, 'RATIO_MIN_DENOMINATOR', 5)

    def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features from metadata and encode categoricals.

        Same logic as FluForecastingModel.prepare_features() but with
        shared label encoders across horizons.
        """
        feature_cols = [col for col in data.columns
                        if col not in ['date', 'location', 'location_name', 'value', 'weekly_rate']]

        if self.feature_columns is None:
            self.feature_columns = feature_cols
        else:
            feature_cols = self.feature_columns

        X = data[feature_cols].copy()
        y = data['value'].copy() if 'value' in data.columns else pd.Series(np.zeros(len(data)))

        # Encode categoricals
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].astype(str)
                unseen_mask = ~X[col].isin(self.label_encoders[col].classes_)
                if unseen_mask.any():
                    X.loc[unseen_mask, col] = self.label_encoders[col].classes_[0]
                X[col] = self.label_encoders[col].transform(X[col])

        X = X.fillna(0)
        return X, y

    def _prepare_horizon_data(self, data: pd.DataFrame, horizon: int
                              ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for a specific horizon (same target-shifting as XGBoost ensemble).

        Returns:
            (X, y_transformed, y_raw)
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        target_col = f'target_h{horizon}'
        df[target_col] = df.groupby('location')['value'].shift(-horizon)
        df = df.dropna(subset=[target_col])

        y_raw = df[target_col].copy()

        if self.target_mode == "ratio":
            denominator = df['value'].clip(lower=1e-6)
            y = (y_raw / denominator).clip(lower=self.ratio_clip_min, upper=self.ratio_clip_max)
        elif self.target_mode == "log":
            y = np.log1p(y_raw)
        else:
            y = y_raw.copy()

        df = df.drop(columns=[target_col])
        X, _ = self._prepare_features(df)

        return X, y, y_raw

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train all horizon models.

        Args:
            data: Full dataset with engineered features
            validation_split: Fraction for validation (temporal split)

        Returns:
            Training results dictionary
        """
        self._training_data = data.copy()
        results = {
            'training_date': datetime.now().isoformat(),
            'target_mode': self.target_mode,
            'device': str(self.device),
            'n_quantiles': len(self.quantiles),
            'quantiles': self.quantiles,
            'horizons': {}
        }

        for horizon in range(1, self.forecast_horizon + 1):
            print(f"\nTraining NN for horizon {horizon} (target_mode={self.target_mode})...")

            X, y, y_raw = self._prepare_horizon_data(data, horizon)
            n_features = X.shape[1]

            # Temporal split
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            y_val_raw = y_raw.iloc[split_idx:]

            # Fit scaler on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.values)
            X_val_scaled = scaler.transform(X_val.values)
            self.scalers[horizon] = scaler

            # Convert to tensors
            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val.values, dtype=torch.float32).to(self.device)

            # Create DataLoader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, drop_last=False)

            # Initialize model
            model = QuantileRegressionNet(
                n_features=n_features,
                n_quantiles=len(self.quantiles),
                hidden_dims=self.hidden_dims,
                dropout=self.dropout
            ).to(self.device)

            criterion = PinballLoss(self.quantiles).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                         weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
            )

            # Training loop with early stopping
            best_val_loss = float('inf')
            best_state = None
            epochs_without_improvement = 0

            for epoch in range(self.max_epochs):
                model.train()
                train_losses = []

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                # Validation
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_t)
                    val_loss = criterion(val_preds, y_val_t).item()

                scheduler.step(val_loss)
                avg_train_loss = np.mean(train_losses)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, best={best_val_loss:.4f}")

                if epochs_without_improvement >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            # Load best model
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models[horizon] = model

            # Compute validation metrics in original scale
            with torch.no_grad():
                val_preds_np = model(X_val_t).cpu().numpy()

            # Median prediction (index of 0.5 quantile)
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            val_median_transformed = val_preds_np[:, median_idx]

            if self.target_mode == "log":
                val_median_counts = np.expm1(val_median_transformed)
                val_median_counts = np.maximum(0, val_median_counts)
            elif self.target_mode == "ratio":
                current_vals = data.sort_values(['location', 'date']).reset_index(drop=True)
                current_vals = current_vals.iloc[split_idx:split_idx + len(X_val)]['value'].values
                val_median_counts = val_median_transformed * current_vals
                val_median_counts = np.maximum(0, val_median_counts)
            else:
                val_median_counts = val_median_transformed

            val_mae = float(np.mean(np.abs(val_median_counts - y_val_raw.values)))
            mask = y_val_raw.values > 0
            if mask.sum() > 0:
                val_mape = float(np.mean(np.abs(
                    (val_median_counts[mask] - y_val_raw.values[mask]) / y_val_raw.values[mask]
                )) * 100)
            else:
                val_mape = float('nan')

            results['horizons'][horizon] = {
                'n_features': n_features,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1,
                'val_mae_counts': val_mae,
                'val_mape_counts': val_mape
            }

            print(f"  Horizon {horizon}: Val MAE (counts)={val_mae:.2f}, "
                  f"Val MAPE={val_mape:.1f}%, epochs={epoch+1}")

        self.is_trained = True
        self.training_info = results
        print(f"\nNN training complete: {self.forecast_horizon} models trained on {self.device}.")
        return results

    def _enforce_quantile_ordering(self, predictions: Dict[float, float]) -> Dict[float, float]:
        """Sort predictions to ensure quantile monotonicity."""
        sorted_quantiles = sorted(predictions.keys())
        values = sorted([predictions[q] for q in sorted_quantiles])
        return {q: v for q, v in zip(sorted_quantiles, values)}

    def generate_forecasts(self, data: pd.DataFrame, cutoff_date: str,
                           locations: Optional[List[str]] = None,
                           use_floor_constraint: bool = True,
                           floor_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate quantile forecasts for all horizons.

        Same output format as QuantileDirectForecastEnsemble.generate_forecasts():
        Columns: location, cutoff_date, forecast_date, forecast_week,
                 predicted, predicted_q05, predicted_q25, predicted_q50,
                 predicted_q75, predicted_q95, target_mode

        Args:
            data: Historical data with features
            cutoff_date: Cutoff date string (YYYY-MM-DD)
            locations: Locations to forecast (None = all)
            use_floor_constraint: Apply floor constraint
            floor_ratio: Minimum ratio of last known value

        Returns:
            DataFrame with quantile forecasts
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before generating forecasts")

        cutoff_dt = pd.to_datetime(cutoff_date)
        historical_data = data[data['date'] <= cutoff_dt].copy()

        if locations is None:
            locations = historical_data['location'].unique().tolist()

        forecasts = []

        for location in locations:
            loc_data = historical_data[historical_data['location'] == location].copy()
            if len(loc_data) == 0:
                continue

            loc_data = loc_data.sort_values('date')
            last_row = loc_data.tail(1)
            last_value = last_row['value'].values[0]

            # Prepare features from last row
            X_last, _ = self._prepare_features(last_row)

            prev_predictions = {q: last_value for q in self.quantiles}

            for horizon in range(1, self.forecast_horizon + 1):
                forecast_date = cutoff_dt + timedelta(weeks=horizon)

                # Scale features
                X_scaled = self.scalers[horizon].transform(X_last.values)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

                # Predict
                self.models[horizon].eval()
                with torch.no_grad():
                    raw_preds = self.models[horizon](X_tensor).cpu().numpy()[0]

                # Map to quantile dict and apply inverse transform
                predictions = {}
                for i, q in enumerate(self.quantiles):
                    raw_pred = float(raw_preds[i])

                    if self.target_mode == "ratio":
                        if last_value < self.ratio_min_denominator:
                            pred = raw_pred * self.ratio_min_denominator
                        else:
                            pred = raw_pred * last_value
                    elif self.target_mode == "log":
                        pred = np.expm1(raw_pred)
                    else:
                        pred = raw_pred

                    predictions[q] = max(0.0, pred)

                predictions = self._enforce_quantile_ordering(predictions)

                # Floor constraint
                if use_floor_constraint:
                    horizon_floor_ratio = floor_ratio * (1 - (horizon - 1) * 0.05)
                    for q in self.quantiles:
                        floor_value = prev_predictions[q] * horizon_floor_ratio
                        predictions[q] = max(predictions[q], floor_value)
                    predictions = self._enforce_quantile_ordering(predictions)

                forecast = {
                    'location': location,
                    'cutoff_date': cutoff_date,
                    'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                    'forecast_week': horizon,
                    'predicted': predictions[0.5],
                    'predicted_q05': predictions[0.05],
                    'predicted_q25': predictions[0.25],
                    'predicted_q50': predictions[0.5],
                    'predicted_q75': predictions[0.75],
                    'predicted_q95': predictions[0.95],
                    'target_mode': self.target_mode
                }
                forecasts.append(forecast)
                prev_predictions = predictions.copy()

        return pd.DataFrame(forecasts)

    def save_models(self, output_dir: str) -> str:
        """Save all NN models, scalers, and metadata."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")

        models_dir = os.path.join(output_dir, 'nn_quantile_models')
        os.makedirs(models_dir, exist_ok=True)

        for horizon in range(1, self.forecast_horizon + 1):
            # Save PyTorch model
            model_path = os.path.join(models_dir, f'nn_h{horizon}.pt')
            torch.save({
                'model_state_dict': self.models[horizon].state_dict(),
                'n_features': self.models[horizon].n_features,
                'n_quantiles': self.models[horizon].n_quantiles,
            }, model_path)

            # Save scaler
            import pickle
            scaler_path = os.path.join(models_dir, f'scaler_h{horizon}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[horizon], f)

        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'quantiles': self.quantiles,
            'target_mode': self.target_mode,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'training_info': self.training_info,
            'label_encoders': {
                col: list(le.classes_) for col, le in self.label_encoders.items()
            }
        }
        with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return models_dir

    def load_models(self, models_dir: str) -> None:
        """Load all NN models, scalers, and metadata."""
        import pickle

        # Load metadata
        with open(os.path.join(models_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.feature_columns = metadata['feature_columns']
        self.quantiles = metadata['quantiles']
        self.target_mode = metadata['target_mode']
        self.hidden_dims = metadata.get('hidden_dims', [256, 128, 64])
        self.dropout = metadata.get('dropout', 0.2)
        self.training_info = metadata.get('training_info', {})

        # Restore label encoders
        for col, classes in metadata.get('label_encoders', {}).items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            self.label_encoders[col] = le

        for horizon in range(1, self.forecast_horizon + 1):
            # Load scaler
            scaler_path = os.path.join(models_dir, f'scaler_h{horizon}.pkl')
            with open(scaler_path, 'rb') as f:
                self.scalers[horizon] = pickle.load(f)

            # Load model
            model_path = os.path.join(models_dir, f'nn_h{horizon}.pt')
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

            model = QuantileRegressionNet(
                n_features=checkpoint['n_features'],
                n_quantiles=checkpoint['n_quantiles'],
                hidden_dims=self.hidden_dims,
                dropout=self.dropout
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models[horizon] = model

        self.is_trained = True
