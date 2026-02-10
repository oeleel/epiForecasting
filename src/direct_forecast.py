"""
Direct Forecasting Ensemble for flu hospitalization prediction.
Instead of recursive forecasting (which accumulates errors), this module
trains separate models for each forecast horizon.

Supports multiple target transformation modes to address ceiling effects:
- "raw": Direct hospitalization counts (original behavior)
- "ratio": Predict value(t+h) / value(t), then multiply by current value
- "log": Predict log1p(value(t+h)), then apply expm1 to convert back
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

from src.model import FluForecastingModel
from src import config


class DirectForecastEnsemble:
    """
    Separate XGBoost model for each forecast horizon (1, 2, 3, 4 weeks ahead).
    This avoids error accumulation from recursive forecasting.

    Supports target transformation modes:
    - "raw": Predict absolute hospitalization counts
    - "ratio": Predict growth ratio (value_future / value_current)
    - "log": Predict log-transformed values
    """

    def __init__(self, forecast_horizon: int = 4, model_params: Optional[Dict] = None,
                 target_mode: Optional[str] = None):
        """
        Initialize the direct forecast ensemble.

        Args:
            forecast_horizon: Number of weeks ahead to forecast
            model_params: XGBoost parameters (uses config defaults if None)
            target_mode: Target transformation mode ("raw", "ratio", or "log")
                        Uses config.TARGET_MODE if None
        """
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params or config.XGBOOST_PARAMS.copy()
        self.models = {h: FluForecastingModel(self.model_params.copy())
                      for h in range(1, forecast_horizon + 1)}
        self.is_trained = False
        self.training_info = {}

        # Target transformation settings
        self.target_mode = target_mode or getattr(config, 'TARGET_MODE', 'raw')
        self.ratio_clip_min = getattr(config, 'RATIO_CLIP_MIN', 0.05)
        self.ratio_clip_max = getattr(config, 'RATIO_CLIP_MAX', 10.0)
        self.ratio_min_denominator = getattr(config, 'RATIO_MIN_DENOMINATOR', 5)
    
    def prepare_horizon_data(self, data: pd.DataFrame, horizon: int,
                             return_raw_targets: bool = False) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Prepare training data for a specific horizon by shifting the target.

        Args:
            data: DataFrame with features and 'value' column
            horizon: Number of weeks ahead for this model
            return_raw_targets: If True, also return raw (untransformed) targets for evaluation

        Returns:
            Tuple of (features DataFrame, transformed target Series, raw target Series or None)
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        # Create target: value shifted backward by 'horizon' weeks
        # This means for training row at time t, target is value at t+horizon
        target_col = f'target_h{horizon}'
        df[target_col] = df.groupby('location')['value'].shift(-horizon)

        # Drop rows where target is NaN (last 'horizon' rows per location)
        df = df.dropna(subset=[target_col])

        # Store raw target values (for evaluation)
        y_raw = df[target_col].copy()

        # Apply target transformation based on mode
        if self.target_mode == "ratio":
            # Ratio mode: target = value(t+h) / value(t)
            current_value = df['value'].copy()
            # Avoid division by zero - use small epsilon for very small values
            denominator = current_value.clip(lower=1e-6)
            y = y_raw / denominator
            # Clip ratios to reasonable bounds
            y = y.clip(lower=self.ratio_clip_min, upper=self.ratio_clip_max)

        elif self.target_mode == "log":
            # Log mode: target = log1p(value(t+h))
            y = np.log1p(y_raw)

        else:
            # Raw mode: target = value(t+h) directly
            y = y_raw.copy()

        # Drop the target column so it's not included in features
        df = df.drop(columns=[target_col])

        # Prepare features (will use 'value' as dummy target, we don't need it)
        X, _ = self.models[horizon].prepare_features(df)

        if return_raw_targets:
            return X, y, y_raw
        return X, y, None
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train all horizon models.

        Args:
            data: Full dataset with features
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training results for each horizon
        """
        results = {
            'training_date': datetime.now().isoformat(),
            'target_mode': self.target_mode,
            'horizons': {}
        }

        # Store the full data for later use in generating forecasts
        self._training_data = data.copy()

        for horizon in range(1, self.forecast_horizon + 1):
            print(f"Training model for horizon {horizon} (target_mode={self.target_mode})...")

            # Prepare data for this horizon (with raw targets for evaluation)
            X, y, y_raw = self.prepare_horizon_data(data, horizon, return_raw_targets=True)

            # Split into train/validation (temporal split)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            # Also split raw targets for evaluation
            if y_raw is not None:
                y_val_raw = y_raw.iloc[split_idx:]
            else:
                y_val_raw = y_val

            # Train model on transformed targets
            train_results = self.models[horizon].train(X_train, y_train, (X_val, y_val))

            # For evaluation, compute metrics on actual hospitalization counts
            # Apply inverse transformation to predictions
            val_predictions_transformed = self.models[horizon].predict(X_val)

            if self.target_mode == "ratio":
                # Get current values for validation set
                # We need to reconstruct these from the original data
                df_sorted = data.sort_values(['location', 'date']).reset_index(drop=True)
                # The validation indices correspond to positions after split_idx
                full_X, full_y, full_y_raw = self.prepare_horizon_data(data, horizon, return_raw_targets=True)
                current_values_val = df_sorted.iloc[split_idx:split_idx + len(X_val)]['value'].values

                # Convert ratio predictions back to counts
                val_predictions_counts = val_predictions_transformed * current_values_val
                val_predictions_counts = np.maximum(0, val_predictions_counts)

            elif self.target_mode == "log":
                # Apply expm1 to convert back from log scale
                val_predictions_counts = np.expm1(val_predictions_transformed)
                val_predictions_counts = np.maximum(0, val_predictions_counts)

            else:
                # Raw mode - predictions are already counts
                val_predictions_counts = val_predictions_transformed

            # Compute metrics on actual counts
            val_mae_counts = np.mean(np.abs(val_predictions_counts - y_val_raw.values))
            val_rmse_counts = np.sqrt(np.mean((val_predictions_counts - y_val_raw.values) ** 2))

            # Also compute MAPE (avoid division by zero)
            mask = y_val_raw.values > 0
            if mask.sum() > 0:
                val_mape = np.mean(np.abs((val_predictions_counts[mask] - y_val_raw.values[mask]) /
                                          y_val_raw.values[mask])) * 100
            else:
                val_mape = np.nan

            results['horizons'][horizon] = {
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'train_mae': train_results.get('train_mae'),  # On transformed scale
                'val_mae': train_results.get('val_mae'),      # On transformed scale
                'train_rmse': train_results.get('train_rmse'),
                'val_rmse': train_results.get('val_rmse'),
                # Metrics on actual hospitalization counts
                'val_mae_counts': float(val_mae_counts),
                'val_rmse_counts': float(val_rmse_counts),
                'val_mape_counts': float(val_mape) if not np.isnan(val_mape) else None
            }

            print(f"  Horizon {horizon}: Train MAE (transformed)={train_results.get('train_mae', 0):.4f}, "
                  f"Val MAE (counts)={val_mae_counts:.2f}, Val MAPE={val_mape:.1f}%")

        self.is_trained = True
        self.training_info = results
        return results
    
    def predict(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Make predictions for a specific horizon.
        
        Args:
            X: Features for prediction
            horizon: Which horizon model to use (1-4)
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if horizon not in self.models:
            raise ValueError(f"Invalid horizon {horizon}. Must be 1-{self.forecast_horizon}")
        
        return self.models[horizon].predict(X)
    
    def generate_forecasts(self, data: pd.DataFrame, cutoff_date: str,
                          locations: Optional[List[str]] = None,
                          use_floor_constraint: bool = True,
                          floor_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate forecasts for all horizons using direct forecasting.

        Applies inverse transformation based on target_mode:
        - "raw": Predictions are used directly
        - "ratio": Multiply predicted ratio by current value
        - "log": Apply expm1() to convert from log scale

        Args:
            data: Historical data with features
            cutoff_date: Date to use as cutoff for forecasting
            locations: List of locations to forecast (None = all)
            use_floor_constraint: If True, predictions won't fall below floor_ratio * last_value
            floor_ratio: Minimum ratio of last known value (default 0.3 = 30%)

        Returns:
            DataFrame with forecasts (always in hospitalization counts)
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

            # Get the last available data point for this location
            loc_data = loc_data.sort_values('date')
            last_row = loc_data.tail(1)
            last_value = last_row['value'].values[0]

            # Prepare features from last row
            X_last, _ = self.models[1].prepare_features(last_row)

            # Generate forecast for each horizon
            prev_prediction = last_value
            for horizon in range(1, self.forecast_horizon + 1):
                forecast_date = cutoff_dt + timedelta(weeks=horizon)

                # Use the horizon-specific model to get raw prediction
                raw_prediction = self.models[horizon].predict(X_last)[0]

                # Apply inverse transformation based on target_mode
                if self.target_mode == "ratio":
                    # Ratio mode: prediction is a ratio, multiply by current value
                    if last_value < self.ratio_min_denominator:
                        # Edge case: if current value is very small, fall back to using
                        # a minimum base value to avoid instability
                        # The predicted ratio times a small epsilon might be too small
                        # Use the raw prediction scaled appropriately
                        prediction = raw_prediction * self.ratio_min_denominator
                    else:
                        prediction = raw_prediction * last_value

                elif self.target_mode == "log":
                    # Log mode: apply expm1 to convert back from log scale
                    prediction = np.expm1(raw_prediction)

                else:
                    # Raw mode: prediction is already in hospitalization counts
                    prediction = raw_prediction

                # Ensure non-negative before floor constraint
                prediction = max(0, prediction)

                # Apply floor constraint AFTER inverse transformation (on final counts)
                if use_floor_constraint:
                    # Floor decays with horizon: 30% for week 1, ~25% for week 4
                    horizon_floor_ratio = floor_ratio * (1 - (horizon - 1) * 0.05)
                    floor_value = prev_prediction * horizon_floor_ratio
                    prediction = max(prediction, floor_value)

                forecast = {
                    'location': location,
                    'cutoff_date': cutoff_date,
                    'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                    'forecast_week': horizon,
                    'forecast': prediction,
                    'raw_model_output': raw_prediction,  # Store for debugging
                    'target_mode': self.target_mode
                }
                forecasts.append(forecast)
                prev_prediction = prediction

        return pd.DataFrame(forecasts)
    
    def save_models(self, output_dir: str) -> str:
        """
        Save all horizon models.
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Path to saved models directory
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        models_dir = os.path.join(output_dir, 'direct_forecast_models')
        os.makedirs(models_dir, exist_ok=True)
        
        for horizon, model in self.models.items():
            model_path = os.path.join(models_dir, f'model_horizon_{horizon}')
            model.save_model(model_path)
        
        # Save training info
        info_path = os.path.join(models_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)
        
        return models_dir
    
    def load_models(self, models_dir: str) -> None:
        """
        Load all horizon models.
        
        Args:
            models_dir: Directory containing saved models
        """
        for horizon in range(1, self.forecast_horizon + 1):
            model_path = os.path.join(models_dir, f'model_horizon_{horizon}')
            self.models[horizon].load_model(model_path)
        
        # Load training info if available
        info_path = os.path.join(models_dir, 'training_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
        
        self.is_trained = True
    
    def get_feature_importance(self, horizon: int) -> pd.DataFrame:
        """
        Get feature importance for a specific horizon model.
        
        Args:
            horizon: Which horizon model
            
        Returns:
            DataFrame with feature importance
        """
        return self.models[horizon].get_feature_importance()
    
    def compare_horizons(self) -> pd.DataFrame:
        """
        Compare feature importance across all horizon models.
        
        Returns:
            DataFrame comparing top features across horizons
        """
        importance_dfs = []
        
        for horizon in range(1, self.forecast_horizon + 1):
            imp = self.models[horizon].get_feature_importance()
            imp['horizon'] = horizon
            importance_dfs.append(imp)
        
        combined = pd.concat(importance_dfs, ignore_index=True)
        
        # Pivot to compare features across horizons
        pivot = combined.pivot_table(
            index='feature',
            columns='horizon',
            values='importance',
            fill_value=0
        )
        pivot['avg_importance'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg_importance', ascending=False)
        
        return pivot


class QuantileDirectForecastEnsemble:
    """
    Quantile regression ensemble for prediction intervals.
    Trains separate models for each horizon and quantile combination.

    For 4 horizons and 5 quantiles = 20 models total.
    """

    def __init__(self, forecast_horizon: int = 4, model_params: Optional[Dict] = None,
                 target_mode: Optional[str] = None, quantiles: Optional[List[float]] = None):
        """
        Initialize the quantile forecast ensemble.

        Args:
            forecast_horizon: Number of weeks ahead to forecast
            model_params: XGBoost parameters (uses config defaults if None)
            target_mode: Target transformation mode ("raw", "ratio", or "log")
            quantiles: List of quantiles to predict (default from config.QUANTILES)
        """
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params or config.XGBOOST_PARAMS.copy()
        self.target_mode = target_mode or getattr(config, 'TARGET_MODE', 'raw')
        self.quantiles = quantiles or getattr(config, 'QUANTILES', [0.05, 0.25, 0.5, 0.75, 0.95])

        # Initialize models: self.models[horizon][quantile]
        self.models = {}
        for h in range(1, forecast_horizon + 1):
            self.models[h] = {}
            for q in self.quantiles:
                self.models[h][q] = FluForecastingModel(
                    self.model_params.copy(),
                    use_monotonic=True,
                    quantile=q
                )

        self.is_trained = False
        self.training_info = {}

        # Target transformation settings
        self.ratio_clip_min = getattr(config, 'RATIO_CLIP_MIN', 0.05)
        self.ratio_clip_max = getattr(config, 'RATIO_CLIP_MAX', 10.0)
        self.ratio_min_denominator = getattr(config, 'RATIO_MIN_DENOMINATOR', 5)

    def prepare_horizon_data(self, data: pd.DataFrame, horizon: int,
                             return_raw_targets: bool = False,
                             quantile: Optional[float] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Prepare training data for a specific horizon by shifting the target.
        (Same as DirectForecastEnsemble)

        Args:
            data: DataFrame with features
            horizon: Forecast horizon
            return_raw_targets: Whether to return raw (untransformed) targets
            quantile: If specified, use this quantile's model for feature prep
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        target_col = f'target_h{horizon}'
        df[target_col] = df.groupby('location')['value'].shift(-horizon)
        df = df.dropna(subset=[target_col])
        y_raw = df[target_col].copy()

        if self.target_mode == "ratio":
            current_value = df['value'].copy()
            denominator = current_value.clip(lower=1e-6)
            y = y_raw / denominator
            y = y.clip(lower=self.ratio_clip_min, upper=self.ratio_clip_max)
        elif self.target_mode == "log":
            y = np.log1p(y_raw)
        else:
            y = y_raw.copy()

        df = df.drop(columns=[target_col])

        # Use specified quantile model or first one for feature preparation
        q = quantile if quantile is not None else self.quantiles[0]
        X, _ = self.models[horizon][q].prepare_features(df)

        if return_raw_targets:
            return X, y, y_raw
        return X, y, None

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train all horizon-quantile models.

        Args:
            data: Full dataset with features
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training results for each horizon and quantile
        """
        results = {
            'training_date': datetime.now().isoformat(),
            'target_mode': self.target_mode,
            'n_quantiles': len(self.quantiles),
            'n_total_models': self.forecast_horizon * len(self.quantiles),
            'quantiles': self.quantiles,
            'horizons': {}
        }

        self._training_data = data.copy()
        total_models = self.forecast_horizon * len(self.quantiles)
        current_model = 0

        for horizon in range(1, self.forecast_horizon + 1):
            print(f"\nTraining models for horizon {horizon} (target_mode={self.target_mode})...")
            results['horizons'][horizon] = {'quantiles': {}}

            for q in self.quantiles:
                current_model += 1
                print(f"  [{current_model}/{total_models}] Training horizon {horizon}, quantile {q}...")

                # Prepare features specifically for this quantile model
                X_q, y_q, y_raw_q = self.prepare_horizon_data(data, horizon, return_raw_targets=True, quantile=q)
                split_idx_q = int(len(X_q) * (1 - validation_split))
                X_train_q, X_val_q = X_q.iloc[:split_idx_q], X_q.iloc[split_idx_q:]
                y_train_q, y_val_q = y_q.iloc[:split_idx_q], y_q.iloc[split_idx_q:]
                y_val_raw_q = y_raw_q.iloc[split_idx_q:] if y_raw_q is not None else y_val_q

                # Train quantile model
                train_results = self.models[horizon][q].train(X_train_q, y_train_q, (X_val_q, y_val_q))

                # Compute metrics on actual counts
                val_predictions_transformed = self.models[horizon][q].predict(X_val_q)

                if self.target_mode == "ratio":
                    # Get current values for this quantile's validation set
                    df_sorted = data.sort_values(['location', 'date']).reset_index(drop=True)
                    current_values_val_q = df_sorted.iloc[split_idx_q:split_idx_q + len(X_val_q)]['value'].values
                    val_predictions_counts = val_predictions_transformed * current_values_val_q
                    val_predictions_counts = np.maximum(0, val_predictions_counts)
                elif self.target_mode == "log":
                    val_predictions_counts = np.expm1(val_predictions_transformed)
                    val_predictions_counts = np.maximum(0, val_predictions_counts)
                else:
                    val_predictions_counts = val_predictions_transformed

                val_mae_counts = np.mean(np.abs(val_predictions_counts - y_val_raw_q.values))

                results['horizons'][horizon]['quantiles'][q] = {
                    'train_mae': train_results.get('train_mae'),
                    'val_mae': train_results.get('val_mae'),
                    'val_mae_counts': float(val_mae_counts),
                    'n_train_samples': len(X_train_q),
                    'n_val_samples': len(X_val_q)
                }

            results['horizons'][horizon]['n_train_samples'] = len(X_train_q)
            results['horizons'][horizon]['n_val_samples'] = len(X_val_q)

        self.is_trained = True
        self.training_info = results
        print(f"\nTraining complete: {total_models} models trained.")
        return results

    def _enforce_quantile_ordering(self, predictions: Dict[float, float]) -> Dict[float, float]:
        """
        Enforce monotonicity of quantile predictions.
        If q25 > q50 after prediction, sort them to maintain ordering.

        Args:
            predictions: Dict mapping quantile -> predicted value

        Returns:
            Corrected predictions with monotonic ordering
        """
        sorted_quantiles = sorted(predictions.keys())
        values = [predictions[q] for q in sorted_quantiles]

        # Sort values to ensure monotonicity
        sorted_values = sorted(values)

        return {q: v for q, v in zip(sorted_quantiles, sorted_values)}

    def generate_forecasts(self, data: pd.DataFrame, cutoff_date: str,
                          locations: Optional[List[str]] = None,
                          use_floor_constraint: bool = True,
                          floor_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate quantile forecasts for all horizons.

        Returns DataFrame with columns:
        - location, cutoff_date, forecast_date, forecast_week
        - predicted (q50 median)
        - predicted_q05, predicted_q25, predicted_q50, predicted_q75, predicted_q95

        Args:
            data: Historical data with features
            cutoff_date: Date to use as cutoff for forecasting
            locations: List of locations to forecast (None = all)
            use_floor_constraint: If True, predictions won't fall below floor_ratio * last_value
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
            first_quantile = self.quantiles[0]
            X_last, _ = self.models[1][first_quantile].prepare_features(last_row)

            prev_predictions = {q: last_value for q in self.quantiles}

            for horizon in range(1, self.forecast_horizon + 1):
                forecast_date = cutoff_dt + timedelta(weeks=horizon)

                # Get predictions for all quantiles
                raw_predictions = {}
                predictions = {}

                for q in self.quantiles:
                    raw_pred = self.models[horizon][q].predict(X_last)[0]
                    raw_predictions[q] = raw_pred

                    # Apply inverse transformation
                    if self.target_mode == "ratio":
                        if last_value < self.ratio_min_denominator:
                            pred = raw_pred * self.ratio_min_denominator
                        else:
                            pred = raw_pred * last_value
                    elif self.target_mode == "log":
                        pred = np.expm1(raw_pred)
                    else:
                        pred = raw_pred

                    pred = max(0, pred)
                    predictions[q] = pred

                # Enforce quantile ordering
                predictions = self._enforce_quantile_ordering(predictions)

                # Apply floor constraint to ALL quantile predictions
                if use_floor_constraint:
                    horizon_floor_ratio = floor_ratio * (1 - (horizon - 1) * 0.05)
                    for q in self.quantiles:
                        floor_value = prev_predictions[q] * horizon_floor_ratio
                        predictions[q] = max(predictions[q], floor_value)

                # Re-enforce ordering after floor constraint
                predictions = self._enforce_quantile_ordering(predictions)

                forecast = {
                    'location': location,
                    'cutoff_date': cutoff_date,
                    'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                    'forecast_week': horizon,
                    'predicted': predictions[0.5],  # Point prediction is median
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
        """Save all quantile models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")

        models_dir = os.path.join(output_dir, 'quantile_forecast_models')
        os.makedirs(models_dir, exist_ok=True)

        for horizon in range(1, self.forecast_horizon + 1):
            for q in self.quantiles:
                model_path = os.path.join(models_dir, f'model_h{horizon}_q{int(q*100):02d}')
                self.models[horizon][q].save_model(model_path)

        info_path = os.path.join(models_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)

        return models_dir

    def load_models(self, models_dir: str) -> None:
        """Load all quantile models."""
        for horizon in range(1, self.forecast_horizon + 1):
            for q in self.quantiles:
                model_path = os.path.join(models_dir, f'model_h{horizon}_q{int(q*100):02d}')
                self.models[horizon][q].load_model(model_path)

        info_path = os.path.join(models_dir, 'training_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)

        self.is_trained = True


class ClusteredDirectForecastEnsemble:
    """
    Cluster-specific forecasting ensemble.
    Trains separate models for each cluster of locations, where clusters are
    determined by similarity in flu hospitalization patterns.

    For n_clusters clusters and 4 horizons = 4 × n_clusters models.
    Optionally supports quantile mode for prediction intervals.
    """

    def __init__(self, forecast_horizon: int = 4, model_params: Optional[Dict] = None,
                 target_mode: Optional[str] = None, n_clusters: int = None,
                 enable_quantiles: bool = False, quantiles: Optional[List[float]] = None):
        """
        Initialize the clustered forecast ensemble.

        Args:
            forecast_horizon: Number of weeks ahead to forecast
            model_params: XGBoost parameters (uses config defaults if None)
            target_mode: Target transformation mode ("raw", "ratio", or "log")
            n_clusters: Number of location clusters (default from config)
            enable_quantiles: If True, train quantile models for each cluster
            quantiles: List of quantiles (only used if enable_quantiles=True)
        """
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params or config.XGBOOST_PARAMS.copy()
        self.target_mode = target_mode or getattr(config, 'TARGET_MODE', 'raw')
        self.n_clusters = n_clusters or getattr(config, 'N_LOCATION_CLUSTERS', 5)

        # Quantile mode settings
        self.enable_quantiles = enable_quantiles
        self.quantiles = quantiles or getattr(config, 'QUANTILES', [0.05, 0.25, 0.5, 0.75, 0.95])

        # Models will be initialized after clustering
        # Structure: self.models[cluster_id][horizon] for point mode
        # Structure: self.models[cluster_id][horizon][quantile] for quantile mode
        self.models = {}

        # Clustering info (set during fit)
        self.clusterer = None
        self.location_to_cluster = {}
        self.cluster_profiles = {}

        self.is_trained = False
        self.training_info = {}

        # Target transformation settings
        self.ratio_clip_min = getattr(config, 'RATIO_CLIP_MIN', 0.05)
        self.ratio_clip_max = getattr(config, 'RATIO_CLIP_MAX', 10.0)
        self.ratio_min_denominator = getattr(config, 'RATIO_MIN_DENOMINATOR', 5)

    def fit_clusters(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Compute location profiles and cluster locations.

        Args:
            data: Full dataset with date, location, value columns

        Returns:
            Dictionary mapping location -> cluster_id
        """
        from location_clustering import LocationClusterer

        print(f"Clustering locations into {self.n_clusters} clusters...")
        self.clusterer = LocationClusterer(n_clusters=self.n_clusters)

        # Compute profiles and cluster
        profiles = self.clusterer.compute_location_profiles(data)
        self.location_to_cluster = self.clusterer.cluster_locations(profiles)
        self.cluster_profiles = self.clusterer.cluster_profiles

        # Initialize models for each cluster
        unique_clusters = set(self.location_to_cluster.values())
        print(f"Initializing models for {len(unique_clusters)} clusters...")

        for cluster_id in unique_clusters:
            self.models[cluster_id] = {}
            for h in range(1, self.forecast_horizon + 1):
                if self.enable_quantiles:
                    self.models[cluster_id][h] = {}
                    for q in self.quantiles:
                        self.models[cluster_id][h][q] = FluForecastingModel(
                            self.model_params.copy(),
                            use_monotonic=True,
                            quantile=q
                        )
                else:
                    self.models[cluster_id][h] = FluForecastingModel(
                        self.model_params.copy(),
                        use_monotonic=True
                    )

        return self.location_to_cluster

    def prepare_horizon_data(self, data: pd.DataFrame, horizon: int,
                             return_raw_targets: bool = False,
                             model: Optional[FluForecastingModel] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Prepare training data for a specific horizon.

        Args:
            data: DataFrame with features
            horizon: Forecast horizon
            return_raw_targets: Whether to return raw (untransformed) targets
            model: Model to use for feature preparation (uses default if None)
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        target_col = f'target_h{horizon}'
        df[target_col] = df.groupby('location')['value'].shift(-horizon)
        df = df.dropna(subset=[target_col])
        y_raw = df[target_col].copy()

        if self.target_mode == "ratio":
            current_value = df['value'].copy()
            denominator = current_value.clip(lower=1e-6)
            y = y_raw / denominator
            y = y.clip(lower=self.ratio_clip_min, upper=self.ratio_clip_max)
        elif self.target_mode == "log":
            y = np.log1p(y_raw)
        else:
            y = y_raw.copy()

        df = df.drop(columns=[target_col])

        # Use provided model or get a reference model for feature prep
        if model is not None:
            X, _ = model.prepare_features(df)
        else:
            # Get any model for feature prep
            first_cluster = list(self.models.keys())[0]
            if self.enable_quantiles:
                ref_model = self.models[first_cluster][horizon][self.quantiles[0]]
            else:
                ref_model = self.models[first_cluster][horizon]
            X, _ = ref_model.prepare_features(df)

        if return_raw_targets:
            return X, y, y_raw
        return X, y, None

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train all cluster-specific models.

        Args:
            data: Full dataset with features
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training results
        """
        # First, fit clusters if not already done
        if not self.location_to_cluster:
            self.fit_clusters(data)

        results = {
            'training_date': datetime.now().isoformat(),
            'target_mode': self.target_mode,
            'n_clusters': len(set(self.location_to_cluster.values())),
            'enable_quantiles': self.enable_quantiles,
            'clusters': {}
        }

        self._training_data = data.copy()
        unique_clusters = sorted(set(self.location_to_cluster.values()))

        for cluster_id in unique_clusters:
            print(f"\n{'='*60}")
            print(f"Training models for Cluster {cluster_id}")

            # Get locations in this cluster
            cluster_locations = [loc for loc, c in self.location_to_cluster.items()
                                if c == cluster_id]
            cluster_data = data[data['location'].isin(cluster_locations)].copy()

            print(f"  Locations ({len(cluster_locations)}): {cluster_locations[:5]}{'...' if len(cluster_locations) > 5 else ''}")
            print(f"  Training samples: {len(cluster_data)}")

            results['clusters'][cluster_id] = {
                'n_locations': len(cluster_locations),
                'locations': cluster_locations,
                'horizons': {}
            }

            for horizon in range(1, self.forecast_horizon + 1):
                if self.enable_quantiles:
                    results['clusters'][cluster_id]['horizons'][horizon] = {'quantiles': {}}

                    for q in self.quantiles:
                        model = self.models[cluster_id][horizon][q]
                        X, y, y_raw = self.prepare_horizon_data(
                            cluster_data, horizon, return_raw_targets=True, model=model
                        )

                        if len(X) < 20:
                            print(f"  Warning: Cluster {cluster_id} has insufficient data for horizon {horizon}")
                            continue

                        split_idx = int(len(X) * (1 - validation_split))
                        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                        y_val_raw = y_raw.iloc[split_idx:] if y_raw is not None else y_val

                        train_results = model.train(X_train, y_train, (X_val, y_val))

                        val_pred = model.predict(X_val)
                        if self.target_mode == "log":
                            val_pred = np.expm1(val_pred)
                        val_mae = np.mean(np.abs(val_pred - y_val_raw.values))

                        results['clusters'][cluster_id]['horizons'][horizon]['quantiles'][q] = {
                            'val_mae_counts': float(val_mae)
                        }

                    print(f"  Horizon {horizon}: Trained {len(self.quantiles)} quantile models")
                else:
                    model = self.models[cluster_id][horizon]
                    X, y, y_raw = self.prepare_horizon_data(
                        cluster_data, horizon, return_raw_targets=True, model=model
                    )

                    if len(X) < 20:
                        print(f"  Warning: Cluster {cluster_id} has insufficient data for horizon {horizon}")
                        continue

                    split_idx = int(len(X) * (1 - validation_split))
                    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                    y_val_raw = y_raw.iloc[split_idx:] if y_raw is not None else y_val

                    train_results = model.train(X_train, y_train, (X_val, y_val))

                    val_pred = model.predict(X_val)
                    if self.target_mode == "log":
                        val_pred = np.expm1(val_pred)
                    val_mae = np.mean(np.abs(val_pred - y_val_raw.values))

                    mask = y_val_raw.values > 0
                    if mask.sum() > 0:
                        val_mape = np.mean(np.abs((val_pred[mask] - y_val_raw.values[mask]) /
                                                   y_val_raw.values[mask])) * 100
                    else:
                        val_mape = np.nan

                    results['clusters'][cluster_id]['horizons'][horizon] = {
                        'val_mae_counts': float(val_mae),
                        'val_mape_counts': float(val_mape) if not np.isnan(val_mape) else None
                    }

                    print(f"  Horizon {horizon}: MAE={val_mae:.1f}, MAPE={val_mape:.1f}%")

        self.is_trained = True
        self.training_info = results

        # Print summary
        self.clusterer.print_cluster_summary()

        return results

    def _enforce_quantile_ordering(self, predictions: Dict[float, float]) -> Dict[float, float]:
        """Enforce monotonicity of quantile predictions."""
        sorted_quantiles = sorted(predictions.keys())
        values = [predictions[q] for q in sorted_quantiles]
        sorted_values = sorted(values)
        return {q: v for q, v in zip(sorted_quantiles, sorted_values)}

    def generate_forecasts(self, data: pd.DataFrame, cutoff_date: str,
                          locations: Optional[List[str]] = None,
                          use_floor_constraint: bool = True,
                          floor_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate forecasts using cluster-specific models.

        Routes each location to its cluster's models for prediction.

        Args:
            data: Historical data with features
            cutoff_date: Date to use as cutoff for forecasting
            locations: List of locations to forecast (None = all)
            use_floor_constraint: If True, predictions won't fall below floor_ratio * last_value
            floor_ratio: Minimum ratio of last known value

        Returns:
            DataFrame with forecasts (includes cluster assignment)
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

            # Get cluster for this location
            cluster_id = self.location_to_cluster.get(location, 0)

            # Handle case where cluster doesn't have models (fallback to cluster 0)
            if cluster_id not in self.models:
                cluster_id = list(self.models.keys())[0]

            # Get the appropriate model for feature preparation
            if self.enable_quantiles:
                ref_model = self.models[cluster_id][1][self.quantiles[0]]
            else:
                ref_model = self.models[cluster_id][1]

            X_last, _ = ref_model.prepare_features(last_row)

            if self.enable_quantiles:
                prev_predictions = {q: last_value for q in self.quantiles}
            else:
                prev_prediction = last_value

            for horizon in range(1, self.forecast_horizon + 1):
                forecast_date = cutoff_dt + timedelta(weeks=horizon)

                if self.enable_quantiles:
                    predictions = {}
                    for q in self.quantiles:
                        raw_pred = self.models[cluster_id][horizon][q].predict(X_last)[0]

                        if self.target_mode == "ratio":
                            if last_value < self.ratio_min_denominator:
                                pred = raw_pred * self.ratio_min_denominator
                            else:
                                pred = raw_pred * last_value
                        elif self.target_mode == "log":
                            pred = np.expm1(raw_pred)
                        else:
                            pred = raw_pred

                        predictions[q] = max(0, pred)

                    predictions = self._enforce_quantile_ordering(predictions)

                    if use_floor_constraint:
                        horizon_floor_ratio = floor_ratio * (1 - (horizon - 1) * 0.05)
                        for q in self.quantiles:
                            floor_value = prev_predictions[q] * horizon_floor_ratio
                            predictions[q] = max(predictions[q], floor_value)
                        predictions = self._enforce_quantile_ordering(predictions)

                    forecast = {
                        'location': location,
                        'cluster': cluster_id,
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
                    prev_predictions = predictions.copy()
                else:
                    raw_pred = self.models[cluster_id][horizon].predict(X_last)[0]

                    if self.target_mode == "ratio":
                        if last_value < self.ratio_min_denominator:
                            prediction = raw_pred * self.ratio_min_denominator
                        else:
                            prediction = raw_pred * last_value
                    elif self.target_mode == "log":
                        prediction = np.expm1(raw_pred)
                    else:
                        prediction = raw_pred

                    prediction = max(0, prediction)

                    if use_floor_constraint:
                        horizon_floor_ratio = floor_ratio * (1 - (horizon - 1) * 0.05)
                        floor_value = prev_prediction * horizon_floor_ratio
                        prediction = max(prediction, floor_value)

                    forecast = {
                        'location': location,
                        'cluster': cluster_id,
                        'cutoff_date': cutoff_date,
                        'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                        'forecast_week': horizon,
                        'predicted': prediction,
                        'target_mode': self.target_mode
                    }
                    prev_prediction = prediction

                forecasts.append(forecast)

        return pd.DataFrame(forecasts)

    def save_models(self, output_dir: str) -> str:
        """Save all cluster models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")

        models_dir = os.path.join(output_dir, 'clustered_forecast_models')
        os.makedirs(models_dir, exist_ok=True)

        for cluster_id in self.models:
            for horizon in range(1, self.forecast_horizon + 1):
                if self.enable_quantiles:
                    for q in self.quantiles:
                        model_path = os.path.join(
                            models_dir,
                            f'model_c{cluster_id}_h{horizon}_q{int(q*100):02d}'
                        )
                        self.models[cluster_id][horizon][q].save_model(model_path)
                else:
                    model_path = os.path.join(models_dir, f'model_c{cluster_id}_h{horizon}')
                    self.models[cluster_id][horizon].save_model(model_path)

        # Save clustering info
        cluster_info = {
            'location_to_cluster': {k: int(v) for k, v in self.location_to_cluster.items()},
            'cluster_profiles': self.cluster_profiles,
            'training_info': self.training_info
        }
        info_path = os.path.join(models_dir, 'cluster_info.json')
        with open(info_path, 'w') as f:
            json.dump(cluster_info, f, indent=2)

        # Also save clustering visualization
        if self.clusterer:
            self.clusterer.save_results(os.path.join(output_dir, 'location_clusters'))

        return models_dir

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary of cluster assignments and performance.

        Returns:
            DataFrame with cluster information
        """
        if not self.is_trained:
            return pd.DataFrame()

        summary = []
        for cluster_id, profile in self.cluster_profiles.items():
            cluster_results = self.training_info.get('clusters', {}).get(cluster_id, {})

            # Average MAE across horizons
            horizons = cluster_results.get('horizons', {})
            if self.enable_quantiles:
                avg_mae = np.nan
            else:
                maes = [h.get('val_mae_counts', np.nan) for h in horizons.values()]
                avg_mae = np.nanmean(maes) if maes else np.nan

            summary.append({
                'cluster': cluster_id,
                'n_locations': profile['n_locations'],
                'locations': ', '.join(profile['locations'][:5]) + ('...' if profile['n_locations'] > 5 else ''),
                'description': self.clusterer.get_cluster_description(cluster_id) if self.clusterer else '',
                'avg_val_mae': avg_mae
            })

        return pd.DataFrame(summary)


class HybridForecastGenerator:
    """
    Combines recursive and direct forecasting methods with optional ensemble.
    """
    
    def __init__(self, model: Optional[FluForecastingModel] = None,
                 direct_ensemble: Optional[DirectForecastEnsemble] = None,
                 ensemble_weights: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        Initialize hybrid generator.
        
        Args:
            model: Recursive forecasting model
            direct_ensemble: Direct forecasting ensemble
            ensemble_weights: Dict mapping horizon -> (recursive_weight, direct_weight)
                             Default: more weight to direct for longer horizons
        """
        self.recursive_model = model
        self.direct_ensemble = direct_ensemble
        
        # Default weights: trust recursive more for week 1, direct more for weeks 2-4
        self.ensemble_weights = ensemble_weights or {
            1: (0.6, 0.4),  # Week 1: 60% recursive, 40% direct
            2: (0.4, 0.6),  # Week 2: 40% recursive, 60% direct
            3: (0.3, 0.7),  # Week 3: 30% recursive, 70% direct
            4: (0.2, 0.8),  # Week 4: 20% recursive, 80% direct
        }
    
    def generate_ensemble_forecasts(self, data: pd.DataFrame, cutoff_date: str,
                                   locations: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate forecasts using weighted ensemble of recursive and direct methods.
        
        Args:
            data: Historical data with features
            cutoff_date: Cutoff date for forecasting
            locations: List of locations to forecast
            
        Returns:
            DataFrame with ensemble forecasts
        """
        from predict import FluForecastGenerator
        
        # Generate recursive forecasts
        if self.recursive_model is not None:
            recursive_gen = FluForecastGenerator(model=self.recursive_model)
            recursive_forecasts = recursive_gen.generate_forecasts(
                data=data, cutoff_date=cutoff_date, locations=locations
            )
        else:
            recursive_forecasts = None
        
        # Generate direct forecasts
        if self.direct_ensemble is not None:
            direct_forecasts = self.direct_ensemble.generate_forecasts(
                data=data, cutoff_date=cutoff_date, locations=locations
            )
        else:
            direct_forecasts = None
        
        # If only one method available, return that
        if recursive_forecasts is None:
            return direct_forecasts
        if direct_forecasts is None:
            return recursive_forecasts
        
        # Combine forecasts using ensemble weights
        combined = recursive_forecasts.copy()
        combined = combined.rename(columns={'forecast': 'forecast_recursive'})
        
        # Merge direct forecasts
        direct_subset = direct_forecasts[['location', 'forecast_date', 'forecast_week', 'forecast']]
        direct_subset = direct_subset.rename(columns={'forecast': 'forecast_direct'})
        
        combined = combined.merge(
            direct_subset,
            on=['location', 'forecast_date', 'forecast_week'],
            how='left'
        )
        
        # Apply ensemble weights
        def weighted_ensemble(row):
            horizon = int(row['forecast_week'])
            rec_weight, dir_weight = self.ensemble_weights.get(horizon, (0.5, 0.5))
            return rec_weight * row['forecast_recursive'] + dir_weight * row['forecast_direct']
        
        combined['forecast'] = combined.apply(weighted_ensemble, axis=1)
        
        # Clean up and return
        result = combined[['location', 'cutoff_date', 'forecast_date', 'forecast_week', 'forecast']]
        return result


def main():
    """Example usage of direct forecasting"""
    from data_loader import FluDataLoader
    from feature_engineering import FeatureEngineer
    
    print("Loading data...")
    loader = FluDataLoader()
    data = loader.load_and_preprocess("2024-11-02")
    
    print("Creating features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(data)
    
    print(f"Data shape: {features_df.shape}")
    
    # Initialize and train direct forecast ensemble
    print("\nTraining direct forecast ensemble...")
    ensemble = DirectForecastEnsemble(forecast_horizon=4)
    results = ensemble.train(features_df)
    
    print("\nTraining Results:")
    for horizon, metrics in results['horizons'].items():
        print(f"  Horizon {horizon}: Val MAE = {metrics.get('val_mae', 'N/A'):.2f}")
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = ensemble.generate_forecasts(
        data=features_df,
        cutoff_date="2024-11-02",
        locations=['US', '06', '48']  # US, California, Texas
    )
    
    print("\nSample Forecasts:")
    print(forecasts.head(12))
    
    # Compare feature importance across horizons
    print("\nFeature Importance Comparison (Top 10):")
    comparison = ensemble.compare_horizons()
    print(comparison.head(10))


if __name__ == "__main__":
    main()
