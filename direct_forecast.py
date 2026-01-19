"""
Direct Forecasting Ensemble for flu hospitalization prediction.
Instead of recursive forecasting (which accumulates errors), this module
trains separate models for each forecast horizon.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

from model import FluForecastingModel
import config


class DirectForecastEnsemble:
    """
    Separate XGBoost model for each forecast horizon (1, 2, 3, 4 weeks ahead).
    This avoids error accumulation from recursive forecasting.
    """
    
    def __init__(self, forecast_horizon: int = 4, model_params: Optional[Dict] = None):
        """
        Initialize the direct forecast ensemble.
        
        Args:
            forecast_horizon: Number of weeks ahead to forecast
            model_params: XGBoost parameters (uses config defaults if None)
        """
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params or config.XGBOOST_PARAMS.copy()
        self.models = {h: FluForecastingModel(self.model_params.copy()) 
                      for h in range(1, forecast_horizon + 1)}
        self.is_trained = False
        self.training_info = {}
    
    def prepare_horizon_data(self, data: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for a specific horizon by shifting the target.
        
        Args:
            data: DataFrame with features and 'value' column
            horizon: Number of weeks ahead for this model
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)
        
        # Create target: value shifted backward by 'horizon' weeks
        # This means for training row at time t, target is value at t+horizon
        target_col = f'target_h{horizon}'
        df[target_col] = df.groupby('location')['value'].shift(-horizon)
        
        # Drop rows where target is NaN (last 'horizon' rows per location)
        df = df.dropna(subset=[target_col])
        
        # Extract target before dropping the column
        y = df[target_col].copy()
        
        # Drop the target column so it's not included in features
        df = df.drop(columns=[target_col])
        
        # Prepare features (will use 'value' as dummy target, we don't need it)
        X, _ = self.models[horizon].prepare_features(df)
        
        return X, y
    
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
            'horizons': {}
        }
        
        for horizon in range(1, self.forecast_horizon + 1):
            print(f"Training model for horizon {horizon}...")
            
            # Prepare data for this horizon
            X, y = self.prepare_horizon_data(data, horizon)
            
            # Split into train/validation (temporal split)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            train_results = self.models[horizon].train(X_train, y_train, (X_val, y_val))
            
            results['horizons'][horizon] = {
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'train_mae': train_results.get('train_mae'),
                'val_mae': train_results.get('val_mae'),
                'train_rmse': train_results.get('train_rmse'),
                'val_rmse': train_results.get('val_rmse')
            }
            
            print(f"  Horizon {horizon}: Train MAE={train_results.get('train_mae', 0):.2f}, "
                  f"Val MAE={train_results.get('val_mae', 0):.2f}")
        
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
        
        Args:
            data: Historical data with features
            cutoff_date: Date to use as cutoff for forecasting
            locations: List of locations to forecast (None = all)
            use_floor_constraint: If True, predictions won't fall below floor_ratio * last_value
            floor_ratio: Minimum ratio of last known value (default 0.3 = 30%)
            
        Returns:
            DataFrame with forecasts
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
                
                # Use the horizon-specific model
                prediction = self.models[horizon].predict(X_last)[0]
                
                # Apply floor constraint to prevent extreme under-predictions
                if use_floor_constraint:
                    # Floor decays with horizon: 30% for week 1, 20% for week 4
                    horizon_floor_ratio = floor_ratio * (1 - (horizon - 1) * 0.05)
                    floor_value = prev_prediction * horizon_floor_ratio
                    prediction = max(prediction, floor_value)
                
                prediction = max(0, prediction)  # Ensure non-negative
                
                forecast = {
                    'location': location,
                    'cutoff_date': cutoff_date,
                    'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                    'forecast_week': horizon,
                    'forecast': prediction
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
