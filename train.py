"""
Model training module for flu forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config
from model import FluForecastingModel


class FluModelTrainer:
    """Trainer for flu forecasting models with walk-forward validation"""
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the trainer
        
        Args:
            model_params: XGBoost parameters (uses config defaults if None)
        """
        self.model_params = model_params or config.XGBOOST_PARAMS.copy()
        self.model = FluForecastingModel(self.model_params)
        self.validation_results = {}
        
    def prepare_training_data(self, data: pd.DataFrame, cutoff_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and validation data based on cutoff date
        
        Args:
            data: Full dataset
            cutoff_date: Date to use as cutoff for training data
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        # Convert cutoff date to datetime
        cutoff_dt = pd.to_datetime(cutoff_date)
        
        # Split data
        train_data = data[data['date'] <= cutoff_dt].copy()
        val_data = data[data['date'] > cutoff_dt].copy()
        
        # Sort by date
        train_data = train_data.sort_values('date')
        val_data = val_data.sort_values('date')
        
        return train_data, val_data
    
    def walk_forward_validation(self, data: pd.DataFrame, 
                              cutoff_dates: List[str],
                              locations: Optional[List[str]] = None) -> Dict:
        """
        Perform walk-forward validation across multiple cutoff dates
        
        Args:
            data: Full dataset
            cutoff_dates: List of cutoff dates for validation
            locations: List of locations to include (if None, use all)
            
        Returns:
            Dictionary with validation results
        """
        if locations is not None:
            data = data[data['location'].isin(locations)].copy()
        
        results = {
            'cutoff_dates': cutoff_dates,
            'validation_splits': [],
            'overall_metrics': {},
            'metrics_by_location': {},
            'metrics_by_horizon': {}
        }
        
        for cutoff_date in cutoff_dates:
            print(f"Validating with cutoff date: {cutoff_date}")
            
            # Prepare data for this cutoff
            train_data, val_data = self.prepare_training_data(data, cutoff_date)
            
            if len(train_data) == 0 or len(val_data) == 0:
                print(f"Warning: Insufficient data for cutoff {cutoff_date}")
                continue
            
            # Train model
            X_train, y_train = self.model.prepare_features(train_data)
            X_val, y_val = self.model.prepare_features(val_data)
            
            # Train model
            train_results = self.model.train(X_train, y_train, (X_val, y_val))
            
            # Make predictions
            val_predictions = self.model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, val_predictions)
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100
            smape = np.mean(2 * np.abs(y_val - val_predictions) / (np.abs(y_val) + np.abs(val_predictions))) * 100
            
            # Store results for this split
            split_results = {
                'cutoff_date': cutoff_date,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'smape': smape
                },
                'predictions': val_predictions.tolist(),
                'actuals': y_val.tolist()
            }
            
            results['validation_splits'].append(split_results)
            
            # Aggregate metrics
            if 'mae' not in results['overall_metrics']:
                results['overall_metrics'] = {'mae': [], 'rmse': [], 'mape': [], 'smape': []}
            
            for metric in ['mae', 'rmse', 'mape', 'smape']:
                results['overall_metrics'][metric].append(split_results['metrics'][metric])
        
        # Calculate overall metrics
        for metric in results['overall_metrics']:
            values = results['overall_metrics'][metric]
            results['overall_metrics'][f'{metric}_mean'] = np.mean(values)
            results['overall_metrics'][f'{metric}_std'] = np.std(values)
        
        # Calculate metrics by location
        if locations is not None:
            for location in locations:
                location_metrics = []
                for split in results['validation_splits']:
                    # Get predictions for this location
                    location_mask = val_data['location'] == location
                    if location_mask.any():
                        loc_pred = val_predictions[location_mask]
                        loc_actual = y_val[location_mask]
                        
                        if len(loc_pred) > 0:
                            loc_mae = mean_absolute_error(loc_actual, loc_pred)
                            location_metrics.append(loc_mae)
                
                if location_metrics:
                    results['metrics_by_location'][location] = {
                        'mae_mean': np.mean(location_metrics),
                        'mae_std': np.std(location_metrics)
                    }
        
        self.validation_results = results
        return results
    
    def train_final_model(self, data: pd.DataFrame, cutoff_date: str,
                         locations: Optional[List[str]] = None) -> Dict:
        """
        Train the final model on all available data up to cutoff date
        
        Args:
            data: Full dataset
            cutoff_date: Cutoff date for training data
            locations: List of locations to include (if None, use all)
            
        Returns:
            Dictionary with training results
        """
        # Filter data
        if locations is not None:
            data = data[data['location'].isin(locations)].copy()
        
        # Prepare training data
        train_data, _ = self.prepare_training_data(data, cutoff_date)
        
        if len(train_data) == 0:
            raise ValueError(f"No training data available for cutoff date {cutoff_date}")
        
        # Split training data for validation (use last 20% for validation)
        split_idx = int(len(train_data) * 0.8)
        train_subset = train_data.iloc[:split_idx].copy()
        val_subset = train_data.iloc[split_idx:].copy()
        
        # Prepare features
        X_train, y_train = self.model.prepare_features(train_subset)
        X_val, y_val = self.model.prepare_features(val_subset)
        
        # Train model with validation data
        train_results = self.model.train(X_train, y_train, (X_val, y_val))
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance()
        
        results = {
            'training_date': datetime.now().isoformat(),
            'cutoff_date': cutoff_date,
            'n_samples': len(train_data),
            'n_features': X_train.shape[1],
            'training_metrics': train_results,
            'feature_importance': feature_importance.to_dict('records')
        }
        
        return results
    
    def save_training_results(self, results: Dict, output_dir: str) -> str:
        """
        Save training results to file
        
        Args:
            results: Training results dictionary
            output_dir: Output directory
            
        Returns:
            Path to saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"training_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results_file
    
    def get_validation_summary(self) -> Dict:
        """
        Get summary of validation results
        
        Returns:
            Dictionary with validation summary
        """
        if not self.validation_results:
            return {'status': 'no_validation_performed'}
        
        summary = {
            'n_splits': len(self.validation_results['validation_splits']),
            'overall_performance': self.validation_results['overall_metrics'],
            'best_cutoff': None,
            'worst_cutoff': None
        }
        
        # Find best and worst performing cutoffs
        if self.validation_results['validation_splits']:
            mae_scores = [split['metrics']['mae'] for split in self.validation_results['validation_splits']]
            best_idx = np.argmin(mae_scores)
            worst_idx = np.argmax(mae_scores)
            
            summary['best_cutoff'] = {
                'date': self.validation_results['validation_splits'][best_idx]['cutoff_date'],
                'mae': mae_scores[best_idx]
            }
            summary['worst_cutoff'] = {
                'date': self.validation_results['validation_splits'][worst_idx]['cutoff_date'],
                'mae': mae_scores[worst_idx]
            }
        
        return summary
