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
from src import config
from src.model import FluForecastingModel


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


def print_train_val_gap(training_results: Dict, model_name: str = "Model") -> Dict:
    """
    Print and return the train/validation MAE gap for each horizon.

    This function helps monitor overfitting by showing the ratio between
    validation and training MAE. A ratio close to 1.0 indicates good
    generalization, while high ratios (e.g., 40x) indicate severe overfitting.

    IMPORTANT: Gap ratio is computed on the same scale (transformed scale)
    to ensure fair comparison. For count-based metrics, see val_mae_counts.

    Args:
        training_results: Dictionary with training results from DirectForecastEnsemble.train()
                         or similar, containing 'horizons' with train_mae and val_mae
        model_name: Name to display in the output

    Returns:
        Dictionary with gap analysis for each horizon
    """
    print("\n" + "=" * 70)
    print(f"TRAIN/VALIDATION GAP ANALYSIS - {model_name}")
    print("=" * 70)
    print(f"{'Horizon':<10} {'Train MAE':<15} {'Val MAE':<15} {'Gap Ratio':<12} {'Status'}")
    print("-" * 70)

    gap_analysis = {}
    target_mode = training_results.get('target_mode', 'raw')

    horizons = training_results.get('horizons', {})
    for horizon in sorted(horizons.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
        h_data = horizons[horizon]

        # IMPORTANT: Compare on the same scale (transformed scale) for fair gap ratio
        # train_mae and val_mae are both on transformed scale from XGBoost
        train_mae = h_data.get('train_mae', 0)
        # Use val_mae (transformed scale) for gap calculation, NOT val_mae_counts
        val_mae = h_data.get('val_mae', 0)

        if train_mae and train_mae > 0:
            gap_ratio = val_mae / train_mae
        else:
            gap_ratio = float('inf')

        # Determine status based on gap ratio
        if gap_ratio < 2:
            status = "EXCELLENT"
        elif gap_ratio < 5:
            status = "GOOD"
        elif gap_ratio < 10:
            status = "MODERATE"
        elif gap_ratio < 20:
            status = "HIGH"
        else:
            status = "SEVERE OVERFITTING"

        print(f"{horizon:<10} {train_mae:<15.4f} {val_mae:<15.2f} {gap_ratio:<12.1f}x {status}")

        gap_analysis[str(horizon)] = {
            'train_mae': float(train_mae) if train_mae else None,
            'val_mae': float(val_mae) if val_mae else None,
            'gap_ratio': float(gap_ratio) if gap_ratio != float('inf') else None,
            'status': status
        }

    # Calculate average gap ratio
    valid_ratios = [g['gap_ratio'] for g in gap_analysis.values() if g['gap_ratio'] is not None]
    if valid_ratios:
        avg_gap = np.mean(valid_ratios)
        print("-" * 70)
        print(f"{'AVERAGE':<10} {'':<15} {'':<15} {avg_gap:<12.1f}x")
        gap_analysis['average_gap_ratio'] = float(avg_gap)

    print("=" * 70)
    print("Gap Ratio Interpretation:")
    print("  < 2x  : Excellent generalization")
    print("  2-5x  : Good generalization")
    print("  5-10x : Moderate overfitting, consider more regularization")
    print("  10-20x: High overfitting, needs stronger regularization")
    print("  > 20x : Severe overfitting, model is memorizing training data")
    print("=" * 70 + "\n")

    return gap_analysis
