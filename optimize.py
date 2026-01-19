"""
Optuna hyperparameter optimization for flu forecasting model
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import FluDataLoader
from feature_engineering import FeatureEngineer
from model import FluForecastingModel
from train import FluModelTrainer
import config


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimizer for flu forecasting"""
    
    def __init__(self, n_trials: int = 50, study_name: Optional[str] = None):
        """
        Initialize the optimizer
        
        Args:
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
        """
        self.n_trials = n_trials
        self.study_name = study_name or f"flu_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def objective(self, trial, data: pd.DataFrame, cutoff_dates: List[str], 
                  locations: Optional[List[str]] = None) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            data: Full dataset with features
            cutoff_dates: List of cutoff dates for walk-forward validation
            locations: List of locations to include
            
        Returns:
            Mean MAE across validation splits (to minimize)
        """
        # Suggest hyperparameters with expanded search space
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'tree_method': 'hist',
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'random_state': 42,
            'early_stopping_rounds': 50
        }
        
        # Add max_leaves for lossguide policy
        if params['grow_policy'] == 'lossguide':
            params['max_leaves'] = trial.suggest_int('max_leaves', 10, 256)
        
        # Filter data by locations if specified
        if locations is not None:
            data_filtered = data[data['location'].isin(locations)].copy()
        else:
            data_filtered = data.copy()
        
        # Perform walk-forward validation
        mae_scores = []
        
        for cutoff_date in cutoff_dates:
            # Prepare training and validation data
            cutoff_dt = pd.to_datetime(cutoff_date)
            train_data = data_filtered[data_filtered['date'] <= cutoff_dt].copy()
            val_data = data_filtered[data_filtered['date'] > cutoff_dt].copy()
            
            if len(train_data) == 0 or len(val_data) == 0:
                continue
            
            # Sort by date
            train_data = train_data.sort_values('date')
            val_data = val_data.sort_values('date')
            
            # Create model with suggested parameters
            model = FluForecastingModel(model_params=params)
            
            # Prepare features
            X_train, y_train = model.prepare_features(train_data)
            X_val, y_val = model.prepare_features(val_data)
            
            # Train model
            try:
                model.train(X_train, y_train, (X_val, y_val))
                
                # Make predictions
                val_predictions = model.predict(X_val)
                
                # Calculate MAE
                mae = mean_absolute_error(y_val, val_predictions)
                mae_scores.append(mae)
                
            except Exception as e:
                # If training fails, return a high penalty
                print(f"Trial failed with error: {str(e)}")
                return 1e6
        
        # Return mean MAE across all validation splits
        if len(mae_scores) == 0:
            return 1e6
        
        return np.mean(mae_scores)
    
    def optimize(self, data: pd.DataFrame, cutoff_dates: List[str],
                 locations: Optional[List[str]] = None,
                 direction: str = 'minimize',
                 metric: str = 'mae') -> Dict:
        """
        Run hyperparameter optimization
        
        Args:
            data: Full dataset with features
            cutoff_dates: List of cutoff dates for walk-forward validation
            locations: List of locations to include
            direction: Optimization direction ('minimize' or 'maximize')
            metric: Metric to optimize ('mae' or 'rmse')
            
        Returns:
            Dictionary with optimization results
        """
        print("="*60)
        print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("="*60)
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Cutoff dates: {cutoff_dates}")
        print(f"Locations: {locations if locations else 'All'}")
        print("="*60)
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, data, cutoff_dates, locations),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"Best {metric.upper()}: {self.best_value:.6f}")
        print("\nBest hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print("="*60)
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'study_name': self.study_name,
            'optimization_date': datetime.now().isoformat()
        }
    
    def save_results(self, results: Dict, output_dir: str = "outputs") -> str:
        """
        Save optimization results to file
        
        Args:
            results: Optimization results dictionary
            output_dir: Output directory
            
        Returns:
            Path to saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"optuna_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nOptimization results saved to: {results_file}")
        return results_file
    
    def save_best_params_to_config(self, output_file: str = "best_params.json") -> str:
        """
        Save best parameters to a JSON file that can be used for training
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to saved parameters file
        """
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Prepare parameters in XGBoost format
        best_xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': self.best_params['max_depth'],
            'learning_rate': self.best_params['learning_rate'],
            'n_estimators': self.best_params['n_estimators'],
            'subsample': self.best_params['subsample'],
            'colsample_bytree': self.best_params['colsample_bytree'],
            'min_child_weight': self.best_params['min_child_weight'],
            'gamma': self.best_params['gamma'],
            'reg_alpha': self.best_params['reg_alpha'],
            'reg_lambda': self.best_params['reg_lambda'],
            'random_state': 42,
            'early_stopping_rounds': 50
        }
        
        with open(output_file, 'w') as f:
            json.dump(best_xgb_params, f, indent=2)
        
        print(f"Best parameters saved to: {output_file}")
        return output_file
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame
        
        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        trials_df = self.study.trials_dataframe()
        return trials_df


def load_optimized_params(filepath: str) -> Dict:
    """
    Load optimized hyperparameters from JSON file
    
    Args:
        filepath: Path to JSON file with optimized parameters
        
    Returns:
        Dictionary with XGBoost parameters
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params


def main():
    """Main optimization execution"""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization with Optuna')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    parser.add_argument('--cutoff-date', type=str, default=config.DEFAULT_CUTOFF_DATE,
                       help='Cutoff date for training data (YYYY-MM-DD)')
    parser.add_argument('--validation-cutoffs', nargs='+', default=None,
                       help='List of cutoff dates for validation (default: uses config.VALIDATION_CUTOFFS)')
    parser.add_argument('--locations', nargs='+', default=None,
                       help='List of locations to optimize for (if not specified, use all)')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Name for the Optuna study')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine validation cutoff dates
    if args.validation_cutoffs:
        cutoff_dates = args.validation_cutoffs
    else:
        cutoff_dates = config.VALIDATION_CUTOFFS
    
    print("Loading and preparing data...")
    
    # Load data
    loader = FluDataLoader()
    data = loader.load_and_preprocess(args.cutoff_date)
    
    # Filter locations if specified
    if args.locations:
        data = data[data['location'].isin(args.locations)]
        print(f"Filtered to {len(args.locations)} locations")
    
    # Create features
    print("Creating features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(data)
    
    print(f"Data shape: {features_df.shape}")
    print(f"Features: {len(engineer.get_feature_columns(features_df))}")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        n_trials=args.n_trials,
        study_name=args.study_name
    )
    
    # Run optimization
    results = optimizer.optimize(
        data=features_df,
        cutoff_dates=cutoff_dates,
        locations=args.locations
    )
    
    # Save results
    results_file = optimizer.save_results(results, args.output_dir)
    
    # Save best parameters
    best_params_file = os.path.join(args.output_dir, "best_hyperparameters.json")
    optimizer.save_best_params_to_config(best_params_file)
    
    # Get and display optimization history
    history_df = optimizer.get_optimization_history()
    history_file = os.path.join(args.output_dir, f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    history_df.to_csv(history_file, index=False)
    print(f"Optimization history saved to: {history_file}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Best MAE: {results['best_value']:.6f}")
    print(f"Total trials: {results['n_trials']}")
    print(f"\nTo use these parameters, load them from: {best_params_file}")
    print("="*60)


if __name__ == "__main__":
    main()

