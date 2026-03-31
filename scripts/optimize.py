"""
Optuna hyperparameter optimization for flu forecasting model

Updated for Step 5:
- Expanded validation cutoffs (18 dates across 3 seasons)
- MedianPruner for early stopping of poor trials
- Phase-aware validation tracking
- Integration with DirectForecastEnsemble
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
import argparse
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.model import FluForecastingModel
from src.direct_forecast import DirectForecastEnsemble, QuantileDirectForecastEnsemble
from src.evaluate import QuantileEvaluator
from src.train import FluModelTrainer
from src import config


def get_season_phase(date_str: str) -> str:
    """
    Determine the flu season phase for a given date.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Phase name: "onset", "peak", or "decline"
    """
    dt = pd.to_datetime(date_str)
    month = dt.month

    if month in [10, 11]:  # October, November
        return "onset"
    elif month in [12, 1]:  # December, January
        return "peak"
    elif month in [2, 3, 4]:  # February, March, April
        return "decline"
    else:
        return "off_season"


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimizer for flu forecasting"""

    def __init__(self, n_trials: int = 100, study_name: Optional[str] = None,
                 use_pruning: bool = True):
        """
        Initialize the optimizer

        Args:
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            use_pruning: Whether to use MedianPruner for early stopping
        """
        self.n_trials = n_trials
        self.study_name = study_name or f"flu_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = None
        self.best_params = None
        self.best_value = None
        self.use_pruning = use_pruning
        self.optimization_time = None
        self.phase_metrics = {}  # Track metrics by season phase
        
    def objective(self, trial, data: pd.DataFrame, cutoff_dates: List[str],
                  locations: Optional[List[str]] = None) -> float:
        """
        Objective function for Optuna optimization with pruning support.

        Args:
            trial: Optuna trial object
            data: Full dataset with features
            cutoff_dates: List of cutoff dates for walk-forward validation
            locations: List of locations to include

        Returns:
            Mean MAE across validation splits (to minimize)
        """
        # Suggest hyperparameters with search space optimized for regularization
        params = {
            'objective': 'reg:squarederror',
            # Tree structure - constrained to prevent overfitting
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            # Sampling - reduced ranges for regularization
            'subsample': trial.suggest_float('subsample', 0.5, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 0.8),
            # Regularization parameters - expanded ranges
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            # Other parameters
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'tree_method': 'hist',
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'random_state': 42,
            'early_stopping_rounds': 50
        }

        # Add max_leaves for lossguide policy
        if params['grow_policy'] == 'lossguide':
            params['max_leaves'] = trial.suggest_int('max_leaves', 8, 64)

        # Filter data by locations if specified
        if locations is not None:
            data_filtered = data[data['location'].isin(locations)].copy()
        else:
            data_filtered = data.copy()

        # Perform walk-forward validation with pruning support
        mae_scores = []

        for i, cutoff_date in enumerate(cutoff_dates):
            # Prepare training and validation data
            cutoff_dt = pd.to_datetime(cutoff_date)
            train_data = data_filtered[data_filtered['date'] <= cutoff_dt].copy()

            # Only evaluate on next 4 weeks (avoid data leakage from future cutoffs)
            val_end_dt = cutoff_dt + pd.Timedelta(weeks=4)
            val_data = data_filtered[
                (data_filtered['date'] > cutoff_dt) &
                (data_filtered['date'] <= val_end_dt)
            ].copy()

            if len(train_data) < 100 or len(val_data) == 0:
                continue

            # Sort by date
            train_data = train_data.sort_values('date')
            val_data = val_data.sort_values('date')

            # Create model with suggested parameters
            model = FluForecastingModel(model_params=params)

            # Prepare features
            X_train, y_train = model.prepare_features(train_data)
            X_val, y_val = model.prepare_features(val_data)

            if len(X_val) == 0:
                continue

            # Train model
            try:
                model.train(X_train, y_train, (X_val, y_val))

                # Make predictions
                val_predictions = model.predict(X_val)

                # Calculate MAE
                mae = mean_absolute_error(y_val, val_predictions)
                mae_scores.append(mae)

                # Report intermediate value for pruning
                if self.use_pruning and len(mae_scores) > 0:
                    intermediate_value = np.mean(mae_scores)
                    trial.report(intermediate_value, i)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                # If training fails, continue to next cutoff
                continue

        # Return mean MAE across all validation splits
        if len(mae_scores) == 0:
            return 1e6

        return np.mean(mae_scores)

    def objective_wis(self, trial, data: pd.DataFrame, cutoff_dates: List[str],
                      locations: Optional[List[str]] = None) -> float:
        """
        Objective function for WIS-based optimization.
        Trains quantile models and minimizes Weighted Interval Score.

        Args:
            trial: Optuna trial object
            data: Full dataset with features
            cutoff_dates: List of cutoff dates for walk-forward validation
            locations: List of locations to include

        Returns:
            Mean WIS across validation splits (to minimize)
        """
        # Same search space but with reduced n_estimators for speed (20 models per trial)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'tree_method': 'hist',
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'random_state': 42,
            'early_stopping_rounds': 50
        }

        if params['grow_policy'] == 'lossguide':
            params['max_leaves'] = trial.suggest_int('max_leaves', 8, 64)

        if locations is not None:
            data_filtered = data[data['location'].isin(locations)].copy()
        else:
            data_filtered = data.copy()

        wis_scores = []
        evaluator = QuantileEvaluator()

        for i, cutoff_date in enumerate(cutoff_dates):
            cutoff_dt = pd.to_datetime(cutoff_date)

            # Load full data up to end of validation window for evaluation
            cutoff_data = data_filtered[data_filtered['date'] <= cutoff_dt].copy()
            if len(cutoff_data) < 100:
                continue

            try:
                # Train quantile ensemble with trial params
                q_ensemble = QuantileDirectForecastEnsemble(
                    forecast_horizon=4, xgb_params_override=params
                )
                q_ensemble.train(cutoff_data)

                # Generate forecasts
                cutoff_str = cutoff_dt.strftime('%Y-%m-%d')
                forecasts = q_ensemble.generate_forecasts(
                    cutoff_data, cutoff_str,
                    locations=locations if locations else None
                )

                # Evaluate WIS against actual data
                results = evaluator.evaluate_quantile_forecasts(forecasts, data_filtered)
                if 'error' not in results and results['quantile_metrics'].get('wis') is not None:
                    wis_scores.append(results['quantile_metrics']['wis'])

                # Pruning support
                if self.use_pruning and wis_scores:
                    trial.report(np.mean(wis_scores), i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                continue

        return np.mean(wis_scores) if wis_scores else 1e6

    def optimize(self, data: pd.DataFrame, cutoff_dates: List[str],
                 locations: Optional[List[str]] = None,
                 direction: str = 'minimize',
                 metric: str = 'mae') -> Dict:
        """
        Run hyperparameter optimization with optional pruning.

        Args:
            data: Full dataset with features
            cutoff_dates: List of cutoff dates for walk-forward validation
            locations: List of locations to include
            direction: Optimization direction ('minimize' or 'maximize')
            metric: Metric to optimize ('mae' or 'rmse')

        Returns:
            Dictionary with optimization results
        """
        print("=" * 70)
        print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("=" * 70)
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Pruning enabled: {self.use_pruning}")
        print(f"Number of cutoff dates: {len(cutoff_dates)}")
        print(f"Cutoff dates by phase:")

        # Group cutoffs by phase
        phase_counts = {"onset": 0, "peak": 0, "decline": 0, "off_season": 0}
        for cutoff in cutoff_dates:
            phase = get_season_phase(cutoff)
            phase_counts[phase] += 1
        for phase, count in phase_counts.items():
            if count > 0:
                print(f"  {phase}: {count} cutoffs")

        print(f"Locations: {locations if locations else 'All'}")
        print("=" * 70)

        # Create pruner if enabled
        pruner = None
        if self.use_pruning:
            n_startup = 5 if metric == 'wis' else 10
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=n_startup,
                n_warmup_steps=2 if metric == 'wis' else 3,
                interval_steps=1
            )

        # Create Optuna study
        self.study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner
        )

        # Track optimization time
        start_time = time.time()

        # Select objective function based on metric
        if metric == 'wis':
            obj_func = lambda trial: self.objective_wis(trial, data, cutoff_dates, locations)
        else:
            obj_func = lambda trial: self.objective(trial, data, cutoff_dates, locations)

        # Optimize
        self.study.optimize(
            obj_func,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1  # Single-threaded for reproducibility
        )

        end_time = time.time()
        self.optimization_time = (end_time - start_time) / 60  # minutes

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Count completed and pruned trials
        n_completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETED")
        print("=" * 70)
        print(f"Best {metric.upper()}: {self.best_value:.4f}")
        print(f"Optimization time: {self.optimization_time:.1f} minutes")
        print(f"Trials completed: {n_completed}")
        print(f"Trials pruned: {n_pruned}")
        print("\nBest hyperparameters:")
        for param, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        print("=" * 70)

        # Try to get parameter importances
        param_importances = None
        try:
            param_importances = optuna.importance.get_param_importances(self.study)
            print("\nParameter importances:")
            for param, importance in sorted(param_importances.items(), key=lambda x: -x[1]):
                print(f"  {param}: {importance:.4f}")
        except Exception:
            pass

        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'n_completed': n_completed,
            'n_pruned': n_pruned,
            'optimization_time_minutes': self.optimization_time,
            'param_importances': param_importances,
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
    
    def save_best_params_to_config(self, output_file: str = "best_params.json") -> Dict:
        """
        Save best parameters to a JSON file that can be used for training.

        Args:
            output_file: Output file path

        Returns:
            Dictionary with best XGBoost parameters
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
            'colsample_bylevel': self.best_params.get('colsample_bylevel', 0.6),
            'min_child_weight': self.best_params['min_child_weight'],
            'gamma': self.best_params['gamma'],
            'reg_alpha': self.best_params['reg_alpha'],
            'reg_lambda': self.best_params['reg_lambda'],
            'max_delta_step': self.best_params.get('max_delta_step', 0),
            'random_state': 42,
            'early_stopping_rounds': 50
        }

        # Add grow_policy and max_leaves if present
        if 'grow_policy' in self.best_params:
            best_xgb_params['grow_policy'] = self.best_params['grow_policy']
            best_xgb_params['tree_method'] = 'hist'
        if 'max_leaves' in self.best_params:
            best_xgb_params['max_leaves'] = self.best_params['max_leaves']

        with open(output_file, 'w') as f:
            json.dump(best_xgb_params, f, indent=2)

        print(f"Best parameters saved to: {output_file}")
        return best_xgb_params
    
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
    parser.add_argument('--metric', type=str, default='mae', choices=['mae', 'wis'],
                       help='Metric to optimize: mae (default) or wis (Weighted Interval Score)')

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
        locations=args.locations,
        metric=args.metric
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
    print(f"Best {args.metric.upper()}: {results['best_value']:.6f}")
    print(f"Total trials: {results['n_trials']}")
    print(f"\nTo use these parameters, load them from: {best_params_file}")
    print("="*60)


if __name__ == "__main__":
    main()

