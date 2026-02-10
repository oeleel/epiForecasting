"""
Step 5: Expanded Validation and Re-optimization

This script:
1. Runs Optuna optimization with expanded validation cutoffs (18 dates vs 3)
2. Saves optimized V3 parameters
3. Retrains with V3 params
4. Evaluates with phase-aware metrics
5. Generates comprehensive performance tracking
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List
import time

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble
from scripts.optimize import HyperparameterOptimizer, get_season_phase
from src.train import print_train_val_gap
from src import config


def run_expanded_validation_with_phases(ensemble: DirectForecastEnsemble,
                                        features_df: pd.DataFrame,
                                        cutoff_dates: List[str],
                                        actual_data: pd.DataFrame) -> Dict:
    """
    Run validation across expanded cutoffs with phase-aware metrics.

    Args:
        ensemble: Trained DirectForecastEnsemble
        features_df: Feature-engineered DataFrame
        cutoff_dates: List of cutoff dates to evaluate
        actual_data: Full dataset with actual values

    Returns:
        Dictionary with phase-aware validation results
    """
    all_results = []

    for cutoff_date in cutoff_dates:
        cutoff_dt = pd.to_datetime(cutoff_date)
        phase = get_season_phase(cutoff_date)

        # Generate forecasts for this cutoff
        try:
            forecasts = ensemble.generate_forecasts(
                data=features_df,
                cutoff_date=cutoff_date,
                use_floor_constraint=True,
                floor_ratio=0.3
            )
        except Exception as e:
            print(f"   Skipping {cutoff_date}: {e}")
            continue

        # Match with actuals
        for _, row in forecasts.iterrows():
            forecast_date = pd.to_datetime(row['forecast_date'])
            location = row['location']
            horizon = row['forecast_week']

            actual_row = actual_data[
                (actual_data['date'] == forecast_date) &
                (actual_data['location'] == location)
            ]

            if len(actual_row) > 0:
                actual_value = actual_row['value'].values[0]
                all_results.append({
                    'location': location,
                    'cutoff_date': cutoff_date,
                    'phase': phase,
                    'forecast_date': row['forecast_date'],
                    'horizon': horizon,
                    'forecast': row['forecast'],
                    'actual': actual_value
                })

    if not all_results:
        return {'error': 'No forecasts could be evaluated'}

    results_df = pd.DataFrame(all_results)

    # Calculate overall metrics
    y_true = results_df['actual'].values
    y_pred = results_df['forecast'].values
    mask = y_true > 0

    overall = {
        'mape': float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
        'mae': float(np.mean(np.abs(y_true - y_pred))),
        'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        'n_forecasts': len(results_df)
    }

    # Calculate metrics by phase
    by_phase = {}
    for phase in ['onset', 'peak', 'decline']:
        phase_data = results_df[results_df['phase'] == phase]
        if len(phase_data) > 0:
            y_true_p = phase_data['actual'].values
            y_pred_p = phase_data['forecast'].values
            mask_p = y_true_p > 0

            by_phase[phase] = {
                'mean_mae': float(np.mean(np.abs(y_true_p - y_pred_p))),
                'mean_mape': float(np.mean(np.abs((y_true_p[mask_p] - y_pred_p[mask_p]) /
                                                   y_true_p[mask_p])) * 100) if mask_p.sum() > 0 else None,
                'n_forecasts': len(phase_data)
            }

    # Calculate metrics by horizon
    by_horizon = {}
    for h in range(1, 5):
        h_data = results_df[results_df['horizon'] == h]
        if len(h_data) > 0:
            y_true_h = h_data['actual'].values
            y_pred_h = h_data['forecast'].values
            mask_h = y_true_h > 0

            by_horizon[str(h)] = {
                'mape': float(np.mean(np.abs((y_true_h[mask_h] - y_pred_h[mask_h]) /
                                             y_true_h[mask_h])) * 100) if mask_h.sum() > 0 else None,
                'mae': float(np.mean(np.abs(y_true_h - y_pred_h))),
                'rmse': float(np.sqrt(np.mean((y_true_h - y_pred_h) ** 2)))
            }

    return {
        'overall': overall,
        'by_phase': by_phase,
        'by_horizon': by_horizon,
        'n_cutoff_dates': len(set(results_df['cutoff_date']))
    }


def main():
    """Run Step 5: Expanded validation and re-optimization."""

    output_dir = "outputs/performance_tracking"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("# STEP 5: EXPANDED VALIDATION AND RE-OPTIMIZATION")
    print("#" * 70)

    # Step 1: Load data
    print("\n1. Loading data...")
    loader = FluDataLoader()
    raw_data = loader.fetch_data()
    print(f"   Total records: {len(raw_data)}")
    print(f"   Date range: {raw_data['date'].min()} to {raw_data['date'].max()}")

    # Step 2: Prepare features
    print("\n2. Preparing features...")
    engineer = FeatureEngineer(feature_version=config.FEATURE_VERSION)
    features_df = engineer.create_all_features(raw_data)
    print(f"   Features shape: {features_df.shape}")

    # Step 3: Get expanded validation cutoffs
    print("\n3. Setting up expanded validation cutoffs...")
    cutoff_dates = config.VALIDATION_CUTOFFS_EXPANDED

    # Filter cutoffs to those with sufficient data
    valid_cutoffs = []
    min_date = features_df['date'].min()
    max_date = features_df['date'].max()

    for cutoff in cutoff_dates:
        cutoff_dt = pd.to_datetime(cutoff)
        # Need at least 52 weeks before and 4 weeks after
        if (cutoff_dt - pd.Timedelta(weeks=52) >= min_date and
            cutoff_dt + pd.Timedelta(weeks=4) <= max_date):
            valid_cutoffs.append(cutoff)

    print(f"   Total cutoff dates: {len(cutoff_dates)}")
    print(f"   Valid cutoffs (with sufficient data): {len(valid_cutoffs)}")

    # Group by phase
    phase_counts = {"onset": 0, "peak": 0, "decline": 0}
    for cutoff in valid_cutoffs:
        phase = get_season_phase(cutoff)
        if phase in phase_counts:
            phase_counts[phase] += 1
    print(f"   By phase: {phase_counts}")

    # Step 4: Run Optuna optimization
    print("\n4. Running Optuna optimization...")
    print(f"   Trials: {config.OPTUNA_N_TRIALS}")
    print(f"   Pruning: {config.OPTUNA_USE_PRUNING}")

    optimizer = HyperparameterOptimizer(
        n_trials=config.OPTUNA_N_TRIALS,
        use_pruning=config.OPTUNA_USE_PRUNING
    )

    optuna_results = optimizer.optimize(
        data=features_df,
        cutoff_dates=valid_cutoffs
    )

    # Save V3 parameters
    v3_params_file = os.path.join(output_dir, "xgboost_params_v3.json")
    v3_params = optimizer.save_best_params_to_config(v3_params_file)

    # Step 5: Retrain with V3 params
    print("\n5. Retraining with optimized V3 parameters...")

    # Filter training data
    training_cutoff = config.DEFAULT_CUTOFF_DATE
    training_data = loader.load_and_preprocess(training_cutoff)
    training_features = engineer.create_all_features(training_data)

    ensemble = DirectForecastEnsemble(
        forecast_horizon=4,
        model_params=v3_params,
        target_mode=config.TARGET_MODE
    )
    training_results = ensemble.train(training_features, validation_split=0.2)

    # Print train/val gap
    gap_analysis = print_train_val_gap(training_results, "V3 Optuna-Optimized")

    # Step 6: Evaluate with phase-aware metrics
    print("\n6. Running expanded validation with phase-aware metrics...")

    # Use cutoffs that have future data for evaluation
    eval_cutoffs = [c for c in valid_cutoffs if pd.to_datetime(c) >= pd.to_datetime("2024-01-01")]
    print(f"   Evaluation cutoffs: {len(eval_cutoffs)}")

    eval_results = run_expanded_validation_with_phases(
        ensemble=ensemble,
        features_df=features_df,
        cutoff_dates=eval_cutoffs,
        actual_data=raw_data
    )

    # Extract train/val gap
    train_val_gap = {}
    for h in range(1, 5):
        h_results = training_results['horizons'].get(h, {})
        train_mae = h_results.get('train_mae', 0)
        val_mae = h_results.get('val_mae', 0)
        gap_ratio = val_mae / train_mae if train_mae > 0 else None

        train_val_gap[str(h)] = {
            'train_mae': float(train_mae) if train_mae else None,
            'val_mae': float(val_mae) if val_mae else None,
            'val_mae_counts': float(h_results.get('val_mae_counts', 0)),
            'gap_ratio': float(gap_ratio) if gap_ratio else None
        }

    # Load step 4 results for comparison
    step4_mape = None
    step4_path = os.path.join(output_dir, "step4_feature_engineering.json")
    if os.path.exists(step4_path):
        with open(step4_path, 'r') as f:
            step4_results = json.load(f)
            step4_mape = step4_results.get('overall_metrics', {}).get('mape')

    # Compile final results
    baseline_mape = 59.0
    new_mape = eval_results['overall']['mape']
    mape_improvement = ((baseline_mape - new_mape) / baseline_mape) * 100

    results = {
        "step": "5_expanded_validation_and_reoptimization",
        "timestamp": datetime.now().isoformat(),
        "optuna_results": {
            "n_trials": optuna_results['n_trials'],
            "n_completed": optuna_results['n_completed'],
            "n_pruned": optuna_results['n_pruned'],
            "best_params": v3_params,
            "best_val_mae": optuna_results['best_value'],
            "optimization_time_minutes": round(optuna_results['optimization_time_minutes'], 1),
            "param_importances": optuna_results.get('param_importances')
        },
        "expanded_validation_results": {
            "n_cutoff_dates": eval_results['n_cutoff_dates'],
            "cutoff_dates_used": eval_cutoffs,
            "by_phase": eval_results['by_phase']
        },
        "overall_metrics": eval_results['overall'],
        "by_horizon": eval_results['by_horizon'],
        "train_val_gap": train_val_gap,
        "comparison_to_baseline": {
            "baseline_overall_mape": baseline_mape,
            "new_overall_mape": new_mape,
            "mape_improvement_pct": round(mape_improvement, 2)
        },
        "comparison_to_previous_step": {
            "step4_overall_mape": step4_mape,
            "new_overall_mape": new_mape,
            "incremental_improvement_pct": round(((step4_mape - new_mape) / step4_mape) * 100, 2) if step4_mape else None
        }
    }

    # Save results
    output_file = os.path.join(output_dir, "step5_expanded_validation.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("STEP 5 RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nOptuna Optimization:")
    print(f"  Trials completed: {optuna_results['n_completed']}")
    print(f"  Trials pruned: {optuna_results['n_pruned']}")
    print(f"  Best validation MAE: {optuna_results['best_value']:.4f}")
    print(f"  Optimization time: {optuna_results['optimization_time_minutes']:.1f} minutes")

    print(f"\nBest V3 Parameters:")
    for param, value in v3_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nExpanded Validation ({eval_results['n_cutoff_dates']} cutoffs):")
    print(f"  By Phase:")
    for phase, metrics in eval_results['by_phase'].items():
        print(f"    {phase}: MAE={metrics['mean_mae']:.1f}, MAPE={metrics['mean_mape']:.1f}%")

    print(f"\nOverall Metrics:")
    print(f"  MAPE: {new_mape:.2f}%")
    print(f"  MAE:  {eval_results['overall']['mae']:.2f}")
    print(f"  RMSE: {eval_results['overall']['rmse']:.2f}")

    print(f"\nBy Horizon:")
    for h, metrics in eval_results['by_horizon'].items():
        print(f"  Horizon {h}: MAPE={metrics['mape']:.2f}%, MAE={metrics['mae']:.2f}")

    print(f"\nTrain/Val Gap:")
    for h, gap in train_val_gap.items():
        ratio = gap['gap_ratio']
        print(f"  Horizon {h}: Gap={ratio:.1f}x")

    print(f"\nComparison to Baseline:")
    print(f"  Baseline MAPE: {baseline_mape:.2f}%")
    print(f"  New MAPE:      {new_mape:.2f}%")
    if mape_improvement > 0:
        print(f"  Improvement:   {mape_improvement:.2f}% BETTER")
    else:
        print(f"  Change:        {-mape_improvement:.2f}% WORSE")

    if step4_mape:
        incremental = ((step4_mape - new_mape) / step4_mape) * 100
        print(f"\nComparison to Step 4:")
        print(f"  Step 4 MAPE: {step4_mape:.2f}%")
        print(f"  New MAPE:    {new_mape:.2f}%")
        if incremental > 0:
            print(f"  Incremental: {incremental:.2f}% BETTER")
        else:
            print(f"  Incremental: {-incremental:.2f}% WORSE")

    print("=" * 70)

    # Print V3 params in config.py format
    print("\n" + "-" * 70)
    print("Add to config.py as XGBOOST_PARAMS_V3:")
    print("-" * 70)
    print("XGBOOST_PARAMS_V3 = {")
    for param, value in v3_params.items():
        if isinstance(value, str):
            print(f"    '{param}': '{value}',")
        elif isinstance(value, float):
            print(f"    '{param}': {value},")
        else:
            print(f"    '{param}': {value},")
    print("}")
    print("-" * 70)

    return results


if __name__ == "__main__":
    main()
