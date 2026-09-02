"""
Evaluate target transformation approaches for flu forecasting.

This script trains the DirectForecastEnsemble with different target modes
(raw, ratio, log) and generates performance tracking reports.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble
from src.evaluate import ModelEvaluator
from src import config


def evaluate_ensemble_forecasts(ensemble: DirectForecastEnsemble,
                                features_df: pd.DataFrame,
                                cutoff_dates: List[str],
                                actual_data: pd.DataFrame) -> Dict:
    """
    Evaluate ensemble forecasts against actual data across multiple cutoff dates.

    Args:
        ensemble: Trained DirectForecastEnsemble
        features_df: Feature-engineered DataFrame
        cutoff_dates: List of cutoff dates to evaluate
        actual_data: Full dataset with actual values for comparison

    Returns:
        Dictionary with evaluation metrics
    """
    all_forecasts = []
    all_actuals = []

    for cutoff_date in cutoff_dates:
        print(f"  Generating forecasts for cutoff {cutoff_date}...")

        # Generate forecasts
        forecasts = ensemble.generate_forecasts(
            data=features_df,
            cutoff_date=cutoff_date,
            use_floor_constraint=True,
            floor_ratio=0.3
        )

        # Get actual values for comparison
        cutoff_dt = pd.to_datetime(cutoff_date)

        for _, row in forecasts.iterrows():
            forecast_date = pd.to_datetime(row['forecast_date'])
            location = row['location']
            horizon = row['forecast_week']

            # Find actual value
            actual_row = actual_data[
                (actual_data['date'] == forecast_date) &
                (actual_data['location'] == location)
            ]

            if len(actual_row) > 0:
                actual_value = actual_row['value'].values[0]
                all_forecasts.append({
                    'location': location,
                    'cutoff_date': cutoff_date,
                    'forecast_date': row['forecast_date'],
                    'horizon': horizon,
                    'forecast': row['forecast'],
                    'actual': actual_value
                })

    if not all_forecasts:
        return {'error': 'No forecasts could be evaluated'}

    results_df = pd.DataFrame(all_forecasts)

    # Calculate overall metrics
    y_true = results_df['actual'].values
    y_pred = results_df['forecast'].values

    mask = y_true > 0
    overall_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

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
        'overall_metrics': {
            'mape': float(overall_mape),
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'n_forecasts': len(results_df)
        },
        'by_horizon': by_horizon,
        'results_df': results_df
    }


def run_evaluation(target_mode: str, output_dir: str = "outputs/performance_tracking") -> Dict:
    """
    Run full evaluation pipeline for a given target mode.

    Args:
        target_mode: "raw", "ratio", or "log"
        output_dir: Directory to save results

    Returns:
        Dictionary with complete evaluation results
    """
    print("=" * 70)
    print(f"EVALUATING TARGET MODE: {target_mode.upper()}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("\n1. Loading data...")
    loader = FluDataLoader()
    # Load ALL data (not just up to cutoff) for evaluation
    raw_data = loader.fetch_data()
    print(f"   Total records: {len(raw_data)}")

    # Step 2: Load data with training cutoff
    print("\n2. Preparing training data...")
    training_data = loader.load_and_preprocess(config.DEFAULT_CUTOFF_DATE)

    # Step 3: Engineer features
    print("\n3. Engineering features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(training_data)
    print(f"   Features shape: {features_df.shape}")

    # Step 4: Train ensemble with specified target mode
    print(f"\n4. Training DirectForecastEnsemble (target_mode={target_mode})...")
    ensemble = DirectForecastEnsemble(
        forecast_horizon=4,
        target_mode=target_mode
    )
    training_results = ensemble.train(features_df, validation_split=0.2)

    # Extract train/val gap
    train_val_gap = {}
    for h in range(1, 5):
        h_results = training_results['horizons'].get(h, {})
        train_val_gap[str(h)] = {
            'train_mae': h_results.get('train_mae'),
            'val_mae': h_results.get('val_mae_counts'),
            'val_mape': h_results.get('val_mape_counts')
        }

    # Step 5: Evaluate on held-out period
    print("\n5. Evaluating forecasts on held-out period...")

    # Use cutoff dates that have future data to evaluate
    evaluation_cutoffs = [
        "2024-11-02",
        "2024-11-09",
        "2024-11-16",
        "2024-11-23",
        "2024-11-30",
        "2024-12-07",
        "2024-12-14",
        "2024-12-21",
        "2024-12-28",
        "2025-01-04",
        "2025-01-11",
    ]

    # Filter cutoffs to those where we have sufficient future data
    valid_cutoffs = []
    max_date = raw_data['date'].max()
    for cutoff in evaluation_cutoffs:
        cutoff_dt = pd.to_datetime(cutoff)
        # Need at least 4 weeks of future data
        if cutoff_dt + pd.Timedelta(7 * (4), unit="D") <= max_date:
            valid_cutoffs.append(cutoff)

    if not valid_cutoffs:
        print("   Warning: No valid cutoff dates for evaluation. Using training cutoff only.")
        valid_cutoffs = [config.DEFAULT_CUTOFF_DATE]

    print(f"   Evaluating {len(valid_cutoffs)} cutoff dates...")

    # Re-engineer features on full data for evaluation
    full_features = engineer.create_all_features(raw_data)

    eval_results = evaluate_ensemble_forecasts(
        ensemble=ensemble,
        features_df=full_features,
        cutoff_dates=valid_cutoffs,
        actual_data=raw_data
    )

    # Step 6: Compile results
    print("\n6. Compiling results...")

    baseline_mape = 59.0  # From previous raw model evaluation

    results = {
        "step": "2_ratio_target_transformation",
        "target_mode": target_mode,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ratio_clip_min": getattr(config, 'RATIO_CLIP_MIN', 0.05),
            "ratio_clip_max": getattr(config, 'RATIO_CLIP_MAX', 10.0),
            "ratio_min_denominator": getattr(config, 'RATIO_MIN_DENOMINATOR', 5),
            "floor_constraint": True,
            "floor_ratio": 0.3
        },
        "overall_metrics": eval_results['overall_metrics'],
        "by_horizon": eval_results['by_horizon'],
        "train_val_gap": train_val_gap,
        "comparison_to_baseline": {
            "baseline_overall_mape": baseline_mape,
            "new_overall_mape": eval_results['overall_metrics']['mape'],
            "mape_improvement_pct": ((baseline_mape - eval_results['overall_metrics']['mape']) /
                                     baseline_mape * 100)
        },
        "evaluation_details": {
            "cutoff_dates_evaluated": valid_cutoffs,
            "n_forecasts": eval_results['overall_metrics']['n_forecasts']
        }
    }

    # Save results
    output_file = os.path.join(output_dir, f"step2_{target_mode}_transform.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Target Mode: {target_mode}")
    print(f"\nOverall Metrics:")
    print(f"  MAPE: {results['overall_metrics']['mape']:.2f}%")
    print(f"  MAE:  {results['overall_metrics']['mae']:.2f}")
    print(f"  RMSE: {results['overall_metrics']['rmse']:.2f}")
    print(f"\nBy Horizon:")
    for h, metrics in results['by_horizon'].items():
        print(f"  Horizon {h}: MAPE={metrics['mape']:.2f}%, MAE={metrics['mae']:.2f}")
    print(f"\nComparison to Baseline (raw mode):")
    print(f"  Baseline MAPE: {baseline_mape:.2f}%")
    print(f"  New MAPE:      {results['overall_metrics']['mape']:.2f}%")
    improvement = results['comparison_to_baseline']['mape_improvement_pct']
    if improvement > 0:
        print(f"  Improvement:   {improvement:.2f}% BETTER")
    else:
        print(f"  Change:        {-improvement:.2f}% WORSE")
    print("=" * 70)

    return results


def main():
    """Run evaluation for ratio mode, with log mode as fallback if issues occur."""

    output_dir = "outputs/performance_tracking"
    os.makedirs(output_dir, exist_ok=True)

    # Try ratio mode first
    print("\n" + "#" * 70)
    print("# STEP 2: TARGET TRANSFORMATION EVALUATION")
    print("#" * 70)

    try:
        ratio_results = run_evaluation("ratio", output_dir)

        # Check if ratio mode has issues (very high MAPE or NaN values)
        ratio_mape = ratio_results['overall_metrics']['mape']

        if np.isnan(ratio_mape) or ratio_mape > 200:
            print("\n" + "!" * 70)
            print("! WARNING: Ratio mode shows potential instability")
            print("! Running log mode as alternative...")
            print("!" * 70)

            log_results = run_evaluation("log", output_dir)

            # Compare and recommend
            print("\n" + "=" * 70)
            print("COMPARISON: RATIO vs LOG MODE")
            print("=" * 70)
            print(f"Ratio MAPE: {ratio_mape:.2f}%")
            print(f"Log MAPE:   {log_results['overall_metrics']['mape']:.2f}%")

            if log_results['overall_metrics']['mape'] < ratio_mape:
                print("\nRECOMMENDATION: Use LOG mode")
            else:
                print("\nRECOMMENDATION: Use RATIO mode (despite potential instability)")

    except Exception as e:
        print(f"\nError with ratio mode: {e}")
        print("Falling back to log mode...")

        log_results = run_evaluation("log", output_dir)

    # Also run raw mode for comparison
    print("\n" + "#" * 70)
    print("# RUNNING RAW MODE FOR COMPARISON")
    print("#" * 70)

    try:
        raw_results = run_evaluation("raw", output_dir)
    except Exception as e:
        print(f"Error with raw mode: {e}")

    print("\n" + "#" * 70)
    print("# EVALUATION COMPLETE")
    print("#" * 70)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
