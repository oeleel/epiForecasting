"""
Evaluate regularization improvements for flu forecasting.

This script tests the V2 regularized parameters against the baseline
to measure the reduction in overfitting (train/val gap).
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
from src.train import print_train_val_gap
from src import config


def evaluate_with_params(params: Dict, target_mode: str = "log",
                         output_dir: str = "outputs/performance_tracking") -> Dict:
    """
    Run full evaluation pipeline with specified parameters.

    Args:
        params: XGBoost parameters to use
        target_mode: Target transformation mode
        output_dir: Directory to save results

    Returns:
        Dictionary with complete evaluation results
    """
    print("=" * 70)
    print("REGULARIZATION EVALUATION")
    print("=" * 70)
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("\n1. Loading data...")
    loader = FluDataLoader()
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

    # Step 4: Train ensemble with specified params
    print(f"\n4. Training DirectForecastEnsemble (target_mode={target_mode})...")
    ensemble = DirectForecastEnsemble(
        forecast_horizon=4,
        model_params=params,
        target_mode=target_mode
    )
    training_results = ensemble.train(features_df, validation_split=0.2)

    # Step 5: Print train/val gap analysis
    gap_analysis = print_train_val_gap(training_results, "V2 Regularized Parameters")

    # Step 6: Evaluate on held-out period
    print("\n5. Evaluating forecasts on held-out period...")

    evaluation_cutoffs = [
        "2024-11-02", "2024-11-09", "2024-11-16", "2024-11-23",
        "2024-11-30", "2024-12-07", "2024-12-14", "2024-12-21",
        "2024-12-28", "2025-01-04", "2025-01-11",
    ]

    # Filter cutoffs to those where we have sufficient future data
    valid_cutoffs = []
    max_date = raw_data['date'].max()
    for cutoff in evaluation_cutoffs:
        cutoff_dt = pd.to_datetime(cutoff)
        if cutoff_dt + pd.Timedelta(7 * (4), unit="D") <= max_date:
            valid_cutoffs.append(cutoff)

    if not valid_cutoffs:
        valid_cutoffs = [config.DEFAULT_CUTOFF_DATE]

    print(f"   Evaluating {len(valid_cutoffs)} cutoff dates...")

    # Re-engineer features on full data for evaluation
    full_features = engineer.create_all_features(raw_data)

    # Generate and evaluate forecasts
    all_forecasts = []
    for cutoff_date in valid_cutoffs:
        print(f"   Generating forecasts for cutoff {cutoff_date}...")
        forecasts = ensemble.generate_forecasts(
            data=full_features,
            cutoff_date=cutoff_date,
            use_floor_constraint=True,
            floor_ratio=0.3
        )

        cutoff_dt = pd.to_datetime(cutoff_date)
        for _, row in forecasts.iterrows():
            forecast_date = pd.to_datetime(row['forecast_date'])
            location = row['location']
            horizon = row['forecast_week']

            actual_row = raw_data[
                (raw_data['date'] == forecast_date) &
                (raw_data['location'] == location)
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

    results_df = pd.DataFrame(all_forecasts)

    # Calculate metrics
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

    # Extract train/val gap for results
    # IMPORTANT: Compare on same scale (transformed) for fair gap ratio
    train_val_gap = {}
    for h in range(1, 5):
        h_results = training_results['horizons'].get(h, {})
        train_mae = h_results.get('train_mae', 0)
        # Use val_mae (transformed scale) for gap ratio, val_mae_counts for actual performance
        val_mae_transformed = h_results.get('val_mae', 0)
        val_mae_counts = h_results.get('val_mae_counts', val_mae_transformed)
        gap_ratio = val_mae_transformed / train_mae if train_mae > 0 else None

        train_val_gap[str(h)] = {
            'train_mae': float(train_mae) if train_mae else None,
            'val_mae': float(val_mae_transformed) if val_mae_transformed else None,
            'val_mae_counts': float(val_mae_counts) if val_mae_counts else None,
            'gap_ratio': float(gap_ratio) if gap_ratio else None
        }

    return {
        'overall_metrics': {
            'mape': float(overall_mape),
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'n_forecasts': len(results_df)
        },
        'by_horizon': by_horizon,
        'train_val_gap': train_val_gap,
        'gap_analysis': gap_analysis
    }


def main():
    """Run regularization evaluation and save performance tracking."""

    output_dir = "outputs/performance_tracking"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("# STEP 3: REGULARIZATION EVALUATION")
    print("#" * 70)

    # Use V2 parameters
    params = config.XGBOOST_PARAMS_V2.copy()

    eval_results = evaluate_with_params(params, target_mode="log", output_dir=output_dir)

    # Baseline values (from V1 with raw mode)
    baseline_mape = 59.0
    baseline_train_val_gap_h1 = 272.8 / 6.84  # ~39.9x

    # Calculate improvements
    new_mape = eval_results['overall_metrics']['mape']
    new_gap_ratio_h1 = eval_results['train_val_gap']['1']['gap_ratio']

    mape_improvement = ((baseline_mape - new_mape) / baseline_mape) * 100
    gap_improvement = ((baseline_train_val_gap_h1 - new_gap_ratio_h1) / baseline_train_val_gap_h1) * 100 if new_gap_ratio_h1 else None

    # Compile final results
    results = {
        "step": "3_regularization",
        "params_used": params,
        "timestamp": datetime.now().isoformat(),
        "overall_metrics": eval_results['overall_metrics'],
        "by_horizon": eval_results['by_horizon'],
        "train_val_gap": eval_results['train_val_gap'],
        "comparison_to_baseline": {
            "baseline_overall_mape": baseline_mape,
            "baseline_train_val_gap_ratio_h1": round(baseline_train_val_gap_h1, 1),
            "new_overall_mape": new_mape,
            "new_train_val_gap_ratio_h1": new_gap_ratio_h1,
            "mape_improvement_pct": round(mape_improvement, 2),
            "gap_ratio_improvement_pct": round(gap_improvement, 2) if gap_improvement else None
        }
    }

    # Save results
    output_file = os.path.join(output_dir, "step3_regularization.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("REGULARIZATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  MAPE: {new_mape:.2f}%")
    print(f"  MAE:  {eval_results['overall_metrics']['mae']:.2f}")
    print(f"  RMSE: {eval_results['overall_metrics']['rmse']:.2f}")

    print(f"\nBy Horizon:")
    for h, metrics in eval_results['by_horizon'].items():
        print(f"  Horizon {h}: MAPE={metrics['mape']:.2f}%, MAE={metrics['mae']:.2f}")

    print(f"\nTrain/Val Gap (Overfitting Indicator - on transformed scale):")
    for h, gap in eval_results['train_val_gap'].items():
        ratio = gap['gap_ratio']
        val_counts = gap.get('val_mae_counts', gap['val_mae'])
        print(f"  Horizon {h}: Train={gap['train_mae']:.4f}, Val={gap['val_mae']:.4f}, Gap={ratio:.1f}x, Val(counts)={val_counts:.1f}")

    print(f"\nComparison to Baseline (V1 params):")
    print(f"  Baseline MAPE: {baseline_mape:.2f}%")
    print(f"  New MAPE:      {new_mape:.2f}%")
    if mape_improvement > 0:
        print(f"  MAPE Change:   {mape_improvement:.2f}% BETTER")
    else:
        print(f"  MAPE Change:   {-mape_improvement:.2f}% WORSE")

    print(f"\n  Baseline Gap Ratio (H1): {baseline_train_val_gap_h1:.1f}x")
    print(f"  New Gap Ratio (H1):      {new_gap_ratio_h1:.1f}x")
    if gap_improvement and gap_improvement > 0:
        print(f"  Gap Reduction:           {gap_improvement:.1f}% BETTER (less overfitting)")
    elif gap_improvement:
        print(f"  Gap Change:              {-gap_improvement:.1f}% WORSE (more overfitting)")

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
