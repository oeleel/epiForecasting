"""
Evaluate feature engineering v2 changes for flu forecasting.

This script tests the impact of:
- Removing 7 low-importance features (based on SHAP analysis)
- Adding 8 new trend-capturing features
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble
from src.train import print_train_val_gap
from src import config


def run_feature_evaluation(output_dir: str = "outputs/performance_tracking") -> Dict:
    """
    Run full evaluation pipeline with v2 features.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary with complete evaluation results
    """
    print("=" * 70)
    print("STEP 4: FEATURE ENGINEERING EVALUATION")
    print("=" * 70)
    print(f"Feature version: {config.FEATURE_VERSION}")
    print(f"Features removed: {config.FEATURES_REMOVED}")
    print(f"Features added: {config.FEATURES_ADDED}")
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

    # Step 3: Engineer features with v2
    print("\n3. Engineering features (v2)...")
    engineer = FeatureEngineer(feature_version=config.FEATURE_VERSION)
    features_df = engineer.create_all_features(training_data)
    feature_cols = engineer.get_feature_columns(features_df)
    total_features = len(feature_cols)
    print(f"   Total feature count: {total_features}")

    # Step 4: Train ensemble
    print(f"\n4. Training DirectForecastEnsemble...")
    params = config.XGBOOST_PARAMS.copy()
    ensemble = DirectForecastEnsemble(
        forecast_horizon=4,
        model_params=params,
        target_mode=config.TARGET_MODE
    )
    training_results = ensemble.train(features_df, validation_split=0.2)

    # Step 5: Print train/val gap analysis
    gap_analysis = print_train_val_gap(training_results, "V2 Features")

    # Step 6: Evaluate on held-out period
    print("\n5. Evaluating forecasts on held-out period...")

    evaluation_cutoffs = [
        "2024-11-02", "2024-11-09", "2024-11-16", "2024-11-23",
        "2024-11-30", "2024-12-07", "2024-12-14", "2024-12-21",
        "2024-12-28", "2025-01-04", "2025-01-11",
    ]

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

    # Extract train/val gap
    train_val_gap = {}
    for h in range(1, 5):
        h_results = training_results['horizons'].get(h, {})
        train_mae = h_results.get('train_mae', 0)
        val_mae_transformed = h_results.get('val_mae', 0)
        val_mae_counts = h_results.get('val_mae_counts', val_mae_transformed)
        gap_ratio = val_mae_transformed / train_mae if train_mae > 0 else None

        train_val_gap[str(h)] = {
            'train_mae': float(train_mae) if train_mae else None,
            'val_mae': float(val_mae_transformed) if val_mae_transformed else None,
            'val_mae_counts': float(val_mae_counts) if val_mae_counts else None,
            'gap_ratio': float(gap_ratio) if gap_ratio else None
        }

    # Load step 3 results for comparison
    step3_mape = None
    step3_path = os.path.join(output_dir, "step3_regularization.json")
    if os.path.exists(step3_path):
        with open(step3_path, 'r') as f:
            step3_results = json.load(f)
            step3_mape = step3_results.get('overall_metrics', {}).get('mape')

    # Compile results
    baseline_mape = 59.0
    mape_improvement_baseline = ((baseline_mape - overall_mape) / baseline_mape) * 100

    results = {
        "step": "4_feature_engineering",
        "feature_version": config.FEATURE_VERSION,
        "features_removed": config.FEATURES_REMOVED,
        "features_added": config.FEATURES_ADDED,
        "total_feature_count": total_features,
        "timestamp": datetime.now().isoformat(),
        "overall_metrics": {
            "mape": float(overall_mape),
            "mae": float(overall_mae),
            "rmse": float(overall_rmse),
            "n_forecasts": len(results_df)
        },
        "by_horizon": by_horizon,
        "train_val_gap": train_val_gap,
        "comparison_to_baseline": {
            "baseline_overall_mape": baseline_mape,
            "new_overall_mape": float(overall_mape),
            "mape_improvement_pct": round(mape_improvement_baseline, 2)
        },
        "comparison_to_previous_step": {
            "step3_overall_mape": step3_mape,
            "new_overall_mape": float(overall_mape),
            "incremental_improvement_pct": round(((step3_mape - overall_mape) / step3_mape) * 100, 2) if step3_mape else None
        }
    }

    # Save results
    output_file = os.path.join(output_dir, "step4_feature_engineering.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING V2 RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFeature Changes:")
    print(f"  Removed: {len(config.FEATURES_REMOVED)} features")
    for f in config.FEATURES_REMOVED:
        print(f"    - {f}")
    print(f"  Added: {len(config.FEATURES_ADDED)} features")
    for f in config.FEATURES_ADDED:
        print(f"    + {f}")
    print(f"  Total features: {total_features}")

    print(f"\nOverall Metrics:")
    print(f"  MAPE: {overall_mape:.2f}%")
    print(f"  MAE:  {overall_mae:.2f}")
    print(f"  RMSE: {overall_rmse:.2f}")

    print(f"\nBy Horizon:")
    for h, metrics in by_horizon.items():
        print(f"  Horizon {h}: MAPE={metrics['mape']:.2f}%, MAE={metrics['mae']:.2f}")

    print(f"\nTrain/Val Gap:")
    for h, gap in train_val_gap.items():
        ratio = gap['gap_ratio']
        print(f"  Horizon {h}: Gap={ratio:.1f}x")

    print(f"\nComparison to Baseline:")
    print(f"  Baseline MAPE: {baseline_mape:.2f}%")
    print(f"  New MAPE:      {overall_mape:.2f}%")
    if mape_improvement_baseline > 0:
        print(f"  Improvement:   {mape_improvement_baseline:.2f}% BETTER")
    else:
        print(f"  Change:        {-mape_improvement_baseline:.2f}% WORSE")

    if step3_mape:
        print(f"\nComparison to Step 3 (Regularization):")
        print(f"  Step 3 MAPE: {step3_mape:.2f}%")
        print(f"  New MAPE:    {overall_mape:.2f}%")
        incremental = ((step3_mape - overall_mape) / step3_mape) * 100
        if incremental > 0:
            print(f"  Incremental: {incremental:.2f}% BETTER")
        else:
            print(f"  Incremental: {-incremental:.2f}% WORSE")

    print("=" * 70)

    return results


if __name__ == "__main__":
    run_feature_evaluation()
