"""
Step 6: Quantile Regression for Prediction Intervals

This script:
1. Trains QuantileDirectForecastEnsemble (4 horizons x 5 quantiles = 20 models)
2. Generates quantile forecasts with prediction intervals
3. Evaluates coverage, calibration, and interval scores
4. Saves comprehensive performance tracking to step6_quantile_regression.json
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import QuantileDirectForecastEnsemble
from src.evaluate import QuantileEvaluator, ModelEvaluator
from src import config


def run_quantile_evaluation(ensemble: QuantileDirectForecastEnsemble,
                           features_df: pd.DataFrame,
                           cutoff_dates: List[str],
                           actual_data: pd.DataFrame) -> Dict:
    """
    Run quantile evaluation across multiple cutoff dates.

    Args:
        ensemble: Trained QuantileDirectForecastEnsemble
        features_df: Feature-engineered DataFrame
        cutoff_dates: List of cutoff dates to evaluate
        actual_data: Full dataset with actual values

    Returns:
        Dictionary with comprehensive evaluation results
    """
    all_forecasts = []

    for cutoff_date in cutoff_dates:
        cutoff_dt = pd.to_datetime(cutoff_date)

        try:
            forecasts = ensemble.generate_forecasts(
                data=features_df,
                cutoff_date=cutoff_date,
                use_floor_constraint=True,
                floor_ratio=0.3
            )
            all_forecasts.append(forecasts)
        except Exception as e:
            print(f"   Skipping {cutoff_date}: {e}")
            continue

    if not all_forecasts:
        return {'error': 'No forecasts could be generated'}

    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Run quantile evaluation
    evaluator = QuantileEvaluator()
    results = evaluator.evaluate_quantile_forecasts(combined_forecasts, actual_data)

    return results


def main():
    """Run Step 6: Quantile Regression evaluation."""

    output_dir = "outputs/performance_tracking"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("# STEP 6: QUANTILE REGRESSION FOR PREDICTION INTERVALS")
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

    # Step 3: Load V3 params if available, otherwise use V2
    print("\n3. Loading model parameters...")
    v3_params_file = os.path.join(output_dir, "xgboost_params_v3.json")
    if os.path.exists(v3_params_file):
        with open(v3_params_file, 'r') as f:
            model_params = json.load(f)
        print(f"   Using V3 parameters from Optuna optimization")
    else:
        model_params = config.XGBOOST_PARAMS_V2.copy()
        print(f"   Using V2 parameters (V3 not found)")

    # Remove early_stopping_rounds if present (can cause issues with quantile)
    model_params_clean = model_params.copy()
    if 'early_stopping_rounds' in model_params_clean:
        del model_params_clean['early_stopping_rounds']

    # Step 4: Initialize and train quantile ensemble
    print("\n4. Initializing QuantileDirectForecastEnsemble...")
    print(f"   Quantiles: {config.QUANTILES}")
    print(f"   Horizons: {config.FORECAST_HORIZON}")
    print(f"   Total models: {config.FORECAST_HORIZON * len(config.QUANTILES)}")

    ensemble = QuantileDirectForecastEnsemble(
        forecast_horizon=config.FORECAST_HORIZON,
        model_params=model_params_clean,
        target_mode=config.TARGET_MODE,
        quantiles=config.QUANTILES
    )

    # Filter training data
    training_cutoff = config.DEFAULT_CUTOFF_DATE
    training_data = loader.load_and_preprocess(training_cutoff)
    training_features = engineer.create_all_features(training_data)

    print("\n5. Training quantile models...")
    training_results = ensemble.train(training_features, validation_split=0.2)

    # Step 5: Get evaluation cutoff dates
    print("\n6. Setting up evaluation cutoffs...")
    cutoff_dates = config.VALIDATION_CUTOFFS_EXPANDED

    # Filter cutoffs to those with sufficient data
    valid_cutoffs = []
    min_date = features_df['date'].min()
    max_date = features_df['date'].max()

    for cutoff in cutoff_dates:
        cutoff_dt = pd.to_datetime(cutoff)
        if (cutoff_dt - pd.Timedelta(weeks=52) >= min_date and
            cutoff_dt + pd.Timedelta(weeks=4) <= max_date):
            valid_cutoffs.append(cutoff)

    # Use cutoffs that have future data for evaluation
    eval_cutoffs = [c for c in valid_cutoffs if pd.to_datetime(c) >= pd.to_datetime("2024-01-01")]
    print(f"   Evaluation cutoffs: {len(eval_cutoffs)}")

    # Step 6: Run quantile evaluation
    print("\n7. Running quantile evaluation...")
    eval_results = run_quantile_evaluation(
        ensemble=ensemble,
        features_df=features_df,
        cutoff_dates=eval_cutoffs,
        actual_data=raw_data
    )

    # Print summary
    evaluator = QuantileEvaluator()
    evaluator.evaluation_results = eval_results
    evaluator.print_quantile_summary()

    # Step 7: Load step 5 results for comparison
    step5_mape = None
    step5_path = os.path.join(output_dir, "step5_expanded_validation.json")
    if os.path.exists(step5_path):
        with open(step5_path, 'r') as f:
            step5_results = json.load(f)
            step5_mape = step5_results.get('overall_metrics', {}).get('mape')

    # Step 8: Compile final results
    baseline_mape = 59.0
    point_metrics = eval_results.get('point_prediction_metrics', {})
    new_mape = point_metrics.get('mape', 0)
    mape_improvement = ((baseline_mape - new_mape) / baseline_mape) * 100 if new_mape > 0 else 0

    # Build by_horizon for point predictions
    by_horizon = {}
    for h, m in eval_results.get('metrics_by_horizon', {}).items():
        by_horizon[str(h)] = {
            'mape': m.get('mape'),
            'mae': m.get('mae'),
            'coverage_90pct': m.get('coverage_90pct'),
            'mean_interval_width_90pct': m.get('mean_interval_width_90pct'),
            'winkler_score_90pct': m.get('winkler_score_90pct')
        }

    results = {
        "step": "6_quantile_regression",
        "timestamp": datetime.now().isoformat(),
        "n_quantiles": len(config.QUANTILES),
        "quantiles": config.QUANTILES,
        "n_total_models": config.FORECAST_HORIZON * len(config.QUANTILES),
        "point_prediction_metrics": {
            "overall": {
                "mape": point_metrics.get('mape'),
                "mae": point_metrics.get('mae'),
                "rmse": point_metrics.get('rmse'),
                "n_forecasts": eval_results.get('n_forecasts_evaluated')
            },
            "by_horizon": by_horizon
        },
        "quantile_metrics": eval_results.get('quantile_metrics', {}),
        "comparison_to_baseline": {
            "baseline_overall_mape": baseline_mape,
            "new_overall_mape": new_mape,
            "mape_improvement_pct": round(mape_improvement, 2)
        },
        "comparison_to_previous_step": {
            "step5_overall_mape": step5_mape,
            "new_overall_mape": new_mape,
            "incremental_improvement_pct": round(((step5_mape - new_mape) / step5_mape) * 100, 2) if step5_mape else None
        }
    }

    # Save results
    output_file = os.path.join(output_dir, "step6_quantile_regression.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print final summary
    print("\n" + "=" * 70)
    print("STEP 6 RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Quantiles: {config.QUANTILES}")
    print(f"  Total models trained: {config.FORECAST_HORIZON * len(config.QUANTILES)}")

    print(f"\nPoint Prediction (Median) Metrics:")
    print(f"  MAPE: {new_mape:.2f}%")
    print(f"  MAE:  {point_metrics.get('mae', 0):.2f}")
    print(f"  RMSE: {point_metrics.get('rmse', 0):.2f}")

    qm = results['quantile_metrics']
    print(f"\n90% Prediction Interval (q05-q95):")
    print(f"  Coverage: {qm.get('coverage_90pct', 0)*100:.1f}% (target: 90%)")
    print(f"  Mean Width: {qm.get('mean_interval_width_90pct', 0):.1f}")
    print(f"  Winkler Score: {qm.get('mean_winkler_score_90pct', 0):.1f}")

    print(f"\n50% Prediction Interval (q25-q75):")
    print(f"  Coverage: {qm.get('coverage_50pct', 0)*100:.1f}% (target: 50%)")
    print(f"  Mean Width: {qm.get('mean_interval_width_50pct', 0):.1f}")

    print(f"\nCalibration:")
    for q, val in qm.get('calibration', {}).items():
        target = int(q[1:]) / 100
        status = "OK" if abs(val - target) < 0.10 else "MISCALIBRATED"
        print(f"  {q}: {val:.3f} (target: {target:.2f}) - {status}")

    print(f"\nBy Horizon:")
    for h, m in sorted(by_horizon.items()):
        print(f"  Horizon {h}: MAPE={m.get('mape', 0):.1f}%, "
              f"Coverage={m.get('coverage_90pct', 0)*100:.1f}%, "
              f"Width={m.get('mean_interval_width_90pct', 0):.1f}")

    print(f"\nComparison to Baseline:")
    print(f"  Baseline MAPE: {baseline_mape:.2f}%")
    print(f"  New MAPE:      {new_mape:.2f}%")
    if mape_improvement > 0:
        print(f"  Improvement:   {mape_improvement:.2f}% BETTER")
    else:
        print(f"  Change:        {-mape_improvement:.2f}% WORSE")

    if step5_mape:
        incremental = ((step5_mape - new_mape) / step5_mape) * 100
        print(f"\nComparison to Step 5:")
        print(f"  Step 5 MAPE: {step5_mape:.2f}%")
        print(f"  New MAPE:    {new_mape:.2f}%")
        if incremental > 0:
            print(f"  Incremental: {incremental:.2f}% BETTER")
        elif incremental < 0:
            print(f"  Incremental: {-incremental:.2f}% WORSE")
        else:
            print(f"  Incremental: No change")

    print("=" * 70)

    # Save sample forecasts as CSV
    print("\nGenerating sample forecast CSV...")
    sample_forecasts = ensemble.generate_forecasts(
        data=features_df,
        cutoff_date="2024-11-02",
        locations=['US', '06', '48', '36', '12']  # US, CA, TX, NY, FL
    )

    # Select columns for output
    output_cols = ['location', 'forecast_date', 'forecast_week',
                   'predicted', 'predicted_q05', 'predicted_q25',
                   'predicted_q75', 'predicted_q95']
    sample_forecasts = sample_forecasts[output_cols]

    csv_path = os.path.join(output_dir, "sample_quantile_forecasts.csv")
    sample_forecasts.to_csv(csv_path, index=False)
    print(f"Sample forecasts saved to: {csv_path}")

    return results


if __name__ == "__main__":
    main()
