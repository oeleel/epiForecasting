"""
Step 7: Location Clustering for Homogeneous Models

This script:
1. Clusters locations based on flu hospitalization patterns
2. Trains cluster-specific models (4 horizons x n_clusters)
3. Compares unified vs clustered model performance
4. Saves comprehensive performance tracking to step7_location_clustering.json
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import QuantileDirectForecastEnsemble, ClusteredDirectForecastEnsemble
from src.location_clustering import LocationClusterer
from src.evaluate import ClusteredEvaluator, ModelEvaluator
from src import config


def run_evaluation(ensemble, features_df: pd.DataFrame,
                   cutoff_dates: List[str], actual_data: pd.DataFrame,
                   is_clustered: bool = False) -> Dict:
    """
    Run forecast evaluation across multiple cutoff dates.

    Args:
        ensemble: Trained forecast ensemble
        features_df: Feature-engineered DataFrame
        cutoff_dates: List of cutoff dates to evaluate
        actual_data: Full dataset with actual values
        is_clustered: Whether this is a clustered model

    Returns:
        Dictionary with forecasts and basic metrics
    """
    all_forecasts = []

    for cutoff_date in cutoff_dates:
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
        return {'error': 'No forecasts could be generated', 'forecasts': pd.DataFrame()}

    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Basic evaluation
    evaluator = ModelEvaluator()

    # Prepare prediction column
    if 'predicted' in combined_forecasts.columns:
        pred_col = 'predicted'
    elif 'predicted_q50' in combined_forecasts.columns:
        pred_col = 'predicted_q50'
    else:
        pred_col = 'forecast'

    # Rename for merge
    forecasts_eval = combined_forecasts.copy()
    forecasts_eval['forecast'] = forecasts_eval[pred_col]

    results = evaluator.evaluate_forecasts(forecasts_eval, actual_data)
    results['forecasts'] = combined_forecasts

    return results


def main():
    """Run Step 7: Location Clustering evaluation."""

    output_dir = "outputs/performance_tracking"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("# STEP 7: LOCATION CLUSTERING FOR HOMOGENEOUS MODELS")
    print("#" * 70)

    # Step 1: Load data
    print("\n1. Loading data...")
    loader = FluDataLoader()
    raw_data = loader.fetch_data()
    print(f"   Total records: {len(raw_data)}")
    print(f"   Date range: {raw_data['date'].min()} to {raw_data['date'].max()}")
    print(f"   Unique locations: {raw_data['location'].nunique()}")

    # Step 2: Prepare features
    print("\n2. Preparing features...")
    engineer = FeatureEngineer(feature_version=config.FEATURE_VERSION)
    features_df = engineer.create_all_features(raw_data)
    print(f"   Features shape: {features_df.shape}")

    # Step 3: Load model parameters
    print("\n3. Loading model parameters...")
    v3_params_file = os.path.join(output_dir, "xgboost_params_v3.json")
    if os.path.exists(v3_params_file):
        with open(v3_params_file, 'r') as f:
            model_params = json.load(f)
        print(f"   Using V3 parameters from Optuna optimization")
    else:
        model_params = config.XGBOOST_PARAMS_V2.copy()
        print(f"   Using V2 parameters (V3 not found)")

    # Remove early_stopping_rounds if present
    model_params_clean = model_params.copy()
    if 'early_stopping_rounds' in model_params_clean:
        del model_params_clean['early_stopping_rounds']

    # Step 4: Filter training data
    training_cutoff = config.DEFAULT_CUTOFF_DATE
    training_data = loader.load_and_preprocess(training_cutoff)
    training_features = engineer.create_all_features(training_data)

    # Step 5: Set up evaluation cutoffs
    print("\n4. Setting up evaluation cutoffs...")
    cutoff_dates = config.VALIDATION_CUTOFFS_EXPANDED

    valid_cutoffs = []
    min_date = features_df['date'].min()
    max_date = features_df['date'].max()

    for cutoff in cutoff_dates:
        cutoff_dt = pd.to_datetime(cutoff)
        if (cutoff_dt - pd.Timedelta(7 * (52), unit="D") >= min_date and
            cutoff_dt + pd.Timedelta(7 * (4), unit="D") <= max_date):
            valid_cutoffs.append(cutoff)

    eval_cutoffs = [c for c in valid_cutoffs if pd.to_datetime(c) >= pd.to_datetime("2024-01-01")]
    print(f"   Evaluation cutoffs: {len(eval_cutoffs)}")

    # Step 6: Train UNIFIED model (baseline for comparison)
    print("\n5. Training UNIFIED model (baseline)...")
    unified_ensemble = QuantileDirectForecastEnsemble(
        forecast_horizon=config.FORECAST_HORIZON,
        model_params=model_params_clean,
        target_mode=config.TARGET_MODE,
        quantiles=config.QUANTILES
    )
    unified_ensemble.train(training_features, validation_split=0.2)

    # Generate unified forecasts
    print("\n6. Generating unified model forecasts...")
    unified_results = run_evaluation(
        ensemble=unified_ensemble,
        features_df=features_df,
        cutoff_dates=eval_cutoffs,
        actual_data=raw_data,
        is_clustered=False
    )
    unified_forecasts = unified_results.get('forecasts', pd.DataFrame())

    # Step 7: Train CLUSTERED models
    print("\n7. Training CLUSTERED models...")
    n_clusters = config.N_LOCATION_CLUSTERS
    print(f"   Number of clusters: {n_clusters}")

    clustered_ensemble = ClusteredDirectForecastEnsemble(
        forecast_horizon=config.FORECAST_HORIZON,
        model_params=model_params_clean,
        target_mode=config.TARGET_MODE,
        n_clusters=n_clusters,
        enable_quantiles=True,
        quantiles=config.QUANTILES
    )

    # Fit clusters on all data
    clustered_ensemble.fit_clusters(raw_data)
    location_to_cluster = clustered_ensemble.location_to_cluster

    # Train on training data
    clustered_ensemble.train(training_features, validation_split=0.2)

    # Generate clustered forecasts
    print("\n8. Generating clustered model forecasts...")
    clustered_results = run_evaluation(
        ensemble=clustered_ensemble,
        features_df=features_df,
        cutoff_dates=eval_cutoffs,
        actual_data=raw_data,
        is_clustered=True
    )
    clustered_forecasts = clustered_results.get('forecasts', pd.DataFrame())

    # Step 8: Compare models
    print("\n9. Comparing unified vs clustered models...")
    clustered_evaluator = ClusteredEvaluator(location_to_cluster)

    # Add cluster column to unified forecasts for fair comparison
    if len(unified_forecasts) > 0:
        unified_forecasts['cluster'] = unified_forecasts['location'].map(location_to_cluster)

    comparison = clustered_evaluator.compare_models(
        unified_forecasts=unified_forecasts,
        clustered_forecasts=clustered_forecasts,
        actual_data=raw_data,
        is_quantile=True
    )

    # Evaluate clustered model with breakdown
    clustered_eval = clustered_evaluator.evaluate_forecasts(
        forecasts=clustered_forecasts,
        actual_data=raw_data,
        is_quantile=True
    )

    # Print summaries
    clustered_evaluator.print_cluster_summary(clustered_eval)
    clustered_evaluator.print_comparison_summary(comparison)

    # Step 9: Get cluster descriptions
    cluster_descriptions = {}
    for cluster_id, profile in clustered_ensemble.cluster_profiles.items():
        desc = clustered_ensemble.clusterer.get_cluster_description(cluster_id)
        cluster_descriptions[int(cluster_id)] = {
            'description': desc,
            'n_locations': profile['n_locations'],
            'locations': profile['locations']
        }

    # Step 10: Load step 6 results for comparison
    step6_mape = None
    step6_path = os.path.join(output_dir, "step6_quantile_regression.json")
    if os.path.exists(step6_path):
        with open(step6_path, 'r') as f:
            step6_results = json.load(f)
            step6_mape = step6_results.get('point_prediction_metrics', {}).get('overall', {}).get('mape')

    # Step 11: Compile final results
    baseline_mape = 59.0
    unified_mape = comparison['unified_model']['mape']
    clustered_mape = comparison['clustered_model']['mape']

    # Build metrics by horizon
    by_horizon = {}
    for h, m in clustered_eval.get('metrics_by_horizon', {}).items():
        by_horizon[str(h)] = {
            'mape': m.get('mape'),
            'mae': m.get('mae'),
            'coverage_90pct': m.get('coverage_90pct') if 'coverage_90pct' in m else None
        }

    # Build metrics by cluster
    by_cluster = {}
    for c, m in clustered_eval.get('metrics_by_cluster', {}).items():
        by_cluster[str(c)] = {
            'mape': m.get('mape'),
            'mae': m.get('mae'),
            'n_locations': m.get('n_locations'),
            'coverage_90pct': m.get('coverage_90pct') if 'coverage_90pct' in m else None
        }

    results = {
        "step": "7_location_clustering",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_clusters": n_clusters,
            "min_cluster_size": config.MIN_CLUSTER_SIZE,
            "clustering_method": clustered_ensemble.clusterer.clustering_method,
            "enable_quantiles": True,
            "quantiles": config.QUANTILES,
            "n_total_models": n_clusters * config.FORECAST_HORIZON * len(config.QUANTILES)
        },
        "cluster_info": cluster_descriptions,
        "unified_model_metrics": {
            "mape": unified_mape,
            "mae": comparison['unified_model']['mae'],
            "rmse": comparison['unified_model']['rmse'],
            "n_forecasts": comparison['unified_model']['n_forecasts']
        },
        "clustered_model_metrics": {
            "overall": {
                "mape": clustered_mape,
                "mae": comparison['clustered_model']['mae'],
                "rmse": comparison['clustered_model']['rmse'],
                "n_forecasts": comparison['clustered_model']['n_forecasts']
            },
            "by_horizon": by_horizon,
            "by_cluster": by_cluster
        },
        "model_comparison": {
            "mape_reduction_pct": comparison['improvement']['mape_reduction_pct'],
            "mae_reduction_pct": comparison['improvement']['mae_reduction_pct'],
            "clustered_better": bool(comparison['improvement']['clustered_better'])
        },
        "comparison_to_baseline": {
            "baseline_mape": baseline_mape,
            "clustered_mape": clustered_mape,
            "improvement_pct": round(((baseline_mape - clustered_mape) / baseline_mape) * 100, 2)
        },
        "comparison_to_previous_step": {
            "step6_mape": step6_mape,
            "clustered_mape": clustered_mape,
            "incremental_improvement_pct": round(((step6_mape - clustered_mape) / step6_mape) * 100, 2) if step6_mape else None
        }
    }

    # Save results
    output_file = os.path.join(output_dir, "step7_location_clustering.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Save clustering results
    clustered_ensemble.clusterer.save_results(os.path.join(output_dir, "../location_clusters"))

    # Print final summary
    print("\n" + "=" * 70)
    print("STEP 7 RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Clustering method: {clustered_ensemble.clusterer.clustering_method}")
    print(f"  Total models trained: {n_clusters * config.FORECAST_HORIZON * len(config.QUANTILES)}")

    print(f"\nCluster Breakdown:")
    for c_id, c_info in cluster_descriptions.items():
        print(f"  Cluster {c_id}: {c_info['description']}")
        locs = c_info['locations'][:5]
        suffix = '...' if len(c_info['locations']) > 5 else ''
        print(f"    Locations: {', '.join(locs)}{suffix}")

    print(f"\nUnified Model (Baseline):")
    print(f"  MAPE: {unified_mape:.2f}%")
    print(f"  MAE:  {comparison['unified_model']['mae']:.2f}")

    print(f"\nClustered Model:")
    print(f"  MAPE: {clustered_mape:.2f}%")
    print(f"  MAE:  {comparison['clustered_model']['mae']:.2f}")

    print(f"\nComparison:")
    improvement = comparison['improvement']
    winner = "CLUSTERED" if improvement['clustered_better'] else "UNIFIED"
    print(f"  MAPE Reduction: {improvement['mape_reduction_pct']:.2f}%")
    print(f"  MAE Reduction:  {improvement['mae_reduction_pct']:.2f}%")
    print(f"  Winner: {winner}")

    print(f"\nMetrics by Cluster:")
    for c, m in sorted(by_cluster.items()):
        cov_str = f", Coverage={m['coverage_90pct']*100:.1f}%" if m.get('coverage_90pct') else ""
        print(f"  Cluster {c}: MAPE={m.get('mape', 0):.1f}%, MAE={m.get('mae', 0):.1f}{cov_str}")

    if step6_mape:
        incremental = ((step6_mape - clustered_mape) / step6_mape) * 100
        print(f"\nComparison to Step 6:")
        print(f"  Step 6 MAPE: {step6_mape:.2f}%")
        print(f"  New MAPE:    {clustered_mape:.2f}%")
        if incremental > 0:
            print(f"  Incremental: {incremental:.2f}% BETTER")
        elif incremental < 0:
            print(f"  Incremental: {-incremental:.2f}% WORSE")
        else:
            print(f"  Incremental: No change")

    print(f"\nComparison to Baseline:")
    print(f"  Baseline MAPE: {baseline_mape:.2f}%")
    print(f"  New MAPE:      {clustered_mape:.2f}%")
    baseline_improvement = ((baseline_mape - clustered_mape) / baseline_mape) * 100
    if baseline_improvement > 0:
        print(f"  Improvement:   {baseline_improvement:.2f}% BETTER")
    else:
        print(f"  Change:        {-baseline_improvement:.2f}% WORSE")

    print("=" * 70)

    # Save sample forecasts
    print("\nGenerating sample forecast CSV...")
    sample_forecasts = clustered_ensemble.generate_forecasts(
        data=features_df,
        cutoff_date="2024-11-02",
        locations=['US', '06', '48', '36', '12']
    )

    output_cols = ['location', 'cluster', 'forecast_date', 'forecast_week',
                   'predicted', 'predicted_q05', 'predicted_q25',
                   'predicted_q75', 'predicted_q95']
    available_cols = [c for c in output_cols if c in sample_forecasts.columns]
    sample_forecasts = sample_forecasts[available_cols]

    csv_path = os.path.join(output_dir, "sample_clustered_forecasts.csv")
    sample_forecasts.to_csv(csv_path, index=False)
    print(f"Sample forecasts saved to: {csv_path}")

    return results


if __name__ == "__main__":
    main()
