"""
Generate quantile and point hindcasts from November 2024 to April 2025.

Retrains both QuantileDirectForecastEnsemble and DirectForecastEnsemble
at each weekly cutoff. Evaluates with QuantileEvaluator (coverage, calibration,
Winkler, pinball loss, CRPS) and ModelEvaluator (MAE/MAPE).

Outputs saved to outputs/quantile_hindcasts/.

Usage:
    python scripts/generate_quantile_hindcasts.py
    python scripts/generate_quantile_hindcasts.py --start-date 2025-01-01 --end-date 2025-02-28
    python scripts/generate_quantile_hindcasts.py --locations US 06 48
"""

import sys
import os

# Ensure project root is on path when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from typing import List, Optional

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble, QuantileDirectForecastEnsemble
from src.evaluate import ModelEvaluator, QuantileEvaluator
from src.visualization import plot_forecast
from src import config


def generate_quantile_hindcasts(
    start_date: str = "2024-11-02",
    end_date: str = "2025-04-26",
    locations: Optional[List[str]] = None,
    output_dir: str = "outputs/quantile_hindcasts"
) -> dict:
    """
    Generate both quantile and point hindcasts over the specified period.

    Args:
        start_date: Start of hindcast period
        end_date: End of hindcast period
        locations: Specific locations (None = all)
        output_dir: Output directory

    Returns:
        Dict with all_point_forecasts, all_quantile_forecasts DataFrames and evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # Load data
    print("Loading data...")
    loader = FluDataLoader()
    try:
        if os.path.exists('influ_hospit.csv'):
            data = pd.read_csv('influ_hospit.csv')
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            end_dt = pd.to_datetime(end_date)
            data = data[data['date'] <= end_dt].copy()
            print(f"Loaded {len(data)} records from local file")
        else:
            data = loader.load_and_preprocess(end_date)
    except Exception as e:
        print(f"Error loading data: {e}, fetching from CDC...")
        data = loader.load_and_preprocess(end_date)

    engineer = FeatureEngineer()

    # Weekly Saturday cutoffs
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    cutoff_dates = pd.date_range(start=start_dt, end=end_dt, freq='W-SAT')

    all_point_forecasts = []
    all_quantile_forecasts = []

    # Key dates for visualization
    key_cutoffs = ['2024-11-02', '2025-01-11', '2025-03-15']

    for cutoff_date in cutoff_dates:
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        print(f"\n{'='*50}")
        print(f"Cutoff: {cutoff_str}")
        print(f"{'='*50}")

        cutoff_data = data[data['date'] <= cutoff_date].copy()
        if len(cutoff_data) == 0:
            print(f"  No data available, skipping")
            continue

        try:
            features_df = engineer.create_all_features(cutoff_data)

            # --- Point forecasts ---
            print("  Training point ensemble...")
            p_ensemble = DirectForecastEnsemble(forecast_horizon=4)
            p_ensemble.train(features_df)
            p_forecasts = p_ensemble.generate_forecasts(
                features_df, cutoff_str, locations=locations
            )
            p_forecasts['cutoff_date'] = cutoff_str
            all_point_forecasts.append(p_forecasts)
            print(f"  Point: {len(p_forecasts)} forecasts")

            # --- Quantile forecasts ---
            print("  Training quantile ensemble...")
            q_ensemble = QuantileDirectForecastEnsemble(forecast_horizon=4)
            q_ensemble.train(features_df)
            q_forecasts = q_ensemble.generate_forecasts(
                features_df, cutoff_str, locations=locations
            )
            q_forecasts['cutoff_date'] = cutoff_str
            all_quantile_forecasts.append(q_forecasts)
            print(f"  Quantile: {len(q_forecasts)} forecasts")

            # Verify quantile ordering
            ordering_ok = (
                (q_forecasts['predicted_q05'] <= q_forecasts['predicted_q25']).all() and
                (q_forecasts['predicted_q25'] <= q_forecasts['predicted_q50']).all() and
                (q_forecasts['predicted_q50'] <= q_forecasts['predicted_q75']).all() and
                (q_forecasts['predicted_q75'] <= q_forecasts['predicted_q95']).all()
            )
            if not ordering_ok:
                print("  WARNING: Quantile ordering violated!")

            # Plot for key dates
            if cutoff_str in key_cutoffs:
                for loc in ['US']:
                    try:
                        save_path = os.path.join(
                            output_dir, 'plots',
                            f'quantile_{loc}_{cutoff_str}.png'
                        )
                        plot_forecast(
                            location=loc,
                            cutoff_date=cutoff_str,
                            model_type='quantile',
                            data=data,
                            features_df=features_df,
                            ensemble=q_ensemble,
                            forecasts=q_forecasts,
                            save_path=save_path
                        )
                        plt.close('all')
                    except Exception as plot_err:
                        print(f"  Plot error for {loc}: {plot_err}")

        except Exception as e:
            print(f"  Error: {str(e)}")
            continue

    # Combine all forecasts
    if not all_point_forecasts or not all_quantile_forecasts:
        print("ERROR: No forecasts generated")
        return {}

    combined_point = pd.concat(all_point_forecasts, ignore_index=True)
    combined_quantile = pd.concat(all_quantile_forecasts, ignore_index=True)

    # Save CSVs
    point_csv = os.path.join(output_dir, 'point_forecasts_nov_apr.csv')
    quantile_csv = os.path.join(output_dir, 'quantile_forecasts_nov_apr.csv')
    combined_point.to_csv(point_csv, index=False)
    combined_quantile.to_csv(quantile_csv, index=False)
    print(f"\nSaved {len(combined_point)} point forecasts to {point_csv}")
    print(f"Saved {len(combined_quantile)} quantile forecasts to {quantile_csv}")

    # --- Evaluation ---
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Point evaluation
    print("\n--- Point Forecast Evaluation ---")
    point_evaluator = ModelEvaluator()
    point_results = point_evaluator.evaluate_forecasts(combined_point, data)

    if 'error' not in point_results:
        report = point_evaluator.generate_evaluation_report(point_results)
        pm = point_results['overall_metrics']
        print(f"  MAE:  {pm['mae']:.2f}")
        print(f"  MAPE: {pm['mape']:.2f}%")
        print(f"  RMSE: {pm['rmse']:.2f}")
        print(f"  R2:   {pm['r2']:.3f}")
    else:
        print(f"  {point_results['error']}")

    # Quantile evaluation
    print("\n--- Quantile Forecast Evaluation ---")
    quantile_evaluator = QuantileEvaluator()
    quantile_results = quantile_evaluator.evaluate_quantile_forecasts(combined_quantile, data)

    if 'error' not in quantile_results:
        quantile_evaluator.print_quantile_summary(quantile_results)
    else:
        print(f"  {quantile_results['error']}")

    # Save comparison JSON
    comparison = {
        'generation_date': datetime.now().isoformat(),
        'period': {'start': start_date, 'end': end_date},
        'n_cutoff_dates': len(cutoff_dates),
        'point_forecasts': {
            'n_total': len(combined_point),
            'overall_metrics': point_results.get('overall_metrics', {}),
            'metrics_by_horizon': {
                str(k): v for k, v in point_results.get('metrics_by_horizon', {}).items()
            }
        },
        'quantile_forecasts': {
            'n_total': len(combined_quantile),
            'point_prediction_metrics': quantile_results.get('point_prediction_metrics', {}),
            'quantile_metrics': quantile_results.get('quantile_metrics', {}),
            'metrics_by_horizon': {
                str(k): v for k, v in quantile_results.get('metrics_by_horizon', {}).items()
            }
        }
    }

    # Remove non-serializable entries
    for key in ['evaluation_data']:
        if key in point_results:
            comparison['point_forecasts'].pop(key, None)

    comparison_file = os.path.join(output_dir, 'comparison_results.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to {comparison_file}")

    return {
        'point_forecasts': combined_point,
        'quantile_forecasts': combined_quantile,
        'point_results': point_results,
        'quantile_results': quantile_results,
        'comparison': comparison
    }


def main():
    parser = argparse.ArgumentParser(description='Generate quantile hindcasts Nov 2024 - Apr 2025')
    parser.add_argument('--start-date', type=str, default='2024-11-02')
    parser.add_argument('--end-date', type=str, default='2025-04-26')
    parser.add_argument('--locations', nargs='+', default=None,
                        help='Specific locations (e.g., US 06 48)')
    parser.add_argument('--output-dir', type=str, default='outputs/quantile_hindcasts')
    args = parser.parse_args()

    print("=" * 70)
    print("QUANTILE HINDCAST GENERATION: NOV 2024 - APR 2025")
    print("=" * 70)

    generate_quantile_hindcasts(
        start_date=args.start_date,
        end_date=args.end_date,
        locations=args.locations,
        output_dir=args.output_dir
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
