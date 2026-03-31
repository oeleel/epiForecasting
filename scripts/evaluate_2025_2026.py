"""
Evaluate forecasting model on the 2025-2026 flu season.

Loads WIS-optimized parameters (if available) and compares against default
parameters using walk-forward hindcasts on 2025-2026 season data.

Usage:
    python scripts/evaluate_2025_2026.py
    python scripts/evaluate_2025_2026.py --params-file outputs/best_hyperparameters.json
    python scripts/evaluate_2025_2026.py --start-date 2025-10-04 --end-date 2026-03-14
    python scripts/evaluate_2025_2026.py --locations US 06 48
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime
from typing import List, Optional, Dict

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble, QuantileDirectForecastEnsemble
from src.evaluate import ModelEvaluator, QuantileEvaluator
from src import config


def load_optimized_params(filepath: str) -> Dict:
    """Load optimized hyperparameters from JSON file."""
    with open(filepath, 'r') as f:
        params = json.load(f)
    print(f"Loaded optimized params from {filepath}")
    return params


def run_hindcasts(data: pd.DataFrame, cutoff_dates: pd.DatetimeIndex,
                  locations: Optional[List[str]], xgb_params: Optional[Dict] = None,
                  label: str = "default") -> dict:
    """
    Run walk-forward hindcasts over the given cutoff dates.

    Returns dict with combined point/quantile forecast DataFrames.
    """
    engineer = FeatureEngineer()
    all_point = []
    all_quantile = []

    for cutoff_date in cutoff_dates:
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        print(f"  [{label}] Cutoff: {cutoff_str}")

        cutoff_data = data[data['date'] <= cutoff_date].copy()
        if len(cutoff_data) < 100:
            print(f"    Insufficient data ({len(cutoff_data)} rows), skipping")
            continue

        try:
            features_df = engineer.create_all_features(cutoff_data)

            # Point forecasts
            p_ensemble = DirectForecastEnsemble(
                forecast_horizon=4, xgb_params_override=xgb_params
            )
            p_ensemble.train(features_df)
            p_forecasts = p_ensemble.generate_forecasts(
                features_df, cutoff_str, locations=locations
            )
            p_forecasts['cutoff_date'] = cutoff_str
            all_point.append(p_forecasts)

            # Quantile forecasts
            q_ensemble = QuantileDirectForecastEnsemble(
                forecast_horizon=4, xgb_params_override=xgb_params
            )
            q_ensemble.train(features_df)
            q_forecasts = q_ensemble.generate_forecasts(
                features_df, cutoff_str, locations=locations
            )
            q_forecasts['cutoff_date'] = cutoff_str
            all_quantile.append(q_forecasts)

            print(f"    {len(p_forecasts)} point, {len(q_forecasts)} quantile forecasts")
        except Exception as e:
            print(f"    Error: {e}")
            continue

    result = {}
    if all_point:
        result['point'] = pd.concat(all_point, ignore_index=True)
    if all_quantile:
        result['quantile'] = pd.concat(all_quantile, ignore_index=True)
    return result


def evaluate_and_report(forecasts: dict, actual_data: pd.DataFrame, label: str) -> dict:
    """Evaluate point and quantile forecasts, return metrics dict."""
    metrics = {'label': label}

    if 'point' in forecasts:
        evaluator = ModelEvaluator()
        point_results = evaluator.evaluate_forecasts(forecasts['point'], actual_data)
        if 'error' not in point_results:
            metrics['point'] = point_results['overall_metrics']
            metrics['point_by_horizon'] = point_results.get('metrics_by_horizon', {})

    if 'quantile' in forecasts:
        q_evaluator = QuantileEvaluator()
        q_results = q_evaluator.evaluate_quantile_forecasts(forecasts['quantile'], actual_data)
        if 'error' not in q_results:
            metrics['quantile_point'] = q_results['point_prediction_metrics']
            metrics['quantile_metrics'] = q_results['quantile_metrics']
            metrics['quantile_by_horizon'] = q_results.get('metrics_by_horizon', {})

    return metrics


def print_comparison(default_metrics: dict, tuned_metrics: dict):
    """Print side-by-side comparison of default vs tuned model."""
    print("\n" + "=" * 80)
    print("2025-2026 SEASON EVALUATION: DEFAULT vs WIS-TUNED")
    print("=" * 80)

    # Point prediction comparison
    dp = default_metrics.get('point', {})
    tp = tuned_metrics.get('point', {})

    print(f"\n{'METRIC':<30} {'DEFAULT':<20} {'WIS-TUNED':<20} {'CHANGE':<15}")
    print("-" * 85)

    for metric_name in ['mae', 'rmse', 'mape', 'r2', 'bias']:
        d_val = dp.get(metric_name, float('nan'))
        t_val = tp.get(metric_name, float('nan'))
        if not np.isnan(d_val) and not np.isnan(t_val) and d_val != 0:
            pct = (t_val - d_val) / abs(d_val) * 100
            change_str = f"{pct:+.1f}%"
        else:
            change_str = "N/A"
        fmt = '.3f' if metric_name == 'r2' else '.2f'
        print(f"  {metric_name.upper():<28} {d_val:<20{fmt}} {t_val:<20{fmt}} {change_str:<15}")

    # Quantile/probabilistic metrics
    dq = default_metrics.get('quantile_metrics', {})
    tq = tuned_metrics.get('quantile_metrics', {})

    print(f"\n--- Probabilistic Metrics ---")
    prob_metrics = [
        ('WIS', 'wis'),
        ('CRPS', 'crps'),
        ('Winkler 90%', 'mean_winkler_score_90pct'),
        ('Coverage 90%', 'coverage_90pct'),
        ('Coverage 50%', 'coverage_50pct'),
    ]

    for display_name, key in prob_metrics:
        d_val = dq.get(key, float('nan'))
        t_val = tq.get(key, float('nan'))
        if d_val is None:
            d_val = float('nan')
        if t_val is None:
            t_val = float('nan')

        if key.startswith('coverage'):
            d_str = f"{d_val*100:.1f}%" if not np.isnan(d_val) else "N/A"
            t_str = f"{t_val*100:.1f}%" if not np.isnan(t_val) else "N/A"
            if not np.isnan(d_val) and not np.isnan(t_val):
                change_str = f"{(t_val - d_val)*100:+.1f}pp"
            else:
                change_str = "N/A"
            print(f"  {display_name:<28} {d_str:<20} {t_str:<20} {change_str:<15}")
        else:
            if not np.isnan(d_val) and not np.isnan(t_val) and d_val != 0:
                pct = (t_val - d_val) / abs(d_val) * 100
                change_str = f"{pct:+.1f}%"
            else:
                change_str = "N/A"
            d_str = f"{d_val:.2f}" if not np.isnan(d_val) else "N/A"
            t_str = f"{t_val:.2f}" if not np.isnan(t_val) else "N/A"
            print(f"  {display_name:<28} {d_str:<20} {t_str:<20} {change_str:<15}")

    # Per-horizon comparison
    dh = default_metrics.get('quantile_by_horizon', {})
    th = tuned_metrics.get('quantile_by_horizon', {})

    if dh and th:
        print(f"\n--- WIS by Horizon ---")
        print(f"{'Horizon':<10} {'Default WIS':<15} {'Tuned WIS':<15} {'Change':<15}")
        print("-" * 55)
        for h in sorted(set(list(dh.keys()) + list(th.keys()))):
            d_wis = dh.get(h, {}).get('wis', float('nan'))
            t_wis = th.get(h, {}).get('wis', float('nan'))
            if d_wis is None:
                d_wis = float('nan')
            if t_wis is None:
                t_wis = float('nan')
            if not np.isnan(d_wis) and not np.isnan(t_wis) and d_wis != 0:
                change = f"{(t_wis - d_wis)/abs(d_wis)*100:+.1f}%"
            else:
                change = "N/A"
            d_str = f"{d_wis:.2f}" if not np.isnan(d_wis) else "N/A"
            t_str = f"{t_wis:.2f}" if not np.isnan(t_wis) else "N/A"
            print(f"  H{h:<7} {d_str:<15} {t_str:<15} {change:<15}")

    # Verdict
    d_wis_overall = dq.get('wis')
    t_wis_overall = tq.get('wis')
    if d_wis_overall is not None and t_wis_overall is not None:
        if t_wis_overall < d_wis_overall:
            improvement = (d_wis_overall - t_wis_overall) / d_wis_overall * 100
            print(f"\nVERDICT: WIS-tuned model IMPROVES WIS by {improvement:.1f}%")
        else:
            degradation = (t_wis_overall - d_wis_overall) / d_wis_overall * 100
            print(f"\nVERDICT: WIS-tuned model DEGRADES WIS by {degradation:.1f}%")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on 2025-2026 flu season')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to WIS-optimized params JSON (default: use config defaults)')
    parser.add_argument('--start-date', type=str, default='2025-10-04',
                        help='Start of evaluation period (default: 2025-10-04)')
    parser.add_argument('--end-date', type=str, default='2026-03-14',
                        help='End of evaluation period (default: 2026-03-14)')
    parser.add_argument('--locations', nargs='+', default=None,
                        help='Specific locations (e.g., US 06 48)')
    parser.add_argument('--output-dir', type=str, default='outputs/eval_2025_2026')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("2025-2026 SEASON EVALUATION")
    print("=" * 80)

    # Load data
    print("\nLoading latest CDC data...")
    loader = FluDataLoader()
    try:
        data = loader.load_and_preprocess(args.end_date)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying to load from local file...")
        if os.path.exists('influ_hospit.csv'):
            data = pd.read_csv('influ_hospit.csv')
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
        else:
            raise

    print(f"Data range: {data['date'].min()} to {data['date'].max()}")
    print(f"Total records: {len(data)}")

    # Check data availability for 2025-2026 season
    season_data = data[data['date'] >= pd.to_datetime(args.start_date)]
    if len(season_data) == 0:
        print(f"\nWARNING: No data available after {args.start_date}.")
        print("The 2025-2026 season data may not be available yet.")
        print("Adjust --start-date to use available data.")
        return

    # Weekly Saturday cutoffs
    start_dt = pd.to_datetime(args.start_date)
    end_dt = pd.to_datetime(args.end_date)
    cutoff_dates = pd.date_range(start=start_dt, end=end_dt, freq='W-SAT')

    print(f"\nEvaluation period: {args.start_date} to {args.end_date}")
    print(f"Cutoff dates: {len(cutoff_dates)}")
    print(f"Locations: {args.locations or 'All'}")

    # --- Run with default params ---
    print("\n" + "-" * 40)
    print("Running hindcasts with DEFAULT params...")
    print("-" * 40)
    default_forecasts = run_hindcasts(data, cutoff_dates, args.locations, label="default")

    # --- Run with tuned params (if available) ---
    tuned_params = None
    if args.params_file and os.path.exists(args.params_file):
        tuned_params = load_optimized_params(args.params_file)
    else:
        # Try default locations
        for candidate in ['outputs/best_hyperparameters.json', 'best_params.json']:
            if os.path.exists(candidate):
                tuned_params = load_optimized_params(candidate)
                break

    if tuned_params:
        print("\n" + "-" * 40)
        print("Running hindcasts with WIS-TUNED params...")
        print("-" * 40)
        tuned_forecasts = run_hindcasts(data, cutoff_dates, args.locations,
                                        xgb_params=tuned_params, label="tuned")
    else:
        print("\nNo tuned params found. Run `python scripts/optimize.py --metric wis` first.")
        print("Showing default params results only.\n")
        tuned_forecasts = {}

    # --- Evaluate ---
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    default_metrics = evaluate_and_report(default_forecasts, data, "default")

    if tuned_forecasts:
        tuned_metrics = evaluate_and_report(tuned_forecasts, data, "tuned")
        print_comparison(default_metrics, tuned_metrics)
    else:
        # Print default-only results
        if 'quantile_metrics' in default_metrics:
            q_evaluator = QuantileEvaluator()
            q_evaluator.evaluation_results = {
                'evaluation_date': datetime.now().isoformat(),
                'n_forecasts_evaluated': len(default_forecasts.get('quantile', [])),
                'point_prediction_metrics': default_metrics.get('quantile_point', {}),
                'quantile_metrics': default_metrics.get('quantile_metrics', {}),
                'metrics_by_horizon': default_metrics.get('quantile_by_horizon', {})
            }
            q_evaluator.print_quantile_summary()
        tuned_metrics = {}

    # --- Save results ---
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'period': {'start': args.start_date, 'end': args.end_date},
        'n_cutoff_dates': len(cutoff_dates),
        'default_metrics': {
            k: v for k, v in default_metrics.items()
            if k not in ('label',) and isinstance(v, (dict, list, str, int, float))
        },
    }
    if tuned_metrics:
        report['tuned_metrics'] = {
            k: v for k, v in tuned_metrics.items()
            if k not in ('label',) and isinstance(v, (dict, list, str, int, float))
        }
        if tuned_params:
            report['tuned_params'] = tuned_params

    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Save forecast CSVs
    for label, forecasts in [('default', default_forecasts), ('tuned', tuned_forecasts)]:
        if not forecasts:
            continue
        for ftype in ['point', 'quantile']:
            if ftype in forecasts:
                csv_path = os.path.join(args.output_dir, f'{label}_{ftype}_forecasts.csv')
                forecasts[ftype].to_csv(csv_path, index=False)
                print(f"Saved {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
