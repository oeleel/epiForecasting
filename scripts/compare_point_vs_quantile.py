"""
Compare point vs quantile XGBoost forecasts.

Loads outputs from generate_quantile_hindcasts.py and generates a formatted
report establishing the XGBoost baseline numbers to beat with NN models.

Usage:
    python scripts/compare_point_vs_quantile.py
    python scripts/compare_point_vs_quantile.py --input-dir outputs/quantile_hindcasts
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
import argparse
import json

from src.data_loader import FluDataLoader
from src.evaluate import ModelEvaluator, QuantileEvaluator


def load_hindcast_outputs(input_dir: str):
    """Load point and quantile forecast CSVs from hindcast output directory."""
    point_csv = os.path.join(input_dir, 'point_forecasts_nov_apr.csv')
    quantile_csv = os.path.join(input_dir, 'quantile_forecasts_nov_apr.csv')

    if not os.path.exists(point_csv) or not os.path.exists(quantile_csv):
        raise FileNotFoundError(
            f"Hindcast outputs not found in {input_dir}. "
            "Run `python scripts/generate_quantile_hindcasts.py` first."
        )

    point_df = pd.read_csv(point_csv)
    quantile_df = pd.read_csv(quantile_csv)

    print(f"Loaded {len(point_df)} point forecasts, {len(quantile_df)} quantile forecasts")
    return point_df, quantile_df


def generate_comparison_report(point_df, quantile_df, actual_data, output_dir):
    """Generate full comparison report with plots and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Point evaluation ---
    point_eval = ModelEvaluator()
    point_results = point_eval.evaluate_forecasts(point_df, actual_data)

    # --- Quantile evaluation ---
    quantile_eval = QuantileEvaluator()
    quantile_results = quantile_eval.evaluate_quantile_forecasts(quantile_df, actual_data)

    # --- Print report ---
    print("\n" + "=" * 70)
    print("XGBOOST BASELINE REPORT: POINT vs QUANTILE FORECASTS")
    print("=" * 70)

    if 'error' in point_results or 'error' in quantile_results:
        print("ERROR: Could not evaluate forecasts")
        if 'error' in point_results:
            print(f"  Point: {point_results['error']}")
        if 'error' in quantile_results:
            print(f"  Quantile: {quantile_results['error']}")
        return

    pm = point_results['overall_metrics']
    qm = quantile_results['point_prediction_metrics']
    qq = quantile_results['quantile_metrics']

    print(f"\n{'METRIC':<30} {'POINT (XGB)':<20} {'QUANTILE (XGB)':<20}")
    print("-" * 70)
    print(f"{'MAE':<30} {pm['mae']:<20.2f} {qm['mae']:<20.2f}")
    print(f"{'RMSE':<30} {pm['rmse']:<20.2f} {qm['rmse']:<20.2f}")
    print(f"{'MAPE (%)':<30} {pm['mape']:<20.2f} {qm['mape']:<20.2f}")
    print(f"{'R²':<30} {pm['r2']:<20.3f} {qm['r2']:<20.3f}")
    print(f"{'Bias':<30} {pm['bias']:<20.2f} {qm['bias']:<20.2f}")

    print(f"\n--- Probabilistic Metrics (Quantile model only) ---")
    print(f"{'90% Coverage':<30} {qq['coverage_90pct']*100:.1f}% (target: 90%)")
    print(f"{'50% Coverage':<30} {qq['coverage_50pct']*100:.1f}% (target: 50%)")
    print(f"{'Winkler Score (90%)':<30} {qq['mean_winkler_score_90pct']:.1f}")
    if qq.get('wis') is not None:
        print(f"{'WIS':<30} {qq['wis']:.2f}")
    if qq.get('crps') is not None:
        print(f"{'CRPS':<30} {qq['crps']:.2f}")
    if qq.get('mean_pinball_loss') is not None:
        print(f"{'Mean Pinball Loss':<30} {qq['mean_pinball_loss']:.2f}")

    # Horizon comparison
    print(f"\n--- Metrics by Horizon ---")
    print(f"{'Horizon':<10} {'Point MAE':<15} {'Q-Median MAE':<15} {'WIS':<12} {'CRPS':<12} {'Coverage90':<12}")
    print("-" * 76)

    point_by_h = point_results.get('metrics_by_horizon', {})
    quant_by_h = quantile_results.get('metrics_by_horizon', {})

    for h in sorted(set(list(point_by_h.keys()) + list(quant_by_h.keys()))):
        p_mae = point_by_h.get(h, {}).get('mae', float('nan'))
        q_mae = quant_by_h.get(h, {}).get('mae', float('nan'))
        q_wis = quant_by_h.get(h, {}).get('wis', float('nan'))
        q_crps = quant_by_h.get(h, {}).get('crps', float('nan'))
        q_cov = quant_by_h.get(h, {}).get('coverage_90pct', float('nan'))
        cov_str = f"{q_cov*100:.1f}%" if not np.isnan(q_cov) else "N/A"
        wis_str = f"{q_wis:.2f}" if not np.isnan(q_wis) else "N/A"
        crps_str = f"{q_crps:.2f}" if not np.isnan(q_crps) else "N/A"
        print(f"  H{h:<7} {p_mae:<15.2f} {q_mae:<15.2f} {wis_str:<12} {crps_str:<12} {cov_str:<12}")

    # Pinball losses per quantile
    if qq.get('pinball_losses'):
        print(f"\n--- Pinball Losses by Quantile ---")
        for q_name, loss in qq['pinball_losses'].items():
            print(f"  {q_name}: {loss:.2f}")

    print("\n" + "=" * 70)
    print("BASELINE NUMBERS TO BEAT:")
    print(f"  Point MAE:  {pm['mae']:.2f}")
    print(f"  Point MAPE: {pm['mape']:.2f}%")
    if qq.get('wis') is not None:
        print(f"  WIS:        {qq['wis']:.2f}")
    if qq.get('crps') is not None:
        print(f"  CRPS:       {qq['crps']:.2f}")
    print(f"  Winkler:    {qq['mean_winkler_score_90pct']:.1f}")
    print("=" * 70)

    # --- Create comparison plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE by horizon
    horizons = sorted(point_by_h.keys())
    p_maes = [point_by_h[h].get('mae', 0) for h in horizons]
    q_maes = [quant_by_h.get(h, {}).get('mae', 0) for h in horizons]
    x = np.arange(len(horizons))
    axes[0].bar(x - 0.15, p_maes, 0.3, label='Point', color='#2E86AB')
    axes[0].bar(x + 0.15, q_maes, 0.3, label='Quantile (median)', color='#E74C3C')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'H{h}' for h in horizons])
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE by Horizon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CRPS by horizon
    crps_vals = [quant_by_h.get(h, {}).get('crps', 0) for h in horizons]
    axes[1].bar(x, crps_vals, color='#E74C3C')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'H{h}' for h in horizons])
    axes[1].set_ylabel('CRPS')
    axes[1].set_title('CRPS by Horizon')
    axes[1].grid(True, alpha=0.3)

    # Coverage by horizon
    cov_vals = [quant_by_h.get(h, {}).get('coverage_90pct', 0) * 100 for h in horizons]
    axes[2].bar(x, cov_vals, color='#27AE60')
    axes[2].axhline(y=90, color='red', linestyle='--', label='Target (90%)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'H{h}' for h in horizons])
    axes[2].set_ylabel('Coverage (%)')
    axes[2].set_title('90% PI Coverage by Horizon')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('XGBoost Baseline: Point vs Quantile Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()

    plot_path = os.path.join(output_dir, 'point_vs_quantile_comparison.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # Save report JSON
    report = {
        'point_overall': pm,
        'quantile_point_metrics': qm,
        'quantile_probabilistic_metrics': qq,
        'point_by_horizon': {str(k): v for k, v in point_by_h.items()},
        'quantile_by_horizon': {str(k): v for k, v in quant_by_h.items()},
        'baseline_targets': {
            'point_mae': pm['mae'],
            'point_mape': pm['mape'],
            'wis': qq.get('wis'),
            'crps': qq.get('crps'),
            'winkler_90': qq['mean_winkler_score_90pct']
        }
    }
    report_path = os.path.join(output_dir, 'baseline_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare point vs quantile XGBoost forecasts')
    parser.add_argument('--input-dir', type=str, default='outputs/quantile_hindcasts',
                        help='Directory with hindcast outputs')
    parser.add_argument('--output-dir', type=str, default='outputs/quantile_hindcasts',
                        help='Directory for comparison outputs')
    args = parser.parse_args()

    # Load forecasts
    point_df, quantile_df = load_hindcast_outputs(args.input_dir)

    # Load actual data — need full date range to cover forecast dates
    print("Loading actual data...")
    loader = FluDataLoader()
    # Determine the latest forecast date to ensure we load enough data
    max_forecast_date = max(
        pd.to_datetime(point_df['forecast_date']).max(),
        pd.to_datetime(quantile_df['forecast_date']).max()
    ).strftime('%Y-%m-%d')
    try:
        if os.path.exists('influ_hospit.csv'):
            actual_data = pd.read_csv('influ_hospit.csv')
            actual_data['date'] = pd.to_datetime(actual_data['date'])
        else:
            actual_data = loader.load_and_preprocess(max_forecast_date)
    except Exception:
        actual_data = loader.load_and_preprocess(max_forecast_date)

    generate_comparison_report(point_df, quantile_df, actual_data, args.output_dir)


if __name__ == "__main__":
    main()
