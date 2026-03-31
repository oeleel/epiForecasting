"""
Side-by-side comparison of XGBoost vs Neural Network quantile forecasts.

Runs both models on the same hindcast period (Nov 2024 - Apr 2025),
evaluates with QuantileEvaluator (including CRPS and pinball loss),
and generates comparison plots and metrics.

Usage:
    python scripts/compare_xgb_vs_nn.py
    python scripts/compare_xgb_vs_nn.py --start-date 2025-01-01 --end-date 2025-02-28
    python scripts/compare_xgb_vs_nn.py --locations US 06 48
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
from src.direct_forecast import QuantileDirectForecastEnsemble
from src.nn_model import NNQuantileDirectForecastEnsemble
from src.evaluate import ModelEvaluator, QuantileEvaluator
from src.visualization import plot_forecast
from src import config


def run_comparison(
    start_date: str = "2024-11-02",
    end_date: str = "2025-04-26",
    locations: Optional[List[str]] = None,
    output_dir: str = "outputs/xgb_vs_nn",
    nn_kwargs: Optional[dict] = None
) -> dict:
    """
    Run both XGBoost and NN on same hindcast period and compare.

    Args:
        start_date: Start of hindcast period
        end_date: End of hindcast period
        locations: Specific locations (None = all)
        output_dir: Output directory
        nn_kwargs: Additional kwargs for NNQuantileDirectForecastEnsemble

    Returns:
        Comparison results dict
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    nn_kwargs = nn_kwargs or {}

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

    # Weekly cutoffs
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    cutoff_dates = pd.date_range(start=start_dt, end=end_dt, freq='W-SAT')

    all_xgb_forecasts = []
    all_nn_forecasts = []

    # Key cutoffs for plots
    key_cutoffs = ['2024-11-02', '2025-01-11', '2025-03-15']

    for cutoff_date in cutoff_dates:
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        print(f"\n{'='*50}")
        print(f"Cutoff: {cutoff_str}")
        print(f"{'='*50}")

        cutoff_data = data[data['date'] <= cutoff_date].copy()
        if len(cutoff_data) == 0:
            print("  No data, skipping")
            continue

        try:
            features_df = engineer.create_all_features(cutoff_data)

            # --- XGBoost Quantile ---
            print("  Training XGBoost quantile ensemble...")
            xgb_ensemble = QuantileDirectForecastEnsemble(forecast_horizon=4)
            xgb_ensemble.train(features_df)
            xgb_fc = xgb_ensemble.generate_forecasts(features_df, cutoff_str, locations=locations)
            xgb_fc['cutoff_date'] = cutoff_str
            xgb_fc['model'] = 'xgboost'
            all_xgb_forecasts.append(xgb_fc)

            # --- Neural Network Quantile ---
            print("  Training NN quantile ensemble...")
            nn_ensemble = NNQuantileDirectForecastEnsemble(
                forecast_horizon=4, **nn_kwargs
            )
            nn_ensemble.train(features_df)
            nn_fc = nn_ensemble.generate_forecasts(features_df, cutoff_str, locations=locations)
            nn_fc['cutoff_date'] = cutoff_str
            nn_fc['model'] = 'nn'
            all_nn_forecasts.append(nn_fc)

            print(f"  XGBoost: {len(xgb_fc)} forecasts, NN: {len(nn_fc)} forecasts")

            # Comparison plots for key cutoffs
            if cutoff_str in key_cutoffs:
                for loc in ['US']:
                    try:
                        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

                        plot_forecast(
                            location=loc, cutoff_date=cutoff_str,
                            model_type='quantile', data=data,
                            features_df=features_df, ensemble=xgb_ensemble,
                            forecasts=xgb_fc, ax=axes[0]
                        )
                        axes[0].set_title(f'XGBoost - {loc} ({cutoff_str})')

                        plot_forecast(
                            location=loc, cutoff_date=cutoff_str,
                            model_type='quantile', data=data,
                            features_df=features_df, ensemble=nn_ensemble,
                            forecasts=nn_fc, ax=axes[1]
                        )
                        axes[1].set_title(f'Neural Network - {loc} ({cutoff_str})')

                        fig.tight_layout()
                        fig.savefig(os.path.join(
                            output_dir, 'plots',
                            f'comparison_{loc}_{cutoff_str}.png'
                        ), dpi=200, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as plot_err:
                        print(f"  Plot error: {plot_err}")

        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not all_xgb_forecasts or not all_nn_forecasts:
        print("ERROR: No forecasts generated")
        return {}

    combined_xgb = pd.concat(all_xgb_forecasts, ignore_index=True)
    combined_nn = pd.concat(all_nn_forecasts, ignore_index=True)

    # Save CSVs
    combined_xgb.to_csv(os.path.join(output_dir, 'xgb_quantile_forecasts.csv'), index=False)
    combined_nn.to_csv(os.path.join(output_dir, 'nn_quantile_forecasts.csv'), index=False)

    # --- Evaluate both ---
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    xgb_evaluator = QuantileEvaluator()
    nn_evaluator = QuantileEvaluator()

    xgb_results = xgb_evaluator.evaluate_quantile_forecasts(combined_xgb, data)
    nn_results = nn_evaluator.evaluate_quantile_forecasts(combined_nn, data)

    # --- Print comparison ---
    print("\n" + "=" * 70)
    print("XGBOOST vs NEURAL NETWORK COMPARISON")
    print("=" * 70)

    if 'error' in xgb_results or 'error' in nn_results:
        print("ERROR evaluating models")
        return {}

    xgb_pm = xgb_results['point_prediction_metrics']
    nn_pm = nn_results['point_prediction_metrics']
    xgb_qm = xgb_results['quantile_metrics']
    nn_qm = nn_results['quantile_metrics']

    print(f"\n{'METRIC':<30} {'XGBOOST':<20} {'NEURAL NET':<20} {'WINNER':<10}")
    print("-" * 80)

    metrics_comparison = [
        ('MAE', xgb_pm['mae'], nn_pm['mae'], True),
        ('RMSE', xgb_pm['rmse'], nn_pm['rmse'], True),
        ('MAPE (%)', xgb_pm['mape'], nn_pm['mape'], True),
        ('R²', xgb_pm['r2'], nn_pm['r2'], False),
        ('90% Coverage', xgb_qm['coverage_90pct'], nn_qm['coverage_90pct'], None),
        ('50% Coverage', xgb_qm['coverage_50pct'], nn_qm['coverage_50pct'], None),
        ('Winkler (90%)', xgb_qm['mean_winkler_score_90pct'], nn_qm['mean_winkler_score_90pct'], True),
    ]

    if xgb_qm.get('crps') is not None and nn_qm.get('crps') is not None:
        metrics_comparison.append(('CRPS', xgb_qm['crps'], nn_qm['crps'], True))

    if xgb_qm.get('mean_pinball_loss') is not None and nn_qm.get('mean_pinball_loss') is not None:
        metrics_comparison.append(('Mean Pinball', xgb_qm['mean_pinball_loss'], nn_qm['mean_pinball_loss'], True))

    nn_wins = 0
    xgb_wins = 0

    for name, xgb_val, nn_val, lower_better in metrics_comparison:
        if lower_better is None:
            # Coverage: closer to target is better
            target = 0.9 if '90%' in name else 0.5
            winner = 'NN' if abs(nn_val - target) < abs(xgb_val - target) else 'XGBoost'
        elif lower_better:
            winner = 'NN' if nn_val < xgb_val else 'XGBoost'
        else:
            winner = 'NN' if nn_val > xgb_val else 'XGBoost'

        if winner == 'NN':
            nn_wins += 1
        else:
            xgb_wins += 1

        if isinstance(xgb_val, float) and xgb_val < 1:
            print(f"{name:<30} {xgb_val:<20.4f} {nn_val:<20.4f} {winner:<10}")
        else:
            print(f"{name:<30} {xgb_val:<20.2f} {nn_val:<20.2f} {winner:<10}")

    # By horizon
    print(f"\n--- CRPS by Horizon ---")
    print(f"{'Horizon':<10} {'XGBoost':<15} {'Neural Net':<15} {'Winner':<10}")
    print("-" * 50)

    xgb_by_h = xgb_results.get('metrics_by_horizon', {})
    nn_by_h = nn_results.get('metrics_by_horizon', {})

    for h in sorted(set(list(xgb_by_h.keys()) + list(nn_by_h.keys()))):
        xgb_crps = xgb_by_h.get(h, {}).get('crps', float('nan'))
        nn_crps = nn_by_h.get(h, {}).get('crps', float('nan'))
        if np.isnan(xgb_crps) or np.isnan(nn_crps):
            winner = 'N/A'
        else:
            winner = 'NN' if nn_crps < xgb_crps else 'XGBoost'
        print(f"  H{h:<7} {xgb_crps:<15.2f} {nn_crps:<15.2f} {winner:<10}")

    # Verdict
    print(f"\n{'='*70}")
    overall_winner = "NEURAL NETWORK" if nn_wins > xgb_wins else "XGBOOST"
    print(f"OVERALL: {overall_winner} wins ({nn_wins} vs {xgb_wins} metrics)")

    xgb_crps = xgb_qm.get('crps')
    nn_crps = nn_qm.get('crps')
    if xgb_crps and nn_crps:
        crps_winner = "NN" if nn_crps < xgb_crps else "XGBoost"
        improvement = (xgb_crps - nn_crps) / xgb_crps * 100
        print(f"CRPS: {crps_winner} wins by {abs(improvement):.1f}%")
        if nn_crps < xgb_crps:
            print("NN OUTPERFORMS XGBOOST on CRPS -> Phase A success!")
        else:
            print("XGBoost still better on CRPS -> Consider Phase B (sequence model)")
    print("=" * 70)

    # --- Create summary plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE by horizon
    horizons = sorted(xgb_by_h.keys())
    x = np.arange(len(horizons))
    xgb_maes = [xgb_by_h[h].get('mae', 0) for h in horizons]
    nn_maes = [nn_by_h.get(h, {}).get('mae', 0) for h in horizons]
    axes[0, 0].bar(x - 0.15, xgb_maes, 0.3, label='XGBoost', color='#2E86AB')
    axes[0, 0].bar(x + 0.15, nn_maes, 0.3, label='Neural Net', color='#E74C3C')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'H{h}' for h in horizons])
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('MAE by Horizon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # CRPS by horizon
    xgb_crps_h = [xgb_by_h[h].get('crps', 0) for h in horizons]
    nn_crps_h = [nn_by_h.get(h, {}).get('crps', 0) for h in horizons]
    axes[0, 1].bar(x - 0.15, xgb_crps_h, 0.3, label='XGBoost', color='#2E86AB')
    axes[0, 1].bar(x + 0.15, nn_crps_h, 0.3, label='Neural Net', color='#E74C3C')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'H{h}' for h in horizons])
    axes[0, 1].set_ylabel('CRPS')
    axes[0, 1].set_title('CRPS by Horizon')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Coverage comparison
    cov_labels = ['90% PI', '50% PI']
    xgb_covs = [xgb_qm['coverage_90pct'] * 100, xgb_qm['coverage_50pct'] * 100]
    nn_covs = [nn_qm['coverage_90pct'] * 100, nn_qm['coverage_50pct'] * 100]
    targets = [90, 50]
    x2 = np.arange(len(cov_labels))
    axes[1, 0].bar(x2 - 0.15, xgb_covs, 0.3, label='XGBoost', color='#2E86AB')
    axes[1, 0].bar(x2 + 0.15, nn_covs, 0.3, label='Neural Net', color='#E74C3C')
    for i, t in enumerate(targets):
        axes[1, 0].hlines(y=t, xmin=i - 0.3, xmax=i + 0.3, colors='gray',
                          linestyles='--', linewidth=1.5)
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(cov_labels)
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('Coverage Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Winkler comparison by horizon
    xgb_winkler = [xgb_by_h[h].get('winkler_score_90pct', 0) for h in horizons]
    nn_winkler = [nn_by_h.get(h, {}).get('winkler_score_90pct', 0) for h in horizons]
    axes[1, 1].bar(x - 0.15, xgb_winkler, 0.3, label='XGBoost', color='#2E86AB')
    axes[1, 1].bar(x + 0.15, nn_winkler, 0.3, label='Neural Net', color='#E74C3C')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'H{h}' for h in horizons])
    axes[1, 1].set_ylabel('Winkler Score')
    axes[1, 1].set_title('Winkler Score by Horizon (lower = better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'XGBoost vs Neural Network Comparison\n'
                 f'Winner: {overall_winner} ({nn_wins} vs {xgb_wins})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'xgb_vs_nn_summary.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Save results JSON
    comparison_results = {
        'date': datetime.now().isoformat(),
        'period': {'start': start_date, 'end': end_date},
        'n_cutoff_dates': len(cutoff_dates),
        'xgboost': {
            'n_forecasts': len(combined_xgb),
            'point_metrics': xgb_pm,
            'quantile_metrics': xgb_qm,
            'by_horizon': {str(k): v for k, v in xgb_by_h.items()}
        },
        'neural_network': {
            'n_forecasts': len(combined_nn),
            'point_metrics': nn_pm,
            'quantile_metrics': nn_qm,
            'by_horizon': {str(k): v for k, v in nn_by_h.items()}
        },
        'comparison': {
            'overall_winner': overall_winner,
            'nn_metric_wins': nn_wins,
            'xgb_metric_wins': xgb_wins,
            'crps_winner': 'NN' if (nn_crps and xgb_crps and nn_crps < xgb_crps) else 'XGBoost'
        }
    }

    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description='Compare XGBoost vs Neural Network forecasts')
    parser.add_argument('--start-date', type=str, default='2024-11-02')
    parser.add_argument('--end-date', type=str, default='2025-04-26')
    parser.add_argument('--locations', nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='outputs/xgb_vs_nn')
    parser.add_argument('--nn-epochs', type=int, default=300,
                        help='Max epochs for NN training')
    parser.add_argument('--nn-patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--nn-lr', type=float, default=1e-3,
                        help='NN learning rate')
    parser.add_argument('--nn-batch-size', type=int, default=256,
                        help='NN batch size')
    args = parser.parse_args()

    print("=" * 70)
    print("XGBOOST vs NEURAL NETWORK COMPARISON")
    print("=" * 70)

    nn_kwargs = {
        'max_epochs': args.nn_epochs,
        'patience': args.nn_patience,
        'learning_rate': args.nn_lr,
        'batch_size': args.nn_batch_size,
    }

    run_comparison(
        start_date=args.start_date,
        end_date=args.end_date,
        locations=args.locations,
        output_dir=args.output_dir,
        nn_kwargs=nn_kwargs
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
