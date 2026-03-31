"""
CLI for plotting flu forecasts.

Usage:
    python scripts/plot_forecast.py --location US --cutoff-date 2025-01-11
    python scripts/plot_forecast.py --location US --cutoff-date 2025-01-11 --quantile
    python scripts/plot_forecast.py --location US 06 48 --cutoff-date 2025-01-11 --grid
    python scripts/plot_forecast.py --location US --cutoff-date 2025-01-11 2025-02-01 --compare
"""

import argparse
import sys
import os

# Ensure project root is on path when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble, QuantileDirectForecastEnsemble
from src.visualization import plot_forecast, plot_forecast_comparison, plot_forecast_grid


def main():
    parser = argparse.ArgumentParser(description='Plot flu forecasts')
    parser.add_argument('--location', nargs='+', default=['US'],
                        help='Location FIPS codes (e.g., US 06 48)')
    parser.add_argument('--cutoff-date', nargs='+', required=True,
                        help='Cutoff date(s) (YYYY-MM-DD)')
    parser.add_argument('--quantile', action='store_true',
                        help='Use quantile forecasting mode')
    parser.add_argument('--grid', action='store_true',
                        help='Plot locations as a grid of subplots')
    parser.add_argument('--compare', action='store_true',
                        help='Compare forecasts from multiple cutoff dates')
    parser.add_argument('--weeks-history', type=int, default=12,
                        help='Weeks of history to show (default: 12)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (shows interactively if not set)')
    parser.add_argument('--no-actual', action='store_true',
                        help='Do not show actual future values')

    args = parser.parse_args()

    model_type = "quantile" if args.quantile else "point"
    show_actual = not args.no_actual
    locations = args.location
    cutoff_dates = args.cutoff_date

    # Load full data (no cutoff) so actual future values are available for comparison
    print("Loading data...")
    loader = FluDataLoader()
    data = loader.fetch_data()

    if args.compare:
        # Comparison mode: multiple cutoffs, single location
        location = locations[0]
        print(f"Comparing {len(cutoff_dates)} cutoffs for {location}...")
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'comparison_{location}.png')

        plot_forecast_comparison(
            location=location,
            cutoff_dates=cutoff_dates,
            model_type=model_type,
            weeks_history=args.weeks_history,
            show_actual=show_actual,
            save_path=save_path,
            data=data
        )

    elif args.grid:
        # Grid mode: multiple locations, single cutoff
        cutoff_date = cutoff_dates[0]
        print(f"Plotting grid for {len(locations)} locations at cutoff {cutoff_date}...")
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'grid_{cutoff_date}.png')

        plot_forecast_grid(
            locations=locations,
            cutoff_date=cutoff_date,
            model_type=model_type,
            weeks_history=args.weeks_history,
            show_actual=show_actual,
            save_path=save_path,
            data=data
        )

    else:
        # Single plot mode: one location, one cutoff
        cutoff_date = cutoff_dates[0]
        location = locations[0]
        print(f"Plotting {model_type} forecast for {location} at cutoff {cutoff_date}...")
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            mode_suffix = '_quantile' if args.quantile else '_point'
            save_path = os.path.join(args.save_dir, f'forecast_{location}_{cutoff_date}{mode_suffix}.png')

        plot_forecast(
            location=location,
            cutoff_date=cutoff_date,
            model_type=model_type,
            weeks_history=args.weeks_history,
            show_actual=show_actual,
            save_path=save_path,
            data=data
        )

    if not args.save_dir:
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
