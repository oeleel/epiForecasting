"""
Forecast visualization module for flu forecasting.

Provides matplotlib-based plotting functions for comparing point and quantile
forecasts against ground truth data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Optional, List, Dict, Tuple
import os

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble, QuantileDirectForecastEnsemble
from src import config

# Use a clean style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except Exception:
    plt.style.use('default')

# Location name lookup (FIPS -> name)
_LOCATION_NAMES = {}


def _get_location_name(location: str, data: Optional[pd.DataFrame] = None) -> str:
    """Get human-readable location name from FIPS code."""
    global _LOCATION_NAMES
    if not _LOCATION_NAMES and data is not None and 'location_name' in data.columns:
        for _, row in data[['location', 'location_name']].drop_duplicates().iterrows():
            _LOCATION_NAMES[str(row['location'])] = row['location_name']
    return _LOCATION_NAMES.get(str(location), str(location))


def _load_data_and_model(cutoff_date: str, model_type: str = "point",
                         data: Optional[pd.DataFrame] = None,
                         features_df: Optional[pd.DataFrame] = None,
                         ensemble=None):
    """
    Load data and train model if not provided.

    Returns:
        (data, features_df, ensemble, forecasts)
    """
    if data is None:
        loader = FluDataLoader()
        # Load full data (no cutoff) so actual future values are available for plotting
        data = loader.fetch_data()

    if features_df is None:
        cutoff_data = data[data['date'] <= pd.to_datetime(cutoff_date)].copy()
        engineer = FeatureEngineer()
        features_df = engineer.create_all_features(cutoff_data)

    if ensemble is None:
        if model_type == "quantile":
            ensemble = QuantileDirectForecastEnsemble()
        else:
            ensemble = DirectForecastEnsemble()
        ensemble.train(features_df)

    forecasts = ensemble.generate_forecasts(features_df, cutoff_date)
    return data, features_df, ensemble, forecasts


def plot_forecast(location: str, cutoff_date: str,
                  model_type: str = "point",
                  weeks_history: int = 12,
                  show_actual: bool = True,
                  ax: Optional[plt.Axes] = None,
                  save_path: Optional[str] = None,
                  data: Optional[pd.DataFrame] = None,
                  features_df: Optional[pd.DataFrame] = None,
                  ensemble=None,
                  forecasts: Optional[pd.DataFrame] = None) -> plt.Figure:
    """
    Plot forecast for a single location and cutoff date.

    Args:
        location: FIPS code or 'US'
        cutoff_date: Cutoff date string (YYYY-MM-DD)
        model_type: 'point' or 'quantile'
        weeks_history: Number of weeks of history to show before cutoff
        show_actual: Whether to overlay actual future values
        ax: Existing matplotlib axes (creates new figure if None)
        save_path: Path to save figure (None = don't save)
        data: Raw data DataFrame (loads if None)
        features_df: Engineered features (computes if None)
        ensemble: Trained ensemble (trains if None)
        forecasts: Pre-computed forecasts DataFrame (generates if None)

    Returns:
        matplotlib Figure
    """
    # Load/compute as needed
    if forecasts is None:
        data, features_df, ensemble, forecasts = _load_data_and_model(
            cutoff_date, model_type, data, features_df, ensemble
        )
    elif data is None:
        loader = FluDataLoader()
        # Load full data (no cutoff) so actual future values are available
        data = loader.fetch_data()

    cutoff_dt = pd.to_datetime(cutoff_date)
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])

    # Filter to this location
    loc_data = data[data['location'] == location].sort_values('date')
    loc_forecasts = forecasts[forecasts['location'] == location].copy()

    if len(loc_forecasts) == 0:
        print(f"No forecasts found for location {location}")
        return plt.gcf()

    loc_forecasts['forecast_date'] = pd.to_datetime(loc_forecasts['forecast_date'])

    # History window
    history_start = cutoff_dt - timedelta(weeks=weeks_history)
    history = loc_data[(loc_data['date'] >= history_start) & (loc_data['date'] <= cutoff_dt)]

    # Future actuals (for evaluation)
    max_forecast_date = loc_forecasts['forecast_date'].max()
    future_actual = loc_data[(loc_data['date'] > cutoff_dt) & (loc_data['date'] <= max_forecast_date)]

    # Create figure if needed
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Plot history
    ax.plot(history['date'], history['value'], 'o-', color='#2E86AB',
            linewidth=2, markersize=4, label='History', zorder=3)

    # Cutoff line
    ax.axvline(x=cutoff_dt, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Cutoff: {cutoff_date}')

    # Plot forecasts
    loc_forecasts_sorted = loc_forecasts.sort_values('forecast_date')

    if model_type == "quantile" and 'predicted_q50' in loc_forecasts.columns:
        # Quantile mode: shaded prediction intervals
        dates = loc_forecasts_sorted['forecast_date']

        # 90% PI (light shading)
        ax.fill_between(dates,
                        loc_forecasts_sorted['predicted_q05'],
                        loc_forecasts_sorted['predicted_q95'],
                        alpha=0.2, color='#E74C3C', label='90% PI')

        # 50% PI (darker shading)
        ax.fill_between(dates,
                        loc_forecasts_sorted['predicted_q25'],
                        loc_forecasts_sorted['predicted_q75'],
                        alpha=0.35, color='#E74C3C', label='50% PI')

        # Median line
        ax.plot(dates, loc_forecasts_sorted['predicted_q50'],
                's-', color='#E74C3C', linewidth=2, markersize=5,
                label='Forecast (median)', zorder=4)
    else:
        # Point forecast
        forecast_col = 'forecast' if 'forecast' in loc_forecasts.columns else 'predicted'
        ax.plot(loc_forecasts_sorted['forecast_date'],
                loc_forecasts_sorted[forecast_col],
                's-', color='#E74C3C', linewidth=2, markersize=5,
                label='Forecast', zorder=4)

    # Actual future values
    if show_actual and len(future_actual) > 0:
        ax.plot(future_actual['date'], future_actual['value'],
                'D-', color='#27AE60', linewidth=2, markersize=5,
                label='Actual', zorder=5)

    # Title and labels
    loc_name = _get_location_name(location, data)
    mode_str = "Quantile" if model_type == "quantile" else "Point"
    ax.set_title(f'{loc_name} - {mode_str} Forecast (cutoff: {cutoff_date})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Hospitalizations', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_forecast_comparison(location: str, cutoff_dates: List[str],
                             model_type: str = "point",
                             weeks_history: int = 12,
                             show_actual: bool = True,
                             save_path: Optional[str] = None,
                             data: Optional[pd.DataFrame] = None) -> plt.Figure:
    """
    Overlay forecasts from multiple cutoff dates for a single location.

    Args:
        location: FIPS code or 'US'
        cutoff_dates: List of cutoff date strings
        model_type: 'point' or 'quantile'
        weeks_history: Weeks of history to show before earliest cutoff
        show_actual: Whether to overlay actual future values
        save_path: Path to save figure
        data: Raw data DataFrame (loads if None)

    Returns:
        matplotlib Figure
    """
    if data is None:
        loader = FluDataLoader()
        latest_cutoff = max(cutoff_dates)
        data = loader.load_and_preprocess(latest_cutoff)

    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot history up to earliest cutoff
    earliest_cutoff = pd.to_datetime(min(cutoff_dates))
    latest_cutoff = pd.to_datetime(max(cutoff_dates))
    history_start = earliest_cutoff - timedelta(weeks=weeks_history)

    loc_data = data[data['location'] == location].sort_values('date')
    history = loc_data[(loc_data['date'] >= history_start) & (loc_data['date'] <= latest_cutoff)]

    ax.plot(history['date'], history['value'], 'o-', color='#2E86AB',
            linewidth=2, markersize=3, label='History', zorder=2)

    # Colors for different cutoffs
    colors = plt.cm.Set1(np.linspace(0, 1, len(cutoff_dates)))

    for i, cutoff_date in enumerate(cutoff_dates):
        cutoff_dt = pd.to_datetime(cutoff_date)
        cutoff_data = data[data['date'] <= cutoff_dt].copy()

        engineer = FeatureEngineer()
        features_df = engineer.create_all_features(cutoff_data)

        if model_type == "quantile":
            ensemble = QuantileDirectForecastEnsemble()
        else:
            ensemble = DirectForecastEnsemble()
        ensemble.train(features_df)
        forecasts = ensemble.generate_forecasts(features_df, cutoff_date)

        loc_fc = forecasts[forecasts['location'] == location].copy()
        loc_fc['forecast_date'] = pd.to_datetime(loc_fc['forecast_date'])
        loc_fc = loc_fc.sort_values('forecast_date')

        if len(loc_fc) == 0:
            continue

        # Cutoff marker
        ax.axvline(x=cutoff_dt, color=colors[i], linestyle=':', alpha=0.5, linewidth=1)

        if model_type == "quantile" and 'predicted_q50' in loc_fc.columns:
            ax.fill_between(loc_fc['forecast_date'],
                            loc_fc['predicted_q05'], loc_fc['predicted_q95'],
                            alpha=0.1, color=colors[i])
            ax.plot(loc_fc['forecast_date'], loc_fc['predicted_q50'],
                    's-', color=colors[i], linewidth=1.5, markersize=4,
                    label=f'Forecast ({cutoff_date})')
        else:
            forecast_col = 'forecast' if 'forecast' in loc_fc.columns else 'predicted'
            ax.plot(loc_fc['forecast_date'], loc_fc[forecast_col],
                    's-', color=colors[i], linewidth=1.5, markersize=4,
                    label=f'Forecast ({cutoff_date})')

    # Actual future values
    if show_actual:
        max_date = latest_cutoff + timedelta(weeks=4)
        future = loc_data[(loc_data['date'] > latest_cutoff) & (loc_data['date'] <= max_date)]
        if len(future) > 0:
            ax.plot(future['date'], future['value'], 'D-', color='#27AE60',
                    linewidth=2, markersize=4, label='Actual', zorder=5)

    loc_name = _get_location_name(location, data)
    ax.set_title(f'{loc_name} - Forecast Comparison ({len(cutoff_dates)} cutoffs)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Hospitalizations', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_forecast_grid(locations: List[str], cutoff_date: str,
                       model_type: str = "point",
                       ncols: int = 3,
                       weeks_history: int = 12,
                       show_actual: bool = True,
                       save_path: Optional[str] = None,
                       data: Optional[pd.DataFrame] = None,
                       features_df: Optional[pd.DataFrame] = None,
                       ensemble=None,
                       forecasts: Optional[pd.DataFrame] = None) -> plt.Figure:
    """
    Create a grid of forecast subplots for multiple locations.

    Args:
        locations: List of FIPS codes
        cutoff_date: Cutoff date string
        model_type: 'point' or 'quantile'
        ncols: Number of columns in grid
        weeks_history: Weeks of history per subplot
        show_actual: Whether to overlay actual future values
        save_path: Path to save figure
        data: Raw data DataFrame (loads if None)
        features_df: Engineered features (computes if None)
        ensemble: Trained ensemble (trains if None)
        forecasts: Pre-computed forecasts (generates if None)

    Returns:
        matplotlib Figure
    """
    if forecasts is None:
        data, features_df, ensemble, forecasts = _load_data_and_model(
            cutoff_date, model_type, data, features_df, ensemble
        )
    elif data is None:
        loader = FluDataLoader()
        # Load full data (no cutoff) so actual future values are available
        data = loader.fetch_data()

    nrows = (len(locations) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, location in enumerate(locations):
        plot_forecast(
            location=location,
            cutoff_date=cutoff_date,
            model_type=model_type,
            weeks_history=weeks_history,
            show_actual=show_actual,
            ax=axes[i],
            data=data,
            features_df=features_df,
            ensemble=ensemble,
            forecasts=forecasts
        )

    # Hide unused axes
    for j in range(len(locations), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Forecast Grid - {model_type.capitalize()} (cutoff: {cutoff_date})',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved grid plot to {save_path}")

    return fig
