"""
Week-by-week interactive forecast visualization tool

This script creates interactive HTML visualizations to inspect forecasts week-by-week,
ensuring they properly capture increases rather than remaining flat during growth periods.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import webbrowser

from data_loader import FluDataLoader
import config


class WeeklyForecastVisualizer:
    """Creates interactive visualizations for week-by-week forecast inspection"""
    
    def __init__(self, forecast_file: str = "outputs/forecasts_improved_direct.csv"):
        """
        Initialize the visualizer
        
        Args:
            forecast_file: Path to forecast CSV file
        """
        self.forecast_file = forecast_file
        self.forecasts = None
        self.actual_data = None
        self.merged_data = None
        self.output_dir = Path("outputs/forecast_viz")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load forecast and actual data"""
        print("Loading forecast data...")
        self.forecasts = pd.read_csv(self.forecast_file)
        self.forecasts['cutoff_date'] = pd.to_datetime(self.forecasts['cutoff_date'])
        self.forecasts['forecast_date'] = pd.to_datetime(self.forecasts['forecast_date'])
        
        print("Loading actual data...")
        loader = FluDataLoader()
        # Load all available data
        self.actual_data = loader.fetch_data()
        
        print(f"Loaded {len(self.forecasts)} forecasts and {len(self.actual_data)} actual observations")
        
    def merge_forecasts_with_actuals(self):
        """Merge forecasts with actual values for comparison"""
        print("Merging forecasts with actual values...")
        
        # Merge on location and forecast_date = actual date
        # Include location_name from actual_data
        merged = self.forecasts.merge(
            self.actual_data[['location', 'date', 'value', 'location_name']],
            left_on=['location', 'forecast_date'],
            right_on=['location', 'date'],
            how='left'
        )
        
        merged = merged.rename(columns={'value': 'actual'})
        merged = merged.drop(columns=['date'], errors='ignore')
        
        # For locations without matches in actual data, try to get names from actual_data
        if 'location_name' not in merged.columns or merged['location_name'].isnull().any():
            location_name_map = self.actual_data[['location', 'location_name']].drop_duplicates().set_index('location')['location_name'].to_dict()
            if 'location_name' not in merged.columns:
                merged['location_name'] = merged['location'].map(location_name_map)
            else:
                merged['location_name'] = merged['location_name'].fillna(merged['location'].map(location_name_map))
        
        # Calculate errors
        merged['error'] = merged['forecast'] - merged['actual']
        merged['abs_error'] = np.abs(merged['error'])
        merged['pct_error'] = (merged['error'] / np.maximum(merged['actual'], 1)) * 100
        
        self.merged_data = merged
        print(f"Merged data: {len(self.merged_data)} records with actuals")
        
    def detect_flat_forecasts(self) -> pd.DataFrame:
        """
        Detect periods where forecasts are flat during significant increases
        
        Returns:
            DataFrame with flat forecast flags
        """
        print("Detecting flat forecasts during increase periods...")
        
        df = self.merged_data.copy()
        
        # Sort by location, cutoff_date, and forecast_week
        df = df.sort_values(['location', 'cutoff_date', 'forecast_week'])
        
        # For each location and cutoff, calculate week-over-week changes
        df['forecast_wow_change'] = df.groupby(['location', 'cutoff_date'])['forecast'].diff()
        df['forecast_wow_pct_change'] = df.groupby(['location', 'cutoff_date'])['forecast'].pct_change() * 100
        
        df['actual_wow_change'] = df.groupby(['location', 'cutoff_date'])['actual'].diff()
        df['actual_wow_pct_change'] = df.groupby(['location', 'cutoff_date'])['actual'].pct_change() * 100
        
        # Flag flat forecasts during increases
        # Flat: forecast change < 5% while actual change > 20%
        df['is_flat_during_increase'] = (
            (np.abs(df['forecast_wow_pct_change']) < 5) & 
            (df['actual_wow_pct_change'] > 20)
        )
        
        # Flag underestimated growth
        # Forecast increases but much less than actual
        df['underestimates_growth'] = (
            (df['forecast_wow_pct_change'] > 0) &
            (df['actual_wow_pct_change'] > 0) &
            (df['forecast_wow_pct_change'] < df['actual_wow_pct_change'] * 0.5)
        )
        
        # Flag correct direction
        df['correct_direction'] = (
            (np.sign(df['forecast_wow_change']) == np.sign(df['actual_wow_change'])) &
            (df['actual_wow_change'].notna())
        )
        
        n_flat = df['is_flat_during_increase'].sum()
        n_underest = df['underestimates_growth'].sum()
        n_correct = df['correct_direction'].sum()
        
        print(f"  Flat forecasts during increases: {n_flat}")
        print(f"  Underestimated growth: {n_underest}")
        print(f"  Correct direction: {n_correct} / {len(df[df['correct_direction'].notna()])}")
        
        self.merged_data = df
        return df
    
    def create_horizon_visualization(self, horizon: int, location_filter: Optional[str] = None) -> go.Figure:
        """
        Create interactive visualization for a specific forecast horizon
        
        Args:
            horizon: Forecast horizon (1-4 weeks)
            location_filter: Optional location to filter (None = create dropdown)
            
        Returns:
            Plotly figure object
        """
        df = self.merged_data[self.merged_data['forecast_week'] == horizon].copy()
        
        if len(df) == 0:
            print(f"No data for horizon {horizon}")
            return None
        
        # Get unique locations and create mapping to names
        locations = sorted(df['location'].unique())
        
        # Create location code to name mapping
        location_names = {}
        for loc in locations:
            loc_data = df[df['location'] == loc]
            if len(loc_data) > 0 and 'location_name' in loc_data.columns:
                name = loc_data['location_name'].iloc[0]
                location_names[loc] = f"{name} ({loc})"
            else:
                location_names[loc] = loc
        
        # If location_filter specified, use it; otherwise use first location as default
        if location_filter:
            locations = [location_filter]
            default_location = location_filter
        else:
            default_location = 'US' if 'US' in locations else locations[0]
        
        # Create figure
        fig = go.Figure()
        
        # For each location, add traces (initially visible only for default)
        for location in locations:
            loc_data = df[df['location'] == location].sort_values('forecast_date')
            
            visible = (location == default_location)
            
            # Actual values line
            fig.add_trace(go.Scatter(
                x=loc_data['forecast_date'],
                y=loc_data['actual'],
                mode='lines+markers',
                name=f'{location} - Actual',
                line=dict(color='black', width=3),
                marker=dict(size=8),
                visible=visible,
                legendgroup=location,
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.0f}<extra></extra>'
            ))
            
            # Forecast values line with color based on performance
            colors = []
            for _, row in loc_data.iterrows():
                if pd.isna(row['actual']):
                    colors.append('gray')
                elif row.get('is_flat_during_increase', False):
                    colors.append('red')
                elif row.get('underestimates_growth', False):
                    colors.append('orange')
                elif row.get('correct_direction', False):
                    colors.append('green')
                else:
                    colors.append('blue')
            
            fig.add_trace(go.Scatter(
                x=loc_data['forecast_date'],
                y=loc_data['forecast'],
                mode='lines+markers',
                name=f'{location} - Forecast',
                line=dict(color='blue', width=2, dash='dash'),
                marker=dict(size=6, color=colors),
                visible=visible,
                legendgroup=location,
                hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:.0f}<br>Cutoff: %{customdata[0]}<br>Error: %{customdata[1]:.0f}<br>% Error: %{customdata[2]:.1f}%<extra></extra>',
                customdata=loc_data[['cutoff_date', 'error', 'pct_error']].values
            ))
            
            # Add markers for problematic forecasts
            flat_data = loc_data[loc_data['is_flat_during_increase'] == True]
            if len(flat_data) > 0:
                fig.add_trace(go.Scatter(
                    x=flat_data['forecast_date'],
                    y=flat_data['forecast'],
                    mode='markers',
                    name=f'{location} - Flat during increase',
                    marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
                    visible=visible,
                    legendgroup=location,
                    hovertemplate='<b>⚠️ Flat during increase</b><br>Date: %{x}<br>Forecast: %{y:.0f}<br>Actual: %{customdata[0]:.0f}<extra></extra>',
                    customdata=flat_data[['actual']].values
                ))
            else:
                # Add empty trace to maintain trace indexing
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    name=f'{location} - Flat during increase',
                    visible=visible,
                    legendgroup=location
                ))
        
        # Create dropdown menu for location selection
        if not location_filter:
            buttons = []
            for i, location in enumerate(locations):
                # Each location has 3 traces (actual, forecast, flat markers)
                visible_array = [False] * (len(locations) * 3)
                visible_array[i * 3] = True      # Actual
                visible_array[i * 3 + 1] = True  # Forecast
                visible_array[i * 3 + 2] = True  # Flat markers
                
                display_name = location_names.get(location, location)
                buttons.append(dict(
                    label=display_name,
                    method='update',
                    args=[{'visible': visible_array},
                          {'title': f'Horizon {horizon} Week Ahead Forecasts - {display_name}'}]
                ))
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=buttons,
                        direction='down',
                        pad={'r': 10, 't': 10},
                        showactive=True,
                        x=0.01,
                        xanchor='left',
                        y=1.15,
                        yanchor='top'
                    )
                ]
            )
        
        # Update layout
        default_display_name = location_names.get(default_location, default_location)
        fig.update_layout(
            title=f'Horizon {horizon} Week Ahead Forecasts - {default_display_name}',
            xaxis_title='Forecast Target Date',
            yaxis_title='Hospitalizations',
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='right',
                x=1.15
            ),
            annotations=[
                dict(
                    text='Color coding: <span style="color:green">●</span> Correct direction | '
                         '<span style="color:orange">●</span> Underestimated growth | '
                         '<span style="color:red">●</span> Flat during increase',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=-0.15,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=10)
                )
            ],
            margin=dict(b=100)
        )
        
        return fig
    
    def create_combined_dashboard(self, location: str = 'US') -> go.Figure:
        """
        Create combined dashboard with all 4 horizons for a specific location
        
        Args:
            location: Location code to visualize
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{i} Week Ahead' for i in range(1, 5)],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for horizon, (row, col) in zip(range(1, 5), positions):
            df = self.merged_data[
                (self.merged_data['forecast_week'] == horizon) &
                (self.merged_data['location'] == location)
            ].sort_values('forecast_date')
            
            if len(df) == 0:
                continue
            
            # Actual line
            fig.add_trace(
                go.Scatter(
                    x=df['forecast_date'],
                    y=df['actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='black', width=2),
                    marker=dict(size=6),
                    legendgroup='actual',
                    showlegend=(horizon == 1),
                    hovertemplate='Actual: %{y:.0f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Forecast line
            colors = []
            for _, r in df.iterrows():
                if pd.isna(r['actual']):
                    colors.append('gray')
                elif r.get('is_flat_during_increase', False):
                    colors.append('red')
                elif r.get('underestimates_growth', False):
                    colors.append('orange')
                elif r.get('correct_direction', False):
                    colors.append('green')
                else:
                    colors.append('blue')
            
            fig.add_trace(
                go.Scatter(
                    x=df['forecast_date'],
                    y=df['forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='blue', width=2, dash='dash'),
                    marker=dict(size=5, color=colors),
                    legendgroup='forecast',
                    showlegend=(horizon == 1),
                    hovertemplate='Forecast: %{y:.0f}<br>Error: %{customdata:.0f}<extra></extra>',
                    customdata=df['error'].values
                ),
                row=row, col=col
            )
            
            # Flat forecast markers
            flat_data = df[df['is_flat_during_increase'] == True]
            if len(flat_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=flat_data['forecast_date'],
                        y=flat_data['forecast'],
                        mode='markers',
                        name='Flat during increase',
                        marker=dict(size=10, color='red', symbol='x', line=dict(width=2)),
                        legendgroup='flat',
                        showlegend=(horizon == 1),
                        hovertemplate='⚠️ Flat<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Hospitalizations')
        
        fig.update_layout(
            title=f'All Forecast Horizons - {location}',
            height=900,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics about forecast quality
        
        Returns:
            DataFrame with summary stats
        """
        df = self.merged_data.copy()
        
        summary = []
        
        for horizon in sorted(df['forecast_week'].unique()):
            horizon_data = df[df['forecast_week'] == horizon]
            
            stats = {
                'horizon': horizon,
                'total_forecasts': len(horizon_data),
                'flat_during_increase': horizon_data['is_flat_during_increase'].sum(),
                'flat_pct': (horizon_data['is_flat_during_increase'].sum() / len(horizon_data) * 100),
                'underestimated_growth': horizon_data['underestimates_growth'].sum(),
                'correct_direction': horizon_data['correct_direction'].sum(),
                'correct_direction_pct': (horizon_data['correct_direction'].sum() / 
                                        len(horizon_data[horizon_data['correct_direction'].notna()]) * 100),
                'mean_abs_error': horizon_data['abs_error'].mean(),
                'mean_pct_error': horizon_data['pct_error'].abs().mean()
            }
            summary.append(stats)
        
        summary_df = pd.DataFrame(summary)
        return summary_df
    
    def visualize_all(self, location_filter: Optional[str] = None, 
                     horizon_filter: Optional[int] = None,
                     open_browser: bool = False):
        """
        Generate all visualizations
        
        Args:
            location_filter: Optional specific location to visualize
            horizon_filter: Optional specific horizon to visualize
            open_browser: Whether to open HTML files in browser
        """
        # Load and process data
        self.load_data()
        self.merge_forecasts_with_actuals()
        self.detect_flat_forecasts()
        
        # Generate summary statistics
        print("\nGenerating summary statistics...")
        summary_df = self.generate_summary_statistics()
        print("\nForecast Quality Summary:")
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = self.output_dir / "forecast_quality_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
        created_files = []
        
        # Generate individual horizon visualizations
        horizons = [horizon_filter] if horizon_filter else [1, 2, 3, 4]
        
        for horizon in horizons:
            print(f"\nCreating visualization for horizon {horizon}...")
            fig = self.create_horizon_visualization(horizon, location_filter)
            
            if fig:
                output_file = self.output_dir / f"horizon_{horizon}_week_interactive.html"
                fig.write_html(str(output_file))
                print(f"  Saved: {output_file}")
                created_files.append(output_file)
        
        # Generate combined dashboard for US (or specified location)
        if not horizon_filter:  # Only create dashboard if showing all horizons
            dashboard_location = location_filter or 'US'
            print(f"\nCreating combined dashboard for {dashboard_location}...")
            fig_combined = self.create_combined_dashboard(dashboard_location)
            
            if fig_combined:
                output_file = self.output_dir / f"all_horizons_dashboard_{dashboard_location}.html"
                fig_combined.write_html(str(output_file))
                print(f"  Saved: {output_file}")
                created_files.append(output_file)
        
        print("\n" + "="*70)
        print("Visualization Complete!")
        print("="*70)
        print("\nCreated files:")
        for f in created_files:
            print(f"  {f}")
        
        # Open in browser if requested
        if open_browser and created_files:
            print(f"\nOpening {created_files[0]} in browser...")
            webbrowser.open(f"file://{created_files[0].absolute()}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate week-by-week interactive forecast visualizations'
    )
    parser.add_argument(
        '--forecast-file',
        type=str,
        default='outputs/forecasts_improved_direct.csv',
        help='Path to forecast CSV file'
    )
    parser.add_argument(
        '--location',
        type=str,
        default=None,
        help='Specific location to visualize (e.g., US, CA)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help='Specific horizon to visualize (1-4 weeks)'
    )
    parser.add_argument(
        '--open-browser',
        action='store_true',
        help='Open visualization in browser after creation'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Week-by-Week Forecast Visualization Tool")
    print("="*70)
    
    visualizer = WeeklyForecastVisualizer(forecast_file=args.forecast_file)
    visualizer.visualize_all(
        location_filter=args.location,
        horizon_filter=args.horizon,
        open_browser=args.open_browser
    )


if __name__ == "__main__":
    main()
