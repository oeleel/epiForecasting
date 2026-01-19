"""
Generate forecasts from November 2024 to April 2025
Create plots for key dates and evaluate performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict, Tuple

from data_loader import FluDataLoader
from feature_engineering import FeatureEngineer
from model import FluForecastingModel
from train import FluModelTrainer
from predict import FluForecastGenerator
from evaluate import ModelEvaluator
import config

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')


def generate_forecasts_for_period(
    start_date: str,
    end_date: str,
    model_path: str,
    locations: List[str] = None,
    forecast_horizon: int = 4
) -> pd.DataFrame:
    """
    Generate forecasts for multiple cutoff dates from start to end date
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model_path: Path to trained model
        locations: List of locations to forecast (None = all)
        forecast_horizon: Number of weeks ahead to forecast
        
    Returns:
        DataFrame with all forecasts
    """
    print(f"Generating forecasts from {start_date} to {end_date}")
    
    # Load data - use local file if available, otherwise fetch
    loader = FluDataLoader()
    # Try to load from local file first
    try:
        if os.path.exists('influ_hospit.csv'):
            data = pd.read_csv('influ_hospit.csv')
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            # Filter to end_date
            end_dt = pd.to_datetime(end_date)
            data = data[data['date'] <= end_dt].copy()
            print(f"Loaded {len(data)} records from local file")
        else:
            data = loader.load_and_preprocess(end_date)
    except Exception as e:
        print(f"Error loading data: {e}, trying to fetch...")
        data = loader.load_and_preprocess(end_date)
    
    # Load model
    model = FluForecastingModel()
    model.load_model(model_path)
    
    # Initialize forecast generator
    generator = FluForecastGenerator(model=model)
    generator.forecast_horizon = forecast_horizon
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Generate cutoff dates (weekly)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    cutoff_dates = pd.date_range(start=start_dt, end=end_dt, freq='W-SAT')  # Weekly on Saturdays
    
    all_forecasts = []
    
    for cutoff_date in cutoff_dates:
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        print(f"\nGenerating forecasts for cutoff date: {cutoff_str}")
        
        # Filter data up to cutoff
        cutoff_data = data[data['date'] <= cutoff_date].copy()
        
        if len(cutoff_data) == 0:
            print(f"  Warning: No data available for cutoff {cutoff_str}")
            continue
        
        # Create features
        try:
            features_df = engineer.create_all_features(cutoff_data)
            
            # Generate forecasts
            forecasts = generator.generate_forecasts(
                data=features_df,
                cutoff_date=cutoff_str,
                locations=locations,
                with_confidence=False
            )
            
            # Add cutoff date to forecasts
            forecasts['cutoff_date'] = cutoff_str
            all_forecasts.append(forecasts)
            print(f"  Generated {len(forecasts)} forecasts")
            
        except Exception as e:
            print(f"  Error generating forecasts for {cutoff_str}: {str(e)}")
            continue
    
    # Combine all forecasts
    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        return combined_forecasts
    else:
        return pd.DataFrame()


def create_forecast_plots(
    forecasts: pd.DataFrame,
    actual_data: pd.DataFrame,
    key_dates: Dict[str, str],
    output_dir: str = "outputs/forecast_plots"
):
    """
    Create plots for key dates (season onset, peak, decline)
    
    Args:
        forecasts: Forecasts DataFrame
        actual_data: Actual data DataFrame
        key_dates: Dictionary with keys like 'onset', 'peak', 'decline' and dates
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert date columns
    forecasts['forecast_date'] = pd.to_datetime(forecasts['forecast_date'])
    forecasts['cutoff_date'] = pd.to_datetime(forecasts['cutoff_date'])
    actual_data['date'] = pd.to_datetime(actual_data['date'])
    
    # Get US data for overall trend
    us_actual = actual_data[actual_data['location'] == 'US'].copy()
    us_forecasts = forecasts[forecasts['location'] == 'US'].copy()
    
    # Plot 1: Overall US trend with forecasts
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual data
    us_actual_sorted = us_actual.sort_values('date')
    ax.plot(us_actual_sorted['date'], us_actual_sorted['value'], 
            'o-', label='Actual', linewidth=2, markersize=4, color='#2E86AB')
    
    # Plot forecasts for each cutoff date
    for cutoff in forecasts['cutoff_date'].unique():
        cutoff_forecasts = us_forecasts[us_forecasts['cutoff_date'] == cutoff].copy()
        cutoff_forecasts = cutoff_forecasts.sort_values('forecast_week')
        
        if len(cutoff_forecasts) > 0:
            ax.plot(cutoff_forecasts['forecast_date'], cutoff_forecasts['forecast'],
                   '--', alpha=0.5, linewidth=1.5, 
                   label=f"Forecast from {cutoff.strftime('%Y-%m-%d')}")
    
    # Mark key dates
    for key, date_str in key_dates.items():
        date = pd.to_datetime(date_str)
        ax.axvline(x=date, color='red', linestyle=':', linewidth=2, 
                  label=f'{key.capitalize()}: {date_str}', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Hospitalizations', fontsize=12)
    ax.set_title('US Flu Hospitalizations: Actual vs Forecasts (Nov 2024 - Apr 2025)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'us_overall_forecasts.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2-4: Detailed plots for key dates
    for key, date_str in key_dates.items():
        cutoff_date = pd.to_datetime(date_str)
        
        # Get forecasts from this cutoff date
        key_forecasts = forecasts[forecasts['cutoff_date'] == cutoff_date].copy()
        
        if len(key_forecasts) == 0:
            print(f"  No forecasts found for {key} date {date_str}")
            continue
        
        # Focus on US for main plot
        us_key_forecasts = key_forecasts[key_forecasts['location'] == 'US'].copy()
        
        if len(us_key_forecasts) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get historical data up to cutoff
        hist_data = actual_data[
            (actual_data['date'] <= cutoff_date) & 
            (actual_data['location'] == 'US')
        ].sort_values('date')
        
        # Get actual data after cutoff (for evaluation)
        future_actual = actual_data[
            (actual_data['date'] > cutoff_date) & 
            (actual_data['date'] <= cutoff_date + timedelta(weeks=4)) &
            (actual_data['location'] == 'US')
        ].sort_values('date')
        
        # Plot historical data
        ax.plot(hist_data['date'], hist_data['value'], 
               'o-', label='Historical', linewidth=2, markersize=5, color='#2E86AB')
        
        # Plot forecasts
        us_key_forecasts = us_key_forecasts.sort_values('forecast_week')
        ax.plot(us_key_forecasts['forecast_date'], us_key_forecasts['forecast'],
               's-', label='Forecast', linewidth=2.5, markersize=6, color='#A23B72')
        
        # Plot actual future data if available
        if len(future_actual) > 0:
            ax.plot(future_actual['date'], future_actual['value'],
                   'o-', label='Actual (Future)', linewidth=2, markersize=5, color='#F18F01')
        
        # Mark cutoff date
        ax.axvline(x=cutoff_date, color='red', linestyle='--', linewidth=2, 
                  label=f'Cutoff: {date_str}', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Hospitalizations', fontsize=12)
        ax.set_title(f'Flu Forecasts from {key.capitalize()} Period ({date_str})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'forecast_{key}_{date_str}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Created plot for {key} period")
    
    # Plot 5: Multiple states comparison for peak period
    if 'peak' in key_dates:
        peak_date = pd.to_datetime(key_dates['peak'])
        peak_forecasts = forecasts[forecasts['cutoff_date'] == peak_date].copy()
        
        # Select a few key states
        key_states = ['US', '06', '12', '48', '36']  # US, CA, FL, TX, NY
        available_states = [s for s in key_states if s in peak_forecasts['location'].unique()]
        
        if len(available_states) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for state in available_states:
                state_forecasts = peak_forecasts[peak_forecasts['location'] == state].sort_values('forecast_week')
                state_name = actual_data[actual_data['location'] == state]['location_name'].iloc[0] if len(actual_data[actual_data['location'] == state]) > 0 else state
                
                ax.plot(state_forecasts['forecast_date'], state_forecasts['forecast'],
                       'o-', label=state_name, linewidth=2, markersize=5)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Hospitalizations', fontsize=12)
            ax.set_title(f'Forecasts by State from Peak Period ({key_dates["peak"]})', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'forecast_multiple_states_peak.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nAll plots saved to {output_dir}")


def evaluate_forecasts(
    forecasts: pd.DataFrame,
    actual_data: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict:
    """
    Evaluate forecasts using MSE, MAE, and MAPE
    
    Args:
        forecasts: Forecasts DataFrame
        actual_data: Actual data DataFrame
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    print("\nEvaluating forecasts...")
    
    # Convert date columns
    forecasts['forecast_date'] = pd.to_datetime(forecasts['forecast_date'])
    actual_data['date'] = pd.to_datetime(actual_data['date'])
    
    # Merge forecasts with actual data
    evaluation_data = forecasts.merge(
        actual_data[['location', 'date', 'value']],
        left_on=['location', 'forecast_date'],
        right_on=['location', 'date'],
        how='inner'
    )
    
    if len(evaluation_data) == 0:
        print("  Warning: No matching actual data found for evaluation")
        return {}
    
    # Rename for clarity
    evaluation_data = evaluation_data.rename(columns={'value': 'actual'})
    
    # Calculate overall metrics
    y_true = evaluation_data['actual'].values
    y_pred = evaluation_data['forecast'].values
    
    # Remove any invalid values
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0) & (y_pred >= 0)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("  Warning: No valid predictions for evaluation")
        return {}
    
    # Calculate metrics
    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1))) * 100
    rmse = np.sqrt(mse)
    
    # Calculate by forecast horizon
    metrics_by_horizon = {}
    for horizon in sorted(evaluation_data['forecast_week'].unique()):
        horizon_data = evaluation_data[evaluation_data['forecast_week'] == horizon]
        h_true = horizon_data['actual'].values
        h_pred = horizon_data['forecast'].values
        
        h_mask = np.isfinite(h_true) & np.isfinite(h_pred) & (h_true >= 0) & (h_pred >= 0)
        h_true_clean = h_true[h_mask]
        h_pred_clean = h_pred[h_mask]
        
        if len(h_true_clean) > 0:
            metrics_by_horizon[horizon] = {
                'mse': np.mean((h_true_clean - h_pred_clean) ** 2),
                'mae': np.mean(np.abs(h_true_clean - h_pred_clean)),
                'mape': np.mean(np.abs((h_true_clean - h_pred_clean) / np.maximum(h_true_clean, 1))) * 100,
                'rmse': np.sqrt(np.mean((h_true_clean - h_pred_clean) ** 2)),
                'n_samples': len(h_true_clean)
            }
    
    # Calculate by location (top locations)
    metrics_by_location = {}
    top_locations = evaluation_data['location'].value_counts().head(10).index
    for location in top_locations:
        loc_data = evaluation_data[evaluation_data['location'] == location]
        l_true = loc_data['actual'].values
        l_pred = loc_data['forecast'].values
        
        l_mask = np.isfinite(l_true) & np.isfinite(l_pred) & (l_true >= 0) & (l_pred >= 0)
        l_true_clean = l_true[l_mask]
        l_pred_clean = l_pred[l_mask]
        
        if len(l_true_clean) > 0:
            metrics_by_location[location] = {
                'mse': np.mean((l_true_clean - l_pred_clean) ** 2),
                'mae': np.mean(np.abs(l_true_clean - l_pred_clean)),
                'mape': np.mean(np.abs((l_true_clean - l_pred_clean) / np.maximum(l_true_clean, 1))) * 100,
                'rmse': np.sqrt(np.mean((l_true_clean - l_pred_clean) ** 2)),
                'n_samples': len(l_true_clean)
            }
    
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'n_forecasts_evaluated': len(evaluation_data),
        'overall_metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'n_samples': len(y_true_clean)
        },
        'metrics_by_horizon': {str(k): v for k, v in metrics_by_horizon.items()},
        'metrics_by_location': {str(k): v for k, v in metrics_by_location.items()}
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("FORECAST EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total forecasts evaluated: {len(evaluation_data)}")
    print(f"\nOverall Metrics:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    print(f"\nMetrics by Forecast Horizon:")
    for horizon, metrics in sorted(metrics_by_horizon.items()):
        print(f"  Week {horizon}: MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'forecast_evaluation_nov_apr.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {results_file}")
    
    return results


def main():
    """Main execution function"""
    print("="*60)
    print("FLU FORECAST GENERATION: NOV 2024 - APR 2025")
    print("="*60)
    
    # Configuration
    start_date = "2024-11-02"
    end_date = "2025-04-26"  # End of April 2025
    # Use enhanced model with new features (YoY, rate-of-change, season phase)
    model_path = "models/flu_model_enhanced_2024-11-02"
    
    # Key dates for plotting (season onset, peak, decline)
    key_dates = {
        'onset': '2024-11-02',      # Early season onset
        'peak': '2025-01-11',      # Mid-season peak
        'decline': '2025-03-15'    # Late season decline
    }
    
    # Load actual data for evaluation
    print("\nLoading actual data...")
    loader = FluDataLoader()
    try:
        if os.path.exists('influ_hospit.csv'):
            actual_data = pd.read_csv('influ_hospit.csv')
            actual_data['date'] = pd.to_datetime(actual_data['date'])
            actual_data = actual_data.sort_values('date').reset_index(drop=True)
            end_dt = pd.to_datetime(end_date)
            actual_data = actual_data[actual_data['date'] <= end_dt].copy()
        else:
            actual_data = loader.load_and_preprocess(end_date)
    except Exception as e:
        print(f"Error loading actual data: {e}, trying to fetch...")
        actual_data = loader.load_and_preprocess(end_date)
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = generate_forecasts_for_period(
        start_date=start_date,
        end_date=end_date,
        model_path=model_path,
        locations=None,  # All locations
        forecast_horizon=4
    )
    
    if len(forecasts) == 0:
        print("Error: No forecasts generated")
        return
    
    # Save forecasts
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    forecast_file = os.path.join(output_dir, "forecasts_nov_2024_apr_2025.csv")
    forecasts.to_csv(forecast_file, index=False)
    print(f"\nForecasts saved to {forecast_file}")
    print(f"Total forecasts: {len(forecasts)}")
    
    # Create plots
    print("\nCreating forecast plots...")
    create_forecast_plots(
        forecasts=forecasts,
        actual_data=actual_data,
        key_dates=key_dates,
        output_dir=os.path.join(output_dir, "forecast_plots")
    )
    
    # Evaluate forecasts
    print("\nEvaluating forecasts...")
    evaluation_results = evaluate_forecasts(
        forecasts=forecasts,
        actual_data=actual_data,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("FORECAST GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Forecasts: {forecast_file}")
    print(f"  - Evaluation: {os.path.join(output_dir, 'forecast_evaluation_nov_apr.json')}")
    print(f"  - Plots: {os.path.join(output_dir, 'forecast_plots')}")


if __name__ == "__main__":
    main()
