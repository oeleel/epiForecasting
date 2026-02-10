"""
Forecast generation module for flu forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
from src.model import FluForecastingModel
from src import config


class FluForecastGenerator:
    """Generator for flu hospitalization forecasts"""
    
    def __init__(self, model: Optional[FluForecastingModel] = None):
        """
        Initialize the forecast generator
        
        Args:
            model: Trained model (if None, will need to be loaded)
        """
        self.model = model
        self.forecast_horizon = config.FORECAST_HORIZON
        
    def generate_forecasts(self, data: pd.DataFrame, 
                        cutoff_date: str,
                        locations: Optional[List[str]] = None,
                        with_confidence: bool = False,
                        n_bootstrap: int = 100) -> pd.DataFrame:
        """
        Generate forecasts for specified locations
        
        Args:
            data: Historical data
            cutoff_date: Date to use as cutoff
            locations: List of locations to forecast (if None, forecast all locations)
            with_confidence: Whether to include confidence intervals
            n_bootstrap: Number of bootstrap samples for confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model must be provided or loaded before generating forecasts")
        
        # Filter data up to cutoff date
        cutoff_dt = pd.to_datetime(cutoff_date)
        historical_data = data[data['date'] <= cutoff_dt].copy()
        
        # If locations not specified, use all unique locations in data
        if locations is None:
            locations = historical_data['location'].unique().tolist()
        
        forecasts = []
        
        for location in locations:
            print(f"Generating forecasts for location: {location}")
            
            # Get historical data for this location
            loc_data = historical_data[historical_data['location'] == location].copy()
            
            if len(loc_data) == 0:
                print(f"Warning: No historical data for location {location}")
                continue
            
            # Sort by date
            loc_data = loc_data.sort_values('date')
            
            # Generate forecasts for this location
            loc_forecasts = self._generate_location_forecasts(
                loc_data, location, cutoff_dt, with_confidence, n_bootstrap
            )
            
            forecasts.extend(loc_forecasts)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecasts)
        
        if len(forecast_df) > 0:
            forecast_df = forecast_df.sort_values(['location', 'forecast_date'])
        
        return forecast_df
    
    def _generate_location_forecasts(self, loc_data: pd.DataFrame, 
                                   location: str, 
                                   cutoff_dt: pd.Timestamp,
                                   with_confidence: bool,
                                   n_bootstrap: int) -> List[Dict]:
        """
        Generate forecasts for a single location
        
        Args:
            loc_data: Historical data for the location
            location: Location identifier
            cutoff_dt: Cutoff datetime
            with_confidence: Whether to include confidence intervals
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            List of forecast dictionaries
        """
        forecasts = []
        
        # Prepare features for the last available data point
        X_last = self.model.prepare_features(loc_data.tail(1))[0]
        
        # Generate recursive forecasts
        current_data = loc_data.copy()
        
        for week in range(1, self.forecast_horizon + 1):
            forecast_date = cutoff_dt + timedelta(weeks=week)
            
            # Make prediction
            if with_confidence:
                prediction, confidence_lower, confidence_upper = self._predict_with_confidence(
                    current_data, n_bootstrap
                )
            else:
                prediction = self.model.predict(X_last)[0]
                confidence_lower = None
                confidence_upper = None
            
            # Create forecast record
            forecast = {
                'location': location,
                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                'forecast_week': week,
                'forecast': max(0, prediction)  # Ensure non-negative
            }
            
            if with_confidence:
                forecast['confidence_lower'] = max(0, confidence_lower)
                forecast['confidence_upper'] = max(0, confidence_upper)
            
            forecasts.append(forecast)
            
            # Update data for next iteration (recursive forecasting)
            if week < self.forecast_horizon:
                # Create new row with prediction - preserve dtypes!
                # Using dict to DataFrame avoids the dtype conversion issue
                new_row_data = {col: [current_data.iloc[-1][col]] for col in current_data.columns}
                new_row_data['date'] = [forecast_date]
                new_row_data['value'] = [prediction]
                new_row_df = pd.DataFrame(new_row_data)
                
                # Ensure dtypes match before concat
                for col in current_data.columns:
                    if col in new_row_df.columns:
                        try:
                            new_row_df[col] = new_row_df[col].astype(current_data[col].dtype)
                        except (ValueError, TypeError):
                            pass  # Keep as-is if conversion fails
                
                # Add to current data
                current_data = pd.concat([current_data, new_row_df], ignore_index=True)
                
                # CRITICAL: Recompute features for the new row
                # This updates lag features, rolling statistics, and temporal features
                current_data = self._update_features_for_prediction(current_data, forecast_date)
                
                # Now extract features from the updated last row
                X_last = self.model.prepare_features(current_data.tail(1))[0]
        
        return forecasts
    
    def _predict_with_confidence(self, data: pd.DataFrame, n_bootstrap: int) -> Tuple[float, float, float]:
        """
        Generate prediction with confidence intervals using bootstrap
        
        Args:
            data: Historical data
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (prediction, lower_bound, upper_bound)
        """
        # Get base prediction
        X = self.model.prepare_features(data.tail(1))[0]
        base_prediction = self.model.predict(X)[0]
        
        # Generate bootstrap predictions
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement from recent data
            sample_data = data.sample(n=min(len(data), 10), replace=True)
            
            # Add some noise to simulate uncertainty
            noise_factor = np.random.normal(1, 0.1)
            sample_data = sample_data.copy()
            sample_data['value'] = sample_data['value'] * noise_factor
            
            # Make prediction
            try:
                X_sample = self.model.prepare_features(sample_data.tail(1))[0]
                pred = self.model.predict(X_sample)[0]
                bootstrap_predictions.append(pred)
            except:
                # If prediction fails, use base prediction with noise
                bootstrap_predictions.append(base_prediction * np.random.normal(1, 0.1))
        
        # Calculate confidence intervals
        predictions_array = np.array(bootstrap_predictions)
        lower_bound = np.percentile(predictions_array, 2.5)
        upper_bound = np.percentile(predictions_array, 97.5)
        
        return base_prediction, lower_bound, upper_bound
    
    def _update_features_for_prediction(self, data: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
        """
        Update features for the last row after appending a new prediction.
        This recomputes lag features, rolling statistics, and temporal features.
        
        Args:
            data: DataFrame with the new prediction appended
            forecast_date: The date of the forecast
            
        Returns:
            DataFrame with updated features for the last row
        """
        df = data.copy()
        n = len(df)
        last_idx = n - 1
        
        # Get values as array for efficient computation
        values = df['value'].values
        
        # Update temporal features for the new date
        df.loc[last_idx, 'year'] = forecast_date.year
        df.loc[last_idx, 'month'] = forecast_date.month
        df.loc[last_idx, 'week_of_year'] = forecast_date.isocalendar()[1]
        df.loc[last_idx, 'day_of_year'] = forecast_date.timetuple().tm_yday
        
        # Cyclical encoding
        week_of_year = df.loc[last_idx, 'week_of_year']
        month = df.loc[last_idx, 'month']
        day_of_year = df.loc[last_idx, 'day_of_year']
        
        df.loc[last_idx, 'week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        df.loc[last_idx, 'week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        df.loc[last_idx, 'month_sin'] = np.sin(2 * np.pi * month / 12)
        df.loc[last_idx, 'month_cos'] = np.cos(2 * np.pi * month / 12)
        df.loc[last_idx, 'day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        df.loc[last_idx, 'day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Update lag features
        # value_lag_1: value from 1 step ago
        lag_features = config.LAG_FEATURES  # [1, 2, 3, 4, 8, 12]
        for lag in lag_features:
            col_name = f'value_lag_{lag}'
            if col_name in df.columns:
                if n > lag:
                    df.loc[last_idx, col_name] = values[n - 1 - lag]
                else:
                    # Not enough history, use the earliest available or 0
                    df.loc[last_idx, col_name] = values[0] if n > 0 else 0
        
        # Update rolling features
        rolling_windows = config.ROLLING_WINDOWS  # [4, 8]
        for window in rolling_windows:
            # Get the window of values (last 'window' values, or all if less)
            start_idx = max(0, n - window)
            window_values = values[start_idx:n]
            
            # Rolling mean
            col_mean = f'value_rolling_mean_{window}'
            if col_mean in df.columns:
                df.loc[last_idx, col_mean] = np.mean(window_values)
            
            # Rolling std
            col_std = f'value_rolling_std_{window}'
            if col_std in df.columns:
                df.loc[last_idx, col_std] = np.std(window_values) if len(window_values) > 1 else 0
            
            # Rolling min
            col_min = f'value_rolling_min_{window}'
            if col_min in df.columns:
                df.loc[last_idx, col_min] = np.min(window_values)
            
            # Rolling max
            col_max = f'value_rolling_max_{window}'
            if col_max in df.columns:
                df.loc[last_idx, col_max] = np.max(window_values)
        
        # Update derived features
        # value_trend_4w: current value - rolling_mean_4
        if 'value_trend_4w' in df.columns and 'value_rolling_mean_4' in df.columns:
            df.loc[last_idx, 'value_trend_4w'] = values[last_idx] - df.loc[last_idx, 'value_rolling_mean_4']
        
        # recent_vs_historical: rolling_mean_4 / rolling_mean_8
        if 'recent_vs_historical' in df.columns:
            if 'value_rolling_mean_4' in df.columns and 'value_rolling_mean_8' in df.columns:
                mean_4 = df.loc[last_idx, 'value_rolling_mean_4']
                mean_8 = df.loc[last_idx, 'value_rolling_mean_8']
                df.loc[last_idx, 'recent_vs_historical'] = mean_4 / (mean_8 + 1e-8)
        
        # Update YoY features
        if 'value_lag_52' in df.columns and 'yoy_ratio' in df.columns:
            lag_52 = df.loc[last_idx, 'value_lag_52']
            df.loc[last_idx, 'yoy_ratio'] = values[last_idx] / (lag_52 + 1)
            if 'yoy_pct_change' in df.columns:
                df.loc[last_idx, 'yoy_pct_change'] = (values[last_idx] - lag_52) / (lag_52 + 1)
            if 'yoy_diff' in df.columns:
                df.loc[last_idx, 'yoy_diff'] = values[last_idx] - lag_52
        
        # Update rate-of-change features
        if 'wow_change' in df.columns and n > 1:
            df.loc[last_idx, 'wow_change'] = values[last_idx] - values[last_idx - 1]
        if 'wow_pct_change' in df.columns and n > 1:
            prev_val = values[last_idx - 1]
            pct_change = (values[last_idx] - prev_val) / (prev_val + 1e-8) if prev_val != 0 else 0
            df.loc[last_idx, 'wow_pct_change'] = np.clip(pct_change, -10, 10)
        if 'acceleration' in df.columns and n > 2:
            wow_curr = values[last_idx] - values[last_idx - 1]
            wow_prev = values[last_idx - 1] - values[last_idx - 2]
            df.loc[last_idx, 'acceleration'] = wow_curr - wow_prev
        if 'momentum_4w' in df.columns and n > 4:
            df.loc[last_idx, 'momentum_4w'] = values[last_idx] - values[last_idx - 4]
        if 'momentum_4w_pct' in df.columns and n > 4:
            val_4_ago = values[last_idx - 4]
            pct = (values[last_idx] - val_4_ago) / (val_4_ago + 1) if val_4_ago != 0 else 0
            df.loc[last_idx, 'momentum_4w_pct'] = np.clip(pct, -10, 10)
        
        # Update season phase features
        if 'is_flu_season' in df.columns:
            df.loc[last_idx, 'is_flu_season'] = 1 if month in [10, 11, 12, 1, 2, 3, 4] else 0
        if 'season_phase' in df.columns:
            week = int(week_of_year)
            if week >= 40 and week <= 48:
                phase = 1  # onset
            elif week >= 49 or week <= 4:
                phase = 2  # peak
            elif week >= 5 and week <= 16:
                phase = 3  # decline
            else:
                phase = 0  # off-season
            df.loc[last_idx, 'season_phase'] = phase
            if 'season_phase_sin' in df.columns:
                df.loc[last_idx, 'season_phase_sin'] = np.sin(2 * np.pi * phase / 4)
            if 'season_phase_cos' in df.columns:
                df.loc[last_idx, 'season_phase_cos'] = np.cos(2 * np.pi * phase / 4)
        
        # Update seasonal_deviation
        if 'seasonal_deviation' in df.columns and 'value_lag_52' in df.columns and 'value_lag_1' in df.columns:
            lag_1 = df.loc[last_idx, 'value_lag_1']
            lag_52 = df.loc[last_idx, 'value_lag_52']
            deviation = (lag_1 - lag_52) / (lag_52 + 1)
            df.loc[last_idx, 'seasonal_deviation'] = np.clip(deviation, -10, 10)
        
        # Update rate-based features (if weekly_rate is available)
        if 'weekly_rate' in df.columns:
            rates = df['weekly_rate'].values
            if 'rate_lag_1' in df.columns and n > 1:
                df.loc[last_idx, 'rate_lag_1'] = rates[last_idx - 1]
            if 'rate_lag_4' in df.columns and n > 4:
                df.loc[last_idx, 'rate_lag_4'] = rates[last_idx - 4]
            if 'rate_rolling_mean_4' in df.columns:
                start_idx = max(0, n - 4)
                df.loc[last_idx, 'rate_rolling_mean_4'] = np.mean(rates[start_idx:n])
            if 'rate_trend_4w' in df.columns and 'rate_rolling_mean_4' in df.columns:
                df.loc[last_idx, 'rate_trend_4w'] = rates[last_idx] - df.loc[last_idx, 'rate_rolling_mean_4']
        
        # Note: US lag and rolling features are based on national data and should
        # ideally be updated if we're forecasting US. For state forecasts, we keep
        # the historical US features as context
        
        return df
    
    def save_forecasts(self, forecasts: pd.DataFrame, output_dir: str) -> str:
        """
        Save forecasts to CSV file
        
        Args:
            forecasts: Forecasts DataFrame
            output_dir: Output directory
            
        Returns:
            Path to saved forecasts file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        forecast_file = os.path.join(output_dir, f"forecasts_{timestamp}.csv")
        
        forecasts.to_csv(forecast_file, index=False)
        
        return forecast_file
    
    def generate_forecast_report(self, forecasts: pd.DataFrame) -> Dict:
        """
        Generate summary report for forecasts
        
        Args:
            forecasts: Forecasts DataFrame
            
        Returns:
            Dictionary with forecast summary
        """
        if len(forecasts) == 0:
            return {'status': 'no_forecasts'}
        
        report = {
            'generation_date': datetime.now().isoformat(),
            'n_locations': forecasts['location'].nunique(),
            'forecast_horizon': forecasts['forecast_week'].max(),
            'total_forecasts': len(forecasts),
            'summary_by_location': {},
            'summary_by_week': {}
        }
        
        # Summary by location
        for location in forecasts['location'].unique():
            loc_forecasts = forecasts[forecasts['location'] == location]
            report['summary_by_location'][location] = {
                'n_forecasts': len(loc_forecasts),
                'avg_forecast': loc_forecasts['forecast'].mean(),
                'min_forecast': loc_forecasts['forecast'].min(),
                'max_forecast': loc_forecasts['forecast'].max()
            }
        
        # Summary by week
        for week in forecasts['forecast_week'].unique():
            week_forecasts = forecasts[forecasts['forecast_week'] == week]
            report['summary_by_week'][f'week_{week}'] = {
                'n_forecasts': len(week_forecasts),
                'avg_forecast': week_forecasts['forecast'].mean(),
                'min_forecast': week_forecasts['forecast'].min(),
                'max_forecast': week_forecasts['forecast'].max()
            }
        
        return report
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model
        
        Args:
            model_path: Path to the model file
        """
        if self.model is None:
            self.model = FluForecastingModel()
        
        self.model.load_model(model_path)
    
    def get_forecast_insights(self, forecasts: pd.DataFrame) -> Dict:
        """
        Generate insights from forecasts
        
        Args:
            forecasts: Forecasts DataFrame
            
        Returns:
            Dictionary with insights
        """
        if len(forecasts) == 0:
            return {'insights': 'No forecasts available'}
        
        insights = {
            'trend_analysis': {},
            'peak_forecasts': {},
            'location_rankings': {}
        }
        
        # Trend analysis
        for location in forecasts['location'].unique():
            loc_forecasts = forecasts[forecasts['location'] == location].sort_values('forecast_week')
            
            if len(loc_forecasts) > 1:
                # Calculate trend
                values = loc_forecasts['forecast'].values
                trend = np.polyfit(range(len(values)), values, 1)[0]
                
                insights['trend_analysis'][location] = {
                    'trend_slope': trend,
                    'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
                }
        
        # Peak forecasts
        for location in forecasts['location'].unique():
            loc_forecasts = forecasts[forecasts['location'] == location]
            peak_week = loc_forecasts.loc[loc_forecasts['forecast'].idxmax(), 'forecast_week']
            peak_value = loc_forecasts['forecast'].max()
            
            insights['peak_forecasts'][location] = {
                'peak_week': peak_week,
                'peak_value': peak_value
            }
        
        # Location rankings by average forecast
        avg_forecasts = forecasts.groupby('location')['forecast'].mean().sort_values(ascending=False)
        insights['location_rankings'] = {
            'highest_risk': avg_forecasts.index[0] if len(avg_forecasts) > 0 else None,
            'lowest_risk': avg_forecasts.index[-1] if len(avg_forecasts) > 0 else None,
            'ranking': avg_forecasts.to_dict()
        }
        
        return insights
