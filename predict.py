"""
Forecast generation module for flu forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
from model import FluForecastingModel
import config


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
                        locations: List[str],
                        with_confidence: bool = False,
                        n_bootstrap: int = 100) -> pd.DataFrame:
        """
        Generate forecasts for specified locations
        
        Args:
            data: Historical data
            cutoff_date: Date to use as cutoff
            locations: List of locations to forecast
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
            # Add the prediction as the next data point
            next_row = current_data.iloc[-1].copy()
            next_row['date'] = forecast_date
            next_row['value'] = prediction
            
            # Add to current data for next iteration
            current_data = pd.concat([current_data, next_row.to_frame().T], ignore_index=True)
            
            # Update features for next prediction
            if week < self.forecast_horizon:
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
