"""
Model evaluation module for flu forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluator for flu forecasting model performance"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        # Remove any NaN or infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid predictions to evaluate'}
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1e-8))) * 100
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / 
                       (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
        
        # R-squared
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean bias
        bias = np.mean(y_pred_clean - y_true_clean)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2,
            'bias': bias,
            'n_samples': len(y_true_clean)
        }
    
    def evaluate_forecasts(self, forecasts: pd.DataFrame, 
                          actual_data: pd.DataFrame,
                          evaluation_period: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Evaluate forecast performance against actual data
        
        Args:
            forecasts: Forecasts DataFrame
            actual_data: Actual data DataFrame
            evaluation_period: Optional tuple of (start_date, end_date)
            
        Returns:
            Dictionary with evaluation results
        """
        if len(forecasts) == 0:
            return {'error': 'No forecasts to evaluate'}
        
        # Filter actual data if evaluation period is specified
        if evaluation_period:
            start_date, end_date = evaluation_period
            actual_data = actual_data[
                (actual_data['date'] >= start_date) & 
                (actual_data['date'] <= end_date)
            ].copy()
        
        # Merge forecasts with actual data
        evaluation_data = self._merge_forecasts_with_actual(forecasts, actual_data)
        
        if len(evaluation_data) == 0:
            return {'error': 'No matching actual data found for evaluation'}
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(
            evaluation_data['actual'].values,
            evaluation_data['forecast'].values
        )
        
        # Calculate metrics by location
        metrics_by_location = {}
        for location in evaluation_data['location'].unique():
            loc_data = evaluation_data[evaluation_data['location'] == location]
            metrics_by_location[location] = self.calculate_metrics(
                loc_data['actual'].values,
                loc_data['forecast'].values
            )
        
        # Calculate metrics by forecast horizon
        metrics_by_horizon = {}
        for horizon in evaluation_data['forecast_week'].unique():
            horizon_data = evaluation_data[evaluation_data['forecast_week'] == horizon]
            metrics_by_horizon[horizon] = self.calculate_metrics(
                horizon_data['actual'].values,
                horizon_data['forecast'].values
            )
        
        # Store results
        self.evaluation_results = {
            'evaluation_date': datetime.now().isoformat(),
            'n_forecasts_evaluated': len(evaluation_data),
            'overall_metrics': overall_metrics,
            'metrics_by_location': metrics_by_location,
            'metrics_by_horizon': metrics_by_horizon,
            'evaluation_data': evaluation_data.to_dict('records')
        }
        
        return self.evaluation_results
    
    def _merge_forecasts_with_actual(self, forecasts: pd.DataFrame, 
                                   actual_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge forecasts with actual data for evaluation
        
        Args:
            forecasts: Forecasts DataFrame
            actual_data: Actual data DataFrame
            
        Returns:
            Merged DataFrame for evaluation
        """
        # Convert date columns
        forecasts['forecast_date'] = pd.to_datetime(forecasts['forecast_date'])
        actual_data['date'] = pd.to_datetime(actual_data['date'])
        
        # Merge on location and date
        merged = forecasts.merge(
            actual_data[['location', 'date', 'value']],
            left_on=['location', 'forecast_date'],
            right_on=['location', 'date'],
            how='inner'
        )
        
        # Rename columns for clarity
        merged = merged.rename(columns={'value': 'actual'})
        
        return merged
    
    def generate_evaluation_report(self, results: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results (uses stored results if None)
            
        Returns:
            Dictionary with evaluation report
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            return {'error': 'No evaluation results available'}
        
        report = {
            'summary': {
                'evaluation_date': results['evaluation_date'],
                'n_forecasts': results['n_forecasts_evaluated'],
                'overall_performance': results['overall_metrics']
            },
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Performance analysis
        overall = results['overall_metrics']
        
        # Categorize performance
        if overall['mape'] < 20:
            performance_level = 'Excellent'
        elif overall['mape'] < 40:
            performance_level = 'Good'
        elif overall['mape'] < 60:
            performance_level = 'Fair'
        else:
            performance_level = 'Poor'
        
        report['performance_analysis'] = {
            'performance_level': performance_level,
            'mae_interpretation': self._interpret_mae(overall['mae']),
            'r2_interpretation': self._interpret_r2(overall['r2']),
            'bias_interpretation': self._interpret_bias(overall['bias'])
        }
        
        # Generate recommendations
        recommendations = []
        
        if overall['mape'] > 50:
            recommendations.append("Consider retraining the model with more recent data")
        
        if overall['r2'] < 0.3:
            recommendations.append("Model explains little variance - consider feature engineering")
        
        if abs(overall['bias']) > overall['mae'] * 0.5:
            recommendations.append("Model shows significant bias - consider bias correction")
        
        if len(results['metrics_by_location']) > 1:
            # Check for location-specific issues
            location_maes = {loc: metrics['mae'] for loc, metrics in results['metrics_by_location'].items()}
            worst_location = max(location_maes, key=location_maes.get)
            if location_maes[worst_location] > overall['mae'] * 1.5:
                recommendations.append(f"Model performs poorly for location {worst_location} - consider location-specific models")
        
        report['recommendations'] = recommendations
        
        return report
    
    def _interpret_mae(self, mae: float) -> str:
        """Interpret MAE value"""
        if mae < 100:
            return "Low error - good performance"
        elif mae < 500:
            return "Moderate error - acceptable performance"
        else:
            return "High error - needs improvement"
    
    def _interpret_r2(self, r2: float) -> str:
        """Interpret R-squared value"""
        if r2 > 0.8:
            return "Excellent model fit"
        elif r2 > 0.6:
            return "Good model fit"
        elif r2 > 0.4:
            return "Fair model fit"
        else:
            return "Poor model fit"
    
    def _interpret_bias(self, bias: float) -> str:
        """Interpret bias value"""
        if abs(bias) < 50:
            return "Minimal bias"
        elif abs(bias) < 200:
            return "Moderate bias"
        else:
            return "Significant bias"
    
    def create_evaluation_plots(self, results: Optional[Dict] = None, 
                              save_path: Optional[str] = None) -> List[str]:
        """
        Create evaluation visualization plots
        
        Args:
            results: Evaluation results (uses stored results if None)
            save_path: Directory to save plots (if None, don't save)
            
        Returns:
            List of plot file paths
        """
        if results is None:
            results = self.evaluation_results
        
        if not results or 'evaluation_data' not in results:
            return []
        
        # Convert evaluation data back to DataFrame
        eval_data = pd.DataFrame(results['evaluation_data'])
        
        plot_files = []
        
        # 1. Scatter plot: Actual vs Predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(eval_data['actual'], eval_data['forecast'], alpha=0.6)
        plt.plot([eval_data['actual'].min(), eval_data['actual'].max()], 
                [eval_data['actual'].min(), eval_data['actual'].max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, 'actual_vs_predicted.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plot_files.append(file_path)
        
        plt.show()
        
        # 2. Residuals plot
        plt.figure(figsize=(10, 6))
        residuals = eval_data['forecast'] - eval_data['actual']
        plt.scatter(eval_data['forecast'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            file_path = os.path.join(save_path, 'residuals_plot.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plot_files.append(file_path)
        
        plt.show()
        
        # 3. Performance by location
        if len(eval_data['location'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            location_maes = []
            locations = []
            
            for location in eval_data['location'].unique():
                loc_data = eval_data[eval_data['location'] == location]
                mae = mean_absolute_error(loc_data['actual'], loc_data['forecast'])
                location_maes.append(mae)
                locations.append(location)
            
            plt.bar(locations, location_maes)
            plt.xlabel('Location')
            plt.ylabel('MAE')
            plt.title('Model Performance by Location')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                file_path = os.path.join(save_path, 'performance_by_location.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plot_files.append(file_path)
            
            plt.show()
        
        # 4. Performance by forecast horizon
        if len(eval_data['forecast_week'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            horizon_maes = []
            horizons = []
            
            for horizon in sorted(eval_data['forecast_week'].unique()):
                horizon_data = eval_data[eval_data['forecast_week'] == horizon]
                mae = mean_absolute_error(horizon_data['actual'], horizon_data['forecast'])
                horizon_maes.append(mae)
                horizons.append(f'Week {horizon}')
            
            plt.bar(horizons, horizon_maes)
            plt.xlabel('Forecast Horizon')
            plt.ylabel('MAE')
            plt.title('Model Performance by Forecast Horizon')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                file_path = os.path.join(save_path, 'performance_by_horizon.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plot_files.append(file_path)
            
            plt.show()
        
        return plot_files
    
    def save_evaluation_results(self, results: Dict, output_dir: str) -> str:
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
            
        Returns:
            Path to saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results_file
