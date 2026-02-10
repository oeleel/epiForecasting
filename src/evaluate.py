"""
Model evaluation module for flu forecasting

Includes:
- ModelEvaluator: Standard point prediction metrics (MAE, RMSE, MAPE)
- QuantileEvaluator: Prediction interval metrics (coverage, calibration, Winkler score)
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
from src import config


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


class QuantileEvaluator:
    """
    Evaluator for quantile/probabilistic forecasts.

    Metrics:
    - Coverage: Fraction of actuals within prediction interval
    - Interval width: Average width of prediction intervals
    - Calibration: Fraction of actuals below each quantile (should match quantile level)
    - Winkler score: Interval score penalizing both width and non-coverage
    """

    def __init__(self, quantiles: Optional[List[float]] = None):
        """
        Initialize quantile evaluator.

        Args:
            quantiles: List of quantiles used in forecasts (default from config)
        """
        self.quantiles = quantiles or getattr(config, 'QUANTILES', [0.05, 0.25, 0.5, 0.75, 0.95])
        self.evaluation_results = {}

    def calculate_coverage(self, actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Calculate coverage: fraction of actuals within [lower, upper] interval.

        Args:
            actuals: Actual values
            lower: Lower bound predictions
            upper: Upper bound predictions

        Returns:
            Coverage fraction (0-1)
        """
        covered = (actuals >= lower) & (actuals <= upper)
        return float(np.mean(covered))

    def calculate_interval_width(self, lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Calculate average interval width.

        Args:
            lower: Lower bound predictions
            upper: Upper bound predictions

        Returns:
            Mean interval width
        """
        return float(np.mean(upper - lower))

    def calculate_calibration(self, actuals: np.ndarray, predictions: np.ndarray,
                             quantile: float) -> float:
        """
        Calculate calibration for a single quantile.
        Returns fraction of actuals below the predicted quantile.
        Should be close to the quantile level (e.g., 0.05 for q05).

        Args:
            actuals: Actual values
            predictions: Predicted quantile values
            quantile: Quantile level (0-1)

        Returns:
            Fraction of actuals below prediction
        """
        below = actuals < predictions
        return float(np.mean(below))

    def calculate_winkler_score(self, actuals: np.ndarray, lower: np.ndarray,
                               upper: np.ndarray, alpha: float = 0.1) -> float:
        """
        Calculate Winkler score (interval score) for a prediction interval.

        The Winkler score rewards narrow intervals but penalizes when actuals
        fall outside the interval. Lower is better.

        For a (1-alpha) prediction interval:
        - If actual is within interval: score = width
        - If actual < lower: score = width + (2/alpha) * (lower - actual)
        - If actual > upper: score = width + (2/alpha) * (actual - upper)

        Args:
            actuals: Actual values
            lower: Lower bound predictions (alpha/2 quantile)
            upper: Upper bound predictions (1-alpha/2 quantile)
            alpha: Significance level (0.1 for 90% interval)

        Returns:
            Mean Winkler score
        """
        width = upper - lower
        scores = width.copy()

        # Penalty for actuals below lower bound
        below_mask = actuals < lower
        scores[below_mask] += (2 / alpha) * (lower[below_mask] - actuals[below_mask])

        # Penalty for actuals above upper bound
        above_mask = actuals > upper
        scores[above_mask] += (2 / alpha) * (actuals[above_mask] - upper[above_mask])

        return float(np.mean(scores))

    def evaluate_quantile_forecasts(self, forecasts: pd.DataFrame,
                                   actual_data: pd.DataFrame) -> Dict:
        """
        Evaluate quantile forecasts against actual data.

        Args:
            forecasts: DataFrame with quantile predictions (predicted_q05, q25, q50, q75, q95)
            actual_data: DataFrame with actual values

        Returns:
            Dictionary with quantile evaluation metrics
        """
        # Merge forecasts with actuals
        forecasts = forecasts.copy()
        actual_data = actual_data.copy()

        forecasts['forecast_date'] = pd.to_datetime(forecasts['forecast_date'])
        actual_data['date'] = pd.to_datetime(actual_data['date'])

        merged = forecasts.merge(
            actual_data[['location', 'date', 'value']],
            left_on=['location', 'forecast_date'],
            right_on=['location', 'date'],
            how='inner'
        )
        merged = merged.rename(columns={'value': 'actual'})

        if len(merged) == 0:
            return {'error': 'No matching actual data found for evaluation'}

        actuals = merged['actual'].values

        # Point prediction metrics (using q50 median)
        point_predictions = merged['predicted_q50'].values
        point_metrics = ModelEvaluator().calculate_metrics(actuals, point_predictions)

        # 90% interval coverage and width (q05 to q95)
        q05 = merged['predicted_q05'].values
        q95 = merged['predicted_q95'].values
        coverage_90 = self.calculate_coverage(actuals, q05, q95)
        width_90 = self.calculate_interval_width(q05, q95)
        winkler_90 = self.calculate_winkler_score(actuals, q05, q95, alpha=0.1)

        # 50% interval coverage and width (q25 to q75)
        q25 = merged['predicted_q25'].values
        q75 = merged['predicted_q75'].values
        coverage_50 = self.calculate_coverage(actuals, q25, q75)
        width_50 = self.calculate_interval_width(q25, q75)

        # Calibration for each quantile
        calibration = {}
        quantile_cols = {
            0.05: 'predicted_q05',
            0.25: 'predicted_q25',
            0.50: 'predicted_q50',
            0.75: 'predicted_q75',
            0.95: 'predicted_q95'
        }

        for q, col in quantile_cols.items():
            if col in merged.columns:
                calibration[f'q{int(q*100):02d}'] = self.calculate_calibration(
                    actuals, merged[col].values, q
                )

        # Metrics by horizon
        metrics_by_horizon = {}
        for horizon in merged['forecast_week'].unique():
            h_data = merged[merged['forecast_week'] == horizon]
            h_actuals = h_data['actual'].values
            h_q05 = h_data['predicted_q05'].values
            h_q95 = h_data['predicted_q95'].values
            h_q50 = h_data['predicted_q50'].values

            # Point metrics
            h_point_metrics = ModelEvaluator().calculate_metrics(h_actuals, h_q50)

            metrics_by_horizon[int(horizon)] = {
                'coverage_90pct': self.calculate_coverage(h_actuals, h_q05, h_q95),
                'mean_interval_width_90pct': self.calculate_interval_width(h_q05, h_q95),
                'winkler_score_90pct': self.calculate_winkler_score(h_actuals, h_q05, h_q95, 0.1),
                'mape': h_point_metrics.get('mape'),
                'mae': h_point_metrics.get('mae'),
                'n_samples': len(h_data)
            }

        self.evaluation_results = {
            'evaluation_date': datetime.now().isoformat(),
            'n_forecasts_evaluated': len(merged),
            'point_prediction_metrics': point_metrics,
            'quantile_metrics': {
                'coverage_90pct': coverage_90,
                'coverage_50pct': coverage_50,
                'mean_interval_width_90pct': width_90,
                'mean_interval_width_50pct': width_50,
                'calibration': calibration,
                'mean_winkler_score_90pct': winkler_90
            },
            'metrics_by_horizon': metrics_by_horizon
        }

        return self.evaluation_results

    def print_quantile_summary(self, results: Optional[Dict] = None) -> None:
        """Print a summary of quantile evaluation results."""
        if results is None:
            results = self.evaluation_results

        if not results:
            print("No evaluation results available")
            return

        print("\n" + "=" * 70)
        print("QUANTILE FORECAST EVALUATION SUMMARY")
        print("=" * 70)

        print(f"\nEvaluation date: {results['evaluation_date']}")
        print(f"Forecasts evaluated: {results['n_forecasts_evaluated']}")

        # Point prediction metrics
        pm = results['point_prediction_metrics']
        print(f"\nPoint Prediction (Median) Metrics:")
        print(f"  MAPE:  {pm.get('mape', 0):.2f}%")
        print(f"  MAE:   {pm.get('mae', 0):.2f}")
        print(f"  RMSE:  {pm.get('rmse', 0):.2f}")

        # Quantile metrics
        qm = results['quantile_metrics']
        print(f"\n90% Prediction Interval (q05-q95):")
        print(f"  Coverage: {qm['coverage_90pct']*100:.1f}% (target: 90%)")
        print(f"  Mean Width: {qm['mean_interval_width_90pct']:.1f}")
        print(f"  Winkler Score: {qm['mean_winkler_score_90pct']:.1f}")

        print(f"\n50% Prediction Interval (q25-q75):")
        print(f"  Coverage: {qm['coverage_50pct']*100:.1f}% (target: 50%)")
        print(f"  Mean Width: {qm['mean_interval_width_50pct']:.1f}")

        print(f"\nCalibration (fraction of actuals below quantile):")
        for q, val in qm['calibration'].items():
            target = int(q[1:]) / 100
            status = "OK" if abs(val - target) < 0.05 else "MISCALIBRATED"
            print(f"  {q}: {val:.3f} (target: {target:.2f}) - {status}")

        # By horizon
        print(f"\nMetrics by Horizon:")
        for h, m in sorted(results['metrics_by_horizon'].items()):
            print(f"  Horizon {h}: Coverage={m['coverage_90pct']*100:.1f}%, "
                  f"Width={m['mean_interval_width_90pct']:.1f}, "
                  f"MAPE={m['mape']:.1f}%")

        print("=" * 70)


class ClusteredEvaluator:
    """
    Evaluator for clustered forecasting models.

    Provides metrics broken down by cluster and comparison between
    unified and clustered model approaches.
    """

    def __init__(self, location_to_cluster: Optional[Dict[str, int]] = None):
        """
        Initialize clustered evaluator.

        Args:
            location_to_cluster: Mapping of location -> cluster ID
        """
        self.location_to_cluster = location_to_cluster or {}
        self.model_evaluator = ModelEvaluator()
        self.quantile_evaluator = QuantileEvaluator()
        self.evaluation_results = {}

    def set_cluster_mapping(self, location_to_cluster: Dict[str, int]) -> None:
        """Set or update the location to cluster mapping."""
        self.location_to_cluster = location_to_cluster

    def evaluate_forecasts(self, forecasts: pd.DataFrame,
                          actual_data: pd.DataFrame,
                          is_quantile: bool = False) -> Dict:
        """
        Evaluate forecast performance with cluster breakdown.

        Args:
            forecasts: Forecasts DataFrame (must have 'cluster' column or location_to_cluster set)
            actual_data: Actual data DataFrame with date, location, value
            is_quantile: If True, also compute quantile metrics

        Returns:
            Dictionary with overall, by-horizon, and by-cluster metrics
        """
        forecasts = forecasts.copy()
        actual_data = actual_data.copy()

        # Add cluster column if not present
        if 'cluster' not in forecasts.columns:
            if not self.location_to_cluster:
                raise ValueError("Cluster mapping not available. Either set location_to_cluster or include 'cluster' column in forecasts.")
            forecasts['cluster'] = forecasts['location'].map(self.location_to_cluster)
        else:
            # Update our mapping from the forecasts
            for _, row in forecasts[['location', 'cluster']].drop_duplicates().iterrows():
                self.location_to_cluster[row['location']] = row['cluster']

        # Convert dates
        forecasts['forecast_date'] = pd.to_datetime(forecasts['forecast_date'])
        actual_data['date'] = pd.to_datetime(actual_data['date'])

        # Merge with actuals
        merged = forecasts.merge(
            actual_data[['location', 'date', 'value']],
            left_on=['location', 'forecast_date'],
            right_on=['location', 'date'],
            how='inner'
        )
        merged = merged.rename(columns={'value': 'actual'})

        if len(merged) == 0:
            return {'error': 'No matching actual data found for evaluation'}

        # Overall metrics
        if 'predicted' in merged.columns:
            predictions = merged['predicted'].values
        elif 'predicted_q50' in merged.columns:
            predictions = merged['predicted_q50'].values
        else:
            predictions = merged['forecast'].values

        actuals = merged['actual'].values
        overall_metrics = self.model_evaluator.calculate_metrics(actuals, predictions)

        # Metrics by horizon
        metrics_by_horizon = {}
        for horizon in sorted(merged['forecast_week'].unique()):
            h_data = merged[merged['forecast_week'] == horizon]
            h_preds = h_data['predicted'].values if 'predicted' in h_data.columns else h_data['predicted_q50'].values
            h_metrics = self.model_evaluator.calculate_metrics(
                h_data['actual'].values,
                h_preds
            )
            if is_quantile and 'predicted_q05' in h_data.columns:
                h_metrics['coverage_90pct'] = self.quantile_evaluator.calculate_coverage(
                    h_data['actual'].values,
                    h_data['predicted_q05'].values,
                    h_data['predicted_q95'].values
                )
            metrics_by_horizon[int(horizon)] = h_metrics

        # Metrics by cluster
        metrics_by_cluster = {}
        for cluster_id in sorted(merged['cluster'].unique()):
            c_data = merged[merged['cluster'] == cluster_id]
            c_preds = c_data['predicted'].values if 'predicted' in c_data.columns else c_data['predicted_q50'].values
            c_metrics = self.model_evaluator.calculate_metrics(
                c_data['actual'].values,
                c_preds
            )
            c_metrics['n_locations'] = c_data['location'].nunique()
            c_metrics['locations'] = c_data['location'].unique().tolist()

            # Quantile metrics per cluster
            if is_quantile and 'predicted_q05' in c_data.columns:
                c_metrics['coverage_90pct'] = self.quantile_evaluator.calculate_coverage(
                    c_data['actual'].values,
                    c_data['predicted_q05'].values,
                    c_data['predicted_q95'].values
                )
                c_metrics['mean_interval_width_90pct'] = self.quantile_evaluator.calculate_interval_width(
                    c_data['predicted_q05'].values,
                    c_data['predicted_q95'].values
                )

            metrics_by_cluster[int(cluster_id)] = c_metrics

        self.evaluation_results = {
            'evaluation_date': datetime.now().isoformat(),
            'n_forecasts_evaluated': len(merged),
            'n_clusters': len(metrics_by_cluster),
            'overall_metrics': overall_metrics,
            'metrics_by_horizon': metrics_by_horizon,
            'metrics_by_cluster': metrics_by_cluster
        }

        return self.evaluation_results

    def compare_models(self, unified_forecasts: pd.DataFrame,
                      clustered_forecasts: pd.DataFrame,
                      actual_data: pd.DataFrame,
                      is_quantile: bool = False) -> Dict:
        """
        Compare unified model vs clustered model performance.

        Args:
            unified_forecasts: Forecasts from unified (single) model
            clustered_forecasts: Forecasts from clustered models
            actual_data: Actual data DataFrame
            is_quantile: If True, also compute quantile comparison

        Returns:
            Dictionary with comparison metrics
        """
        # Prepare unified forecasts with 'forecast' column for ModelEvaluator
        unified_df = unified_forecasts.copy()
        if 'forecast' not in unified_df.columns:
            if 'predicted' in unified_df.columns:
                unified_df['forecast'] = unified_df['predicted']
            elif 'predicted_q50' in unified_df.columns:
                unified_df['forecast'] = unified_df['predicted_q50']

        # Evaluate unified model
        unified_eval = self.model_evaluator.evaluate_forecasts(unified_df, actual_data)
        unified_overall = unified_eval.get('overall_metrics', {})

        # Evaluate clustered model
        clustered_eval = self.evaluate_forecasts(clustered_forecasts, actual_data, is_quantile)
        clustered_overall = clustered_eval.get('overall_metrics', {})

        # Compute improvements
        unified_mape = unified_overall.get('mape', 0)
        clustered_mape = clustered_overall.get('mape', 0)

        unified_mae = unified_overall.get('mae', 0)
        clustered_mae = clustered_overall.get('mae', 0)

        comparison = {
            'evaluation_date': datetime.now().isoformat(),
            'unified_model': {
                'mape': unified_mape,
                'mae': unified_mae,
                'rmse': unified_overall.get('rmse'),
                'n_forecasts': unified_eval.get('n_forecasts_evaluated', 0)
            },
            'clustered_model': {
                'mape': clustered_mape,
                'mae': clustered_mae,
                'rmse': clustered_overall.get('rmse'),
                'n_clusters': clustered_eval.get('n_clusters', 0),
                'n_forecasts': clustered_eval.get('n_forecasts_evaluated', 0)
            },
            'improvement': {
                'mape_reduction_pct': ((unified_mape - clustered_mape) / unified_mape * 100) if unified_mape > 0 else 0,
                'mae_reduction_pct': ((unified_mae - clustered_mae) / unified_mae * 100) if unified_mae > 0 else 0,
                'clustered_better': clustered_mape < unified_mape
            },
            'cluster_breakdown': clustered_eval.get('metrics_by_cluster', {})
        }

        return comparison

    def get_worst_clusters(self, n: int = 3) -> List[Dict]:
        """
        Get the N clusters with worst performance.

        Args:
            n: Number of worst clusters to return

        Returns:
            List of cluster info dictionaries
        """
        if not self.evaluation_results or 'metrics_by_cluster' not in self.evaluation_results:
            return []

        cluster_metrics = self.evaluation_results['metrics_by_cluster']
        sorted_clusters = sorted(
            cluster_metrics.items(),
            key=lambda x: x[1].get('mape', 0),
            reverse=True
        )

        worst = []
        for cluster_id, metrics in sorted_clusters[:n]:
            worst.append({
                'cluster': cluster_id,
                'mape': metrics.get('mape'),
                'mae': metrics.get('mae'),
                'n_locations': metrics.get('n_locations'),
                'n_samples': metrics.get('n_samples')
            })

        return worst

    def get_best_clusters(self, n: int = 3) -> List[Dict]:
        """
        Get the N clusters with best performance.

        Args:
            n: Number of best clusters to return

        Returns:
            List of cluster info dictionaries
        """
        if not self.evaluation_results or 'metrics_by_cluster' not in self.evaluation_results:
            return []

        cluster_metrics = self.evaluation_results['metrics_by_cluster']
        sorted_clusters = sorted(
            cluster_metrics.items(),
            key=lambda x: x[1].get('mape', float('inf'))
        )

        best = []
        for cluster_id, metrics in sorted_clusters[:n]:
            best.append({
                'cluster': cluster_id,
                'mape': metrics.get('mape'),
                'mae': metrics.get('mae'),
                'n_locations': metrics.get('n_locations'),
                'n_samples': metrics.get('n_samples')
            })

        return best

    def print_cluster_summary(self, results: Optional[Dict] = None) -> None:
        """Print a summary of cluster-based evaluation results."""
        if results is None:
            results = self.evaluation_results

        if not results:
            print("No evaluation results available")
            return

        print("\n" + "=" * 70)
        print("CLUSTERED MODEL EVALUATION SUMMARY")
        print("=" * 70)

        print(f"\nEvaluation date: {results['evaluation_date']}")
        print(f"Forecasts evaluated: {results['n_forecasts_evaluated']}")
        print(f"Number of clusters: {results['n_clusters']}")

        # Overall metrics
        om = results['overall_metrics']
        print(f"\nOverall Metrics:")
        print(f"  MAPE:  {om.get('mape', 0):.2f}%")
        print(f"  MAE:   {om.get('mae', 0):.2f}")
        print(f"  RMSE:  {om.get('rmse', 0):.2f}")

        # By horizon
        print(f"\nMetrics by Horizon:")
        for h, m in sorted(results['metrics_by_horizon'].items()):
            cov_str = f", Coverage={m['coverage_90pct']*100:.1f}%" if 'coverage_90pct' in m else ""
            print(f"  Horizon {h}: MAPE={m.get('mape', 0):.1f}%, MAE={m.get('mae', 0):.1f}{cov_str}")

        # By cluster
        print(f"\nMetrics by Cluster:")
        for c, m in sorted(results['metrics_by_cluster'].items()):
            locations_str = ', '.join(m.get('locations', [])[:3])
            if len(m.get('locations', [])) > 3:
                locations_str += '...'
            cov_str = f", Coverage={m['coverage_90pct']*100:.1f}%" if 'coverage_90pct' in m else ""
            print(f"  Cluster {c} ({m.get('n_locations', 0)} locs): MAPE={m.get('mape', 0):.1f}%, "
                  f"MAE={m.get('mae', 0):.1f}{cov_str}")
            print(f"    Locations: {locations_str}")

        print("=" * 70)

    def print_comparison_summary(self, comparison: Dict) -> None:
        """Print a summary of unified vs clustered model comparison."""
        print("\n" + "=" * 70)
        print("UNIFIED vs CLUSTERED MODEL COMPARISON")
        print("=" * 70)

        unified = comparison['unified_model']
        clustered = comparison['clustered_model']
        improvement = comparison['improvement']

        print(f"\nUnified Model:")
        print(f"  MAPE: {unified['mape']:.2f}%")
        print(f"  MAE:  {unified['mae']:.2f}")
        print(f"  RMSE: {unified['rmse']:.2f}")

        print(f"\nClustered Model ({clustered['n_clusters']} clusters):")
        print(f"  MAPE: {clustered['mape']:.2f}%")
        print(f"  MAE:  {clustered['mae']:.2f}")
        print(f"  RMSE: {clustered['rmse']:.2f}")

        winner = "CLUSTERED" if improvement['clustered_better'] else "UNIFIED"
        print(f"\nComparison:")
        print(f"  MAPE Reduction: {improvement['mape_reduction_pct']:.2f}%")
        print(f"  MAE Reduction:  {improvement['mae_reduction_pct']:.2f}%")
        print(f"  Winner: {winner}")

        print("=" * 70)
