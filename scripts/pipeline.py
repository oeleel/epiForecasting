"""
End-to-end pipeline for flu hospitalization forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import argparse
import json

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src.direct_forecast import DirectForecastEnsemble
from src.evaluate import ModelEvaluator
from src import config


class FluForecastingPipeline:
    """End-to-end pipeline for flu hospitalization forecasting"""

    def __init__(self, output_dir: str = "outputs", model_dir: str = "models"):
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.evaluator = ModelEvaluator()
        self.ensemble = None

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    def run_full_pipeline(self, cutoff_date: str,
                         locations: Optional[List[str]] = None,
                         retrain: bool = True,
                         generate_forecasts: bool = True) -> Dict:
        """
        Run the complete forecasting pipeline

        Args:
            cutoff_date: Date to use as cutoff for training data
            locations: List of locations to forecast (if None, use all)
            retrain: Whether to retrain the model
            generate_forecasts: Whether to generate forecasts

        Returns:
            Dictionary with pipeline results
        """
        print("="*60)
        print("FLU FORECASTING PIPELINE")
        print("="*60)
        print(f"Cutoff date: {cutoff_date}")
        print(f"Retrain model: {retrain}")
        print(f"Generate forecasts: {generate_forecasts}")
        print("="*60)

        results = {
            'cutoff_date': cutoff_date,
            'pipeline_start_time': datetime.now().isoformat(),
            'success': False
        }

        try:
            # Step 1: Load and preprocess data
            print("\n1. Loading and preprocessing data...")
            loader = FluDataLoader()
            data = loader.load_and_preprocess(cutoff_date)

            # Filter locations if specified
            if locations is not None:
                data = data[data['location'].isin(locations)]
                print(f"Filtered to {len(locations)} locations")

            results['data_shape'] = data.shape
            results['locations'] = data['location'].unique().tolist()

            # Step 2: Feature engineering
            print("\n2. Creating features...")
            engineer = FeatureEngineer()
            features_df = engineer.create_all_features(data)
            feature_columns = engineer.get_feature_columns(features_df)

            results['feature_count'] = len(feature_columns)
            results['features'] = feature_columns

            # Step 3: Model training (if requested)
            if retrain:
                print("\n3. Training DirectForecastEnsemble...")
                self.ensemble = DirectForecastEnsemble()

                # Pre-filter data to cutoff
                cutoff_dt = pd.to_datetime(cutoff_date)
                training_data = features_df[features_df['date'] <= cutoff_dt].copy()

                training_results = self.ensemble.train(training_data)

                # Save models
                models_dir = self.ensemble.save_models(self.model_dir)

                # Save feature importance from horizon 1 as representative
                feature_importance = self.ensemble.get_feature_importance(horizon=1)
                feature_importance_file = os.path.join(self.output_dir, f"feature_importance_{cutoff_date}.csv")
                feature_importance.to_csv(feature_importance_file, index=False)

                results['training_results'] = training_results
                results['model_path'] = models_dir
                results['feature_importance_file'] = feature_importance_file
                results['model_trained'] = True

                print(f"Models saved to: {models_dir}")
                print(f"Feature importance saved to: {feature_importance_file}")

            # Step 4: Generate forecasts (if requested)
            if generate_forecasts:
                print("\n4. Generating forecasts...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                if not retrain:
                    # Load existing models
                    models_dir = os.path.join(self.model_dir, 'direct_forecast_models')
                    if not os.path.exists(models_dir):
                        raise FileNotFoundError(
                            "No trained model found. Set retrain=True to train a new model."
                        )
                    self.ensemble = DirectForecastEnsemble()
                    self.ensemble.load_models(models_dir)

                forecasts = self.ensemble.generate_forecasts(
                    data=features_df,
                    cutoff_date=cutoff_date,
                    locations=locations
                )

                # Save forecasts
                forecast_file = os.path.join(
                    self.output_dir, f"forecasts_{cutoff_date}_{timestamp}.csv"
                )
                forecasts.to_csv(forecast_file, index=False)

                results['forecasts'] = forecasts
                results['forecast_file'] = forecast_file
                results['forecast_count'] = len(forecasts)

                print(f"Forecasts saved to: {forecast_file}")

            # Step 5: Model evaluation
            print("\n5. Evaluating model...")
            evaluation_results = self.evaluate_model_performance(
                cutoff_date, locations, features_df=features_df
            )
            results['evaluation_results'] = evaluation_results
            results['evaluation_completed'] = True

            if 'evaluation_file' in evaluation_results:
                results['evaluation_file'] = evaluation_results['evaluation_file']

            results['success'] = True
            results['pipeline_end_time'] = datetime.now().isoformat()

            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)

        except Exception as e:
            print(f"\nPipeline failed with error: {str(e)}")
            results['error'] = str(e)
            results['pipeline_end_time'] = datetime.now().isoformat()

        return results

    def update_with_new_data(self, new_cutoff_date: str,
                           locations: Optional[List[str]] = None) -> Dict:
        """
        Update model with new data and generate fresh forecasts

        Args:
            new_cutoff_date: New cutoff date for updated data
            locations: List of locations to forecast

        Returns:
            Dictionary with update results
        """
        print(f"Updating pipeline with new data up to {new_cutoff_date}")

        return self.run_full_pipeline(
            cutoff_date=new_cutoff_date,
            locations=locations,
            retrain=True,
            generate_forecasts=True
        )

    def evaluate_model_performance(self, cutoff_date: str,
                                 locations: Optional[List[str]] = None,
                                 features_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate model performance using DirectForecastEnsemble's built-in
        train/validation split metrics.

        Args:
            cutoff_date: Date to use as cutoff for evaluation
            locations: List of locations to evaluate
            features_df: Pre-computed features (avoids reloading data)

        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating model performance up to {cutoff_date}")

        # Use pre-computed features if available, otherwise load fresh
        if features_df is None:
            loader = FluDataLoader()
            data = loader.load_and_preprocess(cutoff_date)

            if locations is not None:
                data = data[data['location'].isin(locations)]

            engineer = FeatureEngineer()
            features_df = engineer.create_all_features(data)

        # Train a DirectForecastEnsemble (which does its own 80/20 split)
        ensemble = DirectForecastEnsemble()
        cutoff_dt = pd.to_datetime(cutoff_date)
        training_data = features_df[features_df['date'] <= cutoff_dt].copy()
        training_results = ensemble.train(training_data)

        # Extract real per-horizon metrics from training results
        overall_mae_list = []
        overall_rmse_list = []
        overall_mape_list = []

        metrics_by_horizon = {}
        for horizon, h_metrics in training_results.get('horizons', {}).items():
            horizon_key = int(horizon)
            metrics_by_horizon[horizon_key] = {
                'mae': h_metrics.get('val_mae_counts'),
                'rmse': h_metrics.get('val_rmse_counts'),
                'mape': h_metrics.get('val_mape_counts'),
                'n_train_samples': h_metrics.get('n_train_samples'),
                'n_val_samples': h_metrics.get('n_val_samples')
            }
            if h_metrics.get('val_mae_counts') is not None:
                overall_mae_list.append(h_metrics['val_mae_counts'])
            if h_metrics.get('val_rmse_counts') is not None:
                overall_rmse_list.append(h_metrics['val_rmse_counts'])
            if h_metrics.get('val_mape_counts') is not None:
                overall_mape_list.append(h_metrics['val_mape_counts'])

        overall_metrics = {
            'mae': float(np.mean(overall_mae_list)) if overall_mae_list else None,
            'rmse': float(np.mean(overall_rmse_list)) if overall_rmse_list else None,
            'mape': float(np.mean(overall_mape_list)) if overall_mape_list else None,
            'target_mode': training_results.get('target_mode')
        }

        evaluation_results = {
            'cutoff_date': cutoff_date,
            'training_results': training_results,
            'locations_evaluated': len(features_df['location'].unique()),
            'overall_metrics': overall_metrics,
            'metrics_by_horizon': metrics_by_horizon
        }

        # Save evaluation results
        eval_file = os.path.join(
            self.output_dir,
            f"evaluation_{cutoff_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        print(f"Evaluation results saved to {eval_file}")
        evaluation_results['evaluation_file'] = eval_file

        return evaluation_results

    def generate_forecast_report(self, forecast_file: str) -> str:
        """
        Generate a summary report of forecasts

        Args:
            forecast_file: Path to forecast CSV file

        Returns:
            Path to generated report
        """
        if not os.path.exists(forecast_file):
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")

        forecasts = pd.read_csv(forecast_file)

        # Determine column names (handle both old and new formats)
        date_col = 'forecast_date' if 'forecast_date' in forecasts.columns else 'date'
        week_col = 'forecast_week' if 'forecast_week' in forecasts.columns else 'prediction_horizon'
        forecast_col = 'forecast' if 'forecast' in forecasts.columns else 'predicted_value'

        # Generate report
        report_lines = [
            "# Flu Hospitalization Forecast Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Forecast file: {forecast_file}",
            "",
            "## Summary Statistics",
            f"Total predictions: {len(forecasts)}",
            f"Locations: {forecasts['location'].nunique()}",
            f"Date range: {forecasts[date_col].min()} to {forecasts[date_col].max()}",
            f"Prediction horizons: {sorted(forecasts[week_col].unique())}",
            "",
            "## Forecasts by Location and Horizon",
            ""
        ]

        # Add forecasts by location and horizon
        for location in sorted(forecasts['location'].unique()):
            loc_data = forecasts[forecasts['location'] == location]
            location_name = loc_data['location_name'].iloc[0] if 'location_name' in loc_data.columns else location

            report_lines.append(f"### {location} - {location_name}")

            for horizon in sorted(loc_data[week_col].unique()):
                horizon_data = loc_data[loc_data[week_col] == horizon]

                report_lines.append(f"**{horizon}-week ahead:**")
                for _, row in horizon_data.iterrows():
                    pred_val = row[forecast_col]
                    if 'confidence_lower' in row and 'confidence_upper' in row:
                        ci_lower = row['confidence_lower']
                        ci_upper = row['confidence_upper']
                        report_lines.append(f"  - {row[date_col]}: {pred_val:.1f} (95% CI: {ci_lower:.1f} - {ci_upper:.1f})")
                    elif 'pred_lower_95' in row and 'pred_upper_95' in row:
                        ci_lower = row['pred_lower_95']
                        ci_upper = row['pred_upper_95']
                        report_lines.append(f"  - {row[date_col]}: {pred_val:.1f} (95% CI: {ci_lower:.1f} - {ci_upper:.1f})")
                    else:
                        report_lines.append(f"  - {row[date_col]}: {pred_val:.1f}")
                report_lines.append("")

        # Save report
        report_file = forecast_file.replace('.csv', '_report.md')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Forecast report saved to {report_file}")
        return report_file


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Flu Forecasting Pipeline')
    parser.add_argument('--cutoff-date', type=str, default=config.DEFAULT_CUTOFF_DATE,
                       help='Cutoff date for training data (YYYY-MM-DD)')
    parser.add_argument('--locations', nargs='+', default=None,
                       help='List of locations to forecast (if not specified, forecast all)')
    parser.add_argument('--retrain', action='store_true', default=True,
                       help='Whether to retrain the model')
    parser.add_argument('--no-forecasts', action='store_true', default=False,
                       help='Skip forecast generation')
    parser.add_argument('--confidence', action='store_true', default=False,
                       help='Include confidence intervals (requires QuantileDirectForecastEnsemble, not yet wired in)')
    parser.add_argument('--evaluate-only', action='store_true', default=False,
                       help='Only run evaluation, skip training and forecasting')

    args = parser.parse_args()

    if args.confidence:
        print("WARNING: --confidence flag is not yet supported with DirectForecastEnsemble.")
        print("         Use QuantileDirectForecastEnsemble for confidence intervals (future work).")

    # Initialize pipeline
    pipeline = FluForecastingPipeline()

    if args.evaluate_only:
        # Run evaluation only
        results = pipeline.evaluate_model_performance(
            cutoff_date=args.cutoff_date,
            locations=args.locations
        )
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            cutoff_date=args.cutoff_date,
            locations=args.locations,
            retrain=args.retrain,
            generate_forecasts=not args.no_forecasts
        )

        # Generate report if forecasts were created
        if results['success'] and 'forecast_file' in results:
            pipeline.generate_forecast_report(results['forecast_file'])

    # Save pipeline results
    results_file = os.path.join(
        pipeline.output_dir,
        f"pipeline_results_{args.cutoff_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Pipeline results saved to {results_file}")


if __name__ == "__main__":
    main()
