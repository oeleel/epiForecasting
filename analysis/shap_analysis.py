"""
SHAP Analysis Module for Flu Forecasting Models

Provides comprehensive SHAP-based feature importance analysis for the
DirectForecastEnsemble, including per-horizon analysis, cross-horizon
comparisons, and feature recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class SHAPAnalyzer:
    """
    SHAP-based analysis for flu forecasting models.

    Generates comprehensive feature importance analysis including:
    - Per-horizon bar and beeswarm plots
    - Dependence plots for key features
    - Cross-horizon feature importance comparison
    - Feature recommendations for model simplification
    """

    # Features to generate dependence plots for (updated for feature version v2)
    DEPENDENCE_FEATURES = [
        # Core lag and rolling features
        'value_lag_1', 'value_lag_2', 'value_rolling_mean_4',
        'value_rolling_mean_8', 'value_rolling_std_8', 'us_total_lag_1',
        'us_total_rolling_mean_4',
        # Temporal features
        'week_sin', 'week_cos',
        # Rate of change features
        'wow_change', 'acceleration', 'momentum_4w',
        # New v2 trend features
        'linear_trend_slope_4w', 'linear_trend_slope_3w',
        'value_to_rolling_mean_ratio', 'consecutive_increase_weeks',
        'consecutive_decrease_weeks', 'historical_percentile',
        'ratio_to_historical_max', 'surge_indicator'
    ]

    def __init__(self):
        """Initialize the SHAP analyzer."""
        self.shap_values = {}  # Store SHAP values per horizon
        self.explainers = {}   # Store explainers per horizon
        self.feature_importance = {}  # Store mean |SHAP| per horizon

    def _calculate_figure_size(self, n_features: int, width: int = 14) -> Tuple[int, int]:
        """
        Calculate dynamic figure size based on number of features.

        Args:
            n_features: Number of features to display
            width: Figure width in inches

        Returns:
            Tuple of (width, height) in inches
        """
        height = max(12, n_features * 0.35)
        return (width, height)

    def _compute_shap_values(self, model, X: pd.DataFrame, horizon: int) -> shap.Explanation:
        """
        Compute SHAP values for a given model and data.

        Args:
            model: The XGBoost model (FluForecastingModel instance)
            X: Feature DataFrame
            horizon: Horizon number (for logging)

        Returns:
            SHAP Explanation object
        """
        print(f"  Computing SHAP values for horizon {horizon}...")

        # Get the underlying XGBoost model
        xgb_model = model.model

        # Create TreeExplainer
        explainer = shap.TreeExplainer(xgb_model)
        self.explainers[horizon] = explainer

        # Compute SHAP values
        shap_values = explainer(X)

        return shap_values

    def _generate_bar_plot(self, shap_values: shap.Explanation,
                          output_path: str, horizon: int) -> None:
        """
        Generate bar summary plot showing all features ranked by mean |SHAP|.

        Args:
            shap_values: SHAP Explanation object
            output_path: Path to save the plot
            horizon: Horizon number for title
        """
        n_features = shap_values.values.shape[1]
        fig_width, fig_height = self._calculate_figure_size(n_features)

        plt.figure(figsize=(fig_width, fig_height))
        shap.plots.bar(shap_values, max_display=n_features, show=False)
        plt.title(f'Feature Importance (Mean |SHAP|) - Horizon {horizon} Week',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close('all')

    def _generate_beeswarm_plot(self, shap_values: shap.Explanation,
                                output_path: str, horizon: int) -> None:
        """
        Generate beeswarm summary plot showing all features.

        Args:
            shap_values: SHAP Explanation object
            output_path: Path to save the plot
            horizon: Horizon number for title
        """
        n_features = shap_values.values.shape[1]
        fig_width, fig_height = self._calculate_figure_size(n_features)

        plt.figure(figsize=(fig_width, fig_height))
        shap.plots.beeswarm(shap_values, max_display=n_features, show=False)
        plt.title(f'SHAP Beeswarm Plot - Horizon {horizon} Week',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close('all')

    def _generate_dependence_plots(self, shap_values: shap.Explanation,
                                   X: pd.DataFrame, output_dir: str,
                                   horizon: int) -> None:
        """
        Generate dependence plots for specified features.

        Args:
            shap_values: SHAP Explanation object
            X: Feature DataFrame
            output_dir: Directory to save plots
            horizon: Horizon number
        """
        dependence_dir = os.path.join(output_dir, 'dependence_plots')
        os.makedirs(dependence_dir, exist_ok=True)

        for feature in self.DEPENDENCE_FEATURES:
            if feature in X.columns:
                try:
                    plt.figure(figsize=(10, 7))
                    shap.plots.scatter(shap_values[:, feature], show=False)
                    plt.title(f'SHAP Dependence: {feature} - Horizon {horizon} Week',
                              fontsize=12, fontweight='bold')
                    plt.tight_layout()

                    output_path = os.path.join(dependence_dir, f'{feature}.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close('all')
                except Exception as e:
                    print(f"    Warning: Could not generate dependence plot for {feature}: {e}")
                    plt.close('all')

    def _compute_feature_importance(self, shap_values: shap.Explanation,
                                    feature_names: List[str]) -> pd.DataFrame:
        """
        Compute mean |SHAP| for each feature.

        Args:
            shap_values: SHAP Explanation object
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        importance_df['rank'] = range(1, len(importance_df) + 1)

        return importance_df

    def _analyze_horizon(self, ensemble, X: pd.DataFrame,
                        horizon: int, output_dir: str) -> pd.DataFrame:
        """
        Run complete SHAP analysis for a single horizon.

        Args:
            ensemble: DirectForecastEnsemble instance
            X: Feature DataFrame
            horizon: Horizon number (1-4)
            output_dir: Directory to save outputs

        Returns:
            DataFrame with feature importance for this horizon
        """
        print(f"\nAnalyzing Horizon {horizon}...")

        # Create output directory for this horizon
        horizon_dir = os.path.join(output_dir, f'horizon_{horizon}')
        os.makedirs(horizon_dir, exist_ok=True)

        # Get the model for this horizon
        model = ensemble.models[horizon]

        # Compute SHAP values
        shap_values = self._compute_shap_values(model, X, horizon)
        self.shap_values[horizon] = shap_values

        # Generate bar plot
        print(f"  Generating bar plot...")
        bar_path = os.path.join(horizon_dir, 'feature_importance_bar.png')
        self._generate_bar_plot(shap_values, bar_path, horizon)

        # Generate beeswarm plot
        print(f"  Generating beeswarm plot...")
        beeswarm_path = os.path.join(horizon_dir, 'feature_importance_beeswarm.png')
        self._generate_beeswarm_plot(shap_values, beeswarm_path, horizon)

        # Generate dependence plots
        print(f"  Generating dependence plots...")
        self._generate_dependence_plots(shap_values, X, horizon_dir, horizon)

        # Compute feature importance
        feature_names = list(X.columns)
        importance_df = self._compute_feature_importance(shap_values, feature_names)
        self.feature_importance[horizon] = importance_df

        # Save importance to CSV
        importance_path = os.path.join(horizon_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)

        return importance_df

    def _generate_cross_horizon_comparison(self, output_dir: str) -> pd.DataFrame:
        """
        Generate cross-horizon feature importance comparison.

        Args:
            output_dir: Directory to save outputs

        Returns:
            DataFrame with cross-horizon comparison
        """
        print("\nGenerating cross-horizon comparison...")

        # Get all features from horizon 1 (should be same across all)
        all_features = self.feature_importance[1]['feature'].tolist()

        # Build comparison DataFrame
        comparison_data = {'feature_name': all_features}

        for h in range(1, 5):
            imp_df = self.feature_importance[h]

            # Create lookup dictionaries
            shap_lookup = dict(zip(imp_df['feature'], imp_df['mean_abs_shap']))
            rank_lookup = dict(zip(imp_df['feature'], imp_df['rank']))

            comparison_data[f'mean_abs_shap_h{h}'] = [
                shap_lookup.get(f, 0) for f in all_features
            ]
            comparison_data[f'rank_h{h}'] = [
                rank_lookup.get(f, len(all_features)) for f in all_features
            ]

        comparison_df = pd.DataFrame(comparison_data)

        # Calculate average rank across horizons
        rank_cols = [f'rank_h{h}' for h in range(1, 5)]
        comparison_df['avg_rank'] = comparison_df[rank_cols].mean(axis=1)

        # Sort by average rank
        comparison_df = comparison_df.sort_values('avg_rank').reset_index(drop=True)

        # Save to CSV
        csv_path = os.path.join(output_dir, 'feature_importance_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"  Saved comparison CSV to {csv_path}")

        return comparison_df

    def _generate_cross_horizon_heatmap(self, comparison_df: pd.DataFrame,
                                        output_dir: str) -> None:
        """
        Generate heatmap visualization of feature importance across horizons.

        Args:
            comparison_df: Cross-horizon comparison DataFrame
            output_dir: Directory to save the plot
        """
        print("  Generating cross-horizon heatmap...")

        # Prepare data for heatmap
        feature_names = comparison_df['feature_name'].tolist()
        shap_cols = [f'mean_abs_shap_h{h}' for h in range(1, 5)]
        heatmap_data = comparison_df[shap_cols].values

        # Normalize by row for better visualization
        row_maxes = heatmap_data.max(axis=1, keepdims=True)
        row_maxes[row_maxes == 0] = 1  # Avoid division by zero
        heatmap_normalized = heatmap_data / row_maxes

        # Calculate figure size
        n_features = len(feature_names)
        fig_width, fig_height = self._calculate_figure_size(n_features, width=12)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create heatmap
        im = ax.imshow(heatmap_normalized, cmap='YlOrRd', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Horizon 1', 'Horizon 2', 'Horizon 3', 'Horizon 4'])
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names, fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Normalized Importance (within feature)', fontsize=10)

        # Add title
        ax.set_title('Feature Importance Across Horizons\n(Row-normalized by feature)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, 'cross_horizon_heatmap.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        print(f"  Saved heatmap to {heatmap_path}")

    def _generate_feature_recommendations(self, comparison_df: pd.DataFrame,
                                          output_dir: str) -> None:
        """
        Generate feature recommendations report.

        Args:
            comparison_df: Cross-horizon comparison DataFrame
            output_dir: Directory to save the report
        """
        print("\nGenerating feature recommendations...")

        report_lines = [
            "=" * 80,
            "FEATURE RECOMMENDATIONS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]

        n_features = len(comparison_df)

        # Find top feature's SHAP value for threshold calculation
        shap_cols = [f'mean_abs_shap_h{h}' for h in range(1, 5)]
        max_shap_per_horizon = comparison_df[shap_cols].max()
        overall_max_shap = max_shap_per_horizon.max()
        threshold = overall_max_shap * 0.01  # 1% of top feature

        # =====================================================================
        # Section 1: Low importance features (candidates for removal)
        # =====================================================================
        report_lines.append("-" * 80)
        report_lines.append("SECTION 1: LOW IMPORTANCE FEATURES (Candidates for Removal)")
        report_lines.append("-" * 80)
        report_lines.append(f"Threshold: mean |SHAP| < 1% of top feature ({threshold:.4f})")
        report_lines.append("")

        low_importance_features = []
        for _, row in comparison_df.iterrows():
            feature = row['feature_name']
            shap_values = [row[f'mean_abs_shap_h{h}'] for h in range(1, 5)]
            max_shap = max(shap_values)

            if max_shap < threshold:
                low_importance_features.append({
                    'feature': feature,
                    'max_shap': max_shap,
                    'avg_rank': row['avg_rank']
                })

        if low_importance_features:
            report_lines.append(f"Found {len(low_importance_features)} features with low importance across ALL horizons:")
            report_lines.append("")
            for item in sorted(low_importance_features, key=lambda x: x['max_shap']):
                report_lines.append(f"  - {item['feature']}")
                report_lines.append(f"      Max |SHAP| across horizons: {item['max_shap']:.6f}")
                report_lines.append(f"      Average rank: {item['avg_rank']:.1f}")
        else:
            report_lines.append("No features found with consistently low importance.")

        report_lines.append("")

        # =====================================================================
        # Section 2: Inconsistent features (candidates for per-horizon sets)
        # =====================================================================
        report_lines.append("-" * 80)
        report_lines.append("SECTION 2: INCONSISTENT FEATURES (Candidates for Per-Horizon Feature Sets)")
        report_lines.append("-" * 80)
        report_lines.append("Features that are top-10 in some horizons but bottom-50% in others:")
        report_lines.append("")

        inconsistent_features = []
        bottom_50_threshold = n_features // 2

        for _, row in comparison_df.iterrows():
            feature = row['feature_name']
            ranks = {h: row[f'rank_h{h}'] for h in range(1, 5)}

            is_top_10_somewhere = any(r <= 10 for r in ranks.values())
            is_bottom_50_somewhere = any(r > bottom_50_threshold for r in ranks.values())

            if is_top_10_somewhere and is_bottom_50_somewhere:
                inconsistent_features.append({
                    'feature': feature,
                    'ranks': ranks,
                    'rank_range': max(ranks.values()) - min(ranks.values())
                })

        if inconsistent_features:
            report_lines.append(f"Found {len(inconsistent_features)} features with inconsistent importance:")
            report_lines.append("")
            for item in sorted(inconsistent_features, key=lambda x: -x['rank_range']):
                report_lines.append(f"  - {item['feature']}")
                rank_str = ", ".join([f"H{h}: {item['ranks'][h]}" for h in range(1, 5)])
                report_lines.append(f"      Ranks: [{rank_str}]")
                report_lines.append(f"      Rank range: {item['rank_range']}")
        else:
            report_lines.append("No features found with highly inconsistent importance.")

        report_lines.append("")

        # =====================================================================
        # Section 3: Top features per horizon (flag for dependence inspection)
        # =====================================================================
        report_lines.append("-" * 80)
        report_lines.append("SECTION 3: TOP 5 FEATURES PER HORIZON (Inspect Dependence Plots)")
        report_lines.append("-" * 80)
        report_lines.append("These features should be visually inspected for plateauing at high values:")
        report_lines.append("")

        for h in range(1, 5):
            report_lines.append(f"Horizon {h}:")
            imp_df = self.feature_importance[h].head(5)
            for _, row in imp_df.iterrows():
                report_lines.append(f"  {row['rank']}. {row['feature']} (mean |SHAP|: {row['mean_abs_shap']:.4f})")
                report_lines.append(f"      -> Check: outputs/shap_analysis/horizon_{h}/dependence_plots/{row['feature']}.png")
            report_lines.append("")

        # =====================================================================
        # Section 4: Summary statistics
        # =====================================================================
        report_lines.append("-" * 80)
        report_lines.append("SECTION 4: SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total features analyzed: {n_features}")
        report_lines.append(f"Low importance features (< 1% threshold): {len(low_importance_features)}")
        report_lines.append(f"Inconsistent features: {len(inconsistent_features)}")
        report_lines.append("")

        # Top 10 most stable features (lowest rank variance)
        rank_cols = [f'rank_h{h}' for h in range(1, 5)]
        comparison_df['rank_std'] = comparison_df[rank_cols].std(axis=1)
        stable_features = comparison_df.nsmallest(10, 'rank_std')

        report_lines.append("Top 10 most stable features (consistent importance across horizons):")
        for _, row in stable_features.iterrows():
            report_lines.append(f"  - {row['feature_name']} (rank std: {row['rank_std']:.2f}, avg rank: {row['avg_rank']:.1f})")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Write report
        report_path = os.path.join(output_dir, 'feature_recommendations.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  Saved recommendations to {report_path}")

    def run_full_analysis(self, ensemble, X_train: pd.DataFrame,
                         output_dir: str) -> Dict:
        """
        Run complete SHAP analysis for all horizons.

        Args:
            ensemble: DirectForecastEnsemble instance (must be trained)
            X_train: Training features DataFrame (used for SHAP analysis)
            output_dir: Directory to save all outputs

        Returns:
            Dictionary with analysis results
        """
        print("=" * 60)
        print("SHAP ANALYSIS FOR FLU FORECASTING MODEL")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of samples: {X_train.shape[0]}")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Analyze each horizon
        for horizon in range(1, 5):
            self._analyze_horizon(ensemble, X_train, horizon, output_dir)

        # Generate cross-horizon comparison
        comparison_df = self._generate_cross_horizon_comparison(output_dir)

        # Generate cross-horizon heatmap
        self._generate_cross_horizon_heatmap(comparison_df, output_dir)

        # Generate feature recommendations
        self._generate_feature_recommendations(comparison_df, output_dir)

        print("\n" + "=" * 60)
        print("SHAP ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"All outputs saved to: {output_dir}")
        print("=" * 60)

        return {
            'shap_values': self.shap_values,
            'feature_importance': self.feature_importance,
            'comparison': comparison_df,
            'output_dir': output_dir
        }


def main():
    """
    Standalone runner for SHAP analysis.

    Loads data, trains the DirectForecastEnsemble, and runs full SHAP analysis.
    """
    from src.data_loader import FluDataLoader
    from src.feature_engineering import FeatureEngineer
    from src.direct_forecast import DirectForecastEnsemble
    from src import config
    import json

    print("=" * 60)
    print("SHAP ANALYSIS - STANDALONE RUNNER")
    print("=" * 60)

    # Output directory
    output_dir = "outputs/shap_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("\n1. Loading data...")
    loader = FluDataLoader()
    data = loader.load_and_preprocess(config.DEFAULT_CUTOFF_DATE)
    print(f"   Loaded {len(data)} records")

    # Step 2: Engineer features with v2 feature set
    print("\n2. Engineering features...")
    feature_version = getattr(config, 'FEATURE_VERSION', 'v2')
    engineer = FeatureEngineer(feature_version=feature_version)
    features_df = engineer.create_all_features(data)
    print(f"   Feature version: {feature_version}")
    print(f"   Created {len(engineer.get_feature_columns(features_df))} features")

    # Step 3: Load model parameters and train the DirectForecastEnsemble
    print("\n3. Training DirectForecastEnsemble...")

    # Try to load V3 optimized params, fall back to V2
    v3_params_file = "outputs/performance_tracking/xgboost_params_v3.json"
    if os.path.exists(v3_params_file):
        with open(v3_params_file, 'r') as f:
            model_params = json.load(f)
        print("   Using V3 parameters from Optuna optimization")
    else:
        model_params = config.XGBOOST_PARAMS.copy()
        print("   Using default parameters from config")

    # Remove early_stopping_rounds to avoid issues
    model_params_clean = model_params.copy()
    if 'early_stopping_rounds' in model_params_clean:
        del model_params_clean['early_stopping_rounds']

    ensemble = DirectForecastEnsemble(
        forecast_horizon=4,
        model_params=model_params_clean,
        target_mode=config.TARGET_MODE
    )
    training_results = ensemble.train(features_df, validation_split=0.2)

    print("\n   Training Results:")
    for horizon, metrics in training_results['horizons'].items():
        print(f"   Horizon {horizon}: Train MAE = {metrics.get('train_mae', 'N/A'):.2f}, "
              f"Val MAE = {metrics.get('val_mae', 'N/A'):.2f}")

    # Step 4: Prepare feature matrix for SHAP analysis
    print("\n4. Preparing feature matrix for SHAP analysis...")

    # Get feature columns (exclude non-feature columns)
    feature_cols = engineer.get_feature_columns(features_df)

    # We need to prepare features the same way the model does
    # Use the first horizon model to prepare features
    X_full, _ = ensemble.models[1].prepare_features(features_df)

    # Use a sample if the dataset is very large (SHAP can be slow)
    max_samples = 5000
    if len(X_full) > max_samples:
        print(f"   Sampling {max_samples} records for SHAP analysis (from {len(X_full)} total)")
        sample_idx = np.random.choice(len(X_full), max_samples, replace=False)
        X_train = X_full.iloc[sample_idx].reset_index(drop=True)
    else:
        X_train = X_full

    print(f"   Feature matrix shape: {X_train.shape}")

    # Step 5: Run SHAP analysis
    print("\n5. Running SHAP analysis...")
    analyzer = SHAPAnalyzer()
    results = analyzer.run_full_analysis(ensemble, X_train, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - horizon_1/ through horizon_4/: Per-horizon analysis")
    print("    - feature_importance_bar.png")
    print("    - feature_importance_beeswarm.png")
    print("    - feature_importance.csv")
    print("    - dependence_plots/*.png")
    print("  - feature_importance_comparison.csv")
    print("  - cross_horizon_heatmap.png")
    print("  - feature_recommendations.txt")


if __name__ == "__main__":
    main()
