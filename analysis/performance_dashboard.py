"""
Performance Dashboard for Flu Forecasting Model Improvement Tracking

Reads step JSON files from outputs/performance_tracking/ and generates:
1. Summary report (improvement_summary.md)
2. Visualization plots (plots/)
3. Console summary
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class PerformanceDashboard:
    """Dashboard for tracking model improvement across steps."""

    BASELINE_MAPE = 59.0
    BASELINE_GAP_RATIO = 39.9  # Original train/val gap ratio

    # Step definitions with names and descriptions
    STEP_INFO = {
        1: {"name": "SHAP Analysis", "short": "SHAP", "has_metrics": False},
        2: {"name": "Target Transformation", "short": "Log Transform", "has_metrics": True},
        3: {"name": "Regularization", "short": "Regularization", "has_metrics": True},
        4: {"name": "Feature Engineering", "short": "Feature Eng.", "has_metrics": True},
        5: {"name": "Expanded Validation", "short": "Optuna Opt.", "has_metrics": True},
        6: {"name": "Quantile Regression", "short": "Quantile", "has_metrics": True},
        7: {"name": "Location Clustering", "short": "Clustering", "has_metrics": True},
    }

    def __init__(self, tracking_dir: str = "outputs/performance_tracking"):
        """
        Initialize the dashboard.

        Args:
            tracking_dir: Directory containing step JSON files
        """
        self.tracking_dir = tracking_dir
        self.plots_dir = os.path.join(tracking_dir, "plots")
        self.step_data = {}

    def load_step_data(self) -> Dict[int, Dict]:
        """
        Load all step JSON files.

        Returns:
            Dictionary mapping step number to step data
        """
        step_files = {
            2: "step2_log_transform.json",
            3: "step3_regularization.json",
            4: "step4_feature_engineering.json",
            5: "step5_expanded_validation.json",
            6: "step6_quantile_regression.json",
            7: "step7_location_clustering.json",
        }

        for step_num, filename in step_files.items():
            filepath = os.path.join(self.tracking_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.step_data[step_num] = json.load(f)
            else:
                print(f"Warning: {filename} not found")

        return self.step_data

    def extract_metrics(self) -> pd.DataFrame:
        """
        Extract key metrics from all steps into a DataFrame.

        Returns:
            DataFrame with metrics for each step
        """
        rows = []

        # Add baseline as step 0
        rows.append({
            'step': 0,
            'name': 'Baseline',
            'mape': self.BASELINE_MAPE,
            'mae': None,
            'rmse': None,
            'gap_ratio_h1': self.BASELINE_GAP_RATIO,
            'mape_h1': None,
            'mape_h2': None,
            'mape_h3': None,
            'mape_h4': None,
        })

        for step_num in sorted(self.step_data.keys()):
            data = self.step_data[step_num]
            step_info = self.STEP_INFO.get(step_num, {"name": f"Step {step_num}", "short": f"S{step_num}"})

            # Extract overall metrics
            if step_num == 6:
                # Step 6 has different structure
                overall = data.get('point_prediction_metrics', {}).get('overall', {})
                by_horizon = data.get('point_prediction_metrics', {}).get('by_horizon', {})
            elif step_num == 7:
                # Step 7 uses unified model metrics for comparison
                overall = data.get('unified_model_metrics', {})
                by_horizon = data.get('clustered_model_metrics', {}).get('by_horizon', {})
            else:
                overall = data.get('overall_metrics', {})
                by_horizon = data.get('by_horizon', {})

            # Extract train/val gap ratio
            train_val = data.get('train_val_gap', {})
            gap_ratio_h1 = None
            if '1' in train_val:
                gap_ratio_h1 = train_val['1'].get('gap_ratio')
            elif 1 in train_val:
                gap_ratio_h1 = train_val[1].get('gap_ratio')

            row = {
                'step': step_num,
                'name': step_info['name'],
                'mape': overall.get('mape'),
                'mae': overall.get('mae'),
                'rmse': overall.get('rmse'),
                'gap_ratio_h1': gap_ratio_h1,
                'mape_h1': by_horizon.get('1', by_horizon.get(1, {})).get('mape'),
                'mape_h2': by_horizon.get('2', by_horizon.get(2, {})).get('mape'),
                'mape_h3': by_horizon.get('3', by_horizon.get(3, {})).get('mape'),
                'mape_h4': by_horizon.get('4', by_horizon.get(4, {})).get('mape'),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def calculate_improvements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative and incremental improvements.

        Args:
            df: DataFrame with metrics

        Returns:
            DataFrame with improvement columns added
        """
        df = df.copy()

        # Absolute MAPE reduction in percentage points (primary metric)
        df['mape_reduction_pp'] = self.BASELINE_MAPE - df['mape']

        # Relative improvement from baseline (secondary metric)
        df['relative_improvement'] = (
            (self.BASELINE_MAPE - df['mape']) / self.BASELINE_MAPE * 100
        )

        # Incremental improvement from previous step (relative)
        df['incremental_improvement'] = 0.0
        for i in range(1, len(df)):
            prev_mape = df.iloc[i-1]['mape']
            curr_mape = df.iloc[i]['mape']
            if prev_mape and curr_mape:
                df.loc[df.index[i], 'incremental_improvement'] = (
                    (prev_mape - curr_mape) / prev_mape * 100
                )

        return df

    def generate_markdown_report(self, df: pd.DataFrame) -> str:
        """
        Generate markdown summary report.

        Args:
            df: DataFrame with metrics and improvements

        Returns:
            Markdown string
        """
        final_mape = df.iloc[-1]['mape']
        mape_reduction_pp = df.iloc[-1]['mape_reduction_pp']
        relative_improvement = df.iloc[-1]['relative_improvement']

        lines = [
            "# Model Improvement Summary",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Overview",
            "",
            f"- **Baseline MAPE**: {self.BASELINE_MAPE:.2f}%",
            f"- **Final MAPE**: {final_mape:.2f}%",
            f"- **MAPE Reduction**: {mape_reduction_pp:.1f} percentage points",
            f"- **Relative Improvement**: {relative_improvement:.1f}%",
            "",
            "---",
            "",
            "## Overall Metrics by Step",
            "",
            "| Step | Name | MAPE (%) | MAE | RMSE | MAPE Reduction (pp) | Relative Impr. |",
            "|------|------|----------|-----|------|---------------------|----------------|",
        ]

        for _, row in df.iterrows():
            mape = f"{row['mape']:.2f}" if pd.notna(row['mape']) else "N/A"
            mae = f"{row['mae']:.1f}" if pd.notna(row['mae']) else "N/A"
            rmse = f"{row['rmse']:.1f}" if pd.notna(row['rmse']) else "N/A"
            reduction_pp = f"{row['mape_reduction_pp']:.1f}" if pd.notna(row['mape_reduction_pp']) else "N/A"
            rel_imp = f"{row['relative_improvement']:.1f}%" if pd.notna(row['relative_improvement']) else "N/A"
            lines.append(f"| {int(row['step'])} | {row['name']} | {mape} | {mae} | {rmse} | {reduction_pp} | {rel_imp} |")

        lines.extend([
            "",
            "---",
            "",
            "## Per-Horizon MAPE by Step",
            "",
            "| Step | Name | H1 (%) | H2 (%) | H3 (%) | H4 (%) |",
            "|------|------|--------|--------|--------|--------|",
        ])

        for _, row in df.iterrows():
            if row['step'] == 0:
                lines.append(f"| 0 | Baseline | N/A | N/A | N/A | N/A |")
            else:
                h1 = f"{row['mape_h1']:.1f}" if pd.notna(row['mape_h1']) else "N/A"
                h2 = f"{row['mape_h2']:.1f}" if pd.notna(row['mape_h2']) else "N/A"
                h3 = f"{row['mape_h3']:.1f}" if pd.notna(row['mape_h3']) else "N/A"
                h4 = f"{row['mape_h4']:.1f}" if pd.notna(row['mape_h4']) else "N/A"
                lines.append(f"| {int(row['step'])} | {row['name']} | {h1} | {h2} | {h3} | {h4} |")

        lines.extend([
            "",
            "---",
            "",
            "## Train/Validation Gap Ratio by Step",
            "",
            "| Step | Name | Gap Ratio (H1) | Status |",
            "|------|------|----------------|--------|",
        ])

        for _, row in df.iterrows():
            if pd.notna(row['gap_ratio_h1']):
                gap = row['gap_ratio_h1']
                if gap < 1.5:
                    status = "Excellent"
                elif gap < 2.0:
                    status = "Good"
                elif gap < 5.0:
                    status = "Moderate"
                else:
                    status = "Overfitting"
                lines.append(f"| {int(row['step'])} | {row['name']} | {gap:.2f} | {status} |")
            else:
                lines.append(f"| {int(row['step'])} | {row['name']} | N/A | - |")

        lines.extend([
            "",
            "---",
            "",
            "## Incremental Improvements",
            "",
            "| Step | Name | Incremental Improvement |",
            "|------|------|-------------------------|",
        ])

        for _, row in df.iterrows():
            if row['step'] == 0:
                continue
            inc_imp = row['incremental_improvement']
            if inc_imp > 0:
                status = f"+{inc_imp:.2f}%"
            elif inc_imp < 0:
                status = f"{inc_imp:.2f}% (regression)"
            else:
                status = "0.00%"
            lines.append(f"| {int(row['step'])} | {row['name']} | {status} |")

        # Find best step
        df_with_metrics = df[df['step'] > 0].copy()
        if len(df_with_metrics) > 0:
            best_step_idx = df_with_metrics['incremental_improvement'].idxmax()
            best_step = df_with_metrics.loc[best_step_idx]

            lines.extend([
                "",
                "---",
                "",
                "## Key Findings",
                "",
                f"### Most Impactful Step: **Step {int(best_step['step'])} - {best_step['name']}**",
                f"- Incremental Improvement: **{best_step['incremental_improvement']:.2f}%**",
                "",
                "### Summary",
                f"- Started at **{self.BASELINE_MAPE:.2f}%** MAPE (baseline)",
                f"- Ended at **{final_mape:.2f}%** MAPE",
                f"- **MAPE Reduction: {mape_reduction_pp:.1f} percentage points**",
                f"- Relative Improvement: {relative_improvement:.1f}%",
                "",
            ])

        return "\n".join(lines)

    def generate_plots(self, df: pd.DataFrame) -> List[str]:
        """
        Generate visualization plots.

        Args:
            df: DataFrame with metrics

        Returns:
            List of saved plot paths
        """
        os.makedirs(self.plots_dir, exist_ok=True)
        plot_paths = []

        # Filter to steps with metrics
        df_plot = df[df['mape'].notna()].copy()

        # 1. Overall MAPE line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        steps = df_plot['step'].values
        mapes = df_plot['mape'].values
        names = df_plot['name'].values

        ax.plot(steps, mapes, 'b-o', linewidth=2, markersize=10)
        ax.axhline(y=self.BASELINE_MAPE, color='r', linestyle='--', label=f'Baseline ({self.BASELINE_MAPE}%)')

        # Add step names as annotations
        for i, (step, mape, name) in enumerate(zip(steps, mapes, names)):
            ax.annotate(f'{mape:.1f}%', (step, mape), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('Overall MAPE Across Improvement Steps', fontsize=14, fontweight='bold')
        ax.set_xticks(steps)
        ax.set_xticklabels([f"S{int(s)}\n{n[:10]}" for s, n in zip(steps, names)], fontsize=8, rotation=0)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.plots_dir, 'mape_progression.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(path)

        # 2. Per-horizon MAPE grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 7))

        # Filter steps with horizon data
        df_horizon = df_plot[df_plot['mape_h1'].notna()].copy()
        steps = df_horizon['step'].values
        names = df_horizon['name'].values

        x = np.arange(len(steps))
        width = 0.2

        h1 = df_horizon['mape_h1'].values
        h2 = df_horizon['mape_h2'].values
        h3 = df_horizon['mape_h3'].values
        h4 = df_horizon['mape_h4'].values

        bars1 = ax.bar(x - 1.5*width, h1, width, label='Horizon 1', color='#2ecc71')
        bars2 = ax.bar(x - 0.5*width, h2, width, label='Horizon 2', color='#3498db')
        bars3 = ax.bar(x + 0.5*width, h3, width, label='Horizon 3', color='#9b59b6')
        bars4 = ax.bar(x + 1.5*width, h4, width, label='Horizon 4', color='#e74c3c')

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('Per-Horizon MAPE Across Improvement Steps', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{int(s)}: {n[:12]}" for s, n in zip(steps, names)], fontsize=9, rotation=15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        path = os.path.join(self.plots_dir, 'horizon_mape_comparison.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(path)

        # 3. Train/Val gap ratio line chart
        fig, ax = plt.subplots(figsize=(12, 6))

        df_gap = df_plot[df_plot['gap_ratio_h1'].notna()].copy()
        if len(df_gap) > 0:
            steps = df_gap['step'].values
            gaps = df_gap['gap_ratio_h1'].values
            names = df_gap['name'].values

            ax.plot(steps, gaps, 'g-o', linewidth=2, markersize=10)
            ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Excellent (<1.5)')
            ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Good (<2.0)')

            for step, gap, name in zip(steps, gaps, names):
                ax.annotate(f'{gap:.2f}', (step, gap), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9)

            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Train/Val Gap Ratio', fontsize=12)
            ax.set_title('Train/Validation Gap Ratio (H1) Across Steps', fontsize=14, fontweight='bold')
            ax.set_xticks(steps)
            ax.set_xticklabels([f"S{int(s)}\n{n[:10]}" for s, n in zip(steps, names)], fontsize=8)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            path = os.path.join(self.plots_dir, 'gap_ratio_progression.png')
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(path)

        # 4. Incremental improvement bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        df_inc = df_plot[df_plot['step'] > 0].copy()
        steps = df_inc['step'].values
        names = df_inc['name'].values
        improvements = df_inc['incremental_improvement'].values

        colors = ['green' if imp >= 0 else 'red' for imp in improvements]
        bars = ax.bar(range(len(steps)), improvements, color=colors, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Incremental Improvement (%)', fontsize=12)
        ax.set_title('Incremental MAPE Improvement at Each Step', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"S{int(s)}: {n[:12]}" for s, n in zip(steps, names)], fontsize=9, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{imp:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9)

        path = os.path.join(self.plots_dir, 'incremental_improvements.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(path)

        return plot_paths

    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print concise summary to stdout.

        Args:
            df: DataFrame with metrics
        """
        print("\n" + "=" * 70)
        print("MODEL IMPROVEMENT DASHBOARD")
        print("=" * 70)

        final_row = df.iloc[-1]
        print(f"\nBaseline MAPE:        {self.BASELINE_MAPE:.2f}%")
        print(f"Final MAPE:           {final_row['mape']:.2f}%")
        print(f"MAPE Reduction:       {final_row['mape_reduction_pp']:.1f} percentage points")
        print(f"Relative Improvement: {final_row['relative_improvement']:.1f}%")

        print("\n" + "-" * 70)
        print("MAPE Progression:")
        print("-" * 70)
        print(f"{'Step':<6} {'Name':<25} {'MAPE':>10} {'Improvement':>15}")
        print("-" * 70)

        for _, row in df.iterrows():
            if pd.notna(row['mape']):
                mape = f"{row['mape']:.2f}%"
                if row['step'] == 0:
                    imp = "-"
                else:
                    imp = f"{row['incremental_improvement']:+.2f}%"
                print(f"{int(row['step']):<6} {row['name']:<25} {mape:>10} {imp:>15}")

        # Find best step
        df_with_metrics = df[df['step'] > 0].copy()
        if len(df_with_metrics) > 0:
            best_idx = df_with_metrics['incremental_improvement'].idxmax()
            best = df_with_metrics.loc[best_idx]
            print("\n" + "-" * 70)
            print(f"Most Impactful: Step {int(best['step'])} - {best['name']}")
            print(f"                Contributed {best['incremental_improvement']:.2f}% improvement")

        print("=" * 70)

    def run(self) -> None:
        """Run the full dashboard generation."""
        print("Loading step data...")
        self.load_step_data()

        if not self.step_data:
            print("No step data found!")
            return

        print("Extracting metrics...")
        df = self.extract_metrics()

        print("Calculating improvements...")
        df = self.calculate_improvements(df)

        print("Generating markdown report...")
        report = self.generate_markdown_report(df)
        report_path = os.path.join(self.tracking_dir, "improvement_summary.md")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")

        print("Generating plots...")
        plot_paths = self.generate_plots(df)
        for path in plot_paths:
            print(f"  - {path}")

        # Print summary
        self.print_summary(df)


def main():
    """Run the performance dashboard."""
    dashboard = PerformanceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
