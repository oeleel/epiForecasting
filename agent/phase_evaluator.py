"""Phase-aware evaluation for forecasting models.

Assigns epidemic phases to time periods and computes per-phase metrics,
enabling diagnosis of when/where the model performs poorly.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class PhaseEvaluator:
    """Evaluate forecasting performance by epidemic phase and time step.

    Phases are calendar-based (adapted from scripts/optimize.py:get_season_phase):
        onset:      October-November (weeks 40-48)
        peak:       December-January (weeks 49-52, 1-4)
        decline:    February-April (weeks 5-16)
        off_season: May-September
    """

    PHASE_MAP = {
        10: "onset", 11: "onset",
        12: "peak", 1: "peak",
        2: "decline", 3: "decline", 4: "decline",
        5: "off_season", 6: "off_season", 7: "off_season",
        8: "off_season", 9: "off_season",
    }

    @classmethod
    def assign_phase(cls, date_str: str) -> str:
        """Assign epidemic phase based on calendar month.

        Args:
            date_str: Date string parseable by pd.to_datetime

        Returns:
            Phase name: "onset", "peak", "decline", or "off_season"
        """
        month = pd.to_datetime(date_str).month
        return cls.PHASE_MAP.get(month, "off_season")

    @classmethod
    def merge_forecasts_actuals(
        cls,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge forecasts with actuals and compute error columns.

        Joins on (location, forecast_date == date). Adds columns:
            actual, error (signed), abs_error, pct_error, abs_pct_error, phase

        Args:
            forecasts: DataFrame with columns: location, forecast_date, forecast, horizon
            actuals: DataFrame with columns: date, location, value, location_name

        Returns:
            Merged DataFrame with error columns and phase labels
        """
        forecasts = forecasts.copy()
        actuals = actuals.copy()

        # Normalize date columns
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        actuals["date"] = pd.to_datetime(actuals["date"])

        # Ensure location is string for consistent merge
        forecasts["location"] = forecasts["location"].astype(str).str.zfill(2)
        actuals["location"] = actuals["location"].astype(str).str.zfill(2)

        # Merge: forecast_date matches actual date
        merged = forecasts.merge(
            actuals[["date", "location", "value", "location_name"]],
            left_on=["location", "forecast_date"],
            right_on=["location", "date"],
            how="inner",
        )

        if len(merged) == 0:
            raise ValueError(
                "No matching rows after merging forecasts with actuals. "
                "Check that forecast_date values overlap with actual date values."
            )

        merged.rename(columns={"value": "actual"}, inplace=True)

        # Compute error columns
        merged["error"] = merged["forecast"] - merged["actual"]
        merged["abs_error"] = merged["error"].abs()

        # Percentage errors (guard against division by zero)
        actual_safe = merged["actual"].replace(0, np.nan)
        merged["pct_error"] = (merged["error"] / actual_safe) * 100
        merged["abs_pct_error"] = merged["pct_error"].abs()

        # Assign phase based on forecast date
        merged["phase"] = merged["forecast_date"].apply(
            lambda d: cls.assign_phase(d.isoformat())
        )

        return merged

    @classmethod
    def evaluate_by_phase(cls, merged_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute metrics broken down by epidemic phase.

        Args:
            merged_df: Output of merge_forecasts_actuals()

        Returns:
            Dict mapping phase name to {"mape", "mae", "bias", "n"}
        """
        result = {}
        for phase, group in merged_df.groupby("phase"):
            if phase == "off_season":
                continue  # Skip off-season — not meaningful for flu forecasting
            result[phase] = {
                "mape": round(group["abs_pct_error"].mean(), 1),
                "mae": round(group["abs_error"].mean(), 1),
                "bias": round(group["error"].mean(), 1),
                "n": len(group),
            }
        return result

    @classmethod
    def evaluate_by_horizon(cls, merged_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Compute metrics broken down by forecast horizon.

        Args:
            merged_df: Output of merge_forecasts_actuals()

        Returns:
            Dict mapping horizon (1-4) to {"mape", "mae", "bias", "n"}
        """
        result = {}
        for horizon, group in merged_df.groupby("horizon"):
            result[int(horizon)] = {
                "mape": round(group["abs_pct_error"].mean(), 1),
                "mae": round(group["abs_error"].mean(), 1),
                "bias": round(group["error"].mean(), 1),
                "n": len(group),
            }
        return result

    @classmethod
    def evaluate_by_timestep(cls, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per (cutoff_date, location, horizon) error metrics.

        Args:
            merged_df: Output of merge_forecasts_actuals()

        Returns:
            DataFrame with one row per (cutoff_date, location, horizon)
        """
        group_cols = ["cutoff_date", "location", "location_name", "horizon"]
        available_cols = [c for c in group_cols if c in merged_df.columns]

        return merged_df.groupby(available_cols).agg(
            mae=("abs_error", "mean"),
            mape=("abs_pct_error", "mean"),
            bias=("error", "mean"),
            n=("error", "count"),
        ).reset_index()

    @classmethod
    def evaluate_by_location(
        cls, merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute metrics broken down by location.

        Returns:
            DataFrame with one row per location, sorted by MAPE descending
        """
        group_cols = ["location"]
        if "location_name" in merged_df.columns:
            group_cols.append("location_name")

        result = merged_df.groupby(group_cols).agg(
            mape=("abs_pct_error", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            n=("error", "count"),
        ).reset_index()

        return result.sort_values("mape", ascending=False)

    @classmethod
    def identify_worst_segments(
        cls, merged_df: pd.DataFrame, n: int = 10
    ) -> List[Dict[str, Any]]:
        """Find the N worst individual predictions by absolute error.

        Args:
            merged_df: Output of merge_forecasts_actuals()
            n: Number of worst predictions to return

        Returns:
            List of dicts, each describing one bad prediction
        """
        worst = merged_df.nlargest(n, "abs_error")

        segments = []
        for _, row in worst.iterrows():
            segment = {
                "location": row.get("location_name", row["location"]),
                "fips": str(row["location"]),
                "date": str(row["forecast_date"].date()) if hasattr(row["forecast_date"], "date") else str(row["forecast_date"]),
                "horizon": int(row["horizon"]),
                "predicted": round(float(row["forecast"]), 1),
                "actual": round(float(row["actual"]), 1),
                "error": round(float(row["error"]), 1),
                "pct_error": round(float(row["pct_error"]), 1) if pd.notna(row["pct_error"]) else None,
            }
            if "cutoff_date" in row.index:
                segment["cutoff_date"] = str(row["cutoff_date"])
            segments.append(segment)

        return segments

    @classmethod
    def compute_overall_metrics(cls, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate metrics across all predictions.

        Returns:
            Dict with mape, mae, rmse, bias, n_forecasts
        """
        return {
            "mape": round(float(merged_df["abs_pct_error"].mean()), 1),
            "mae": round(float(merged_df["abs_error"].mean()), 1),
            "rmse": round(float(np.sqrt((merged_df["error"] ** 2).mean())), 1),
            "bias": round(float(merged_df["error"].mean()), 1),
            "n_forecasts": len(merged_df),
        }
