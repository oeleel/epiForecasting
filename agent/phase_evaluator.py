"""Phase-aware evaluation for forecasting models.

Assigns epidemic phases to time periods and computes per-phase metrics,
enabling diagnosis of when/where the model performs poorly.

Supports two forecast formats:
    Point forecasts:    requires column 'forecast'
    Quantile forecasts: requires columns 'predicted_q05', 'predicted_q25',
                        'predicted_q50', 'predicted_q75', 'predicted_q95'
                        (the FluSight standard 5-quantile output). When
                        quantiles are present, WIS and 95% coverage are
                        computed in addition to MAPE/MAE/RMSE/bias.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# FluSight standard quantile levels (also matches src.config.QUANTILES default)
QUANTILE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_COLS = [f"predicted_q{int(q * 100):02d}" for q in QUANTILE_LEVELS]


def _has_quantiles(df: pd.DataFrame) -> bool:
    """True if the DataFrame contains the full 5-quantile column set."""
    return all(c in df.columns for c in QUANTILE_COLS)


def _compute_wis_row(actual: float, q: Dict[float, float]) -> float:
    """Weighted Interval Score for a single forecast row.

    Uses the FluSight definition with the standard 5-quantile set
    [0.05, 0.25, 0.50, 0.75, 0.95], which corresponds to two prediction
    intervals (90% and 50%) plus a median:

        WIS = (1/(K + 0.5)) * (
            0.5 * |y - q50|
            + sum over k of (alpha_k / 2) * IS_alpha_k
        )

    where alpha_k is the *miscoverage* level (0.10 for the 90% PI, 0.50
    for the 50% PI) and IS_alpha is the interval score:

        IS_alpha(l, u, y) = (u - l)
            + (2/alpha) * (l - y) * I(y < l)
            + (2/alpha) * (y - u) * I(y > u)

    Args:
        actual: ground truth value
        q: dict mapping quantile level (0.05/0.25/0.5/0.75/0.95) -> predicted value

    Returns:
        WIS for this single observation. Lower is better.
    """
    median = q[0.50]
    # 90% interval (alpha=0.10)
    l90, u90 = q[0.05], q[0.95]
    is90 = (u90 - l90)
    if actual < l90:
        is90 += (2.0 / 0.10) * (l90 - actual)
    elif actual > u90:
        is90 += (2.0 / 0.10) * (actual - u90)

    # 50% interval (alpha=0.50)
    l50, u50 = q[0.25], q[0.75]
    is50 = (u50 - l50)
    if actual < l50:
        is50 += (2.0 / 0.50) * (l50 - actual)
    elif actual > u50:
        is50 += (2.0 / 0.50) * (actual - u50)

    K = 2  # number of intervals
    return (1.0 / (K + 0.5)) * (
        0.5 * abs(actual - median)
        + (0.10 / 2.0) * is90
        + (0.50 / 2.0) * is50
    )


def _compute_wis_vector(actual: np.ndarray, q05, q25, q50, q75, q95) -> np.ndarray:
    """Vectorized WIS for an array of rows. Same formula as _compute_wis_row."""
    actual = np.asarray(actual, dtype=float)
    q05 = np.asarray(q05, dtype=float)
    q25 = np.asarray(q25, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q75 = np.asarray(q75, dtype=float)
    q95 = np.asarray(q95, dtype=float)

    # 90% interval
    is90 = (q95 - q05) \
        + (2.0 / 0.10) * np.maximum(q05 - actual, 0.0) \
        + (2.0 / 0.10) * np.maximum(actual - q95, 0.0)
    # 50% interval
    is50 = (q75 - q25) \
        + (2.0 / 0.50) * np.maximum(q25 - actual, 0.0) \
        + (2.0 / 0.50) * np.maximum(actual - q75, 0.0)

    K = 2
    return (1.0 / (K + 0.5)) * (
        0.5 * np.abs(actual - q50)
        + (0.10 / 2.0) * is90
        + (0.50 / 2.0) * is50
    )


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

        If the forecasts DataFrame includes the FluSight 5-quantile columns
        (predicted_q05, predicted_q25, predicted_q50, predicted_q75,
        predicted_q95), also adds:
            wis            — Weighted Interval Score (lower is better)
            in_pi95        — 1 if actual is in [q05, q95] else 0

        The point-forecast 'forecast' column is required either way. For
        quantile CSVs that name the point column 'predicted', it is
        renamed to 'forecast' on entry.

        Args:
            forecasts: DataFrame with columns: location, forecast_date, horizon,
                       and either 'forecast' or 'predicted' (point), plus
                       optional quantile columns.
            actuals: DataFrame with columns: date, location, value, location_name

        Returns:
            Merged DataFrame with error columns and phase labels.
        """
        forecasts = forecasts.copy()
        actuals = actuals.copy()

        # Allow CSVs that use 'predicted' (the quantile output's point column)
        if "forecast" not in forecasts.columns and "predicted" in forecasts.columns:
            forecasts = forecasts.rename(columns={"predicted": "forecast"})

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

        # Quantile-derived metrics (WIS + 95% PI coverage) when available
        if _has_quantiles(merged):
            merged["wis"] = _compute_wis_vector(
                merged["actual"].values,
                merged["predicted_q05"].values,
                merged["predicted_q25"].values,
                merged["predicted_q50"].values,
                merged["predicted_q75"].values,
                merged["predicted_q95"].values,
            )
            in95 = (
                (merged["actual"] >= merged["predicted_q05"]) &
                (merged["actual"] <= merged["predicted_q95"])
            )
            merged["in_pi95"] = in95.astype(int)

        return merged

    @staticmethod
    def _row_metrics(group: pd.DataFrame) -> Dict[str, Any]:
        """Compute the standard metric set for one slice. Includes WIS +
        coverage_95 only when the slice has those columns."""
        out = {
            "mape": round(float(group["abs_pct_error"].mean()), 1),
            "mae": round(float(group["abs_error"].mean()), 1),
            "bias": round(float(group["error"].mean()), 1),
            "n": int(len(group)),
        }
        if "wis" in group.columns:
            out["wis"] = round(float(group["wis"].mean()), 2)
        if "in_pi95" in group.columns:
            out["coverage_95"] = round(float(group["in_pi95"].mean()), 3)
        return out

    @classmethod
    def evaluate_by_phase(cls, merged_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute metrics broken down by epidemic phase.

        Args:
            merged_df: Output of merge_forecasts_actuals()

        Returns:
            Dict mapping phase name to {"mape", "mae", "bias", "n",
            "wis"?, "coverage_95"?}
        """
        result = {}
        for phase, group in merged_df.groupby("phase"):
            if phase == "off_season":
                continue  # Skip off-season — not meaningful for flu forecasting
            result[phase] = cls._row_metrics(group)
        return result

    @classmethod
    def evaluate_by_horizon(cls, merged_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Compute metrics broken down by forecast horizon.

        Args:
            merged_df: Output of merge_forecasts_actuals()

        Returns:
            Dict mapping horizon (1-4) to {"mape", "mae", "bias", "n",
            "wis"?, "coverage_95"?}
        """
        result = {}
        for horizon, group in merged_df.groupby("horizon"):
            result[int(horizon)] = cls._row_metrics(group)
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
            DataFrame with one row per location, sorted by MAPE descending.
            Also includes wis + coverage_95 columns when quantile data
            is present in merged_df.
        """
        group_cols = ["location"]
        if "location_name" in merged_df.columns:
            group_cols.append("location_name")

        agg_kwargs = {
            "mape": ("abs_pct_error", "mean"),
            "mae": ("abs_error", "mean"),
            "bias": ("error", "mean"),
            "n": ("error", "count"),
        }
        if "wis" in merged_df.columns:
            agg_kwargs["wis"] = ("wis", "mean")
        if "in_pi95" in merged_df.columns:
            agg_kwargs["coverage_95"] = ("in_pi95", "mean")

        result = merged_df.groupby(group_cols).agg(**agg_kwargs).reset_index()
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
            Dict with mape, mae, rmse, bias, n_forecasts. When quantile
            data is present, also includes wis and coverage_95.
        """
        out = {
            "mape": round(float(merged_df["abs_pct_error"].mean()), 1),
            "mae": round(float(merged_df["abs_error"].mean()), 1),
            "rmse": round(float(np.sqrt((merged_df["error"] ** 2).mean())), 1),
            "bias": round(float(merged_df["error"].mean()), 1),
            "n_forecasts": len(merged_df),
        }
        if "wis" in merged_df.columns:
            out["wis"] = round(float(merged_df["wis"].mean()), 2)
        if "in_pi95" in merged_df.columns:
            out["coverage_95"] = round(float(merged_df["in_pi95"].mean()), 3)
        return out
