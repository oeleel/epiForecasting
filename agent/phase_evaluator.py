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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---- phase schemes ----------------------------------------------------------
#
# A phase scheme maps a calendar date to an epidemic phase by (month, day).
# Every metrics dict records the scheme it was scored under ("phase_scheme"),
# because the same phase name means different dates in different schemes
# ("peak" is Dec-Jan in calendar_v1 and Jan 1 - end of Feb in adiga_2026_10).
#
# Windows are inclusive (month, day) ranges that never wrap the year; a phase
# that wraps (calendar_v1 peak) is two windows. `_validate_scheme` checks at
# import time that each scheme covers every day of a leap year exactly once.

PHASE_SCHEME_CALENDAR_V1 = "calendar_v1"
PHASE_SCHEME_ADIGA_2026_10 = "adiga_2026_10"
DEFAULT_PHASE_SCHEME = PHASE_SCHEME_ADIGA_2026_10

# Metrics dicts written before phase schemes existed carry no "phase_scheme"
# key; every one of them was scored with the month map now named calendar_v1.
LEGACY_PHASE_SCHEME = PHASE_SCHEME_CALENDAR_V1

OFF_SEASON = "off_season"

# A. Adiga's 2026-10-15 definitions do not assign October. ASSUMPTION, awaiting
# his confirmation: October is off-season (unscored). Change this one constant
# if he places it elsewhere (for example "growth").
ADIGA_OCTOBER_PHASE = OFF_SEASON

MonthDay = Tuple[int, int]
PhaseWindow = Tuple[MonthDay, MonthDay, str]  # (first day, last day, phase), inclusive

PHASE_SCHEMES: Dict[str, Tuple[PhaseWindow, ...]] = {
    # Month map used through 2026-10-15 (adapted from scripts/optimize.py).
    PHASE_SCHEME_CALENDAR_V1: (
        ((1, 1), (1, 31), "peak"),
        ((2, 1), (4, 30), "decline"),
        ((5, 1), (9, 30), OFF_SEASON),
        ((10, 1), (11, 30), "onset"),
        ((12, 1), (12, 31), "peak"),
    ),
    # A. Adiga, meeting 2026-10-15.
    PHASE_SCHEME_ADIGA_2026_10: (
        ((1, 1), (2, 29), "peak"),  # through the end of February, leap day included
        ((3, 1), (4, 15), "decline"),
        ((4, 16), (5, 31), "plateau"),
        ((6, 1), (9, 30), OFF_SEASON),
        ((10, 1), (10, 31), ADIGA_OCTOBER_PHASE),
        ((11, 1), (12, 31), "growth"),
    ),
}

# Scored phases (everything except off_season) in season order, per scheme.
# Reports, prompts and tables iterate these instead of hard-coding names.
SCHEME_PHASE_ORDER: Dict[str, Tuple[str, ...]] = {
    PHASE_SCHEME_CALENDAR_V1: ("onset", "peak", "decline"),
    PHASE_SCHEME_ADIGA_2026_10: ("growth", "peak", "decline", "plateau"),
}

# One-line human description per scheme, for prompts and report headers.
SCHEME_DESCRIPTIONS: Dict[str, str] = {
    PHASE_SCHEME_CALENDAR_V1: (
        "onset Oct-Nov, peak Dec-Jan, decline Feb-Apr, off_season May-Sep"
    ),
    PHASE_SCHEME_ADIGA_2026_10: (
        "growth Nov 1-Dec 31, peak Jan 1-end of Feb, decline Mar 1-Apr 15, "
        f"plateau Apr 16-May 31, off_season Jun 1-Sep 30, October {ADIGA_OCTOBER_PHASE} "
        "(assumed, awaiting A. Adiga)"
    ),
}

_LEAP_YEAR = 2024  # any leap year: the coverage check must include Feb 29


def resolve_phase_scheme(scheme: Optional[str] = None) -> str:
    """Return a known scheme name; None means DEFAULT_PHASE_SCHEME.

    Raises:
        ValueError: if `scheme` is not a key of PHASE_SCHEMES.
    """
    if scheme is None:
        return DEFAULT_PHASE_SCHEME
    if scheme not in PHASE_SCHEMES:
        raise ValueError(
            f"unknown phase scheme {scheme!r}; known schemes are {sorted(PHASE_SCHEMES)}"
        )
    return scheme


def metrics_phase_scheme(metrics: Optional[Dict[str, Any]]) -> str:
    """The scheme a metrics dict was scored under; a missing key means LEGACY_PHASE_SCHEME."""
    if not isinstance(metrics, dict):
        return LEGACY_PHASE_SCHEME
    return str(metrics.get("phase_scheme") or LEGACY_PHASE_SCHEME)


def phase_order(scheme: Optional[str] = None) -> Tuple[str, ...]:
    """Scored phases of `scheme` in season order (off_season excluded)."""
    return SCHEME_PHASE_ORDER[resolve_phase_scheme(scheme)]


def describe_phase_scheme(scheme: Optional[str] = None) -> str:
    """`<name> (<windows>)`, e.g. for a prompt line or a report header."""
    name = resolve_phase_scheme(scheme)
    return f"{name} ({SCHEME_DESCRIPTIONS[name]})"


def _month_day_key(month: int, day: int) -> int:
    return month * 100 + day


def _validate_scheme(name: str, windows: Tuple[PhaseWindow, ...]) -> None:
    """Every day of a leap year falls in exactly one window; raise naming the gap/overlap."""
    days = pd.date_range(f"{_LEAP_YEAR}-01-01", f"{_LEAP_YEAR}-12-31", freq="D")
    keys = days.month * 100 + days.day
    hits = np.zeros(len(days), dtype=int)
    for (m1, d1), (m2, d2), _phase in windows:
        hits += (keys >= _month_day_key(m1, d1)) & (keys <= _month_day_key(m2, d2))
    bad = days[hits != 1]
    if len(bad):
        raise ValueError(
            f"phase scheme {name!r} must cover every day exactly once; "
            f"first bad day {bad[0].strftime('%m-%d')} is covered {int(hits[hits != 1][0])} times"
        )
    if set(SCHEME_PHASE_ORDER[name]) != {p for _, _, p in windows} - {OFF_SEASON}:
        raise ValueError(
            f"SCHEME_PHASE_ORDER[{name!r}] = {SCHEME_PHASE_ORDER[name]} does not match the "
            f"scheme's scored phases {sorted({p for _, _, p in windows} - {OFF_SEASON})}"
        )


for _name, _windows in PHASE_SCHEMES.items():
    _validate_scheme(_name, _windows)


def assign_phases(dates: Any, scheme: Optional[str] = None) -> np.ndarray:
    """Vectorized phase labels for anything `pd.to_datetime` accepts as a 1-D array.

    Raises:
        ValueError: on an unknown scheme.
    """
    windows = PHASE_SCHEMES[resolve_phase_scheme(scheme)]
    parsed = pd.DatetimeIndex(pd.to_datetime(dates))
    keys = np.asarray(parsed.month * 100 + parsed.day)
    conditions = [
        (keys >= _month_day_key(m1, d1)) & (keys <= _month_day_key(m2, d2))
        for (m1, d1), (m2, d2), _phase in windows
    ]
    labels = [phase for _, _, phase in windows]
    # Coverage is validated at import, so the default is unreachable for real dates.
    return np.select(conditions, labels, default=OFF_SEASON).astype(object)


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

    Phases come from a named calendar scheme (PHASE_SCHEMES), assigned by
    (month, day) of the forecast's target date. DEFAULT_PHASE_SCHEME is
    adiga_2026_10 (A. Adiga, 2026-10-15):
        growth:     Nov 1 - Dec 31
        peak:       Jan 1 - end of February
        decline:    Mar 1 - Apr 15
        plateau:    Apr 16 - May 31
        off_season: Jun 1 - Oct 31 (October assumed, see ADIGA_OCTOBER_PHASE)
    calendar_v1 (the month map used before 2026-10-15) stays selectable:
    onset Oct-Nov, peak Dec-Jan, decline Feb-Apr, off_season May-Sep.
    """

    @classmethod
    def assign_phase(cls, date_str: str, scheme: Optional[str] = None) -> str:
        """Assign the epidemic phase of one date under `scheme` (default DEFAULT_PHASE_SCHEME).

        Args:
            date_str: Date string parseable by pd.to_datetime
            scheme: A PHASE_SCHEMES key, or None for DEFAULT_PHASE_SCHEME

        Returns:
            A phase name of the scheme, or "off_season".

        Raises:
            ValueError: on an unknown scheme.
        """
        return str(assign_phases([date_str], scheme)[0])

    @classmethod
    def merge_forecasts_actuals(
        cls,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        phase_scheme: Optional[str] = None,
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
            phase_scheme: PHASE_SCHEMES key for the phase column (None = default)

        Returns:
            Merged DataFrame with error columns and phase labels.

        Raises:
            ValueError: on an unknown phase scheme, or when nothing merges.
        """
        phase_scheme = resolve_phase_scheme(phase_scheme)
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

        # Assign phase based on forecast (target) date
        merged["phase"] = assign_phases(merged["forecast_date"], phase_scheme)

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
            if phase == OFF_SEASON:
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
