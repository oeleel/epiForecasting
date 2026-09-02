"""Bridge between the CDC FluSight frame and the bank's long format, both ways.

The repo's loader (`src.data_loader.FluDataLoader`) yields the CDC layout:
    date, location, location_name, value, weekly_rate
The bank speaks the Nixtla long layout (`unique_id`, `ds`, `y`, extras). Every
model sees only the long layout, so an in-house model written against Nixtla
conventions plugs in unchanged, and the legacy XGBoost pipeline converts back
with `from_long` because its feature engineering expects CDC column names.

`to_forecast_csv` renders a validated prediction frame in the flat CSV layout
the rest of the repo already consumes (`agent.phase_evaluator`, the
orchestrator, `python -m agent summarize`):
    location, cutoff_date, forecast_date, forecast_week, horizon, forecast,
    predicted, predicted_q05..predicted_q95, reference_date, target_end_date,
    model_family
so WIS / coverage / phase metrics work for any family with zero changes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from src.model_bank.contract import (
    HORIZON_COL,
    ID_COL,
    TARGET_COL,
    TIME_COL,
    ModelBankError,
    quantile_column,
    validate_forecast_frame,
    validate_history_frame,
)

__all__ = [
    "MissingPolicy",
    "CDC_DATE_COL",
    "CDC_LOCATION_COL",
    "CDC_VALUE_COL",
    "FORECAST_CSV_COLUMNS",
    "from_long",
    "to_forecast_csv",
    "to_long",
]

_log = logging.getLogger(__name__)

MissingPolicy = Literal["interpolate", "raise"]

CDC_DATE_COL = "date"
CDC_LOCATION_COL = "location"
CDC_VALUE_COL = "value"

# Point-forecast column the evaluator requires plus the FluSight quantile names.
FORECAST_CSV_COLUMNS: list[str] = [
    "location", "cutoff_date", "forecast_date", "forecast_week", "horizon",
    "forecast", "predicted",
    "predicted_q05", "predicted_q25", "predicted_q50", "predicted_q75", "predicted_q95",
    "reference_date", "target_end_date", "model_family",
]

_CDC_TO_LONG: dict[str, str] = {
    CDC_LOCATION_COL: ID_COL,
    CDC_DATE_COL: TIME_COL,
    CDC_VALUE_COL: TARGET_COL,
}
_LONG_TO_CDC: dict[str, str] = {v: k for k, v in _CDC_TO_LONG.items()}


def to_long(cdc: pd.DataFrame, missing: MissingPolicy = "interpolate") -> pd.DataFrame:
    """CDC frame -> validated long frame. Extra columns pass through.

    The CDC target data has a handful of NaN weeks (off-season 2024 reporting
    gaps for three states) on an otherwise regular weekly grid. Nixtla-style
    models need a gap-free grid, so the default policy fills each series'
    NaNs by linear interpolation (edges: nearest value) and logs how many.
    Pass missing="raise" to refuse instead - the legacy pipeline drops those
    rows after feature engineering, so the two paths differ only there.
    """
    if missing not in ("interpolate", "raise"):
        raise ModelBankError(f"missing must be 'interpolate' or 'raise', got {missing!r}")
    absent = [c for c in _CDC_TO_LONG if c not in cdc.columns]
    if absent:
        raise ModelBankError(f"CDC frame missing columns {absent}; has {list(cdc.columns)}")
    out = cdc.rename(columns=_CDC_TO_LONG).copy()
    out[ID_COL] = out[ID_COL].astype(str).str.zfill(2)
    nan_mask = out[TARGET_COL].isna()
    if nan_mask.any() and missing == "interpolate":
        n_series = int(out.loc[nan_mask, ID_COL].nunique())
        out = out.sort_values([ID_COL, TIME_COL])
        out[TARGET_COL] = (
            out.groupby(ID_COL, sort=False)[TARGET_COL]
            .transform(lambda s: s.interpolate(limit_direction="both"))
        )
        _log.info(
            "to_long: interpolated %d NaN target values across %d series",
            int(nan_mask.sum()), n_series,
        )
    return validate_history_frame(out)


def from_long(history: pd.DataFrame) -> pd.DataFrame:
    """Long frame -> CDC frame (for the legacy feature engineering)."""
    validated = validate_history_frame(history)
    return validated.rename(columns=_LONG_TO_CDC)


def to_forecast_csv(
    prediction: pd.DataFrame,
    cutoff_date: str,
    family: str,
    quantile_levels: Sequence[float],
    horizon: int,
    expected_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Validated prediction frame -> repo-standard forecast CSV frame.

    Returns (csv_frame, n_quantile_crossings_repaired). The evaluator's
    quantile column names are fixed at the FluSight 5 levels; if a family
    was asked for a different set we fail here rather than write a CSV the
    evaluator cannot score.
    """
    clean, n_crossing = validate_forecast_frame(
        prediction, quantile_levels, horizon, expected_ids=expected_ids
    )
    level_to_csv = {0.05: "predicted_q05", 0.25: "predicted_q25", 0.5: "predicted_q50",
                    0.75: "predicted_q75", 0.95: "predicted_q95"}
    missing_levels = [lvl for lvl in level_to_csv if lvl not in quantile_levels]
    if missing_levels:
        raise ModelBankError(
            f"forecast CSV needs quantile levels {sorted(level_to_csv)}; "
            f"family {family!r} was configured with {list(quantile_levels)} "
            f"(missing {missing_levels})"
        )

    out = pd.DataFrame({
        "location": clean[ID_COL],
        "cutoff_date": cutoff_date,
        "forecast_date": clean[TIME_COL].dt.strftime("%Y-%m-%d"),
        "forecast_week": clean[HORIZON_COL],
        "horizon": clean[HORIZON_COL],
    })
    median_col = quantile_column(0.5)
    out["forecast"] = clean[median_col].to_numpy()
    out["predicted"] = clean[median_col].to_numpy()
    for level, csv_col in level_to_csv.items():
        out[csv_col] = clean[quantile_column(level)].to_numpy()
    out["reference_date"] = cutoff_date
    out["target_end_date"] = out["forecast_date"]
    out["model_family"] = family
    return out[FORECAST_CSV_COLUMNS], n_crossing
