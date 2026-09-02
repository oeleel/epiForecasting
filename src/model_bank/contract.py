"""The model-bank contract: what any forecasting model must look like to plug in.

Why this exists
---------------
The agent loop (Milestones 1-3) was hardwired to one model: the in-repo XGBoost
`QuantileDirectForecastEnsemble`. TS-Agent's Stage 1 (model selection over a
bank of candidates) needs many models behind one interface, and the lab we work
with has its own in-house models (LSTM, GNN, ...) that must drop in without any
per-model plumbing. This module is that single interface.

The contract is deliberately tiny so that wrapping an arbitrary model is a
~30-line job:

    class MyModel(ForecastModel):
        family = "my_lstm"
        description = "In-house LSTM, one network per location"

        @classmethod
        def param_space(cls):
            return {"hidden": ParamSpec("hidden", "int", low=8, high=512)}

        def fit(self, history):  ...          # long frame in
        def predict(self, history):  ...      # long quantile frame out

Data contract (both directions use the Nixtla "long" layout)
------------------------------------------------------------
Input to `fit` / `predict` - one row per (series, week), sorted by (id, time):
    unique_id   str    series id (FIPS code for flu: "06", "US", ...)
    ds          datetime64  week-ending date
    y           float  observed target (weekly hospital admissions)
    <extras>    any additional columns pass through untouched (location_name,
                weekly_rate, ...). Models may use or ignore them.
Rows are strictly <= the cutoff date; the runner never leaks future rows.

Output of `predict` - one row per (series, horizon):
    unique_id   str
    ds          datetime64  target week-ending date (= cutoff + horizon weeks)
    horizon     int    1..H
    q05 ... q95 float  one column per requested quantile level, named by
                       `quantile_column(level)`; must be non-decreasing in
                       the level and non-negative.
`validate_forecast_frame` enforces this, so a mis-shaped in-house model fails
at the boundary with a message naming the problem instead of a NaN WIS later.

Design choices
--------------
- Params are validated against `param_space()` at construction. That is what
  lets Agent 2's generic `adjust_hyperparameter` action tune *any* family
  safely: the LLM only ever sees the names and bounds the model declared.
- Warm-start hooks (`get_state` / `set_state`) are optional. Families that
  support resuming from last week's weights set `supports_warm_start = True`;
  the warm-start research layer (roadmap workstream 4) keys off that flag.
- Quantile crossing is repaired by sorting, with a count returned to the
  caller, rather than raised: neural quantile nets cross slightly in practice
  and a hard failure would reject otherwise-valid in-house models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "ID_COL",
    "TIME_COL",
    "TARGET_COL",
    "HORIZON_COL",
    "ForecastModel",
    "ModelBankError",
    "ParamSpec",
    "ParamKind",
    "quantile_column",
    "quantile_columns",
    "validate_history_frame",
    "validate_forecast_frame",
    "validate_params",
]


# Long-format column names (Nixtla convention, adopted repo-wide for the bank).
ID_COL = "unique_id"
TIME_COL = "ds"
TARGET_COL = "y"
HORIZON_COL = "horizon"

# FluSight standard 5-quantile set; must match agent.phase_evaluator.QUANTILE_LEVELS.
DEFAULT_QUANTILE_LEVELS: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95]

ParamKind = Literal["int", "float", "categorical", "bool"]


class ModelBankError(ValueError):
    """Raised for any contract violation: bad params, bad frames, unknown family."""


def quantile_column(level: float) -> str:
    """Column name for a quantile level: 0.05 -> 'q05', 0.5 -> 'q50', 0.975 -> 'q975'."""
    if not (0.0 < level < 1.0):
        raise ModelBankError(f"quantile level must be in (0, 1), got {level}")
    text = f"{level:.3f}".split(".")[1].rstrip("0")
    if len(text) < 2:
        text = text.ljust(2, "0")
    return f"q{text}"


def quantile_columns(levels: Sequence[float]) -> list[str]:
    """Column names for a sequence of levels, in the given order."""
    return [quantile_column(level) for level in levels]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable hyperparameter a family exposes to the improvement loop.

    Bounds are the guardrails Agent 2 must respect. `choices` is required for
    categorical params; `low`/`high` are required for numeric ones.
    """

    name: str
    kind: ParamKind
    low: float | None = None
    high: float | None = None
    choices: tuple | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("int", "float"):
            if self.low is None or self.high is None:
                raise ModelBankError(
                    f"ParamSpec {self.name!r}: numeric kind {self.kind!r} needs low and high"
                )
            if self.low > self.high:
                raise ModelBankError(
                    f"ParamSpec {self.name!r}: low {self.low} > high {self.high}"
                )
        elif self.kind == "categorical":
            if not self.choices:
                raise ModelBankError(f"ParamSpec {self.name!r}: categorical kind needs choices")
        elif self.kind != "bool":
            raise ModelBankError(f"ParamSpec {self.name!r}: unknown kind {self.kind!r}")

    def validate(self, value: Any) -> Any:
        """Return `value` coerced to the declared kind, or raise ModelBankError."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ModelBankError(f"{self.name} must be a bool, got {value!r}")
            return value
        if self.kind == "categorical":
            if value not in self.choices:  # type: ignore[operator]
                raise ModelBankError(
                    f"{self.name} must be one of {list(self.choices)}, got {value!r}"  # type: ignore[arg-type]
                )
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelBankError(f"{self.name} must be numeric, got {value!r}")
        if not (self.low <= value <= self.high):  # type: ignore[operator]
            raise ModelBankError(
                f"{self.name}={value} outside guardrail [{self.low}, {self.high}]"
            )
        return int(value) if self.kind == "int" else float(value)

    def to_json(self) -> dict[str, Any]:
        """JSON-friendly view used in the action catalog shown to the LLM."""
        out: dict[str, Any] = {"kind": self.kind, "description": self.description}
        if self.kind in ("int", "float"):
            out["low"] = self.low
            out["high"] = self.high
        if self.kind == "categorical":
            out["choices"] = list(self.choices)  # type: ignore[arg-type]
        return out


def validate_params(
    params: dict[str, Any], space: dict[str, ParamSpec], family: str
) -> dict[str, Any]:
    """Validate a params dict against a family's param space.

    Unknown names are rejected: a typo in a config must fail loudly rather
    than silently train the default model.
    """
    unknown = sorted(set(params) - set(space))
    if unknown:
        raise ModelBankError(
            f"{family}: unknown params {unknown}; allowed: {sorted(space)}"
        )
    return {name: space[name].validate(value) for name, value in params.items()}


class ForecastModel(ABC):
    """Base class every bank model implements. See module docstring for the contract."""

    #: Registry key and the value of config["model"]["family"].
    family: ClassVar[str] = ""
    #: One line shown to the LLM and in `select-model` output.
    description: ClassVar[str] = ""
    #: True when get_state/set_state are implemented (roadmap workstream 4).
    supports_warm_start: ClassVar[bool] = False

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        quantile_levels: Sequence[float] | None = None,
        horizon: int = 4,
    ) -> None:
        if not self.family:
            raise ModelBankError(f"{type(self).__name__} must set a non-empty `family`")
        if horizon < 1:
            raise ModelBankError(f"horizon must be >= 1, got {horizon}")
        if quantile_levels is None:
            quantile_levels = DEFAULT_QUANTILE_LEVELS
        levels = list(quantile_levels)
        if sorted(levels) != levels or len(set(levels)) != len(levels):
            raise ModelBankError(f"quantile_levels must be strictly increasing, got {levels}")
        self.quantile_levels: list[float] = levels
        self.horizon: int = int(horizon)
        self.params: dict[str, Any] = {
            **self.default_params(),
            **validate_params(dict(params or {}), self.param_space(), self.family),
        }
        self.is_fitted: bool = False

    # ---- what a family declares ------------------------------------------------

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        """Tunable params with guardrails. Empty = nothing for the LLM to tune."""
        return {}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Starting values for the tunable params (must satisfy param_space)."""
        return {}

    # ---- what a family implements ----------------------------------------------

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> None:
        """Train on a validated long-format history (rows <= cutoff)."""

    @abstractmethod
    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        """Return a long-format quantile frame for horizons 1..self.horizon."""

    # ---- optional warm-start hooks --------------------------------------------

    def get_state(self) -> Any:
        raise NotImplementedError(f"{self.family} does not support warm start")

    def set_state(self, state: Any) -> None:
        raise NotImplementedError(f"{self.family} does not support warm start")

    # ---- convenience -------------------------------------------------------------

    @property
    def quantile_cols(self) -> list[str]:
        return quantile_columns(self.quantile_levels)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(family={self.family!r}, params={self.params})"


def validate_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Check the long-format input frame; return it sorted by (id, time)."""
    missing = [c for c in (ID_COL, TIME_COL, TARGET_COL) if c not in history.columns]
    if missing:
        raise ModelBankError(
            f"history frame missing columns {missing}; has {list(history.columns)}"
        )
    if history.empty:
        raise ModelBankError("history frame is empty")
    if history[TARGET_COL].isna().any():
        n_bad = int(history[TARGET_COL].isna().sum())
        raise ModelBankError(f"history frame has {n_bad} NaN values in {TARGET_COL!r}")
    out = history.copy()
    out[ID_COL] = out[ID_COL].astype(str)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    out[TARGET_COL] = out[TARGET_COL].astype(float)
    dup = out.duplicated(subset=[ID_COL, TIME_COL])
    if dup.any():
        raise ModelBankError(f"history frame has {int(dup.sum())} duplicate (unique_id, ds) rows")
    return out.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)


def validate_forecast_frame(
    forecast: pd.DataFrame,
    quantile_levels: Sequence[float],
    horizon: int,
    expected_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Check a model's prediction frame against the contract.

    Returns (clean_frame, n_rows_with_quantile_crossing). Crossing rows are
    repaired by sorting the quantiles left-to-right; every other violation
    raises ModelBankError so a broken in-house model is caught at the seam.
    """
    qcols = quantile_columns(quantile_levels)
    required = [ID_COL, TIME_COL, HORIZON_COL, *qcols]
    missing = [c for c in required if c not in forecast.columns]
    if missing:
        raise ModelBankError(
            f"forecast frame missing columns {missing}; has {list(forecast.columns)}"
        )
    if forecast.empty:
        raise ModelBankError("forecast frame is empty")

    out = forecast.copy()
    out[ID_COL] = out[ID_COL].astype(str)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    out[HORIZON_COL] = out[HORIZON_COL].astype(int)

    bad_h = sorted(set(out[HORIZON_COL]) - set(range(1, horizon + 1)))
    if bad_h:
        raise ModelBankError(f"forecast frame has horizons {bad_h} outside 1..{horizon}")

    q = out[qcols].to_numpy(dtype=float)
    if not np.isfinite(q).all():
        n_bad = int((~np.isfinite(q)).any(axis=1).sum())
        raise ModelBankError(f"forecast frame has {n_bad} rows with NaN/inf quantiles")
    if (q < 0).any():
        n_bad = int((q < 0).any(axis=1).sum())
        raise ModelBankError(
            f"forecast frame has {n_bad} rows with negative quantiles (admissions are counts)"
        )
    crossing = (np.diff(q, axis=1) < 0).any(axis=1)
    n_crossing = int(crossing.sum())
    if n_crossing:
        out.loc[:, qcols] = np.sort(q, axis=1)

    dup = out.duplicated(subset=[ID_COL, HORIZON_COL])
    if dup.any():
        raise ModelBankError(
            f"forecast frame has {int(dup.sum())} duplicate (unique_id, horizon) rows"
        )
    if expected_ids is not None:
        missing_ids = sorted(set(map(str, expected_ids)) - set(out[ID_COL]))
        if missing_ids:
            raise ModelBankError(f"forecast frame missing series {missing_ids}")

    return out.sort_values([ID_COL, HORIZON_COL]).reset_index(drop=True), n_crossing
