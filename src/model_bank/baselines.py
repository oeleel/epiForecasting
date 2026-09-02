"""Dependency-free baseline families.

These exist for three reasons:
1. They make the whole bank testable and demoable with nothing beyond pandas,
   so the architecture can be exercised on a machine without torch or Nixtla.
2. They are the floor every serious candidate must beat; `select-model` prints
   them alongside the real models.
3. They are the reference implementation of the contract: an in-house model
   author can read `PersistenceModel` (~40 lines) and copy the shape.

Uncertainty is empirical: for each horizon h the model collects in-sample
residuals of its own point rule (y[t+h] - point(t)) over the trailing
`residual_window_weeks`, pooled across series, and adds the residual
quantiles to the point forecast. Simple, honest, and non-degenerate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_bank.contract import (
    HORIZON_COL,
    ID_COL,
    TARGET_COL,
    TIME_COL,
    ForecastModel,
    ModelBankError,
    ParamSpec,
    quantile_column,
    validate_history_frame,
)
from src.model_bank.registry import register

__all__ = ["PersistenceModel", "SeasonalNaiveModel"]

WEEK = pd.Timedelta(7, unit="D")
WEEKS_PER_YEAR = 52
DEFAULT_RESIDUAL_WINDOW_WEEKS = 26
# Guard: fewer pooled residuals than this and the empirical quantiles are noise.
MIN_RESIDUALS_FOR_QUANTILES = 20


class _EmpiricalResidualModel(ForecastModel):
    """Shared machinery: subclass supplies `_point(series_values, horizon)`."""

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "residual_window_weeks": ParamSpec(
                "residual_window_weeks", "int", low=8, high=156,
                description="Trailing weeks of residuals used for the empirical quantiles",
            ),
        }

    @classmethod
    def default_params(cls) -> dict[str, int]:
        return {"residual_window_weeks": DEFAULT_RESIDUAL_WINDOW_WEEKS}

    def _point(self, values: np.ndarray, horizon: int) -> float:
        raise NotImplementedError

    def fit(self, history: pd.DataFrame) -> None:
        hist = validate_history_frame(history)
        window = int(self.params["residual_window_weeks"])
        residuals: dict[int, list[float]] = {h: [] for h in range(1, self.horizon + 1)}
        for _, series in hist.groupby(ID_COL, sort=False):
            values = series[TARGET_COL].to_numpy(dtype=float)
            start = max(0, len(values) - window - self.horizon)
            for t in range(start, len(values)):
                for h in range(1, self.horizon + 1):
                    if t + h >= len(values):
                        break
                    residuals[h].append(values[t + h] - self._point(values[: t + 1], h))
        self._residual_quantiles: dict[int, np.ndarray] = {}
        for h, res in residuals.items():
            if len(res) < MIN_RESIDUALS_FOR_QUANTILES:
                raise ModelBankError(
                    f"{self.family}: only {len(res)} residuals for horizon {h}; "
                    f"need >= {MIN_RESIDUALS_FOR_QUANTILES} (more history or a longer window)"
                )
            self._residual_quantiles[h] = np.quantile(np.asarray(res), self.quantile_levels)
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        hist = validate_history_frame(history)
        rows: list[dict[str, object]] = []
        for uid, series in hist.groupby(ID_COL, sort=False):
            values = series[TARGET_COL].to_numpy(dtype=float)
            last_date = series[TIME_COL].iloc[-1]
            for h in range(1, self.horizon + 1):
                point = self._point(values, h)
                q = np.maximum(point + self._residual_quantiles[h], 0.0)
                row: dict[str, object] = {
                    ID_COL: uid, TIME_COL: last_date + h * WEEK, HORIZON_COL: h,
                }
                for level, value in zip(self.quantile_levels, q, strict=True):
                    row[quantile_column(level)] = float(value)
                rows.append(row)
        return pd.DataFrame(rows)


@register
class PersistenceModel(_EmpiricalResidualModel):
    family = "persistence"
    description = "Last observed value carried forward; empirical residual quantiles"

    def _point(self, values: np.ndarray, horizon: int) -> float:
        return float(values[-1])


@register
class SeasonalNaiveModel(_EmpiricalResidualModel):
    family = "seasonal_naive"
    description = "Same week last year (52-week lag), falling back to persistence"

    def _point(self, values: np.ndarray, horizon: int) -> float:
        lag = WEEKS_PER_YEAR - horizon
        if len(values) > lag:
            return float(values[-1 - lag])
        return float(values[-1])
