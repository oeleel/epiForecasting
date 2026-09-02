"""Nixtla example families: statsforecast, mlforecast, neuralforecast.

These are *examples*, not the point. The lab we work with has its own
in-house models; Nixtla is here because (a) it validates that the contract
handles three very different model kinds - classical statistical, gradient
boosted with lags, and neural - with one thin wrapper each, and (b) the
advisor wants a cheap, uniform bank of competitors to select against.

Each wrapper is deliberately minimal. Every one:
    - builds the Nixtla object from `self.params`
    - fits on the long frame (Nixtla already uses unique_id / ds / y)
    - asks for prediction intervals at the levels implied by our quantile
      set and maps them to q05..q95 columns
    - lets the runner validate/repair the output

Quantile mapping: our levels [0.05, 0.25, 0.5, 0.75, 0.95] correspond to
central prediction intervals of 90% (q05/q95) and 50% (q25/q75) plus the
median. Nixtla names those columns "<Model>-lo-90", "<Model>-hi-90", ...
which `_intervals_to_quantiles` translates. Only symmetric level sets are
supported here; asymmetric sets raise so nobody silently gets the wrong
column.

All three libraries are optional: importing this module without them raises
ImportError, which the registry reports as an unavailable family rather than
failing the whole bank. Install with documentation/requirements-nixtla.txt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.model_bank.contract import (
    HORIZON_COL,
    ID_COL,
    TIME_COL,
    ForecastModel,
    ModelBankError,
    ParamSpec,
    quantile_column,
    validate_history_frame,
)
from src.model_bank.registry import register

__all__ = [
    "MLForecastLightGBMModel",
    "NeuralForecastNHITSModel",
    "StatsForecastAutoARIMAModel",
    "StatsForecastAutoETSModel",
    "StatsForecastAutoThetaModel",
]

WEEKLY_FREQ = "W-SAT"
WEEKLY_SEASON_LENGTH = 52


def _interval_levels(quantile_levels: Sequence[float]) -> list[int]:
    """[0.05, 0.25, 0.5, 0.75, 0.95] -> [90, 50] (central interval widths, %)."""
    levels: list[int] = []
    for q in quantile_levels:
        if q == 0.5:
            continue
        mirror = round(1.0 - q, 6)
        if mirror not in [round(x, 6) for x in quantile_levels]:
            raise ModelBankError(
                f"Nixtla families need symmetric quantile levels; {q} has no mirror {mirror}"
            )
        if q < 0.5:
            levels.append(int(round((1.0 - 2.0 * q) * 100)))
    return sorted(set(levels), reverse=True)


def _intervals_to_quantiles(
    pred: pd.DataFrame, model_name: str, quantile_levels: Sequence[float], horizon: int
) -> pd.DataFrame:
    """Nixtla wide prediction frame -> contract long quantile frame."""
    if model_name not in pred.columns:
        raise ModelBankError(
            f"Nixtla output lacks point column {model_name!r}: {list(pred.columns)}"
        )
    out = pd.DataFrame({
        ID_COL: pred[ID_COL].astype(str).to_numpy(),
        TIME_COL: pd.to_datetime(pred[TIME_COL]).to_numpy(),
    })
    out[HORIZON_COL] = out.groupby(ID_COL).cumcount() + 1
    for q in quantile_levels:
        if q == 0.5:
            src_col = model_name
        else:
            width = int(round(abs(1.0 - 2.0 * q) * 100))
            side = "lo" if q < 0.5 else "hi"
            src_col = f"{model_name}-{side}-{width}"
        if src_col not in pred.columns:
            raise ModelBankError(f"Nixtla output lacks column {src_col!r}: {list(pred.columns)}")
        out[quantile_column(q)] = pred[src_col].to_numpy(dtype=float).clip(min=0.0)
    return out[out[HORIZON_COL] <= horizon].reset_index(drop=True)


# ---------------------------------------------------------------------------
# statsforecast
# ---------------------------------------------------------------------------

class _StatsForecastModel(ForecastModel):
    """One statsforecast model per series; conformal or model-native intervals."""

    model_name: str = ""

    def _make_model(self) -> Any:
        raise NotImplementedError

    def fit(self, history: pd.DataFrame) -> None:
        from statsforecast import StatsForecast

        hist = validate_history_frame(history)[[ID_COL, TIME_COL, "y"]]
        self._sf = StatsForecast(models=[self._make_model()], freq=WEEKLY_FREQ, n_jobs=1)
        self._sf.fit(hist)
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        levels = _interval_levels(self.quantile_levels)
        pred = self._sf.predict(h=self.horizon, level=levels).reset_index()
        return _intervals_to_quantiles(pred, self.model_name, self.quantile_levels, self.horizon)


@register
class StatsForecastAutoARIMAModel(_StatsForecastModel):
    family = "sf_autoarima"
    description = "statsforecast AutoARIMA per location, seasonal period 52"
    model_name = "AutoARIMA"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {"season_length": ParamSpec("season_length", "int", 1, 52)}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"season_length": WEEKLY_SEASON_LENGTH}

    def _make_model(self) -> Any:
        from statsforecast.models import AutoARIMA

        return AutoARIMA(season_length=self.params["season_length"])


@register
class StatsForecastAutoETSModel(_StatsForecastModel):
    family = "sf_autoets"
    description = "statsforecast AutoETS per location, seasonal period 52"
    model_name = "AutoETS"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {"season_length": ParamSpec("season_length", "int", 1, 52)}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"season_length": WEEKLY_SEASON_LENGTH}

    def _make_model(self) -> Any:
        from statsforecast.models import AutoETS

        return AutoETS(season_length=self.params["season_length"])


@register
class StatsForecastAutoThetaModel(_StatsForecastModel):
    family = "sf_autotheta"
    description = "statsforecast AutoTheta per location, seasonal period 52"
    model_name = "AutoTheta"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {"season_length": ParamSpec("season_length", "int", 1, 52)}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"season_length": WEEKLY_SEASON_LENGTH}

    def _make_model(self) -> Any:
        from statsforecast.models import AutoTheta

        return AutoTheta(season_length=self.params["season_length"])


# ---------------------------------------------------------------------------
# mlforecast
# ---------------------------------------------------------------------------

@register
class MLForecastLightGBMModel(ForecastModel):
    family = "mlf_lightgbm"
    description = "mlforecast LightGBM, global model on lag features, conformal intervals"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "n_estimators": ParamSpec("n_estimators", "int", 50, 3000),
            "learning_rate": ParamSpec("learning_rate", "float", 0.005, 0.5),
            "num_leaves": ParamSpec("num_leaves", "int", 4, 256),
            "max_lag": ParamSpec("max_lag", "int", 4, 52, description="Longest lag feature"),
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_lag": 12}

    def fit(self, history: pd.DataFrame) -> None:
        import lightgbm as lgb
        from mlforecast import MLForecast
        from mlforecast.utils import PredictionIntervals

        hist = validate_history_frame(history)[[ID_COL, TIME_COL, "y"]]
        lags = list(range(1, int(self.params["max_lag"]) + 1))
        if WEEKLY_SEASON_LENGTH not in lags:
            lags.append(WEEKLY_SEASON_LENGTH)
        model = lgb.LGBMRegressor(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            num_leaves=self.params["num_leaves"],
            verbosity=-1,
        )
        self._mlf = MLForecast(models={"LGBM": model}, freq=WEEKLY_FREQ, lags=lags)
        self._mlf.fit(
            hist,
            prediction_intervals=PredictionIntervals(n_windows=4, h=self.horizon),
        )
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        levels = _interval_levels(self.quantile_levels)
        pred = self._mlf.predict(h=self.horizon, level=levels)
        return _intervals_to_quantiles(pred, "LGBM", self.quantile_levels, self.horizon)


# ---------------------------------------------------------------------------
# neuralforecast
# ---------------------------------------------------------------------------

@register
class NeuralForecastNHITSModel(ForecastModel):
    family = "nf_nhits"
    description = "neuralforecast NHITS, global model with multi-quantile loss"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "input_size": ParamSpec("input_size", "int", 4, 104, description="Lookback weeks"),
            "max_steps": ParamSpec("max_steps", "int", 20, 5000),
            "learning_rate": ParamSpec("learning_rate", "float", 1e-5, 1e-1),
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"input_size": 26, "max_steps": 300, "learning_rate": 1e-3}

    def fit(self, history: pd.DataFrame) -> None:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS

        hist = validate_history_frame(history)[[ID_COL, TIME_COL, "y"]]
        levels = _interval_levels(self.quantile_levels)
        model = NHITS(
            h=self.horizon,
            input_size=self.params["input_size"],
            max_steps=self.params["max_steps"],
            learning_rate=self.params["learning_rate"],
            loss=MQLoss(level=levels),
            enable_progress_bar=False,
            logger=False,
        )
        self._nf = NeuralForecast(models=[model], freq=WEEKLY_FREQ)
        self._nf.fit(hist)
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        pred = self._nf.predict().reset_index()
        # MQLoss names the median column "<Model>-median"; alias it to the point name.
        if "NHITS-median" in pred.columns and "NHITS" not in pred.columns:
            pred["NHITS"] = pred["NHITS-median"]
        return _intervals_to_quantiles(pred, "NHITS", self.quantile_levels, self.horizon)
