"""Nixtla example families: statsforecast, mlforecast, neuralforecast.

These are *examples*, not the point. The lab we work with has its own
in-house models; Nixtla is here because (a) it validates that the contract
handles three very different model kinds - classical statistical, gradient
boosted with lags, and neural - with one thin wrapper each, and (b) the
lab wants a cheap, uniform bank of competitors to select against.

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

# Fixed so a config reproduces its forecast; the agent tunes the listed params, not the seed.
SKLEARN_RANDOM_STATE = 0
CONFORMAL_WINDOWS = 4
MAX_LAG_SPEC = ParamSpec("max_lag", "int", 4, 52, description="Longest lag feature")


class _MLForecastModel(ForecastModel):
    """One global regressor on lag features (lags 1..max_lag plus 52); conformal intervals."""

    model_name: str = ""

    def _make_regressor(self) -> Any:
        raise NotImplementedError

    def fit(self, history: pd.DataFrame) -> None:
        from mlforecast import MLForecast
        from mlforecast.utils import PredictionIntervals

        hist = validate_history_frame(history)[[ID_COL, TIME_COL, "y"]]
        lags = list(range(1, int(self.params["max_lag"]) + 1))
        if WEEKLY_SEASON_LENGTH not in lags:
            lags.append(WEEKLY_SEASON_LENGTH)
        self._mlf = MLForecast(
            models={self.model_name: self._make_regressor()}, freq=WEEKLY_FREQ, lags=lags
        )
        self._mlf.fit(
            hist,
            prediction_intervals=PredictionIntervals(n_windows=CONFORMAL_WINDOWS, h=self.horizon),
        )
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        levels = _interval_levels(self.quantile_levels)
        pred = self._mlf.predict(h=self.horizon, level=levels)
        return _intervals_to_quantiles(pred, self.model_name, self.quantile_levels, self.horizon)


@register
class MLForecastLightGBMModel(_MLForecastModel):
    family = "mlf_lightgbm"
    description = "mlforecast LightGBM, global model on lag features, conformal intervals"
    model_name = "LGBM"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "n_estimators": ParamSpec("n_estimators", "int", 50, 3000),
            "learning_rate": ParamSpec("learning_rate", "float", 0.005, 0.5),
            "num_leaves": ParamSpec("num_leaves", "int", 4, 256),
            "max_lag": MAX_LAG_SPEC,
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_lag": 12}

    def _make_regressor(self) -> Any:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            num_leaves=self.params["num_leaves"],
            verbosity=-1,
        )


@register
class MLForecastLinearModel(_MLForecastModel):
    family = "mlf_linear"
    description = "mlforecast linear regression on lag features (ridge when alpha > 0), conformal intervals"
    model_name = "Linear"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "alpha": ParamSpec("alpha", "float", 0.0, 1000.0, description="Ridge penalty; 0 = ordinary least squares"),
            "max_lag": MAX_LAG_SPEC,
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"alpha": 0.0, "max_lag": 12}

    def _make_regressor(self) -> Any:
        from sklearn.linear_model import LinearRegression, Ridge

        # Ridge(alpha=0) solves the same problem with a less stable solver, so use OLS there.
        if self.params["alpha"] == 0:
            return LinearRegression()
        return Ridge(alpha=self.params["alpha"])


@register
class MLForecastRandomForestModel(_MLForecastModel):
    family = "mlf_random_forest"
    description = "mlforecast random forest, global model on lag features, conformal intervals"
    model_name = "RandomForest"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "n_estimators": ParamSpec("n_estimators", "int", 20, 1000),
            "max_depth": ParamSpec("max_depth", "int", 2, 32),
            "min_samples_leaf": ParamSpec("min_samples_leaf", "int", 1, 50),
            "max_lag": MAX_LAG_SPEC,
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2, "max_lag": 12}

    def _make_regressor(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            random_state=SKLEARN_RANDOM_STATE,
        )


# ---------------------------------------------------------------------------
# neuralforecast
# ---------------------------------------------------------------------------

INPUT_SIZE_SPEC = ParamSpec("input_size", "int", 4, 104, description="Lookback weeks")
MAX_STEPS_SPEC = ParamSpec("max_steps", "int", 20, 5000)
NEURAL_LEARNING_RATE_SPEC = ParamSpec("learning_rate", "float", 1e-5, 1e-1)


class _NeuralForecastModel(ForecastModel):
    """One global neuralforecast model trained with a multi-quantile loss."""

    model_name: str = ""

    def _model_class(self) -> Any:
        raise NotImplementedError

    def _architecture_kwargs(self) -> dict[str, Any]:
        """Constructor arguments beyond the shared input_size / max_steps / learning_rate."""
        return {}

    def fit(self, history: pd.DataFrame) -> None:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss

        hist = validate_history_frame(history)[[ID_COL, TIME_COL, "y"]]
        levels = _interval_levels(self.quantile_levels)
        model = self._model_class()(
            h=self.horizon,
            input_size=self.params["input_size"],
            max_steps=self.params["max_steps"],
            learning_rate=self.params["learning_rate"],
            loss=MQLoss(level=levels),
            enable_progress_bar=False,
            logger=False,
            **self._architecture_kwargs(),
        )
        self._nf = NeuralForecast(models=[model], freq=WEEKLY_FREQ)
        self._nf.fit(hist)
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        pred = self._nf.predict().reset_index()
        # MQLoss names the median column "<Model>-median"; alias it to the point name.
        median_col = f"{self.model_name}-median"
        if median_col in pred.columns and self.model_name not in pred.columns:
            pred[self.model_name] = pred[median_col]
        return _intervals_to_quantiles(pred, self.model_name, self.quantile_levels, self.horizon)


@register
class NeuralForecastNHITSModel(_NeuralForecastModel):
    family = "nf_nhits"
    description = "neuralforecast NHITS, global model with multi-quantile loss"
    model_name = "NHITS"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "input_size": INPUT_SIZE_SPEC,
            "max_steps": MAX_STEPS_SPEC,
            "learning_rate": NEURAL_LEARNING_RATE_SPEC,
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"input_size": 26, "max_steps": 300, "learning_rate": 1e-3}

    def _model_class(self) -> Any:
        from neuralforecast.models import NHITS

        return NHITS


@register
class NeuralForecastLSTMModel(_NeuralForecastModel):
    family = "nf_lstm"
    description = "neuralforecast LSTM encoder, global model with multi-quantile loss"
    model_name = "LSTM"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "input_size": INPUT_SIZE_SPEC,
            "max_steps": MAX_STEPS_SPEC,
            "learning_rate": NEURAL_LEARNING_RATE_SPEC,
            "encoder_hidden_size": ParamSpec("encoder_hidden_size", "int", 8, 512),
            "encoder_n_layers": ParamSpec("encoder_n_layers", "int", 1, 4),
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "input_size": 26,
            "max_steps": 300,
            "learning_rate": 1e-3,
            "encoder_hidden_size": 64,
            "encoder_n_layers": 2,
        }

    def _model_class(self) -> Any:
        from neuralforecast.models import LSTM

        return LSTM

    def _architecture_kwargs(self) -> dict[str, Any]:
        return {
            "encoder_hidden_size": self.params["encoder_hidden_size"],
            "encoder_n_layers": self.params["encoder_n_layers"],
        }


@register
class NeuralForecastTransformerModel(_NeuralForecastModel):
    family = "nf_transformer"
    description = "neuralforecast VanillaTransformer, global model with multi-quantile loss"
    model_name = "VanillaTransformer"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "input_size": INPUT_SIZE_SPEC,
            "max_steps": MAX_STEPS_SPEC,
            "learning_rate": NEURAL_LEARNING_RATE_SPEC,
            "hidden_size": ParamSpec("hidden_size", "categorical", choices=(32, 64, 128, 256)),
            "n_head": ParamSpec("n_head", "categorical", choices=(2, 4, 8)),
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "input_size": 26,
            "max_steps": 300,
            "learning_rate": 1e-4,
            "hidden_size": 64,
            "n_head": 4,
        }

    def _model_class(self) -> Any:
        from neuralforecast.models import VanillaTransformer

        return VanillaTransformer

    def _architecture_kwargs(self) -> dict[str, Any]:
        return {"hidden_size": self.params["hidden_size"], "n_head": self.params["n_head"]}
