"""The in-repo ensembles exposed as bank families.

    xgboost_direct   QuantileDirectForecastEnsemble (src/direct_forecast.py)
    nn_quantile      NNQuantileDirectForecastEnsemble (src/nn_model.py, PyTorch)

Why wrap them at all: the selection stage (`python -m agent select-model`)
must compare every candidate through one code path, including the model the
lab already runs. Wrapping the incumbent proves the contract works for a
feature-engineered, multi-horizon, in-house style model, which is exactly
what the lab's own LSTM/GNN look like.

Relationship to src/pipeline.py: the improvement loop keeps using the legacy
code path in `src.pipeline.run_pipeline` for family "xgboost_direct" because
that path honors every Agent 2 action (feature-group toggles, sample
reweighting, floor, target transform). These wrappers mirror its default
behavior (same feature engineering, same ensemble, same floor) and expose the
subset of knobs that make sense for selection. `test_model_bank` pins the
wrapper's defaults to `src.config.XGBOOST_PARAMS` so the two cannot drift
silently.

Both wrappers convert the long frame back to the CDC layout, run the repo's
FeatureEngineer, and convert the ensemble's wide forecast frame to the long
quantile layout - so the model code itself is untouched.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import pandas as pd

from src import config as repo_config
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
from src.model_bank.data_bridge import from_long
from src.model_bank.registry import register

__all__ = ["NNQuantileModel", "XGBoostDirectModel", "ensemble_forecast_to_long"]

# Guardrails shared with FluForecastAdapter.ACTION_CATALOG; kept here so the
# bank's param_space and the adapter agree on bounds (the adapter reads them
# from param_space for non-legacy families and from its catalog for this one).
XGBOOST_PARAM_SPACE: dict[str, ParamSpec] = {
    "max_depth": ParamSpec("max_depth", "int", 2, 10, description="Tree depth"),
    "learning_rate": ParamSpec("learning_rate", "float", 0.005, 0.5),
    "n_estimators": ParamSpec("n_estimators", "int", 50, 3000),
    "subsample": ParamSpec("subsample", "float", 0.3, 1.0),
    "colsample_bytree": ParamSpec("colsample_bytree", "float", 0.3, 1.0),
    "min_child_weight": ParamSpec("min_child_weight", "int", 1, 100),
    "reg_alpha": ParamSpec("reg_alpha", "float", 0.0, 20.0, description="L1"),
    "reg_lambda": ParamSpec("reg_lambda", "float", 0.0, 20.0, description="L2"),
    "gamma": ParamSpec("gamma", "float", 0.0, 10.0, description="Min split loss"),
    "target_mode": ParamSpec(
        "target_mode", "categorical", choices=("log", "raw", "sqrt"),
        description="Target transform",
    ),
    "floor_pct": ParamSpec(
        "floor_pct", "float", 0.0, 0.6,
        description="Post-prediction floor as a fraction of the last observed value",
    ),
}
_XGB_TUNABLE = [k for k in XGBOOST_PARAM_SPACE if k not in ("target_mode", "floor_pct")]
DEFAULT_FLOOR_PCT = 0.30

_FLUSIGHT_CSV_LEVELS = {
    "predicted_q05": 0.05, "predicted_q25": 0.25, "predicted_q50": 0.5,
    "predicted_q75": 0.75, "predicted_q95": 0.95,
}


def ensemble_forecast_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Ensemble `generate_forecasts` output (wide FluSight columns) -> long quantile frame."""
    needed = ["location", "forecast_date", "forecast_week", *_FLUSIGHT_CSV_LEVELS]
    missing = [c for c in needed if c not in wide.columns]
    if missing:
        raise ModelBankError(f"ensemble forecast frame missing columns {missing}")
    out = pd.DataFrame({
        ID_COL: wide["location"].astype(str),
        TIME_COL: pd.to_datetime(wide["forecast_date"]),
        HORIZON_COL: wide["forecast_week"].astype(int),
    })
    for csv_col, level in _FLUSIGHT_CSV_LEVELS.items():
        out[quantile_column(level)] = wide[csv_col].to_numpy(dtype=float)
    return out


class _FeatureEngineeredEnsemble(ForecastModel):
    """Shared fit/predict around the repo's FeatureEngineer + an ensemble object."""

    def _build_ensemble(self) -> Any:
        raise NotImplementedError

    def _train_ensemble(self, ensemble: Any, training: pd.DataFrame) -> None:
        raise NotImplementedError

    def _floor_pct(self) -> float:
        return float(self.params.get("floor_pct", DEFAULT_FLOOR_PCT))

    def fit(self, history: pd.DataFrame) -> None:
        from src.feature_engineering import FeatureEngineer

        hist = validate_history_frame(history)
        if sorted(self.quantile_levels) != sorted(_FLUSIGHT_CSV_LEVELS.values()):
            raise ModelBankError(
                f"{self.family} only produces the FluSight levels "
                f"{sorted(_FLUSIGHT_CSV_LEVELS.values())}, got {self.quantile_levels}"
            )
        cdc = from_long(hist)
        with contextlib.redirect_stdout(io.StringIO()):
            features = FeatureEngineer().create_all_features(cdc)
            self._ensemble = self._build_ensemble()
            self._train_ensemble(self._ensemble, features)
        self.is_fitted = True

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        from src.feature_engineering import FeatureEngineer

        if not self.is_fitted:
            raise ModelBankError(f"{self.family}: call fit() before predict()")
        hist = validate_history_frame(history)
        cutoff = hist[TIME_COL].max().strftime("%Y-%m-%d")
        cdc = from_long(hist)
        with contextlib.redirect_stdout(io.StringIO()):
            features = FeatureEngineer().create_all_features(cdc)
            wide = self._ensemble.generate_forecasts(
                data=features,
                cutoff_date=cutoff,
                locations=None,
                use_floor_constraint=self._floor_pct() > 0.0,
                floor_ratio=self._floor_pct(),
            )
        if wide.empty:
            raise ModelBankError(f"{self.family}: ensemble produced no forecasts at {cutoff}")
        return ensemble_forecast_to_long(wide)


@register
class XGBoostDirectModel(_FeatureEngineeredEnsemble):
    family = "xgboost_direct"
    description = (
        "In-repo XGBoost direct ensemble: 4 horizon models x 5 quantile models on 59 features"
    )

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return dict(XGBOOST_PARAM_SPACE)

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        defaults = {k: repo_config.XGBOOST_PARAMS[k] for k in _XGB_TUNABLE}
        defaults["target_mode"] = repo_config.TARGET_MODE
        defaults["floor_pct"] = DEFAULT_FLOOR_PCT
        return defaults

    def _build_ensemble(self) -> Any:
        from src.direct_forecast import QuantileDirectForecastEnsemble

        xgb_params = dict(repo_config.XGBOOST_PARAMS)
        xgb_params.update({k: self.params[k] for k in _XGB_TUNABLE})
        return QuantileDirectForecastEnsemble(
            forecast_horizon=self.horizon,
            target_mode=self.params["target_mode"],
            quantiles=list(self.quantile_levels),
            xgb_params_override=xgb_params,
        )

    def _train_ensemble(self, ensemble: Any, training: pd.DataFrame) -> None:
        ensemble.train(training)


@register
class NNQuantileModel(_FeatureEngineeredEnsemble):
    family = "nn_quantile"
    description = "In-repo PyTorch quantile MLP direct ensemble (pinball loss, 1 net per horizon)"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "dropout": ParamSpec("dropout", "float", 0.0, 0.7),
            "learning_rate": ParamSpec("learning_rate", "float", 1e-5, 1e-1),
            "max_epochs": ParamSpec("max_epochs", "int", 10, 2000),
            "patience": ParamSpec("patience", "int", 3, 200),
            "batch_size": ParamSpec("batch_size", "int", 16, 4096),
            "target_mode": ParamSpec("target_mode", "categorical", choices=("log", "raw")),
            "floor_pct": ParamSpec("floor_pct", "float", 0.0, 0.6),
        }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "dropout": 0.2, "learning_rate": 1e-3, "max_epochs": 300, "patience": 20,
            "batch_size": 256, "target_mode": repo_config.TARGET_MODE,
            "floor_pct": DEFAULT_FLOOR_PCT,
        }

    def _build_ensemble(self) -> Any:
        from src.nn_model import NNQuantileDirectForecastEnsemble

        return NNQuantileDirectForecastEnsemble(
            forecast_horizon=self.horizon,
            target_mode=self.params["target_mode"],
            quantiles=list(self.quantile_levels),
            dropout=self.params["dropout"],
            learning_rate=self.params["learning_rate"],
            batch_size=self.params["batch_size"],
            max_epochs=self.params["max_epochs"],
            patience=self.params["patience"],
        )

    def _train_ensemble(self, ensemble: Any, training: pd.DataFrame) -> None:
        ensemble.train(training)
