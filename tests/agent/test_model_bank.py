"""Unit tests for src.model_bank: contract, registry, data bridge, baselines, runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.phase_evaluator import PhaseEvaluator
from src import config as repo_config
from src.model_bank import (
    ForecastModel,
    ModelBankError,
    ParamSpec,
    from_long,
    list_families,
    quantile_column,
    resolve_family,
    to_forecast_csv,
    to_long,
    validate_forecast_frame,
    validate_params,
)
from src.model_bank.legacy import XGBoostDirectModel, ensemble_forecast_to_long
from src.model_bank.runner import fit_predict, load_history

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
N_WEEKS = 160
SERIES = ["06", "48", "US"]


# ---- factories --------------------------------------------------------------

def _cdc_frame(n_weeks: int = N_WEEKS, series=SERIES, with_nan: bool = False) -> pd.DataFrame:
    """Synthetic CDC-layout frame with a yearly cycle so seasonal naive is non-trivial."""
    dates = pd.date_range("2022-02-05", periods=n_weeks, freq="W-SAT")
    rows = []
    for i, loc in enumerate(series):
        base = 50.0 * (i + 1)
        season = 40.0 * (1 + np.sin(np.arange(n_weeks) * 2 * np.pi / 52))
        values = base + season
        for d, v in zip(dates, values, strict=True):
            rows.append({"date": d, "location": loc, "location_name": f"name_{loc}",
                         "value": float(v), "weekly_rate": float(v) / 100.0})
    df = pd.DataFrame(rows)
    if with_nan:
        df.loc[df.index[5:8], "value"] = np.nan
    return df


def _prediction_frame(ids=SERIES, horizon: int = 4, center: float = 100.0) -> pd.DataFrame:
    rows = []
    for uid in ids:
        for h in range(1, horizon + 1):
            row = {
                "unique_id": uid,
                "ds": pd.Timestamp("2025-01-04") + pd.Timedelta(7 * h, unit="D"),
                "horizon": h,
            }
            for lvl in LEVELS:
                row[quantile_column(lvl)] = center + (lvl - 0.5) * 40.0
            rows.append(row)
    return pd.DataFrame(rows)


class ConstantModel(ForecastModel):
    """Minimal in-house style model used to test the plug-in path end to end."""

    family = "test_constant"
    description = "Predicts a constant with fixed-width quantiles"

    @classmethod
    def param_space(cls) -> dict[str, ParamSpec]:
        return {
            "level": ParamSpec("level", "float", 0.0, 1000.0),
            "wide": ParamSpec("wide", "bool"),
        }

    @classmethod
    def default_params(cls):
        return {"level": 100.0, "wide": False}

    def fit(self, history):
        self.is_fitted = True

    def predict(self, history):
        spread = 40.0 if self.params["wide"] else 10.0
        rows = []
        for uid, s in history.groupby("unique_id"):
            last = s["ds"].max()
            for h in range(1, self.horizon + 1):
                row = {"unique_id": uid, "ds": last + pd.Timedelta(7 * h, unit="D"), "horizon": h}
                for lvl in self.quantile_levels:
                    row[quantile_column(lvl)] = self.params["level"] + (lvl - 0.5) * spread
                rows.append(row)
        return pd.DataFrame(rows)


# ---- contract ---------------------------------------------------------------

def test_quantile_column_names_match_flusight_convention():
    assert [quantile_column(q) for q in LEVELS] == ["q05", "q25", "q50", "q75", "q95"]
    assert quantile_column(0.975) == "q975"


def test_quantile_column_rejects_out_of_range():
    with pytest.raises(ModelBankError):
        quantile_column(1.0)


def test_param_spec_numeric_validate_coerces_and_bounds():
    spec = ParamSpec("depth", "int", 2, 10)
    assert spec.validate(5.0) == 5 and isinstance(spec.validate(5.0), int)
    with pytest.raises(ModelBankError):
        spec.validate(11)
    with pytest.raises(ModelBankError):
        spec.validate("5")


def test_param_spec_categorical_and_bool():
    cat = ParamSpec("mode", "categorical", choices=("log", "raw"))
    assert cat.validate("log") == "log"
    with pytest.raises(ModelBankError):
        cat.validate("sqrt")
    flag = ParamSpec("flag", "bool")
    with pytest.raises(ModelBankError):
        flag.validate(1)


def test_param_spec_rejects_bad_construction():
    with pytest.raises(ModelBankError):
        ParamSpec("x", "int")  # no bounds
    with pytest.raises(ModelBankError):
        ParamSpec("x", "categorical")  # no choices


def test_validate_params_rejects_unknown_names():
    with pytest.raises(ModelBankError) as e:
        validate_params({"nope": 1}, ConstantModel.param_space(), "test_constant")
    assert "unknown params" in str(e.value)


def test_model_constructor_merges_defaults_and_validates():
    m = ConstantModel(params={"level": 5})
    assert m.params == {"level": 5.0, "wide": False}
    with pytest.raises(ModelBankError):
        ConstantModel(params={"level": -1})
    with pytest.raises(ModelBankError):
        ConstantModel(quantile_levels=[0.5, 0.25])


def test_validate_forecast_frame_repairs_crossing_and_counts():
    df = _prediction_frame()
    df.loc[0, "q05"], df.loc[0, "q25"] = 120.0, 90.0  # crossing on row 0
    clean, n = validate_forecast_frame(df, LEVELS, 4)
    assert n == 1
    row = clean.iloc[0][["q05", "q25", "q50", "q75", "q95"]].to_numpy()
    assert (np.diff(row) >= 0).all()


def test_validate_forecast_frame_rejects_nan_negative_missing_series():
    df = _prediction_frame()
    with pytest.raises(ModelBankError):
        validate_forecast_frame(df.assign(q50=np.nan), LEVELS, 4)
    with pytest.raises(ModelBankError):
        validate_forecast_frame(df.assign(q05=-1.0), LEVELS, 4)
    with pytest.raises(ModelBankError):
        validate_forecast_frame(df, LEVELS, 4, expected_ids=["06", "48", "US", "12"])
    with pytest.raises(ModelBankError):
        validate_forecast_frame(df, LEVELS, 3)  # horizon 4 present


# ---- registry ---------------------------------------------------------------

def test_registry_lists_baselines_and_legacy_families():
    names = {f.family for f in list_families() if f.available}
    assert {"persistence", "seasonal_naive", "xgboost_direct", "nn_quantile"} <= names


def test_registry_unknown_family_message_names_dotted_path():
    with pytest.raises(ModelBankError) as e:
        resolve_family("no_such_family")
    assert "pkg" in str(e.value) or "dotted" in str(e.value)


def test_registry_resolves_dotted_path_to_in_house_model():
    cls = resolve_family("tests.agent.test_model_bank:ConstantModel")
    assert cls is ConstantModel


def test_registry_dotted_path_rejects_non_model():
    with pytest.raises(ModelBankError):
        resolve_family("tests.agent.test_model_bank:LEVELS")


# ---- data bridge ------------------------------------------------------------

def test_to_long_from_long_round_trip():
    cdc = _cdc_frame()
    long = to_long(cdc)
    assert {"unique_id", "ds", "y"} <= set(long.columns)
    back = from_long(long)
    assert {"date", "location", "value", "location_name"} <= set(back.columns)
    assert len(back) == len(cdc)


def test_to_long_interpolates_nan_by_default_and_raises_on_request():
    cdc = _cdc_frame(with_nan=True)
    long = to_long(cdc)
    assert not long["y"].isna().any()
    with pytest.raises(ModelBankError):
        to_long(cdc, missing="raise")


def test_to_forecast_csv_has_evaluator_columns_and_scores():
    csv, n = to_forecast_csv(_prediction_frame(), "2025-01-04", "test", LEVELS, 4)
    assert n == 0
    for col in ["location", "forecast_date", "horizon", "forecast",
                "predicted_q05", "predicted_q95", "model_family"]:
        assert col in csv.columns
    actuals = pd.DataFrame({
        "date": [d for _ in SERIES for d in pd.date_range("2025-01-11", periods=4, freq="W-SAT")],
        "location": [s for s in SERIES for _ in range(4)],
        "location_name": [f"name_{s}" for s in SERIES for _ in range(4)],
        "value": 100.0,
    })
    merged = PhaseEvaluator.merge_forecasts_actuals(csv, actuals)
    overall = PhaseEvaluator.compute_overall_metrics(merged)
    assert overall["n_forecasts"] == 12
    assert overall["wis"] >= 0


def test_to_forecast_csv_requires_flusight_levels():
    pred = _prediction_frame()
    with pytest.raises(ModelBankError):
        to_forecast_csv(pred, "2025-01-04", "test", [0.05, 0.5, 0.95], 4)


# ---- baselines --------------------------------------------------------------

def test_persistence_predicts_last_value_as_median():
    long = to_long(_cdc_frame())
    m = resolve_family("persistence")()
    m.fit(long)
    pred = m.predict(long)
    last_us = long[long.unique_id == "US"]["y"].iloc[-1]
    got = pred[(pred.unique_id == "US") & (pred.horizon == 1)]["q50"].iloc[0]
    assert abs(got - last_us) < 60.0  # median = last + median residual (small for a smooth series)
    assert len(pred) == len(SERIES) * 4
    assert (pred["q05"] <= pred["q95"]).all()


def test_seasonal_naive_uses_52_week_lag():
    long = to_long(_cdc_frame())
    m = resolve_family("seasonal_naive")()
    m.fit(long)
    pred = m.predict(long)
    us = long[long.unique_id == "US"]["y"].to_numpy()
    expected_point = us[-1 - (52 - 1)]
    got = pred[(pred.unique_id == "US") & (pred.horizon == 1)]["q50"].iloc[0]
    assert abs(got - expected_point) < 60.0


def test_baseline_refuses_too_short_history():
    long = to_long(_cdc_frame(n_weeks=8))
    with pytest.raises(ModelBankError):
        resolve_family("persistence")().fit(long)


def test_baseline_predict_before_fit_raises():
    with pytest.raises(ModelBankError):
        resolve_family("persistence")().predict(to_long(_cdc_frame()))


# ---- legacy wrapper ---------------------------------------------------------

def test_xgboost_direct_defaults_pin_repo_hyperparameters():
    defaults = XGBoostDirectModel.default_params()
    for k in ["max_depth", "learning_rate", "n_estimators", "subsample", "reg_lambda"]:
        assert defaults[k] == repo_config.XGBOOST_PARAMS[k]
    assert defaults["target_mode"] == repo_config.TARGET_MODE


def test_ensemble_forecast_to_long_maps_flusight_columns():
    wide = pd.DataFrame([{
        "location": "06", "forecast_date": "2025-01-11", "forecast_week": 1,
        "predicted_q05": 1.0, "predicted_q25": 2.0, "predicted_q50": 3.0,
        "predicted_q75": 4.0, "predicted_q95": 5.0,
    }])
    long = ensemble_forecast_to_long(wide)
    assert list(long[["q05", "q25", "q50", "q75", "q95"]].iloc[0]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert long["horizon"].iloc[0] == 1


# ---- runner -----------------------------------------------------------------

def test_load_history_slices_window_and_locations():
    cdc = _cdc_frame()
    hist = load_history("2024-06-01", train_start_date="2023-01-01", locations=["US"], cdc=cdc)
    assert hist["unique_id"].unique().tolist() == ["US"]
    assert hist["ds"].min() >= pd.Timestamp("2023-01-01")
    assert hist["ds"].max() <= pd.Timestamp("2024-06-01")
    with pytest.raises(ModelBankError):
        load_history("2022-01-01", train_start_date="2023-01-01", cdc=cdc)


def test_fit_predict_in_house_model_via_dotted_path():
    cdc = _cdc_frame()
    hist = load_history("2024-06-01", cdc=cdc)
    csv, meta = fit_predict(
        "tests.agent.test_model_bank:ConstantModel", {"level": 42.0}, hist, "2024-06-01"
    )
    assert meta.family == "test_constant"
    assert meta.n_series == len(SERIES)
    assert (csv["predicted_q50"] == 42.0).all()
    assert set(csv["horizon"]) == {1, 2, 3, 4}
    assert (csv["model_family"] == "test_constant").all()


def test_fit_predict_refuses_history_past_cutoff():
    cdc = _cdc_frame()
    hist = load_history("2024-06-01", cdc=cdc)
    with pytest.raises(ModelBankError):
        fit_predict("persistence", None, hist, "2024-01-01")


def test_run_pipeline_dispatches_bank_family():
    """A non-legacy family in config routes through run_bank_model, not the XGBoost path.

    Plain tempfile + manual patch (no pytest fixtures) so the no-pytest
    runner in tests/agent/run_all.py can call this too.
    """
    import tempfile

    import src.model_bank.runner as runner
    from src.config import get_default_config
    from src.pipeline import run_pipeline

    synthetic = _cdc_frame()
    original = runner.load_history

    def _patched(cutoff, train_start_date, locations, cdc=None):
        return original(
            cutoff, train_start_date=train_start_date, locations=locations, cdc=synthetic
        )

    runner.load_history = _patched
    try:
        cfg = get_default_config()
        cfg["model"]["family"] = "persistence"
        cfg["data"]["cutoff_date"] = "2024-06-01"
        with tempfile.TemporaryDirectory() as tmp:
            out = run_pipeline(cfg, output_path=Path(tmp) / "f.csv")
            df = pd.read_csv(out)
    finally:
        runner.load_history = original
    assert (df["model_family"] == "persistence").all()
    assert len(df) == len(SERIES) * 4


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
