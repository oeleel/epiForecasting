"""Unit tests for the pure functions of scripts/experiments/peak_rectification.py.

Covers the parts of the harness that need no data or model: arm expansion,
config-delta application (claim 4: every arm changes exactly one config
path), reward direction (claim 5: WIS lower-is-better), the log -> summary
round trip, and round 2: seed suffixing, phase x horizon scoring, the
across-seed and by-horizon summary tables, `--rescore`, `--merge` and the
data-vintage pin. Nothing here trains a model: the seed tests run
`src.pipeline.run_pipeline` on the cached data up to the ensemble's fit and
stop there, to prove the seed reaches the ensemble the harness trains.

Every test is linear setup -> execute -> verify, with `tempfile` instead of
pytest fixtures so `tests/agent/run_all.py` can run the same functions.

Run with:
    PYTHONPATH=. python tests/agent/test_peak_rectification.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The harness sets these itself on darwin, but only after import; set them
# before xgboost/torch are pulled in by the adapter import below.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from agent.phase_evaluator import DEFAULT_PHASE_SCHEME  # noqa: E402
from scripts.experiments.peak_rectification import (  # noqa: E402
    ARM_BASELINE,
    ARM_LAMBDA,
    ARM_LAMBDA_CALENDAR,
    ARM_OVERSAMPLE,
    ARM_WINDOW,
    BASELINE_CONFIG_ID,
    DATA_CACHE_HASH_KEY,
    DEFAULT_APPROACHING_PEAK_WEEKS,
    DEFAULT_OUT_DIR,
    DEFAULT_SEEDS,
    FORECASTS_DIRNAME,
    LOG_NAME,
    MANIFEST_NAME,
    MIN_TRAIN_WINDOW_WEEKS,
    RESCORED_LOG_NAME,
    RESCORED_SUMMARY_NAME,
    ROUND1_OUT_DIR,
    SUMMARY_MISSING,
    SUMMARY_NAME,
    Arm,
    _data_cache_sha256,
    _ensure_manifest,
    _frozen_data_cache,
    apply_delta,
    arm_lambda,
    arm_lambda_calendar,
    arm_oversample,
    arm_window,
    build_arms,
    log_row,
    merge,
    read_log,
    rescore,
    reward_for,
    score,
    with_seed,
    write_summary,
)
from src.config import get_default_config  # noqa: E402
from src.data_loader import FluDataLoader  # noqa: E402
from src.direct_forecast import QuantileDirectForecastEnsemble  # noqa: E402
from src.pipeline import PhaseOversampledQuantileEnsemble, run_pipeline  # noqa: E402

# A cutoff inside the pinned split; any would do, nothing is fitted.
SEED_TEST_CUTOFF = "2025-12-06"


def _expect_value_error(fn, *needles: str) -> str:
    return _expect_error(ValueError, fn, *needles)


def _expect_error(exc_type: type[Exception], fn, *needles: str) -> str:
    try:
        fn()
    except exc_type as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, f"expected {needle!r} in error: {message}"
        return message
    raise AssertionError(f"expected {exc_type.__name__} mentioning {needles}, nothing raised")


def _metrics(peak_wis: float | None, overall_wis: float = 50.0, peak_bias: float = -10.0) -> dict:
    by_phase = {} if peak_wis is None else {"peak": {"wis": peak_wis, "bias": peak_bias}}
    return {"overall": {"wis": overall_wis}, "by_phase": by_phase, "by_horizon": {}}


def _horizon_metrics(peak_wis: float, overall_by_h: dict[str, float], peak_by_h: dict[str, float]) -> dict:
    metrics = _metrics(peak_wis=peak_wis)
    metrics["by_horizon"] = {h: {"wis": v} for h, v in overall_by_h.items()}
    metrics["by_phase_horizon"] = {"peak": {h: {"wis": v} for h, v in peak_by_h.items()}}
    return metrics


class _HorizonEchoAdapter:
    """Stands in for FluForecastAdapter.compute_metrics: one phase per horizon present.

    by_phase reports "peak" with wis = the mean `predicted_q50` of the rows
    it is given and n = their count, so per-horizon subsets are visible in
    the result. `phase_scheme` mimics the adapter's key.
    """

    exclude_locations: list[str] = []

    def compute_metrics(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> dict:
        block = {"wis": float(forecasts["predicted_q50"].mean()), "bias": 0.0, "n": len(forecasts)}
        by_horizon = {int(h): {"wis": float(g["predicted_q50"].mean())} for h, g in forecasts.groupby("horizon")}
        return {
            "phase_scheme": "test_scheme",
            "overall": {"wis": block["wis"]},
            "by_phase": {"peak": block},
            "by_horizon": by_horizon,
        }


def _forecasts(q50_by_horizon: dict[int, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"location": "01", "horizon": h, "predicted_q50": v} for h, values in q50_by_horizon.items() for v in values]
    )


class _StopBeforeFit(Exception):
    """Raised by the patched ensemble `train`, so run_pipeline never fits a model."""


def _model_params_run_pipeline_builds(config: dict) -> tuple[str, dict]:
    """(ensemble class name, its model_params) as built by run_pipeline for `config`, stopping before the fit."""
    seen: dict = {}

    def stop(self, *args, **kwargs):
        seen.update(cls=type(self).__name__, params=dict(self.model_params))
        raise _StopBeforeFit

    with mock.patch.object(QuantileDirectForecastEnsemble, "train", stop), \
            mock.patch.object(PhaseOversampledQuantileEnsemble, "train", stop), _frozen_data_cache():
        try:
            run_pipeline(config)
        except _StopBeforeFit:
            pass
    return seen["cls"], seen["params"]


def _seeded_config(arm: Arm, seed: int) -> dict:
    config = apply_delta(get_default_config(), with_seed(arm, seed).config_delta)
    config["data"]["cutoff_date"] = SEED_TEST_CUTOFF
    return config


def _write_source(out_dir: Path, seed: int, manifest_overrides: dict | None = None) -> None:
    """A finished one-seed sweep directory: manifest, baseline + lambda_2 rows, their forecasts."""
    manifest = {
        "cutoffs": ["c1"], "n_cutoffs": 1, "seeds": [seed], "phase_scheme": "s", "git_sha": "abc",
        "split": {"train_start_date": "t0", "eval_start_date": "e0", "eval_end_date": "e1", "stride_weeks": 4},
        "exclude_locations": ["US"], "data_cache": {"mtime": "m", "sha256": "f00d"},
        **(manifest_overrides or {}),
    }
    (out_dir / FORECASTS_DIRNAME).mkdir(parents=True)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    base, lam = _metrics(peak_wis=120.0 + seed), _metrics(peak_wis=110.0 + seed)
    for arm, metrics in ((build_arms([])[0], base), (arm_lambda(2.0), lam)):
        seeded = with_seed(arm, seed)
        log_row(out_dir / LOG_NAME, seeded, ["c1"], metrics, reward_for(metrics, base), 1.0)
        _forecasts({1: [1.0]}).to_csv(out_dir / FORECASTS_DIRNAME / f"{seeded.config_id}.csv", index=False)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


# ---- arms -------------------------------------------------------------------


def test_build_arms_always_puts_baseline_first_with_unique_ids():
    arms = build_arms([ARM_LAMBDA, ARM_WINDOW], lambdas=(1.5, 2.0), windows=(12, 26))

    assert arms[0].name == ARM_BASELINE
    assert arms[0].config_id == BASELINE_CONFIG_ID
    assert arms[0].config_delta == {}
    assert [a.config_id for a in arms] == [
        BASELINE_CONFIG_ID, "lambda_1.5", "lambda_2", "window_12", "window_26"
    ]
    assert len({a.config_id for a in arms}) == len(arms)


def test_build_arms_rejects_unknown_arm_name():
    _expect_value_error(lambda: build_arms(["smote"]), "unknown arm", "smote")


def test_arm_factories_validate_their_knob():
    _expect_value_error(lambda: arm_lambda(0.0), "lambda weight")
    _expect_value_error(lambda: arm_lambda(2.0, weeks_before=0), "weeks_before")
    _expect_value_error(lambda: arm_lambda_calendar(-1.0), "calendar lambda")
    _expect_value_error(lambda: arm_window(MIN_TRAIN_WINDOW_WEEKS - 1), "window weeks")
    _expect_value_error(lambda: arm_window(True), "window weeks")
    _expect_value_error(lambda: Arm(name="bogus", config_id="x"), "Arm.name")
    _expect_value_error(lambda: Arm(name=ARM_LAMBDA, config_id="has space"), "Arm.config_id")


def test_every_arm_changes_exactly_one_config_path():
    base = {"data": {"train_window_weeks": None}, "sample_weights": {"by_phase": {}}}
    arms = build_arms(
        [ARM_LAMBDA, ARM_LAMBDA_CALENDAR, ARM_WINDOW],
        lambdas=(3.0,), calendar_lambdas=(2.0,), windows=(52,),
    )
    flat_base = _flatten(base)

    diffs = {}
    for arm in arms:
        flat = _flatten(apply_delta(base, arm.config_delta))
        diffs[arm.config_id] = {k: v for k, v in flat.items() if flat_base.get(k) != v}

    assert diffs[BASELINE_CONFIG_ID] == {}
    assert diffs["lambda_3"] == {
        "sample_weights.approaching_peak.weeks_before": DEFAULT_APPROACHING_PEAK_WEEKS,
        "sample_weights.approaching_peak.weight": 3.0,
    }
    assert diffs["lambda_calendar_2"] == {"sample_weights.by_phase.peak": 2.0}
    assert diffs["window_52"] == {"data.train_window_weeks": 52}


# ---- apply_delta ------------------------------------------------------------


def test_apply_delta_creates_missing_parents_and_leaves_input_untouched():
    base = {"data": {"cutoff_date": "2025-12-06"}}

    out = apply_delta(base, {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}})

    assert out["sample_weights"]["approaching_peak"] == {"weeks_before": 6, "weight": 2.0}
    assert out["data"] == {"cutoff_date": "2025-12-06"}
    assert "sample_weights" not in base


def test_apply_delta_rejects_a_non_mapping_parent():
    _expect_value_error(
        lambda: apply_delta({"data": {"locations": ["06"]}}, {"data.locations.first": "x"}),
        "data.locations.first",
        "not a mapping",
    )


# ---- reward_for -------------------------------------------------------------


def test_reward_for_lower_wis_is_better():
    baseline = _metrics(peak_wis=122.4, overall_wis=60.0)

    better = reward_for(_metrics(peak_wis=100.0, overall_wis=55.0), baseline)
    worse = reward_for(_metrics(peak_wis=130.0), baseline)
    same = reward_for(baseline, baseline)

    assert better["metric"] == "wis" and better["phase"] == "peak"
    assert better["baseline_value"] == 122.4 and better["value"] == 100.0
    assert better["delta"] == -22.4
    assert better["improvement_frac"] == round((122.4 - 100.0) / 122.4, 4)
    assert better["better"] is True
    assert better["guard"] == {"metric": "overall_wis", "baseline_value": 60.0, "value": 55.0}

    assert worse["delta"] == 7.6
    assert worse["improvement_frac"] < 0
    assert worse["better"] is False

    assert same["delta"] == 0.0 and same["improvement_frac"] == 0.0 and same["better"] is False


def test_reward_for_missing_peak_phase_yields_null_reward_not_a_guess():
    baseline = _metrics(peak_wis=122.4)

    row = reward_for(_metrics(peak_wis=None, overall_wis=5.61), baseline)

    assert row["value"] is None
    assert row["delta"] is None and row["improvement_frac"] is None
    assert row["better"] is False
    assert row["guard"]["value"] == 5.61


# ---- log + summary ----------------------------------------------------------


def test_log_row_and_read_log_round_trip_the_state_action_reward_row():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / LOG_NAME
        arm = arm_lambda(2.0)
        baseline = _metrics(peak_wis=122.4)
        metrics = _metrics(peak_wis=110.0)

        written = log_row(log_path, arm, ["2025-12-06", "2026-01-03"], metrics, reward_for(metrics, baseline), 93.25)
        rows = read_log(log_path)

        assert rows == [json.loads(json.dumps(written))]
        row = rows[0]
        assert row["arm"] == ARM_LAMBDA and row["config_id"] == "lambda_2"
        assert row["config_delta"] == {
            "sample_weights.approaching_peak": {"weeks_before": DEFAULT_APPROACHING_PEAK_WEEKS, "weight": 2.0}
        }
        assert row["state"]["model"] == "xgboost_direct"
        assert row["state"]["target_phase"] == "peak" and row["state"]["metric"] == "wis"
        assert row["state"]["n_cutoffs"] == 2
        assert row["reward"]["delta"] == -12.4 and row["reward"]["better"] is True
        assert row["seconds"] == 93.2
        assert row["run_at"].endswith("+00:00")


def test_read_log_missing_file_is_empty_and_malformed_line_names_the_line():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / LOG_NAME
        assert read_log(log_path) == []
        log_path.write_text('{"arm": "baseline"}\nnot json\n')
        _expect_value_error(lambda: read_log(log_path), f"{log_path}:2", "malformed log line")


def test_write_summary_renders_every_logged_config_from_the_log():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        log_path = out_dir / LOG_NAME
        baseline = _metrics(peak_wis=122.4, overall_wis=60.0, peak_bias=-52.4)
        # Log the window arm first to prove ordering follows ARM_NAMES, not log order.
        log_row(log_path, arm_window(12), ["c1"], _metrics(peak_wis=None, overall_wis=61.0),
                reward_for(_metrics(peak_wis=None), baseline), 10.0)
        log_row(log_path, arm_lambda(2.0), ["c1"], _metrics(peak_wis=100.0, overall_wis=58.0),
                reward_for(_metrics(peak_wis=100.0), baseline), 20.0)
        log_row(log_path, build_arms([])[0], ["c1"], baseline, reward_for(baseline, baseline), 5.0)

        path = write_summary(out_dir)
        text = path.read_text()

        assert path == out_dir / SUMMARY_NAME
        table_rows = [line for line in text.splitlines() if line.startswith("| ") and "`" in line]
        assert [r.split("|")[2].strip() for r in table_rows] == [
            f"`{BASELINE_CONFIG_ID}`", "`lambda_2`", "`window_12`"
        ]
        assert "| lambda | `lambda_2` | 100.00 | 58.00 | -10.0 | -22.40 | +18.3% | yes | 20.0 |" in text
        assert f"| window | `window_12` | {SUMMARY_MISSING} | 61.00 | {SUMMARY_MISSING} | {SUMMARY_MISSING} | {SUMMARY_MISSING} | no | 10.0 |" in text
        assert "(3 configs)" in text


def test_write_summary_with_no_log_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        text = write_summary(Path(tmp)).read_text()
        assert "(no rows logged yet)" in text
        assert "(0 configs)" in text


# ---- round 2: arms + seeds --------------------------------------------------


def test_build_arms_oversample_expands_factors_after_the_other_arms():
    arms = build_arms([ARM_OVERSAMPLE, ARM_WINDOW], windows=(52,), oversample_factors=(2, 5))

    assert [a.config_id for a in arms] == [BASELINE_CONFIG_ID, "window_52", "oversample_2", "oversample_5"]
    assert arms[2].config_delta == {"data.oversample": {"phase": "peak", "factor": 2}}
    assert arms[2].describe == (
        "peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each)"
    )


def test_arm_oversample_rejects_a_factor_below_two_or_a_bool():
    _expect_value_error(lambda: arm_oversample(1), "oversample factor", ">= 2", "1")
    _expect_value_error(lambda: arm_oversample(True), "oversample factor", "True")
    _expect_value_error(lambda: arm_oversample(2.0), "oversample factor", "2.0")


def test_with_seed_always_suffixes_the_config_id():
    arm = arm_lambda(2.0)

    seeded = with_seed(arm, 43)

    assert seeded.config_id == "lambda_2__seed43" and seeded.base_config_id == "lambda_2" and seeded.seed == 43
    assert with_seed(arm, 42).config_id == "lambda_2__seed42"
    assert seeded.config_delta == {**arm.config_delta, "xgboost.random_state": 43}
    assert arm.config_delta == {
        "sample_weights.approaching_peak": {"weeks_before": DEFAULT_APPROACHING_PEAK_WEEKS, "weight": 2.0}
    }


def test_with_seed_rejects_a_negative_or_bool_seed():
    _expect_value_error(lambda: with_seed(arm_lambda(2.0), -1), "seed", "-1")
    _expect_value_error(lambda: with_seed(arm_lambda(2.0), True), "seed", "True")


def test_with_seed_random_state_reaches_the_ensemble_run_pipeline_builds():
    config = _seeded_config(arm_lambda(2.0), 7)

    cls, params = _model_params_run_pipeline_builds(config)

    assert cls == "QuantileDirectForecastEnsemble"
    assert params["random_state"] == 7


def test_with_seed_random_state_reaches_the_oversampled_ensemble_run_pipeline_builds():
    config = _seeded_config(arm_oversample(2), 9)

    cls, params = _model_params_run_pipeline_builds(config)

    assert cls == "PhaseOversampledQuantileEnsemble"
    assert params["random_state"] == 9


def test_with_seed_random_state_reaches_the_fitted_xgboost_model():
    config = apply_delta(get_default_config(), with_seed(arm_lambda(2.0), 7).config_delta)
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-06", periods=40, freq="7D"),
        "location": "06",
        "value": np.arange(40, dtype=float),
        "feat_a": np.arange(40, dtype=float),
    })
    ensemble = QuantileDirectForecastEnsemble(
        forecast_horizon=1, target_mode="log", quantiles=[0.5],
        xgb_params_override={**config["xgboost"], "n_estimators": 5},
    )

    ensemble.train(frame)

    assert ensemble.models[1][0.5].model.get_params()["random_state"] == 7


def test_default_out_dir_is_not_the_round_1_directory():
    assert DEFAULT_OUT_DIR != ROUND1_OUT_DIR
    assert DEFAULT_OUT_DIR.name == "peak_rectification_round2"
    assert DEFAULT_SEEDS == (42,)


def test_ensure_manifest_refuses_a_different_seed_list_or_phase_scheme():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        (out_dir / MANIFEST_NAME).write_text(json.dumps({"cutoffs": ["c1"], "seeds": [42, 43]}))
        _expect_value_error(lambda: _ensure_manifest(out_dir, ["c1"], [42]), "seeds", "[42, 43]", "--out")
        # A round-1 manifest has no phase_scheme: it was calendar_v1.
        (out_dir / MANIFEST_NAME).write_text(json.dumps({"cutoffs": ["c1"]}))
        _expect_value_error(lambda: _ensure_manifest(out_dir, ["c1"], [42]), "phase_scheme", "calendar_v1")


def test_ensure_manifest_refuses_a_changed_data_cache():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        manifest = {"cutoffs": ["c1"], "seeds": [42], "phase_scheme": DEFAULT_PHASE_SCHEME,
                    "data_cache": {"sha256": "0" * 64}}
        (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest))

        _expect_value_error(lambda: _ensure_manifest(out_dir, ["c1"], [42]), "sha256", "0" * 64, "--out")


def test_ensure_manifest_resumes_when_the_data_cache_is_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        manifest = {"cutoffs": ["c1"], "seeds": [42], "phase_scheme": DEFAULT_PHASE_SCHEME,
                    "data_cache": {"sha256": _data_cache_sha256()}}
        (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest))

        assert _ensure_manifest(out_dir, ["c1"], [42]) == manifest


def test_frozen_data_cache_never_reports_the_cache_stale():
    loader = FluDataLoader()

    with _frozen_data_cache():
        stale = loader.is_cache_stale(max_age_hours=0)

    assert stale is False


# ---- round 2: phase x horizon -----------------------------------------------


def test_score_adds_by_phase_horizon_from_per_horizon_subsets():
    forecasts = _forecasts({1: [10.0, 20.0], 2: [30.0], 4: [50.0, 70.0]})

    metrics = score(_HorizonEchoAdapter(), forecasts, pd.DataFrame())

    assert metrics["phase_scheme"] == "test_scheme"
    assert metrics["by_phase"]["peak"] == {"wis": 36.0, "bias": 0.0, "n": 5}
    assert metrics["by_phase_horizon"] == {
        "peak": {
            "1": {"wis": 15.0, "bias": 0.0, "n": 2},
            "2": {"wis": 30.0, "bias": 0.0, "n": 1},
            "4": {"wis": 60.0, "bias": 0.0, "n": 2},
        }
    }
    assert sorted(metrics["by_horizon"]) == ["1", "2", "4"]


def test_write_summary_renders_peak_and_overall_wis_by_horizon_with_deltas():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        log_path = out_dir / LOG_NAME
        baseline = _horizon_metrics(120.0, {"1": 29.1, "2": 36.9}, {"1": 68.2, "2": 100.3})
        lam = _horizon_metrics(110.0, {"1": 25.2, "2": 36.0}, {"1": 57.2, "2": 93.7})
        log_row(log_path, build_arms([])[0], ["c1"], baseline, reward_for(baseline, baseline), 1.0)
        log_row(log_path, arm_lambda(5.0), ["c1"], lam, reward_for(lam, baseline), 1.0)

        text = write_summary(out_dir).read_text()

        assert "## Peak WIS by horizon" in text and "## Overall WIS by horizon" in text
        assert "| config | h1 (delta) | h2 (delta) |" in text
        assert "| `lambda_5` | 57.2 (-11.0) | 93.7 (-6.6) |" in text
        assert "| `lambda_5` | 25.2 (-3.9) | 36.0 (-0.9) |" in text
        assert "| `baseline` | 29.1 (+0.0) | 36.9 (+0.0) |" in text


def test_write_summary_round_1_rows_without_phase_horizon_render_missing():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        metrics = _horizon_metrics(120.0, {"1": 29.1}, {})
        log_row(out_dir / LOG_NAME, build_arms([])[0], ["c1"], metrics, reward_for(metrics, metrics), 1.0)

        text = write_summary(out_dir).read_text()

        assert f"| `{BASELINE_CONFIG_ID}` | {SUMMARY_MISSING} |" in text
        assert "Phase scheme: `calendar_v1`." in text


# ---- round 2: seeds in the summary ------------------------------------------


def test_write_summary_reports_seed_mean_sd_and_paired_deltas():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        log_path = out_dir / LOG_NAME
        base42, base43 = _metrics(peak_wis=120.0, overall_wis=50.0), _metrics(peak_wis=124.0, overall_wis=52.0)
        lam42, lam43 = _metrics(peak_wis=110.0, overall_wis=49.0), _metrics(peak_wis=118.0, overall_wis=51.0)
        base = build_arms([])[0]
        log_row(log_path, with_seed(base, 42), ["c1"], base42, reward_for(base42, base42), 1.0)
        log_row(log_path, with_seed(arm_lambda(2.0), 42), ["c1"], lam42, reward_for(lam42, base42), 1.0)
        log_row(log_path, with_seed(base, 43), ["c1"], base43, reward_for(base43, base43), 1.0)
        log_row(log_path, with_seed(arm_lambda(2.0), 43), ["c1"], lam43, reward_for(lam43, base43), 1.0)

        text = write_summary(out_dir).read_text()

        assert "## Across seeds" in text
        assert "| `baseline` | 2 | 122.00 | 2.83 | 51.00 | 1.41 | +0.00 | 42: +0.00, 43: +0.00 |" in text
        assert "| `lambda_2` | 2 | 114.00 | 5.66 | 50.00 | 1.41 | -8.00 | 42: -10.00, 43: -6.00 |" in text
        assert "| lambda | `lambda_2__seed43` | 118.00 | 51.00 | -10.0 | -6.00 |" in text


def test_write_summary_horizon_cells_show_the_paired_delta_range_across_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        log_path = out_dir / LOG_NAME
        base42 = _horizon_metrics(120.0, {"1": 30.0}, {"1": 70.0})
        base43 = _horizon_metrics(124.0, {"1": 32.0}, {"1": 74.0})
        lam42 = _horizon_metrics(110.0, {"1": 26.0}, {"1": 60.0})
        lam43 = _horizon_metrics(118.0, {"1": 30.0}, {"1": 72.0})
        base = build_arms([])[0]
        log_row(log_path, with_seed(base, 42), ["c1"], base42, reward_for(base42, base42), 1.0)
        log_row(log_path, with_seed(arm_lambda(2.0), 42), ["c1"], lam42, reward_for(lam42, base42), 1.0)
        log_row(log_path, with_seed(base, 43), ["c1"], base43, reward_for(base43, base43), 1.0)
        log_row(log_path, with_seed(arm_lambda(2.0), 43), ["c1"], lam43, reward_for(lam43, base43), 1.0)

        text = write_summary(out_dir).read_text()

        assert "| `lambda_2` | 66.0 (-6.0; paired -10.0..-2.0) |" in text
        assert "| `lambda_2` | 28.0 (-3.0; paired -4.0..-2.0) |" in text


def test_write_summary_single_seed_has_no_across_seeds_table():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        metrics = _metrics(peak_wis=120.0)
        log_row(out_dir / LOG_NAME, build_arms([])[0], ["c1"], metrics, reward_for(metrics, metrics), 1.0)

        text = write_summary(out_dir).read_text()

        assert "## Across seeds" not in text


# ---- round 2: rescore -------------------------------------------------------


def test_rescore_writes_siblings_and_leaves_the_original_log_and_summary_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        (out_dir / FORECASTS_DIRNAME).mkdir()
        _forecasts({1: [100.0], 2: [140.0]}).to_csv(out_dir / FORECASTS_DIRNAME / "baseline.csv", index=False)
        _forecasts({1: [80.0], 2: [140.0]}).to_csv(out_dir / FORECASTS_DIRNAME / "lambda_2.csv", index=False)
        stale = _metrics(peak_wis=999.0)
        log_row(out_dir / LOG_NAME, build_arms([])[0], ["c1"], stale, reward_for(stale, stale), 1.0)
        log_row(out_dir / LOG_NAME, arm_lambda(2.0), ["c1"], stale, reward_for(stale, stale), 1.0)
        write_summary(out_dir)
        log_before = (out_dir / LOG_NAME).read_bytes()
        summary_before = (out_dir / SUMMARY_NAME).read_bytes()

        path = rescore(out_dir, _HorizonEchoAdapter(), pd.DataFrame())
        rows = read_log(path)

        assert path == out_dir / RESCORED_LOG_NAME
        assert (out_dir / LOG_NAME).read_bytes() == log_before
        assert (out_dir / SUMMARY_NAME).read_bytes() == summary_before
        assert (out_dir / RESCORED_SUMMARY_NAME).exists()
        assert [r["config_id"] for r in rows] == [BASELINE_CONFIG_ID, "lambda_2"]
        assert rows[1]["metrics"]["by_phase"]["peak"]["wis"] == 110.0
        assert rows[1]["metrics"]["by_phase_horizon"]["peak"]["1"]["wis"] == 80.0
        assert rows[1]["reward"]["baseline_value"] == 120.0 and rows[1]["reward"]["delta"] == -10.0
        assert rows[1]["rescored"]["original_phase_scheme"] == "calendar_v1"
        assert set(rows[1]["rescored"]) == {
            "at", "phase_scheme", "data_cache_mtime", "data_cache_sha256", "original_phase_scheme",
        }
        assert rows[1]["run_at"] == read_log(out_dir / LOG_NAME)[1]["run_at"]


def test_rescore_missing_forecasts_fails_before_writing_anything():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        metrics = _metrics(peak_wis=120.0)
        log_row(out_dir / LOG_NAME, build_arms([])[0], ["c1"], metrics, reward_for(metrics, metrics), 1.0)

        message = _expect_error(FileNotFoundError, lambda: rescore(out_dir, _HorizonEchoAdapter(), pd.DataFrame()))

        assert "baseline" in message
        assert not (out_dir / RESCORED_LOG_NAME).exists()


# ---- round 2: merge ---------------------------------------------------------


def test_merge_combines_per_seed_directories_into_one_log_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_source(root / "s42", 42)
        _write_source(root / "s43", 43)

        log_path = merge(root / "merged", [root / "s42", root / "s43"])
        manifest = json.loads((root / "merged" / MANIFEST_NAME).read_text())
        text = (root / "merged" / SUMMARY_NAME).read_text()

        assert [r["config_id"] for r in read_log(log_path)] == [
            "baseline__seed42", "lambda_2__seed42", "baseline__seed43", "lambda_2__seed43",
        ]
        assert manifest["seeds"] == [42, 43]
        assert [m["seeds"] for m in manifest["merged_from"]] == [[42], [43]]
        assert sorted(p.name for p in (root / "merged" / FORECASTS_DIRNAME).iterdir()) == [
            "baseline__seed42.csv", "baseline__seed43.csv", "lambda_2__seed42.csv", "lambda_2__seed43.csv",
        ]
        assert "## Across seeds" in text


def test_merge_refuses_sources_with_different_data_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_source(root / "s42", 42)
        _write_source(root / "s43", 43, {"data_cache": {"sha256": "beef"}})

        _expect_value_error(lambda: merge(root / "merged", [root / "s42", root / "s43"]), DATA_CACHE_HASH_KEY)

        assert not (root / "merged").exists()


def test_merge_refuses_overlapping_seeds_and_an_existing_target():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_source(root / "a", 42)
        _write_source(root / "b", 42)
        _write_source(root / "c", 43)

        _expect_value_error(lambda: merge(root / "merged", [root / "a", root / "b"]), "share seeds", "42")
        _expect_value_error(lambda: merge(root / "a", [root / "b", root / "c"]), "already has", "--out")


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main() -> None:
    passed = failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001 - test runner surface
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
