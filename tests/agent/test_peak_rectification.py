"""Unit tests for the pure functions of scripts/experiments/peak_rectification.py.

Covers the parts of the harness that need no data or model: arm expansion,
config-delta application (claim 4: every arm changes exactly one config
path), reward direction (claim 5: WIS lower-is-better), and the log ->
summary round trip. Nothing here trains a model.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The harness sets these itself on darwin, but only after import; set them
# before xgboost/torch are pulled in by the adapter import below.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from scripts.experiments.peak_rectification import (  # noqa: E402
    ARM_BASELINE,
    ARM_LAMBDA,
    ARM_LAMBDA_CALENDAR,
    ARM_WINDOW,
    BASELINE_CONFIG_ID,
    DEFAULT_APPROACHING_PEAK_WEEKS,
    LOG_NAME,
    MIN_TRAIN_WINDOW_WEEKS,
    SUMMARY_MISSING,
    SUMMARY_NAME,
    Arm,
    apply_delta,
    arm_lambda,
    arm_lambda_calendar,
    arm_window,
    build_arms,
    log_row,
    read_log,
    reward_for,
    write_summary,
)


def _expect_value_error(fn, *needles: str) -> str:
    try:
        fn()
    except ValueError as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, f"expected {needle!r} in error: {message}"
        return message
    raise AssertionError(f"expected ValueError mentioning {needles}, nothing raised")


def _metrics(peak_wis: float | None, overall_wis: float = 50.0, peak_bias: float = -10.0) -> dict:
    by_phase = {} if peak_wis is None else {"peak": {"wis": peak_wis, "bias": peak_bias}}
    return {"overall": {"wis": overall_wis}, "by_phase": by_phase, "by_horizon": {}}


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
