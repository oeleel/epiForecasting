"""Unit tests for agent.run_report (the end-of-run report, roadmap 6.1).

These cover the three things a wrong report would quietly get away with:
direction-aware deltas (for wis a negative absolute delta is an improvement),
carry-forward rows being flagged rather than credited, and the degenerate
shapes real data actually has — zero-iteration runs, error iterations, string
*and* int horizon keys, a `by_phase` with no `onset` in it, and a best
iteration scored on a different evaluation window than the baseline.

Run with:
    PYTHONPATH=. python tests/agent/test_run_report.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.orchestrator import IterationRecord, RunResult
from agent.run_report import (
    build_report,
    build_report_from_tracker,
    render_markdown,
    write_report,
)

# ---- factories --------------------------------------------------------------

def _metrics(wis: float, mape: float = 30.0, by_phase=None, by_horizon=None) -> dict:
    """An 8-key metrics dict shaped exactly like PhaseEvaluator's real output."""
    return {
        "overall": {
            "mape": mape, "mae": 145.6, "rmse": 626.2, "bias": -78.5,
            "n_forecasts": 212, "wis": wis, "coverage_95": 0.939,
        },
        "by_phase": by_phase if by_phase is not None else {
            "peak": {"mape": 25.8, "mae": 61.2, "bias": 47.2, "n": 53, "wis": wis * 0.5},
            "decline": {"mape": 34.9, "mae": 173.8, "bias": -120.4, "n": 159, "wis": wis * 1.2},
        },
        "by_horizon": by_horizon if by_horizon is not None else {
            "1": {"mape": 25.8, "mae": 61.2, "bias": 47.2, "n": 53, "wis": wis * 0.5},
            "2": {"mape": 33.5, "mae": 100.1, "bias": -3.4, "n": 53, "wis": wis * 0.8},
        },
        "worst_locations": [],
        "best_locations": [],
        "worst_segments": [],
        "date_range": {"min": "2026-01-31", "max": "2026-02-21"},
        "n_locations": 53,
    }


def _rec(iteration, target, status="applied", metrics=None, action=None) -> IterationRecord:
    """One IterationRecord, defaulting to an applied hyperparameter tweak."""
    if action is None and iteration > 0:
        action = {
            "name": "adjust_hyperparameter",
            "params": {"name": "input_size", "value": 52},
            "rationale": "widen the lookback",
            "expected_effect": "better peak fit",
        }
    return IterationRecord(
        iteration=iteration,
        config={"model": {"family": "nf_nhits"}, "data": {"cutoff_date": "2026-01-24"}},
        forecast_path=f"iter_{iteration}.csv",
        metrics=metrics if metrics is not None else _metrics(target if target else 1.0),
        diagnosis=({"summary": "peak underprediction", "suggested_focus": "week3"}
                   if iteration > 0 else None),
        action=action,
        action_status=status,
        target_value=target,
        change_desc=(f"input_size 32 -> 52 (iter {iteration})" if iteration > 0 else None),
    )


def _result(targets, target_metric="wis", stop_reason="max_iterations") -> RunResult:
    """A RunResult whose iteration i has target value targets[i]."""
    records = [
        _rec(i, t, status="baseline" if i == 0 else "applied",
             metrics=_metrics(t) if target_metric == "wis" else _coverage_metrics(t))
        for i, t in enumerate(targets)
    ]
    return RunResult(
        run_id="20260903-145933-0732",
        iterations=records,
        best_iteration=0,
        best_forecast_path="best_forecast.csv",
        stop_reason=stop_reason,
        target_metric=target_metric,
    )


def _coverage_metrics(coverage: float) -> dict:
    m = _metrics(100.0)
    m["overall"]["coverage_95"] = coverage
    return m


def _tracker_row() -> dict:
    """A run row shaped like RunTracker.get_run(): string horizon keys, no status."""
    return {
        "run_id": "20260903-140412-65cd",
        "started_at": "2026-09-03T14:04:12+00:00",
        "finished_at": "2026-09-03T14:20:00+00:00",
        "initial_forecast": "(regenerated)",
        "cutoff_date": "2026-01-24",
        "target_metric": "wis",
        "status": "completed",
        "notes": None,
        "iterations": [
            {"iteration": 0, "metrics": _metrics(481.93), "diagnosis": None, "action": None,
             "config": {"model": {"family": "xgboost_direct"}}, "forecast_path": "iter_0.csv",
             "wis": 481.93, "mape": 32.6, "coverage_95": 0.939, "bias": -78.5},
            {"iteration": 1, "metrics": _metrics(481.93), "diagnosis": None,
             "action": {"name": "toggle_feature", "params": {"feature_group": "yoy"}},
             "config": {"model": {"family": "xgboost_direct"}}, "forecast_path": "iter_1.csv",
             "wis": 481.93, "mape": 32.6, "coverage_95": 0.939, "bias": -78.5},
        ],
    }


# ---- build_report -----------------------------------------------------------

def test_build_report_baseline_only_run():
    result = _result([101.67])

    report = build_report(result)

    assert report.n_iterations == 1
    assert report.best_iteration == 0
    assert report.improvement_frac in (None, 0.0)
    assert report.target_metric == "wis"
    assert report.model_family == "nf_nhits"
    md = render_markdown(report)
    assert isinstance(md, str)
    assert md.startswith("# ")


def test_build_report_handles_zero_iterations():
    result = RunResult(
        run_id="20260903-142315-8f8d",
        iterations=[],
        best_iteration=0,
        best_forecast_path="",
        stop_reason="unknown",
        target_metric="wis",
    )

    report = build_report(result)

    assert report.best_iteration is None
    assert report.baseline_value is None
    assert report.best_value is None
    md = render_markdown(report)
    assert "inconclusive" in md


def test_deltas_are_direction_aware_for_wis():
    report = build_report(_result([100.0, 90.0]))

    line = report.lines[1]
    assert line.delta_vs_prev is not None
    assert line.delta_vs_prev > 0, "a lower wis is an improvement"
    assert line.became_best is True
    assert report.best_value - report.baseline_value < 0
    assert report.improvement_frac > 0


def test_deltas_are_direction_aware_for_coverage():
    report = build_report(_result([0.80, 0.94, 0.99], target_metric="coverage_95"))

    assert report.lines[1].delta_vs_prev > 0, "0.94 is closer to 0.95 than 0.80"
    assert report.lines[2].delta_vs_prev < 0, "0.99 is further from 0.95 than 0.94"
    assert report.best_iteration == 1


def test_carry_forward_iterations_are_flagged_not_credited():
    records = [
        _rec(0, 100.0, status="baseline"),
        _rec(1, 100.0, status="skipped", metrics=_metrics(100.0)),
    ]
    result = RunResult(
        run_id="r", iterations=records, best_iteration=0,
        best_forecast_path="", stop_reason="max_iterations", target_metric="wis",
    )

    report = build_report(result)

    line = report.lines[1]
    assert line.carry_forward is True
    assert line.delta_vs_prev is None
    assert line.became_best is False
    assert "carry-forward" in render_markdown(report)


def test_error_iteration_renders_without_crashing():
    records = [
        _rec(0, 100.0, status="baseline"),
        _rec(1, None, status="applied", metrics={"error": "boom"}),
    ]
    result = RunResult(
        run_id="r", iterations=records, best_iteration=0,
        best_forecast_path="", stop_reason="iteration_error: boom", target_metric="wis",
    )

    report = build_report(result)

    line = report.lines[1]
    assert line.error == "boom"
    assert line.target_value is None
    assert line.delta_vs_prev is None
    md = render_markdown(report)
    assert "Errors:" in md
    assert "boom" in md


# ---- render_markdown --------------------------------------------------------

def test_render_markdown_has_header_tables_and_recommendation():
    md = render_markdown(build_report(_result([101.67, 90.0])))

    assert md.startswith("# ")
    assert "- target metric: wis" in md
    assert "- evaluation window: 2026-01-31 .. 2026-02-21 (53 locations, 212 forecasts)" in md
    assert "| iter |" in md
    assert "Where the benefit came from" in md
    assert "Recommendation:" in md
    assert md.index("What was tried") < md.index("Recommendation:")


def test_recommendation_reports_no_benefit_honestly():
    report = build_report(_result([101.67, 533.06, 300.52, 342.23]))

    md = render_markdown(report)

    assert report.best_iteration == 0
    assert "keep the baseline" in md
    assert "adopt" not in md


def test_phase_table_only_shows_phases_present():
    only_two = {
        "peak": {"mape": 25.8, "mae": 61.2, "bias": 47.2, "n": 53, "wis": 47.2},
        "decline": {"mape": 34.9, "mae": 173.8, "bias": -120.4, "n": 159, "wis": 119.83},
    }
    records = [_rec(0, 101.67, status="baseline", metrics=_metrics(101.67, by_phase=only_two))]
    result = RunResult(
        run_id="r", iterations=records, best_iteration=0,
        best_forecast_path="", stop_reason="max_iterations", target_metric="wis",
    )

    md = render_markdown(build_report(result))

    assert "| peak |" in md
    assert "| decline |" in md
    assert "onset" not in md


def test_horizon_table_renders_with_int_keys_from_live_evaluator():
    # PhaseEvaluator.evaluate_by_horizon returns int keys in-process (the tracker
    # path JSON-stringifies them). Regression: live reports rendered a blank table.
    int_keyed = {
        1: {"mape": 25.8, "mae": 61.2, "bias": 47.2, "n": 53, "wis": 50.83},
        2: {"mape": 33.5, "mae": 100.1, "bias": -3.4, "n": 53, "wis": 81.34},
    }
    records = [
        _rec(0, 101.67, status="baseline", metrics=_metrics(101.67, by_horizon=int_keyed)),
        _rec(1, 90.0, metrics=_metrics(90.0, by_horizon=int_keyed)),
    ]
    result = RunResult(
        run_id="r", iterations=records, best_iteration=1,
        best_forecast_path="", stop_reason="max_iterations", target_metric="wis",
    )

    md = render_markdown(build_report(result))

    assert "| 1 | 50.83 | 50.83 | 25.8 | 25.8 | 53 |" in md
    assert "| 2 | 81.34 | 81.34 | 33.5 | 33.5 | 53 |" in md
    assert "(no horizon breakdown recorded)" not in md


def test_improvement_withheld_when_eval_windows_differ():
    # Baseline scored on one window, best iteration on another: the smaller WIS is
    # not comparable, so no improvement may be claimed.
    best_metrics = _metrics(60.0)
    best_metrics["date_range"] = {"min": "2026-02-07", "max": "2026-03-07"}
    best_metrics["overall"]["n_forecasts"] = 159
    records = [
        _rec(0, 101.67, status="baseline", metrics=_metrics(101.67)),
        _rec(1, 60.0, metrics=best_metrics),
    ]
    result = RunResult(
        run_id="r", iterations=records, best_iteration=1,
        best_forecast_path="", stop_reason="max_iterations", target_metric="wis",
    )

    report = build_report(result)
    md = render_markdown(report)

    assert report.windows_match is False
    assert report.improvement_frac is None
    assert report.best_iteration == 1
    assert "- baseline window: 2026-01-31 .. 2026-02-21 (212 forecasts)" in md
    assert "- best-iteration window: 2026-02-07 .. 2026-03-07 (159 forecasts)" in md
    assert "(not comparable: windows differ)" in md
    assert "Recommendation: inconclusive" in md
    assert "different evaluation windows" in md
    assert "adopt iteration" not in md


def test_improvement_reported_when_eval_windows_match():
    report = build_report(_result([101.67, 90.0]))

    assert report.windows_match is True
    assert report.improvement_frac is not None
    assert report.improvement_frac > 0
    assert "adopt iteration 1" in render_markdown(report)


# ---- tracker path + I/O -----------------------------------------------------

def test_report_from_tracker_marks_degraded_fields():
    report = build_report_from_tracker(_tracker_row())

    assert report.source == "tracker"
    assert "action_status" in report.degraded_fields
    assert "stop_reason" in report.degraded_fields
    md = render_markdown(report)
    assert "| 1 |" in md, "string horizon keys must not break the horizon table"
    assert "| 2 |" in md
    assert "identical metrics across iterations" in md


def test_write_report_creates_markdown_and_json():
    report = build_report(_result([101.67, 90.0]))

    with tempfile.TemporaryDirectory() as tmp:
        md_path, json_path = write_report(report, Path(tmp) / "run")

        assert md_path.exists()
        assert json_path.exists()
        payload = json.loads(json_path.read_text())
        assert payload["run_id"] == report.run_id
        assert payload["target_metric"] == "wis"
        assert len(payload["lines"]) == 2


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
