"""Unit tests for agent.run_tracker.RunTracker."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.run_tracker import RunTracker


def _baseline_metrics(wis: float = 145.2) -> dict:
    return {
        "overall": {
            "wis": wis,
            "mape": 38.4,
            "coverage_95": 0.82,
            "bias": 12.3,
        }
    }


def test_start_log_finish_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv", cutoff_date="2024-11-02")
        t.log_iteration(rid, 0, _baseline_metrics(), forecast_path="x.csv")
        t.log_iteration(
            rid, 1, _baseline_metrics(wis=130.5),
            diagnosis={"summary": "s"},
            action={"name": "adjust_hyperparameter", "params": {"name": "max_depth", "value": 5}},
            config={"xgboost": {"max_depth": 5}},
            forecast_path="iter1.csv",
        )
        t.finish_run(rid, status="completed")

        run = t.get_run(rid)
        assert run is not None
        assert run["status"] == "completed"
        assert len(run["iterations"]) == 2
        assert run["iterations"][1]["action"]["name"] == "adjust_hyperparameter"
        assert run["iterations"][1]["config"]["xgboost"]["max_depth"] == 5


def test_best_iteration_lowest_wis():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv")
        for i, w in enumerate([145.0, 130.0, 138.0]):
            t.log_iteration(rid, i, _baseline_metrics(wis=w))
        best = t.get_best_iteration(rid, metric="wis")
        assert best["iteration"] == 1
        assert best["wis"] == 130.0


def test_best_iteration_coverage_closest_to_095():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv")
        # 0.82 is 0.13 away; 0.92 is 0.03 away (best); 0.99 is 0.04 away
        coverages = [0.82, 0.92, 0.99]
        for i, cov in enumerate(coverages):
            m = _baseline_metrics()
            m["overall"]["coverage_95"] = cov
            t.log_iteration(rid, i, m)
        best = t.get_best_iteration(rid, metric="coverage_95")
        assert best["iteration"] == 1
        assert best["coverage_95"] == 0.92


def test_best_iteration_bias_smallest_absolute():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv")
        biases = [12.3, -3.5, 8.0]  # -3.5 has smallest absolute
        for i, b in enumerate(biases):
            m = _baseline_metrics()
            m["overall"]["bias"] = b
            t.log_iteration(rid, i, m)
        best = t.get_best_iteration(rid, metric="bias")
        assert best["iteration"] == 1


def test_list_runs_aggregates_n_iterations_and_best():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv")
        for i, w in enumerate([145.0, 130.0]):
            t.log_iteration(rid, i, _baseline_metrics(wis=w))
        runs = t.list_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["n_iterations"] == 2
        assert runs[0]["best_wis"] == 130.0


def test_compare_runs_same_run_zero_delta():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv")
        t.log_iteration(rid, 0, _baseline_metrics(wis=145.0))
        cmp = t.compare_runs(rid, rid)
        assert cmp["delta"]["absolute"] == 0.0


def test_get_run_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        assert t.get_run("does-not-exist") is None


def test_invalid_finish_status_raises():
    with tempfile.TemporaryDirectory() as tmp:
        t = RunTracker(db_path=Path(tmp) / "t.db")
        rid = t.start_run(initial_forecast="x.csv")
        try:
            t.finish_run(rid, status="bogus")
            assert False, "should have raised"
        except ValueError:
            pass


ALL = [
    test_start_log_finish_round_trip,
    test_best_iteration_lowest_wis,
    test_best_iteration_coverage_closest_to_095,
    test_best_iteration_bias_smallest_absolute,
    test_list_runs_aggregates_n_iterations_and_best,
    test_compare_runs_same_run_zero_delta,
    test_get_run_missing_returns_none,
    test_invalid_finish_status_raises,
]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print()
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
