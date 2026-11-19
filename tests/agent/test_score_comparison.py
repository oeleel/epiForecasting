"""Tests for scripts/comparison/score_comparison.py (TSci vs our framework, MAE)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.comparison.score_comparison import (  # noqa: E402
    common_rows,
    load_our_run,
    load_tsci_runs,
    summarize,
)


def _row(framework: str, stage: str, predicted: float, location: str = "51", horizon: int = 1) -> dict:
    return {"location": location, "cutoff_date": "2025-12-27", "horizon": horizon,
            "framework": framework, "stage": stage, "predicted": predicted, "actual": 100.0}


def _four_stages(location: str = "51", horizon: int = 1) -> list[dict]:
    return [
        _row("tsci", "baseline", 120.0, location, horizon),
        _row("tsci", "final", 150.0, location, horizon),
        _row("ours", "baseline", 140.0, location, horizon),
        _row("ours", "final", 110.0, location, horizon),
    ]


def test_summarize_reports_mae_and_signed_change_per_framework():
    table = summarize(pd.DataFrame(_four_stages())).set_index("framework")

    assert table.loc["tsci", "baseline_mae"] == 20.0
    assert table.loc["tsci", "final_mae"] == 50.0
    assert table.loc["tsci", "change_pct"] == pytest.approx(150.0)
    assert table.loc["ours", "change_pct"] == pytest.approx(-75.0)
    assert table.loc["ours", "n"] == 1


def test_summarize_groups_by_the_requested_column():
    rows = _four_stages(horizon=1) + _four_stages(horizon=2)

    table = summarize(pd.DataFrame(rows), by=["horizon"])

    assert sorted(table["horizon"].unique()) == [1, 2]
    assert len(table) == 4


def test_common_rows_drops_a_key_one_framework_did_not_forecast():
    ours_only = [_row("ours", "baseline", 90.0, location="06"), _row("ours", "final", 95.0, location="06")]

    shared = common_rows(pd.DataFrame(_four_stages() + ours_only))

    assert set(shared["location"]) == {"51"}
    assert len(shared) == 4


def test_load_tsci_runs_reads_baseline_and_final_per_horizon(tmp_path):
    series_dir, runs_dir = tmp_path / "series", tmp_path / "runs"
    run_dir = runs_dir / "51_2025-12-27"
    run_dir.mkdir(parents=True)
    series_dir.mkdir()
    manifest = {"files": [{"file": "51_2025-12-27.csv", "location": "51",
                           "cutoff_date": "2025-12-27", "horizon": 2}]}
    (series_dir / "manifest.json").write_text(json.dumps(manifest))
    baseline = {"model": "XGBoost", "predictions": [10.0, 20.0], "actuals": [11.0, 22.0]}
    (run_dir / "baseline.json").write_text(json.dumps(baseline))
    results = {"aggregated_results": {"individual_predictions": {"XGBoost": [12.0, 25.0]}}}
    (run_dir / "results.json").write_text(json.dumps(results))

    frame = load_tsci_runs(runs_dir, series_dir)

    final_h2 = frame[(frame["stage"] == "final") & (frame["horizon"] == 2)].iloc[0]
    assert len(frame) == 4
    assert final_h2["predicted"] == 25.0
    assert final_h2["actual"] == 22.0


def test_load_tsci_runs_skips_a_run_without_results(tmp_path):
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    manifest = {"files": [{"file": "06_2025-12-27.csv", "location": "06",
                           "cutoff_date": "2025-12-27", "horizon": 4}]}
    (series_dir / "manifest.json").write_text(json.dumps(manifest))

    assert load_tsci_runs(tmp_path / "runs", series_dir).empty


def test_load_our_run_joins_truth_on_the_target_date(tmp_path):
    forecast = pd.DataFrame({"location": ["51"], "cutoff_date": ["2025-12-27"], "horizon": [1],
                             "target_end_date": ["2026-01-03"], "predicted": [900.0]})
    forecast.to_csv(tmp_path / "iter_0.csv", index=False)
    forecast.assign(predicted=1000.0).to_csv(tmp_path / "best_forecast.csv", index=False)
    truth = pd.DataFrame({"location": ["51", "51"], "date": ["2026-01-03", "2026-01-10"],
                          "actual": [1141.0, 862.0]})

    frame = load_our_run(tmp_path, truth).set_index("stage")

    assert frame.loc["baseline", "predicted"] == 900.0
    assert frame.loc["final", "predicted"] == 1000.0
    assert frame.loc["final", "actual"] == 1141.0
