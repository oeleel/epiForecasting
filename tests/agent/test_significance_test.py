"""Unit tests for scripts/experiments/significance_test.py (no training; synthetic run directories)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.significance_test import (  # noqa: E402
    ALL_PHASES,
    GATE_HORIZON,
    GATE_NONE,
    build_candidate,
    date_deltas,
    pooled_date_series,
    run_test,
)

LOCATIONS = ("01", "02", "06")
CUTOFFS = tuple(d.strftime("%Y-%m-%d") for d in pd.date_range("2025-12-06", periods=12, freq="7D"))
HORIZONS = (1, 2, 3, 4)
QUANTILE_COLS = ("predicted_q05", "predicted_q25", "predicted_q50", "predicted_q75", "predicted_q95")
QUANTILE_OFFSETS = (-30.0, -10.0, 0.0, 10.0, 30.0)


def _forecasts(center_shift: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for location in LOCATIONS:
        for cutoff in CUTOFFS:
            for horizon in HORIZONS:
                target = (date.fromisoformat(cutoff) + timedelta(weeks=horizon)).isoformat()
                center = 100.0 + center_shift + rng.normal(0.0, 5.0)
                row = {"location": location, "cutoff_date": cutoff, "forecast_date": target,
                       "horizon": horizon, "predicted": center}
                row.update({col: center + off for col, off in zip(QUANTILE_COLS, QUANTILE_OFFSETS)})
                rows.append(row)
    return pd.DataFrame(rows)


def _actuals() -> pd.DataFrame:
    dates = pd.date_range("2025-11-01", "2026-04-30", freq="7D")
    return pd.DataFrame(
        [{"date": d.strftime("%Y-%m-%d"), "location": loc, "location_name": loc, "value": 100.0}
         for loc in LOCATIONS for d in dates]
    )


def _write_run(run_dir: Path, seeds: tuple[int, ...], candidate_shift: float) -> None:
    """A finished harness directory: manifest, pinned cache, baseline + candidate forecasts per seed."""
    (run_dir / "forecasts").mkdir(parents=True)
    cache = run_dir / "cache.csv"
    _actuals().to_csv(cache, index=False)
    manifest = {
        "phase_scheme": "adiga_2026_10", "seeds": list(seeds), "split": {"eval_window": "primary"},
        "data_cache": {"path": str(cache), "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        _forecasts(0.0, rng).to_csv(run_dir / "forecasts" / f"baseline__seed{seed}.csv", index=False)
        _forecasts(candidate_shift, rng).to_csv(run_dir / "forecasts" / f"cand__seed{seed}.csv", index=False)


def _scored(deltas_by_row: list[tuple[str, int, float]], phase: str = "peak") -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [{"location": "01", "cutoff_date": c, "horizon": h} for c, h, _ in deltas_by_row]
    base = pd.DataFrame([{**k, "wis": 10.0, "phase": phase} for k in keys])
    cand = pd.DataFrame([{**k, "wis": 10.0 + d, "phase": phase} for k, (_, _, d) in zip(keys, deltas_by_row)])
    return cand, base


def test_pooled_date_series_mean_equals_pooled_row_mean_with_unequal_rows_per_date():
    rows = pd.DataFrame({"cutoff_date": ["a", "b", "b", "b", "c", "c"], "delta": [9.0, -1.0, -1.0, -1.0, 2.0, 0.0]})

    series = pooled_date_series(rows)

    assert series.mean() == pytest.approx(rows["delta"].mean())
    assert list(series.index) == ["a", "b", "c"]


def test_date_deltas_keeps_only_the_requested_phase():
    cand, base = _scored([("d1", 1, -2.0), ("d2", 1, 3.0)])
    base.loc[1, "phase"] = "growth"

    peak = date_deltas(cand, base, "peak")
    every = date_deltas(cand, base, ALL_PHASES)

    assert peak["delta"].tolist() == [-2.0]
    assert every["delta"].tolist() == [-2.0, 3.0]


def test_date_deltas_mismatched_keys_raise():
    cand, base = _scored([("d1", 1, -2.0), ("d2", 1, 3.0)])

    with pytest.raises(ValueError, match="do not share their keys"):
        date_deltas(cand.iloc[:1], base, ALL_PHASES)


def test_build_candidate_horizon_gate_takes_emphasis_only_at_h1_h2():
    rng = np.random.default_rng(0)
    baseline = _forecasts(0.0, rng)
    emphasis = _forecasts(50.0, rng)

    composed = build_candidate(baseline, emphasis, GATE_HORIZON)

    assert set(composed.loc[composed["from_emphasis"], "horizon"]) == {1, 2}
    assert set(composed.loc[~composed["from_emphasis"], "horizon"]) == {3, 4}


def test_build_candidate_unknown_gate_raises():
    with pytest.raises(ValueError, match="gate must be one of"):
        build_candidate(pd.DataFrame(), pd.DataFrame(), "calendar")


def test_run_test_mean_delta_equals_pooled_wis_delta_over_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _write_run(run_dir, seeds=(1, 2), candidate_shift=0.0)

        report = run_test(run_dir, "cand", GATE_NONE, phase="peak")

    assert report.seeds == (1, 2)
    assert report.dm.mean_delta == pytest.approx(report.candidate_wis - report.baseline_wis)
    assert report.n_dates == 12


def test_run_test_clearly_worse_candidate_is_significant_harm():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _write_run(run_dir, seeds=(1,), candidate_shift=40.0)

        report = run_test(run_dir, "cand", GATE_NONE, phase=ALL_PHASES)

    assert report.dm.mean_delta > 0
    assert report.significant is True
    assert report.improvement is False


def test_run_test_changed_data_cache_raises():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _write_run(run_dir, seeds=(1,), candidate_shift=0.0)
        with (run_dir / "cache.csv").open("a") as fh:
            fh.write("2026-05-02,01,01,1.0\n")

        with pytest.raises(ValueError, match="sha256"):
            run_test(run_dir, "cand", GATE_NONE)


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
