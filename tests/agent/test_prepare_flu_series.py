"""Tests for scripts/comparison/prepare_flu_series.py (the TSci input files)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.comparison.prepare_flu_series import build_series  # noqa: E402

TRAIN_START = "2025-01-04"
CUTOFF = "2025-03-01"  # the 9th Saturday from TRAIN_START
N_WEEKS = 20


def _cache(locations=("51", "06"), drop_date: str | None = None) -> pd.DataFrame:
    dates = pd.date_range(TRAIN_START, periods=N_WEEKS, freq="7D")
    rows = [
        {"date": d.strftime("%Y-%m-%d"), "location": loc, "value": float(100 * (i + 1) + week)}
        for i, loc in enumerate(locations)
        for week, d in enumerate(dates)
    ]
    frame = pd.DataFrame(rows)
    if drop_date is not None:
        frame = frame[frame["date"] != drop_date]
    return frame


def test_build_series_ends_four_weeks_after_the_cutoff():
    series = build_series(_cache(), "51", CUTOFF, train_start_date=TRAIN_START)

    assert list(series.columns) == ["date", "OT"]
    assert series["date"].iloc[0] == TRAIN_START
    assert series["date"].tolist()[-4:] == ["2025-03-08", "2025-03-15", "2025-03-22", "2025-03-29"]
    assert len(series) == 9 + 4


def test_build_series_keeps_only_the_requested_location():
    series = build_series(_cache(), "06", CUTOFF, train_start_date=TRAIN_START)

    assert series["OT"].iloc[0] == 200.0
    assert series["OT"].iloc[-1] == 212.0


def test_build_series_rejects_an_unknown_location():
    with pytest.raises(ValueError, match="'99'"):
        build_series(_cache(), "99", CUTOFF, train_start_date=TRAIN_START)


def test_build_series_rejects_a_missing_week():
    with pytest.raises(ValueError, match="expected 13 consecutive weeks"):
        build_series(_cache(drop_date="2025-02-01"), "51", CUTOFF, train_start_date=TRAIN_START)


def test_build_series_rejects_a_cutoff_too_close_to_the_end_of_the_cache():
    with pytest.raises(ValueError, match="consecutive weeks"):
        build_series(_cache(), "51", "2025-05-10", train_start_date=TRAIN_START)
