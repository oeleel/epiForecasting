"""Tests for the direction metric in scripts/experiments/export_hindcast_tables.py."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from scripts.experiments.export_hindcast_tables import (  # noqa: E402
    classify_direction,
    direction_agreement,
    direction_confusion,
    score_rows,
)

CUTOFF = "2026-01-10"
TARGET = "2026-01-17"


def _rows(entries: list[tuple[str, int, str, str, str]]) -> pd.DataFrame:
    """(location, horizon, phase, predicted_direction, actual_direction) -> exported-row frame."""
    return pd.DataFrame(entries, columns=["location", "horizon", "phase", "predicted_direction", "actual_direction"])


def _actuals(anchor: float, observed: float) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": CUTOFF, "location": "51", "location_name": "Virginia", "value": anchor},
        {"date": TARGET, "location": "51", "location_name": "Virginia", "value": observed},
    ])


def _forecast(median: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "location": "51", "cutoff_date": CUTOFF, "target_end_date": TARGET, "forecast_date": TARGET,
        "horizon": 1, "predicted": median, "predicted_q05": median - 20, "predicted_q25": median - 5,
        "predicted_q50": median, "predicted_q75": median + 5, "predicted_q95": median + 20,
    }])


def test_classify_direction_uses_the_relative_band_on_large_anchors():
    change = pd.Series([15.0, 9.9, -10.0, -19.0])
    anchor = pd.Series([100.0, 100.0, 100.0, 200.0])

    labels = classify_direction(change, anchor)

    assert list(labels) == ["up", "flat", "down", "flat"]


def test_classify_direction_floor_keeps_small_anchor_wobbles_flat():
    change = pd.Series([3.0, -2.0, 9.0, 10.0])
    anchor = pd.Series([0.0, 2.0, 1.0, 0.0])

    labels = classify_direction(change, anchor)

    assert list(labels) == ["flat", "flat", "flat", "up"]


def test_classify_direction_rejects_missing_values():
    try:
        classify_direction(pd.Series([1.0, float("nan")]), pd.Series([10.0, 10.0]))
    except ValueError as exc:
        assert "1 changes" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_score_rows_anchors_on_the_cutoff_value():
    rows = score_rows(_forecast(median=130.0), _actuals(anchor=100.0, observed=95.0))

    assert rows[["anchor", "predicted_direction", "actual_direction"]].iloc[0].tolist() == [100.0, "up", "flat"]


def test_score_rows_rejects_a_cutoff_without_an_observed_value():
    actuals = _actuals(anchor=100.0, observed=95.0).iloc[1:]

    try:
        score_rows(_forecast(median=130.0), actuals)
    except ValueError as exc:
        assert CUTOFF in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_direction_agreement_is_the_share_of_matching_rows_per_horizon():
    rows = _rows([
        ("51", 1, "peak", "up", "up"), ("51", 1, "peak", "up", "down"),
        ("06", 2, "peak", "flat", "flat"), ("06", 2, "decline", "down", "down"),
    ])

    table = direction_agreement({"baseline": rows}, "horizon")

    assert table.set_index("horizon")["agreement"].to_dict() == {"1": 0.5, "2": 1.0, "all": 0.75}


def test_direction_agreement_excludes_us():
    rows = _rows([("51", 1, "peak", "up", "up"), ("US", 1, "peak", "up", "down")])

    table = direction_agreement({"baseline": rows}, "phase")

    assert table.set_index("phase")[["agreement", "n"]].loc["all"].tolist() == [1.0, 1]


def test_direction_confusion_counts_overshoot_after_the_turn():
    rows = _rows([
        ("51", 1, "peak", "up", "down"), ("06", 1, "peak", "down", "down"),
        ("51", 2, "decline", "up", "down"), ("US", 1, "peak", "up", "down"),
    ])

    table = direction_confusion({"baseline": rows})

    overshoot = table[(table["actual_direction"] == "down") & (table["predicted_direction"] == "up")]
    assert overshoot[["n", "share_of_actual"]].iloc[0].tolist() == [1, 0.5]


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
