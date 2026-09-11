"""Unit tests for agent.phase_segmentation (curve-based phase labelling).

Covers the two published steps separately — breakpoint detection and the
Algorithm-3 +-delta classification — plus the real-time incremental variant,
whose whole purpose is that already-settled phase labels must not change when
new data arrives.

Run with:
    PYTHONPATH=. python tests/agent/test_phase_segmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from agent.phase_segmentation import (
    DEFAULT_DELTA,
    PHASE_DECLINE,
    PHASE_PLATEAU,
    PHASE_SURGE,
    classify_phases,
    phase_series,
    piecewise_breakpoints,
    segment_incrementally,
    segment_series,
)


# ---- factories ------------------------------------------------------------

def _wave(rise: int = 12, fall: int = 12, peak: float = 1000.0, floor: float = 10.0):
    """One synthetic epidemic wave: exponential rise then exponential fall."""
    up = np.geomspace(floor, peak, rise)
    down = np.geomspace(peak, floor, fall)[1:]
    return np.concatenate([up, down])


def _flat(n: int = 20, level: float = 500.0):
    return np.full(n, level, dtype=float)


def _dates(n: int, start: str = "2023-10-07"):
    import pandas as pd
    return [str(d)[:10] for d in pd.date_range(start=start, periods=n, freq="W-SAT")]


# ---- classification (their Algorithm 3) -----------------------------------

def test_classify_labels_growth_as_surge():
    values = [100.0, 200.0, 400.0]
    segs = classify_phases(values, breakpoints=[], delta=DEFAULT_DELTA)
    assert len(segs) == 1
    assert segs[0].phase == PHASE_SURGE, segs[0].phase


def test_classify_labels_decay_as_decline():
    values = [400.0, 200.0, 100.0]
    segs = classify_phases(values, breakpoints=[], delta=DEFAULT_DELTA)
    assert segs[0].phase == PHASE_DECLINE, segs[0].phase


def test_classify_labels_flat_as_plateau():
    values = _flat(10)
    segs = classify_phases(values, breakpoints=[], delta=DEFAULT_DELTA)
    assert segs[0].phase == PHASE_PLATEAU, segs[0].phase


def test_classify_respects_the_delta_threshold():
    # +8% total change: a plateau at delta=0.10, a surge at delta=0.05.
    values = [100.0, 104.0, 108.0]
    lenient = classify_phases(values, breakpoints=[], delta=0.10)
    strict = classify_phases(values, breakpoints=[], delta=0.05)
    assert lenient[0].phase == PHASE_PLATEAU
    assert strict[0].phase == PHASE_SURGE


def test_classify_handles_zero_start_without_dividing_by_zero():
    segs = classify_phases([0.0, 0.0, 50.0], breakpoints=[], delta=DEFAULT_DELTA)
    assert segs[0].phase == PHASE_SURGE
    zero = classify_phases([0.0, 0.0, 0.0], breakpoints=[], delta=DEFAULT_DELTA)
    assert zero[0].phase == PHASE_PLATEAU


def test_classify_rejects_out_of_range_delta():
    try:
        classify_phases([1.0, 2.0], breakpoints=[], delta=1.5)
        assert False, "should have raised for delta >= 1"
    except ValueError as e:
        assert "delta" in str(e)


def test_classify_splits_at_supplied_breakpoints():
    values = np.concatenate([_wave(10, 10)])
    mid = int(np.argmax(values))
    segs = classify_phases(values, breakpoints=[mid], delta=DEFAULT_DELTA)
    assert len(segs) == 2
    assert segs[0].phase == PHASE_SURGE
    assert segs[1].phase == PHASE_DECLINE


# ---- breakpoints ----------------------------------------------------------

def test_breakpoints_found_near_the_turning_point():
    values = _wave(14, 14)
    true_peak = int(np.argmax(values))
    bps = piecewise_breakpoints(values, min_segment_weeks=3)
    assert bps, "expected at least one breakpoint on a rise-then-fall curve"
    assert min(abs(b - true_peak) for b in bps) <= 3, f"{bps} vs peak {true_peak}"


def test_breakpoints_empty_for_a_series_too_short_to_split():
    assert piecewise_breakpoints([1.0, 2.0, 3.0], min_segment_weeks=3) == []


def test_breakpoints_rejects_non_finite_values():
    try:
        piecewise_breakpoints([1.0, float("nan"), 3.0, 4.0, 5.0, 6.0])
        assert False, "should have raised on NaN"
    except ValueError as e:
        assert "finite" in str(e)


def test_breakpoints_rejects_empty_input():
    try:
        piecewise_breakpoints([])
        assert False, "should have raised on empty input"
    except ValueError:
        pass


# ---- whole-series segmentation -------------------------------------------

def test_segment_series_recovers_surge_then_decline():
    values = _wave(14, 14)
    result = segment_series(values, _dates(values.size))
    phases = [s.phase for s in result.segments]
    assert PHASE_SURGE in phases and PHASE_DECLINE in phases, phases
    assert phases[0] == PHASE_SURGE
    assert result.current_phase == PHASE_DECLINE


def test_segment_series_carries_dates_onto_segments():
    values = _wave(12, 12)
    dates = _dates(values.size)
    result = segment_series(values, dates)
    assert result.segments[0].start_date == dates[0]
    assert result.segments[-1].end_date == dates[-1]


def test_phase_series_expands_to_one_label_per_week():
    values = _wave(12, 12)
    result = segment_series(values, _dates(values.size))
    labels = phase_series(result, values.size)
    assert len(labels) == values.size
    assert set(labels) <= {PHASE_SURGE, PHASE_PLATEAU, PHASE_DECLINE}


def test_to_json_is_serializable():
    import json
    result = segment_series(_wave(10, 10), _dates(19))
    payload = json.dumps(result.to_json())
    assert "current_phase" in payload and "segments" in payload


# ---- incremental / real-time behavior ------------------------------------

def test_incremental_does_not_rewrite_settled_history():
    """The point of Algorithm 1: an early week's label must not change later."""
    values = np.concatenate([_wave(14, 14), _wave(14, 14)])
    dates = _dates(values.size)

    early_cut = 40
    early = segment_incrementally(values[:early_cut], dates[:early_cut], initial_window=15)
    full = segment_incrementally(values, dates, initial_window=15)

    early_labels = phase_series(early, early_cut)
    full_labels = phase_series(full, values.size)[:early_cut]

    # Allow the trailing few weeks to be revised (they are inside the live tail),
    # but everything before the last two breakpoints must be stable.
    stable_to = max(0, early_cut - 20)
    assert early_labels[:stable_to] == full_labels[:stable_to], (
        f"settled labels changed:\n  early={early_labels[:stable_to]}\n  full ={full_labels[:stable_to]}"
    )


def test_incremental_matches_series_when_input_is_short():
    values = _wave(6, 6)
    inc = segment_incrementally(values, _dates(values.size), initial_window=15)
    assert inc.n_observations == values.size
    assert inc.current_phase in {PHASE_SURGE, PHASE_PLATEAU, PHASE_DECLINE}


def test_incremental_rejects_initial_window_smaller_than_two_segments():
    try:
        segment_incrementally(_wave(20, 20), initial_window=4, min_segment_weeks=3)
        assert False, "should have raised on too-small initial_window"
    except ValueError as e:
        assert "initial_window" in str(e)


def test_current_phase_is_the_most_recent_segment():
    rising = np.geomspace(10, 5000, 30)
    result = segment_series(rising, _dates(30))
    assert result.current_phase == PHASE_SURGE


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
