"""Unit tests for the sample-weight helper in src.direct_forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.direct_forecast import _compute_sample_weights


def _meta() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-10-15", "2024-12-15", "2024-12-22", "2025-02-10"]),
        "location": ["06", "06", "12", "12"],
        "horizon": [4, 4, 4, 4],
    })


def test_none_or_empty_returns_none():
    m = _meta()
    assert _compute_sample_weights(m, None) is None
    assert _compute_sample_weights(m, {}) is None
    assert _compute_sample_weights(
        m, {"by_phase": {}, "by_horizon": {}, "by_location": {}}
    ) is None


def test_phase_weighting_only_affects_peak_rows():
    w = _compute_sample_weights(_meta(), {"by_phase": {"peak": 2.0}})
    # Dec 15 + Dec 22 are peak (idx 1, 2). Oct 15 = onset, Feb 10 = decline.
    assert w.tolist() == [1.0, 2.0, 2.0, 1.0]


def test_horizon_weighting_uniform_when_horizon_constant():
    w = _compute_sample_weights(_meta(), {"by_horizon": {"4": 1.5}})
    assert w.tolist() == [1.5, 1.5, 1.5, 1.5]


def test_location_weighting_zero_padded_match():
    w = _compute_sample_weights(_meta(), {"by_location": {"06": 3.0}})
    assert w.tolist() == [3.0, 3.0, 1.0, 1.0]


def test_combined_weights_multiply():
    w = _compute_sample_weights(
        _meta(), {"by_phase": {"peak": 2.0}, "by_location": {"12": 4.0}}
    )
    # Row 0: onset/06 -> 1*1=1
    # Row 1: peak/06 -> 2*1=2
    # Row 2: peak/12 -> 2*4=8
    # Row 3: decline/12 -> 1*4=4
    assert w.tolist() == [1.0, 2.0, 8.0, 4.0]


def test_returns_none_when_all_weights_uniform():
    # Apply a non-matching weight (no row qualifies) — should return None
    w = _compute_sample_weights(_meta(), {"by_phase": {"summer": 5.0}})
    assert w is None


def test_returns_numpy_array_dtype_float():
    w = _compute_sample_weights(_meta(), {"by_phase": {"peak": 2.0}})
    assert isinstance(w, np.ndarray)
    assert w.dtype == np.float64


ALL = [
    test_none_or_empty_returns_none,
    test_phase_weighting_only_affects_peak_rows,
    test_horizon_weighting_uniform_when_horizon_constant,
    test_location_weighting_zero_padded_match,
    test_combined_weights_multiply,
    test_returns_none_when_all_weights_uniform,
    test_returns_numpy_array_dtype_float,
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
