"""Unit tests for scripts/experiments/gated_emphasis.py (gating, composition, floor counterfactual).

Every frame is hand-built: a couple of locations, forecast dates and horizons. Nothing
reads the hindcast run or the data cache, and nothing trains a model.

Run with:
    PYTHONPATH=. python tests/agent/test_gated_emphasis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.experiments.gated_emphasis import (  # noqa: E402
    QUANTILE_COLUMNS,
    calendar_mask,
    compose,
    curve_mask,
    floor_counterfactual,
    horizon_mask,
    rising_flags,
    score_rows,
)


def _keys(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["location", "cutoff_date", "horizon"])


def _forecasts(rows: list[tuple[str, str, int, list[float]]]) -> pd.DataFrame:
    """One forecast row per (location, cutoff_date, horizon, five quantiles)."""
    records = []
    for location, cutoff, horizon, quantiles in rows:
        target = (pd.Timestamp(cutoff) + pd.DateOffset(weeks=horizon)).date().isoformat()
        record = {"location": location, "cutoff_date": cutoff, "horizon": horizon, "forecast_date": target}
        record.update(dict(zip(QUANTILE_COLUMNS, quantiles)))
        record["predicted"] = record["predicted_q50"]
        records.append(record)
    return pd.DataFrame(records)


def _actuals(location: str, values_by_date: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": list(values_by_date),
        "location": location,
        "location_name": f"loc {location}",
        "value": list(values_by_date.values()),
    })


# ---- horizon gate ----------------------------------------------------------


def test_horizon_mask_selects_h1_and_h2_only():
    keys = _keys([("01", "2025-12-06", h) for h in (1, 2, 3, 4)])

    mask = horizon_mask(keys)

    assert mask.tolist() == [True, True, False, False]


# ---- calendar gate ---------------------------------------------------------


def test_calendar_mask_uses_forecast_date_not_target_date():
    # 2025-12-27 + 2 weeks targets January (peak) but the forecast date is in growth.
    keys = _keys([("01", "2025-12-27", 2), ("01", "2026-01-03", 1), ("01", "2025-10-25", 1), ("01", "2025-11-01", 4)])

    mask = calendar_mask(keys)

    assert mask.tolist() == [True, False, False, True]


def test_calendar_mask_rejects_unknown_phase():
    keys = _keys([("01", "2025-12-06", 1)])

    try:
        calendar_mask(keys, phase="onset")
    except ValueError as err:
        assert "onset" in str(err)
    else:
        raise AssertionError("calendar_mask accepted a phase outside adiga_2026_10")


# ---- curve gate ------------------------------------------------------------


def test_rising_flags_compares_value_k_weeks_back():
    actuals = _actuals("01", {"2025-11-22": 100.0, "2025-11-29": 150.0, "2025-12-06": 120.0})
    pairs = _keys([("01", "2025-12-06", 1)])

    two_back = rising_flags(actuals, pairs, lookback_weeks=2, rise_threshold=0.1)
    one_back = rising_flags(actuals, pairs, lookback_weeks=1, rise_threshold=0.1)

    assert two_back["rising"].tolist() == [True]  # 120 > 100 * 1.1
    assert one_back["rising"].tolist() == [False]  # 120 < 150 * 1.1


def test_rising_flags_threshold_is_strict():
    actuals = _actuals("01", {"2025-11-29": 100.0, "2025-12-06": 110.0})
    pairs = _keys([("01", "2025-12-06", 1)])

    flags = rising_flags(actuals, pairs, lookback_weeks=1, rise_threshold=0.1)

    assert flags["rising"].tolist() == [False]


def test_rising_flags_ignores_values_after_the_forecast_date():
    flat = {"2025-11-29": 100.0, "2025-12-06": 100.0}
    actuals = _actuals("01", {**flat, "2025-12-13": 900.0})
    pairs = _keys([("01", "2025-12-06", 1)])

    flags = rising_flags(actuals, pairs, lookback_weeks=1, rise_threshold=0.0)

    assert flags["rising"].tolist() == [False]


def test_rising_flags_missing_week_is_not_rising():
    actuals = _actuals("01", {"2025-12-06": 500.0})
    pairs = _keys([("01", "2025-12-06", 1)])

    flags = rising_flags(actuals, pairs, lookback_weeks=2, rise_threshold=0.0)

    assert flags["rising"].tolist() == [False]
    assert np.isnan(flags["prior"].iloc[0])


def test_rising_flags_rejects_bad_lookback():
    actuals = _actuals("01", {"2025-12-06": 1.0})
    pairs = _keys([("01", "2025-12-06", 1)])

    try:
        rising_flags(actuals, pairs, lookback_weeks=0, rise_threshold=0.1)
    except ValueError as err:
        assert "got 0" in str(err)
    else:
        raise AssertionError("rising_flags accepted lookback_weeks=0")


def test_curve_mask_broadcasts_location_flag_to_every_horizon():
    keys = _keys([("01", "2025-12-06", 1), ("01", "2025-12-06", 2), ("02", "2025-12-06", 1)])
    flags = pd.DataFrame({"location": ["01", "02"], "cutoff_date": ["2025-12-06"] * 2, "rising": [True, False]})

    mask = curve_mask(keys, flags)

    assert mask.tolist() == [True, True, False]


def test_curve_mask_rejects_uncovered_keys():
    keys = _keys([("01", "2025-12-06", 1)])
    flags = pd.DataFrame({"location": ["02"], "cutoff_date": ["2025-12-06"], "rising": [True]})

    try:
        curve_mask(keys, flags)
    except ValueError as err:
        assert "do not cover" in str(err)
    else:
        raise AssertionError("curve_mask accepted keys with no flag")


# ---- composition -----------------------------------------------------------


def test_compose_takes_emphasis_rows_where_masked():
    baseline = _forecasts([("01", "2025-12-06", 1, [1, 2, 3, 4, 5]), ("01", "2025-12-06", 2, [1, 2, 3, 4, 5])])
    emphasis = _forecasts([("01", "2025-12-06", 1, [9, 9, 9, 9, 9]), ("01", "2025-12-06", 2, [8, 8, 8, 8, 8])])

    composed = compose(baseline, emphasis, np.array([True, False]))

    assert composed["predicted_q50"].tolist() == [9, 3]
    assert composed["from_emphasis"].tolist() == [True, False]


def test_compose_rejects_mismatched_keys():
    baseline = _forecasts([("01", "2025-12-06", 1, [1, 2, 3, 4, 5])])
    emphasis = _forecasts([("02", "2025-12-06", 1, [1, 2, 3, 4, 5])])

    try:
        compose(baseline, emphasis, np.array([True]))
    except ValueError as err:
        assert "same" in str(err)
    else:
        raise AssertionError("compose accepted runs with different keys")


# ---- floor counterfactual ------------------------------------------------------


def test_floor_counterfactual_refloors_baseline_row_after_emphasis_row():
    # h2 floor ratio is 0.3 * 0.95 = 0.285; emphasis h1 of 20 floors h2 at 5.7.
    baseline = _forecasts([("01", "2025-12-06", 1, [1, 2, 3, 4, 5]), ("01", "2025-12-06", 2, [5, 6, 7, 8, 9])])
    emphasis = _forecasts([("01", "2025-12-06", 1, [20] * 5), ("01", "2025-12-06", 2, [0] * 5)])
    composed = compose(baseline, emphasis, np.array([True, False]))

    exact = floor_counterfactual(composed, baseline, "zero")

    assert np.allclose(exact.loc[1, list(QUANTILE_COLUMNS)].to_numpy(float), [5.7, 6, 7, 8, 9])


def test_floor_counterfactual_brackets_a_baseline_value_on_its_floor():
    # Baseline h2 q05 = 0.285 * 1 sits on its floor, so its raw value is only known to be <= 0.285.
    baseline = _forecasts([("01", "2025-12-06", 1, [1, 2, 3, 4, 5]), ("01", "2025-12-06", 2, [0.285, 6, 7, 8, 9])])
    emphasis = _forecasts([("01", "2025-12-06", 1, [0.5] * 5), ("01", "2025-12-06", 2, [0] * 5)])
    composed = compose(baseline, emphasis, np.array([True, False]))

    low = floor_counterfactual(composed, baseline, "zero")
    high = floor_counterfactual(composed, baseline, "floor")

    assert np.isclose(low.loc[1, "predicted_q05"], 0.285 * 0.5)
    assert np.isclose(high.loc[1, "predicted_q05"], 0.285)


def test_floor_counterfactual_is_identity_without_emphasis_rows():
    baseline = _forecasts([("01", "2025-12-06", 1, [1, 2, 3, 4, 5]), ("01", "2025-12-06", 2, [0.285, 6, 7, 8, 9])])
    composed = compose(baseline, baseline, np.array([False, False]))

    exact = floor_counterfactual(composed, baseline, "zero")

    assert np.allclose(exact[list(QUANTILE_COLUMNS)].to_numpy(float), baseline[list(QUANTILE_COLUMNS)].to_numpy(float))


# ---- scoring ---------------------------------------------------------------


def test_score_rows_excludes_us():
    forecasts = _forecasts([("01", "2025-12-06", 1, [1, 2, 3, 4, 5]), ("US", "2025-12-06", 1, [1, 2, 3, 4, 5])])
    actuals = pd.concat([_actuals("01", {"2025-12-13": 3.0}), _actuals("US", {"2025-12-13": 3.0})])

    scored = score_rows(forecasts, actuals)

    assert scored["location"].tolist() == ["01"]


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
