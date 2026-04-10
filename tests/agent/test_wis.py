"""Unit tests for WIS computation in agent.phase_evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.phase_evaluator import (
    PhaseEvaluator,
    QUANTILE_COLS,
    _compute_wis_row,
    _compute_wis_vector,
    _has_quantiles,
)


# ---- _compute_wis_row ------------------------------------------------------

def test_wis_row_perfect_prediction_is_just_interval_widths():
    """When all quantiles equal the actual, no penalty fires; WIS = 0."""
    q = {0.05: 100.0, 0.25: 100.0, 0.50: 100.0, 0.75: 100.0, 0.95: 100.0}
    wis = _compute_wis_row(actual=100.0, q=q)
    assert wis == 0.0


def test_wis_row_actual_inside_intervals():
    """Actual within both intervals → WIS is just interval-width term + median error."""
    q = {0.05: 80.0, 0.25: 90.0, 0.50: 100.0, 0.75: 110.0, 0.95: 120.0}
    wis = _compute_wis_row(actual=100.0, q=q)
    # WIS = (1/2.5) * (0.5*0 + 0.05*40 + 0.25*20)
    #     = (1/2.5) * (2 + 5) = 2.8
    assert abs(wis - 2.8) < 1e-9


def test_wis_row_penalizes_when_actual_below_q05():
    """If actual < q05, the 90% interval gets a (2/0.10) penalty."""
    q = {0.05: 80.0, 0.25: 90.0, 0.50: 100.0, 0.75: 110.0, 0.95: 120.0}
    # actual = 60, which is 20 below q05
    wis = _compute_wis_row(actual=60.0, q=q)
    # is90 = 40 + (2/0.10)*20 = 40 + 400 = 440
    # is50 = 20 + (2/0.50)*30 = 20 + 120 = 140    (actual < q25=90)
    # WIS = (1/2.5) * (0.5*40 + 0.05*440 + 0.25*140)
    #     = (1/2.5) * (20 + 22 + 35) = 30.8
    assert abs(wis - 30.8) < 1e-9


def test_wis_row_penalizes_when_actual_above_q95():
    """Symmetric case: actual above q95."""
    q = {0.05: 80.0, 0.25: 90.0, 0.50: 100.0, 0.75: 110.0, 0.95: 120.0}
    # actual = 140, which is 20 above q95
    wis = _compute_wis_row(actual=140.0, q=q)
    # is90 = 40 + (2/0.10)*20 = 440
    # is50 = 20 + (2/0.50)*30 = 140
    # WIS = (1/2.5) * (0.5*40 + 22 + 35) = 30.8
    assert abs(wis - 30.8) < 1e-9


def test_wis_row_lower_is_better():
    """A wider, miscalibrated interval should score worse than a tight one."""
    q_tight = {0.05: 95.0, 0.25: 98.0, 0.50: 100.0, 0.75: 102.0, 0.95: 105.0}
    q_wide = {0.05: 50.0, 0.25: 75.0, 0.50: 100.0, 0.75: 125.0, 0.95: 150.0}
    actual = 100.0
    wis_tight = _compute_wis_row(actual, q_tight)
    wis_wide = _compute_wis_row(actual, q_wide)
    assert wis_tight < wis_wide


# ---- _compute_wis_vector ---------------------------------------------------

def test_wis_vector_matches_row_for_each_element():
    actual = np.array([100.0, 60.0, 140.0])
    q05 = np.array([80.0, 80.0, 80.0])
    q25 = np.array([90.0, 90.0, 90.0])
    q50 = np.array([100.0, 100.0, 100.0])
    q75 = np.array([110.0, 110.0, 110.0])
    q95 = np.array([120.0, 120.0, 120.0])

    vec = _compute_wis_vector(actual, q05, q25, q50, q75, q95)
    # Compare row-by-row
    for i in range(len(actual)):
        q = {0.05: q05[i], 0.25: q25[i], 0.50: q50[i], 0.75: q75[i], 0.95: q95[i]}
        row_wis = _compute_wis_row(float(actual[i]), q)
        assert abs(vec[i] - row_wis) < 1e-9


# ---- _has_quantiles --------------------------------------------------------

def test_has_quantiles_true_with_full_set():
    df = pd.DataFrame(columns=QUANTILE_COLS + ["other"])
    assert _has_quantiles(df) is True


def test_has_quantiles_false_when_partial():
    df = pd.DataFrame(columns=["predicted_q05", "predicted_q95"])
    assert _has_quantiles(df) is False


# ---- End-to-end through merge_forecasts_actuals + compute_overall ---------

def _make_forecasts_and_actuals():
    """Tiny synthetic dataset with known answers."""
    forecasts = pd.DataFrame([
        {
            "location": "06", "forecast_date": "2024-12-15", "horizon": 1,
            "predicted": 100.0,
            "predicted_q05": 80.0, "predicted_q25": 90.0,
            "predicted_q50": 100.0, "predicted_q75": 110.0, "predicted_q95": 120.0,
        },
        {
            "location": "06", "forecast_date": "2024-12-22", "horizon": 1,
            "predicted": 100.0,
            "predicted_q05": 80.0, "predicted_q25": 90.0,
            "predicted_q50": 100.0, "predicted_q75": 110.0, "predicted_q95": 120.0,
        },
    ])
    actuals = pd.DataFrame([
        {"date": "2024-12-15", "location": "06", "value": 100.0, "location_name": "California"},
        {"date": "2024-12-22", "location": "06", "value": 60.0, "location_name": "California"},
    ])
    return forecasts, actuals


def test_merge_forecasts_actuals_adds_wis_and_in_pi95_columns():
    forecasts, actuals = _make_forecasts_and_actuals()
    merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)
    assert "wis" in merged.columns
    assert "in_pi95" in merged.columns
    # Row 0: actual=100, perfectly inside both intervals
    # Row 1: actual=60, way below q05=80
    assert merged.iloc[0]["in_pi95"] == 1
    assert merged.iloc[1]["in_pi95"] == 0


def test_compute_overall_metrics_includes_wis_when_quantiles_present():
    forecasts, actuals = _make_forecasts_and_actuals()
    merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)
    overall = PhaseEvaluator.compute_overall_metrics(merged)
    assert "wis" in overall
    assert "coverage_95" in overall
    assert overall["coverage_95"] == 0.5  # 1 in / 1 out


def test_compute_overall_metrics_omits_wis_for_point_only_csv():
    """When no quantile cols are present, the metric set is unchanged."""
    forecasts = pd.DataFrame([
        {"location": "06", "forecast_date": "2024-12-15", "horizon": 1, "forecast": 100.0},
    ])
    actuals = pd.DataFrame([
        {"date": "2024-12-15", "location": "06", "value": 100.0, "location_name": "California"},
    ])
    merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)
    overall = PhaseEvaluator.compute_overall_metrics(merged)
    assert "wis" not in overall
    assert "coverage_95" not in overall


def test_predicted_column_is_renamed_to_forecast():
    """Quantile CSVs use 'predicted' as the point column; the merge handles it."""
    forecasts, actuals = _make_forecasts_and_actuals()
    # Note: forecasts already has 'predicted', no 'forecast'
    assert "forecast" not in forecasts.columns
    merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)
    assert "forecast" in merged.columns
    assert merged["forecast"].iloc[0] == 100.0


ALL = [
    test_wis_row_perfect_prediction_is_just_interval_widths,
    test_wis_row_actual_inside_intervals,
    test_wis_row_penalizes_when_actual_below_q05,
    test_wis_row_penalizes_when_actual_above_q95,
    test_wis_row_lower_is_better,
    test_wis_vector_matches_row_for_each_element,
    test_has_quantiles_true_with_full_set,
    test_has_quantiles_false_when_partial,
    test_merge_forecasts_actuals_adds_wis_and_in_pi95_columns,
    test_compute_overall_metrics_includes_wis_when_quantiles_present,
    test_compute_overall_metrics_omits_wis_for_point_only_csv,
    test_predicted_column_is_renamed_to_forecast,
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
