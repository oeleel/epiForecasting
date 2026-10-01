"""Unit tests for the sample-weight helper in src.direct_forecast.

Covers the three original dimensions (by_phase / by_horizon / by_location)
and the approaching_peak dimension added for the peak-rectification
experiment (knowledge-bank design doc, rectification action 5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.direct_forecast import (
    DEFAULT_APPROACHING_PEAK_WEEKS,
    DirectForecastEnsemble,
    MIN_SEASON_WEEKS,
    _compute_sample_weights,
)


def _meta() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-10-15", "2024-12-15", "2024-12-22", "2025-02-10"]),
        "location": ["06", "06", "12", "12"],
        "horizon": [4, 4, 4, 4],
    })


# ---- approaching_peak synthetic frame ---------------------------------------
#
# Two locations, weekly Saturdays from 2022-02-05 (the real cache start) to a
# mid-season cutoff of 2025-01-11. That gives, per location:
#   2021-22  partial  (34 weeks, < MIN_SEASON_WEEKS)      -> never labeled
#   2022-23  complete peak 06: 2022-12-24 / 12: 2023-01-14 -> labeled
#   2023-24  complete peak both: 2024-01-06                -> labeled
#   2024-25  in progress, values still rising at the cutoff -> never labeled

SEASON_FRAME_START = "2022-02-05"
MID_SEASON_CUTOFF = "2025-01-11"
PEAK_2223 = {"06": "2022-12-24", "12": "2023-01-14"}
PEAK_2324 = {"06": "2024-01-06", "12": "2024-01-06"}
PEAK_VALUE = 100.0
LAMBDA = 3.0
K_WEEKS = 6


def _season_frame() -> pd.DataFrame:
    dates = pd.date_range(SEASON_FRAME_START, MID_SEASON_CUTOFF, freq="7D")
    frames = []
    for loc in ("06", "12"):
        f = pd.DataFrame({
            "date": dates,
            "location": loc,
            "horizon": 4,
            "value": 1.0,
        })
        f.loc[f["date"] == PEAK_2223[loc], "value"] = PEAK_VALUE
        f.loc[f["date"] == PEAK_2324[loc], "value"] = PEAK_VALUE
        # In-progress 2024-25 season: monotone rise so the running max is the
        # last observed week, which must NOT be treated as a peak.
        rising = f["date"] >= "2024-10-05"
        f.loc[rising, "value"] = np.arange(1, rising.sum() + 1, dtype=float) * 10.0
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _weight_at(frame: pd.DataFrame, weights: np.ndarray, loc: str, date: str) -> float:
    idx = frame.index[(frame["location"] == loc) & (frame["date"] == pd.Timestamp(date))]
    assert len(idx) == 1, f"expected one row for {loc} {date}, got {len(idx)}"
    return float(weights[idx[0]])


def _approaching_peak_config(weight: float = LAMBDA, weeks_before: int = K_WEEKS) -> dict:
    return {"approaching_peak": {"weeks_before": weeks_before, "weight": weight}}


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
    # Apply a non-matching weight (no row qualifies) - should return None
    w = _compute_sample_weights(_meta(), {"by_phase": {"summer": 5.0}})
    assert w is None


def test_returns_numpy_array_dtype_float():
    w = _compute_sample_weights(_meta(), {"by_phase": {"peak": 2.0}})
    assert isinstance(w, np.ndarray)
    assert w.dtype == np.float64


# ---- approaching_peak -------------------------------------------------------

def test_approaching_peak_window_rows_get_weight():
    frame = _season_frame()
    w = _compute_sample_weights(frame, _approaching_peak_config())
    assert w is not None and len(w) == len(frame)
    # 06 peaks 2022-12-24; K=6 window = 11-12 .. 12-17 inclusive
    assert _weight_at(frame, w, "06", "2022-11-12") == LAMBDA
    assert _weight_at(frame, w, "06", "2022-12-17") == LAMBDA
    # one week before the window opens, and the week after the peak: 1.0
    assert _weight_at(frame, w, "06", "2022-11-05") == 1.0
    assert _weight_at(frame, w, "06", "2022-12-31") == 1.0


def test_approaching_peak_peak_row_itself_is_unweighted():
    frame = _season_frame()
    w = _compute_sample_weights(frame, _approaching_peak_config())
    assert _weight_at(frame, w, "06", PEAK_2223["06"]) == 1.0
    assert _weight_at(frame, w, "12", PEAK_2223["12"]) == 1.0
    assert _weight_at(frame, w, "06", PEAK_2324["06"]) == 1.0


def test_approaching_peak_exact_window_count_per_eligible_season():
    frame = _season_frame()
    w = _compute_sample_weights(frame, _approaching_peak_config())
    weighted = frame[w != 1.0]
    # 2 locations x 2 eligible seasons x K rows
    assert len(weighted) == 2 * 2 * K_WEEKS
    assert set(np.unique(w[w != 1.0])) == {LAMBDA}


def test_approaching_peak_in_progress_season_is_unweighted_at_mid_season_cutoff():
    frame = _season_frame()
    w = _compute_sample_weights(frame, _approaching_peak_config())
    in_progress = (frame["date"] >= "2024-10-01").values
    assert in_progress.sum() > 0
    assert np.all(w[in_progress] == 1.0)
    # The running max of the in-progress season is the last row: still 1.0.
    assert _weight_at(frame, w, "06", MID_SEASON_CUTOFF) == 1.0


def test_approaching_peak_short_season_is_unweighted():
    frame = _season_frame()
    # 2021-22 partial season: give it an unmistakable peak in March 2022
    frame.loc[(frame["location"] == "06") & (frame["date"] == "2022-03-05"), "value"] = 500.0
    n_partial = frame[(frame["location"] == "06") & (frame["date"] < "2022-10-01")]["date"].nunique()
    assert n_partial < MIN_SEASON_WEEKS
    w = _compute_sample_weights(frame, _approaching_peak_config())
    partial = ((frame["location"] == "06") & (frame["date"] < "2022-10-01")).values
    assert np.all(w[partial] == 1.0)


def test_approaching_peak_locations_have_independent_peaks():
    frame = _season_frame()
    w = _compute_sample_weights(frame, _approaching_peak_config())
    # 12 peaks 2023-01-14, three weeks after 06. 2023-01-07 is in 12's window
    # but after 06's peak; 2022-11-12 is in 06's window but before 12's.
    assert _weight_at(frame, w, "12", "2023-01-07") == LAMBDA
    assert _weight_at(frame, w, "06", "2023-01-07") == 1.0
    assert _weight_at(frame, w, "06", "2022-11-12") == LAMBDA
    assert _weight_at(frame, w, "12", "2022-11-12") == 1.0


def test_approaching_peak_multiplies_with_by_phase():
    frame = _season_frame()
    cfg = {"by_phase": {"peak": 2.0}, "approaching_peak": {"weeks_before": K_WEEKS, "weight": LAMBDA}}
    w = _compute_sample_weights(frame, cfg)
    # 2022-12-17: December (peak phase) and inside 06's window -> 2 * 3
    assert _weight_at(frame, w, "06", "2022-12-17") == 2.0 * LAMBDA
    # 2022-11-12: November (onset) and inside 06's window -> 1 * 3
    assert _weight_at(frame, w, "06", "2022-11-12") == LAMBDA
    # 2022-12-31: December, after the peak -> 2 * 1
    assert _weight_at(frame, w, "06", "2022-12-31") == 2.0


def test_approaching_peak_weeks_before_defaults_when_omitted():
    frame = _season_frame()
    w_default = _compute_sample_weights(frame, {"approaching_peak": {"weight": LAMBDA}})
    w_explicit = _compute_sample_weights(
        frame, _approaching_peak_config(weeks_before=DEFAULT_APPROACHING_PEAK_WEEKS)
    )
    assert w_default.tolist() == w_explicit.tolist()


def test_approaching_peak_empty_dict_means_not_in_use():
    frame = _season_frame()
    assert _compute_sample_weights(frame, {"approaching_peak": {}}) is None


def _expect_value_error(frame: pd.DataFrame, cfg: dict, label: str) -> None:
    try:
        _compute_sample_weights(frame, cfg)
    except ValueError:
        return
    raise AssertionError(f"{label} should have raised ValueError")


def test_approaching_peak_malformed_spec_raises():
    frame = _season_frame()
    _expect_value_error(frame, {"approaching_peak": 3.0}, "non-dict")
    _expect_value_error(frame, {"approaching_peak": {"weeks_before": 6}}, "missing weight")
    _expect_value_error(
        frame, {"approaching_peak": {"weeks_before": 0, "weight": 2.0}}, "weeks_before=0"
    )
    _expect_value_error(
        frame, {"approaching_peak": {"weeks_before": 6.5, "weight": 2.0}}, "float weeks"
    )
    _expect_value_error(
        frame, {"approaching_peak": {"weeks_before": 6, "weight": -1.0}}, "negative weight"
    )
    _expect_value_error(
        frame, {"approaching_peak": {"weeks_before": 6, "weight": 2.0, "lambda": 1}},
        "unknown key",
    )


def test_approaching_peak_requires_value_column():
    # The legacy 4-row meta has no 'value' column: loud failure, not a no-op.
    _expect_value_error(_meta(), _approaching_peak_config(), "missing value column")


def test_approaching_peak_window_crosses_the_october_season_boundary():
    # Location 06's 2023-24 peak moved to the second week of the season
    # (2023-10-14). The 6-week window opens 2023-09-02, inside the previous
    # season's tail; those rows must still be labeled (the window is measured
    # on the location's full series, eligibility on the season).
    frame = _season_frame()
    early_peak = pd.Timestamp("2023-10-14")
    frame.loc[(frame["location"] == "06") & (frame["date"] == PEAK_2324["06"]), "value"] = 1.0
    frame.loc[(frame["location"] == "06") & (frame["date"] == early_peak), "value"] = PEAK_VALUE

    w = _compute_sample_weights(frame, _approaching_peak_config())

    for date in ("2023-09-02", "2023-09-09", "2023-09-16", "2023-09-23", "2023-09-30", "2023-10-07"):
        assert _weight_at(frame, w, "06", date) == LAMBDA, date
    assert _weight_at(frame, w, "06", "2023-08-26") == 1.0
    assert _weight_at(frame, w, "06", "2023-10-14") == 1.0
    # Still exactly K rows per eligible (location, season): nothing else moved.
    assert int((w != 1.0).sum()) == 2 * 2 * K_WEEKS


def test_train_passes_weights_sliced_to_the_training_portion():
    # DirectForecastEnsemble.train computes weights on the full horizon-shifted
    # frame and passes only the first split_idx of them to the model.
    frame = _season_frame()
    frame["feat_a"] = np.arange(len(frame), dtype=float)
    ensemble = DirectForecastEnsemble(forecast_horizon=1, target_mode="raw")
    _, _, _, meta = ensemble.prepare_horizon_data(frame, 1, return_metadata=True)
    expected_full = _compute_sample_weights(meta, _approaching_peak_config())
    split_idx = int(len(meta) * (1 - 0.2))
    seen = {}
    original_train = ensemble.models[1].train

    def recording_train(X, y, validation_data=None, sample_weight=None, **kwargs):
        seen["sample_weight"] = sample_weight
        seen["n_train"] = len(X)
        return original_train(X, y, validation_data, sample_weight=sample_weight, **kwargs)

    ensemble.models[1].train = recording_train

    ensemble.train(frame, validation_split=0.2, sample_weights=_approaching_peak_config())

    assert seen["n_train"] == split_idx
    assert len(seen["sample_weight"]) == split_idx
    np.testing.assert_array_equal(seen["sample_weight"], expected_full[:split_idx])
    assert LAMBDA in set(np.unique(seen["sample_weight"]))


def test_prepare_horizon_data_metadata_carries_value():
    dates = pd.date_range("2024-01-06", periods=10, freq="7D")
    df = pd.DataFrame({
        "date": dates,
        "location": "06",
        "value": np.arange(10, dtype=float),
        "feat_a": 1.0,
    })
    ensemble = DirectForecastEnsemble(forecast_horizon=2)
    X, y, y_raw, meta = ensemble.prepare_horizon_data(df, 1, return_metadata=True)
    assert meta.columns.tolist() == ["date", "location", "horizon", "value"]
    assert len(meta) == len(X) == 9
    # meta value is the ORIGIN-week value, not the shifted target
    assert meta["value"].tolist() == list(range(9))


ALL = [
    test_none_or_empty_returns_none,
    test_phase_weighting_only_affects_peak_rows,
    test_horizon_weighting_uniform_when_horizon_constant,
    test_location_weighting_zero_padded_match,
    test_combined_weights_multiply,
    test_returns_none_when_all_weights_uniform,
    test_returns_numpy_array_dtype_float,
    test_approaching_peak_window_rows_get_weight,
    test_approaching_peak_peak_row_itself_is_unweighted,
    test_approaching_peak_exact_window_count_per_eligible_season,
    test_approaching_peak_in_progress_season_is_unweighted_at_mid_season_cutoff,
    test_approaching_peak_short_season_is_unweighted,
    test_approaching_peak_locations_have_independent_peaks,
    test_approaching_peak_multiplies_with_by_phase,
    test_approaching_peak_weeks_before_defaults_when_omitted,
    test_approaching_peak_empty_dict_means_not_in_use,
    test_approaching_peak_malformed_spec_raises,
    test_approaching_peak_requires_value_column,
    test_approaching_peak_window_crosses_the_october_season_boundary,
    test_train_passes_weights_sliced_to_the_training_portion,
    test_prepare_horizon_data_metadata_carries_value,
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
