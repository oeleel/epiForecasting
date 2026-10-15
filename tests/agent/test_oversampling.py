"""Unit tests for data.oversample in src/pipeline.py (peak-rectification round 2).

Covers input validation, the copy positions, and the oversampled ensemble's
fit inputs: row counts, copies kept inside their location's block with their
original's target, and a validation block identical to the stock ensemble's.
Model fitting is replaced by a recorder, so nothing here trains XGBoost.

Rows are labelled by TARGET date (origin + h weeks) under the default phase
scheme. The frame's horizon-1 targets fall in June/July (off_season) or
January to early February (peak).

Run with:
    PYTHONPATH=. python tests/agent/test_oversampling.py
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

from src.config import get_default_config  # noqa: E402
from src.direct_forecast import QuantileDirectForecastEnsemble  # noqa: E402
from src.pipeline import (  # noqa: E402
    OversampleSpec,
    PhaseOversampledQuantileEnsemble,
    known_phases,
    oversample_positions,
    run_pipeline,
    validate_oversample,
)

LOCATIONS = ("01", "02", "04", "05", "06")
# 4 off-season, 4 peak, 4 off-season weeks per location (Saturdays).
DATES = (
    ["2023-06-03", "2023-06-10", "2023-06-17", "2023-06-24"]
    + ["2024-01-06", "2024-01-13", "2024-01-20", "2024-01-27"]
    + ["2024-07-06", "2024-07-13", "2024-07-20", "2024-07-27"]
)
QUANTILE = 0.5
FACTOR = 3
# Horizon 1 drops each location's last row: 5 x 11 = 55 rows, split at
# int(55 * 0.8) = 44 = the first four locations exactly.
N_ROWS_H1 = 55
SPLIT_IDX = 44
N_PEAK_TRAIN_ROWS = 4 * 4


def _expect_value_error(fn, *needles: str) -> str:
    try:
        fn()
    except ValueError as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, f"expected {needle!r} in error: {message}"
        return message
    raise AssertionError(f"expected ValueError mentioning {needles}, nothing raised")


def _frame() -> pd.DataFrame:
    rows = [
        {"date": pd.Timestamp(d), "location": loc, "value": 100.0 * i + j, "feat_a": float(j)}
        for i, loc in enumerate(LOCATIONS)
        for j, d in enumerate(DATES)
    ]
    return pd.DataFrame(rows)


def _record_fits(ensemble: QuantileDirectForecastEnsemble) -> dict:
    """Replace the single model's fit/predict with recorders; returns the capture dict."""
    seen: dict = {}

    def recording_train(X, y, validation_data=None, sample_weight=None):
        seen.update(X=X, y=y, X_val=validation_data[0], y_val=validation_data[1], sample_weight=sample_weight)
        return {}

    model = ensemble.models[1][QUANTILE]
    model.train = recording_train
    model.predict = lambda X: np.zeros(len(X))
    return seen


def _oversampled(phase: str = "peak", factor: int = FACTOR) -> PhaseOversampledQuantileEnsemble:
    return PhaseOversampledQuantileEnsemble(
        oversample=OversampleSpec(phase=phase, factor=factor),
        forecast_horizon=1, target_mode="log", quantiles=[QUANTILE],
    )


def _stock() -> QuantileDirectForecastEnsemble:
    return QuantileDirectForecastEnsemble(forecast_horizon=1, target_mode="log", quantiles=[QUANTILE])


def _config_with(**data_overrides) -> dict:
    config = get_default_config()
    config["data"].update(data_overrides)
    return config


# ---- validation -------------------------------------------------------------


def test_oversample_validate_accepts_a_known_phase_and_int_factor():
    spec = validate_oversample({"phase": "peak", "factor": 2})

    assert spec == OversampleSpec(phase="peak", factor=2)
    assert "peak" in known_phases()


def test_oversample_validate_rejects_bad_factor():
    _expect_value_error(lambda: validate_oversample({"phase": "peak", "factor": 1}), "factor", ">= 2", "1")
    _expect_value_error(lambda: validate_oversample({"phase": "peak", "factor": True}), "factor", "bool")
    _expect_value_error(lambda: validate_oversample({"phase": "peak", "factor": 2.0}), "factor", "float")


def test_oversample_validate_rejects_bad_phase_and_shape():
    _expect_value_error(lambda: validate_oversample({"phase": "summer", "factor": 2}), "phase", "'summer'")
    _expect_value_error(lambda: validate_oversample({"phase": 1, "factor": 2}), "phase")
    _expect_value_error(lambda: validate_oversample({"phase": "peak"}), "exactly the keys", "['phase']")
    _expect_value_error(lambda: validate_oversample({"phase": "peak", "factor": 2, "x": 1}), "exactly the keys")
    _expect_value_error(lambda: validate_oversample(["peak", 2]), "must be a dict", "list")


def test_oversample_run_pipeline_rejects_a_bank_family_before_training():
    config = _config_with(oversample={"phase": "peak", "factor": 2})
    config["model"]["family"] = "persistence"

    _expect_value_error(lambda: run_pipeline(config), "data.oversample", "xgboost_direct", "persistence")


def test_oversample_run_pipeline_rejects_the_point_ensemble_before_training():
    config = _config_with(oversample={"phase": "peak", "factor": 2})
    config["quantiles"]["enabled"] = False

    _expect_value_error(lambda: run_pipeline(config), "quantiles.enabled")


def test_oversample_run_pipeline_validates_the_spec_before_training():
    config = _config_with(oversample={"phase": "peak", "factor": 0})

    _expect_value_error(lambda: run_pipeline(config), "data.oversample.factor", "0")


# ---- positions --------------------------------------------------------------


def test_oversample_positions_repeat_phase_rows_next_to_their_original():
    dates = ["2023-06-03", "2024-01-06", "2024-01-13", "2024-07-06"]

    positions = oversample_positions(dates, 1, OversampleSpec(phase="peak", factor=3))

    assert positions.tolist() == [0, 1, 1, 1, 2, 2, 2, 3]


def test_oversample_positions_label_rows_by_target_date_not_origin_date():
    # h4: December origins target January (peak); February origins target March (decline).
    dates = ["2024-12-07", "2025-02-08", "2024-12-28", "2025-02-22"]

    positions = oversample_positions(dates, 4, OversampleSpec(phase="peak", factor=3))

    assert positions.tolist() == [0, 0, 0, 1, 2, 2, 2, 3]


def test_oversample_positions_reject_a_bad_horizon():
    spec = OversampleSpec(phase="peak", factor=2)

    _expect_value_error(lambda: oversample_positions(["2024-01-06"], 0, spec), "horizon", "0")
    _expect_value_error(lambda: oversample_positions(["2024-01-06"], True, spec), "horizon", "True")


# ---- ensemble fit inputs ----------------------------------------------------


def test_oversample_ensemble_adds_factor_minus_one_copies_of_training_peak_rows():
    ensemble = _oversampled()
    seen = _record_fits(ensemble)

    info = ensemble.train(_frame(), validation_split=0.2)

    assert len(seen["X"]) == SPLIT_IDX + (FACTOR - 1) * N_PEAK_TRAIN_ROWS
    assert len(seen["y"]) == len(seen["X"])
    assert info["horizons"][1] == {
        "n_train_rows": SPLIT_IDX,
        "n_train_samples": SPLIT_IDX + (FACTOR - 1) * N_PEAK_TRAIN_ROWS,
        "n_val_samples": N_ROWS_H1 - SPLIT_IDX,
    }


def test_oversample_ensemble_keeps_copies_in_their_location_block_with_the_same_target():
    ensemble = _oversampled()
    seen = _record_fits(ensemble)

    ensemble.train(_frame(), validation_split=0.2)

    # Location "01" is positions 0..10; its January rows are positions 4..7.
    assert seen["X"].index[:19].tolist() == [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 9, 10]
    assert seen["X"].index.is_monotonic_increasing
    assert seen["y"].loc[4].nunique() == 1
    assert float(np.expm1(seen["y"].loc[4].iloc[0])) == 5.0  # value of the week after 2024-01-06
    assert seen["X"].index.max() == SPLIT_IDX - 1


def test_oversample_ensemble_leaves_the_validation_block_identical_to_the_stock_split():
    oversampled, stock = _oversampled(), _stock()
    seen_over, seen_stock = _record_fits(oversampled), _record_fits(stock)

    oversampled.train(_frame(), validation_split=0.2)
    stock.train(_frame(), validation_split=0.2)

    assert seen_over["X_val"].index.tolist() == list(range(SPLIT_IDX, N_ROWS_H1))
    pd.testing.assert_frame_equal(seen_over["X_val"], seen_stock["X_val"])
    pd.testing.assert_series_equal(seen_over["y_val"], seen_stock["y_val"])


def test_oversample_ensemble_with_no_matching_rows_fits_exactly_the_stock_rows():
    oversampled, stock = _oversampled(phase="decline"), _stock()
    seen_over, seen_stock = _record_fits(oversampled), _record_fits(stock)

    oversampled.train(_frame(), validation_split=0.2)
    stock.train(_frame(), validation_split=0.2)

    pd.testing.assert_frame_equal(seen_over["X"], seen_stock["X"])
    pd.testing.assert_series_equal(seen_over["y"], seen_stock["y"])


def test_oversample_ensemble_repeats_sample_weights_with_their_rows():
    ensemble = _oversampled()
    seen = _record_fits(ensemble)

    ensemble.train(_frame(), validation_split=0.2, sample_weights={"by_location": {"01": 2.0}})

    assert len(seen["sample_weight"]) == len(seen["X"])
    assert seen["sample_weight"][:19].tolist() == [2.0] * 19
    assert set(seen["sample_weight"][19:].tolist()) == {1.0}


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
