"""Unit tests for agent.significance (paired per-date forecast-score tests)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.significance import (  # noqa: E402
    KERNEL_BARTLETT,
    KERNEL_RECTANGULAR,
    block_bootstrap_interval,
    diebold_mariano,
)

NOISY_GAIN = [-3.1, -0.4, -2.2, 0.9, -1.7, -2.8, 0.3, -1.1, -2.5, -0.6, -1.9, -0.2]
ALTERNATING = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]


def test_diebold_mariano_lag_zero_equals_one_sample_t_test():
    expected = stats.ttest_1samp(NOISY_GAIN, 0.0)

    result = diebold_mariano(NOISY_GAIN, max_lag=0)

    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert (result.n, result.df, result.variance_kernel) == (12, 11, KERNEL_RECTANGULAR)


def test_diebold_mariano_consistent_gain_is_significant_and_negative():
    result = diebold_mariano(NOISY_GAIN, max_lag=3)

    assert result.mean_delta < 0
    assert result.statistic < 0
    assert result.p_value < 0.05


def test_diebold_mariano_zero_mean_noise_is_not_significant():
    result = diebold_mariano(ALTERNATING, max_lag=0)

    assert result.mean_delta == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_diebold_mariano_falls_back_to_bartlett_when_rectangular_variance_is_negative():
    deltas = [d - 0.1 for d in ALTERNATING]  # gamma_1 = -7/8 gamma_0: rectangular V < 0

    result = diebold_mariano(deltas, max_lag=1)

    assert result.variance_kernel == KERNEL_BARTLETT
    assert np.isfinite(result.statistic)


def test_diebold_mariano_identical_forecasts_give_p_one():
    result = diebold_mariano([0.0, 0.0, 0.0, 0.0], max_lag=1)

    assert (result.statistic, result.p_value) == (0.0, 1.0)


def test_diebold_mariano_constant_nonzero_shift_raises():
    with pytest.raises(ValueError, match="constant shift"):
        diebold_mariano([-2.0, -2.0, -2.0, -2.0], max_lag=1)


def test_diebold_mariano_lag_too_large_for_dates_raises():
    with pytest.raises(ValueError, match="max_lag must be < n - 1"):
        diebold_mariano([1.0, -1.0, 0.5], max_lag=2)


def test_diebold_mariano_too_few_dates_raises():
    with pytest.raises(ValueError, match="at least 3 dates"):
        diebold_mariano([1.0, -1.0], max_lag=0)


def test_diebold_mariano_non_finite_delta_raises():
    with pytest.raises(ValueError, match="finite"):
        diebold_mariano([1.0, float("nan"), 0.5, 0.2], max_lag=0)


def test_block_bootstrap_interval_brackets_the_mean_and_excludes_zero_for_consistent_gain():
    interval = block_bootstrap_interval(NOISY_GAIN, block_length=4)

    assert interval.low < interval.mean_delta < interval.high
    assert interval.high < 0


def test_block_bootstrap_interval_is_reproducible_for_a_seed():
    first = block_bootstrap_interval(NOISY_GAIN, block_length=4, seed=7)
    second = block_bootstrap_interval(NOISY_GAIN, block_length=4, seed=7)

    assert first == second


def test_block_bootstrap_interval_of_constant_deltas_is_that_constant():
    interval = block_bootstrap_interval([-1.5] * 6, block_length=3, n_resamples=200)

    assert (interval.low, interval.high) == pytest.approx((-1.5, -1.5))


def test_block_bootstrap_interval_block_longer_than_series_raises():
    with pytest.raises(ValueError, match="block_length"):
        block_bootstrap_interval([1.0, 2.0, 3.0], block_length=4)


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
