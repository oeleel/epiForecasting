"""Significance of a paired forecast-score difference (hypothesis-driven protocol, 2026-11-05).

Question: did a change lower the score, or is the gain inside the noise? The input is one
number per forecast date, `d_t = score(candidate) - score(reference)`, already averaged
over the rows (locations, horizons) and seeds that date covers. Lower scores are better,
so a negative mean is an improvement.

Why the forecast date is the unit, not the row
----------------------------------------------
The rows from one forecast date come from one model fit, so they are not independent.
Treating hundreds of rows as samples overstates confidence. The dates themselves are
also correlated: forecasts made up to H-1 weeks apart share target weeks, so both
methods below allow for autocorrelation up to `max_lag = H - 1`.

Methods
-------
- `diebold_mariano`: Diebold and Mariano (1995) test of equal predictive accuracy, with
  the Harvey, Leybourne and Newbold (1997) small-sample correction and a t distribution
  with n - 1 degrees of freedom. The long-run variance uses the rectangular kernel up to
  `max_lag` (the original test). With few dates that estimate can be zero or negative;
  then the Bartlett (Newey-West) kernel with the same lag is used and the result says so.
  With `max_lag = 0` the statistic equals the one-sample t statistic.
- `block_bootstrap_interval`: circular block bootstrap percentile interval for the mean
  of `d_t`. Blocks of `block_length` adjacent dates keep the short-range correlation.

Both are domain-agnostic: nothing here knows about influenza, WIS or phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "BootstrapInterval",
    "DieboldMarianoResult",
    "KERNEL_BARTLETT",
    "KERNEL_RECTANGULAR",
    "block_bootstrap_interval",
    "diebold_mariano",
]

KERNEL_RECTANGULAR = "rectangular"
KERNEL_BARTLETT = "bartlett"
MIN_DATES = 3  # below this a variance with n - 1 degrees of freedom is meaningless
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult:
    n: int
    mean_delta: float
    statistic: float  # HLN-corrected; negative = candidate better
    p_value: float  # two-sided
    df: int
    max_lag: int
    variance_kernel: str


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    mean_delta: float
    low: float
    high: float
    level: float
    block_length: int
    n_resamples: int
    seed: int


def diebold_mariano(deltas: Sequence[float], max_lag: int) -> DieboldMarianoResult:
    """Two-sided DM test (HLN-corrected) that the mean of `deltas` is zero."""
    d = _as_finite_array(deltas)
    n = len(d)
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError(f"max_lag must be an int >= 0, got {max_lag!r}")
    if max_lag >= n - 1:
        raise ValueError(f"max_lag must be < n - 1 = {n - 1} for {n} dates, got {max_lag}")
    horizon = max_lag + 1
    hln_factor_sq = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if hln_factor_sq <= 0:
        raise ValueError(f"too few dates ({n}) for max_lag {max_lag}: the HLN correction is undefined")

    mean = float(d.mean())
    gammas = _autocovariances(d, max_lag)
    kernel = KERNEL_RECTANGULAR
    variance = gammas[0] + 2.0 * gammas[1:].sum()
    if variance <= 0:
        kernel = KERNEL_BARTLETT
        weights = 1.0 - np.arange(1, max_lag + 1) / (max_lag + 1)
        variance = gammas[0] + 2.0 * (weights * gammas[1:]).sum()
    if variance <= 0:
        if mean != 0.0:
            raise ValueError(f"deltas have zero variance but mean {mean}; the test is undefined for a constant shift")
        return DieboldMarianoResult(n, 0.0, 0.0, 1.0, n - 1, max_lag, kernel)

    statistic = mean / np.sqrt(variance / n) * np.sqrt(hln_factor_sq)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return DieboldMarianoResult(n, mean, float(statistic), p_value, n - 1, max_lag, kernel)


def block_bootstrap_interval(
    deltas: Sequence[float],
    block_length: int,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    level: float = DEFAULT_BOOTSTRAP_LEVEL,
    seed: int = 0,
) -> BootstrapInterval:
    """Circular block bootstrap percentile interval for the mean of `deltas`."""
    d = _as_finite_array(deltas)
    n = len(d)
    if isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length <= n:
        raise ValueError(f"block_length must be an int in [1, {n}], got {block_length!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_length)  # ceil
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length)
    indices = ((starts[:, :, None] + offsets) % n).reshape(n_resamples, -1)[:, :n]
    means = d[indices].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return BootstrapInterval(float(d.mean()), float(low), float(high), level, block_length, n_resamples, seed)


def _as_finite_array(deltas: Sequence[float]) -> np.ndarray:
    d = np.asarray(deltas, dtype=float)
    if d.ndim != 1:
        raise ValueError(f"deltas must be one-dimensional, got shape {d.shape}")
    if len(d) < MIN_DATES:
        raise ValueError(f"need at least {MIN_DATES} dates, got {len(d)}")
    if not np.isfinite(d).all():
        raise ValueError(f"deltas must be finite, got {d[~np.isfinite(d)][:3].tolist()}")
    return d


def _autocovariances(d: np.ndarray, max_lag: int) -> np.ndarray:
    """gamma_k = (1/n) sum_t (d_t - mean)(d_{t-k} - mean), k = 0..max_lag."""
    centered = d - d.mean()
    n = len(d)
    return np.array([centered[k:] @ centered[: n - k] / n for k in range(max_lag + 1)])
