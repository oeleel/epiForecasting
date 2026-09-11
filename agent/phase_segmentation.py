"""Curve-based epidemic phase segmentation (Adiga et al., IAAI-23 / Big Data 2022).

Why this exists
---------------
`agent/phase_evaluator.py` assigns a phase by **calendar month** — onset is
Oct-Nov, peak is Dec-Jan, decline is Feb-Apr. That is a proxy for a property of
the *curve*, and the lab whose method we are building on measures that property
directly. Their own words, in the 2026-09-03 meeting: "we generally know that
during October things start to grow, but the growth rate is not the same every
season."

This module implements their published segmentation so that phase becomes an
observed property of the series rather than an assumption about the date:

    Adiga, Kaur, Wang, Hurt, Porebski, Venkatramanan, Lewis, Marathe.
    "Phase-Informed Bayesian Ensemble Models Improve Performance of COVID-19
    Forecasts." AAAI-23 (IAAI), pp. 15647-15653.        [Algorithms 1, 3]

    Adiga, Kaur, Hurt, Wang, Porebski, Venkatramanan, Lewis, Marathe.
    "Enhancing COVID-19 Ensemble Forecasting Model Performance Using Auxiliary
    Data Sources." IEEE Big Data 2022, pp. 1594-1602.   [Algorithms 1, 3]

Their taxonomy is SURGE / PLATEAU / DECLINE, not ours. Two differences matter:
we have no `plateau` (the regime where their ablation finds AR and Kalman
methods win), and they have no `peak` — for them the peak is the instant
between surge and decline, not a phase you train on.

Design notes
------------
- **Two steps, kept separate.** `piecewise_breakpoints` finds where the trend
  changes; `classify_phases` labels the intervals between them. They are
  independent so either can be swapped — the papers are explicit that phase
  definitions are "subjective" and may be user-annotated or produced by any
  change-point method, which makes both steps legitimate agent-tunable knobs.

- **No `segmented`/R dependency.** The papers fit breakpoints with the R package
  `segmented` (Muggeo 2008). Porting R into this repo to get breakpoints is not
  worth it, so we use bottom-up segmentation on the log1p curve: start with every
  point a segment, repeatedly merge the adjacent pair whose merge adds the least
  squared error, stop at a residual tolerance or a minimum segment length. This
  is a standard piecewise-linear approximation and produces the same *kind* of
  object their Algorithm 1 returns — an ordered breakpoint set. It is a
  deliberate substitution, not a claim of equivalence; `method` records which
  segmenter produced a result so a later `segmented` port stays comparable.

- **Real-time safety is the whole point.** Their Algorithm 1 re-fits each week
  using only data from the most recent two breakpoints onward, so new data
  refines recent phases without rewriting history. `segment_incrementally`
  reproduces that: it freezes every breakpoint before the last two and re-fits
  only the tail. Anything else silently leaks a later revision of the phase
  label into an earlier week's training set.

- **log1p before fitting.** Epidemic curves are multiplicative; a linear fit on
  raw counts puts all its breakpoints near the peak and none in the low season.
  The +-delta classification below is a *ratio* test, so it is applied to the raw
  values regardless of the fitting space.

Their classification rule (Algorithm 3), with delta = 0.10 in both papers:

    surge    if  y[b_{i+1}] >  (1 + delta) * y[b_i]
    decline  if  y[b_{i+1}] <  (1 - delta) * y[b_i]
    plateau  otherwise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_MIN_SEGMENT_WEEKS",
    "DEFAULT_INITIAL_WINDOW",
    "PHASE_SURGE",
    "PHASE_PLATEAU",
    "PHASE_DECLINE",
    "VALID_CURVE_PHASES",
    "PhaseSegment",
    "SegmentationResult",
    "piecewise_breakpoints",
    "classify_phases",
    "segment_series",
    "segment_incrementally",
    "phase_series",
]

_log = logging.getLogger(__name__)

# Their Algorithm 3 threshold: a >=10% change across a segment makes it a
# surge/decline rather than a plateau. Both papers use 10%; it is a knob.
DEFAULT_DELTA = 0.10

# A phase shorter than this is not a phase, it is noise in a weekly series.
DEFAULT_MIN_SEGMENT_WEEKS = 3

# Their Algorithm 1 seeds the recursive fit with t_0 = 15 observations.
DEFAULT_INITIAL_WINDOW = 15

PHASE_SURGE = "surge"
PHASE_PLATEAU = "plateau"
PHASE_DECLINE = "decline"
VALID_CURVE_PHASES = (PHASE_SURGE, PHASE_PLATEAU, PHASE_DECLINE)


@dataclass(frozen=True, slots=True)
class PhaseSegment:
    """One [start, end] run of weeks carrying a single phase label."""

    start_index: int
    end_index: int
    start_date: str
    end_date: str
    start_value: float
    end_value: float
    phase: str
    ratio: float  # end_value / start_value; 1.0 when start_value == 0

    @property
    def n_weeks(self) -> int:
        return self.end_index - self.start_index + 1

    def to_json(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "phase": self.phase,
            "ratio": round(self.ratio, 4),
            "n_weeks": self.n_weeks,
        }


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Breakpoints plus the labelled segments between them."""

    breakpoints: tuple[int, ...]
    segments: tuple[PhaseSegment, ...]
    delta: float
    method: str
    n_observations: int

    @property
    def current_phase(self) -> str | None:
        """The phase of the most recent segment — what a live run is 'in'."""
        return self.segments[-1].phase if self.segments else None

    def phase_at(self, index: int) -> str | None:
        for seg in self.segments:
            if seg.start_index <= index <= seg.end_index:
                return seg.phase
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "breakpoints": list(self.breakpoints),
            "segments": [s.to_json() for s in self.segments],
            "delta": self.delta,
            "method": self.method,
            "n_observations": self.n_observations,
            "current_phase": self.current_phase,
        }


# ---------------------------------------------------------------------------
# Step 1 — breakpoints
# ---------------------------------------------------------------------------

def piecewise_breakpoints(
    values: "np.ndarray | pd.Series | list[float]",
    *,
    min_segment_weeks: int = DEFAULT_MIN_SEGMENT_WEEKS,
    max_breakpoints: int | None = None,
    tolerance: float = 0.02,
    fit_space: Literal["log1p", "raw"] = "log1p",
) -> list[int]:
    """Return interior breakpoint indices of a piecewise-linear fit.

    Bottom-up segmentation: every point starts as its own segment, and the
    adjacent pair whose merge costs the least extra squared error is merged
    repeatedly until no segment is shorter than `min_segment_weeks` and the
    total residual is within `tolerance` of the curve's own scale (or
    `max_breakpoints` is reached).

    Returns interior breakpoints only — the endpoints 0 and n-1 are implicit,
    matching the papers' `{b_1 ... b_m}` convention.
    """
    y = np.asarray(values, dtype=float).ravel()
    n = y.size
    if n == 0:
        raise ValueError("values must be non-empty")
    if min_segment_weeks < 1:
        raise ValueError(f"min_segment_weeks must be >= 1, got {min_segment_weeks}")
    if not np.all(np.isfinite(y)):
        raise ValueError("values must be finite (no NaN/inf); fill gaps before segmenting")
    if n < 2 * min_segment_weeks:
        return []

    fit = np.log1p(np.clip(y, 0.0, None)) if fit_space == "log1p" else y

    # Stop merging once the cheapest available merge would cost more than this
    # fraction of the curve's total variation. Judging the NEXT merge rather
    # than the accumulated residual is what keeps a clean rise-then-fall from
    # collapsing into one segment: both halves are near-perfect lines, so the
    # running residual stays ~0 right up until the merge that destroys them.
    centered = fit - fit.mean()
    total_ss = float(np.dot(centered, centered))
    max_error = (tolerance ** 2) * total_ss if total_ss > 0 else 0.0

    # Segment boundaries as half-open [start, end) index pairs.
    bounds = [(i, i + 1) for i in range(n)]

    def sse(start: int, end: int) -> float:
        """Squared error of a least-squares line through fit[start:end]."""
        m = end - start
        if m <= 2:
            return 0.0
        x = np.arange(m, dtype=float)
        seg = fit[start:end]
        slope, intercept = np.polyfit(x, seg, 1)
        resid = seg - (slope * x + intercept)
        return float(np.dot(resid, resid))

    costs = [sse(a, c) for (a, _), (_, c) in zip(bounds[:-1], bounds[1:])]

    while len(bounds) > 1:
        i = int(np.argmin(costs))
        shortest = min(b - a for a, b in bounds)
        must_merge = shortest < min_segment_weeks
        too_many = max_breakpoints is not None and len(bounds) - 1 > max_breakpoints
        if not must_merge and not too_many and costs[i] > max_error:
            break

        bounds[i] = (bounds[i][0], bounds[i + 1][1])
        del bounds[i + 1]
        del costs[i]
        if i > 0:
            costs[i - 1] = sse(bounds[i - 1][0], bounds[i][1])
        if i < len(costs):
            costs[i] = sse(bounds[i][0], bounds[i + 1][1])

    return [end for _, end in bounds[:-1]]


# ---------------------------------------------------------------------------
# Step 2 — classification (their Algorithm 3)
# ---------------------------------------------------------------------------

def classify_phases(
    values: "np.ndarray | pd.Series | list[float]",
    breakpoints: "list[int] | tuple[int, ...]",
    dates: "pd.Series | list[str] | None" = None,
    *,
    delta: float = DEFAULT_DELTA,
) -> list[PhaseSegment]:
    """Label each inter-breakpoint interval surge / plateau / decline.

    Implements Algorithm 3 of both papers: an interval is a surge when its end
    value exceeds (1 + delta) times its start value, a decline when it falls
    below (1 - delta) times it, and a plateau otherwise. The comparison is a
    ratio on the RAW values regardless of the space the breakpoints were fit in.
    """
    y = np.asarray(values, dtype=float).ravel()
    n = y.size
    if n == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    edges = [0, *sorted(set(int(b) for b in breakpoints)), n - 1]
    edges = sorted(set(e for e in edges if 0 <= e <= n - 1))

    if dates is not None:
        date_strs = [str(d)[:10] for d in pd.to_datetime(pd.Series(list(dates)))]
        if len(date_strs) != n:
            raise ValueError(f"dates has {len(date_strs)} entries but values has {n}")
    else:
        date_strs = [""] * n

    segments: list[PhaseSegment] = []
    for start, end in zip(edges[:-1], edges[1:]):
        y0, y1 = float(y[start]), float(y[end])
        if y0 > 0:
            ratio = y1 / y0
        else:
            # From zero, any positive value is growth; zero-to-zero is a plateau.
            ratio = float("inf") if y1 > 0 else 1.0

        if ratio > 1.0 + delta:
            phase = PHASE_SURGE
        elif ratio < 1.0 - delta:
            phase = PHASE_DECLINE
        else:
            phase = PHASE_PLATEAU

        segments.append(
            PhaseSegment(
                start_index=start,
                end_index=end,
                start_date=date_strs[start],
                end_date=date_strs[end],
                start_value=y0,
                end_value=y1,
                phase=phase,
                ratio=ratio if np.isfinite(ratio) else 999.0,
            )
        )
    return segments


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def segment_series(
    values: "np.ndarray | pd.Series | list[float]",
    dates: "pd.Series | list[str] | None" = None,
    *,
    delta: float = DEFAULT_DELTA,
    min_segment_weeks: int = DEFAULT_MIN_SEGMENT_WEEKS,
    fit_space: Literal["log1p", "raw"] = "log1p",
) -> SegmentationResult:
    """Segment a whole series at once (retrospective use — plots, analysis)."""
    bps = piecewise_breakpoints(
        values, min_segment_weeks=min_segment_weeks, fit_space=fit_space
    )
    segs = classify_phases(values, bps, dates, delta=delta)
    return SegmentationResult(
        breakpoints=tuple(bps),
        segments=tuple(segs),
        delta=delta,
        method=f"bottom_up_{fit_space}",
        n_observations=int(np.asarray(values).size),
    )


def segment_incrementally(
    values: "np.ndarray | pd.Series | list[float]",
    dates: "pd.Series | list[str] | None" = None,
    *,
    delta: float = DEFAULT_DELTA,
    min_segment_weeks: int = DEFAULT_MIN_SEGMENT_WEEKS,
    initial_window: int = DEFAULT_INITIAL_WINDOW,
    fit_space: Literal["log1p", "raw"] = "log1p",
) -> SegmentationResult:
    """Segment the way a live forecaster must: refit only the recent tail.

    Their Algorithm 1 walks forward one observation at a time and re-fits only
    from the **most recent two breakpoints** onward, so that already-settled
    phases never change under a later observation. Anything else lets a phase
    label computed with future data leak into an earlier week's training set.

    Use this whenever the labels will feed training or evaluation at a cutoff.
    Use `segment_series` only for retrospective plots.
    """
    y = np.asarray(values, dtype=float).ravel()
    n = y.size
    if n == 0:
        raise ValueError("values must be non-empty")
    if initial_window < 2 * min_segment_weeks:
        raise ValueError(
            f"initial_window ({initial_window}) must be >= 2*min_segment_weeks "
            f"({2 * min_segment_weeks}) or the first fit cannot produce a segment"
        )
    if n <= initial_window:
        return segment_series(
            y, dates, delta=delta, min_segment_weeks=min_segment_weeks, fit_space=fit_space
        )

    frozen: list[int] = piecewise_breakpoints(
        y[:initial_window], min_segment_weeks=min_segment_weeks, fit_space=fit_space
    )

    for t in range(initial_window + 1, n + 1):
        # Everything before the last two breakpoints is settled and stays put.
        anchor = frozen[-2] if len(frozen) >= 2 else 0
        settled = [b for b in frozen if b < anchor]
        tail = y[anchor:t]
        if tail.size < 2 * min_segment_weeks:
            continue
        tail_bps = piecewise_breakpoints(
            tail, min_segment_weeks=min_segment_weeks, fit_space=fit_space
        )
        kept = list(settled)
        if anchor > 0:
            kept.append(anchor)
        frozen = sorted(set(kept + [anchor + b for b in tail_bps]))
        frozen = [b for b in frozen if 0 < b < n - 1]

    segs = classify_phases(y, frozen, dates, delta=delta)
    return SegmentationResult(
        breakpoints=tuple(frozen),
        segments=tuple(segs),
        delta=delta,
        method=f"incremental_{fit_space}",
        n_observations=n,
    )


def phase_series(result: SegmentationResult, n: int | None = None) -> list[str]:
    """Expand segments into one phase label per observation, P(t) in the papers."""
    size = n if n is not None else result.n_observations
    out = [PHASE_PLATEAU] * size
    for seg in result.segments:
        for i in range(seg.start_index, min(seg.end_index + 1, size)):
            out[i] = seg.phase
    return out
