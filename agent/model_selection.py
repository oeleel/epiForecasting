"""Model selection stage (TS-Agent Stage 1) over the model bank.

Before the refinement loop (Milestone 2) spends iterations tuning one model,
this stage answers "which model should we even start from?":

    1. warm-up  - fit + predict every candidate family at each cutoff of the
                  pinned evaluation window (rolling origin, no leakage)
    2. score    - concatenate each family's forecasts and run the same
                  PhaseEvaluator the loop uses (WIS, coverage, bias, by phase)
    3. select   - rank by the goal metric and hand the incumbent's config to
                  the orchestrator

Design notes
------------
- Goal-as-parameter (advisor, 09-02): the ranking metric and the phase it is
  read from are arguments (`SelectionGoal`), so "best at peak" vs "best
  overall" is a flag, not a code change. A natural-language front end can map
  onto the same struct later.
- The CDC frame is loaded once and sliced per cutoff. Families are fit fresh
  at every cutoff (retrain-from-scratch); the warm-start layer (roadmap
  workstream 4) will add a resume path next to this one.
- One family failing (missing dependency, a NaN forecast) is recorded on its
  `CandidateResult.error` and does not abort the comparison. That is the
  error boundary that keeps a broken in-house model from hiding the others.
- Everything returned is JSON-serializable so `select-model --json` can feed
  the run reports (workstream 6) and the paper's experiment log.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from agent.phase_evaluator import PhaseEvaluator
from src import config as repo_config
from src.model_bank.runner import fit_predict, load_history

__all__ = [
    "render_markdown",
    "CandidateResult",
    "DEFAULT_GOAL",
    "LOWER_IS_BETTER_METRICS",
    "SelectionGoal",
    "SelectionResult",
    "evaluate_candidates",
    "goal_value",
    "select_incumbent",
]

_log = logging.getLogger(__name__)

# Metrics where a smaller number wins. coverage_95 is scored by distance to 0.95
# and bias by absolute value (same rules as agent.orchestrator._is_better).
LOWER_IS_BETTER_METRICS = frozenset({"wis", "mape", "mae", "rmse"})
VALID_METRICS = tuple(sorted(LOWER_IS_BETTER_METRICS | {"coverage_95", "bias"}))
COVERAGE_TARGET = 0.95
ALL_PHASES = "all"
VALID_PHASES = (ALL_PHASES, "onset", "peak", "decline")


@dataclass(frozen=True, slots=True)
class SelectionGoal:
    """What "best" means for this run. metric read from `phase` ("all" = overall)."""

    metric: str = "wis"
    phase: str = ALL_PHASES

    def __post_init__(self) -> None:
        if self.metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {list(VALID_METRICS)}, got {self.metric!r}")
        if self.phase not in VALID_PHASES:
            raise ValueError(f"phase must be one of {VALID_PHASES}, got {self.phase!r}")

    def describe(self) -> str:
        where = "overall" if self.phase == ALL_PHASES else f"{self.phase} phase"
        return f"{self.metric} ({where})"


DEFAULT_GOAL = SelectionGoal()


@dataclass(slots=True)
class CandidateResult:
    family: str
    params: dict[str, Any]
    n_cutoffs: int
    n_forecast_rows: int
    fit_seconds_total: float
    metrics: dict[str, Any] = field(default_factory=dict)  # PhaseEvaluator output
    error: str | None = None
    forecasts: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("forecasts")
        return out


@dataclass(slots=True)
class SelectionResult:
    goal: SelectionGoal
    cutoffs: list[str]
    candidates: list[CandidateResult]
    incumbent: str | None
    ranking: list[dict[str, Any]]  # [{family, value}] best first

    def to_json(self) -> dict[str, Any]:
        return {
            "goal": asdict(self.goal),
            "cutoffs": list(self.cutoffs),
            "incumbent": self.incumbent,
            "ranking": list(self.ranking),
            "candidates": [c.to_json() for c in self.candidates],
        }


def goal_value(metrics: dict[str, Any], goal: SelectionGoal) -> float | None:
    """Read the goal metric out of a PhaseEvaluator metrics dict; None if absent."""
    if goal.phase == ALL_PHASES:
        section = metrics.get("overall") or {}
    else:
        section = (metrics.get("by_phase") or {}).get(goal.phase) or {}
    value = section.get(goal.metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_key(value: float, goal: SelectionGoal) -> float:
    if goal.metric == "coverage_95":
        return abs(value - COVERAGE_TARGET)
    if goal.metric == "bias":
        return abs(value)
    return value


def evaluate_candidates(
    families: Sequence[str],
    cutoffs: Sequence[str],
    actuals: pd.DataFrame,
    params_by_family: dict[str, dict[str, Any]] | None = None,
    locations: Sequence[str] | None = None,
    train_start_date: str = repo_config.TRAIN_START_DATE,
    horizon: int = repo_config.FORECAST_HORIZON,
    cdc: pd.DataFrame | None = None,
    exclude_locations: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[CandidateResult]:
    """Rolling-origin warm-up of every family over `cutoffs`, scored with PhaseEvaluator.

    `actuals` is the CDC frame used as ground truth (usually the same frame
    as `cdc`, which the history is sliced from). Both are passed explicitly
    so tests can run on synthetic data with no disk access.
    """
    if not families:
        raise ValueError("families must be a non-empty sequence")
    if not cutoffs:
        raise ValueError("cutoffs must be a non-empty sequence")
    params_by_family = params_by_family or {}
    excluded = {str(x) for x in (exclude_locations or ())}
    say = progress or (lambda _msg: None)

    if cdc is None:
        from src.data_loader import FluDataLoader

        with contextlib.redirect_stdout(io.StringIO()):
            cdc = FluDataLoader().fetch_data()

    results: list[CandidateResult] = []
    for family in families:
        params = dict(params_by_family.get(family) or {})
        frames: list[pd.DataFrame] = []
        fit_total = 0.0
        error: str | None = None
        t0 = time.perf_counter()
        for cutoff in cutoffs:
            try:
                history = load_history(
                    cutoff, train_start_date=train_start_date, locations=locations, cdc=cdc
                )
                csv_frame, meta = fit_predict(family, params, history, cutoff, horizon=horizon)
            except Exception as exc:  # error boundary: one family must not sink the comparison
                error = f"{type(exc).__name__} at cutoff {cutoff}: {exc}"
                _log.warning("candidate %s failed: %s", family, error)
                break
            fit_total += meta.fit_seconds
            frames.append(csv_frame)
            say(f"  {family:<20s} {cutoff}  fit {meta.fit_seconds:5.1f}s")

        if error is not None or not frames:
            results.append(CandidateResult(
                family=family, params=params, n_cutoffs=len(frames), n_forecast_rows=0,
                fit_seconds_total=fit_total, error=error or "no forecasts produced",
            ))
            continue

        forecasts = pd.concat(frames, ignore_index=True)
        if excluded:
            forecasts = forecasts[~forecasts["location"].astype(str).isin(excluded)]
        try:
            metrics = _score(forecasts, actuals)
        except Exception as exc:
            results.append(CandidateResult(
                family=family, params=params, n_cutoffs=len(frames),
                n_forecast_rows=len(forecasts), fit_seconds_total=fit_total,
                error=f"scoring failed: {exc}", forecasts=forecasts,
            ))
            continue
        results.append(CandidateResult(
            family=family, params=params, n_cutoffs=len(frames),
            n_forecast_rows=len(forecasts), fit_seconds_total=fit_total,
            metrics=metrics, forecasts=forecasts,
        ))
        say(f"  {family:<20s} done in {time.perf_counter() - t0:.1f}s")
    return results


def select_incumbent(
    candidates: Sequence[CandidateResult],
    goal: SelectionGoal = DEFAULT_GOAL,
    cutoffs: Sequence[str] = (),
) -> SelectionResult:
    """Rank scored candidates by the goal and name the incumbent (None if nothing scored)."""
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        if not cand.ok:
            continue
        value = goal_value(cand.metrics, goal)
        if value is None:
            continue
        scored.append((cand.family, value))
    scored.sort(key=lambda fv: _sort_key(fv[1], goal))
    ranking = [{"family": f, "value": v} for f, v in scored]
    return SelectionResult(
        goal=goal,
        cutoffs=list(cutoffs),
        candidates=list(candidates),
        incumbent=scored[0][0] if scored else None,
        ranking=ranking,
    )


def _score(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, Any]:
    merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)
    return {
        "overall": PhaseEvaluator.compute_overall_metrics(merged),
        "by_phase": PhaseEvaluator.evaluate_by_phase(merged),
        "by_horizon": PhaseEvaluator.evaluate_by_horizon(merged),
        "n_scored": int(len(merged)),
    }


def render_markdown(result: SelectionResult, title: str = "Model selection") -> str:
    """Human-legible summary of a SelectionResult (the professor-facing view of the JSON).

    One table, best first, plus the failures and the incumbent. Kept
    deliberately plain so it reads in a terminal, a PR, or a lab notebook.
    """
    lines = [f"# {title}", ""]
    if result.cutoffs:
        lines.append(
            f"- cutoffs: {len(result.cutoffs)} ({result.cutoffs[0]} .. {result.cutoffs[-1]})"
        )
    lines.append(f"- goal: {result.goal.describe()}")
    lines.append(f"- incumbent: **{result.incumbent or 'none'}**")
    lines += ["", "| family | WIS | MAPE | cov95 | onset WIS | peak WIS | decline WIS | fit s |",
              "|---|---|---|---|---|---|---|---|"]
    ranked = [r["family"] for r in result.ranking]
    ordered = sorted(
        (c for c in result.candidates if c.ok),
        key=lambda c: ranked.index(c.family) if c.family in ranked else len(ranked),
    )

    def _num(value: Any, digits: int = 1) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"

    for cand in ordered:
        overall = cand.metrics.get("overall", {})
        phases = cand.metrics.get("by_phase", {})
        mark = " (incumbent)" if cand.family == result.incumbent else ""
        lines.append(
            f"| {cand.family}{mark} | {_num(overall.get('wis'))} | {_num(overall.get('mape'))} "
            f"| {_num(overall.get('coverage_95'), 2)} "
            f"| {_num((phases.get('onset') or {}).get('wis'))} "
            f"| {_num((phases.get('peak') or {}).get('wis'))} "
            f"| {_num((phases.get('decline') or {}).get('wis'))} "
            f"| {_num(cand.fit_seconds_total)} |"
        )
    failed = [c for c in result.candidates if not c.ok]
    if failed:
        lines += ["", "Failed:", ""]
        lines += [f"- {c.family}: {c.error}" for c in failed]
    return "\n".join(lines) + "\n"
