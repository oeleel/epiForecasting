"""Orchestrator for the two-agent improvement loop (Milestone 2).

This module wires together the Analyst (Agent 1) and the Engineer (Agent 2)
into an iterative loop that:

    1. evaluates the current forecast
    2. asks Agent 1 to produce a structured Diagnosis
    3. asks Agent 2 to propose an Action (with the diagnosis + history)
    4. validates the action against the catalog
    5. (optionally) prompts the user to confirm the action — shadow mode
    6. applies the action to a config dict
    7. retrains the model + generates a new forecast
    8. logs the iteration to the SQLite run tracker
    9. decides whether to stop or loop — on a regression (worse than the best
       config seen so far), reverts the working config to that best config
       before the next diagnosis, so the search never builds on a bad edit

Why plain Python and not LangGraph:

    The original spec called for a LangGraph state machine, but for the
    flat evaluate -> diagnose -> propose -> validate -> apply -> retrain
    -> track -> decide flow we have, LangGraph adds a runtime dependency
    without buying any features we use. We deliberately keep this as a
    plain Orchestrator class so:
        - the agent framework adds zero new deps for the loop
        - it's trivial to test with a fake LLM + fake pipeline
        - the Rivanna deployment doesn't have to ship LangGraph wheels
    Each "node" in the original spec is a method on Orchestrator. If we
    later need branching parallelism or persisted state checkpoints, the
    swap to LangGraph is mechanical.

The `Orchestrator` constructor accepts injectable dependencies (LLM client,
pipeline runner, run tracker, action confirmer, knowledge bank). This is the
seam the fake-harness in tests/agent/fakes.py plugs into.

Knowledge bank at the proposal step (design unit 4, injection point B):

    When a `knowledge_bank` is given, every `_propose` call derives a
    `RetrievalContext` from what the loop knows (diagnosed phase = the first
    `weak_segments` entry with dimension "phase", the active model family,
    the target metric), queries the bank, and hands the retrieved entries
    to Agent 2 as a "Known facts" block with the entry ids visible. The
    validated proposal is then annotated in place:

        action["retrieved_entry_ids"]           what Agent 2 was shown
        action["cited_entries"]                 what it says it relied on,
                                                restricted to ids that were
                                                actually shown
        action["supported_by"]                  retrieved entries whose
                                                recommendation names the
                                                proposed action
        action["supported_by_params_mismatch"]  the subset of supported_by
                                                whose recommended params
                                                disagree with the proposal
                                                on a shared key

    What is shown (`_retrieve_for_proposal`): the bank's ranking is curated >
    derived > experiential, so a plain `LIMIT n` cuts the experiential tail
    first and the loop would never see its own past runs. The proposal limit
    is therefore pinned to the store's `DEFAULT_QUERY_LIMIT`, and when the
    matches still exceed it, up to `DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS`
    experiential entries are guaranteed a place by trimming the lowest-ranked
    curated/derived entries instead.

    Guardrails are advisory with an override log (design doc section 6,
    advisor 2026-09-24): a citation of an id that was never shown is dropped
    with a warning, an action that no retrieved recommendation supports is
    logged, a supporting recommendation whose params differ is logged, and
    nothing is ever blocked. With `knowledge_bank=None` the proposal path
    behaves exactly as before the bank existed: no retrieval, no annotation,
    the prompt carries no known-facts section, and a `cited_entries` key the
    LLM emits anyway is dropped so stored actions keep the pre-bank shape.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.domain_adapter import DomainAdapter
from agent.knowledge import (
    DEFAULT_QUERY_LIMIT,
    KNOWN_PHASES,
    PROVENANCES,
    KnowledgeEntry,
    RetrievalContext,
)
from agent.llm_client import LLMClient
from agent.prompt_templates import (
    CITED_ENTRIES_KEY,
    extract_json_from_response,
    format_action_proposal_prompt,
    format_known_facts_block,
    format_structured_diagnosis_prompt,
    validate_action_proposal,
    validate_diagnosis,
)
from agent.run_tracker import RunTracker
from src.config import get_default_config

_log_module = logging.getLogger(__name__)

# How many bank entries the proposal prompt shows per step. Pinned to the
# store's own query limit and it must never be smaller: the store ranks
# curated > derived > experiential, so a smaller proposal limit cuts the
# experiential tail first and Agent 2 never sees the loop's own past runs
# (observed 2026-10-01: 22 matches at peak, 12 shown, all 9 experiential cut).
# The block states how many were omitted when the matches exceed the limit.
DEFAULT_PROPOSAL_FACTS_LIMIT = DEFAULT_QUERY_LIMIT

# Representation guarantee when the limit truncates: up to this many
# experiential entries are always shown, displacing the lowest-ranked
# curated/derived entries, however large the curated layer grows.
DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS = 6
assert 0 < DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS < DEFAULT_PROPOSAL_FACTS_LIMIT, (
    "experiential slots must leave room for curated entries"
)

# The provenance the representation guarantee protects (store trust rank 1).
EXPERIENTIAL_PROVENANCE = "experiential"
assert EXPERIENTIAL_PROVENANCE in PROVENANCES, "provenance name drifted from the schema"

# Model family assumed when the config carries none: the legacy pipeline.
DEFAULT_MODEL_FAMILY = "xgboost_direct"

# Keys the orchestrator writes onto a validated proposal when a bank is set.
RETRIEVED_ENTRY_IDS_KEY = "retrieved_entry_ids"
SUPPORTED_BY_KEY = "supported_by"
SUPPORTED_BY_PARAMS_MISMATCH_KEY = "supported_by_params_mismatch"

# Characters stripped from a cited id before matching it against the shown ids.
_CITATION_BRACKETS = "[]"


# ----------------------------------------------------------------------------
# Protocols (so the orchestrator can be tested with fakes)
# ----------------------------------------------------------------------------

class LLMLike(Protocol):
    def invoke(self, prompt: str) -> str: ...
    def is_available(self) -> bool: ...


class PipelineRunner(Protocol):
    def __call__(
        self,
        config: Dict[str, Any],
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> str: ...


class KnowledgeBankLike(Protocol):
    """The two reads the proposal step needs; `agent.knowledge.KnowledgeBank` satisfies it."""
    def query(self, ctx: RetrievalContext, *, limit: int) -> List[KnowledgeEntry]: ...
    def count_matching(self, ctx: RetrievalContext) -> int: ...


class ActionConfirmer(Protocol):
    """Decides whether to actually apply a proposed action.

    Returns one of:
        "apply"  — go ahead and retrain
        "skip"   — record the proposal but don't apply (move to next iter)
        "stop"   — abort the run entirely
    """
    def __call__(self, iteration: int, action: Dict[str, Any]) -> str: ...


def auto_apply_confirmer(iteration: int, action: Dict[str, Any]) -> str:
    """Default for --auto-apply mode: always apply."""
    return "apply"


def interactive_confirmer(iteration: int, action: Dict[str, Any]) -> str:
    """Shadow mode: prompt the user [y/n/stop] before applying."""
    name = action.get("name", "?")
    params = action.get("params", {})
    rationale = action.get("rationale", "")
    print(f"\n  proposed action: {name}({params})")
    if rationale:
        print(f"  rationale: {rationale}")
    while True:
        ans = input("  apply this action? [y]es / [n]o-skip / [s]top: ").strip().lower()
        if ans in ("y", "yes"):
            return "apply"
        if ans in ("n", "no", "skip"):
            return "skip"
        if ans in ("s", "stop"):
            return "stop"
        print("  please answer y, n, or s")


# ----------------------------------------------------------------------------
# State + result types
# ----------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """One iteration's worth of state, kept in-memory and persisted to SQLite."""
    iteration: int
    config: Dict[str, Any]
    forecast_path: str
    metrics: Dict[str, Any]
    diagnosis: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    action_status: str = "n/a"  # baseline | applied | skipped | no_op | stop | user_stop
    target_value: Optional[float] = None
    elapsed_s: float = 0.0
    change_desc: Optional[str] = None  # adapter.apply_action's human-readable summary


@dataclass
class RunResult:
    run_id: str
    iterations: List[IterationRecord]
    best_iteration: int
    best_forecast_path: str
    stop_reason: str
    target_metric: str

    def to_summary(self) -> Dict[str, Any]:
        baseline = self.iterations[0].target_value if self.iterations else None
        best = self.iterations[self.best_iteration].target_value if self.iterations else None
        delta = (
            (best - baseline) if (baseline is not None and best is not None) else None
        )
        rel = (delta / baseline * 100.0) if (delta is not None and baseline) else None
        return {
            "run_id": self.run_id,
            "n_iterations": len(self.iterations),
            "best_iteration": self.best_iteration,
            "stop_reason": self.stop_reason,
            "target_metric": self.target_metric,
            "baseline": baseline,
            "best": best,
            "absolute_delta": delta,
            "relative_pct": rel,
            "best_forecast_path": self.best_forecast_path,
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _extract_target(metrics: Dict[str, Any], target_metric: str) -> Optional[float]:
    """Pull the target metric value from a metrics dict.

    Looks first at metrics["overall"] then at the top level. Returns None
    if the metric isn't present (the orchestrator's stop logic treats this
    as "no information" rather than crashing).
    """
    overall = metrics.get("overall", metrics) if isinstance(metrics, dict) else {}
    val = overall.get(target_metric)
    if val is None:
        # Try uppercase
        val = overall.get(target_metric.upper())
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _is_better(target_metric: str, current: Optional[float], best: Optional[float]) -> bool:
    """Direction-aware comparison for improvement.

    Lower is better for wis/mape/mae/rmse. Closer-to-0.95 is better for
    coverage_95. Smaller absolute is better for bias.
    """
    if current is None:
        return False
    if best is None:
        return True
    if target_metric == "coverage_95":
        return abs(current - 0.95) < abs(best - 0.95)
    if target_metric == "bias":
        return abs(current) < abs(best)
    return current < best


def _improvement_pct(target_metric: str, current: float, baseline: float) -> float:
    """Return the *fractional* improvement of `current` over `baseline`.

    Positive = improvement. Negative = regression. Always direction-aware.
    """
    if baseline == 0:
        return 0.0
    if target_metric == "coverage_95":
        return (abs(baseline - 0.95) - abs(current - 0.95)) / abs(baseline - 0.95 or 1)
    if target_metric == "bias":
        return (abs(baseline) - abs(current)) / abs(baseline)
    return (baseline - current) / baseline


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

class Orchestrator:
    """Drives the two-agent improvement loop.

    Constructor takes injectable dependencies so the same code path is
    used in production and in unit tests:

        Orchestrator(
            adapter=FluForecastAdapter(),
            llm=LLMClient(...),
            pipeline=adapter.run_pipeline,
            tracker=RunTracker(),
            confirmer=interactive_confirmer,
        ).run(initial_forecast="...", cutoff_date="2024-11-02", max_iterations=5)
    """

    def __init__(
        self,
        adapter: DomainAdapter,
        llm: LLMLike,
        pipeline: PipelineRunner,
        tracker: RunTracker,
        confirmer: ActionConfirmer = auto_apply_confirmer,
        *,
        target_metric: str = "wis",
        max_iterations: int = 5,
        no_improvement_threshold: float = 0.01,
        run_dir: Optional[Path] = None,
        verbose: bool = True,
        knowledge_bank: Optional[KnowledgeBankLike] = None,
    ):
        self.adapter = adapter
        self.llm = llm
        self.pipeline = pipeline
        self.tracker = tracker
        self.confirmer = confirmer
        self.target_metric = target_metric
        self.max_iterations = max_iterations
        self.no_improvement_threshold = no_improvement_threshold
        self.run_dir = run_dir
        self.verbose = verbose
        self.knowledge_bank = knowledge_bank

    # ---------- public entry ------------------------------------------------

    def run(
        self,
        initial_forecast: Optional[str] = None,
        cutoff_date: Optional[str] = None,
        initial_config: Optional[Dict[str, Any]] = None,
        regenerate_baseline: bool = True,
    ) -> RunResult:
        """Execute the loop. Returns a RunResult with full history.

        Args:
            initial_forecast: Optional pre-existing baseline CSV. Only used
                when regenerate_baseline=False. Otherwise the baseline is
                regenerated from initial_config + cutoff_date so that iter 0
                uses the same evaluation window as iter 1, 2, ..., N
                (apples-to-apples comparison).
            cutoff_date: Required. Cutoff for both training and forecasting.
            initial_config: Optional starting config. Defaults to
                src.config.get_default_config().
            regenerate_baseline: If True (default), iter 0 is generated by
                running the pipeline with the default config. If False, the
                file at initial_forecast is used as-is — only correct when
                you can guarantee it was generated at the same cutoff with
                the same default config the loop will start from.
        """
        if cutoff_date is None and initial_config is None:
            raise ValueError("cutoff_date is required (or pass initial_config with data.cutoff_date set)")

        # ---- bookkeeping ---------------------------------------------------
        run_id = self.tracker.start_run(
            initial_forecast=str(initial_forecast) if initial_forecast else "(regenerated)",
            cutoff_date=cutoff_date,
            target_metric=self.target_metric,
        )
        self._log(f"=== run_id: {run_id} ===")

        if self.run_dir is None:
            self.run_dir = Path("outputs/agent_runs") / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(initial_config) if initial_config else get_default_config()
        if cutoff_date is not None:
            config["data"]["cutoff_date"] = cutoff_date

        iterations: List[IterationRecord] = []
        regression_streak = 0
        no_improvement_streak = 0
        stop_reason: str = "unknown"

        # ---- iteration 0 (baseline) ----------------------------------------
        # Regenerate the baseline forecast from the default config so it
        # covers the same forecast window (cutoff + horizons) as every
        # subsequent iteration. Without this, the baseline CSV may cover
        # a different evaluation set than iter 1+, producing misleading
        # "improvements" that are really just easier eval data.
        try:
            if regenerate_baseline:
                self._log(f"\n[iteration 0/{self.max_iterations}] baseline "
                          f"(regenerating with default config @ {cutoff_date})")
                baseline_path = self.run_dir / "iter_0.csv"
                self._log(f"  retraining... -> {baseline_path}")
                actual_baseline_path = self.pipeline(
                    config=config, output_path=str(baseline_path), verbose=False,
                )
            else:
                if initial_forecast is None:
                    raise ValueError("regenerate_baseline=False requires initial_forecast")
                self._log(f"\n[iteration 0/{self.max_iterations}] baseline "
                          f"(using existing {initial_forecast})")
                actual_baseline_path = initial_forecast

            baseline = self._evaluate_iteration(
                iteration=0,
                forecast_path=actual_baseline_path,
                config=config,
            )
        except Exception as e:
            self._log(f"baseline evaluation failed: {e}")
            self.tracker.finish_run(run_id, status="failed")
            raise
        self.tracker.log_iteration(
            run_id, 0,
            metrics=baseline.metrics,
            forecast_path=baseline.forecast_path,
            config=config,
        )
        iterations.append(baseline)
        best_idx = 0  # index into `iterations` of the best config seen so far

        # ---- improvement loop ---------------------------------------------
        for it in range(1, self.max_iterations + 1):
            self._log(f"\n[iteration {it}/{self.max_iterations}]")

            try:
                # ---- 1. diagnose ---------------------------------------
                # Diagnose from the best-known state, not the last-tried one:
                # after a regression `config` has already been reverted below,
                # so this keeps Agent 1/2 reasoning from a regressed baseline.
                diagnosis = self._diagnose(iterations[best_idx])
                self._print_diagnosis(diagnosis)

                # ---- 2. propose --------------------------------------------
                action = self._propose(diagnosis, iterations, config)
                self._print_action(action)

                # ---- 3. handle stop action --------------------------------
                if action.get("name") == "stop":
                    self._log(f"  Agent 2 chose stop: {action.get('rationale', '')}")
                    record = IterationRecord(
                        iteration=it,
                        config=config,
                        forecast_path=iterations[-1].forecast_path,
                        metrics=iterations[-1].metrics,
                        diagnosis=diagnosis,
                        action=action,
                        action_status="stop",
                        target_value=iterations[-1].target_value,
                    )
                    iterations.append(record)
                    self.tracker.log_iteration(
                        run_id, it,
                        metrics=record.metrics,
                        diagnosis=diagnosis,
                        action=action,
                        config=config,
                        forecast_path=record.forecast_path,
                    )
                    stop_reason = "agent_stop"
                    break

                # ---- 4. confirm (shadow mode) ------------------------------
                decision = self.confirmer(it, action)
                if decision == "stop":
                    self._log("  user chose stop")
                    record = IterationRecord(
                        iteration=it, config=config,
                        forecast_path=iterations[-1].forecast_path,
                        metrics=iterations[-1].metrics,
                        diagnosis=diagnosis, action=action,
                        action_status="user_stop",
                        target_value=iterations[-1].target_value,
                    )
                    iterations.append(record)
                    self.tracker.log_iteration(
                        run_id, it, metrics=record.metrics, diagnosis=diagnosis,
                        action=action, config=config, forecast_path=record.forecast_path,
                    )
                    stop_reason = "user_stop"
                    break

                if decision == "skip":
                    self._log("  user skipped action; moving on without applying")
                    record = IterationRecord(
                        iteration=it, config=config,
                        forecast_path=iterations[-1].forecast_path,
                        metrics=iterations[-1].metrics,
                        diagnosis=diagnosis, action=action,
                        action_status="skipped",
                        target_value=iterations[-1].target_value,
                    )
                    iterations.append(record)
                    self.tracker.log_iteration(
                        run_id, it, metrics=record.metrics, diagnosis=diagnosis,
                        action=action, config=config, forecast_path=record.forecast_path,
                    )
                    continue

                # ---- 5. apply action --------------------------------------
                new_config, change_desc = self.adapter.apply_action(action, config)
                self._log(f"  applied: {change_desc}")

                # Detect no-op (value didn't actually change)
                if new_config == config:
                    self._log("  no-op: config unchanged — skipping retrain")
                    record = IterationRecord(
                        iteration=it, config=config,
                        forecast_path=iterations[-1].forecast_path,
                        metrics=iterations[-1].metrics,
                        diagnosis=diagnosis, action=action,
                        action_status="no_op",
                        target_value=iterations[-1].target_value,
                    )
                    iterations.append(record)
                    self.tracker.log_iteration(
                        run_id, it, metrics=record.metrics, diagnosis=diagnosis,
                        action=action, config=config, forecast_path=record.forecast_path,
                    )
                    continue

                # ---- 6. retrain --------------------------------------------
                forecast_path = self._retrain(it, new_config)

                # ---- 7. evaluate new forecast ------------------------------
                record = self._evaluate_iteration(
                    iteration=it,
                    forecast_path=forecast_path,
                    config=new_config,
                    diagnosis=diagnosis,
                    action=action,
                    action_status="applied",
                    change_desc=change_desc,
                )
                iterations.append(record)

                self.tracker.log_iteration(
                    run_id, it,
                    metrics=record.metrics,
                    diagnosis=diagnosis,
                    action=action,
                    config=new_config,
                    forecast_path=forecast_path,
                )

                # ---- 8. stop checks + revert-on-regression -----------------
                # Compare against the best config seen so far, not merely the
                # previous iteration — otherwise a regressed iteration becomes
                # the new floor and the search degrades from there (TS-Agent
                # gap #2: they revert on regression, we used to commit anyway).
                best_target = iterations[best_idx].target_value
                cur_target = record.target_value
                if cur_target is None or best_target is None:
                    self._log("  warning: target metric missing; cannot evaluate progress")
                    continue

                if _is_better(self.target_metric, cur_target, best_target):
                    delta = _improvement_pct(self.target_metric, cur_target, best_target)
                    self._log(f"  improved by {delta*100:+.1f}% on {self.target_metric}")
                    regression_streak = 0
                    best_idx = len(iterations) - 1
                    config = new_config  # commit: this is now the best-known config
                    if delta < self.no_improvement_threshold:
                        no_improvement_streak += 1
                        if no_improvement_streak >= 2:
                            stop_reason = "no_improvement_x2"
                            break
                    else:
                        no_improvement_streak = 0
                else:
                    regression_streak += 1
                    no_improvement_streak = 0
                    # Equal-to-best counts toward the streak (it is not progress)
                    # but is labelled honestly: a stub pipeline or a no-op
                    # action yields "no change", not a regression.
                    outcome = "no change" if cur_target == best_target else "regression"
                    self._log(
                        f"  {outcome} #{regression_streak} on {self.target_metric} "
                        f"(best so far: iter {best_idx} = {best_target:.3f}); "
                        f"reverting to that config for the next proposal"
                    )
                    config = copy.deepcopy(iterations[best_idx].config)
                    if regression_streak >= 2:
                        stop_reason = "regressed_x2"
                        break

            except Exception as e:
                # Pipeline / LLM / validator failure for this iteration
                self._log(f"  iteration {it} failed: {e}")
                self.tracker.log_iteration(
                    run_id, it,
                    metrics={"error": str(e)},
                    diagnosis=None,
                    action=None,
                    config=config,
                )
                stop_reason = f"iteration_error: {e}"
                break

        else:
            stop_reason = "max_iterations"

        # ---- finalize ------------------------------------------------------
        best_idx = self._find_best_iteration(iterations)
        best_path = iterations[best_idx].forecast_path

        # Stamp a stable best_forecast.csv into run_dir
        best_canonical = self.run_dir / "best_forecast.csv"
        try:
            shutil.copyfile(best_path, best_canonical)
        except Exception:
            pass

        self.tracker.finish_run(run_id, status="completed")
        self._log(f"\n=== finished: stop_reason={stop_reason} best_iter={best_idx} ===")

        result = RunResult(
            run_id=run_id,
            iterations=iterations,
            best_iteration=best_idx,
            best_forecast_path=str(best_canonical if best_canonical.exists() else best_path),
            stop_reason=stop_reason,
            target_metric=self.target_metric,
        )

        # The end-of-run report (roadmap 6.1) is the deliverable a reader
        # actually opens, but it is downstream of a finished run: a formatting
        # bug here must never turn a completed run into a failed one.
        # Imported locally so `run_report` can import this module at top level.
        try:
            from agent.run_report import build_report, write_report

            md_path, _json_path = write_report(build_report(result), self.run_dir)
            self._log(f"    report: {md_path}")
        except Exception as exc:
            self._log(f"    (report generation failed: {exc})")

        return result

    # ---------- node implementations (each is straightforward) -------------

    def _evaluate_iteration(
        self,
        iteration: int,
        forecast_path: str,
        config: Dict[str, Any],
        diagnosis: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
        action_status: str = "baseline",
        change_desc: Optional[str] = None,
    ) -> IterationRecord:
        """Compute metrics for a forecast file and wrap as an IterationRecord."""
        t0 = time.time()
        forecasts, actuals = self.adapter.load_data({"forecast_csv": forecast_path})
        metrics = self.adapter.compute_metrics(forecasts, actuals)
        target = _extract_target(metrics, self.target_metric)
        if target is not None:
            self._log(f"  {self.target_metric}={target:.3f}")
        else:
            self._log(f"  {self.target_metric}=<not in metrics>")
        return IterationRecord(
            iteration=iteration,
            config=copy.deepcopy(config),
            forecast_path=str(forecast_path),
            metrics=metrics,
            diagnosis=diagnosis,
            action=action,
            action_status=action_status,
            target_value=target,
            elapsed_s=time.time() - t0,
            change_desc=change_desc,
        )

    def _diagnose(self, current: IterationRecord) -> Dict[str, Any]:
        """Call Agent 1 and return a validated diagnosis dict."""
        prompt = format_structured_diagnosis_prompt(
            current.metrics, self.adapter.get_domain_context(current.config)
        )
        response = self.llm.invoke(prompt)
        try:
            obj = extract_json_from_response(response)
            validate_diagnosis(obj)
            return obj
        except ValueError as e:
            # One repair attempt — Qwen3 8B occasionally returns prose without JSON.
            self._log(
                f"  ! Agent 1 output didn't validate ({e}). "
                f"Retrying with error feedback (this is normal for small LLMs)."
            )
            repair = (
                prompt
                + "\n\n## Previous attempt failed validation\n"
                + f"Error: {e}\n"
                + "Please re-emit a valid JSON object matching the schema exactly. "
                + "Output ONLY the JSON object — no prose, no fences."
            )
            response = self.llm.invoke(repair)
            obj = extract_json_from_response(response)
            validate_diagnosis(obj)
            return obj

    def _propose(
        self, diagnosis: Dict[str, Any], iterations: List[IterationRecord],
        current_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call Agent 2 and return a validated action dict."""
        catalog = self.adapter.get_available_actions(current_config)
        history = [
            {
                "iteration": it.iteration,
                "action": it.action,
                "metrics": {self.target_metric: it.target_value},
                "delta_vs_baseline": (
                    (it.target_value - iterations[0].target_value)
                    if (it.target_value is not None and iterations[0].target_value is not None)
                    else None
                ),
            }
            for it in iterations
        ]
        retrieved: Optional[List[KnowledgeEntry]] = None
        known_facts: Optional[str] = None
        if self.knowledge_bank is not None:
            ctx = self._retrieval_context(diagnosis, current_config)
            retrieved, omitted = self._retrieve_for_proposal(ctx)
            known_facts = format_known_facts_block(retrieved, omitted=omitted)
            _log_module.debug("proposal known-facts block:\n%s", known_facts)
            self._log(
                f"  knowledge: {len(retrieved)} fact(s) retrieved for phase={ctx.phase} "
                f"model={ctx.model} metric={ctx.metric}"
                + (f" ({omitted} omitted)" if omitted > 0 else "")
            )
        prompt = format_action_proposal_prompt(
            diagnosis=diagnosis,
            history=history,
            action_catalog=catalog,
            domain_context=self.adapter.get_domain_context(current_config),
            target_metric=self.target_metric,
            current_config=current_config,
            known_facts=known_facts,
        )
        response = self.llm.invoke(prompt)
        try:
            obj = extract_json_from_response(response)
            validate_action_proposal(obj, catalog)
        except ValueError as e:
            self._log(
                f"  ! Agent 2 output didn't validate ({e}). "
                f"Retrying with error feedback (this is normal for small LLMs)."
            )
            repair = (
                prompt
                + "\n\n## Previous attempt failed validation\n"
                + f"Error: {e}\n"
                + "Please re-emit a valid action JSON object. "
                + "Output ONLY the JSON object — no prose, no fences."
            )
            response = self.llm.invoke(repair)
            obj = extract_json_from_response(response)
            validate_action_proposal(obj, catalog)
        if retrieved is not None:
            self._annotate_citations(obj, retrieved)
        elif CITED_ENTRIES_KEY in obj:
            # No bank: the prompt never mentioned citations, so anything the
            # LLM emits here is noise. Dropping it keeps stored actions in
            # no-bank runs byte-for-byte in the pre-bank shape (A/B control).
            self._log(
                f"  knowledge: no bank wired, dropping stray {CITED_ENTRIES_KEY!r} "
                f"{obj[CITED_ENTRIES_KEY]!r} from the proposal"
            )
            del obj[CITED_ENTRIES_KEY]
        return obj

    def _retrieve_for_proposal(self, ctx: RetrievalContext) -> tuple[List[KnowledgeEntry], int]:
        """Entries to show Agent 2 for `ctx`, and how many matches were left out.

        Rule: query with `DEFAULT_PROPOSAL_FACTS_LIMIT`. If nothing was cut,
        show exactly the store's ranking. If matches were cut, re-query for
        all of them and rebuild the shown list as the highest-ranked
        curated/derived entries plus up to
        `DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS` experiential entries (in store
        order), trimming the lowest-ranked curated/derived entries so the
        list stays within the limit. Slots the experiential side cannot
        fill go back to curated/derived and vice versa, so the list is as
        long as the limit allows. Order is curated/derived first, then
        experiential: the store's own trust order.
        """
        assert self.knowledge_bank is not None
        limit = DEFAULT_PROPOSAL_FACTS_LIMIT
        ranked = self.knowledge_bank.query(ctx, limit=limit)
        total = self.knowledge_bank.count_matching(ctx)
        omitted = max(total - len(ranked), 0)
        if omitted == 0:
            return ranked, 0

        everything = self.knowledge_bank.query(ctx, limit=limit + omitted)
        experiential = [e for e in everything if e.provenance == EXPERIENTIAL_PROVENANCE]
        if not experiential:
            return ranked, omitted
        others = [e for e in everything if e.provenance != EXPERIENTIAL_PROVENANCE]
        n_experiential = min(len(experiential), DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS)
        n_others = min(len(others), limit - n_experiential)
        n_experiential = min(len(experiential), limit - n_others)
        shown = others[:n_others] + experiential[:n_experiential]
        return shown, total - len(shown)

    def _retrieval_context(
        self, diagnosis: Dict[str, Any], current_config: Optional[Dict[str, Any]],
    ) -> RetrievalContext:
        """What the loop knows at proposal time, as the bank's retrieval keys.

        Phase comes from Agent 1's first phase-dimension weak segment. The
        value is LLM output, so an unknown spelling degrades to "no phase
        constraint" with a log line rather than aborting the iteration.
        """
        phase: Optional[str] = None
        for seg in diagnosis.get("weak_segments") or []:
            if seg.get("dimension") != "phase":
                continue
            candidate = str(seg.get("value", "")).strip().lower()
            if candidate in KNOWN_PHASES:
                phase = candidate
            else:
                self._log(
                    f"  knowledge: diagnosed phase {seg.get('value')!r} is not a known phase "
                    f"{list(KNOWN_PHASES)}; retrieving without a phase constraint"
                )
            break
        model_cfg = (current_config or {}).get("model") or {}
        model = model_cfg.get("family") or DEFAULT_MODEL_FAMILY
        return RetrievalContext(phase=phase, model=model, metric=self.target_metric)

    def _annotate_citations(self, action: Dict[str, Any], retrieved: List[KnowledgeEntry]) -> None:
        """Record what was shown, what was cited, and what supports the action (in place).

        Advisory guardrails only (design doc section 6): a cited id that was
        never shown is a hallucination and is dropped with a warning; an
        action no retrieved recommendation supports is logged as an override;
        a supporting recommendation whose params disagree with the proposal
        on a shared key is listed under `supported_by_params_mismatch` and
        logged (it still counts as support for the action name). The action
        itself is never changed or blocked.
        """
        retrieved_ids = [entry.id for entry in retrieved]
        shown = set(retrieved_ids)
        cited: List[str] = []
        for raw in action.get(CITED_ENTRIES_KEY) or []:
            # The prompt shows ids as "[id]"; qwen3:8b echoes the brackets back
            # about half the time, which is a formatting slip, not a hallucination.
            entry_id = raw.strip().strip(_CITATION_BRACKETS).strip()
            if not entry_id or entry_id in cited:
                continue
            if entry_id not in shown:
                msg = (
                    f"Agent 2 cited knowledge entry {entry_id!r} which was not in the "
                    f"retrieved set {retrieved_ids}; dropping the citation (hallucinated)"
                )
                _log_module.warning(msg)
                self._log(f"  ! warning: {msg}")
                continue
            cited.append(entry_id)

        name = action.get("name")
        supported_by = [
            entry.id
            for entry in retrieved
            if entry.recommendation is not None and entry.recommendation.get("action") == name
        ]
        recommended_ids = [entry.id for entry in retrieved if entry.recommendation is not None]
        if recommended_ids and not supported_by and name != "stop":
            self._log(
                f"  knowledge: override - {name!r} is not recommended by any retrieved entry "
                f"{recommended_ids}; advisory only, continuing"
            )

        # Same action name, different params (e.g. the entry says max_depth=6,
        # the proposal says 5): still support for the action, but worth a
        # record, because "followed the rule" and "followed the rule's value"
        # are different claims in the run report. Only keys both sides set are
        # compared; a key the recommendation leaves open is not a disagreement.
        proposed_params = action.get("params") or {}
        params_mismatch: List[str] = []
        for entry in retrieved:
            if entry.id not in supported_by:
                continue
            recommended_params = entry.recommendation.get("params") or {}
            shared_keys = set(recommended_params) & set(proposed_params)
            if any(recommended_params[k] != proposed_params[k] for k in shared_keys):
                params_mismatch.append(entry.id)
        if params_mismatch:
            self._log(
                f"  knowledge: {name!r} is recommended by {params_mismatch} but with different "
                f"params than proposed ({proposed_params}); advisory only, continuing"
            )

        action[CITED_ENTRIES_KEY] = cited
        action[RETRIEVED_ENTRY_IDS_KEY] = retrieved_ids
        action[SUPPORTED_BY_KEY] = supported_by
        action[SUPPORTED_BY_PARAMS_MISMATCH_KEY] = params_mismatch

    def _retrain(self, iteration: int, config: Dict[str, Any]) -> str:
        """Run the pipeline with the new config; return the new forecast path."""
        out_path = self.run_dir / f"iter_{iteration}.csv"
        self._log(f"  retraining... -> {out_path}")
        path = self.pipeline(config=config, output_path=str(out_path), verbose=False)
        return str(path)

    def _find_best_iteration(self, iterations: List[IterationRecord]) -> int:
        """Return the index of the iteration with the best target value."""
        best_idx = 0
        best_val: Optional[float] = iterations[0].target_value
        for i, it in enumerate(iterations[1:], start=1):
            if _is_better(self.target_metric, it.target_value, best_val):
                best_idx = i
                best_val = it.target_value
        return best_idx

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _print_diagnosis(self, diagnosis: Dict[str, Any]) -> None:
        """Render Agent 1's structured diagnosis for the user."""
        if not self.verbose:
            return
        print()
        print("  Agent 1 (Analyst):")
        summary = (diagnosis.get("summary") or "").strip()
        if summary:
            # Wrap long summaries on two lines for readability
            for line in self._wrap(summary, width=66):
                print(f"    {line}")
        focus = diagnosis.get("suggested_focus")
        if focus and focus != "none":
            print(f"    Focus: {focus}")

        weak = diagnosis.get("weak_segments") or []
        if weak:
            print("    Weak segments:")
            for seg in weak[:4]:
                sev = seg.get("severity", "?")
                dim = seg.get("dimension", "?")
                val = seg.get("value", "?")
                metric = seg.get("metric", "?")
                delta = seg.get("delta_vs_overall")
                delta_s = f"{delta:+.2f}" if isinstance(delta, (int, float)) else str(delta)
                print(f"      [{sev:<6}] {dim}={val} ({metric} delta_vs_overall={delta_s})")

        hyps = diagnosis.get("hypotheses") or []
        if hyps:
            print("    Hypotheses:")
            for h in hyps[:3]:
                conf = h.get("confidence")
                conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
                text = h.get("text", "")
                print(f"      ({conf_s}) {text}")

    def _print_action(self, action: Dict[str, Any]) -> None:
        """Render Agent 2's proposed action + reasoning for the user."""
        if not self.verbose:
            return
        print()
        print("  Agent 2 (Engineer):")
        name = action.get("name", "?")
        params = action.get("params") or {}
        if params:
            params_s = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    Action: {name}({params_s})")
        else:
            print(f"    Action: {name}")
        rationale = (action.get("rationale") or "").strip()
        if rationale:
            for line in self._wrap(f"Rationale: {rationale}", width=66):
                print(f"    {line}")
        expected = (action.get("expected_effect") or "").strip()
        if expected:
            for line in self._wrap(f"Expected: {expected}", width=66):
                print(f"    {line}")

    @staticmethod
    def _wrap(text: str, width: int = 66) -> List[str]:
        """Cheap word-wrap for terminal output."""
        words = text.split()
        lines: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for w in words:
            if cur and cur_len + 1 + len(w) > width:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += (1 if cur_len else 0) + len(w)
        if cur:
            lines.append(" ".join(cur))
        return lines
