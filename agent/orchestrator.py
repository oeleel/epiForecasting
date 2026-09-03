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
pipeline runner, run tracker, action confirmer). This is the seam the
fake-harness in tests/agent/fakes.py plugs into.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.domain_adapter import DomainAdapter
from agent.llm_client import LLMClient
from agent.prompt_templates import (
    extract_json_from_response,
    format_action_proposal_prompt,
    format_structured_diagnosis_prompt,
    validate_action_proposal,
    validate_diagnosis,
)
from agent.run_tracker import RunTracker
from src.config import get_default_config


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
                    self._log(
                        f"  regression #{regression_streak} on {self.target_metric} "
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
        prompt = format_action_proposal_prompt(
            diagnosis=diagnosis,
            history=history,
            action_catalog=catalog,
            domain_context=self.adapter.get_domain_context(current_config),
            target_metric=self.target_metric,
            current_config=current_config,
        )
        response = self.llm.invoke(prompt)
        try:
            obj = extract_json_from_response(response)
            validate_action_proposal(obj, catalog)
            return obj
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
            return obj

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
