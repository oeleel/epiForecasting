#!/usr/bin/env python3
"""
Milestone 2 Demo — Two-Agent Improvement Loop
==============================================

End-to-end walkthrough of the two-agent system:
  1. Loads a baseline forecast and computes its phase-aware metrics
  2. Agent 1 (Analyst) reads the metrics and produces a structured Diagnosis
  3. Agent 2 (Engineer) reads the diagnosis + history and proposes one Action
  4. The action is validated against the catalog + guardrails
  5. The pipeline retrains with the mutated config
  6. Loop until improvement converges (max-iters / regression / no-progress)
  7. Final summary + best forecast retained

Modes:
    --fake-pipeline    Skip XGBoost retraining; copy seed forecast each iter.
                       Lets you exercise the full LLM loop in seconds without
                       waiting for the real model. Recommended for the first
                       run-through.
    --auto-apply       Skip the y/n/skip confirmation prompt between iters.
                       Useful for the demo so you can sit back and watch.

Requirements:
    - pip install langchain-openai
    - ollama serve  (or brew services start ollama)
    - ollama pull qwen3:8b

Usage:
    # Recommended first run: fake pipeline + auto-apply, full LLM
    python scripts/demo_milestone2.py --fake-pipeline --auto-apply

    # Real pipeline (slow — actually retrains XGBoost each iter)
    python scripts/demo_milestone2.py --auto-apply

    # Interactive shadow mode (asks before applying each action)
    python scripts/demo_milestone2.py --fake-pipeline
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.llm_client import LLMClient
from agent.orchestrator import (
    Orchestrator,
    auto_apply_confirmer,
    interactive_confirmer,
)
from agent.run_tracker import RunTracker

DEFAULT_BASELINE = (
    PROJECT_ROOT / "outputs" / "quantile_hindcasts" / "quantile_forecasts_nov_apr.csv"
)
DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"


def banner(title: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def step(label: str) -> None:
    print(f"\n--- {label} ---")


def main():
    p = argparse.ArgumentParser(description="Milestone 2 Demo")
    p.add_argument(
        "--forecast-csv",
        default=str(DEFAULT_BASELINE),
        help=f"Baseline forecast CSV (default: {DEFAULT_BASELINE.name})",
    )
    p.add_argument("--cutoff-date", default="2024-11-02")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--target-metric", default="wis",
                   choices=["wis", "mape", "mae", "rmse", "coverage_95", "bias"],
                   help="Optimization target (default: wis — requires a quantile forecast CSV)")
    p.add_argument("--auto-apply", action="store_true",
                   help="Apply each action without asking (default: shadow mode)")
    p.add_argument("--fake-pipeline", action="store_true",
                   help="Stub the pipeline so the loop runs in seconds")
    p.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    p.add_argument("--model", default=DEFAULT_LLM_MODEL)
    args = p.parse_args()

    baseline = Path(args.forecast_csv).resolve()
    if not baseline.exists():
        print(f"\nERROR: baseline forecast not found: {baseline}")
        print("Run `python scripts/generate_forecasts_nov_apr.py` first, or pass "
              "--forecast-csv pointing at an existing forecast CSV.")
        sys.exit(1)

    banner("Milestone 2 Demo: Two-Agent Improvement Loop")
    print()
    print(f"  Baseline forecast: {baseline.name}")
    print(f"  Cutoff date:       {args.cutoff_date}")
    print(f"  Target metric:     {args.target_metric}")
    print(f"  Max iterations:    {args.max_iterations}")
    print(f"  Mode:              "
          f"{'auto-apply' if args.auto_apply else 'shadow (y/n/s prompts)'}, "
          f"{'fake pipeline' if args.fake_pipeline else 'real pipeline'}")
    print(f"  LLM:               {args.model} via {args.base_url}")

    # ---- LLM availability check ---------------------------------------
    step("Checking LLM availability")
    llm = LLMClient(base_url=args.base_url, model=args.model)
    if not llm.is_available():
        print(f"  ERROR: LLM not reachable at {args.base_url}")
        print("  Start Ollama (`brew services start ollama`) and pull the model")
        print(f"  (`ollama pull {args.model}`), then re-run this demo.")
        sys.exit(2)
    print(f"  OK — {args.model} is reachable")

    # ---- Pipeline plumbing ---------------------------------------------
    adapter = FluForecastAdapter()
    tracker = RunTracker()

    if args.fake_pipeline:
        step("Using stub pipeline (each iteration copies the seed forecast)")
        seed = baseline

        def stub(config, output_path=None, verbose=False):
            assert output_path is not None
            shutil.copyfile(seed, output_path)
            return str(output_path)

        pipeline = stub
        print("  Note: this exercises the LLM loop end-to-end without retraining,")
        print("        so iteration metrics will be IDENTICAL across iterations.")
        print("        That's fine for demoing the agent reasoning — to see real")
        print("        improvements, drop --fake-pipeline.")
    else:
        step("Using real pipeline (XGBoost will retrain each iteration)")
        pipeline = adapter.run_pipeline

    confirmer = auto_apply_confirmer if args.auto_apply else interactive_confirmer

    # ---- Orchestrator --------------------------------------------------
    banner("Running the Loop", char="-")
    orch = Orchestrator(
        adapter=adapter,
        llm=llm,
        pipeline=pipeline,
        tracker=tracker,
        confirmer=confirmer,
        target_metric=args.target_metric,
        max_iterations=args.max_iterations,
        verbose=True,
    )

    t0 = time.time()
    try:
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date=args.cutoff_date,
        )
    except Exception as e:
        print(f"\nERROR: orchestrator run failed: {e}")
        raise

    elapsed = time.time() - t0

    # ---- Final report --------------------------------------------------
    banner("Final Report")
    summary = result.to_summary()
    print(f"  run_id:           {summary['run_id']}")
    print(f"  iterations:       {summary['n_iterations']}")
    print(f"  best iteration:   {summary['best_iteration']}")
    print(f"  stop reason:      {summary['stop_reason']}")
    print(f"  target metric:    {summary['target_metric']}")
    if summary["baseline"] is not None and summary["best"] is not None:
        delta = summary["absolute_delta"] or 0.0
        sign = "+" if delta > 0 else ""
        rel = summary["relative_pct"] or 0.0
        print(
            f"  baseline -> best: {summary['baseline']:.3f} -> "
            f"{summary['best']:.3f} ({sign}{delta:.3f}, {rel:+.1f}%)"
        )
    print(f"  best forecast:    {summary['best_forecast_path']}")
    print(f"  total elapsed:    {elapsed:.1f}s")

    # ---- Iteration log -------------------------------------------------
    banner("Iteration Log", char="-")
    for it in result.iterations:
        target_str = f"{it.target_value:.3f}" if it.target_value is not None else "—"
        action = it.action.get("name") if it.action else "(baseline)"
        params = it.action.get("params") if it.action else {}
        print(f"  iter {it.iteration}: {it.action_status:<10} "
              f"{args.target_metric}={target_str:<10} {action} {params}")

    print()
    print("To inspect this run later:")
    print(f"  python -m agent history")
    print(f"  python -m agent compare {summary['run_id']} <other-run-id>")
    print()


if __name__ == "__main__":
    main()
