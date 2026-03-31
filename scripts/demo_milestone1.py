#!/usr/bin/env python3
"""
Milestone 1 Demo — LLM-Powered Forecast Analysis
==================================================

Demonstrates the agentic framework's ability to:
  1. Load forecast outputs and ground-truth CDC data
  2. Compute phase-aware evaluation metrics (no LLM needed)
  3. Send structured metrics to a local LLM for natural-language diagnosis

Requirements:
  - pip install langchain-openai
  - ollama serve  (running in background)
  - ollama pull qwen3:8b

Usage:
  python scripts/demo_milestone1.py                # full demo with LLM
  python scripts/demo_milestone1.py --no-llm       # metrics only, no Ollama needed
"""

import argparse
import sys
import time
from pathlib import Path

# --- Setup path -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.phase_evaluator import PhaseEvaluator
from agent.prompt_templates import format_summary_prompt

# --- Config ----------------------------------------------------------------
FORECAST_CSV = PROJECT_ROOT / "outputs" / "quantile_hindcasts" / "point_forecasts_nov_apr.csv"
LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1"


def print_header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step: int, total: int, description: str) -> None:
    print(f"\n[Step {step}/{total}] {description}")
    print("-" * 50)


def demo_metrics_only(adapter, forecasts, actuals, metrics):
    """Show what the framework computes without any LLM."""

    print_step(1, 3, "Data Loading")
    print(f"  Forecast file:   {FORECAST_CSV.name}")
    print(f"  Forecast rows:   {len(forecasts):,}")
    print(f"  Actuals rows:    {len(actuals):,}")
    print(f"  Locations:       {metrics['n_locations']}")
    print(f"  Date range:      {metrics['date_range']['min']} to {metrics['date_range']['max']}")

    print_step(2, 3, "Phase-Aware Metrics (computed in Python, no LLM)")

    overall = metrics["overall"]
    bias_dir = "over" if overall["bias"] > 0 else "under"
    print(f"\n  Overall:")
    print(f"    MAPE:  {overall['mape']}%")
    print(f"    MAE:   {overall['mae']} admissions")
    print(f"    RMSE:  {overall['rmse']} admissions")
    print(f"    Bias:  {overall['bias']:+.1f} ({bias_dir}-predicting)")

    print(f"\n  By Forecast Horizon:")
    print(f"    {'Horizon':<12} {'MAPE':>8} {'MAE':>10} {'Bias':>10}")
    for h in sorted(metrics["by_horizon"]):
        m = metrics["by_horizon"][h]
        print(f"    Week {h:<8} {m['mape']:>7.1f}% {m['mae']:>10.1f} {m['bias']:>+10.1f}")

    print(f"\n  By Epidemic Phase:")
    print(f"    {'Phase':<12} {'MAPE':>8} {'MAE':>10} {'Bias':>10}")
    for phase in ["onset", "peak", "decline"]:
        if phase in metrics["by_phase"]:
            m = metrics["by_phase"][phase]
            print(f"    {phase.capitalize():<12} {m['mape']:>7.1f}% {m['mae']:>10.1f} {m['bias']:>+10.1f}")

    print(f"\n  Worst 3 Locations:")
    for loc in metrics["worst_locations"][:3]:
        print(f"    {loc['location']:<25} MAPE={loc['mape']}%")

    print(f"\n  Best 3 Locations:")
    for loc in metrics["best_locations"][:3]:
        print(f"    {loc['location']:<25} MAPE={loc['mape']}%")

    print_step(3, 3, "Key Observations (from metrics alone)")
    print(f"  - Error grows with horizon: Week 1 MAPE={metrics['by_horizon'][1]['mape']}% "
          f"-> Week 4 MAPE={metrics['by_horizon'][4]['mape']}%")
    bias_by_phase = {
        phase: metrics["by_phase"][phase]["bias"]
        for phase in ["onset", "peak", "decline"]
        if phase in metrics["by_phase"]
    }
    worst_phase = max(bias_by_phase, key=lambda p: abs(bias_by_phase[p]))
    print(f"  - Largest bias in {worst_phase} phase: {bias_by_phase[worst_phase]:+.1f} admissions")
    print(f"  - Model systematically {bias_dir}-predicts across all phases")


def demo_with_llm(adapter, metrics):
    """Show the LLM analysis on top of computed metrics."""

    print_step(3, 3, "LLM Analysis")
    print(f"  Sending structured metrics to {LLM_MODEL} via Ollama...")
    print(f"  (The LLM receives pre-computed numbers — it does NO math, only reasoning)\n")

    from agent.llm_client import LLMClient

    # Build the prompt
    domain_context = adapter.get_domain_context()
    prompt = format_summary_prompt(metrics, domain_context)

    # Show prompt size
    print(f"  Prompt size: ~{len(prompt.split()):,} words")

    # Send to LLM
    llm = LLMClient(base_url=LLM_BASE_URL, model=LLM_MODEL)

    start = time.time()
    try:
        response = llm.invoke(prompt)
        elapsed = time.time() - start

        print_header("LLM PERFORMANCE ANALYSIS")
        print(response)
        print("=" * 70)
        print(f"\n  Generated in {elapsed:.1f}s by {LLM_MODEL}")
    except ConnectionError:
        print("  ERROR: Could not connect to Ollama.")
        print("  Make sure 'ollama serve' is running and the model is pulled:")
        print(f"    ollama pull {LLM_MODEL}")
        return
    except ImportError as e:
        print(f"  ERROR: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="Milestone 1 Demo")
    parser.add_argument("--no-llm", action="store_true",
                        help="Show metrics only, skip LLM analysis")
    args = parser.parse_args()

    print_header("Milestone 1 Demo: LLM-Powered Forecast Analysis")
    print()
    print("  This demo shows the agentic framework's current capability:")
    print("  the system computes structured evaluation metrics from forecast")
    print("  outputs, then an LLM interprets the results and writes a")
    print("  diagnostic report for the engineering team.")
    if not args.no_llm:
        print(f"\n  LLM: {LLM_MODEL} via Ollama (localhost)")
    print(f"  Data: CDC FluSight hospital admissions (Nov 2024 - Apr 2025)")

    # Check forecast file exists
    if not FORECAST_CSV.exists():
        print(f"\n  ERROR: Forecast file not found: {FORECAST_CSV}")
        print("  Run the pipeline first: python scripts/generate_forecasts_nov_apr.py")
        sys.exit(1)

    # Load data and compute metrics
    adapter = FluForecastAdapter()
    forecasts, actuals = adapter.load_data({"forecast_csv": str(FORECAST_CSV)})
    metrics = adapter.compute_metrics(forecasts, actuals)

    # Show metrics
    demo_metrics_only(adapter, forecasts, actuals, metrics)

    # LLM analysis
    if args.no_llm:
        print_header("NEXT STEP")
        print("  To see the LLM analysis, run without --no-llm:")
        print("    python scripts/demo_milestone1.py")
        print("  Requires: ollama serve + ollama pull qwen3:8b")
    else:
        demo_with_llm(adapter, metrics)

    print_header("WHAT'S NEXT: Milestone 2")
    print("  Currently the LLM only writes a report. In Milestone 2, it will:")
    print("  1. Diagnose WHY specific segments underperform")
    print("  2. Choose an action (adjust hyperparams, reweight samples, etc.)")
    print("  3. The system applies the action and retrains automatically")
    print("  4. The LLM evaluates improvement and iterates")
    print()


if __name__ == "__main__":
    main()
