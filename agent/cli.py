"""CLI for the agentic forecasting framework.

Commands:
    summarize  — Milestone 1: compute metrics + LLM-generated report.
    improve    — Milestone 2: run the two-agent improvement loop.
    history    — list past improve runs from the SQLite tracker.
    compare    — diff two improve runs side-by-side.

Usage examples:
    python -m agent summarize --forecast-csv outputs/.../forecast.csv --dry-run
    python -m agent improve --forecast-csv outputs/.../baseline.csv --max-iterations 3
    python -m agent improve --forecast-csv outputs/.../baseline.csv --auto-apply --fake-pipeline
    python -m agent history --limit 10
    python -m agent compare 20260410-153022-a3f2 20260411-091844-7d10
"""

import argparse
import json
import sys

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.llm_client import LLMClient
from agent.prompt_templates import format_summary_prompt
from agent.run_tracker import RunTracker


def print_metrics_table(metrics: dict) -> None:
    """Print computed metrics as a formatted table (for --dry-run or LLM fallback)."""
    overall = metrics["overall"]

    print("\n" + "=" * 70)
    print("FORECAST PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"\nEvaluation period: {metrics['date_range']['min']} to {metrics['date_range']['max']}")
    print(f"Locations: {metrics['n_locations']}  |  Forecasts: {overall['n_forecasts']}")

    # Overall
    print(f"\n--- Overall ---")
    bias_dir = "over" if overall["bias"] > 0 else "under"
    print(f"  MAPE: {overall['mape']}%  |  MAE: {overall['mae']}  |  RMSE: {overall['rmse']}  |  Bias: {overall['bias']:+.1f} ({bias_dir}-predicting)")

    # By horizon
    print(f"\n--- By Horizon ---")
    print(f"  {'Horizon':<10} {'MAPE':>8} {'MAE':>8} {'Bias':>10} {'N':>6}")
    for h in sorted(metrics["by_horizon"].keys()):
        m = metrics["by_horizon"][h]
        print(f"  Week {h:<5} {m['mape']:>7.1f}% {m['mae']:>8.1f} {m['bias']:>+10.1f} {m['n']:>6}")

    # By phase
    if metrics["by_phase"]:
        print(f"\n--- By Epidemic Phase ---")
        print(f"  {'Phase':<10} {'MAPE':>8} {'MAE':>8} {'Bias':>10} {'N':>6}")
        for phase in ["onset", "peak", "decline"]:
            if phase in metrics["by_phase"]:
                m = metrics["by_phase"][phase]
                print(f"  {phase.capitalize():<10} {m['mape']:>7.1f}% {m['mae']:>8.1f} {m['bias']:>+10.1f} {m['n']:>6}")

    # Worst locations
    if metrics["worst_locations"]:
        print(f"\n--- Worst 5 Locations (by MAPE) ---")
        for loc in metrics["worst_locations"]:
            print(f"  {loc['location']:<25} MAPE={loc['mape']:>6.1f}%  MAE={loc['mae']:>8.1f}  Bias={loc['bias']:>+8.1f}")

    # Best locations
    if metrics["best_locations"]:
        print(f"\n--- Best 5 Locations (by MAPE) ---")
        for loc in metrics["best_locations"]:
            print(f"  {loc['location']:<25} MAPE={loc['mape']:>6.1f}%  MAE={loc['mae']:>8.1f}  Bias={loc['bias']:>+8.1f}")

    # Worst segments
    if metrics["worst_segments"]:
        print(f"\n--- Top 10 Worst Predictions ---")
        for seg in metrics["worst_segments"]:
            print(f"  {seg['location']:<20} {seg['date']}  H{seg['horizon']}  "
                  f"pred={seg['predicted']:>8.1f}  actual={seg['actual']:>8.1f}  "
                  f"error={seg['error']:>+9.1f}")

    print("\n" + "=" * 70)


def cmd_summarize(args) -> None:
    """Run the summarize command: compute metrics and optionally send to LLM."""
    # Build adapter
    adapter = FluForecastAdapter()

    # Build config from args
    config = {}
    if args.forecast_csv:
        config["forecast_csv"] = args.forecast_csv
    elif args.cutoff_date:
        config["cutoff_date"] = args.cutoff_date
    else:
        print("Error: Must provide --forecast-csv or --cutoff-date", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading data...")
    try:
        forecasts, actuals = adapter.load_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Forecasts: {len(forecasts)} rows")
    print(f"  Actuals: {len(actuals)} rows")

    # Compute metrics
    print("Computing metrics...")
    try:
        metrics = adapter.compute_metrics(forecasts, actuals)
    except ValueError as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Always print metrics table
    print_metrics_table(metrics)

    # If --verbose, also dump raw JSON
    if args.verbose:
        print("\n--- Raw Metrics (JSON) ---")
        print(json.dumps(metrics, indent=2, default=str))

    # If --dry-run, stop here
    if args.dry_run:
        print("\n[Dry run — LLM summarization skipped]")
        return

    # Build prompt
    domain_context = adapter.get_domain_context()
    prompt = format_summary_prompt(metrics, domain_context)

    if args.verbose:
        print("\n--- Prompt Sent to LLM ---")
        print(prompt)
        print("--- End Prompt ---\n")

    # Try LLM
    print(f"\nSending to LLM ({args.base_url}, model={args.model})...")
    llm = LLMClient(base_url=args.base_url, model=args.model)

    try:
        response = llm.invoke(prompt)
        print("\n" + "=" * 70)
        print("LLM ANALYSIS")
        print("=" * 70)
        print(response)
        print("=" * 70)
    except ConnectionError as e:
        print(f"\nLLM server not available at {args.base_url}.", file=sys.stderr)
        print("Showing raw metrics only (see table above).", file=sys.stderr)
        print(f"To use LLM summarization, start a local server:", file=sys.stderr)
        print(f"  Ollama: ollama serve & ollama pull {args.model}", file=sys.stderr)
    except ImportError as e:
        print(f"\n{e}", file=sys.stderr)
    except Exception as e:
        print(f"\nLLM error: {e}", file=sys.stderr)
        print("Showing raw metrics only (see table above).", file=sys.stderr)


# ----------------------------------------------------------------------------
# Milestone 2: improve / history / compare
# ----------------------------------------------------------------------------

def cmd_improve(args) -> None:
    """Run the two-agent improvement loop."""
    from agent.orchestrator import (
        Orchestrator,
        auto_apply_confirmer,
        interactive_confirmer,
    )

    adapter = FluForecastAdapter()
    tracker = RunTracker()

    # Pipeline: real one, or a deterministic stub for testing the LLM in
    # isolation. The stub copies the seed forecast each iteration so the
    # loop can run end-to-end without retraining XGBoost.
    if args.fake_pipeline:
        from pathlib import Path
        import shutil

        seed = Path(args.forecast_csv).resolve()

        def fake_pipeline(config, output_path=None, verbose=False):
            assert output_path is not None
            shutil.copyfile(seed, output_path)
            return str(output_path)

        pipeline = fake_pipeline
        print("[--fake-pipeline] using stub that copies the seed forecast each iteration")
    else:
        pipeline = adapter.run_pipeline

    llm = LLMClient(base_url=args.base_url, model=args.model)
    if not args.dry_run:
        problem = llm.diagnose()
        if problem:
            print(f"\n{problem}", file=sys.stderr)
            sys.exit(2)

    confirmer = auto_apply_confirmer if args.auto_apply else interactive_confirmer

    orch = Orchestrator(
        adapter=adapter,
        llm=llm,
        pipeline=pipeline,
        tracker=tracker,
        confirmer=confirmer,
        target_metric=args.target_metric,
        max_iterations=args.max_iterations,
        no_improvement_threshold=args.no_improvement_threshold,
        verbose=True,
    )

    try:
        result = orch.run(
            initial_forecast=args.forecast_csv,
            cutoff_date=args.cutoff_date,
            regenerate_baseline=not args.no_regenerate_baseline,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    summary = result.to_summary()
    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"  run_id:           {summary['run_id']}")
    print(f"  iterations:       {summary['n_iterations']}")
    print(f"  best iteration:   {summary['best_iteration']}")
    print(f"  stop reason:      {summary['stop_reason']}")
    print(f"  target metric:    {summary['target_metric']}")
    if summary['baseline'] is not None and summary['best'] is not None:
        sign = "+" if (summary['absolute_delta'] or 0) > 0 else ""
        print(
            f"  baseline -> best: {summary['baseline']:.3f} -> {summary['best']:.3f} "
            f"({sign}{summary['absolute_delta']:.3f}, "
            f"{(summary['relative_pct'] or 0):+.1f}%)"
        )
    print(f"  best forecast:    {summary['best_forecast_path']}")
    print()
    print("To inspect this run later:")
    print(f"  python -m agent history")
    print(f"  python -m agent compare {summary['run_id']} <other-run-id>")


def cmd_history(args) -> None:
    """List recent improvement runs."""
    tracker = RunTracker()
    runs = tracker.list_runs(limit=args.limit)
    if not runs:
        print("(no runs yet — start one with `python -m agent improve`)")
        return

    print(
        f"{'run_id':<22}  {'started_at':<22}  {'iters':>5}  "
        f"{'best_wis':>10}  {'status':<10}"
    )
    print("-" * 80)
    for r in runs:
        wis = f"{r['best_wis']:.3f}" if r.get('best_wis') is not None else "—"
        print(
            f"{r['run_id']:<22}  {r['started_at']:<22}  "
            f"{r.get('n_iterations', 0):>5}  {wis:>10}  {r['status']:<10}"
        )


def cmd_compare(args) -> None:
    """Diff two runs side by side."""
    tracker = RunTracker()
    try:
        cmp = tracker.compare_runs(args.run_a, args.run_b, metric=args.metric)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nComparing on metric: {cmp['metric']}\n")
    print(f"{'':<25} {'run A':<25} {'run B':<25}")
    print("-" * 75)

    a, b = cmp["a"], cmp["b"]
    print(f"{'run_id':<25} {a['run_id']:<25} {b['run_id']:<25}")
    print(f"{'started_at':<25} {a['started_at']:<25} {b['started_at']:<25}")
    print(f"{'n_iterations':<25} {a['n_iterations']:<25} {b['n_iterations']:<25}")

    best_a = a.get("best") or {}
    best_b = b.get("best") or {}
    print(f"{'best iteration':<25} "
          f"{best_a.get('iteration', '—'):<25} "
          f"{best_b.get('iteration', '—'):<25}")
    print(f"{f'best {args.metric}':<25} "
          f"{best_a.get(args.metric, '—'):<25} "
          f"{best_b.get(args.metric, '—'):<25}")

    delta = cmp.get("delta")
    if delta:
        sign = "+" if delta["absolute"] > 0 else ""
        rel = delta.get("relative_pct")
        print()
        print(
            f"  delta (B - A): {sign}{delta['absolute']:.4f}"
            + (f"  ({rel:+.1f}%)" if rel is not None else "")
        )


# ----------------------------------------------------------------------------
# Top-level argparse
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agentic Forecasting Framework — LLM-powered model analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # summarize subcommand
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Compute evaluation metrics and generate LLM summary report",
    )
    data_group = summarize_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--forecast-csv",
        help="Path to forecast CSV file",
    )
    data_group.add_argument(
        "--cutoff-date",
        help="Find forecast files for this cutoff date (YYYY-MM-DD)",
    )
    summarize_parser.add_argument(
        "--base-url",
        default=None,
        help="LLM server URL (default: env LLM_BASE_URL or http://localhost:11434/v1)",
    )
    summarize_parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: env LLM_MODEL or qwen2.5:7b)",
    )
    summarize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and display metrics without calling the LLM",
    )
    summarize_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full prompt and raw JSON metrics",
    )

    # ---- improve subcommand ------------------------------------------------
    improve_parser = subparsers.add_parser(
        "improve",
        help="Run the two-agent improvement loop on a baseline forecast",
    )
    improve_parser.add_argument(
        "--forecast-csv", default=None,
        help="Optional pre-existing baseline forecast CSV. By default the loop "
             "regenerates the baseline at iteration 0 using the default config "
             "+ --cutoff-date so all iterations evaluate on the same window. "
             "Pass --no-regenerate-baseline to use the file as-is instead.",
    )
    improve_parser.add_argument(
        "--cutoff-date", required=True,
        help="Cutoff date for training and forecasting (YYYY-MM-DD). "
             "Iteration 0 trains the default-config model up to this date and "
             "forecasts the next 4 weeks; every subsequent iteration uses the "
             "same window so improvements are apples-to-apples.",
    )
    improve_parser.add_argument(
        "--no-regenerate-baseline", action="store_true",
        help="Use the file at --forecast-csv as iteration 0 instead of "
             "regenerating it. Only correct if that CSV was generated with the "
             "default config at this exact cutoff_date.",
    )
    improve_parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Maximum number of agent iterations (default: 5)",
    )
    improve_parser.add_argument(
        "--target-metric", default="wis",
        choices=["wis", "mape", "mae", "rmse", "coverage_95", "bias"],
        help="Metric the loop optimizes (default: wis)",
    )
    improve_parser.add_argument(
        "--no-improvement-threshold", type=float, default=0.01,
        help="Fractional improvement below this counts as 'no improvement' (default: 0.01 = 1%%)",
    )
    improve_parser.add_argument(
        "--auto-apply", action="store_true",
        help="Apply each proposed action without asking (default: shadow mode prompts y/n/stop)",
    )
    improve_parser.add_argument(
        "--fake-pipeline", action="store_true",
        help="Use a stub pipeline that copies the seed forecast each iteration "
             "(for testing the LLM end-to-end without retraining XGBoost)",
    )
    improve_parser.add_argument("--base-url", default=None, help="LLM server URL")
    improve_parser.add_argument("--model", default=None, help="LLM model name")
    improve_parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip the LLM availability check (useful with --fake-pipeline + a known-up server)",
    )

    # ---- history subcommand ------------------------------------------------
    history_parser = subparsers.add_parser(
        "history", help="List recent improvement runs from the SQLite tracker",
    )
    history_parser.add_argument(
        "--limit", type=int, default=10, help="Number of runs to show (default: 10)",
    )

    # ---- compare subcommand ------------------------------------------------
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two improvement runs side by side",
    )
    compare_parser.add_argument("run_a", help="First run_id")
    compare_parser.add_argument("run_b", help="Second run_id")
    compare_parser.add_argument(
        "--metric", default="wis",
        choices=["wis", "mape", "coverage_95", "bias"],
        help="Metric to compare on (default: wis)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "improve":
        cmd_improve(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
