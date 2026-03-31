"""CLI for the agentic forecasting framework.

Usage:
    python -m agent summarize --forecast-csv outputs/quantile_hindcasts/point_forecasts_nov_apr.csv --dry-run
    python -m agent summarize --cutoff-date 2024-11-02
    python -m agent summarize --forecast-csv path/to/forecasts.csv --base-url http://localhost:8247/v1
    python -m agent summarize --forecast-csv path/to/forecasts.csv --verbose
"""

import argparse
import json
import sys

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.llm_client import LLMClient
from agent.prompt_templates import format_summary_prompt


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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "summarize":
        cmd_summarize(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
