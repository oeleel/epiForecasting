"""CLI for the agentic forecasting framework.

Commands:
    check-data — Pre-training data quality checks on raw CDC data.
    summarize  — Compute metrics + LLM-generated performance report.
    improve    — Run the two-agent improvement loop.
    history    — List past improve runs from the SQLite tracker.
    status     — Show detailed iteration-by-iteration view of a run.
    compare    — Diff two improve runs side-by-side.
    report     — Regenerate the end-of-run report for a past run.
    list-models  — Show every model-bank family available in this environment.
    select-model — Stage 1: warm-up every candidate family on the pinned
                   eval window, score with phase-aware WIS, name the incumbent.

Usage examples:
    python -m agent check-data --cutoff-date 2024-11-02 --dry-run
    python -m agent summarize --forecast-csv outputs/.../forecast.csv --dry-run
    python -m agent improve --cutoff-date 2024-11-02 --auto-apply
    python -m agent improve --cutoff-date 2024-11-02 --auto-apply --fake-pipeline
    python -m agent history
    python -m agent status 20260410-153022-a3f2
    python -m agent compare 20260410-153022-a3f2 20260411-091844-7d10
    python -m agent report 20260903-145933-0732
    python -m agent list-models
    python -m agent select-model --stride-weeks 4
    python -m agent select-model --families persistence sf_autoets mlf_lightgbm --metric wis --phase peak
    python -m agent improve --cutoff-date 2025-12-06 --model-family mlf_lightgbm --auto-apply
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


def cmd_check_data(args) -> None:
    """Run pre-training data quality checks on raw CDC data."""
    from agent.data_quality import DataQualityChecker, format_data_quality_prompt

    adapter = FluForecastAdapter()

    # Load raw actuals
    print("Loading raw data...")
    try:
        data = adapter._load_actuals()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter out excluded locations
    if args.exclude_locations:
        exclude = set(args.exclude_locations)
        data = data[~data["location"].astype(str).isin(exclude)]
        print(f"  Excluded locations: {', '.join(sorted(exclude))}")

    print(f"  Rows: {len(data)}  |  Locations: {data['location'].nunique()}")

    # Run checks
    print(f"Running data quality checks (cutoff={args.cutoff_date})...")
    checker = DataQualityChecker(args.cutoff_date)
    report = checker.run(data)

    # Print report
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(f"\n  Cutoff:    {report.cutoff_date}")
    print(f"  Range:     {report.date_range['min']} to {report.date_range['max']}")
    print(f"  Locations: {report.n_locations}")
    print(f"  Rows:      {report.n_rows}")

    stats = report.summary_stats
    if stats:
        print(f"\n  Recent {checker.RECENT_WINDOW_WEEKS} weeks — "
              f"mean: {stats.get('recent_mean', '?')}, "
              f"median: {stats.get('recent_median', '?')}, "
              f"max: {stats.get('recent_max', '?')}")
        print(f"  Null values: {stats.get('null_values', 0)}  |  "
              f"Zero values: {stats.get('zero_values', 0)}")

    if not report.issues:
        print("\n  No issues found — data looks clean.")
    else:
        print(f"\n  Issues: {report.critical_count} critical, {report.warning_count} warnings")
        for severity in ["critical", "warning", "info"]:
            group = [i for i in report.issues if i.severity == severity]
            if not group:
                continue
            print(f"\n  --- {severity.upper()} ---")
            for issue in group[:20]:
                loc = f" [{issue.location}]" if issue.location else ""
                date = f" ({issue.date})" if issue.date else ""
                print(f"    [{issue.check}]{loc}{date}: {issue.detail}")
            if len(group) > 20:
                print(f"    ... and {len(group) - 20} more")

    print("\n" + "=" * 70)

    if args.verbose:
        print("\n--- Raw Report (JSON) ---")
        print(json.dumps(report.to_dict(), indent=2, default=str))

    if args.dry_run:
        print("\n[Dry run — LLM analysis skipped]")
        return

    # LLM interpretation
    domain_context = adapter.get_domain_context()
    prompt = format_data_quality_prompt(report, domain_context)

    if args.verbose:
        print("\n--- Prompt Sent to LLM ---")
        print(prompt)
        print("--- End Prompt ---\n")

    llm = LLMClient(base_url=args.base_url, model=args.model)
    print(f"\nSending to LLM ({llm.base_url}, model={llm.model})...")

    try:
        response = llm.invoke(prompt)
        print("\n" + "=" * 70)
        print("LLM DATA QUALITY ANALYSIS")
        print("=" * 70)
        print(response)
        print("=" * 70)
    except ConnectionError:
        print(f"\nLLM server not available. Showing raw report only (see above).",
              file=sys.stderr)
    except Exception as e:
        print(f"\nLLM error: {e}", file=sys.stderr)


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

    # Filter out excluded locations
    if hasattr(args, 'exclude_locations') and args.exclude_locations:
        exclude = set(args.exclude_locations)
        forecasts = forecasts[~forecasts["location"].astype(str).isin(exclude)]
        print(f"  Excluded locations: {', '.join(sorted(exclude))}")

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
    llm = LLMClient(base_url=args.base_url, model=args.model)
    print(f"\nSending to LLM ({llm.base_url}, model={llm.model})...")

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

    adapter = FluForecastAdapter(exclude_locations=args.exclude_locations)
    tracker = RunTracker()

    if args.exclude_locations:
        print(f"[excluding locations: {', '.join(args.exclude_locations)}]")

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

    initial_config = None
    if args.model_family:
        from src.config import get_default_config
        from src.model_bank.registry import resolve_family

        resolve_family(args.model_family)  # fail fast on an unknown family
        initial_config = get_default_config()
        initial_config["model"]["family"] = args.model_family
        print(f"[model family: {args.model_family}]")

    try:
        result = orch.run(
            initial_forecast=args.forecast_csv,
            cutoff_date=args.cutoff_date,
            initial_config=initial_config,
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

    # The orchestrator already wrote report.md/report.json into the run dir;
    # point at them rather than rebuilding a second, possibly divergent copy.
    from pathlib import Path as _Path

    run_dir = _Path(summary["best_forecast_path"]).parent
    print(f"  report:           {run_dir / 'report.md'}")

    if args.report:
        from agent.run_report import build_report, render_markdown

        out = _Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(build_report(result)))
        print(f"wrote {out}")

    print()
    print("To inspect this run later:")
    print(f"  python -m agent history")
    print(f"  python -m agent compare {summary['run_id']} <other-run-id>")


def cmd_list_models(args) -> None:
    """Print every model-bank family and whether it is importable here."""
    from src.model_bank.registry import list_families

    print(f"\n{'family':<22} {'ok':<4} {'warm':<5} description")
    print("-" * 78)
    for info in list_families():
        ok = "yes" if info.available else "no"
        warm = "yes" if info.supports_warm_start else "-"
        desc = info.description if info.available else info.reason
        print(f"{info.family:<22} {ok:<4} {warm:<5} {desc}")
    print()
    print("In-house models: pass a dotted path as the family, e.g. "
          "--model-family my_lab.models:FluLSTM (see documentation/MODEL_BANK.md).")


DEFAULT_SELECT_FAMILIES = ["persistence", "seasonal_naive", "xgboost_direct"]


def cmd_select_model(args) -> None:
    """Stage 1: warm-up + score every candidate on the pinned split; pick the incumbent."""
    import json as _json
    from pathlib import Path as _Path

    from agent.model_selection import SelectionGoal, evaluate_candidates, select_incumbent
    from src import config as repo_config
    from src.model_bank.registry import resolve_family

    families = args.families or DEFAULT_SELECT_FAMILIES
    for fam in families:
        resolve_family(fam)  # fail fast with the registry's message

    if args.cutoffs:
        cutoffs = list(args.cutoffs)
    else:
        cutoffs = repo_config.generate_eval_cutoffs(stride_weeks=args.stride_weeks)
        if args.max_cutoffs:
            cutoffs = cutoffs[: args.max_cutoffs]
    goal = SelectionGoal(metric=args.metric, phase=args.phase)

    print(f"\nselect-model: {len(families)} families x {len(cutoffs)} cutoffs "
          f"({cutoffs[0]} .. {cutoffs[-1]}), goal = {goal.describe()}")
    print(f"train window starts {repo_config.TRAIN_START_DATE}"
          + (f"; locations {args.locations}" if args.locations else "")
          + (f"; excluding {args.exclude_locations}" if args.exclude_locations else ""))
    print()

    adapter = FluForecastAdapter()
    actuals = adapter._load_actuals()
    candidates = evaluate_candidates(
        families=families,
        cutoffs=cutoffs,
        actuals=actuals,
        locations=args.locations,
        exclude_locations=args.exclude_locations,
        progress=print if not args.quiet else None,
    )
    result = select_incumbent(candidates, goal=goal, cutoffs=cutoffs)

    print()
    print("=" * 96)
    print("MODEL SELECTION")
    print("=" * 96)
    header = (f"{'family':<18} {'wis':>9} {'mape':>7} {'cov95':>6} "
              f"{'onset':>9} {'peak':>9} {'decline':>9} {'fit s':>7}  status")
    print(header)
    print("-" * 96)
    for cand in result.candidates:
        if not cand.ok:
            print(f"{cand.family:<18} {'-':>9} {'-':>7} {'-':>6} {'-':>9} {'-':>9} {'-':>9} "
                  f"{cand.fit_seconds_total:>7.1f}  FAILED: {cand.error}")
            continue
        o = cand.metrics.get("overall", {})
        ph = cand.metrics.get("by_phase", {})

        def _w(phase):
            v = (ph.get(phase) or {}).get("wis")
            return f"{v:>9.1f}" if isinstance(v, (int, float)) else f"{'-':>9}"

        cov = o.get("coverage_95")
        cov_s = f"{cov:>6.2f}" if isinstance(cov, (int, float)) else f"{'-':>6}"
        wis = o.get("wis")
        wis_s = f"{wis:>9.1f}" if isinstance(wis, (int, float)) else f"{'-':>9}"
        mark = "  <- incumbent" if cand.family == result.incumbent else ""
        print(f"{cand.family:<18} {wis_s} {o.get('mape', float('nan')):>7.1f} {cov_s} "
              f"{_w('onset')} {_w('peak')} {_w('decline')} {cand.fit_seconds_total:>7.1f}  ok{mark}")
    print()
    if result.incumbent is None:
        print("no candidate produced a scoreable forecast; nothing selected")
    else:
        print(f"incumbent by {goal.describe()}: {result.incumbent}")
        print("next: refine it with the improvement loop, e.g.")
        print(f"  python -m agent improve --cutoff-date {cutoffs[-1]} "
              f"--model-family {result.incumbent} --auto-apply")

    if args.json:
        out = _Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(result.to_json(), indent=2, default=str))
        print(f"\nwrote {out}")
    if args.report:
        from agent.model_selection import render_markdown

        out = _Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(result))
        print(f"wrote {out}")
    if args.save_forecasts:
        out_dir = _Path(args.save_forecasts)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cand in result.candidates:
            if cand.forecasts is not None:
                cand.forecasts.to_csv(out_dir / f"{cand.family}.csv", index=False)
        print(f"wrote per-family forecasts to {out_dir}/")


def cmd_history(args) -> None:
    """List recent improvement runs."""
    tracker = RunTracker()
    runs = tracker.list_runs(limit=args.limit)
    if not runs:
        print("(no runs yet — start one with `python -m agent improve`)")
        return

    print(f"\n{'run_id':<22}  {'cutoff':<12}  {'iters':>5}  "
          f"{'best_wis':>10}  {'status':<10}  {'started':<20}")
    print("-" * 90)
    for r in runs:
        wis = f"{r['best_wis']:.3f}" if r.get('best_wis') is not None else "—"
        cutoff = r.get('cutoff_date', '—') or '—'
        raw_ts = r.get('started_at', '—')
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(raw_ts).astimezone()
            started = dt.strftime("%b %d %I:%M %p")  # e.g. "Apr 22 2:15 PM"
        except Exception:
            started = raw_ts[:19]
        print(
            f"{r['run_id']:<22}  {cutoff:<12}  "
            f"{r.get('n_iterations', 0):>5}  {wis:>10}  "
            f"{r['status']:<10}  {started:<20}"
        )
    print()


def cmd_status(args) -> None:
    """Show detailed status of a specific run."""
    tracker = RunTracker()
    run = tracker.get_run(args.run_id)
    if run is None:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"RUN: {run['run_id']}")
    print(f"{'=' * 70}")
    print(f"  Status:         {run['status']}")
    print(f"  Cutoff date:    {run.get('cutoff_date', '—')}")
    print(f"  Target metric:  {run.get('target_metric', 'wis')}")
    print(f"  Started:        {run.get('started_at', '—')}")
    print(f"  Finished:       {run.get('finished_at', '—') or '(still running)'}")
    print(f"  Iterations:     {len(run.get('iterations', []))}")

    iterations = run.get("iterations", [])
    if not iterations:
        print("\n  (no iterations recorded)")
        return

    print(f"\n  {'iter':>4}  {'wis':>10}  {'mape':>8}  {'bias':>8}  {'action':<35}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*35}")

    for it in iterations:
        i = it.get("iteration", "?")
        wis = f"{it['wis']:.3f}" if it.get('wis') is not None else "—"
        mape = f"{it['mape']:.1f}%" if it.get('mape') is not None else "—"
        bias = f"{it['bias']:+.1f}" if it.get('bias') is not None else "—"

        action_data = it.get("action")
        if action_data is None:
            action_str = "(baseline)"
        elif isinstance(action_data, dict):
            name = action_data.get("name", "?")
            params = action_data.get("params", {})
            if params:
                params_s = ", ".join(f"{k}={v}" for k, v in params.items())
                action_str = f"{name}({params_s})"
            else:
                action_str = name
        else:
            action_str = str(action_data)[:35]

        print(f"  {i:>4}  {wis:>10}  {mape:>8}  {bias:>8}  {action_str:<35}")

    # Best iteration
    best = tracker.get_best_iteration(args.run_id, metric=run.get("target_metric", "wis"))
    if best:
        metric = run.get("target_metric", "wis")
        print(f"\n  Best: iteration {best.get('iteration')} "
              f"({metric}={best.get(metric, '?')})")

    # Show config details if requested
    if hasattr(args, 'show_config') and args.show_config:
        for it in iterations:
            i = it.get("iteration", "?")
            cfg = it.get("config")
            if cfg is None:
                continue
            print(f"\n  --- Iteration {i} config ---")
            xgb = cfg.get("xgboost", {})
            if xgb:
                print(f"    XGBoost: {', '.join(f'{k}={v}' for k, v in sorted(xgb.items()))}")
            target = cfg.get("target", {})
            if target:
                print(f"    Target: mode={target.get('mode', '?')}")
            floor = cfg.get("floor", {})
            if floor:
                print(f"    Floor: floor_pct={floor.get('floor_pct', '?')}, enabled={floor.get('enabled', '?')}")
            sw = cfg.get("sample_weights", {})
            active = {k: v for k, v in sw.items() if v} if sw else {}
            if active:
                print(f"    Sample weights: {active}")
            fg = cfg.get("features", {}).get("groups_enabled", {})
            disabled = [g for g, v in fg.items() if not v] if fg else []
            if disabled:
                print(f"    Disabled features: {disabled}")
    print()


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


def cmd_report(args) -> None:
    """Print (or write) the end-of-run report for a past run.

    Prefers the `report.md` the orchestrator wrote at the end of the run — it
    carries fields SQLite never stored (`action_status`, `change_desc`,
    `stop_reason`). Falls back to rebuilding from `runs.db`, which degrades
    those fields loudly rather than guessing at them.
    """
    from pathlib import Path as _Path

    from agent.run_report import build_report_from_tracker, render_markdown

    run = RunTracker().get_run(args.run_id)
    if run is None:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        sys.exit(1)

    stored = _Path("outputs/agent_runs") / args.run_id / "report.md"
    if stored.exists() and not args.rebuild:
        markdown = stored.read_text()
        source_note = f"(stored report: {stored})"
    else:
        markdown = render_markdown(build_report_from_tracker(run))
        source_note = "(rebuilt from runs.db)"

    if args.out:
        out = _Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print(f"wrote {out} {source_note}")
        return

    print(source_note, file=sys.stderr)
    print(markdown)


# ----------------------------------------------------------------------------
# Top-level argparse
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agentic Forecasting Framework — LLM-powered model analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- check-data subcommand ------------------------------------------------
    check_data_parser = subparsers.add_parser(
        "check-data",
        help="Run pre-training data quality checks on raw CDC data",
    )
    check_data_parser.add_argument(
        "--cutoff-date", required=True,
        help="Check data up to this date (YYYY-MM-DD)",
    )
    check_data_parser.add_argument(
        "--base-url", default=None, help="LLM server URL",
    )
    check_data_parser.add_argument(
        "--model", default=None, help="LLM model name",
    )
    check_data_parser.add_argument(
        "--dry-run", action="store_true",
        help="Run checks only, skip LLM analysis",
    )
    check_data_parser.add_argument(
        "--verbose", action="store_true",
        help="Print raw JSON report and LLM prompt",
    )
    check_data_parser.add_argument(
        "--exclude-locations",
        nargs="+",
        default=None,
        help="FIPS codes to exclude from checks (e.g., US for national aggregate)",
    )

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
        help="LLM model name (default: env LLM_MODEL or qwen3:8b)",
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
    summarize_parser.add_argument(
        "--exclude-locations",
        nargs="+",
        default=None,
        help="FIPS codes to exclude from evaluation (e.g., US for national aggregate)",
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
    improve_parser.add_argument(
        "--exclude-locations",
        nargs="+",
        default=None,
        help="FIPS codes to exclude from evaluation (e.g., US for national aggregate)",
    )
    improve_parser.add_argument(
        "--model-family", default=None,
        help="Model-bank family to refine (default: xgboost_direct, the legacy "
             "pipeline). Registered names from `list-models` or a dotted path "
             "'pkg.module:Class' for an in-house model.",
    )
    improve_parser.add_argument(
        "--report", default=None,
        help="Also write the run report here (it is always written to the run dir)",
    )

    # ---- history subcommand ------------------------------------------------
    history_parser = subparsers.add_parser(
        "history", help="List recent improvement runs from the SQLite tracker",
    )
    history_parser.add_argument(
        "--limit", type=int, default=10, help="Number of runs to show (default: 10)",
    )

    # ---- status subcommand ---------------------------------------------------
    status_parser = subparsers.add_parser(
        "status", help="Show detailed status of a specific run (iterations + actions)",
    )
    status_parser.add_argument("run_id", help="The run_id to inspect")
    status_parser.add_argument(
        "--config", dest="show_config", action="store_true",
        help="Show full config (hyperparameters, weights, etc.) for each iteration",
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

    # ---- report subcommand -------------------------------------------------
    report_parser = subparsers.add_parser(
        "report", help="Regenerate the end-of-run report for a past run",
    )
    report_parser.add_argument("run_id", help="The run_id to report on")
    report_parser.add_argument(
        "--out", default=None, help="Write the markdown here instead of stdout",
    )
    report_parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild from runs.db even if report.md exists (degrades unpersisted fields)",
    )

    # ---- list-models subcommand --------------------------------------------
    subparsers.add_parser(
        "list-models", help="List model-bank families available in this environment",
    )

    # ---- select-model subcommand -------------------------------------------
    select_parser = subparsers.add_parser(
        "select-model",
        help="Stage 1: warm-up every candidate family on the pinned eval window "
             "and pick the incumbent by the goal metric",
    )
    select_parser.add_argument(
        "--families", nargs="+", default=None,
        help=f"Families to compare (default: {' '.join(DEFAULT_SELECT_FAMILIES)})",
    )
    select_parser.add_argument(
        "--stride-weeks", type=int, default=4,
        help="Use every Nth Saturday cutoff of the pinned eval window (default: 4; 1 = every week)",
    )
    select_parser.add_argument(
        "--max-cutoffs", type=int, default=None,
        help="Cap the number of cutoffs (after striding) for quick demos",
    )
    select_parser.add_argument(
        "--cutoffs", nargs="+", default=None,
        help="Explicit cutoff dates (YYYY-MM-DD); overrides the pinned window",
    )
    select_parser.add_argument(
        "--metric", default="wis",
        choices=["wis", "mape", "mae", "rmse", "coverage_95", "bias"],
        help="Goal metric for ranking (default: wis)",
    )
    select_parser.add_argument(
        "--phase", default="all", choices=["all", "onset", "peak", "decline"],
        help="Read the goal metric from this phase (default: all = overall)",
    )
    select_parser.add_argument(
        "--locations", nargs="+", default=None,
        help="Restrict to these FIPS codes (default: all)",
    )
    select_parser.add_argument(
        "--exclude-locations", nargs="+", default=None,
        help="FIPS codes to drop from scoring (e.g. US)",
    )
    select_parser.add_argument("--json", default=None, help="Write the full result as JSON here")
    select_parser.add_argument(
        "--report", default=None, help="Write a human-readable markdown summary here",
    )
    select_parser.add_argument(
        "--save-forecasts", default=None, help="Directory to write one forecast CSV per family",
    )
    select_parser.add_argument("--quiet", action="store_true", help="Hide per-cutoff progress")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "check-data":
        cmd_check_data(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "improve":
        cmd_improve(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "select-model":
        cmd_select_model(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
