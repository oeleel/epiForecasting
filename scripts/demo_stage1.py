#!/usr/bin/env python3
"""
Stage 1 Demo - Model Bank + Model Selection
============================================

The researcher's walkthrough of the selection stage that sits in front of the
two-agent improvement loop (TS-Agent Stage 1 -> our Stage 2):

  1. List the model families available in this environment
  2. Show the pinned train / evaluation window and the cutoffs it yields
  3. Warm-up: fit + forecast every candidate at every cutoff (no leakage)
  4. Score with the shared phase-aware WIS and rank by the goal metric
  5. Same warm-up, different goals: re-rank by peak and by decline WIS
     (no refitting - the goal is a parameter, exactly as the advisor asked)
  6. Write the JSON + markdown report for the run
  7. (optional) Hand the incumbent to the improvement loop for 2 iterations

Modes:
    default (quick)   6 locations, 3 cutoffs, 4 fast families. Under a minute.
    --full            All 52 locations, every 4th Saturday of the eval window,
                      default families. ~10 minutes.
    --families ...    Any registered family or a dotted "pkg.mod:Class" path
                      to an in-house model.
    --improve         After selection, run the improvement loop on the
                      incumbent (needs an LLM server; skipped with a message
                      if none is reachable).

Usage:
    python scripts/demo_stage1.py
    python scripts/demo_stage1.py --full
    python scripts/demo_stage1.py --families persistence xgboost_direct my_lab.models:FluLSTM
    python scripts/demo_stage1.py --improve --auto-apply
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter  # noqa: E402
from agent.model_selection import (  # noqa: E402
    SelectionGoal,
    evaluate_candidates,
    render_markdown,
    select_incumbent,
)
from src import config as repo_config  # noqa: E402
from src.model_bank.registry import list_families, resolve_family  # noqa: E402

# Quick mode keeps the demo under two minutes on a laptop: a handful of
# large + mid-size states and the national aggregate for context.
QUICK_LOCATIONS = ["06", "48", "12", "36", "17", "US"]
QUICK_MAX_CUTOFFS = 3
QUICK_STRIDE_WEEKS = 8
FULL_STRIDE_WEEKS = 4
# Fast families first so the audience sees output within seconds. The
# in-repo XGBoost is the lab's incumbent and must always be in the lineup.
QUICK_FAMILIES = ["persistence", "seasonal_naive", "xgboost_direct", "sf_autoets"]
FULL_FAMILIES = ["persistence", "seasonal_naive", "xgboost_direct", "sf_autoets",
                 "mlf_lightgbm", "nf_nhits"]
DEMO_GOALS = [SelectionGoal("wis", "all"), SelectionGoal("wis", "peak"),
              SelectionGoal("wis", "decline")]
IMPROVE_ITERATIONS = 2
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_selection"


def banner(title: str, char: str = "=", width: int = 78) -> None:
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def step(label: str) -> None:
    print(f"\n--- {label} ---")


def _fmt(value, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def show_families() -> None:
    step("1. Model families available here (python -m agent list-models)")
    print(f"  {'family':<18} {'ok':<4} description")
    for info in list_families():
        ok = "yes" if info.available else "NO"
        print(f"  {info.family:<18} {ok:<4} {info.description if info.available else info.reason}")
    print("\n  An in-house model needs no registration: pass 'pkg.module:ClassName'")
    print("  as the family. Contract + 30-line recipe: documentation/MODEL_BANK.md")


def show_window(cutoffs: list[str], locations, full: bool) -> None:
    step("2. Pinned evaluation window (src/config.py)")
    print(f"  train from       {repo_config.TRAIN_START_DATE}")
    print(f"  evaluate         {repo_config.EVAL_START_DATE} .. {repo_config.EVAL_END_DATE}")
    print(f"  cutoffs used     {len(cutoffs)}: {', '.join(cutoffs)}")
    print(f"  locations        {'all 52' if locations is None else ', '.join(locations)}")
    print("  At each cutoff every model sees data <= that date only, then forecasts 1-4 weeks.")
    if not full:
        print("  (quick mode - pass --full for all locations and every 4th week)")


def print_ranking(result, label: str) -> None:
    print(f"\n  goal: {label}")
    header = (f"  {'rank':<5}{'family':<18}{'WIS':>8}{'MAPE':>8}{'cov95':>7}"
              f"{'onset':>9}{'peak':>9}{'decline':>9}{'fit s':>8}")
    print(header)
    by_family = {c.family: c for c in result.candidates}
    for rank, row in enumerate(result.ranking, start=1):
        cand = by_family[row["family"]]
        o = cand.metrics.get("overall", {})
        ph = cand.metrics.get("by_phase", {})
        mark = "  <- incumbent" if cand.family == result.incumbent else ""
        print(f"  {rank:<5}{cand.family:<18}{_fmt(o.get('wis')):>8}{_fmt(o.get('mape')):>8}"
              f"{_fmt(o.get('coverage_95'), 2):>7}"
              f"{_fmt((ph.get('onset') or {}).get('wis')):>9}"
              f"{_fmt((ph.get('peak') or {}).get('wis')):>9}"
              f"{_fmt((ph.get('decline') or {}).get('wis')):>9}"
              f"{_fmt(cand.fit_seconds_total):>8}{mark}")
    for cand in result.candidates:
        if not cand.ok:
            print(f"  {'-':<5}{cand.family:<18}FAILED: {cand.error}")


def run_improve(incumbent: str, cutoff: str, locations, auto_apply: bool,
                base_url, model, run_dir: Path) -> None:
    step(f"7. Hand-off: improvement loop on the incumbent ({incumbent})")
    from agent.llm_client import LLMClient
    from agent.orchestrator import Orchestrator, auto_apply_confirmer, interactive_confirmer
    from agent.run_tracker import RunTracker
    from src.config import get_default_config

    llm = LLMClient(base_url=base_url, model=model)
    problem = llm.diagnose()
    if problem:
        print(f"  skipped: no LLM reachable ({problem.splitlines()[0]})")
        print("  start Ollama (`ollama serve` + `ollama pull qwen3:8b`) or set LLM_BASE_URL")
        return

    adapter = FluForecastAdapter(exclude_locations=["US"])
    config = get_default_config()
    config["model"]["family"] = incumbent
    if locations is not None:
        config["data"]["locations"] = list(locations)
    orch = Orchestrator(
        adapter=adapter,
        llm=llm,
        pipeline=adapter.run_pipeline,
        tracker=RunTracker(),
        confirmer=auto_apply_confirmer if auto_apply else interactive_confirmer,
        target_metric="wis",
        max_iterations=IMPROVE_ITERATIONS,
        run_dir=run_dir / "improve",
        verbose=True,
    )
    result = orch.run(cutoff_date=cutoff, initial_config=config)
    summary = result.to_summary()
    print(f"\n  run_id {summary['run_id']}: {summary['n_iterations']} iterations, "
          f"stop={summary['stop_reason']}, best iteration {summary['best_iteration']}")
    if summary["baseline"] is not None and summary["best"] is not None:
        print(f"  WIS baseline -> best: {summary['baseline']:.2f} -> {summary['best']:.2f} "
              f"({(summary['relative_pct'] or 0):+.1f}%)")
    print(f"  inspect: python -m agent status {summary['run_id']} --config")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 demo: model bank + selection")
    p.add_argument("--full", action="store_true", help="all locations, every 4th week")
    p.add_argument("--families", nargs="+", default=None, help="families to compare")
    p.add_argument("--improve", action="store_true",
                   help="run the improvement loop after selection")
    p.add_argument("--auto-apply", action="store_true", help="improve without y/n prompts")
    p.add_argument("--base-url", default=None, help="LLM server URL (improve only)")
    p.add_argument("--model", default=None, help="LLM model name (improve only)")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("src.model_bank").setLevel(logging.ERROR)

    families = args.families or (FULL_FAMILIES if args.full else QUICK_FAMILIES)
    for fam in families:
        resolve_family(fam)  # fail fast with the registry's message
    locations = None if args.full else QUICK_LOCATIONS
    stride = FULL_STRIDE_WEEKS if args.full else QUICK_STRIDE_WEEKS
    cutoffs = repo_config.generate_eval_cutoffs(stride_weeks=stride)
    if not args.full:
        cutoffs = cutoffs[:QUICK_MAX_CUTOFFS]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_ROOT / f"demo_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    banner("STAGE 1 DEMO: model bank + model selection")
    print("  Question a researcher asks: which model should I even start from,")
    print("  before spending iterations tuning it? This stage answers that.")

    show_families()
    show_window(cutoffs, locations, args.full)

    step(f"3. Warm-up: {len(families)} families x {len(cutoffs)} cutoffs")
    actuals = FluForecastAdapter()._load_actuals()
    t0 = time.perf_counter()
    candidates = evaluate_candidates(
        families=families, cutoffs=cutoffs, actuals=actuals,
        locations=locations, exclude_locations=["US"], progress=print,
    )
    print(f"  warm-up done in {time.perf_counter() - t0:.1f}s")

    step("4. Score + rank by the default goal (overall WIS, US excluded)")
    results = {g.describe(): select_incumbent(candidates, goal=g, cutoffs=cutoffs)
               for g in DEMO_GOALS}
    primary = results[DEMO_GOALS[0].describe()]
    print_ranking(primary, DEMO_GOALS[0].describe())
    if not args.full:
        print("\n  quick mode = smoke test on a few states and 3 cutoffs; rankings can flip.")
        print("  Cite the --full run (all states, 9 cutoffs) when reporting results.")

    step("5. Same warm-up, different goals (no refitting)")
    print("  The advisor left the objective open: peak vs average vs overall.")
    print("  The goal is a parameter, so one warm-up answers all three:")
    for goal in DEMO_GOALS[1:]:
        res = results[goal.describe()]
        print(f"  {goal.describe():<22} -> incumbent {res.incumbent}")
    print("  (CLI: python -m agent select-model --metric wis --phase peak)")

    step("6. Artifacts for the run log")
    (run_dir / "selection.json").write_text(
        json.dumps({k: v.to_json() for k, v in results.items()}, indent=2, default=str)
    )
    report = "".join(
        render_markdown(res, title=f"Model selection - goal {label}") + "\n"
        for label, res in results.items()
    )
    (run_dir / "report.md").write_text(report)
    for cand in candidates:
        if cand.forecasts is not None:
            cand.forecasts.to_csv(run_dir / f"forecasts_{cand.family.replace(':', '_')}.csv",
                                  index=False)
    print(f"  {run_dir}/selection.json   full metrics per family, per goal")
    print(f"  {run_dir}/report.md        human-readable summary")
    print(f"  {run_dir}/forecasts_*.csv  every forecast scored, for auditing")

    if args.improve and primary.incumbent:
        run_improve(primary.incumbent, cutoffs[-1], locations, args.auto_apply,
                    args.base_url, args.model, run_dir)

    banner("NEXT")
    if primary.incumbent:
        print(f"  refine the incumbent:  python -m agent improve --cutoff-date {cutoffs[-1]} "
              f"--model-family {primary.incumbent} --auto-apply")
    print("  compare your own model: python -m agent select-model --families "
          "xgboost_direct my_lab.models:FluLSTM")
    print("  full-scale run:         python -m agent select-model --stride-weeks 4 "
          "--exclude-locations US --report outputs/model_selection/report.md")


if __name__ == "__main__":
    main()
