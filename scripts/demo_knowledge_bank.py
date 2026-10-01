#!/usr/bin/env python3
"""
Knowledge Bank Demo - the 2026-10-01 meeting walkthrough
=========================================================

The ask (A. Adiga, meeting 2026-09-10, follow-up 09-24): a knowledge bank the
improvement loop reads before it proposes a change, holding the lab's rules,
facts derived from the data, and lessons from past runs, as *queryable*
entries that the loop cites. This script walks that bank end to end in under
three minutes on a laptop, with no LLM in the loop, so every number shown is
reproducible in the meeting. One optional step (`--live`) runs the real
improve loop against a local Ollama to show the citations landing.

Steps (each prints what it shows and why it matters):

  1. Validate the curated YAML   what the lab authors, file by file
  2. Open the bank               curated rows rebuilt from YAML + persisted
                                 experiential rows, counted per provenance
  3. Retrieval                   the KNOWN FACTS block for peak vs plateau:
                                 the same bank gives different facts per context
  4. One entry in full           the schema, via the lambda rectification entry
  5. Experiment memory           the peak-rectification harness log imported as
                                 deterministic [E] entries, then retrieved
  6. Where it reaches the loop   the adapter's domain context tail + the
                                 improve loop's per-proposal retrieval/citation
  7. (--live) improve loop       2 iterations with --fake-pipeline, then the
                                 cited_entries / supported_by stored per action

Design: `documentation/knowledge-bank-design.md`. Code: `agent/knowledge/`.
Experiential rows come from `scripts/experiments/peak_rectification.py`
(`outputs/experiments/peak_rectification/log.jsonl`); step 5 says SKIP with the
harness command when that log is absent. Step 5 is the one step that WRITES:
it upserts the log's experiential rows into the bank the script opened, by
default the repo bank `knowledge/knowledge.db` (gitignored); the import is
idempotent, so re-running the demo leaves the same rows. `--db` points the
whole demo at another SQLite file (tests use a temp one).

Structure: every step is a function taking the bank and an output stream and
returning a JSON-serialisable record, so `tests/agent/test_demo_knowledge_bank.py`
drives steps 1-6 against a temp bank and `--json` dumps the records for the
prep note. The live step is kept out of the tests (needs Ollama).

Usage:
    python scripts/demo_knowledge_bank.py
    python scripts/demo_knowledge_bank.py --pause
    python scripts/demo_knowledge_bank.py --json outputs/demo_knowledge_bank.json
    python scripts/demo_knowledge_bank.py --db /tmp/demo.db --log tests/agent/fixtures/peak_rectification_log_sample.jsonl
    python scripts/demo_knowledge_bank.py --live        # needs Ollama qwen3:8b
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

# macOS: xgboost + torch each bundle libomp; loading both in one process
# segfaults without these (same guard as scripts/experiments/peak_rectification.py).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from agent.knowledge import (  # noqa: E402
    DEFAULT_CURATED_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_EXPERIMENT_LOG,
    DEFAULT_HEADER,
    EMPTY_BLOCK,
    PROVENANCES,
    ImportResult,
    KnowledgeBank,
    KnowledgeEntry,
    RetrievalContext,
    import_experiment_log,
    list_curated_files,
    load_curated_dir,
    render_known_facts,
)

__all__ = [
    "EXPERIMENT_LOG_PATH",
    "LAMBDA_ENTRY_ID",
    "LIVE_IMPROVE_COMMAND",
    "run_demo",
    "step_experiment_memory",
    "step_live_improve",
    "step_loop_reach",
    "step_open_bank",
    "step_retrieval",
    "step_show_entry",
    "step_validate_yaml",
]

BANNER_WIDTH = 78

# The curated entry whose schema step 4 shows: the lambda loss-weight
# rectification action, the one the peak-rectification experiment tests.
LAMBDA_ENTRY_ID = "rectify-peak-loss-weight-approaching-peak"
DEMO_MODEL_FAMILY = "xgboost_direct"
PEAK_CONTEXT = RetrievalContext(phase="peak", model=DEMO_MODEL_FAMILY)
PLATEAU_CONTEXT = RetrievalContext(phase="plateau", model=DEMO_MODEL_FAMILY)
EXPERIENTIAL_LINE_PREFIX = "- [E"

EXPERIMENT_LOG_PATH = PROJECT_ROOT / DEFAULT_EXPERIMENT_LOG
HARNESS_COMMAND = ".venv/bin/python scripts/experiments/peak_rectification.py"
VALIDATE_COMMAND = "python -m agent knowledge validate"
IMPROVE_COMMAND_HINT = "python -m agent improve --cutoff-date 2025-12-06 --auto-apply"

# Step 7: a fake pipeline copies the seed forecast each iteration, so the only
# thing exercised is the LLM proposal + knowledge retrieval, not XGBoost.
LIVE_SEED_FORECAST = "outputs/experiments/peak_rectification/forecasts/baseline.csv"
LIVE_IMPROVE_COMMAND = [
    "-m", "agent", "improve",
    "--cutoff-date", "2025-12-06",
    "--forecast-csv", LIVE_SEED_FORECAST,
    "--fake-pipeline", "--auto-apply",
    "--max-iterations", "2",
]
RUN_TRACKER_DB = PROJECT_ROOT / "outputs" / "agent_runs" / "runs.db"
# Keys the orchestrator writes onto each stored action when a bank is wired in
# (agent/orchestrator.py). Read with .get: an older run has none of them.
ACTION_KNOWLEDGE_KEYS = (
    "retrieved_entry_ids", "cited_entries", "supported_by", "supported_by_params_mismatch",
)


# ---- printing helpers ------------------------------------------------------


def banner(out: TextIO, title: str, char: str = "=") -> None:
    print(file=out)
    print(char * BANNER_WIDTH, file=out)
    print(f"  {title}", file=out)
    print(char * BANNER_WIDTH, file=out)


def step_title(out: TextIO, label: str) -> None:
    print(f"\n--- {label} ---", file=out)


def why(out: TextIO, *lines: str) -> None:
    print(file=out)
    for line in lines:
        print(f"  WHY: {line}", file=out)


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class _BlockStyleDumper(yaml.SafeDumper):
    """Multi-line strings as `|` blocks, so the folded `note` fields stay readable."""


def _represent_multiline_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockStyleDumper.add_representer(str, _represent_multiline_str)


def _to_yaml(data: dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=_BlockStyleDumper, sort_keys=False, allow_unicode=True,
                     width=BANNER_WIDTH).rstrip()


def _known_facts_tail(domain_context: str) -> str:
    """The KNOWN FACTS block at the end of a domain-context string."""
    marker = domain_context.rfind(DEFAULT_HEADER)
    if marker >= 0:
        return domain_context[marker:]
    marker = domain_context.rfind(EMPTY_BLOCK)
    if marker >= 0:
        return domain_context[marker:]
    raise ValueError(
        "domain context carries no KNOWN FACTS block; expected it to end with "
        f"{DEFAULT_HEADER!r} or {EMPTY_BLOCK!r}"
    )


# ---- steps -----------------------------------------------------------------


def step_validate_yaml(curated_dir: Path, out: TextIO) -> dict[str, Any]:
    """Step 1: load + validate every curated file; show what the lab authors."""
    step_title(out, f"1. Validate the curated YAML ({VALIDATE_COMMAND})")
    entries = load_curated_dir(curated_dir)
    files = list_curated_files(curated_dir)
    per_file: dict[str, int] = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            per_file[path.name] = len(yaml.safe_load(fh))
    skipped = sorted(p.name for p in curated_dir.glob("_*.yaml"))
    by_provenance = Counter(e.provenance for e in entries)
    by_category = Counter(e.category for e in entries)

    print(f"  dir: {curated_dir}", file=out)
    print(f"  {len(entries)} entries in {len(files)} file(s), all valid:", file=out)
    for name, count in per_file.items():
        print(f"    {name:<36} {count:>3} entries", file=out)
    if skipped:
        print(f"  skipped (template / intake stubs, '_' prefix): {', '.join(skipped)}", file=out)
    print(f"  by provenance: {dict(by_provenance)}", file=out)
    print(f"  by category:   {dict(sorted(by_category.items()))}", file=out)
    why(
        out,
        "curated = the lab's human-authored guardrails, git-tracked and PR-reviewed.",
        "One ValueError lists EVERY problem (schema, duplicate id, misnamed file), so an",
        "author fixes a file in one pass; a bank with no curated knowledge refuses to open.",
    )
    return {
        "curated_dir": str(curated_dir),
        "n_entries": len(entries),
        "n_files": len(files),
        "per_file": per_file,
        "skipped_files": skipped,
        "by_provenance": dict(by_provenance),
        "by_category": dict(sorted(by_category.items())),
    }


def step_open_bank(bank: KnowledgeBank, out: TextIO) -> dict[str, Any]:
    """Step 2: counts per provenance in the opened bank."""
    step_title(out, "2. Open the bank (KnowledgeBank.open)")
    counts = {prov: bank.count(provenance=prov) for prov in PROVENANCES}
    print(f"  db: {bank.db_path}", file=out)
    for prov, count in counts.items():
        note = {
            "curated": "rebuilt from YAML on every open; the files are the source of truth",
            "derived": "deterministic jobs over CDC data (not in v1)",
            "experiential": "written by code from logged runs; persisted across rebuilds",
        }[prov]
        print(f"    {prov:<13} {count:>3}   {note}", file=out)
    print(f"  total: {bank.count()}", file=out)
    why(
        out,
        "One SQLite table, three provenances, one trust order: curated > derived > experiential.",
        "Rebuilding curated rows never touches the other two, so a lesson learned from a run",
        "survives every YAML edit, and no id can ever change provenance.",
    )
    return {"db_path": str(bank.db_path), "counts": counts, "total": bank.count()}


def _query_block(bank: KnowledgeBank, ctx: RetrievalContext) -> tuple[list[KnowledgeEntry], str]:
    entries = bank.query(ctx)
    omitted = bank.count_matching(ctx) - len(entries)
    return entries, render_known_facts(entries, omitted=omitted)


def step_retrieval(bank: KnowledgeBank, out: TextIO) -> dict[str, Any]:
    """Step 3: the KNOWN FACTS block for peak, then plateau, and what changed."""
    step_title(out, "3. Retrieval: the block the LLM sees, keyed on (phase, model)")
    peak_entries, peak_block = _query_block(bank, PEAK_CONTEXT)
    plateau_entries, plateau_block = _query_block(bank, PLATEAU_CONTEXT)

    print(f"  RetrievalContext(phase={PEAK_CONTEXT.phase!r}, model={PEAK_CONTEXT.model!r})"
          f" -> {len(peak_entries)} entries", file=out)
    print(_indent(peak_block), file=out)
    print(file=out)
    print(f"  RetrievalContext(phase={PLATEAU_CONTEXT.phase!r}, model={PLATEAU_CONTEXT.model!r})"
          f" -> {len(plateau_entries)} entries", file=out)
    print(_indent(plateau_block), file=out)

    peak_ids = [e.id for e in peak_entries]
    plateau_ids = [e.id for e in plateau_entries]
    only_peak = [i for i in peak_ids if i not in plateau_ids]
    only_plateau = [i for i in plateau_ids if i not in peak_ids]
    print(file=out)
    print(f"  only in the peak block:    {only_peak}", file=out)
    print(f"  only in the plateau block: {only_plateau}", file=out)
    why(
        out,
        "Retrieval is a SQL filter, not a prompt dump: an entry matches when it has no",
        "constraint on a key or lists the value. The peak block carries the rectification",
        "actions; the plateau block swaps them for the long-window lull rule. Ordering is",
        "deterministic (trust, confidence, n_observations, recency, id), so the same bank",
        "and context always render byte-identical text.",
    )
    return {
        "peak": {"context": {"phase": PEAK_CONTEXT.phase, "model": PEAK_CONTEXT.model},
                 "ids": peak_ids, "block": peak_block},
        "plateau": {"context": {"phase": PLATEAU_CONTEXT.phase, "model": PLATEAU_CONTEXT.model},
                    "ids": plateau_ids, "block": plateau_block},
        "only_peak": only_peak,
        "only_plateau": only_plateau,
    }


def step_show_entry(bank: KnowledgeBank, out: TextIO,
                    entry_id: str = LAMBDA_ENTRY_ID) -> dict[str, Any]:
    """Step 4: one entry in its YAML shape so the schema is visible."""
    step_title(out, f"4. One entry in full: {entry_id}")
    entry = bank.get(entry_id)
    if entry is None:
        raise ValueError(f"entry {entry_id!r} is not in the bank at {bank.db_path}")
    as_dict = entry.to_dict()
    print(_indent(_to_yaml(as_dict)), file=out)
    why(
        out,
        "statement = the ONE sentence the LLM reads. context = WHEN it applies (the retrieval",
        "keys). payload.recommendation = the concrete adapter action, rendered as",
        "'-> try action(k=v)'. evidence = who said so / how many runs. Every field is",
        "validated on write and re-validated on read, so a stored row can never be malformed.",
    )
    return {"entry_id": entry_id, "entry": as_dict}


def step_experiment_memory(bank: KnowledgeBank, out: TextIO, log_path: Path,
                           created_at: str) -> dict[str, Any]:
    """Step 5: import the harness log as experiential entries, then retrieve them."""
    step_title(out, "5. Experiment memory: harness log -> experiential entries")
    if not log_path.is_file():
        print(f"  SKIP: no experiment log at {log_path}", file=out)
        print(f"  produce it with:  {HARNESS_COMMAND}", file=out)
        print(f"  (smoke run: {HARNESS_COMMAND} --cutoffs 1 --arms baseline lambda --lambdas 2)",
              file=out)
        return {"skipped": True, "log_path": str(log_path), "harness_command": HARNESS_COMMAND}

    result: ImportResult = import_experiment_log(log_path, bank, created_at=created_at)
    print(f"  log: {log_path}", file=out)
    print(f"  ImportResult(n_rows={result.n_rows}, n_entries={result.n_entries})", file=out)
    print(f"  ids: {list(result.ids)}", file=out)
    print(f"  experiential rows now in bank: {bank.count(provenance='experiential')}", file=out)

    _, peak_block = _query_block(bank, PEAK_CONTEXT)
    e_lines = [ln for ln in peak_block.splitlines() if ln.startswith(EXPERIENTIAL_LINE_PREFIX)]
    print(file=out)
    print(f"  peak query again, [E] lines only ({len(e_lines)}):", file=out)
    for line in e_lines:
        print(f"  {line}", file=out)
    why(
        out,
        "Each (state, action, reward) row becomes one entry DETERMINISTICALLY: no LLM writes",
        "the statement, the numbers or the retrieval keys, so a retrieved [E] fact cannot be",
        "a hallucination. Confidence is 'low' and n_observations is 1 because one run is one",
        "observation whatever the effect size; the baseline row supplies the 'before' numbers.",
        "Re-importing the same log upserts the same ids: idempotent.",
    )
    return {
        "skipped": False,
        "log_path": str(log_path),
        "created_at": created_at,
        "import_result": {"n_rows": result.n_rows, "n_entries": result.n_entries,
                          "ids": list(result.ids)},
        "experiential_lines": e_lines,
    }


def step_loop_reach(bank: KnowledgeBank, out: TextIO) -> dict[str, Any]:
    """Step 6: where the bank reaches the improve loop."""
    step_title(out, "6. Where it reaches the loop")
    from agent.adapters.flu_forecast import FluForecastAdapter
    from src.config import get_default_config

    adapter = FluForecastAdapter(knowledge_bank=bank)
    domain_context = adapter.get_domain_context(get_default_config())
    tail = _known_facts_tail(domain_context)
    print("  (a) FluForecastAdapter.get_domain_context(...) tail, appended to EVERY prompt:",
          file=out)
    print(_indent(tail, "      "), file=out)
    print(file=out)
    print("  (b) per-proposal retrieval in the improve loop (agent/orchestrator.py):", file=out)
    print(f"      {IMPROVE_COMMAND_HINT}", file=out)
    print("      At every proposal the orchestrator keys a RetrievalContext on the phase Agent 1",
          file=out)
    print("      diagnosed as weakest, shows those facts to Agent 2, and stores on the action:",
          file=out)
    print("        retrieved_entry_ids  what Agent 2 was shown", file=out)
    print("        cited_entries        what it says it relied on (restricted to the shown ids)",
          file=out)
    print("        supported_by         shown entries whose recommendation matches the action",
          file=out)
    print("        supported_by_params_mismatch  those of them recommending different params",
          file=out)
    print("      Nothing is blocked: an action no entry recommends is logged as an override.",
          file=out)
    why(
        out,
        "The lab's requirement: the loop CITES the fact it acts on but stays free to",
        "explore past the bank. Citations are stored per iteration in the run tracker, so a",
        "run report can say which rule drove which change, and which changes were explorations.",
    )
    return {
        "domain_context_tail": tail,
        "improve_command": IMPROVE_COMMAND_HINT,
        "action_keys": list(ACTION_KNOWLEDGE_KEYS),
    }


def step_live_improve(out: TextIO, tracker_db: Path = RUN_TRACKER_DB) -> dict[str, Any]:
    """Step 7 (--live): run the improve loop for real and show the stored citations."""
    step_title(out, "7. LIVE: improve loop, 2 iterations, fake pipeline (needs Ollama qwen3:8b)")
    from agent.run_tracker import RunTracker

    tracker = RunTracker(db_path=tracker_db)
    before = {r["run_id"] for r in tracker.list_runs(limit=50)}
    cmd = [sys.executable, *LIVE_IMPROVE_COMMAND]
    print(f"  $ {' '.join(cmd[1:])}", file=out)
    out.flush()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy())
    if proc.returncode != 0:
        raise SystemExit(
            f"live improve exited {proc.returncode}: is Ollama running with qwen3:8b "
            "(`ollama serve` + `ollama pull qwen3:8b`), or set LLM_BASE_URL? "
            "Drop --live to run the offline demo only."
        )

    newest = [r for r in tracker.list_runs(limit=50) if r["run_id"] not in before]
    if not newest:
        raise SystemExit(f"live improve finished but no new run appeared in {tracker_db}")
    run = tracker.get_run(newest[0]["run_id"])
    assert run is not None
    print(f"\n  run {run['run_id']}: status={run.get('status')}, "
          f"{len(run['iterations'])} iteration rows", file=out)
    record: list[dict[str, Any]] = []
    for it in run["iterations"]:
        action = it.get("action") or {}
        if not action:
            print(f"  iter {it['iteration']}: baseline (no action)", file=out)
            continue
        params = action.get("params", {})
        print(f"  iter {it['iteration']}: {action.get('name')}({params})", file=out)
        for key in ACTION_KNOWLEDGE_KEYS:
            print(f"      {key:<20} {action.get(key)}", file=out)
        record.append({"iteration": it["iteration"], "name": action.get("name"),
                       "params": params,
                       **{key: action.get(key) for key in ACTION_KNOWLEDGE_KEYS}})
    why(
        out,
        "This is the closed loop: the bank's facts were retrieved for the diagnosed phase,",
        "Agent 2 cited the ones it used, and the citations are persisted with the action.",
    )
    return {"run_id": run["run_id"], "actions": record}


# ---- driver ----------------------------------------------------------------


def run_demo(bank: KnowledgeBank, out: TextIO, *, curated_dir: Path = DEFAULT_CURATED_DIR,
             log_path: Path = EXPERIMENT_LOG_PATH, created_at: str | None = None,
             pause: bool = False, live: bool = False) -> dict[str, Any]:
    """Run steps 1-6 (and 7 with `live`) against `bank`; return the JSON record."""
    created_at = created_at or dt.date.today().isoformat()

    def maybe_pause() -> None:
        if pause:
            input("\n  [enter to continue]")

    banner(out, "KNOWLEDGE BANK DEMO: what the loop knows before it acts")
    print("  The question: can the improvement loop read the lab's rules and its own", file=out)
    print("  past runs as queryable facts, and cite them when it proposes a change?", file=out)

    record: dict[str, Any] = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                              "steps": {}}
    record["steps"]["1_validate_yaml"] = step_validate_yaml(curated_dir, out)
    maybe_pause()
    record["steps"]["2_open_bank"] = step_open_bank(bank, out)
    maybe_pause()
    record["steps"]["3_retrieval"] = step_retrieval(bank, out)
    maybe_pause()
    record["steps"]["4_entry"] = step_show_entry(bank, out)
    maybe_pause()
    record["steps"]["5_experiment_memory"] = step_experiment_memory(bank, out, log_path, created_at)
    maybe_pause()
    record["steps"]["6_loop_reach"] = step_loop_reach(bank, out)
    if live:
        maybe_pause()
        record["steps"]["7_live"] = step_live_improve(out)

    banner(out, "NEXT")
    print(f"  author a rule:        edit knowledge/curated/*.yaml, then {VALIDATE_COMMAND}",
          file=out)
    print("  query the bank:       python -m agent knowledge query --phase peak "
          f"--model {DEMO_MODEL_FAMILY}", file=out)
    shown_log = log_path.relative_to(PROJECT_ROOT) if log_path.is_relative_to(PROJECT_ROOT) \
        else log_path
    print(f"  import an experiment: python -m agent knowledge import-experiment --log {shown_log}",
          file=out)
    print(f"  run the loop:         {IMPROVE_COMMAND_HINT}", file=out)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description="Knowledge bank demo (meeting 2026-10-01)")
    p.add_argument("--pause", action="store_true", help="wait for enter between steps")
    p.add_argument("--live", action="store_true",
                   help="also run the improve loop for 2 iterations (needs Ollama qwen3:8b)")
    p.add_argument("--json", type=Path, default=None, metavar="PATH",
                   help="dump a machine-readable record of the steps to PATH")
    p.add_argument("--log", type=Path, default=EXPERIMENT_LOG_PATH,
                   help="peak-rectification log.jsonl for step 5")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                   help="SQLite bank to open (and write step 5's rows into); "
                        "default is the repo bank knowledge/knowledge.db")
    args = p.parse_args()

    bank = KnowledgeBank.open(curated_dir=DEFAULT_CURATED_DIR, db_path=args.db)
    record = run_demo(bank, sys.stdout, log_path=args.log, pause=args.pause, live=args.live)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
