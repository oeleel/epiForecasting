# Handoff 2026-09-11 — MacBook → desktop, branch `feature/run-report-and-nl-goal`

Written for the next working session (human or agent) on the desktop. The branch
is pushed and shareable, **not merge-ready**: the adversarial review of the
run-report + NL-goal work returned 25 findings (3 blockers) that have not been
addressed. They previously lived only in the MacBook session's temp files; the
authoritative list is §3 below.

## 1. Desktop setup

```bash
git fetch && git checkout feature/run-report-and-nl-goal

# venv does not sync (gitignored) — rebuild it:
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r documentation/requirements.txt pytest langchain-openai
uv pip install --python .venv/bin/python -r documentation/requirements-nixtla.txt

# macOS-specific: the KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS libomp workaround
# lives in the MacBook's ~/.zshrc. Linux desktop likely does NOT need it, but if
# torch-family runs segfault, set:  export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

# verify (expected: pytest 183 passed; run_all 170/170; gap 13 exactly)
.venv/bin/python -m pytest tests/agent/ -q
PYTHONPATH=. .venv/bin/python tests/agent/run_all.py

# LLM for the agent loops: ollama serve + qwen3:8b (pull if absent)
```

## 2. What the branch carries (5 commits ahead of main)

- **Run-report generator** (roadmap 6.1) — `agent/run_report.py`, `agent report <run_id>`,
  auto-written `report.md`/`report.json` per improve run. Built, findings open.
- **Natural-language goals** (roadmap 2.3) — `agent/goal_parser.py`,
  `select-model --goal "..." [--no-llm] [--explain-goal]`. Built, findings open.
- **Curve-based phase segmentation** — `agent/phase_segmentation.py` (Adiga
  Algorithms 1+3, surge/plateau/decline), 19 tests. Solid.
- **Paper summaries** — `ts-agent-paper-summary.md`, `phase-based-training-papers.md`
  (both supersede `ts-agent-comparison-notes.md` where they conflict).
- Two WIP-autosave checkpoint commits from the Stop hook — squash away at merge time.

## 3. Open review findings (final verify pass, 4 reviewers, 2026-09-09)

### Blockers — fix before anything else

1. **`run_report.py` ~519/529 — horizon table blank on every LIVE report.**
   `_segment_keys` stringifies keys but `_segment` indexes the original dict with
   the string; `PhaseEvaluator.evaluate_by_horizon` returns **int** keys in-process
   (the tracker/rebuild path JSON-stringifies them, which is why `agent report
   --rebuild` looks fine while the orchestrator-written report.md is blank).
   Fix: normalize the block once (`{str(k): v ...}`) and look up in that.
   Add a regression test whose `by_horizon` uses int keys.
2. **`run_report.py` ~426/429 — improvement computed across mismatched windows.**
   `improvement_frac` and the adopt/keep recommendation never check that baseline
   and best were scored on the same date range / forecast count; only the best
   iteration's window is printed. Fix: compare `date_range`/`n_forecasts`; on
   mismatch print both windows, set `improvement_frac=None`, recommendation
   "inconclusive — windows differ".

### Majors

3. `goal_parser.py:90` — bare-substring keyword match: "wis" fires inside
   **"Wisconsin"/"otherwise"**, "peak" inside "speaking" (line 105, minor twin),
   and precedence silently beats mape/rmse/mae. Fix: word-boundary regex for the
   short acronym entries.
4. `cli.py:454` — explicit `--metric`/`--phase` override rebuilds the goal
   **without re-running `_guard`**, so `--metric rmse --phase peak` sneaks past
   the exact trap the guard exists for; provenance tag printed is also wrong.
   Promote `_guard` to a public `check_goal()` and call it on the merged goal.
5. `run_report.py:497` — for coverage_95/bias the direction-aware fraction
   divides by distance-to-target ≈ 0 on a well-calibrated baseline → unbounded
   nonsense percentages. Report distance-to-target in metric units instead.
6. `run_report.py:597` — attribution tables hardcode wis/mape regardless of the
   run's target metric (a coverage_95 run's "where the benefit came from" never
   shows coverage). Emit target-metric columns first.
7. `run_report.py:660` — single-cutoff runs (every `improve` run): phase table is
   a pure relabelling of the horizon table and no caveat says so.
8. `run_report.py:134` — **no config anywhere in report.md/report.json**; the
   recommendation says "adopt iteration N's config" without printing it. Add
   params of the best iteration.
9. `run_report.py:546` — no statement of how many forecast origins/cutoffs were
   evaluated; smoke run and citable run format identically. Print origin count +
   an explicit "single-cutoff smoke run — not citable" line.
10. `goal_parser.py:229` — two metrics in one sentence: first-in-tuple wins and
    the LLM is never consulted. Return all matches; >1 distinct value = unresolved.
11. `documentation/*` — spec §5 documentation touches skipped entirely: roadmap
    rows 73 (2.3) / 106 (6.1) still unticked, demo-runbook gap items open (and its
    line ~38 "runs.db empty" claim is false — 15 real runs), `.claude/CLAUDE.md`
    test count stale (now 183) and missing `run_report.py`/`goal_parser.py`/
    `phase_segmentation.py` + `report` command in its trees.

### Minors / nits (fix opportunistically)

- `goal_parser.py:214` — REJECTED_PHRASES fires even when a legal phase is also
  named ("peak, not the off-season" is hard-refused). Resolve keywords first.
- `run_report.py:712` — recommendation gates on `improvement_frac` instead of
  `best_iteration`; baseline exactly 0 (coverage/bias) → tells you to keep a
  strictly-worse baseline.
- `run_report.py:633/646` — "identical metrics across iterations with different
  configs" caveat never compares configs; false-accuses no-op/skip/stop rows on
  the tracker path (live example: run 20260422-192720-2306).
- `run_report.py:257` — Errors: section unreachable on the live path (orchestrator
  breaks without appending the failed IterationRecord), so report.md and
  `--rebuild` disagree about the same run.
- `run_report.py:606` — segment table falls back to baseline `n` for the best
  column; advertise per-column n or "-".
- `cli.py:438` — `if args.goal:` treats `--goal ""` as absent; use `is not None`.
- `cli.py:452` — warnings printed before the override block, so it warns about a
  field the user explicitly supplied.
- `goal_parser.py:385` — `_degraded` calls `_guard` and can raise on an LLM
  failure, contradicting the "never raises because of the LLM" contract
  (unreachable from CLI, reachable via API with a custom fallback).
- `run_report.py:687` — standing caveat describes columns the table doesn't render.
- Spec-name drift (informational): `_line_from_record`→`_normalize_record`+
  `_build_lines`, `_line_from_row`→`_normalize_row` — behavior-equivalent.

## 4. Suggested order of work on the desktop

1. Blockers 1–2 (+ minor 712 while in `_recommendation`), with regression tests.
2. Majors 3, 4, 10 (goal-parser input handling cluster) + minors 214/438/452.
3. Majors 5–9 (report content/honesty cluster) + remaining minors.
4. Major 11 (docs sync) last, in the same commit series.
5. Re-run both gates; then squash-merge to main per repo convention.

## 5. Next research actions (independent of the fixes)

- **Re-run Stage-1 selection under curve-based phase labels** (`phase_segmentation`)
  and check whether "NHITS wins decline" survives — the replication result for the
  January paper.
- Advisor inputs pending from the 09-10 meeting (capture outcomes when known):
  δ per-week vs per-segment; replace-vs-parallel phase labels; his best-practices
  list (knowledge-bank seed); January framing; day-to-day use case.
- Novel directions pitched (see the briefing artifact / `phase-based-training-papers.md` §7):
  phase→model roster, phase-boundary retrain trigger, self-writing knowledge bank.
