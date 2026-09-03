# Fall 2026 roadmap — model bank, selection stage, and warm-start research

Work plan coming out of the 2026-09-02 advisor sync
(`meeting-notes/2026-09-02-ts-agent-review-and-nixtla-direction.md`) and the
TS-Agent gap analysis (`ts-agent-comparison-notes.md`).

**Frame:** Milestones 1–3 (summarize, improvement loop, tracking) are complete.
This roadmap adds TS-Agent's Stage 1 (model selection over a Nixtla model bank)
in front of our existing Stage 2 (constrained refinement loop), plus the
warm-start research layer and the knowledge/reporting pieces needed for the
**~January paper submission**.

**Scope guardrails (decided 09-02):**
- Internal exploration tool — not real-time, no operational submissions.
- No forecast-summarization agent (deferred).
- LLM refines configs against templates; it never writes model code.
- **The lab's in-house models are the real model bank.** Nixtla families are
  examples that prove the architecture (classical / boosted / neural through
  one contract); do not invest in tuning them. Everything must stay
  plug-and-play for whatever models the team brings
  (`documentation/MODEL_BANK.md`).

---

## Workstream 0 — Repo hygiene & data refresh

| # | Task | Status |
|---|---|---|
| 0.1 | Untrack committed `__pycache__` artifacts (gitignore already covers them) | ✅ 2026-09-02 |
| 0.2 | Commit meeting notes + TS-Agent comparison notes | ✅ 2026-09-02 |
| 0.3 | Refresh CDC FluSight cache (`python -m src.data_loader update`) — cache now runs through 2026-07-04, covering the whole eval window | ✅ 2026-09-02 |
| 0.4 | Pin the agreed split as named constants in `src/config.py`: train **2022 → 2025**, evaluate **Oct 2025 → May 2026** (`TRAIN_START_DATE`, `EVAL_START_DATE`, `EVAL_END_DATE`, `generate_eval_cutoffs`) | ✅ 2026-09-02 |

## Workstream 1 — Model bank (next-meeting goal)

One contract (`src/model_bank/contract.py: ForecastModel`) that any model
implements; Nixtla families are the worked examples. Landed 2026-09-02 on
`feature/model-bank`.

| # | Task | Notes |
|---|---|---|
| 1.1 | Add Nixtla deps (`statsforecast`, `mlforecast`, `neuralforecast`) to requirements; keep isolated from the legacy pipeline | ✅ `documentation/requirements-nixtla.txt`; imports are guarded, families show as unavailable when missing |
| 1.2 | Data bridge: CDC weekly admissions → Nixtla long format (`unique_id`, `ds`, `y`), reusing `FluDataLoader` | ✅ `src/model_bank/data_bridge.py` (both directions + forecast-CSV writer) |
| 1.3 | Wrap an initial bank behind one interface | ✅ 9 families: `persistence`, `seasonal_naive`, `xgboost_direct`, `nn_quantile`, `sf_autoarima`, `sf_autoets`, `sf_autotheta`, `mlf_lightgbm`, `nf_nhits`; in-house models via dotted path `pkg.mod:Class` with no repo edit |
| 1.4 | Quantile support: map model outputs to our levels so `PhaseEvaluator` WIS/coverage work unchanged | ✅ validated at the seam (`validate_forecast_frame`), crossings repaired + counted |
| 1.5 | Plug into the adapter contract (`run_pipeline(config)` with a `model.family` key) so the existing orchestrator can drive any bank model | ✅ pipeline dispatches on `model.family`; Agent 2's `adjust_hyperparameter` is generated from each family's `param_space()` |

**Acceptance:** `python -m agent select-model --stride-weeks 4` runs every
candidate on the pinned split and prints phase-aware WIS per family. ✅

First run (2026-09-02, 9 cutoffs Oct 2025 → May 2026 at stride 4, US excluded,
all defaults, no tuning; full result in `outputs/model_selection/demo_2026-09-02.json`):

| family | WIS | MAPE | cov95 | onset WIS | peak WIS | decline WIS | fit s |
|---|---|---|---|---|---|---|---|
| xgboost_direct (incumbent) | 50.6 | 67.5 | 0.83 | 11.0 | 122.4 | 49.3 | 92.7 |
| nf_nhits | 65.6 | 83.6 | 0.65 | 12.9 | 189.7 | 41.7 | 66.6 |
| sf_autoets | 82.2 | 88.6 | 0.86 | 16.0 | 235.1 | 50.6 | 26.7 |
| persistence | 90.5 | 89.0 | 0.72 | 20.4 | 235.8 | 70.9 | 0.1 |
| mlf_lightgbm | 138.7 | 108.0 | 0.64 | 15.8 | 408.6 | 85.8 | 1.6 |
| seasonal_naive | 153.2 | 163.2 | 0.76 | 23.2 | 304.6 | 209.2 | 0.1 |

Read: the lab's XGBoost beats every untuned example, peak is where everyone
loses, and NHITS already wins the decline phase — a concrete case for the
goal-as-parameter selection (2.3).

## Workstream 2 — Model selection stage (TS-Agent Stage 1)

| # | Task | Notes |
|---|---|---|
| 2.1 | Warm-up runs: cheap short-horizon fits of each bank model at the cutoff before committing to an incumbent | ✅ `agent/model_selection.py: evaluate_candidates` (rolling origin over the pinned cutoffs, one family failing never sinks the rest) |
| 2.2 | Incumbent selection by WIS (or the specified goal metric); hand off to the existing refinement loop | ✅ `select_incumbent`; `improve --model-family <incumbent>` |
| 2.3 | Goal-as-parameter: selection/refinement objective specified per run (peak performance vs. average vs. overall), settable via natural-language command | ◐ `SelectionGoal(metric, phase)` + `--metric/--phase` flags done; natural-language front end not started |

## Workstream 3 — Orchestrator fixes (from the gap analysis, ranked)

| # | Task | Notes |
|---|---|---|
| 3.1 | **Revert-on-regression** — restore best-known config when an iteration regresses instead of committing it (orchestrator step 8) | ✅ 2026-09-03 `agent/orchestrator.py`: comparisons now use `iterations[best_idx]` instead of the previous iteration; on regression, `config = copy.deepcopy(iterations[best_idx].config)` before the next diagnosis. Live A/B on `nf_nhits` @ 2026-01-24: before, iter2 built on a regressed `input_size=52` and reached WIS 300.5 (3x baseline) before stopping at `max_iterations`; after, iter2 diagnosed from the reverted baseline, proposed an independent fix, reached WIS 105.7 (4% off baseline), and stopped correctly at `regressed_x2`. All 138 tests still pass (`test_two_regressions_stop`, `test_best_forecast_retained_on_regression` unchanged). |
| 3.2 | Case retrieval over `runs.db` — surface "what worked at similar cutoffs/phases in past runs" in Agent 2's proposal prompt | ☐ |

## Workstream 4 — Warm-start research layer (paper centerpiece)

The advisor's core research question: when does incremental fine-tuning beat
retraining from scratch — and can the framework decide *a priori*?

| # | Task | Notes |
|---|---|---|
| 4.1 | Persist trained model state per run so later runs can resume ("last week's model") | ☐ |
| 4.2 | Add `fine_tune` vs. `retrain_from_scratch` as explicit actions/configs | ☐ |
| 4.3 | Regime-change awareness: flag phase transitions (off-season → onset → peak) where a warm start trained on the prior regime is suspect | ☐ |
| 4.4 | Experiment harness: user specifies the investigation ("train these models; find up to what point fine-tuning sufficed vs. full retrain"); framework runs it and logs the answer | ☐ |

## Workstream 5 — Epi knowledge bank

| # | Task | Notes |
|---|---|---|
| 5.1 | Curated heuristics file (read-only, versioned like data) injected into Agent 2's prompt — e.g. "entering the season → window ≈ 12 weeks", "peak underprediction → target transform / peak reweight", "low coverage → widen quantiles" | ☐ |
| 5.2 | Collect the advisor's best practices in the weekly syncs; add literature-sourced entries | ☐ |
| 5.3 | (Later) retrieval over the bank if it outgrows a single prompt injection | ☐ |

## Workstream 6 — Run reporting ("most important output")

| # | Task | Notes |
|---|---|---|
| 6.1 | End-of-run report generator: what was tried, per-iteration deltas, where the benefit came from, final recommendation — markdown, agent- and human-legible | ☐ |
| 6.2 | Cross-run assessment view: which models did well in which phases with how much (re)training — the evidence base for promoting a model into the lab's software | ☐ |

## Workstream 7 — Paper (January)

| # | Task | Notes |
|---|---|---|
| 7.1 | Keep an experiments log suitable for the paper (runs.db + reports already give the audit trail) | ☐ |
| 7.2 | Frame contribution vs. TS-Agent: epi-specific phase evaluation, small-LLM constrained-action safety, warm-start regime analysis | ☐ |
| 7.3 | Draft outline by ~November; results freeze ~December | ☐ |

---

## Sequencing

1. **This week:** ~~0.3–0.4, then Workstream 1 (demo for Thursday)~~ done 09-02.
2. **Next:** 3.1 (revert-on-regression, trivial), then 2.3's natural-language goal,
   then wrap the lab's first in-house model against the contract.
3. **Then:** 4 (warm-start layer) and 5 (knowledge bank) in parallel — both feed the paper.
4. **Ongoing:** 6 (reporting) as soon as multi-model runs exist; 7 from November.

Per repo conventions: each workstream lands on its own `feature/…` branch,
squash-merged into `main` at milestone quality.
