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

---

## Workstream 0 — Repo hygiene & data refresh

| # | Task | Status |
|---|---|---|
| 0.1 | Untrack committed `__pycache__` artifacts (gitignore already covers them) | ✅ 2026-09-02 |
| 0.2 | Commit meeting notes + TS-Agent comparison notes | ✅ 2026-09-02 |
| 0.3 | Refresh CDC FluSight cache (`python -m src.data_loader update`) — last data commit is through 2026-03-07; eval window ends May 2026 | ☐ |
| 0.4 | Pin the agreed split as named constants in `src/config.py`: train **2022 → 2025**, evaluate **Oct 2025 → May 2026** | ☐ |

## Workstream 1 — Nixtla model bank (next-meeting goal)

Uniform template for many candidate models; replaces hand-building competitors.

| # | Task | Notes |
|---|---|---|
| 1.1 | Add Nixtla deps (`statsforecast`, `mlforecast`, `neuralforecast`) to requirements; keep isolated from the legacy pipeline | ☐ |
| 1.2 | Data bridge: CDC weekly admissions → Nixtla long format (`unique_id`, `ds`, `y`), reusing `FluDataLoader` | ☐ |
| 1.3 | Wrap an initial bank (e.g. AutoARIMA/ETS/Theta from statsforecast; XGBoost/LightGBM via mlforecast for continuity; NHITS/LSTM via neuralforecast) behind one interface | ☐ |
| 1.4 | Quantile support: map model outputs to our levels `[0.05, 0.25, 0.5, 0.75, 0.95]` so `PhaseEvaluator` WIS/coverage work unchanged | ☐ |
| 1.5 | Plug into the adapter contract (`run_pipeline(config)` with a `model.family` key) so the existing orchestrator can drive any bank model | ☐ |

**Acceptance:** one command runs ≥3 Nixtla models on the pinned split and emits
phase-aware WIS per model. Target: working demo by the next Thursday sync.

## Workstream 2 — Model selection stage (TS-Agent Stage 1)

| # | Task | Notes |
|---|---|---|
| 2.1 | Warm-up runs: cheap short-horizon fits of each bank model at the cutoff before committing to an incumbent | ☐ |
| 2.2 | Incumbent selection by WIS (or the specified goal metric); hand off to the existing refinement loop | ☐ |
| 2.3 | Goal-as-parameter: selection/refinement objective specified per run (peak performance vs. average vs. overall), settable via natural-language command | ☐ |

## Workstream 3 — Orchestrator fixes (from the gap analysis, ranked)

| # | Task | Notes |
|---|---|---|
| 3.1 | **Revert-on-regression** — restore previous config when an iteration regresses instead of committing it (orchestrator step 8). Few lines; do first | ☐ |
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

1. **This week:** 0.3–0.4, then Workstream 1 (Nixtla demo for Thursday).
2. **Next:** 3.1 (revert-on-regression, trivial), then Workstream 2.
3. **Then:** 4 (warm-start layer) and 5 (knowledge bank) in parallel — both feed the paper.
4. **Ongoing:** 6 (reporting) as soon as multi-model runs exist; 7 from November.

Per repo conventions: each workstream lands on its own `feature/…` branch,
squash-merged into `main` at milestone quality.
