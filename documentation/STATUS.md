# Status

Milestone log for the agentic forecasting framework. Read on demand; `AGENTS.md` holds the
working instructions. Newest work is at the bottom of each section.

## Milestones

**Milestone 1: LLM summarization** - complete. `python -m agent summarize` computes
phase-aware metrics and writes a natural-language performance report.

**Milestone 2: improvement loop** - complete. Evaluate -> Agent 1 diagnoses -> Agent 2
proposes an action -> validate -> apply -> retrain -> re-evaluate (`agent/orchestrator.py`,
`agent/run_tracker.py`, callable pipeline `src/pipeline.py`). Sample weights wired through
all ensemble types; feature-group toggling through `FeatureEngineer`. Seven constrained
actions: `adjust_hyperparameter`, `reweight_training_samples`, `toggle_feature`,
`adjust_floor_constraint`, `change_target_transform`, `set_training_window`, `stop`.

**Data-quality agent** - complete. `python -m agent check-data` runs pre-training checks
(missing weeks, nulls, spikes, zero-reporting, coverage); the LLM interprets findings.

**Milestone 3: tracking and history** - complete. `history`, `status`, `compare`;
reweighting by phase, horizon, and location.

**Stage 1: model bank and selection (TS-Agent Stage 1)** - complete 2026-09-02.
`src/model_bank/` (contract, registry, data bridge, runner); 9 families (2 baselines, the
in-repo XGBoost and PyTorch ensembles, 5 Nixtla examples); `agent/model_selection.py` and
`select-model` (rolling-origin warm-up on the pinned split, phase-aware WIS, incumbent by
`SelectionGoal(metric, phase)`); `improve --model-family X` refines any family.

**Run report and curve-based phases** - landed 2026-09-11. `agent/run_report.py` (roadmap
6.1) writes `report.md` + `report.json` per improve run. `agent/phase_segmentation.py`:
surge/plateau/decline segmentation from the curve. NL goal parser (roadmap 2.3) demoted to
`scrap/goal_parser/` on 2026-09-30 (A. Adiga, 2026-09-10: cosmetic). Open review findings on
the report: `documentation/handoff-2026-09-11-desktop.md` §3.

**Knowledge bank v1** - landed 2026-09-30 (design: `documentation/knowledge-bank-design.md`;
priority set 2026-09-10).
- Entry schema, SQLite store, curated YAML intake, query, `knowledge` CLI, KNOWN FACTS
  renderer (`agent/knowledge/`). Seeded entries in `knowledge/curated/`: the five
  rectification actions from A. Adiga (Teams ~2026-09-27), training-strategy rules, domain
  context migrated out of `FluForecastAdapter.get_domain_context`.
- Lab intake stub: `knowledge/curated/_training-strategy-by-model-class.yaml` (rename to
  activate).
- Unit 4 (2026-10-01): per-proposal retrieval on weak phase + model + metric, entries shown
  with ids, `cited_entries` recorded (unretrieved citations dropped and logged),
  `supported_by` / `supported_by_params_mismatch` / `retrieved_entry_ids`; report shows
  `cites:`; `improve --no-knowledge` switches off the KNOWN FACTS block and retrieval.
- Unit 5 slice (2026-10-01): `knowledge import-experiment` writes the sweep log as
  experiential entries (confidence low, one observation each).
- `set_training_window` (2026-10-01): window entries now recommend it (12 / 12 / 52 weeks).

**Peak-rectification harness** - `scripts/experiments/peak_rectification.py`. Round 1
(`calendar_v1`, seed 42): `outputs/experiments/peak_rectification/`. Round 2 (Adiga phase
windows, phase x horizon WIS, oversampling, seeds 42-44):
`outputs/experiments/peak_rectification_round2/`.

**Milestone 4: generalization** - planned. Template adapter for new domains.

## Deferred

- Derived-knowledge refresh jobs; the end-of-run experiential bank-writer that merges
  repeated observations (design doc §9).
- Rivanna port.

## Pointers

- Roadmap: `documentation/fall-2026-roadmap.md`
- Model bank plug-in recipe: `documentation/MODEL_BANK.md`
- Experiments: `documentation/experiments/`
- Weekly reports: `documentation/reports/`
