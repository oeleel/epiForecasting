# Draft paper outline

Draft for sign-off by A. Adiga. Started 2026-10-15 from the framing given in the 2026-10-15 sync
(`documentation/meeting-notes/2026-10-15-sync.md`). Every claim about the system is tied to a file in
this repo; anything not built is marked PLANNED or IN PROGRESS. `[cite]` marks a reference still to be
found; no citation below has been invented.

Status legend:

- **EXISTS** - code with tests (`tests/agent/`, 477 tests on `feature/experiment-round-2`, 2026-10-15).
- **IN PROGRESS** - code exists, results do not yet (round 2 of the peak-rectification sweep is running).
- **PLANNED** - designed or agreed, not built.

Timeline constraint (2026-09-03 sync): January venue primary, February fallback; November is exams;
winter break is for writing and experiments.

---

## 1. Working title options

1. *An Agentic Test Bench for Evaluating and Refining Epidemic Forecasting Models*
2. *Phase- and Horizon-Aware Agentic Refinement of Influenza Hospitalization Forecasts*
3. *Constrained LLM Agents with a Curated Knowledge Bank for Epidemic Forecast Model Refinement*

Recommendation: option 1. It matches Adiga's stated objective (a test bench to evaluate and refine models)
and does not promise a refinement result before one exists. Options 2 and 3 fit only if the results land
on phase/horizon findings or on the knowledge bank respectively.

## 2. Problem statement and objective

Epidemic forecasting groups maintain several model families and must repeatedly decide which model to
trust, where it fails, and what change to its training would fix that failure. In practice this is a manual
loop: run a model, inspect error tables, adjust a parameter or the training data, and run again. Season-level
scores hide where a model fails, because errors concentrate in specific epidemic phases (most visibly the
peak) and grow with forecast horizon, and a change that helps one segment can hurt another while leaving the
pooled score nearly unchanged. Objective: an agentic framework that serves as a test bench to evaluate
forecasting models on a fixed protocol, disaggregate their weighted interval score (WIS) by epidemic phase and
forecast horizon, and refine them through a small, validated set of training actions proposed by a local LLM,
guided by a curated knowledge bank of the lab's practices and the framework's own past outcomes, with a
researcher deciding when to test, which phases matter, and which strategies are admissible.

## 3. Contributions

Only what the code supports or what is planned and marked so.

- **C1. A model-agnostic test bench for probabilistic epidemic forecasts (EXISTS).** One `ForecastModel`
  contract (`src/model_bank/contract.py`) with validation at the seam; 9 families shipped (2 baselines, the
  lab's XGBoost direct ensemble, a PyTorch quantile ensemble, 5 Nixtla examples); in-house models plug in by
  dotted path without editing the repo (`documentation/MODEL_BANK.md`). The plug-in path is exercised only on
  the repo's own families; the first in-house onboarding is PLANNED (C5).
- **C2. Phase- and horizon-disaggregated WIS as the evaluation surface (EXISTS).**
  `agent/phase_evaluator.py` scores WIS, 95% coverage and signed bias by phase, horizon and location under a
  named phase scheme (default: Adiga's windows); every metrics dict records its scheme and cross-scheme
  comparisons are refused.
- **C3. A constrained-action refinement loop that runs on a small local LLM (EXISTS).** Two-agent loop
  (diagnose, propose) over a validated action catalog with numeric guardrails, accept-or-revert against the
  best-so-far, and a full audit trail (`agent/orchestrator.py`, `agent/adapters/flu_forecast.py`,
  `agent/run_tracker.py`). Default model Qwen 3 8B via Ollama; the LLM edits configs, never model code.
- **C4. A provenance-typed knowledge bank wired into every proposal (EXISTS v1; evaluation PLANNED).**
  Curated / derived / experiential entries in SQLite compiled from YAML; deterministic context-keyed retrieval;
  checked citations; advisory override logging; `--no-knowledge` ablation switch (`agent/knowledge/`,
  `documentation/knowledge-bank-design.md`).
- **C5. An empirical case study on 2025-26 US influenza hospitalizations (PARTIAL).** Model-bank comparison
  (EXISTS, untuned, preliminary); peak-rectification ablation (round 1 EXISTS, round 2 IN PROGRESS); onboarding
  an in-house lab model and vanilla vs refined comparison (PLANNED, blocked on Adiga naming a model and code
  access).

Candidate contribution, scope to be decided (see open questions): warm start vs retrain across phase changes
(`documentation/fall-2026-roadmap.md` Workstream 4, all items PLANNED; `ForecastModel.get_state` / `set_state`
hooks exist but nothing calls them).

## 4. Related work

Sources: `documentation/ts-agent-paper-summary.md`, `documentation/ts-agent-comparison-notes.md`,
`documentation/phase-based-training-papers.md`.

- **Structured agentic workflows for time-series modeling.** TS-Agent (Ang, Bao, Jiang, Tao, Tung, Szpruch, Ni,
  NeurIPS 2025): model selection, code refinement and fine-tuning guided by read-only case, refinement and
  code banks; accept-only-if-loss-improves; reports smaller backbone gaps when refining vetted implementations
  instead of writing models from scratch. Closest prior work.
  - Shared ground: selection stage, accept-or-revert, audit trail, external knowledge banks.
  - Differences to claim: epidemic phase and horizon disaggregation as the evaluation surface; config-space
    actions instead of code edits, chosen so a small local LLM stays safe; human approval per action;
    provenance-typed bank with experiential entries from the framework's own runs.
  - Their backbone-agnosticism result supports the constrained-action design; we have not reproduced it.
- **AutoML and hyperparameter search.** AutoGluon, Optuna, Auto-WEKA [cite each]: static search over generic
  losses, no domain knowledge, no diagnosis. Optuna is used in this repo for the base model's tuning
  (`scripts/optimize.py`).
- **General LLM agents for data science.** AutoGPT, DS-Agent, ResearchAgent [cite each]: end-to-end automation,
  weaker on auditability; TS-Agent uses DS-Agent and ResearchAgent as baselines.
- **Phase-informed epidemic forecasting (the lab's own work).**
  - Adiga et al., *Phase-Informed Bayesian Ensemble Models Improve Performance of COVID-19 Forecasts*,
    AAAI-23 / IAAI-23: phase-matched training sets for ensemble weights; gains concentrated in surge phases;
    model class affinities by phase (SEIR, LSTM, AR / Kalman).
  - Adiga et al., *Enhancing COVID-19 Ensemble Forecasting Model Performance Using Auxiliary Data Sources*,
    IEEE Big Data 2022: piecewise-linear phase segmentation with a 10% rule, transfer-entropy leading
    indicators, phase prediction.
  - Segmentation method: R package `segmented` (Muggeo 2008) [cite].
  - Framing to borrow: report per-phase results because the win concentrates in critical phases.
- **Probabilistic forecast scoring.** WIS for quantile forecasts [cite]; CDC FluSight hub and its target data
  [cite].
- **Forecasting libraries used as example families.** statsforecast, mlforecast, neuralforecast (Nixtla)
  [cite]; NHITS [cite]; XGBoost [cite]; LightGBM [cite].
- **Learning from experience as RL memory.** The bank stores (state, action, reward) rows, framed as RL
  memory per the 2026-09-09 sync; needs a reference for experience replay / case-based reasoning in agents
  [cite]. Knowledge-graph vs flat-memory trade-off in `knowledge-bank-design.md` section 7 [cite the sources
  summarized there].

## 5. System

### 5.1 Model bank and selection (TS-Agent Stage 1 analogue) - EXISTS

- `ForecastModel` contract: `fit(history)`, `predict(history)`, `param_space()`, `default_params()`; long
  format in, quantile columns `q05..q95` out; validated by `validate_forecast_frame` (NaN, negatives, horizons,
  missing series raise; quantile crossings repaired and counted).
- Registry: `@register` or dotted path `pkg.mod:Class`; `python -m agent list-models`.
- Families: `persistence`, `seasonal_naive`, `xgboost_direct`, `nn_quantile`, `sf_autoarima`, `sf_autoets`,
  `sf_autotheta`, `mlf_lightgbm`, `nf_nhits`. Nixtla families are scaffolding; in-house lab models are the
  intended bank.
- Selection: `agent/model_selection.py` runs a rolling-origin warm-up of every family over the pinned
  cutoffs, scores phase-aware WIS, picks the incumbent by `SelectionGoal(metric, phase)`; one family failing
  does not abort the comparison.
- Not built: parallel warm-up (serial today); knowledge-bank retrieval at selection (design injection point A,
  PLANNED).

### 5.2 Improvement loop with constrained actions (TS-Agent Stage 2 analogue) - EXISTS

- Plain-Python orchestrator (`agent/orchestrator.py`): evaluate, Agent 1 diagnoses, Agent 2 proposes, validate,
  apply, retrain, re-evaluate.
- Action catalog (`FluForecastAdapter.ACTION_CATALOG`): `adjust_hyperparameter` (generated from each family's
  `param_space()` for bank families), `reweight_training_samples`, `toggle_feature`, `adjust_floor_constraint`,
  `change_target_transform`, `set_training_window` (8-104 weeks), `stop`.
- Guardrails: schema and range validation on every action (for example reweighting weights in [1, 5]);
  LLM output repair-retry.
- Accept-or-revert: compares against the best-so-far iteration and restores its config when an iteration does
  not beat it; stops after two consecutive iterations that fail to beat the best-so-far (`regressed_x2`;
  equal-to-best counts, an improvement resets the streak), or after two consecutive marginal improvements
  below the no-improvement threshold (`no_improvement_x2`).
- Ledger: every iteration's metrics, diagnosis, action and config in SQLite (`outputs/agent_runs/runs.db`).
- Legacy-only actions: feature toggles, reweighting, floor and target transform act on `xgboost_direct`; bank
  families get `adjust_hyperparameter` only. Worth stating as a limitation.
- Not built: chain-of-edits factorization (strategy, then knob); case retrieval over `runs.db` (roadmap 3.2).

### 5.3 Knowledge bank - EXISTS (v1)

- Provenance-first taxonomy: curated (human), derived (computed from data), experiential (from runs);
  categories as tags.
- Storage: curated YAML in `knowledge/curated/` compiled into SQLite `knowledge/knowledge.db`;
  `python -m agent knowledge validate | rebuild | list | query | import-experiment | remove`.
- Seeded content: Adiga's five rectification actions, training-strategy rules, domain context; 9
  experiential entries imported from the round-1 sweep (confidence low, one observation each).
- Retrieval: deterministic SQL filter on phase, model, metric; ordering by provenance trust, confidence,
  observation count. No embeddings, no LLM-written queries.
- Injection: a `KNOWN FACTS` block in prompts, plus per-proposal retrieval with entry ids; the LLM is asked to
  cite ids (not enforced; a missing citation list is accepted); cited ids that were not in the retrieved set
  are dropped and logged; `supported_by` and override log lines are recorded.
- Guardrails are advisory with an override log, not hard blocks.
- Ablation switch: `improve --no-knowledge`.
- Format decision (2026-10-15): YAML stays; the advisor agent reads it as is.
- Not built: derived-knowledge jobs (unit 3, blocked on onset threshold), end-of-run bank-writer that merges
  repeated observations (unit 5 remainder), persisted override table, behavioral evaluation (unit 7), log
  summarization agent (deferred by Adiga, 2026-10-15).

### 5.4 Run report - EXISTS

- `agent/run_report.py`: every `improve` run writes `report.md` and `report.json`; `python -m agent report
  <run_id> [--rebuild]` regenerates.
- Contents: what was tried per iteration with deltas, by-phase and by-horizon tables, cited bank entries,
  caveats, final recommendation.
- Refuses to claim an improvement across different evaluation windows (reported as inconclusive).
- Not built: cross-run assessment view (roadmap 6.2).

### 5.5 Phase-aware and horizon-aware evaluation

- **EXISTS:** `PhaseEvaluator` computes WIS, 95% coverage, MAE, MAPE and signed bias overall, by phase, by
  horizon, by location; `identify_worst_segments` feeds the diagnosis prompt.
- **EXISTS:** calendar_v1 phase scheme, month-based (`PHASE_SCHEMES["calendar_v1"]`, formerly `PhaseEvaluator.PHASE_MAP`): onset Oct-Nov, peak Dec-Jan,
  decline Feb-Apr, off_season May-Sep.
- **EXISTS:** curve-based surge / plateau / decline segmentation after the lab's method
  (`agent/phase_segmentation.py`); not yet used for scoring.
- **EXISTS:** Adiga's phase windows (2026-10-15, `PHASE_SCHEMES["adiga_2026_10"]`, the default): growth Nov 1 - Dec 31, peak Jan 1 - end of Feb,
  decline Mar 1 - Apr 15, plateau Apr 16 - May 31, off_season Jun 1 - Oct 31. October as off_season is an
  ASSUMPTION pending Adiga. Date-based, not month-based, because April is split.
- **EXISTS:** WIS by horizon (1-4 weeks) in the summarize and diagnosis prompts
  (`agent/prompt_templates.py`), the run report (`agent/run_report.py`) and the harness `log.jsonl`
  (`metrics.by_horizon`).
- **EXISTS:** phase x horizon WIS in the harness log and `summary.md`; `--rescore` re-scores kept forecasts
  under the current scheme without retraining.

### 5.6 Experiment harness - EXISTS

- `scripts/experiments/peak_rectification.py`: fixed config written to `manifest.json` with provenance (git
  SHA, data-cache time, library versions); one knob per config; resumable; one (state, action, reward) JSON row
  per config in `log.jsonl`, the same shape as an experiential bank entry.
- Scoring reuses the loop's scorer and `_is_better`; no private metric code.
- Arms: baseline, approaching-peak lambda, calendar-peak lambda, training window, phase oversampling (EXISTS);
  `--seeds`, `--merge` for per-seed parallel runs; manifest pins seeds, phase scheme and data-cache sha256.
  Round-2 results IN PROGRESS.

## 6. Human in the loop

Where the researcher decides, and how the framework records it.

- **What to test:** the researcher picks families for `select-model` and the family and cutoff for `improve`
  (EXISTS).
- **Which phases matter:** the researcher sets the goal `SelectionGoal(metric, phase)` and the target metric
  (EXISTS); the phase scheme itself is a lab decision (calendar_v1 vs Adiga windows vs curve-based).
- **Which strategies are admissible:** the action catalog and its guardrail ranges are fixed by the
  researcher in code (EXISTS); curated bank entries encode the lab's strategies and are reviewed via PR
  (EXISTS).
- **Per-action approval:** shadow mode by default; each proposed action prompts y / n / stop; `--auto-apply`
  skips it (EXISTS, `agent/cli.py`).
- **Override of the bank:** advisory, logged when the agent proposes against bank recommendations (EXISTS as
  a log line). A researcher directive that outranks the bank (precedence user > bank > model priors, raised by
  Adiga 2026-09-09) is PLANNED.
- **Promotion of experience:** experiential entries enter at low confidence; merging and promotion need the
  bank-writer (PLANNED) and human review.
- **Paper evidence to collect:** how often the human rejects proposals, how often the agent overrides the bank,
  and whether those overrides were right (needs the persisted override table, PLANNED).

## 7. Evaluation protocol

- **Data:** CDC FluSight weekly influenza hospital admissions, state level; national `US` row excluded from
  scoring.
- **Pinned split** (`src/config.py`, decided 2026-09-02): train from `TRAIN_START_DATE` 2022-02-05; evaluate
  `EVAL_START_DATE`..`EVAL_END_DATE` = 2025-10-01..2026-05-31.
- **Cutoffs:** `generate_eval_cutoffs(4)`: 9 Saturday origins at stride 4: 2025-10-04, 11-01, 11-29, 12-27,
  2026-01-24, 02-21, 03-21, 04-18, 05-16. Rolling origin; history is truncated at each cutoff.
- **Forecast target:** horizons 1-4 weeks; quantiles 0.05, 0.25, 0.5, 0.75, 0.95.
- **Score:** WIS is the only keep-or-drop score (lab practice, 2026-10-15). Coverage, bias, MAE reported for
  diagnosis only.
- **Disaggregation:** WIS by phase, by horizon, and by phase x horizon; pooled numbers always shown next to
  the horizon breakdown, because the pooled mean hid opposite effects in round 1.
- **Phase scheme:** Adiga windows as primary (EXISTS); calendar_v1 kept for comparison with round 1;
  curve-based as a robustness check (re-scoring kept forecasts, no retraining).
- **Seeds:** round 1 used one seed (42); five parallel baselines were identical, so the harness is
  deterministic, but seed variance is unmeasured. Round 2 uses 3 seeds (running); report mean and spread,
  and treat differences inside the seed spread as no effect.
- **Comparisons:** every delta against a baseline row in the same output directory, never across runs (the
  09-02 model-bank-path run and the 09-30 pipeline-path run gave different XGBoost baselines, peak WIS 122.37
  vs 121.01; cause not traced, code path or data vintage).
- **Vanilla vs refined:** protocol not settled. `improve` refines at a single `--cutoff-date`; scoring the
  refined config on the same window it was tuned on would overstate the gain. Proposal: refine on cutoffs
  from an earlier season (2024-25), then score vanilla and refined on the pinned 2025-26 cutoffs. Needs Adiga's
  sign-off (open question 6).
- **Knowledge-bank ablation:** the same `improve` runs with and without `--no-knowledge` (PLANNED, design unit
  7).

## 8. Results (placeholders)

No result below is final. Numbers shown are preliminary and labeled with their source.

### 8.1 Models onboarded

| model | source | type | onboarded | notes |
|---|---|---|---|---|
| 9 repo families | `src/model_bank/` | baselines, boosted, neural, classical | yes | Nixtla families untuned examples |
| in-house model 1 | lab | TBD | no | blocked: Adiga to name model and grant code access |
| in-house model 2..N | lab | TBD | no | count is open question 2 |

- Placeholder: table of WIS overall, by phase, by horizon per family on the pinned split, multi-seed.
- Preliminary (2026-09-02, `outputs/model_selection/demo_2026-09-02.json`, calendar_v1, one seed, no tuning;
  six of the nine families compared, `nn_quantile`, `sf_autoarima`, `sf_autotheta` not run): `xgboost_direct`
  best overall (WIS 50.6); peak is the worst phase for each of the six; `nf_nhits` best in decline.
  Needs re-scoring under Adiga windows and by horizon.

### 8.2 Vanilla vs refined

- Placeholder: per model, vanilla WIS vs refined WIS by phase and horizon on held-out cutoffs, with the
  actions the loop took and the bank entries it cited.
- Placeholder: same with `--no-knowledge`.
- Anecdote available, not a result: 2026-09-03 live run on `nf_nhits` at 2026-01-24, where revert-on-regression
  stopped a 52-week lookback from compounding (`documentation/fall-2026-roadmap.md` 3.1).

### 8.3 Peak-rectification ablation

- Round 1 (EXISTS, `documentation/experiments/2026-10-peak-rectification-results.md`; calendar_v1; one seed):
  - No arm beat baseline on peak WIS (121.01); approaching-peak lambda moved peak bias from -58.8 to
    -35.7..+7.9 (overshooting at lambda_5) but peak WIS rose 2.53-2.86; calendar-peak lambda rose 1.65-2.69;
    peak 95% coverage fell.
  - Overall WIS by horizon h1..h4: baseline 29.1 / 36.9 / 53.8 / 80.4; lambda_5 25.2 / 36.0 / 57.4 / 83.5;
    lambda_calendar_3 25.5 / 34.5 / 57.1 / 83.8; window_52 39.5 / 51.5 / 80.4 / 116.0. Upweighting helps
    h1-h2 and hurts h3-h4; the pooled mean hid it.
  - Unconditional short training windows (12 / 26 / 52 weeks) worsened every phase for `xgboost_direct`.
- Round 2 (IN PROGRESS): Adiga windows, oversampling arm, 3 seeds, WIS by phase x horizon. Placeholder table:
  arm x horizon, peak WIS mean +- spread.
- Open thread: whether the h1-h2 gain survives seeds, and whether a horizon-specific weight (only on the
  h1-h2 models) is worth an arm.

## 9. Limitations

- One evaluation season (2025-26) and one data source; results may not generalize across seasons or diseases.
- Phase definitions are subjective and the scheme changed mid-project; round 1 and round 2 numbers are not
  directly comparable.
- Most actions apply only to the legacy `xgboost_direct` path; bank families get hyperparameter edits only, and
  `set_training_window` is honored only on the legacy path.
- XGBoost's validation split is positional over (location, date), so it holds out locations, not weeks; sample
  weights on those rows are unused.
- Small local LLM (Qwen 3 8B) by default; Qwen 2.5 72B on Rivanna is planned but compute access is not
  settled. Backbone comparison not run.
- Knowledge bank v1: experiential entries are single observations; no merging; overrides are log lines, so
  "does memory change behavior" is not yet measurable.
- No real-time operation: hindcasts on finalized data, no backfill / revision handling. Real-time weekly
  retraining is the end goal, out of scope.
- Nixtla families are untuned; comparisons against them say little about those model classes.

## 10. Open questions for Adiga

1. **Venue and deadline.** Which January venue, and which February fallback? Page limit and format decide how
   much of sections 5-6 survives.
2. **How many models.** How many in-house models should the paper report as onboarded, and are the 9 repo
   families (with untuned Nixtla examples) worth reporting at all?
3. **Which in-house models, and access.** Which model first, and who grants code access? This gates the agreed
   success metric.
4. **October.** Off_season (current assumption) or growth (growth window would become Oct 1 - Dec 31)?
5. **Real-time scope.** Is any real-time or weekly-retraining result in scope for this paper, or is it framed as
   future work?
6. **Vanilla vs refined protocol.** Refine on 2024-25 cutoffs and score on 2025-26 (proposed), or another split?
7. **Warm start vs retrain.** Contribution of this paper, or future work?
8. **Framing.** Epidemic specialization of TS-Agent (phase/horizon evaluation, small-LLM safety, lab knowledge
   bank), or a general test bench demonstrated on flu?
9. **Authorship and lab content.** Who contributes the curated per-model-class strategies
   (`knowledge/curated/_training-strategy-by-model-class.yaml` is still a stub), and how are they credited?
