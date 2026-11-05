# Draft paper outline

Draft for sign-off by A. Adiga. Started 2026-10-15 from the framing given in the 2026-10-15 sync
(`documentation/meeting-notes/2026-10-15-sync.md`); updated 2026-11-04 for round 2, the weekly hindcast, the
inspection round and the 2026-11-05 scope decision (hypothesis-driven runs with a significance test). Every claim about the system is tied to a file in
this repo; anything not built is marked PLANNED or IN PROGRESS. `[cite]` marks a reference still to be
found; no citation below has been invented.

Status legend:

- **EXISTS** - code with tests (`tests/agent/`, 516 tests on `main`, 2026-11-04).
- **IN PROGRESS** - code exists, results do not yet.
- **PLANNED** - designed or agreed, not built.

Timeline (2026-10-21 sync): paper deadline in the first or second week of January 2027; exams in early
December; winter break is for writing and experiments. A. Adiga is writing a separate, more formal Bayesian
treatment of the framework (2026-10-21) and will share the relevant parts.

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
  `agent/phase_evaluator.py` scores WIS, 90% interval coverage (q0.05-q0.95, stored as `coverage_95`) and
  signed bias by phase, horizon and location under a named phase scheme (default: Adiga's windows); every metrics dict records its scheme and cross-scheme
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
  (EXISTS, untuned, preliminary); peak-rectification ablation (rounds 1 and 2 EXISTS, weekly hindcast and
  inspection EXISTS, out-of-sample gate test PLANNED); onboarding
  an in-house lab model and vanilla vs refined comparison (PLANNED, blocked on Adiga naming a model and code
  access).

- **C6. Hypothesis-driven refinement with a significance test (test and held-out split EXIST; agent run
  mode PLANNED; scope set 2026-11-05).** The
  researcher (or the agent) states a hypothesis as one action fixed before the run; the framework runs it and
  reports whether the gain over baseline, and over previously accepted changes, is statistically significant.
  A. Adiga's Bayesian treatment frames this as a posterior that the change caused the gain. See section 7.

Candidate contribution, scope to be decided (see open questions): warm start vs retrain across phase changes
(`documentation/fall-2026-roadmap.md` Workstream 4, all items PLANNED; `ForecastModel.get_state` / `set_state`
hooks exist but nothing calls them).

## 4. Related work

Sources: `documentation/ts-agent-paper-summary.md`, `documentation/ts-agent-comparison-notes.md`,
`documentation/phase-based-training-papers.md`, `documentation/ts-scientist-paper-summary.md`,
`documentation/related-work-survey.md` (47 works, each checked against arXiv or doi.org; links there).

- **Structured agentic workflows for time-series modeling.** TS-Agent (Ang, Bao, Jiang, Tao, Tung, Szpruch, Ni,
  NeurIPS 2025): model selection, code refinement and fine-tuning guided by read-only case, refinement and
  code banks; accept-only-if-loss-improves; reports smaller backbone gaps when refining vetted implementations
  instead of writing models from scratch. Closest prior work.
  - Shared ground: selection stage, accept-or-revert, audit trail, external knowledge banks.
  - Differences to claim: epidemic phase and horizon disaggregation as the evaluation surface; config-space
    actions instead of code edits, chosen so a small local LLM stays safe; human approval per action;
    provenance-typed bank with experiential entries from the framework's own runs.
  - Their backbone-agnosticism result supports the constrained-action design; we have not reproduced it.
- **TimeSeriesScientist (TSci)** (Zhao et al. 2025, arXiv:2510.01538): Curator, Planner, Forecaster and
  Reporter agents for univariate point forecasting; the Planner narrows a 21-model library by validation MAPE.
  - Differences to claim: no knowledge bank (listed as their future work); pooled MAE / MAPE on point
    forecasts, no interval scoring, no phases; one pass, no accept-or-revert; no seeds, variance or
    significance tests. On ILI, their only health dataset, Prophet has about half TSci's MAE (their Fig. 11,
    p. 22), which the paper does not discuss.
  - To borrow: the missing control (search the action space by validation alone at equal compute) is one our
    paper also needs.
- **LLM agents for respiratory-disease forecasting.** ERA (Martinson, Brenner et al. 2026, arXiv:2605.16238):
  LLM-guided tree search writes flu, COVID and RSV forecasting code; its ensemble ranked first in the CDC
  2025-26 post-season evaluation. Same task, metric and season as ours, so this paper does not compete on
  accuracy; it claims auditable, phase-aware refinement of a lab's own model and a statistical account of each
  change. EpiEvolve (Lu et al. 2026, arXiv:2606.05513): episodic memory with regime-aware retrieval for COVID
  hospitalization trends, the closest analogue to the phase-keyed bank.
- **Hypothesis validation and Bayesian experimentation.** Hypothesis learning in automated experiment
  (Ziatdinov et al. 2021, arXiv:2112.06649; named by A. Adiga as the model for the abductive loop): competing
  hypotheses as structured Gaussian-process mean functions, with a policy that picks the next experiment.
  POPPER (Huang et al., ICML 2025, arXiv:2502.09858): LLM-designed falsification tests with sequential e-values
  and Type-I error control, the closest template for C6. BED-LLM (arXiv:2508.21184) and AutoDiscovery
  (arXiv:2507.00310) for Bayesian framing. Diebold-Mariano (1995) for paired forecast comparison.
- **AutoML and hyperparameter search.** AutoGluon, Optuna, Auto-WEKA [cite each]: static search over generic
  losses, no domain knowledge, no diagnosis. Optuna is used in this repo for the base model's tuning
  (`scripts/optimize.py`).
- **General LLM agents for data science.** AutoGPT [cite], DS-Agent (arXiv:2402.17453), ResearchAgent [cite]: end-to-end automation,
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
- **Probabilistic forecast scoring.** WIS for quantile forecasts (Bracher, Ray, Gneiting, Reich 2021, PLOS
  Comput. Biol.; DOI in the survey); CDC FluSight hub and its target data
  [cite].
- **Forecasting libraries used as example families.** statsforecast, mlforecast, neuralforecast (Nixtla)
  [cite]; NHITS [cite]; XGBoost [cite]; LightGBM [cite].
- **Learning from experience as RL memory.** The bank stores (state, action, reward) rows, framed as RL
  memory per the 2026-09-09 sync; experience memory in agents: Reflexion (arXiv:2303.11366), ExpeL
  (arXiv:2308.10144); knowledge-guided ML agents: MLCopilot (EACL 2024, arXiv:2304.14979), AutoMind
  (arXiv:2506.10974). Knowledge-graph vs flat-memory trade-off in `knowledge-bank-design.md` section 7 [cite the sources
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
- Target: the loop optimizes one pooled metric (`wis`, `mape`, `mae`, `rmse`, `coverage_95`, `bias`) at a
  single `--cutoff-date` (`agent/orchestrator.py` `_extract_target`); it cannot target peak WIS or a horizon.
  Phase- and horizon-targeted experiments run in the harness (5.6), not in the agent.
- Accept rule: any strictly lower point estimate is accepted (`_is_better`); the 1% no-improvement threshold
  is also a point-estimate rule. No significance test (C6, PLANNED).
- Phase oversampling and horizon-specific emphasis exist in the harness only, not as agent actions. The
  `reweight_training_samples` horizon dimension cannot express "emphasis on h1-h2 only", because each
  per-horizon model sees one horizon and a uniform weight targets nothing. A per-horizon phase-emphasis action
  is PLANNED.
- Not built: a fixed-hypothesis run mode (execute one given action and test it, C6); chain-of-edits
  factorization (strategy, then knob); case retrieval over `runs.db` (roadmap 3.2).

### 5.3 Knowledge bank - EXISTS (v1)

- Provenance-first taxonomy: curated (human), derived (computed from data), experiential (from runs);
  categories as tags.
- Storage: curated YAML in `knowledge/curated/` compiled into SQLite `knowledge/knowledge.db`;
  `python -m agent knowledge validate | rebuild | list | query | import-experiment | remove`.
- Seeded content: Adiga's five rectification actions, training-strategy rules, domain context; 9
  experiential entries imported from the round-1 sweep (calendar_v1, confidence low, one observation each).
  Round 2 is not imported: the importer expects a seedless baseline row, and an honest entry needs the 3 seeds
  merged into one observation set (bank-writer work).
- Stale against the evidence: the curated short-window entry recommends 12 weeks at high confidence, while
  every window arm lost (by about 68 peak WIS in round 2 at 52 weeks, 44-78 in the weekly hindcast). The bank
  has no mechanism yet to reconcile a curated prior with contradicting experience.
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

- **EXISTS:** `PhaseEvaluator` computes WIS, 90% interval coverage (q0.05-q0.95; the metric is named
  `coverage_95` for compatibility, and its target was corrected to 0.90 on 2026-11-04), MAE, MAPE and signed
  bias overall, by phase, by horizon, by location; `identify_worst_segments` feeds the diagnosis prompt.
- **EXISTS:** WIS is computed on the five emitted quantiles (two central intervals and the median), not on
  FluSight's 23. Deltas between configurations are valid; absolute values are not comparable with hub-scored
  numbers.
- **EXISTS:** calendar_v1 phase scheme, month-based (`PHASE_SCHEMES["calendar_v1"]`, formerly `PhaseEvaluator.PHASE_MAP`): onset Oct-Nov, peak Dec-Jan,
  decline Feb-Apr, off_season May-Sep.
- **EXISTS:** curve-based surge / plateau / decline segmentation after the lab's method
  (`agent/phase_segmentation.py`); not yet used for scoring. The inspection (8.4) shows errors follow each
  state's own turn, not the calendar, so re-scoring under per-state curve phases is PLANNED.
- **EXISTS:** Adiga's phase windows (2026-10-15, `PHASE_SCHEMES["adiga_2026_10"]`, the default): growth Nov 1 - Dec 31, peak Jan 1 - end of Feb,
  decline Mar 1 - Apr 15, plateau Apr 16 - May 31, off_season Jun 1 - Oct 31. October as off_season is an
  ASSUMPTION pending Adiga. Date-based, not month-based, because April is split.
- **EXISTS:** WIS by horizon (1-4 weeks) in the summarize and diagnosis prompts
  (`agent/prompt_templates.py`), the run report (`agent/run_report.py`) and the harness `log.jsonl`
  (`metrics.by_horizon`).
- **EXISTS:** phase x horizon WIS in the harness log and `summary.md`; `--rescore` re-scores kept forecasts
  under the current scheme without retraining.
- **EXISTS:** WIS per location and per forecast date (`scripts/experiments/export_hindcast_tables.py`,
  cross-checked against `log.jsonl`). Because WIS is in admissions, pooled numbers are dominated by the five
  largest states.
- **EXISTS:** direction agreement as a second evaluation surface (2026-10-21: the lab reads plot direction, not
  only WIS): does the median's change from the last observed value point the same way as the truth, with a
  flat band of max(10% of the last value, 10 admissions), an ASSUMPTION
  (`scripts/experiments/export_hindcast_tables.py`).
- **EXISTS:** lab-style forecast panels (`src/forecast_panels.py`, `scripts/plot_forecasts.py`): observed in
  black, median with 50% and 90% bands for h1-h4, hollow markers for what happened next.

### 5.6 Experiment harness - EXISTS

- `scripts/experiments/peak_rectification.py`: fixed config written to `manifest.json` with provenance (git
  SHA, data-cache time, library versions); one knob per config; resumable; one (state, action, reward) JSON row
  per config in `log.jsonl`, the same shape as an experiential bank entry.
- Scoring reuses the loop's scorer and `_is_better`; no private metric code.
- Arms: baseline, approaching-peak lambda, calendar-peak lambda, training window, phase oversampling (EXISTS);
  `--seeds`, `--merge` for per-seed parallel runs; manifest pins seeds, phase scheme and data-cache sha256.
  Round 2 EXISTS (8.3).
- Weekly hindcast (EXISTS): `--stride-weeks 1` reruns every arm from all 35 weekly origins of the test window
  (13 configs, seed 42); `--pin-data-sha256` refuses to run unless the data cache matches round 2's. It feeds
  the location, week and direction breakdowns; seed-variance claims rest on the 3-seed stride-4 run. Its
  baseline peak WIS (119.73) differs from round 2's (107.28) because the origins differ; the two are never
  compared.
- Gated emphasis (EXISTS, `scripts/experiments/gated_emphasis.py`): no training; each row is taken from an
  emphasis run or the baseline run by a horizon, calendar or curve gate. Because `xgboost_direct` fits one
  model per horizon, the horizon gate equals emphasis trained into h1-h2 only, up to the cross-horizon floor
  (which moves pooled WIS by at most 2e-6). Results are in-sample (8.4).

## 6. Human in the loop

Where the researcher decides, and how the framework records it.

- **What to test:** the researcher picks families for `select-model` and the family and cutoff for `improve`
  (EXISTS).
- **What to hypothesize:** the researcher supplies the hypotheses (one fixed action each) and the framework
  tests them in turn against the baseline and the changes already accepted (PLANNED, C6). For this round the
  framework executes the given hypothesis and does not generate new ones until one wins (2026-11-05).
- **Which phases matter:** the researcher sets `SelectionGoal(metric, phase)` for model selection (EXISTS);
  the refinement loop optimizes a pooled metric at one origin, so phase- or horizon-targeted refinement runs
  in the harness (limitation, 5.2). The phase scheme itself is a lab decision (calendar_v1 vs Adiga windows vs
  curve-based).
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
- **Inspection:** the lab inspects plots before every submission, so the bench ships the same view: a viewer
  that renders any location, origin and configuration from the committed forecast tables
  (`documentation/experiments/weekly_hindcast/viewer.html`, on GitHub Pages), forecast CSVs the lab can plot
  with its own scripts, and the weekly report pages (`documentation/reports/`) (EXISTS).
- **Paper evidence to collect:** how often the human rejects proposals, how often the agent overrides the bank,
  and whether those overrides were right (needs the persisted override table, PLANNED).

## 7. Evaluation protocol

- **Data:** CDC FluSight weekly influenza hospital admissions, state level. The national `US` row is excluded
  because the state-trained model (training max about 4.7k) cannot reach national scale (about 21k at peak);
  its h1 forecasts hit the floor constraint with all quantiles equal, so they are not model forecasts.
- **Pinned split** (`src/config.py`, decided 2026-09-02): train from `TRAIN_START_DATE` 2022-02-05; evaluate
  `EVAL_START_DATE`..`EVAL_END_DATE` = 2025-10-01..2026-05-31.
- **Cutoffs:** `generate_eval_cutoffs(4)`: 9 Saturday origins at stride 4: 2025-10-04, 11-01, 11-29, 12-27,
  2026-01-24, 02-21, 03-21, 04-18, 05-16. Rolling origin; history is truncated at each cutoff. The weekly
  hindcast uses `generate_eval_cutoffs(1)` (35 origins) for inspection only.
- **Forecast target:** horizons 1-4 weeks; quantiles 0.05, 0.25, 0.5, 0.75, 0.95 (5, not FluSight's 23;
  switching needs a rerun of round 2, open question 10).
- **Score:** WIS is the only keep-or-drop score (lab practice, 2026-10-15). 90% coverage, bias, MAE and
  direction agreement reported for diagnosis only.
- **Disaggregation:** WIS by phase, by horizon, and by phase x horizon; pooled numbers always shown next to
  the horizon breakdown, because the pooled mean hid opposite effects in round 1.
- **Phase scheme:** Adiga windows as primary (EXISTS); calendar_v1 kept for comparison with round 1;
  curve-based as a robustness check (PLANNED: re-score kept forecasts, no retraining; nothing scores with it
  yet).
- **Seeds:** round 1 used one seed (42); five parallel baselines were identical, so the harness is
  deterministic. Round 2 used 3 seeds (42, 43, 44); the harness reports mean, sample sd and the min..max of
  same-seed paired deltas. "Inside the seed spread = no effect" is a reading rule, not a test; three seeds
  bound the seed noise but do not estimate it well.
- **Hypothesis-driven protocol (2026-11-05, A. Adiga).** Each run executes one hypothesis given before the
  run (one action) and reports whether its gain over baseline is statistically significant on held-out
  origins. It does not generate further hypotheses until one wins, because a search that stops at the first
  win inflates the false-positive rate.
  - Held-out split (EXISTS): `src/config.py` `EVAL_WINDOWS`, `holdout_2024_25` = 35 weekly origins
    2024-10-05..2025-05-31; harness `--eval-window`. A hypothesis chosen on 2025-26 is confirmed on 2024-25.
  - Test (EXISTS, `agent/significance.py`, `scripts/experiments/significance_test.py`): the forecast date is
    the unit (rows from one fit are not independent). Per-date WIS deltas are scaled by rows per date so their
    mean equals the pooled WIS delta, then averaged over seeds. Primary: two-sided Diebold-Mariano with the
    Harvey-Leybourne-Newbold correction, autocovariance to lag H-1 = 3, alpha 0.05. Secondary: circular block
    bootstrap interval (block 4), descriptive only, because it is too narrow with about 12 dates.
  - Pre-registration: hypothesis, endpoint, test and decision rule committed before the run
    (`documentation/experiments/holdout_2024_25/preregistration.md`).
  - PLANNED: the agent run mode that executes a given action under this protocol, and A. Adiga's Bayesian
    treatment, which may replace the test with a posterior over competing hypotheses.
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
- Preliminary: the 2026-09-02 selection run (`outputs/model_selection/demo_2026-09-02/`, one seed, no
  tuning, 9 stride-4 cutoffs, US excluded; six of the nine families, `nn_quantile`, `sf_autoarima` and
  `sf_autotheta` not run). Its saved forecasts were re-scored on 2026-11-05 under `adiga_2026_10` against
  round 2's pinned data snapshot (sha256 `c91d9d81b8ab...`), no retraining. WIS, 1,872 rows per family:

| family | overall | growth | peak | decline | plateau | h1 | h4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xgboost_direct` | 50.6 | 63.7 | 108.4 | 32.7 | 7.5 | 29.4 | 81.0 |
| `nf_nhits` | 65.5 | 73.6 | 155.2 | 33.2 | 10.1 | 27.1 | 115.2 |
| `sf_autoets` | 82.2 | 102.5 | 167.8 | 56.4 | 18.1 | 27.9 | 144.5 |
| `persistence` | 90.7 | 131.7 | 155.0 | 79.0 | 22.2 | 33.6 | 159.4 |
| `mlf_lightgbm` | 138.7 | 102.0 | 378.3 | 67.7 | 31.3 | 49.8 | 225.2 |
| `seasonal_naive` | 153.0 | 111.1 | 423.2 | 90.3 | 13.7 | 160.9 | 164.2 |

  - `xgboost_direct` is best overall and in every phase. The calendar_v1 claim "`nf_nhits` best in decline"
    does not hold under the Adiga windows (33.2 vs 32.7).
  - Peak is the worst phase for every family.
  - `nf_nhits` and `sf_autoets` beat `xgboost_direct` at h1 (27.1 and 27.9 vs 29.4) and lose badly at h4.
    This is the same short-horizon pattern as peak emphasis, and it is a candidate for a horizon ensemble.
  - One seed and no significance test: these are point estimates.

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
    peak 90% coverage fell.
  - Overall WIS by horizon h1..h4: baseline 29.1 / 36.9 / 53.8 / 80.4; lambda_5 25.2 / 36.0 / 57.4 / 83.5;
    lambda_calendar_3 25.5 / 34.5 / 57.1 / 83.8; window_52 39.5 / 51.5 / 80.4 / 116.0. Upweighting helps
    h1-h2 and hurts h3-h4; the pooled mean hid it.
  - Unconditional short training windows (12 / 26 / 52 weeks) worsened every phase for `xgboost_direct`.
- Round 2 (EXISTS, `documentation/experiments/2026-10-peak-rectification-round2-results.md`; adiga_2026_10;
  3 seeds; 9 stride-4 origins; 11 configs):
  - Pooled over horizons, no arm beats baseline by more than seed noise: baseline peak WIS 107.28 (seed sd
    1.24); best arm peak weight 2 at 105.43 (paired delta -1.86, range -2.96..-0.47).
  - By horizon, peak-phase weighting or oversampling lowers peak WIS at h1-h2 by 4-13 and raises it at h4 by
    9-22, with the same sign in all 3 seeds; h3 crosses zero. The pooled mean hides the trade-off.
  - Oversampling and weighting land within seed noise of each other, as expected for XGBoost without row
    subsampling. The 52-week window is worse at every horizon (+68.50 peak WIS).
  - The weight and oversampling factors (1.5-5) were chosen to find a pattern, not tuned.

### 8.4 Inspection and gated emphasis (weekly hindcast, seed 42)

- Source: `documentation/experiments/weekly_hindcast/` (summary, inspection, direction, by-location and
  by-week tables, gated emphasis).
- Three regimes across 7 inspected states and 11 origins: during the surge every config under-predicts and
  peak emphasis helps; at the turn the h3-h4 median rebounds into a V (14% of post-fall baseline forecasts,
  26-35% with peak emphasis); after the turn the largest states are forecast to collapse (h1 median 0.65 of the
  last value vs 0.83 observed) in every config.
- Direction agreement: baseline 72.5% over all phases, 53.8% in the peak phase.
- By location and week: ungated peak emphasis beats the baseline in only 17-22 of 52 locations, and the five
  largest states drive the pooled gain; emphasis wins the surge weeks (Nov 29 - Dec 20) and loses the two weeks
  after the turn (Jan 3, Jan 10).
- Gated emphasis (in-sample on 2025-26): the horizon gate turns `oversample_5` from a pooled loss (+0.99
  overall, +4.97 peak) into a gain (-1.18 overall, -3.13 peak; 40 of 52 locations better). The calendar gate
  scores best only because this season's turn fell next to Dec 31; it is the overfit reference.
- Under the per-date DM test the in-sample horizon-gate gain is not significant either (p = 0.13, seed 42).

### 8.5 Held-out test of the horizon gate (2024-25, 3 seeds) - EXISTS

- Source: `documentation/experiments/holdout_2024_25/results.md`; pre-registered.
- Primary result: H1 not confirmed. Horizon-gated `oversample_5` changes peak WIS by -0.19 (359.06 ->
  358.87), DM p = 0.99 over 11 dates. Harm guard (all weeks): -0.84, p = 0.82.
- Why: the h1 gain replicates (-23 to -29 peak WIS in every seed) but h2 reverses (+24 to +26 in every seed),
  so the h1-h2 gate cancels. "Emphasis on h1 only" is post hoc and needs a third season (2023-24 is the only
  one left).
- No emphasis configuration is significant on either season. 2024-25 is a far larger season (baseline peak
  WIS about 3 times 2025-26; peak bias about -440), so only paired deltas are compared.

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
  "does memory change behavior" is not yet measurable. One high-confidence curated entry (short windows) is
  contradicted by the framework's own evidence, with no reconciliation mechanism.
- The improve loop and model selection still accept on point estimates; the significance test exists only
  as a harness tool (section 7). Power is low: a season has about 12 peak-phase forecast dates.
- Horizons are fitted independently but coupled at prediction time by the floor constraint (horizon h is
  floored at 0.3 x (1 - 0.05 (h - 1)) of horizon h-1); for the gate experiment the coupling moves pooled WIS
  by at most 2e-6, but any horizon-specific claim must state it.
- WIS on 5 quantiles; absolute values are not comparable with FluSight hub scores.
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
10. **5 vs 23 quantiles.** Switch to FluSight's 23 once, before the results are written (needs a rerun of
    round 2), or keep 5 and report deltas only?
11. **Significance test or posterior.** For C6: a frequentist paired test (block bootstrap over forecast
    dates, Diebold-Mariano) now, or wait for the Bayesian treatment? Which comparison set: baseline only, or
    baseline plus every accepted change?
12. **Held-out gate test.** Run the horizon-gate test on 2024-25 origins as the first given hypothesis?
13. **Direction "flat" threshold.** Is max(10%, 10 admissions) the lab's notion of flat?
