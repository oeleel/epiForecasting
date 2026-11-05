# epiForecasting: project overview

As of 2026-11-05 (`main` after the 2026-11-05 merge). This document describes what the project is, how
to run it, what every command does, and one end-to-end example of each workflow. For setup detail and
troubleshooting, see the root `README.md` and `documentation/QUICKSTART.md`.

## 1. What the project is

An LLM agent framework that refines an existing flu-hospitalization forecaster. The base model is the
lab's XGBoost direct ensemble (`xgboost_direct`: 4 horizon models x 5 quantile models, weekly state-level
admissions, 1-4 weeks ahead). The agent reads structured, phase-aware metrics, proposes one constrained
change to the model's configuration, retrains, re-scores, and keeps or reverts the change.

There are three workflows:

| Workflow | Question it answers | Entry point |
|---|---|---|
| Model selection (Stage 1) | Which model family should we start from? | `python -m agent select-model` |
| Improvement loop (Stage 2) | Can the agent refine that model? | `python -m agent improve` |
| Hypothesis test (new, 2026-11-05) | Does a stated change really help, on a season nobody tuned on? | `scripts/experiments/peak_rectification.py` + `scripts/experiments/significance_test.py` |

The improvement loop, one iteration:

```
 evaluate forecasts --> Agent 1 (Analyst) --> Agent 2 (Engineer) --> validate action --> retrain --> re-evaluate
   (phase, horizon,       diagnoses the          picks ONE action        (schema and        (pipeline)     keep if better,
    location metrics)     weakest segment        from the catalog,       ranges; unknown                   else revert;
                                                 cites bank entries      action = error)                   stop rules
                          \______ both prompts carry KNOWN FACTS from the knowledge bank ______/
```

What exists and what does not:

| Part | Status |
|---|---|
| Phase-aware evaluation (WIS, 90% coverage, signed bias by phase, horizon, location) | Done |
| Two-agent loop with accept-or-revert, run tracker, run report | Done |
| Model bank (9 families, plug-in contract for in-house models) and Stage 1 selection | Done |
| Knowledge bank (curated, derived, experiential entries; cited in every proposal) | Done |
| Experiment harness for peak rectification (reweighting, windows, oversampling, seeds) | Done |
| Held-out season (2024-25) and paired significance test | Done (2026-11-05) |
| Significance or posterior gate inside the `improve` accept rule | Not built; waits on the test-vs-posterior decision |
| Fixed-hypothesis run mode for `improve` | Not built |
| In-house lab models in the bank | Blocked on model names and code access |

## 2. How to run it

```bash
# 1. Environment (Python 3.12, uv)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r documentation/requirements.txt pytest langchain-openai
uv pip install --python .venv/bin/python -r documentation/requirements-nixtla.txt   # optional example families
source .venv/bin/activate

# 2. macOS only: xgboost and torch ship separate OpenMP libraries; without these, one process segfaults
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# 3. Local LLM (only for commands that call the LLM)
brew install ollama && ollama serve &
ollama pull qwen3:8b                       # the code default

# Rivanna instead: point at the vLLM server
export LLM_BASE_URL=http://localhost:8247/v1
export LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ

# 4. Check the install (no LLM, no training; about 10 s)
python -m pytest tests/agent/ -q            # 547 tests
python -m agent --help
```

CDC data is fetched on first use and cached in `data/raw/` for one week. `python -m src.data_loader
update` refreshes it. Experiments pin the cache by sha256, so a refreshed cache cannot silently change a
past result.

## 3. Commands

### 3.1 Agent CLI (`python -m agent <command>`)

| Command | What it does | Needs LLM | Trains | Key flags |
|---|---|---|---|---|
| `check-data` | Checks the raw CDC data before training: missing weeks, nulls, spikes, zero reporting, coverage. The LLM explains the findings. | optional | no | `--cutoff-date`, `--dry-run` (checks only), `--exclude-locations` |
| `summarize` | Scores one forecast CSV (overall, by phase, horizon, location) and writes a plain-language report. | optional | no | `--forecast-csv`, `--dry-run` (metrics only), `--verbose` (show prompt) |
| `list-models` | Lists the model-bank families that load in this environment. | no | no | none |
| `select-model` | Stage 1. Runs every candidate family over rolling cutoffs on the pinned 2025-26 window and ranks them by a goal. | no | yes | `--families`, `--stride-weeks`, `--metric`, `--phase`, `--json`, `--report`, `--save-forecasts` |
| `improve` | Stage 2. The two-agent loop on one family at one cutoff. Writes `report.md` and `report.json` per run. | yes | yes | `--cutoff-date`, `--model-family`, `--max-iterations` (default 5), `--target-metric`, `--auto-apply`, `--no-knowledge`, `--fake-pipeline` |
| `history` | Lists recent `improve` runs from the tracker (`outputs/agent_runs/runs.db`). | no | no | `--limit` |
| `status <run_id>` | Shows each iteration of one run: action, metrics, accept or revert. | no | no | `--config` |
| `compare <a> <b>` | Compares two runs side by side. | no | no | `--metric` |
| `report <run_id>` | Regenerates the end-of-run report of a past run. | no | no | `--out`, `--rebuild` |
| `knowledge <action>` | Manages the knowledge bank (next table). | no | no | see below |

`--auto-apply` skips the human prompt. Without it, the loop asks y / n / stop before each action (shadow
mode). `--fake-pipeline` replaces retraining with a copy of the seed forecast; use it only to test the LLM
path, because every iteration then scores the same.

### 3.2 Knowledge bank (`python -m agent knowledge <action>`)

| Action | What it does |
|---|---|
| `validate [--dir D]` | Checks curated YAML. Exit 1 on any error. This is the lab's authoring check. |
| `rebuild` | Reloads `knowledge/curated/*.yaml` into `knowledge/knowledge.db` and prints counts. |
| `list [--provenance P] [--category C]` | Prints a table of stored entries. |
| `query [--phase] [--model] [--metric] [--season-week]` | Prints the KNOWN FACTS block that the agent would see in that context. |
| `import-experiment --log PATH` | Writes a harness log as experiential entries. Multi-seed logs become one entry per arm, with the per-seed deltas. |
| `remove --id ID [--dry-run]` | Deletes one derived or experiential entry. Curated entries are removed by editing the YAML and running `rebuild`. |

Entry types: `[C]` curated (from the lab, for example the five rectification actions from A. Adiga),
`[D]` derived, `[E]` experiential (from our own experiments, always `low` confidence). Every proposal
records which entries it cites, and citations of entries that were not retrieved are dropped.

### 3.3 Experiment and analysis scripts

| Script | What it does | Trains |
|---|---|---|
| `scripts/experiments/peak_rectification.py` | The experiment harness. Runs a baseline and rectification arms (`lambda`, `lambda_calendar`, `window`, `oversample`) per seed over rolling cutoffs, and logs (state, action, reward) rows. Resumable; the manifest pins cutoffs, seeds, phase scheme and data sha256. | yes |
| `... --eval-window holdout_2024_25 --out DIR` | The same harness on the held-out 2024-25 season (35 weekly cutoffs). | yes |
| `... --summary-only`, `--rescore`, `--merge` | Rebuild `summary.md`; re-score saved forecasts; merge per-seed directories. | no |
| `scripts/experiments/significance_test.py` | Paired per-date Diebold-Mariano test (Harvey-Leybourne-Newbold correction) of a candidate against the baseline, plus a block-bootstrap interval. `--gate horizon` composes emphasis rows for h1-h2 with baseline rows for h3-h4. | no |
| `scripts/experiments/gated_emphasis.py` | In-sample gate experiment: baseline and peak emphasis combined by horizon, calendar or curve. | no |
| `scripts/experiments/export_hindcast_tables.py` | Exports a run as tracked tables (per-row forecasts and WIS, by location, by week). | no |
| `scripts/plot_forecasts.py` | Inspection figure: one location, one forecast date, one panel per config. An interactive viewer is at `documentation/experiments/weekly_hindcast/viewer.html`. | no |
| `scripts/demo_stage1.py` | Short Stage 1 demo (`--full`, `--improve`). | yes |
| `scripts/demo_knowledge_bank.py` | Knowledge-bank demo (`--live` runs a 2-iteration `improve`). | optional |
| `scripts/pipeline.py --cutoff-date D` | The underlying forecasting pipeline alone: train and forecast. | yes |
| `scripts/optimize.py --n-trials N` | Optuna hyperparameter search for the base model. | yes |

## 4. Evaluation conventions

These rules apply to every number the project reports.

- **Phases.** The default scheme is `adiga_2026_10`: growth Nov 1 - Dec 31, peak Jan 1 - end of Feb,
  decline Mar 1 - Apr 15, plateau Apr 16 - May 31. June to October is off-season and is not scored.
  (October as off-season is an assumption that waits for confirmation.) A row's phase is the phase of
  its target date. Every metrics dict records its scheme, and reports refuse to compare across schemes.
- **WIS on 5 quantiles** (0.05, 0.25, 0.5, 0.75, 0.95), not FluSight's 23. Absolute WIS is therefore
  not comparable with hub scores; only differences between our own arms are.
- **Coverage.** The metric `coverage_95` counts actuals inside [q05, q95], which is the 90% interval. The
  name is legacy. The loop targets 0.90.
- **US is excluded** from scoring, because this model trains on state rows and its national forecast is
  invalid.
- **Splits.** Primary window 2025-10-01 to 2026-05-31: hypotheses are formed here. Held-out window
  2024-10-01 to 2025-05-31: hypotheses are confirmed here, under a pre-registration. Each cutoff trains
  only on data before it.
- **Significance.** The unit of analysis is the forecast date, because rows from one fit are correlated.
  A change counts as an improvement only if its mean per-date WIS difference is negative and the
  two-sided Diebold-Mariano p < 0.05. The `improve` loop does not apply this rule yet.

## 5. Example: a hypothesis test, end to end

This is the workflow from the 2026-11-05 meeting. It ran on 2026-11-05, and the numbers below are real.

**Step 1. Form the hypothesis on 2025-26.** Round 2 and the weekly hindcast showed that peak emphasis
(`oversample_5`: peak-phase training rows copied 5 times) helps at 1-2 weeks and hurts at 3-4 weeks. A gate
that uses emphasis for h1-h2 only gave -3.13 peak WIS in-sample.

**Step 2. Pre-register it** before any held-out forecast exists:
`documentation/experiments/holdout_2024_25/preregistration.md` fixes the hypothesis, the rows, the test and
the decision rule.

**Step 3. Run both arms on the held-out season** (one process per seed in parallel, then merge; both arms took about 23 minutes per
seed on a MacBook Pro):

```bash
for s in 42 43 44; do
  python scripts/experiments/peak_rectification.py --eval-window holdout_2024_25 --stride-weeks 1 \
      --arms baseline oversample --oversample-factors 5 --seeds $s \
      --pin-data-sha256 c91d9d81b8ab47ee5fe5458123167a3ce9c4171bddca682a89eb573d4b5336cb \
      --out outputs/experiments/holdout-seed$s &
done; wait
python scripts/experiments/peak_rectification.py --merge outputs/experiments/holdout-seed4{2,3,4} \
    --out outputs/experiments/peak_rectification_holdout_2024_25
```

**Step 4. Test it** (no training, a few seconds):

```bash
python scripts/experiments/significance_test.py \
    --run-dir outputs/experiments/peak_rectification_holdout_2024_25 --candidate oversample_5 --gate horizon
```

```
oversample_5 (gate horizon) vs baseline, peak WIS, holdout_2024_25, seeds [42, 43, 44]
  dates 11 (2024-12-07 .. 2025-02-15); WIS 359.06 -> 358.87
  mean delta per date -0.192; by horizon h1 -25.58, h2 +24.81, h3 +0.00, h4 +0.00
  Diebold-Mariano (HLN, lag 3, rectangular): stat -0.012, df 10, p 0.9903
  block bootstrap 95% CI (block 4): [-17.657, +17.503]
  verdict at alpha 0.05: not significant
```

**Step 5. Read the result.** The hypothesis is not confirmed. The 1-week gain replicates (it is larger on
2024-25, with the same sign in all 3 seeds), but the 2-week gain reverses with the same consistency, so
the gate cancels. Full write-up: `documentation/experiments/holdout_2024_25/results.md`.

**Step 6. Record it in the knowledge bank,** so that later agent runs see the evidence:

```bash
python -m agent knowledge import-experiment --log outputs/experiments/peak_rectification_round2/log.jsonl
python -m agent knowledge query --phase peak --model xgboost_direct --metric wis
```

## 6. Example: an `improve` run

A real run on `main` on 2026-11-05: run `20261105-191315-3fde`, Qwen 3 8B on Ollama, 6 min 41 s on a
MacBook Pro. It used the CDC data of that day, which is newer than the experiments' pinned snapshot.

```bash
python -m agent improve --cutoff-date 2025-12-06 --max-iterations 2 --auto-apply --exclude-locations US
```

**What happens, step by step** (excerpts from the console):

```
[iteration 0/2] baseline (regenerating with default config @ 2025-12-06)
  retraining... -> outputs/agent_runs/20261105-191315-3fde/iter_0.csv
  wis=192.510

[iteration 1/2]
  Agent 1 (Analyst):
    The model exhibits significant underprediction during the peak phase ...
    Focus: peak_underprediction
    Weak segments:
      [high  ] phase=peak (mape delta_vs_overall=+10.00)
      [high  ] horizon=3 (mape delta_vs_overall=+12.60)
      [high  ] location=72 (mape delta_vs_overall=+54.90)
    Hypotheses:
      (0.85) Peak phase underprediction stems from insufficient training data capturing high-variance spikes
  knowledge: 30 fact(s) retrieved for phase=peak model=xgboost_direct metric=wis (2 omitted)

  Agent 2 (Engineer):
    Action: reweight_training_samples(dimension=approaching_peak, value=8, weight=3.0)
    Rationale: Upweight samples in the 8 weeks before each peak to address under-prediction of peaks ...
  applied: sample_weights.approaching_peak = {weeks_before: 8, weight: 3.0}
  retraining... -> outputs/agent_runs/20261105-191315-3fde/iter_1.csv
  wis=170.130
  improved by +11.6% on wis

[iteration 2/2]
  ...
  ! warning: Agent 2 cited knowledge entry 'iter 1' which was not in the retrieved set [...]
  Agent 2 (Engineer):
    Action: reweight_training_samples(dimension=approaching_peak, value=8, weight=5.0)
  wis=163.590
  improved by +3.8% on wis

=== finished: stop_reason=max_iterations best_iter=2 ===
    report: outputs/agent_runs/20261105-191315-3fde/report.md
```

1. Iteration 0 retrains the default model at the cutoff and scores it. This is the baseline.
2. Agent 1 reads the phase, horizon and location metrics and names the weakest segments, with ranked
   hypotheses.
3. The orchestrator retrieves the knowledge-bank entries for this context (phase, model, metric) and
   shows them to Agent 2 with their ids.
4. Agent 2 picks one action from the catalog and cites the entries that support it. The adapter checks the
   action against its schema and ranges, then applies it to the config.
5. The model retrains, the new forecast is scored on the same window, and the change is kept if it beats
   the best so far, else reverted.
6. The run stops at `--max-iterations`, after two regressions in a row, or when Agent 2 chooses `stop`.
   `report.md` records every iteration, the phase and horizon breakdown, and a recommendation.

**From the report:**

| iter | action | status | WIS |
|---|---|---|---|
| 0 | (baseline) | baseline | 192.51 |
| 1 | `reweight_training_samples(approaching_peak, weeks_before=8, weight=3.0)` | new best | 170.13 |
| 2 | `reweight_training_samples(approaching_peak, weeks_before=8, weight=5.0)` | new best | 163.59 |

The report recommends iteration 2, a 15.0% WIS gain.

**How to read this run.** The run shows the mechanics and the guardrails, and it also shows why the
accept rule must change:

- **The guardrails work.** The action was validated before it was applied. In iteration 2, Agent 2 cited
  a nonexistent entry ("iter 1"), and the loop logged the citation and dropped it.
- **The gain is not evidence.** `improve` scores one cutoff: 4 target weeks (2025-12-13 to 2026-01-03),
  208 rows, one seed. That cutoff is in the surge, where peak emphasis is known to help (the 2025-12-06
  forecast date gains -34.6 WIS with peak oversampling, `documentation/reports/2026-10-29/`). Over 9 cutoffs
  and 3 seeds, round 2 found `approaching_peak` weight 5 (with `weeks_before` 6, not 8) worse on peak WIS
  in every seed (+3.99 mean). The loop accepted the change on a single point estimate.
- **The agent did not use the evidence it was shown.** The knowledge bank holds the round 2 results for
  weights 1.5 to 5 (all worse on the mean), and Agent 2 still proposed a close variant. Retrieval puts facts in the prompt; it does not make the model weigh
  them.

The significance gate (section 4) and a multi-cutoff evaluation inside `improve` are the fixes. Both wait
on the decision between a frequentist test and a posterior.

Inspect the run afterwards:

```bash
python -m agent history                        # the run is listed with its best WIS
python -m agent status 20261105-191315-3fde    # each iteration, action, accept or revert
python -m agent report 20261105-191315-3fde    # print the report again
```

## 7. Where things are

| Path | Contents |
|---|---|
| `agent/` | Orchestrator, evaluator, prompts, LLM client, run tracker, run report, model selection, significance test, knowledge bank code |
| `agent/adapters/flu_forecast.py` | The flu adapter: action catalog, validation, how an action changes the config |
| `src/` | The forecasting pipeline and the model bank (`src/model_bank/`) |
| `knowledge/curated/` | Human-written bank entries (YAML) |
| `scripts/experiments/` | Harness, significance test, gate experiment, exports |
| `outputs/agent_runs/<run_id>/` | Per-run forecasts and `report.md` / `report.json` |
| `outputs/experiments/` | Harness runs: `manifest.json`, `log.jsonl`, `forecasts/`, `summary.md` |
| `documentation/experiments/` | Tracked experiment write-ups, pre-registrations and results |
| `documentation/paper/outline.md` | The paper outline and the current state of every claim |
| `documentation/MODEL_BANK.md` | How to plug an in-house model into the bank |

## 8. Known limitations

- The `improve` loop accepts any lower point estimate. The significance test exists but is not wired into
  the accept rule.
- `improve` optimizes one pooled metric. It cannot target a phase, although every experiment targets peak
  WIS. Phase targeting exists only in `select-model` and the harness.
- Phase oversampling exists only in the harness, not as an agent action.
- With 11 peak forecast dates per season, a single season gives low power. "Not confirmed" is not evidence
  of no effect.
- The local model is Qwen 3 8B. The 72B model on Rivanna is planned, not yet used for the reported runs.
