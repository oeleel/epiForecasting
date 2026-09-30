# Peak rectification experiment - design

This is the design of the controlled XGBoost experiment the advisor asked for on 2026-09-24: what is fixed, what is varied, how each configuration is scored and logged. It is written for sign-off before the numbers are read; the numbers live in the results doc.

Status (2026-09-30): harness built and smoke-tested; the full sweep finished on 2026-09-30 (see [Results](#results)).

## 1. Question

`xgboost_direct` fails most at the peak. On the pinned 2025-26 split (9 cutoffs, stride 4, US excluded; `outputs/model_selection/demo_2026-09-02.json`):

| calendar phase | WIS | bias |
|---|---:|---:|
| onset | 10.98 | -8.3 |
| decline | 49.28 | -26.7 |
| **peak** | **122.37** | **-52.4** |

The failure is under-prediction. Question: do the advisor's two cheap rectification actions (loss weight on approaching-peak rows, and a short training window) lower peak WIS without hurting overall WIS?

Source: advisor's five actions, Teams ~09-27, recorded in [`../meeting-notes/2026-09-24-controlled-experiment-and-curated-guardrails.md`](../meeting-notes/2026-09-24-controlled-experiment-and-curated-guardrails.md) and seeded as curated entries in [`../../knowledge/curated/rectification-actions.yaml`](../../knowledge/curated/rectification-actions.yaml).

## 2. Objective

| role | metric | where it comes from | decision use |
|---|---|---|---|
| primary | peak WIS (lower is better) | `metrics.by_phase.peak.wis` | the reward; `better` is computed on this only |
| guard | overall WIS | `metrics.overall.wis`, logged as `reward.guard` | an arm that wins the peak but loses overall is flagged, not called a win |
| reported | peak bias | `metrics.by_phase.peak.bias` | shows whether the arm fixes the under-prediction or only narrows intervals |

Reward direction and improvement fraction come from `agent.orchestrator._is_better` and `_improvement_pct`, the same functions the improve loop and `agent/run_report.py` use. Scoring is `FluForecastAdapter.compute_metrics`, the same scorer as `select-model` and the loop. The harness has no private scorer.

Every delta is against the baseline row **in the same output directory**, never against the 09-02 number. The 09-02 number came through the model-bank path (`src/model_bank/legacy.py::XGBoostDirectModel`); the harness trains through `src/pipeline.run_pipeline`. Same ensemble class and features, but the two paths can differ slightly, so the results doc reports both and explains any gap before reading an arm.

## 3. Fixed vs varied

Everything fixed is written to `manifest.json` in the output directory. Values below are read from `scripts/experiments/peak_rectification.py` (`FIXED`) and `src/config.py`.

| fixed | value | source |
|---|---|---|
| model family | `xgboost_direct` (4 horizon models, quantile ensemble) | `MODEL_FAMILY` |
| training start | 2022-02-05 (first date in the CDC cache) | `TRAIN_START_DATE` |
| eval window | 2025-10-01 .. 2026-05-31 | `EVAL_START_DATE`, `EVAL_END_DATE` |
| cutoffs | stride 4: 2025-10-04, 11-01, 11-29, 12-27, 2026-01-24, 02-21, 03-21, 04-18, 05-16 | `generate_eval_cutoffs(4)` |
| scoring exclusions | `US` (national aggregate) | `EXCLUDE_LOCATIONS` |
| XGBoost params | `XGBOOST_PARAMS_V2`: depth 3, lr 0.05, 800 trees, subsample 0.7, colsample 0.6, min_child_weight 30, alpha 1.0, lambda 5.0, gamma 1.0, early stopping 50 | `src/config.py` |
| seed | 42 | `XGBOOST_PARAMS_V2["random_state"]` |
| features | version `v3` | `FEATURE_VERSION` |
| quantiles | 0.05, 0.25, 0.5, 0.75, 0.95 | `QUANTILES` |
| target transform | `log` (log1p / expm1) | `TARGET_MODE` |
| horizon | 4 weeks | `FORECAST_HORIZON` |
| floor | enabled, 30% of last value, decays 0.05 per horizon | `get_default_config()["floor"]` |
| validation split | positional 80/20 over (location, date)-sorted rows | `src/direct_forecast.py`, `train(validation_split=0.2)` |
| phase labels for scoring | calendar, by forecast (target) date | `PhaseEvaluator.PHASE_MAP` |
| provenance | git SHA, dirty flag, data-cache mtime, host, python / xgboost / pandas versions | `build_manifest` |

| varied | one knob per config, all others stock |
|---|---|
| sample weights | `sample_weights.approaching_peak` or `sample_weights.by_phase.peak` |
| fit window | `data.train_window_weeks` |

Each log row records its `config_delta` (dotted config path -> value), so a row is self-describing without the script.

Known property of the fixed setup: the validation split is positional over rows sorted by (location, date), so it holds out the last ~20% of **locations** (by FIPS order), not the last 20% of weeks. Sample weights on those rows are never used, and early stopping runs on location-held-out data. This is identical in every arm, so it does not bias the comparison, but it dilutes every weight arm. A per-location temporal split is a separate change.

## 4. The five actions and which are run

| # | advisor action | run now? | how | why / why not |
|---|---|---|---|---|
| 5 | Reweight the loss: w = lambda for approaching-peak rows, 1 otherwise | **yes, two forms** | `lambda` and `lambda_calendar` arms | Cheap; the knob already existed for calendar phases |
| 4 | Shorter window when the current season differs from past ones | **yes, window only** | `window` arm | The "season differs" trigger is not defined yet; this tests the lever unconditionally |
| 2 | Oversample training windows from the failing phase | no | - | For XGBoost, integer sample weights equal row duplication except under `subsample` (0.7 here). Running it as a separate arm needs literal row duplication to stay distinct from action 5. Planned after lambda results say whether upweighting helps at all |
| 3 | SMOTE | no | - | Advisor expects it to fail. Lowest priority; run last if at all |
| 1 | Add failing-phase examples from other seasons / locations / outbreaks | no | - | All locations are already pooled. The only lever left is data before 2022-02-05 (COVID-era NHSN reporting), which is an admissibility question for the advisor, not an engineering one |

Entry #4 and the two window rules in `knowledge/curated/training-strategy.yaml` (`train-short-window-on-takeoff`, `train-long-window-in-lull`) carry `action: not_yet_available` until a `set_training_window` adapter action exists. Until then the harness sets `data.train_window_weeks` directly; the improve loop cannot take this action yet.

### Arms and configs (10 configs)

| arm | config ids | config delta | tests |
|---|---|---|---|
| `baseline` | `baseline` | none | the reference for every delta |
| `lambda` | `lambda_1.5`, `lambda_2`, `lambda_3`, `lambda_5` | `sample_weights.approaching_peak = {weeks_before: 6, weight: lambda}` | action 5 with the curve-anchored label |
| `lambda_calendar` | `lambda_calendar_2`, `lambda_calendar_3` | `sample_weights.by_phase.peak = lambda` (Dec-Jan origin dates) | action 5 with the calendar label: does the label definition matter? |
| `window` | `window_12`, `window_26`, `window_52` | `data.train_window_weeks = N` | action 4 |

Lambda values are `DEFAULT_LAMBDAS = (1.5, 2.0, 3.0, 5.0)`; the adapter's guardrail for `reweight_training_samples` weights is [1.0, 5.0], so 5 is the top of what the loop may propose. Windows are `DEFAULT_WINDOWS = (12, 26, 52)`: the advisor's takeoff window, a half year, and his lull window. 104 weeks (the other lull value he named) is not in this sweep; it is a one-flag addition (`--arms baseline window --windows 104`) if 52 looks promising.

### How the sweep was run

The sweep was run as five parallel background processes, one output directory each, each with its own baseline. With five processes sharing the machine, a weight config took about 30-60 s per cutoff (265-525 s for 9 cutoffs, `seconds` in the split logs); the window configs were faster (78-192 s). All ten configs finished in about 21 minutes of wall time (manifests created 20:02:49-20:03:04 UTC, last row logged 20:24:12 UTC).

| process | output dir | configs |
|---|---|---|
| 1 | `outputs/experiments/pr-split-1` | baseline, lambda 1.5, 2 |
| 2 | `outputs/experiments/pr-split-2` | baseline, lambda 3, 5 |
| 3 | `outputs/experiments/pr-split-3` | baseline, lambda_calendar 2, 3 |
| 4 | `outputs/experiments/pr-split-4` | baseline, window 12, 26 |
| 5 | `outputs/experiments/pr-split-5` | baseline, window 52 |

Each delta is computed against its own directory's baseline. The five baselines use the same config and seed, and they matched exactly: identical `metrics` in all five `log.jsonl` baseline rows (peak WIS 121.01, overall WIS 50.05) and byte-identical `forecasts/baseline.csv` files. So the run is deterministic and every split shares one reference. The merged log, `summary.md` and `manifest.json` are in `outputs/experiments/peak_rectification/`, which is also the harness default output directory for a single-process run.

## 5. The approaching-peak label

Implemented in `src/direct_forecast.py::_approaching_peak_mask`. It labels **training rows only**, where hindsight is legitimate.

- **Season**: Oct 1 .. Sep 30, per location (`SEASON_START_MONTH = 10`).
- **Eligible season**: complete relative to the data in hand (`season_end <= max date in the training rows`) **and** at least `MIN_SEASON_WEEKS = 40` observed weeks.
- **Peak**: `argmax(value)` within an eligible season, per location.
- **Label**: a row with origin date `d` is approaching-peak when `peak - K weeks <= d < peak`. The peak row and everything after it get 1.0. `K = DEFAULT_APPROACHING_PEAK_WEEKS = 6`, chosen so that the 4-week horizons whose targets land on the rise are covered.
- **Weight**: lambda on labeled rows, 1.0 on every other row. It multiplies with any other weight dimension.

The eligibility rule is the leakage rule:

| season | status | why |
|---|---|---|
| 2021-22 | never labeled | partial: the cache starts 2022-02-05, fewer than 40 weeks |
| 2022-23, 2023-24 | labeled at every cutoff | complete |
| 2024-25 | not labeled at the first cutoff (2025-10-04), labeled from 2025-11-01 on | the meta frame is horizon-shifted, so its last date is cutoff minus h weeks; at 2025-10-04 that is still before Sep 30, so the season is not yet "complete" |
| 2025-26 (eval season) | never labeled | in progress at every eval cutoff: its running max is not a peak, and labeling it would leak the future |

The first-cutoff case is the conservative side of the rule, not a bug. It means only two seasons carry weights at 2025-10-04.

### Alternative definition (not implemented)

The advisor's 09-03 wording was "when entering a growth phase", a curve property. `agent/phase_segmentation.py` already implements the Adiga surge / plateau / decline segmentation (bottom-up piecewise-linear fit on log1p, `DEFAULT_DELTA = 0.10`, `DEFAULT_MIN_SEGMENT_WEEKS = 3`). The alternative label is: rows inside the `surge` segment that ends at an eligible season's peak.

| | K weeks before max (implemented) | surge segment (alternative) |
|---|---|---|
| window length | fixed K for every season and location | follows the curve: long for a slow rise, short for a sharp one |
| free parameters | K | delta, minimum segment length |
| leakage handling | eligibility rule above | same eligibility rule; for anything touching the in-progress season, `segment_incrementally` (freezes breakpoints, refits only the tail) instead of `segment_series` |
| cost to switch | - | a new mask function behind the same config key |

The implemented rule is the simpler one to audit. Which one the lab wants is advisor question 1.

## 6. Window knob semantics

`data.train_window_weeks = N` (`src/pipeline.py`):

- Applied **after** feature engineering: training rows are the feature-engineered rows with `cutoff - N weeks < date <= cutoff`. Lag, rolling and seasonal features still see the full history from 2022-02-05. Only the rows the model fits on shrink.
- It is **not** `data.train_start_date`. That filter runs before feature engineering and would also cut the lag features' history, which is a different experiment.
- The horizon-h model drops the last h rows per location (no target yet), so each location contributes about N - h rows to the horizon-h model.
- Minimum `MIN_TRAIN_WINDOW_WEEKS = FORECAST_HORIZON + 4 = 8`; smaller values raise `ValueError`. The pipeline logs the resulting training-row count.
- Absent (default) = unchanged behavior.

## 7. Log schema

One JSON line per config in `log.jsonl`, written by `log_row`. It is a (state, action, reward) row with the same fields an experiential bank entry carries (design doc §3, worked example 3), so the summarizer and the future bank-writer read it unchanged.

Real line from the smoke run (`outputs/experiments/peak_rectification-smoke-1cutoffs/log.jsonl`, line 2), pretty-printed, values unchanged. This is a schema example, **not a result**: one October cutoff has only onset-phase targets, which is why `by_phase` has no `peak` key and the reward is null.

```json
{
  "run_at": "2026-09-30T18:56:44+00:00",
  "arm": "lambda",
  "config_id": "lambda_2",
  "config_delta": {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}},
  "describe": "approaching-peak rows (K=6 weeks before the season max) weighted 2x",
  "state": {
    "model": "xgboost_direct",
    "target_phase": "peak",
    "metric": "wis",
    "split": {"train_start_date": "2022-02-05", "eval_start_date": "2025-10-01", "eval_end_date": "2026-05-31", "stride_weeks": 4},
    "n_cutoffs": 1
  },
  "metrics": {
    "overall": {"mape": 93.1, "mae": 8.1, "rmse": 15.5, "bias": 0.1, "n_forecasts": 208, "wis": 5.7, "coverage_95": 0.933},
    "by_phase": {"onset": {"mape": 93.1, "mae": 8.1, "bias": 0.1, "n": 208, "wis": 5.7, "coverage_95": 0.933}},
    "by_horizon": {
      "1": {"mape": 58.8, "mae": 4.6, "bias": -0.1, "n": 52, "wis": 3.27, "coverage_95": 0.923},
      "2": {"mape": 82.8, "mae": 7.2, "bias": -0.7, "n": 52, "wis": 4.7, "coverage_95": 0.923},
      "3": {"mape": 94.5, "mae": 8.3, "bias": -0.5, "n": 52, "wis": 5.95, "coverage_95": 0.942},
      "4": {"mape": 136.7, "mae": 12.4, "bias": 1.6, "n": 52, "wis": 8.87, "coverage_95": 0.942}
    }
  },
  "reward": {
    "metric": "wis", "phase": "peak",
    "baseline_value": null, "value": null, "delta": null, "improvement_frac": null, "better": false,
    "guard": {"metric": "overall_wis", "baseline_value": 5.61, "value": 5.7}
  },
  "seconds": 63.6
}
```

| field | meaning |
|---|---|
| `state` | what was being optimized: model, target phase, metric, split, number of cutoffs |
| `arm`, `config_id`, `config_delta`, `describe` | the action: which knob, which value |
| `metrics` | full scorer output: `overall`, `by_phase`, `by_horizon` |
| `reward` | target-phase value vs baseline, `delta`, `improvement_frac`, `better`, plus the overall-WIS `guard` |
| `seconds` | wall time for all cutoffs of this config |

Other files per output directory: `manifest.json` (the fixed part), `forecasts/<config_id>.csv` (pooled forecasts across cutoffs, for re-scoring under another phase definition), `summary.md` (a table regenerated from `log.jsonl` after every config, never edited by hand). Only `summary.md` and `manifest.json` are committed; the rest is gitignored.

Invariants the harness enforces: one output directory pins one cutoff list (a mismatch fails loudly); baseline runs first; a logged `config_id` is skipped on rerun (resumable); a missing target phase yields a null reward, not a fabricated one.

Reproduce:

```bash
.venv/bin/python scripts/experiments/peak_rectification.py                     # full sweep, 10 configs
.venv/bin/python scripts/experiments/peak_rectification.py --arms baseline lambda --lambdas 2 3
.venv/bin/python scripts/experiments/peak_rectification.py --summary-only       # rebuild summary.md
```

Harness tests: `tests/agent/test_peak_rectification.py`; weight tests in `tests/agent/test_sample_weights.py`; window validation in `tests/agent/test_config.py`. Full suite: 281 tests (`.venv/bin/python -m pytest tests/agent`).

## 8. What "an agent summarizes the logs" consumes next

Not built yet. The plan, in order:

1. **Input**: the merged `log.jsonl` plus `manifest.json`. Nothing else, so the summarizer sees exactly what the experiment recorded.
2. **Deterministic pass first**: rank configs by `reward.delta`, flag guard violations (peak WIS down, overall WIS up), group by `arm`. No LLM touches these numbers.
3. **LLM pass**: the local model (Qwen 3 8B) writes "which strategy is best, where, and how confident" from the deterministic table, citing `config_id`s. The advisor expects a small model to summarize poorly; where it fails tells us how to organize the bank.
4. **Output**: candidate `experiential` entries in the bank schema (`payload.state / action / reward`, `evidence.n_observations`, `evidence.reward_delta`, `confidence: low` for a single split), for human review before promotion. The same path later becomes the end-of-run bank-writer (design doc §5, unit 5).

## Results

The sweep finished on 2026-09-30: all 10 configs over 9 cutoffs, merged into `outputs/experiments/peak_rectification/log.jsonl`. Results and their reading, with every number traced to a `log.jsonl` line, are in [`2026-10-peak-rectification-results.md`](2026-10-peak-rectification-results.md).

## 9. Questions for the advisor

1. **Approaching-peak definition and K.** Keep "K weeks before the season max" with K = 6, or switch to the surge segment from `phase_segmentation`? If K, what value?
2. **2020-22 seasons.** Are the COVID-era seasons admissible as extra peak examples (action 1), or does training stay pinned at 2022-02-05?
3. **Lambda range.** Current sweep is 1.5 / 2 / 3 / 5, capped at 5 by the adapter guardrail. Wider, finer, or learned per season or per location?
4. **Warm-start vs retrain.** Raised 08-27, not in the five actions. Add it as a bank category now and an experiment arm later?
5. **Calendar vs curve phases for scoring.** Every number here scores by calendar phase (Dec-Jan = peak, by target date). Keep both vocabularies, or switch scoring to curve phases? The pooled forecasts are kept, so any arm can be re-scored either way.
