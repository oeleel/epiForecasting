# Peak rectification - results

This is the findings doc for the controlled XGBoost peak-rectification sweep designed in [`2026-10-peak-rectification-design.md`](2026-10-peak-rectification-design.md): 10 configs, 9 cutoffs each, finished 2026-09-30. It is written for the 2026-10-01 meeting with A. Adiga and stands alone; every number below is a row in `outputs/experiments/peak_rectification/log.jsonl` (mirrored in `summary.md` next to it), except the 09-02 demo numbers, which are labelled as such.

## Headline

Neither cheap rectification action lowers peak WIS. Weighting approaching-peak rows leaves peak WIS at 123.54-123.87 against a 121.01 baseline while moving peak bias from -58.8 to -6.4 (lambda 3) and cutting peak 95% coverage from 0.855 to 0.808; shorter training windows make every phase worse (peak WIS 238.26 / 240.24 / 197.99 for 12 / 26 / 52 weeks). Lambda fixes the level, not the score. Applied unconditionally, a short fit window hurts this model class at every phase; the rule's takeoff trigger was not tested.

## Results table

Target: peak WIS (lower is better). Guard: overall WIS. Delta is vs the harness `baseline` row. Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoffs at stride 4, `US` excluded from scoring, `xgboost_direct`, seed 42. Phases are calendar phases by target date (peak = Dec-Jan).

| arm | config | peak WIS | delta vs baseline | overall WIS | peak bias | peak coverage_95 | peak MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `baseline` | 121.01 | +0.00 | 50.05 | -58.8 | 0.855 | 171.6 |
| lambda | `lambda_1.5` | 123.74 | +2.73 | 50.92 | -35.7 | 0.835 | 177.0 |
| lambda | `lambda_2` | 123.54 | +2.53 | 50.64 | -17.9 | 0.821 | 180.5 |
| lambda | `lambda_3` | 123.71 | +2.70 | 50.40 | -6.4 | 0.808 | 177.7 |
| lambda | `lambda_5` | 123.87 | +2.86 | 50.53 | +7.9 | 0.769 | 180.4 |
| lambda_calendar | `lambda_calendar_2` | 123.70 | +2.69 | 50.60 | -29.8 | 0.827 | 175.2 |
| lambda_calendar | `lambda_calendar_3` | 122.66 | +1.65 | 50.21 | -6.2 | 0.806 | 174.2 |
| window | `window_12` | 238.26 | +117.25 | 96.02 | -193.0 | 0.746 | 326.9 |
| window | `window_26` | 240.24 | +119.23 | 92.16 | -188.2 | 0.709 | 325.9 |
| window | `window_52` | 197.99 | +76.98 | 71.85 | +67.5 | 0.829 | 329.5 |

Config deltas: `lambda_*` = `sample_weights.approaching_peak = {weeks_before: 6, weight: lambda}`; `lambda_calendar_*` = `sample_weights.by_phase.peak = lambda`; `window_*` = `data.train_window_weeks = N`. No config is `better` on the target metric.

Guard detail (the other phases and the overall bias), same rows:

| config | onset WIS | decline WIS | overall bias | overall coverage_95 | horizon-4 WIS (all phases) |
|---|---:|---:|---:|---:|---:|
| `baseline` | 11.07 | 48.61 | -26.4 | 0.831 | 80.42 |
| `lambda_1.5` | 11.04 | 49.15 | -21.6 | 0.825 | 83.22 |
| `lambda_2` | 11.05 | 48.50 | -16.3 | 0.828 | 83.90 |
| `lambda_3` | 10.95 | 47.69 | -13.2 | 0.829 | 82.25 |
| `lambda_5` | 10.94 | 47.89 | -9.0 | 0.822 | 83.54 |
| `lambda_calendar_2` | 11.07 | 48.32 | -17.7 | 0.826 | 83.21 |
| `lambda_calendar_3` | 11.01 | 48.01 | -11.2 | 0.821 | 83.76 |
| `window_12` | 24.15 | 87.63 | -45.5 | 0.739 | 141.33 |
| `window_26` | 23.93 | 72.95 | -56.0 | 0.697 | 137.50 |
| `window_52` | 14.42 | 53.84 | -2.2 | 0.847 | 116.04 |

## Reproducibility

**Five identical baselines.** The sweep ran as five background processes (`outputs/experiments/pr-split-1` .. `pr-split-5`, gitignored), each with its own `baseline` row, same config and seed. The five baseline `metrics` dicts are equal in every field (overall, by phase, by horizon), not just in peak WIS. The merged `log.jsonl` keeps split 1's baseline row; every arm row is byte-for-byte the row from its split.

| split | baseline `run_at` (UTC) | seconds | peak WIS |
|---|---|---:|---:|
| 1 | 2026-09-30T20:10:48 | 478.7 | 121.01 |
| 2 | 2026-09-30T20:11:39 | 524.8 | 121.01 |
| 3 | 2026-09-30T20:11:49 | 524.2 | 121.01 |
| 4 | 2026-09-30T20:08:45 | 340.6 | 121.01 |
| 5 | 2026-09-30T20:11:47 | 522.9 | 121.01 |

So the harness is deterministic at this seed, and the deltas above are not noise between processes. What the sweep cannot say is how large a delta is real across seeds (see caveats).

**Harness path vs the 09-02 demo path.** The 09-02 demo (`outputs/model_selection/demo_2026-09-02.json`, model-bank path `src/model_bank/legacy.py::XGBoostDirectModel`, same split and cutoffs) scored `xgboost_direct` at peak WIS 122.37, peak bias -52.4, onset WIS 10.98, decline WIS 49.28 (demo numbers, not from this sweep). The harness trains through `src/pipeline.run_pipeline`, the config-driven path the improve loop uses, and its baseline is 121.01 / -58.8 / 11.07 / 48.61. The gap is the path difference between the two code routes, same ensemble class and features; it has not been traced further. Every delta in this doc is against the harness baseline, never against 122.37.

**Provenance** (from `manifest.json`): git `13be50b0de` (dirty), data cache `data/raw/flusight_hospital_admissions.csv` dated 2026-09-30T18:44:41Z, Python 3.12.5, xgboost 3.4.1, pandas 2.3.3, `XGBOOST_PARAMS_V2`, features `v3`, log target, floor 0.3 decaying 0.05 per horizon, K = 6, minimum window 8 weeks.

**Runtime** (from the `seconds` field, 9 cutoffs per config, five processes in parallel with `OMP_NUM_THREADS=1`): full-history configs took 265.3-524.8 s each, about 30-60 s per cutoff; the window configs took 77.8-192.1 s because they fit on far fewer rows. First manifest 20:02:49Z, last row 20:24:12Z: about 21 minutes of wall time for the whole sweep.

**Artifacts.** `log.jsonl` (one (state, action, reward) row per config), `forecasts/<config_id>.csv` (pooled forecasts across cutoffs, so any arm can be re-scored, for example by curve phase), `manifest.json`, `summary.md`. Only `summary.md` and `manifest.json` are exempt from `.gitignore` (commit them by hand); the log and forecasts are gitignored and live on the machine that ran the sweep.

## Interpretation by arm

**Approaching-peak lambda (K = 6).** The weight does exactly what the loss proposed by A. Adiga says it should do to the level: peak bias moves monotonically with lambda, -58.8 -> -35.7 -> -17.9 -> -6.4 -> +7.9 for lambda 1.5 / 2 / 3 / 5, and overall bias shrinks from -26.4 to -9.0. But peak WIS does not follow: 123.74 / 123.54 / 123.71 / 123.87, all about +2.5 to +2.9 worse than 121.01, with no trend in lambda. Two numbers explain why. Peak MAE rises (171.6 -> 177.0 / 180.5 / 177.7 / 180.4): bias is a signed mean over locations, so pulling the level up cancels under-prediction in some locations while creating over-prediction in others, and the absolute error per row grows. And peak 95% coverage falls (0.855 -> 0.835 / 0.821 / 0.808 / 0.769): the same row weight multiplies into all five quantile models, so the whole fan shifts up with the median instead of widening around it, and the lower quantiles rise past observations they used to cover. WIS is dominated by interval sharpness and calibration plus the per-location median error, and both got worse. The guard holds: onset WIS 10.94-11.07 vs 11.07, decline 47.69-49.15 vs 48.61, overall WIS 50.40-50.92 vs 50.05. The cost shows up most at horizon 4 (80.42 -> 82.25-83.90), where the shifted fan has the least data to calibrate on.

**Calendar-peak lambda (Dec-Jan origin rows).** Lambda 2 gives peak WIS 123.70, bias -29.8, coverage 0.827; lambda 3 gives 122.66, bias -6.2, coverage 0.806. The question this arm asked was whether the label definition matters. At matched bias it does not: `lambda_3` and `lambda_calendar_3` land at bias -6.4 vs -6.2, coverage 0.808 vs 0.806, and peak WIS 123.71 vs 122.66. The calendar label reaches the same level shift with the same lambda and the same coverage loss, so for this model the two labels are interchangeable ways of upweighting the same rows. The 1.05 WIS gap between them is the best number in the sweep, but it is one seed and one split, so it is not evidence that the calendar label is better.

**Training window (12 / 26 / 52 weeks).** This arm is a clear negative, and the interesting part is why. Peak WIS is 238.26 / 240.24 / 197.99 (bias -193.0 / -188.2 / +67.5, coverage 0.746 / 0.709 / 0.829), and every phase degrades: onset 24.15 / 23.93 / 14.42 vs 11.07, decline 87.63 / 72.95 / 53.84 vs 48.61, overall WIS 96.02 / 92.16 / 71.85 vs 50.05. The knob keeps the lag and seasonal features intact and only shrinks the rows the model fits on (design doc, section 6), so each horizon-h model sees about 12 - h rows per location for the 12-week window (8 for the horizon-4 model; computed from the window semantics, not logged). `XGBOOST_PARAMS_V2` was tuned for the full pooled history (min_child_weight 30, 800 trees with early stopping on a location hold-out), and on a handful of rows per location the pooled, feature-rich model starves; the 12- and 26-week runs also finished in 77.8 and 87.9 s against 340.6 s for the baseline in the same process (split 4), consistent with a much smaller training set. Seconds depend on contention (the same baseline config took 340.6-524.8 s across the five processes), so this is a hint, not a measurement. The 52-week window flips the sign of the peak bias to +67.5, so the failure mode changes with N rather than fading, and that has not been examined. Adiga's short-window rule comes from per-series statistical models, where "the last 12 weeks" is the last 12 points of one series and no feature history is lost. The 09-24 point from Adiga was that training strategy is per model class. This sweep is one concrete observation of that. Applied unconditionally, a short fit window hurts `xgboost_direct` at every phase on this split and seed; the rule's takeoff trigger was not tested (design doc, section 4). That is evidence the rule needs model-class scoping, not proof it is wrong as written.

## What this means for the knowledge bank

**Rules must carry model-class context.** `knowledge/curated/training-strategy.yaml::train-short-window-on-takeoff` has `context: {phase: [surge, onset, approaching_peak]}` and no `model` key, so it is retrieved for every family, including `xgboost_direct`, for which an unconditional 12-, 26- or 52-week window hurt every phase in this sweep. The fix is not to delete the rule but to scope it: `context.model` on the per-class entries the lab promised (`knowledge/curated/_training-strategy-by-model-class.yaml` is the stub), and a negative experiential entry for `xgboost_direct` so the loop sees the warning next to the rule. This is the first case where the bank would send the loop down a known-bad path once `set_training_window` exists, and it is the reason the design keeps guardrails advisory with an override log rather than hard blocks.

**The nine outcomes are in the bank as experiential entries.** On 2026-10-01 `python -m agent knowledge import-experiment` turned the nine non-baseline `log.jsonl` rows into experiential entries (ids such as `exp-lambda-3-2025`). Import is deterministic code, not hand authoring. Each entry has confidence `low` and one observation, because a config is one run on one seed and one split. They render after the curated entries, for example in `knowledge query --phase peak --model xgboost_direct`. One real line:

```
- [E, low, 1 run] approaching_peak weight 3.0 (weeks_before 6): peak WIS 123.71 vs 121.01 baseline (+2.70, worse); peak bias -58.8 -> -6.4; peak coverage_95 0.855 -> 0.808
```

Merging repeated observations of the same action into one entry with higher confidence needs the end-of-run bank-writer (design doc unit 5), which is not built. The window outcome sits next to the curated window rule, which now recommends the `set_training_window` action, so the loop sees the warning beside the rule.

## Caveats

1. Scoring uses calendar phases (Dec-Jan = peak, by target date), not curve phases.
2. Single pinned split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoffs at stride 4.
3. The approaching-peak label uses K = 6, an assumption, not a tuned value.
4. At the first cutoff (2025-10-04) only 2022-23 and 2023-24 carry approaching-peak weights; 2024-25 is not yet complete in the horizon-shifted frame.
5. The XGBoost validation split holds out the last ~20% of locations, so weights on those rows are unused.
6. The window arm shrinks the rows the model fits on (lags stay intact), so small windows leave few rows per location.
7. Single seed (42). The five baselines prove the harness is deterministic, not that a 1-3 WIS difference between arms would survive a seed change.
8. K = 6 is untuned; the label width was not varied, so "lambda does not help" is "lambda at K = 6 does not help".
9. Lambda was applied only as XGBoost `sample_weight`, the same row weight in every one of the 20 horizon-by-quantile models. Nothing in the arm acts on interval width, which is where the coverage loss shows up.

## Arms not run

Actions 1 (add failing-phase examples from other seasons, locations or outbreaks), 2 (oversample failing-phase windows) and 3 (SMOTE) were not run, for the reasons in the design doc, section 4: all locations are already pooled and the only extra seasons are pre-2022-02-05 (open question 2); oversampling is equivalent to integer sample weights for XGBoost except under `subsample` and needs literal row duplication to be a distinct arm; SMOTE is expected to fail. The lambda result lowers the priority of 2 and 3 further: both are ways of upweighting the same rows, and upweighting fixed the level without fixing the score.

Rectification entry #4 and the two training-strategy window rules stay `not_yet_available` until a `set_training_window` adapter action exists; the harness set `data.train_window_weeks` directly. See the design doc, section 4.

## Proposed next arms

In priority order, each cheap on the existing harness:

1. **Seed replication of the baseline and `lambda_3`** (three seeds). Needed before any further arm: it sizes the noise floor that decides whether a 1-3 WIS difference means anything.
2. **Bias-only post-hoc correction as a control.** Shift every quantile by a constant estimated from training residuals in the peak phase, no retraining. If it reproduces lambda's profile (bias toward 0, coverage down, WIS flat), lambda is doing nothing a level shift cannot, and the search should move to interval width.
3. **Lambda with interval width.** Weight the quantile models separately: lambda on the upper quantiles (0.75, 0.95) only, or a smaller lambda on the 0.05 / 0.25 models than on the median, so the fan widens as the median rises instead of shifting with it. Coverage is the metric to watch, WIS the reward.
4. **Re-score by curve phase.** The pooled forecasts are kept; re-scoring every arm with the Adiga surge / plateau / decline labels is a script, not a rerun (open question 5).
5. **Oversampling and SMOTE (actions 2 and 3)**, low priority given the lambda result.
6. **Window 104 and a window arm on a statistical family** (`sf_autoets` or `persistence`) to test the model-class claim directly rather than infer it.

## Open questions

Same five as the design doc, section 9.

1. **Approaching-peak definition and K.** Keep "K weeks before the season max" with K = 6, or switch to the surge segment from `phase_segmentation`? If K, what value?
2. **2020-22 seasons.** Are the COVID-era seasons admissible as extra peak examples (action 1), or does training stay pinned at 2022-02-05?
3. **Lambda range.** Current sweep is 1.5 / 2 / 3 / 5, capped at 5 by the adapter guardrail. Wider, finer, or learned per season or per location?
4. **Warm-start vs retrain.** Raised 08-27, not in the five actions. Add it as a bank category now and an experiment arm later?
5. **Calendar vs curve phases for scoring.** Every number here scores by calendar phase (Dec-Jan = peak, by target date). Keep both vocabularies, or switch scoring to curve phases? The pooled forecasts are kept, so any arm can be re-scored either way.
