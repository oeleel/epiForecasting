# TimeSeriesScientist vs our framework - first-level comparison (XGBoost, flu)

Run 2026-11-19 on Rivanna (`bii-gpu`, V100). Both frameworks used the same LLM (`qwen3:8b` served by
Ollama 0.19.0, 16k context), the same pinned data cache (sha256 `c91d9d81...`), the same training start
(2022-02-05), the same three cutoffs and the same truth. Scored rows: 5 states (CA, FL, NY, TX, VA) x 3
cutoffs (2025-11-29, 2025-12-27, 2026-01-24) x 4 weekly horizons = 60. Raw tables:
`documentation/experiments/tsci_comparison/`.

**Read the limits section before quoting any number.** This is one seed, 15 series, and the two frameworks
do not select their final model in the same way.

## Result

Metric: MAE of the point forecast, in weekly hospital admissions. TSci returns point forecasts only, so WIS
is not available for it. Our point forecast is the median.

| Framework | Baseline MAE | Final MAE | Change vs own baseline |
|---|---|---|---|
| Ours (`agent improve`, `xgboost_direct`) | 541.7 | 473.5 | -12.6% |
| TimeSeriesScientist (XGBoost only) | 644.4 | 683.3 | +6.0% |

| Cutoff | Ours baseline | Ours final | Ours change | TSci baseline | TSci final | TSci change |
|---|---|---|---|---|---|---|
| 2025-11-29 (growth) | 498.2 | 367.3 | -26.3% | 741.9 | 879.2 | +18.5% |
| 2025-12-27 (peak) | 490.2 | 490.2 | 0.0% | 828.7 | 673.6 | -18.7% |
| 2026-01-24 (peak) | 636.8 | 563.1 | -11.6% | 362.5 | 497.2 | +37.2% |

| Horizon | Ours baseline | Ours final | TSci baseline | TSci final |
|---|---|---|---|---|
| 1 week | 389.7 | 370.3 | 195.2 | 254.2 |
| 2 weeks | 481.7 | 475.5 | 447.8 | 478.9 |
| 3 weeks | 610.9 | 487.4 | 851.8 | 864.9 |
| 4 weeks | 684.5 | 560.8 | 1082.7 | 1135.4 |

"Baseline" is each framework's own starting point: for us, iteration 0 of the loop (the repo's default
config); for TSci, its own XGBoost function at library defaults on the raw series, with no LLM involved.
"Final" is the best iteration of our loop, and TSci's XGBoost after its Curator, Planner and Forecaster.

## Question A - does each framework improve the model over its own baseline?

- **Ours: yes on two of three cutoffs.** WIS, the metric the loop optimizes, fell 19.2% (124.4 to 100.6) at
  2025-11-29 and 13.9% (102.9 to 88.6) at 2026-01-24. At 2025-12-27 no proposed action helped; the loop
  stopped after two regressions and kept the baseline. On the five scored states MAE fell 12.6%; over all
  52 locations it fell 8.1% (172.8 to 158.8).
- **TSci: no, on average.** Its tuned XGBoost beat its own default XGBoost in only 6 of 15 runs, and was
  6.0% worse overall. It helped at one cutoff (2025-12-27) and hurt at the other two.
- Row by row, our final forecast had the smaller error on 40 of 60 rows.
- **TSci's simple XGBoost is the better one-week forecaster on these states** (195 against 390 at horizon
  1, before any tuning). Its 10-lag recursive model tracks the last observation closely. Our model wins from
  horizon 3 onward, where a recursive forecast compounds its own errors. This is a property of the two
  models, not of the agents, and it is worth a look on its own.

Why TSci's tuning does not help here, from its code and logs:

1. It picks hyperparameters by fitting on the first 80% of the input window and forecasting the last 20%
   recursively. For a 200-week window that is a 40-week-ahead forecast, used to choose settings for a
   4-week task.
2. It tunes once. Nothing in the system sees the test error and reacts, so a bad choice stands.
3. The 15 runs chose 15 distinct hyperparameter settings, and the LLM proposed a different search space
   each time, so there is no stable answer for the tuning to converge on at this series length.

## Question B - what diagnosis does our framework give that theirs does not?

| | Ours | TimeSeriesScientist |
|---|---|---|
| Where the error is | By epidemic phase, horizon and location, with signed bias. Example (cutoff 2025-11-29): "severe under-prediction during growth phase", WIS 17 at horizon 1 against 303 at horizon 4, worst states listed with their bias | One pooled MSE / MAE / MAPE per series |
| Domain reading of the curve | Phases from the knowledge bank; the season's peak is expected | Generic statistics. The epidemic peak is called a "structural break" and an "outlier" (Virginia report). In 5 of 15 analyses the LLM stated that no seasonality is present; none stated that it is. In 15 of 15 runs the LLM recommended clipping outliers; nothing was clipped only because the IQR detector found none |
| Why a change was made | Each action carries a rationale, an expected effect and the knowledge-bank entries it cites (for example `phase-growth-adiga-2026-10`, `train-short-window-on-takeoff`) | A free-text "reason" for the hyperparameter ranges |
| What the change did | Table of every iteration: action, WIS, change against the previous and the best iteration; a regression is reverted | Not applicable: one tuning pass, no before and after |
| What it recommends | A concrete config delta: at 2025-11-29, `sample_weights.by_phase[growth] = 5.0`; at 2026-01-24, `by_phase[peak] = 2.0` | Generic advice: "apply normalization", "add lag features", "include ARIMA / LSTM" |
| Uncertainty | Five quantiles, coverage reported | Its report notes its own intervals "match prediction values (no variability shown)" |

## Limits

1. **Selection is not symmetric, and this favours us.** Our loop keeps the best iteration as judged on the
   same four weeks that are scored here, so our "final" can never be worse than our baseline on WIS and is
   an optimistic estimate. TSci selects on a split inside the history and is scored on weeks it never saw.
   A fair version scores our chosen config on a later cutoff. Until that is run, "-12.6%" is an upper
   bound on our gain, and the comparison of the two change columns is not like for like.
2. **Small and single-seed.** 15 series, 3 cutoffs, one LLM sample per run. No significance test was run.
   The LLM is stochastic in both frameworks.
3. **Different models under one name.** Ours is one global model over all states with 59 features, one
   model per horizon and quantile. TSci fits 10 lags of one state's series and forecasts recursively.
4. **Different objectives.** Our loop optimized WIS and is scored here on MAE. TSci optimized MSE.
5. **TSci's released code was changed to run at all.** Two call sites in its graph pass the wrong
   arguments (upstream issue 4 reports the failure). The fix is `scripts/comparison/tsci_minimal_fixes.patch`.
   Its `requirements.txt` does not resolve either. TSci was also restricted to XGBoost; its default is to
   choose 3 of 21 models and ensemble them, which this run does not test.
6. **Its plotting step failed in 9 of 15 runs** (the LLM's plot plan was not valid JSON). The error is
   caught and the run continues; forecasts are not affected.
7. **A small local model.** The TSci paper used GPT-4o-class models. `qwen3:8b` was used for both so the
   LLM is not a confound, but TSci may do better with the model it was built for.

## Reproduce

`documentation/QUICKSTART.md`, section "TimeSeriesScientist comparison on Rivanna". Our runs:
`20261119-174325-2b0e`, `20261119-174824-ec95`, `20261119-175022-dd44`. Slurm jobs 23162203 (TSci, 31
minutes for 15 runs) and 23162245 (ours, 12 minutes for 3 cutoffs).

## Next

1. Remove limit 1: score each framework's chosen configuration on the next cutoff.
2. All 52 locations and all 9 pinned cutoffs, three seeds, with the paired significance test.
3. TSci with its full model list, to test the framework rather than one model.
