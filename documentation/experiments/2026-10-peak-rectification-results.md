# Peak rectification - results

This is the results and findings doc for the controlled XGBoost peak experiment designed in [`2026-10-peak-rectification-design.md`](2026-10-peak-rectification-design.md). Status (2026-09-30): the sweep is still running, so this doc carries no findings yet - only the configs logged so far and their `run_at` stamps.

## Headline

Pending. No number goes here until the sweep finishes and every figure has a `log.jsonl` row behind it.

Reference point (not from this sweep): in the 09-02 demo run (`outputs/model_selection/demo_2026-09-02.json`), `xgboost_direct` scored peak WIS 122.37 with bias -52.4 (under-prediction), versus onset 10.98 and decline 49.28.

## Sweep status

Five background processes, one output directory each, each with its own baseline (split in the design doc, "How the sweep is being run").

| process | output dir | planned configs | logged so far (`run_at`, UTC) |
|---|---|---|---|
| 1 | `outputs/experiments/pr-split-1` | baseline, lambda 1.5, 2 | `baseline` 2026-09-30T20:10:48 |
| 2 | `outputs/experiments/pr-split-2` | baseline, lambda 3, 5 | `baseline` 2026-09-30T20:11:39 |
| 3 | `outputs/experiments/pr-split-3` | baseline, lambda_calendar 2, 3 | `baseline` 2026-09-30T20:11:49 |
| 4 | `outputs/experiments/pr-split-4` | baseline, window 12, 26 | `baseline` 2026-09-30T20:08:45, `window_12` 20:10:03, `window_26` 20:11:31 |
| 5 | `outputs/experiments/pr-split-5` | baseline, window 52 | `baseline` 2026-09-30T20:11:47 |

Before the merged table is written, the five baselines must be checked for equality. A mismatch is a finding (nondeterminism), not something to average away.

## Results table

Pending. Built from the five `summary.md` files once every config is logged.

## Interpretation by arm

Pending for all three run arms: approaching-peak lambda, calendar-peak lambda, training window.

## Caveats

1. Scoring uses calendar phases (Dec-Jan = peak, by target date), not curve phases.
2. Single pinned split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoffs at stride 4.
3. The approaching-peak label uses K = 6, an assumption, not a tuned value.
4. At the first cutoff (2025-10-04) only 2022-23 and 2023-24 carry approaching-peak weights; 2024-25 is not yet complete in the horizon-shifted frame.
5. The XGBoost validation split holds out the last ~20% of locations, so weights on those rows are unused.
6. The window arm shrinks the rows the model fits on (lags stay intact), so small windows leave few rows per location.

## Arms not run

Rectification entry #4 and the two training-strategy window rules stay `not_yet_available` until a `set_training_window` adapter action exists. See the design doc, section 4.

## Questions

See the design doc, section 9.
