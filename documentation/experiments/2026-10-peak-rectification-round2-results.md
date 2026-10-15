# Peak rectification - round 2 results

Round 2 of the controlled XGBoost peak-rectification sweep, run 2026-10-15 after the 2026-10-15 meeting with A. Adiga. It changes three things from [round 1](2026-10-peak-rectification-results.md): phases are A. Adiga's windows (`adiga_2026_10`), there is an oversampling arm, and every configuration runs with 3 seeds. Every number below comes from `outputs/experiments/peak_rectification_round2/summary.md`. The interactive version is the [2026-10-15 weekly report](../reports/2026-10-15/index.html).

## Headline

Pooled over horizons 1-4, no arm beats the baseline by more than seed noise: baseline peak WIS is 107.28 with a seed-to-seed sd of 1.24. By horizon there is a clear trade-off that holds in every seed. Weighting or oversampling peak-phase rows lowers peak WIS at 1 and 2 weeks ahead by 4-13 and raises it at 4 weeks ahead by 9-22. The pooled score hides it because the two effects cancel.

## Setup

- Model `xgboost_direct`, seeds 42, 43, 44 (`xgboost.random_state`).
- Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoffs at stride 4, `US` excluded from scoring.
- Phases by target date, `adiga_2026_10`: growth Nov 1 - Dec 31, peak Jan 1 - end of Feb, decline Mar 1 - Apr 15, plateau Apr 16 - May 31, off-season otherwise. October as off-season is an assumption awaiting confirmation.
- Data cache 2026-10-15 (sha256 `c91d9d81b8ab...` in the manifest).
- 11 configurations: baseline; approaching-peak weight 1.5/2/3/5 (K = 6 weeks before the season max); peak-phase weight 2/3; peak-phase oversampling 2/3/5 (copies of training rows whose target date is in the peak); 52-week training window. The 12- and 26-week windows were dropped after losing by about 70 peak WIS in round 1.
- Deltas are paired: config minus baseline at the same seed.

Round-1 and round-2 numbers are not comparable: round 1 used the old month-based phases (peak = Dec-Jan), one seed, and the 2026-09-30 data cache.

## Pooled peak WIS

| config | peak WIS (mean of 3) | seed sd | paired delta, mean (min..max) | overall WIS |
|---|---:|---:|---|---:|
| baseline | 107.28 | 1.24 | - | 50.33 |
| approaching-peak weight 1.5 | 108.85 | 1.10 | +1.56 (+1.35..+1.70) | 50.60 |
| approaching-peak weight 2 | 109.39 | 1.12 | +2.11 (+1.43..+2.84) | 50.56 |
| approaching-peak weight 3 | 108.70 | 1.69 | +1.41 (-1.66..+2.97) | 50.16 |
| approaching-peak weight 5 | 111.28 | 0.57 | +3.99 (+3.17..+4.58) | 50.56 |
| peak weight 2 | 105.43 | 2.03 | -1.86 (-2.96..-0.47) | 49.59 |
| peak weight 3 | 106.96 | 0.83 | -0.33 (-0.87..-0.01) | 50.15 |
| peak oversample 2 | 106.47 | 0.90 | -0.81 (-2.37..+0.22) | 50.12 |
| peak oversample 3 | 106.62 | 2.01 | -0.67 (-2.37..+1.55) | 50.26 |
| peak oversample 5 | 107.20 | 0.36 | -0.08 (-1.75..+1.33) | 50.19 |
| 52-week window | 175.79 | 4.11 | +68.50 (+66.11..+72.98) | 71.75 |

## Peak WIS by horizon

Mean of 3 seeds; paired delta vs baseline in parentheses.

| config | 1 wk | 2 wk | 3 wk | 4 wk |
|---|---:|---:|---:|---:|
| baseline | 67.9 | 101.1 | 129.7 | 150.1 |
| peak weight 2 | 61.6 (-6.3) | 92.8 (-8.2) | 129.8 (+0.1) | 159.3 (+9.2) |
| peak weight 3 | 60.7 (-7.2) | 89.1 (-12.0) | 130.9 (+1.2) | 170.2 (+20.1) |
| peak oversample 2 | 63.7 (-4.3) | 95.1 (-6.0) | 128.6 (-1.1) | 159.9 (+9.8) |
| peak oversample 3 | 61.8 (-6.2) | 90.5 (-10.6) | 128.2 (-1.6) | 168.5 (+18.4) |
| peak oversample 5 | 59.4 (-8.5) | 88.0 (-13.1) | 133.4 (+3.7) | 171.9 (+21.8) |
| approaching-peak weight 5 | 57.7 (-10.3) | 93.3 (-7.7) | 147.5 (+17.8) | 173.4 (+23.3) |
| 52-week window | 92.5 (+24.6) | 156.3 (+55.2) | 231.2 (+101.4) | 264.8 (+114.7) |

For every peak-weight and peak-oversample arm, the 1-, 2- and 4-week paired deltas have the same sign in all 3 seeds. The 3-week deltas cross zero. Per-seed ranges and the other phases are in the summary and the report's explorer.

## Interpretation

- **The pooled score hides a horizon trade-off.** Emphasising the peak helps the short-horizon models and hurts the 4-week model; averaged, the effects cancel. Round 1 already showed the same pattern once its forecasts were re-scored under the new windows.
- **Oversampling behaves like weighting.** In XGBoost, k copies of a row act like weight k unless rows are subsampled; the two arms land within seed noise of each other.
- **Approaching-peak weighting is slightly worse pooled** at every value, though it gives the largest 1-week gain at weight 5.
- **The 52-week window is worse at every horizon**, consistent with round 1.
- **Overall WIS** stays within about 1 of baseline for every arm except the window.

## Caveats

- One evaluation season (2025-26). Each peak cell rests on 2-3 forecast dates.
- The "emphasis on short horizons only" idea below was spotted on this season; it is a hypothesis until tested on a season it was not chosen on.
- Three seeds bound the seed noise but do not estimate it well.

## Proposed next arm

`xgboost_direct` trains one model per horizon, so apply peak weighting or oversampling to the 1- and 2-week models only and keep the baseline for 3 and 4 weeks. Choose the factor on 2024-25 cutoffs and score on 2025-26.

## Reproduce

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
for s in 42 43 44; do
  .venv/bin/python scripts/experiments/peak_rectification.py --seeds $s --windows 52 --out outputs/experiments/pr2-seed$s &
done; wait
.venv/bin/python scripts/experiments/peak_rectification.py --merge outputs/experiments/pr2-seed42 outputs/experiments/pr2-seed43 outputs/experiments/pr2-seed44 --out outputs/experiments/peak_rectification_round2
```

About 50 minutes on a MacBook with the three seeds in parallel.
