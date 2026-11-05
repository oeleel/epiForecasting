# Pre-registration: horizon-gated peak emphasis on the held-out 2024-25 season

Written and committed on 2026-11-05, before any 2024-25 forecast existed. This file fixes the hypothesis, the
data, the test and the decision rule. Any analysis not listed here is exploratory and is labelled as such
in the results.

## Why this hypothesis

On the 2025-26 weekly hindcast, emphasis on peak-phase training rows helped the 1-2 week forecasts and hurt
the 3-4 week forecasts (round 2: same sign in all 3 seeds). Taking emphasis rows for h1-h2 and baseline rows
for h3-h4 turned `oversample_5` from a pooled loss into a gain (-3.13 peak WIS,
`documentation/experiments/weekly_hindcast/gated_emphasis.md`). That gate was designed after looking at
2025-26, so the 2025-26 result is in-sample. 2024-25 is a season no experiment has looked at.

## Hypothesis

**H1.** On the 2024-25 season, the horizon-gated `oversample_5` forecast has lower peak-phase WIS than the
baseline.

- Gated forecast: `oversample_5` rows (peak-phase training rows copied 5 times) for horizons 1-2, baseline
  rows for horizons 3-4. Composition by `scripts/experiments/gated_emphasis.py::compose` with
  `horizon_mask`, no training.
- H0: no difference in expected peak-phase WIS.

## Data and runs

- Runs: `scripts/experiments/peak_rectification.py --eval-window holdout_2024_25 --stride-weeks 1
  --arms baseline oversample --oversample-factors 5`, seeds 42, 43 and 44 (one process each, merged).
- 35 weekly cutoffs, 2024-10-05 to 2025-05-31. Each cutoff trains only on data before it.
- Data cache pinned to round 2's snapshot: sha256 `c91d9d81b8ab47ee5fe5458123167a3ce9c4171bddca682a89eb573d4b5336cb`.
- Model, features and every other setting as in round 2 (`xgboost_direct`, `XGBOOST_PARAMS_V2`, features v3).
- Output: `outputs/experiments/peak_rectification_holdout_2024_25/`.

## Primary endpoint and test

- Rows: target date in the `adiga_2026_10` peak phase (2025-01-01 to 2025-02-28), all horizons, `US`
  excluded.
- Unit of analysis: the forecast date. For each date t, `d_t` = sum over that date's peak rows of
  (WIS gated - WIS baseline), divided by the mean number of peak rows per date, then averaged over the 3
  seeds. The mean of `d_t` therefore equals the pooled peak WIS delta. Rows from one fit are not
  independent, so rows are never the unit. (Amended, see below.)
- Test: two-sided Diebold-Mariano test on `d_t`, with the Harvey-Leybourne-Newbold small-sample correction
  and a t distribution with n - 1 degrees of freedom. Variance: autocovariances up to lag 3, because
  forecasts from dates up to 3 weeks apart share target weeks. If that estimate is not positive, the
  Bartlett (Newey-West) kernel with the same lag is used, and the results say so.
- Decision rule: H1 is supported if and only if the mean of `d_t` is negative and p < 0.05.
- Expected power is low: about 11 forecast dates have peak-phase targets. A non-significant result is
  reported as "not confirmed", not as evidence of no effect.

## Secondary analyses (reported, no decision)

1. Circular block bootstrap 95% interval for the mean of `d_t`: block length 4, 10,000 resamples, generator
   seed 0.
2. Harm guard: the same test on all target weeks (overall WIS).
3. Per-horizon peak WIS deltas, descriptive.
4. Ungated `oversample_5` against the baseline, same test.
5. The same test applied to the 2025-26 weekly hindcast (seed 42), labelled in-sample.

## Known caveats

- `XGBOOST_PARAMS_V2` and the v3 feature set were tuned before this project's experiments, and the tuning
  scripts default to a 2024-11-02 cutoff (`src/config.py` `DEFAULT_CUTOFF_DATE`), so the base model may have
  been fitted to 2024-25. Both arms share those settings; the hypothesis itself was not chosen on 2024-25.
- The 2024-25 models train on less history (from 2022-02-05), so absolute WIS is not comparable with
  2025-26. Only paired deltas within one season are compared.
- The floor constraint couples horizons at forecast time; on 2025-26 this moved the composed WIS by at most
  2e-6 (`gated_emphasis.md`).

## Amendment 1 (2026-11-05, before any 2024-25 result was read)

The first version defined `d_t` as the unweighted mean over a date's peak rows. Dates hold different numbers
of peak rows (the first peak date has only h4 rows in the peak, the last only h1 rows), so the mean of
unweighted `d_t` is a different estimand from pooled peak WIS. On the 2025-26 in-sample data it gave -2.39
for ungated `oversample_5`, while its pooled peak WIS delta is +4.97. The hypothesis is about pooled peak
WIS, so `d_t` is now scaled to make its mean equal the pooled delta. The amendment was found on 2025-26 data
only; the held-out runs were still training and no 2024-25 forecast had been scored.
