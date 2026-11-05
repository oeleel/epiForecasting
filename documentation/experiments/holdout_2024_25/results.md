# Held-out 2024-25 test of horizon-gated peak emphasis: results

Run 2026-11-05 under the pre-registration in [`preregistration.md`](preregistration.md) (committed `9eb9a76`
before the runs, amendment 1 made before any 2024-25 result was read). Runs:
`outputs/experiments/peak_rectification_holdout_2024_25/` (baseline and `oversample_5`, seeds 42, 43, 44,
35 weekly cutoffs 2024-10-05..2025-05-31, data sha256 `c91d9d81b8ab...`, git `9eb9a76`, clean tree). Test:
`scripts/experiments/significance_test.py`. The JSON beside this file holds every number below.

## Verdict

**H1 is not confirmed.** On 2024-25, horizon-gated `oversample_5` changes pooled peak WIS by -0.19
(359.06 -> 358.87). The two-sided Diebold-Mariano test gives p = 0.99 over 11 forecast dates.

The reason is not noise alone. The 2025-26 pattern "emphasis helps h1 and h2" holds only for h1 in 2024-25:

| `oversample_5` peak WIS delta vs baseline (ungated; the gate keeps h1-h2, takes h3-h4 from baseline) | h1 | h2 | h3 | h4 |
|---|---:|---:|---:|---:|
| 2025-26 (in-sample, seed 42) | -7.66 | -4.85 | +5.50 | +26.90 |
| 2024-25 seed 42 | -29.0 | +23.9 | +22.5 | +19.2 |
| 2024-25 seed 43 | -23.1 | +25.8 | +18.0 | +19.6 |
| 2024-25 seed 44 | -24.6 | +24.7 | +20.7 | +19.2 |

The h1 gain and the h2 loss have the same sign in all 3 seeds. The horizon gate keeps h1 and h2, so the two
cancel.

## All pre-registered tests

| test | rows | WIS baseline -> candidate | mean delta | DM p | bootstrap 95% CI |
|---|---|---|---:|---:|---|
| **Primary:** gated, peak, 2024-25 | 11 dates | 359.06 -> 358.87 | -0.19 | 0.99 | [-17.7, +17.5] |
| Harm guard: gated, all weeks, 2024-25 | 35 dates | 117.69 -> 116.85 | -0.84 | 0.82 | [-6.5, +4.4] |
| Ungated, peak, 2024-25 | 11 dates | 359.06 -> 368.79 | +9.73 | 0.72 | [-19.2, +39.2] |
| Ungated, all weeks, 2024-25 | 35 dates | 117.69 -> 121.13 | +3.44 | 0.51 | [-3.8, +11.9] |
| Gated, peak, 2025-26 (in-sample, seed 42) | 12 dates | 119.73 -> 116.60 | -3.13 | 0.13 | [-6.1, -0.6] |

Every DM test used the rectangular kernel to lag 3, except the in-sample one, which fell back to Bartlett.

## What this means

- **No emphasis configuration is significant on either season.** The pooled point estimates the project
  acted on so far (round 2, gated emphasis) are inside the noise of a per-date test.
- **The h1 gain replicates; the h1-h2 gate does not.** "Emphasis on h1 only" is the natural next
  hypothesis, but it was chosen after seeing 2024-25, so it is post hoc. It needs a third season to be
  tested honestly. 2023-24 is the only one left, and its models would train on less than two years of
  history.
- **The bootstrap interval is too narrow with about 12 dates.** On 2025-26 it excludes zero while DM gives
  p = 0.13. DM is the pre-registered primary test; the bootstrap interval is descriptive.
- **2024-25 is a different season.** Baseline peak WIS is about 3 times the 2025-26 value, and the baseline
  under-predicts the peak by about 440 admissions per row (bias -436 to -443). Absolute WIS is not comparable
  across the two seasons; only the paired deltas are.

## Caveats

- Power is low: 11 peak dates. "Not confirmed" is not evidence of no effect.
- `XGBOOST_PARAMS_V2` and the v3 features may have been tuned on 2024-25 data (see the pre-registration).
  Both arms share them.
