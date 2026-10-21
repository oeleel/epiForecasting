# Gated peak emphasis (2026-10-21)

> **Every number on this page is in-sample.** The gates were designed after looking at
> this same 2025-26 season (the week table and the horizon table in `summary.md`), and they
> are scored on that season. They show what a gate *could* have done this season, not what
> it will do next season. Nothing here is evidence that a gate generalizes until the
> out-of-sample test at the end is run.

Source: the weekly hindcast in this directory (35 forecast dates 2025-10-04..2026-05-30,
seed 42, data cache sha256 `c91d9d81b8ab`). No model was trained. Script:
`scripts/experiments/gated_emphasis.py`; full generated tables: [`gated/tables.md`](gated/tables.md)
and the CSVs in `gated/`.

## Question

Emphasis arms (peak-phase rows oversampled or upweighted) cut WIS for forecasts made
during the surge (forecast dates Nov 29 - Dec 20) and add a lot just after the turn (Jan 3,
Jan 10), and they help h1-h2 while hurting h3-h4. Can a gate that uses only information
available at the forecast date keep the gains and drop the losses?

## Method

Each composed forecast set takes every row (location x forecast date x horizon) from either
an emphasis run or the baseline run, both seed 42 on the same data:

| gate | takes the emphasis row when |
|---|---|
| ungated | always (the emphasis arm itself, for reference) |
| horizon | horizon is 1 or 2 |
| calendar | the forecast date (`cutoff_date`) is in growth, Nov 1 - Dec 31 (`adiga_2026_10`) |
| curve | the location is rising at the forecast date: value(cutoff) > value(cutoff - K wk) x (1 + r); default K = 2, r = 0.10 |
| horizon+curve | horizon is 1 or 2 and the location is rising |

Emphasis configs: `oversample_2`, `oversample_5`, `lambda_calendar_3`. The curve gate reads
only the observed values dated at the forecast date and K weeks before it. It reads the
pinned 2026-10-15 data vintage, not the vintage live on each forecast date; the models were
fitted on that same vintage, so the gate is as real-time as the hindcast and ignores
reporting backfill exactly as the hindcast does.

Scoring is the harness's: `PhaseEvaluator.merge_forecasts_actuals` against the pinned cache,
US excluded (52 locations), overall = every target week, phase = target week's phase. The
script re-scores the four source runs first and refuses to continue unless they match
`log.jsonl`: baseline reproduces **overall 53.01, peak 119.73**, and `oversample_2` (52.77,
119.61), `oversample_5` (54.00, 124.70) and `lambda_calendar_3` (53.72, 123.63) match too.
The composition reads the run's full-precision forecasts in
`outputs/experiments/peak_rectification_weekly/forecasts/`, not the 1-decimal export.

## Is the horizon gate the same as training emphasis into h1-h2 only?

**Yes, exactly up to the floor constraint, whose effect is below 1e-5 WIS.**

Trained independently per horizon:

- One `FluForecastingModel` per (horizon, quantile), each its own `XGBRegressor` with the
  same params and `random_state` (`src/direct_forecast.py:765-773`, `src/model.py:141`).
- Each horizon's training frame is built separately by `prepare_horizon_data`
  (`src/direct_forecast.py:799-835`, a per-location `shift(-h)`), and so is its positional
  train/validation split and its early-stopping set (`src/direct_forecast.py:876-898`).
- `sample_weights.by_phase` (lambda_calendar) is computed per horizon from that horizon's
  metadata, labelling each row by its own target date (`src/direct_forecast.py:888-891`,
  `:250-283`).
- `data.oversample` copies are chosen per horizon inside the per-horizon loop
  (`src/pipeline.py:246-259`).
- Shared across horizons but identical between arms: data loading and the 10 feature
  passes (`src/pipeline.py:393-412`), which see no weights or copies. Label encoders are fit
  per model on its own rows (`src/model.py:69-70`). The pipeline builds
  `QuantileDirectForecastEnsemble` or `PhaseOversampledQuantileEnsemble`, never the
  clustered ensemble (`src/pipeline.py:418-438`), so location clustering plays no part.
  Features are every engineered column except date, location and value (`src/model.py:59`);
  nothing is selected by fitting.

The one cross-horizon coupling is at forecast time: horizon h is floored at
0.3 x (1 - 0.05 (h-1)) times horizon h-1's emitted quantiles
(`src/direct_forecast.py:998`, `:1028-1033`, `:1055`). In a single run with emphasis only in
h1-h2, h3 would be floored against the emphasis h2. `floor_counterfactual` rebuilds that run's
h3-h4 values (bracketing the rows where a baseline value sat on its own floor, so its raw
value is unknown). 27-60 scored rows change, all tiny lower quantiles in small states, and
pooled WIS moves by at most 2e-6 (peak: 0). The calendar and curve gates switch whole
(location, forecast date) forecasts, so they are exact selections between two runs, with no
coupling at all.

Not verified experimentally: that XGBoost is bit-for-bit deterministic across the two
runs. Same seed and same data make that the expected case; I did not retrain to confirm it.

## Results (in-sample, seed 42)

WIS change vs baseline; negative is better. Baseline: overall 53.01, growth 59.17, peak
119.73, decline 30.60, plateau 6.49; h1 32.53, h2 47.59, h3 61.53, h4 70.40. "Locations"
counts the 52 states whose mean WIS (all target weeks, or peak target weeks) went down / up.

| emphasis | gate | overall | peak | growth | decline | h1 | h2 | h3 | h4 | locations, overall | locations, peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oversample_2` | ungated | -0.24 | -0.12 | -1.98 | +1.97 | -1.27 | -0.35 | -0.83 | +1.50 | 21 / 31 | 22 / 30 |
| `oversample_2` | horizon | -0.41 | -1.08 | -0.45 | +0.04 | -1.27 | -0.35 | 0 | 0 | 35 / 17 | 37 / 15 |
| `oversample_2` | calendar | -1.22 | -2.66 | -2.06 | 0 | -0.49 | -0.67 | -2.41 | -1.28 | 38 / 14 | 37 / 15 |
| `oversample_2` | curve | -0.93 | -2.10 | -1.87 | +0.57 | -1.19 | -0.50 | -1.92 | -0.09 | 28 / 24 | 32 / 20 |
| `oversample_2` | horizon+curve | -0.42 | -1.17 | -0.45 | -0.03 | -1.19 | -0.50 | 0 | 0 | 42 / 10 | 45 / 7 |
| `oversample_5` | ungated | +0.99 | +4.97 | -4.24 | +4.95 | -2.59 | -2.11 | +0.81 | +7.85 | 14 / 38 | 17 / 35 |
| `oversample_5` | horizon | -1.18 | -3.13 | -1.66 | +0.53 | -2.59 | -2.11 | 0 | 0 | 40 / 12 | 43 / 9 |
| `oversample_5` | calendar | **-2.00** | **-3.43** | -4.37 | 0 | -1.28 | -2.45 | -3.18 | -1.11 | 41 / 11 | 40 / 12 |
| `oversample_5` | curve | -0.55 | +0.77 | -4.02 | +1.62 | -2.10 | -2.40 | -1.13 | +3.42 | 24 / 28 | 24 / 28 |
| `oversample_5` | horizon+curve | -1.13 | -2.78 | -1.64 | +0.06 | -2.10 | -2.40 | 0 | 0 | **47 / 5** | 46 / 6 |
| `lambda_calendar_3` | ungated | +0.71 | +3.89 | -3.51 | +3.92 | -2.12 | -1.54 | +0.08 | +6.42 | 19 / 33 | 21 / 31 |
| `lambda_calendar_3` | horizon | -0.92 | -2.24 | -1.34 | +0.21 | -2.12 | -1.54 | 0 | 0 | 40 / 12 | 41 / 11 |
| `lambda_calendar_3` | calendar | -1.70 | -3.01 | -3.60 | 0 | -0.93 | -2.16 | -3.21 | -0.50 | 41 / 11 | 41 / 11 |
| `lambda_calendar_3` | curve | -0.55 | +0.04 | -3.38 | +1.78 | -1.78 | -2.08 | -1.38 | +3.03 | 25 / 27 | 22 / 30 |
| `lambda_calendar_3` | horizon+curve | -0.96 | -2.46 | -1.33 | +0.07 | -1.78 | -2.08 | 0 | 0 | 47 / 5 | **48 / 4** |

Plateau changes are within +-0.3 everywhere (see `gated/tables.md`).

Around the turn, `oversample_5` (the other two emphasis configs show the same shape):

| forecast date | baseline WIS | locations rising (K=2, r=0.10) | ungated | horizon | calendar | curve | horizon+curve |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-11-29 | 124.9 | 91% | -10.6 | +0.4 | -10.6 | -10.0 | +0.3 |
| 2025-12-06 | 192.5 | 96% | -34.6 | -4.2 | -34.6 | -34.5 | -4.3 |
| 2025-12-13 | 189.7 | 100% | -25.8 | -8.7 | -25.8 | -25.8 | -8.7 |
| 2025-12-20 | 128.7 | 98% | -9.2 | -12.9 | -9.2 | -9.3 | -12.9 |
| 2025-12-27 | 135.9 | 100% | +9.0 | -8.7 | +9.0 | +9.0 | -8.7 |
| 2026-01-03 | 175.7 | 100% | +38.8 | -6.2 | 0 | +38.8 | -6.2 |
| 2026-01-10 | 141.1 | 13% | +56.9 | +6.4 | 0 | -0.3 | -0.7 |
| 2026-01-17 | 83.5 | 0% | +8.4 | +2.9 | 0 | 0 | 0 |
| 2026-01-24 | 100.4 | 2% | -12.9 | -4.2 | 0 | 0 | 0 |

What the tables say:

- **Every gate turns an emphasis arm that loses overall into one that wins overall.** The
  ungated `oversample_5` loses (+0.99 overall, +4.97 peak); gated, it wins by 0.55 to 2.00
  overall.
- **Calendar has the largest pooled gain** because the Dec 31 boundary happens to fall
  between this season's last good emphasis week and its first bad one. It drops emphasis on
  Jan 3 and Jan 10 by date, not by anything the data shows. That is the gate most fitted to
  this season: a season that turns in early December would keep emphasis through its turn,
  and one that turns in February would lose the surge gains.
- **Curve alone does not fix the turn.** At the Jan 3 forecast every location is still above
  its value two weeks earlier, so the gate keeps emphasis on the worst week (+38.8). A
  K-week comparison lags the turn by about K weeks. It does remove the Jan 10 loss.
- **The horizon gate is the robust one.** No thresholds, h1-h2 gains in every config, and
  40-43 of 52 states improve. It keeps half of the surge gain (-4 to -13 per surge week
  instead of -9 to -35) and shrinks the post-turn loss (+6.4 on Jan 10 instead of +56.9).
- **Horizon+curve is the most consistent across states** (46-48 of 52 better with
  `oversample_5` / `lambda_calendar_3`) at about the same pooled gain as horizon alone. It
  removes the h1-h2 loss on Jan 10 but adds two tuned constants.

The gains are small: the best defensible gate (horizon, `oversample_5`) is -1.18 overall
(2.2%) and -3.13 peak (2.6%), on one seed. Round 2's seed-to-seed sd of peak WIS was about
1.2 per arm; the sd of a paired gated-minus-baseline delta is not measured.

## Sensitivity of the curve gate

K in {1, 2, 3} weeks, r in {0, 0.10, 0.25}; range over the 9 settings (WIS change vs
baseline). The default (K=2, r=0.10) was set before the grid ran; the grid is reported
instead of its best cell.

| emphasis | gate | overall, min..max | peak, min..max |
|---|---|---:|---:|
| `oversample_2` | curve | -0.97..-0.25 | -2.41..+0.32 |
| `oversample_2` | horizon+curve | -0.43..-0.20 | -1.21..-0.60 |
| `oversample_5` | curve | -0.96..+0.94 | -1.22..+6.21 |
| `oversample_5` | horizon+curve | -1.21..-0.82 | -3.21..-1.98 |
| `lambda_calendar_3` | curve | -0.89..+0.71 | -1.52..+4.74 |
| `lambda_calendar_3` | horizon+curve | -1.05..-0.68 | -2.85..-1.46 |

Curve alone is fragile: K = 3 flips it to a loss for `oversample_5` and `lambda_calendar_3`
(peak +4.4 to +6.2), because a longer look-back keeps emphasis on further past the turn.
K = 1 is its best setting. Horizon+curve stays a gain on every setting, and its spread is
about +-0.2 overall, so its result does not hinge on the constants. Every cell is in
`gated/tables.md`.

## Out-of-sample test (proposed, not run)

The test needs new training runs, so whether and when to run it is a decision for Leo and
A. Adiga.

1. **Pre-register before training:** gates = horizon (primary), horizon+curve at K=2,
   r=0.10 (secondary), calendar (reported only, as the overfit reference); emphasis =
   `oversample_5` and `lambda_calendar_3`; metrics = overall and peak WIS, and states
   improved. No threshold is re-tuned after seeing the held-out season.
2. **Hold out 2024-25:** a second pinned split with weekly forecast dates 2024-10-05..
   2025-05-31, each fitted only on data before its forecast date (as now), with baseline and
   the two emphasis configs, seeds 42 43 44. The data starts 2022-02-05, so these fits see
   about 2.7 seasons instead of 3.7; that is a real difference from the 2025-26 runs and
   should be stated. At roughly 15 min per config per seed for 35 dates (this run: 760-1050
   s), that is about 2.5 h single-threaded, or under an hour with one process per seed.
3. **Pass criterion:** the primary gate beats baseline AND the ungated emphasis arm on both
   overall and peak WIS in the mean over seeds, with every per-seed paired delta negative,
   and improves a majority of the 52 states. If 2023-24 can also be held out (about 1.7
   seasons of training), it is a second, weaker check.

## Recommendation

- Do not adopt any gate on this evidence; it is in-sample and one seed.
- If one gate goes to the out-of-sample test, make it the **horizon gate** with
  `oversample_5` or `lambda_calendar_3` for h1-h2. It needs no thresholds, its h1-h2 gain
  appears in every emphasis config, and it is exact relative to a single training run. In
  practice it is "run the emphasis models for h1-h2 and the baseline models for h3-h4",
  which the pipeline can do without new machinery.
- Carry horizon+curve as the secondary: more consistent across states, stable across its
  constants, but two more constants to justify.
- Do not carry the calendar gate forward as a method, despite the best pooled number; its
  advantage comes from this season's turn falling next to Dec 31.
- Drop curve alone: it lags the turn and changes sign with K.

## Files

- `scripts/experiments/gated_emphasis.py`: composition, gates, floor counterfactual,
  scoring, tables. Rerun: `.venv/bin/python scripts/experiments/gated_emphasis.py` (about 5 s).
- `gated/tables.md` (generated): every table, all three configs.
- `gated/gate_summary.csv`, `gated/gate_by_week.csv`, `gated/curve_sensitivity.csv`,
  `gated/floor_coupling.csv`.
- `tests/agent/test_gated_emphasis.py`: gating, composition and floor-counterfactual tests.
