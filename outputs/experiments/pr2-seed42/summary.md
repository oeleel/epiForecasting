# peak_rectification: results

Generated 2026-10-15T17:09:42+00:00 from `log.jsonl` (11 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row (of the same seed, when several seeds ran).

Phase scheme: `adiga_2026_10`.

Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoff(s) at stride 4: 2025-10-04, 2025-11-01, 2025-11-29, 2025-12-27, 2026-01-24, 2026-02-21, 2026-03-21, 2026-04-18, 2026-05-16. Excluded from scoring: US. Seeds: 42. Git 7bee4a9573, data cache 2026-10-15T15:23:56+00:00 (sha256 c91d9d81b8ab).

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline__seed42` | 108.53 | 50.59 | -38.2 | +0.00 | +0.0% | no | 195.7 |
| lambda | `lambda_1.5__seed42` | 109.88 | 50.79 | -13.5 | +1.35 | -1.2% | no | 205.9 |
| lambda | `lambda_2__seed42` | 109.96 | 50.67 | 2.1 | +1.43 | -1.3% | no | 203.4 |
| lambda | `lambda_3__seed42` | 106.87 | 49.71 | 11.3 | -1.66 | +1.5% | yes | 290.0 |
| lambda | `lambda_5__seed42` | 111.70 | 50.82 | 20.7 | +3.17 | -2.9% | no | 428.8 |
| lambda_calendar | `lambda_calendar_2__seed42` | 106.39 | 49.90 | -10.9 | -2.14 | +2.0% | yes | 459.4 |
| lambda_calendar | `lambda_calendar_3__seed42` | 107.66 | 50.28 | 2.0 | -0.87 | +0.8% | yes | 234.5 |
| window | `window_52__seed42` | 174.95 | 71.33 | 77.7 | +66.42 | -61.2% | no | 96.3 |
| oversample | `oversample_2__seed42` | 106.16 | 49.77 | -26.5 | -2.37 | +2.2% | yes | 220.3 |
| oversample | `oversample_3__seed42` | 106.16 | 50.08 | -12.0 | -2.37 | +2.2% | yes | 231.7 |
| oversample | `oversample_5__seed42` | 106.78 | 49.89 | 5.5 | -1.75 | +1.6% | yes | 240.0 |

## Peak WIS by horizon

Peak-phase WIS on each horizon's forecasts alone (`by_phase_horizon`), mean over seeds; delta vs `baseline`, and with several seeds the min..max of the same-seed paired deltas. `n/a` when the phase is absent from that horizon's forecasts or the row was logged before phase x horizon was recorded.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 68.4 (+0.0) | 99.4 (+0.0) | 132.1 (+0.0) | 154.3 (+0.0) |
| `lambda_1.5` | 67.2 (-1.2) | 98.5 (-0.9) | 135.3 (+3.2) | 159.9 (+5.6) |
| `lambda_2` | 62.4 (-6.0) | 96.0 (-3.4) | 140.5 (+8.4) | 164.7 (+10.4) |
| `lambda_3` | 59.8 (-8.6) | 91.8 (-7.6) | 139.5 (+7.3) | 160.0 (+5.7) |
| `lambda_5` | 58.0 (-10.4) | 93.3 (-6.1) | 144.3 (+12.2) | 178.0 (+23.7) |
| `lambda_calendar_2` | 62.8 (-5.6) | 93.4 (-6.0) | 131.0 (-1.2) | 160.1 (+5.8) |
| `lambda_calendar_3` | 60.9 (-7.5) | 89.7 (-9.8) | 129.8 (-2.4) | 173.7 (+19.4) |
| `window_52` | 94.4 (+26.0) | 154.4 (+55.0) | 232.0 (+99.9) | 259.2 (+104.9) |
| `oversample_2` | 63.7 (-4.7) | 96.3 (-3.1) | 127.0 (-5.1) | 159.0 (+4.7) |
| `oversample_3` | 61.1 (-7.3) | 90.1 (-9.3) | 125.9 (-6.2) | 170.1 (+15.8) |
| `oversample_5` | 60.1 (-8.2) | 85.7 (-13.7) | 131.5 (-0.6) | 173.1 (+18.8) |

## Overall WIS by horizon

All scored phases (`by_horizon`), mean over seeds; delta vs `baseline`. The pooled WIS above is the mean of these over horizons, so it can hide an arm that helps h1/h2 and hurts h3/h4.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 29.0 (+0.0) | 36.8 (+0.0) | 54.9 (+0.0) | 81.6 (+0.0) |
| `lambda_1.5` | 28.7 (-0.3) | 36.4 (-0.4) | 55.0 (+0.1) | 83.1 (+1.5) |
| `lambda_2` | 26.8 (-2.2) | 35.9 (-0.9) | 56.1 (+1.2) | 83.8 (+2.2) |
| `lambda_3` | 26.0 (-3.0) | 35.0 (-1.8) | 55.9 (+0.9) | 81.9 (+0.3) |
| `lambda_5` | 25.4 (-3.6) | 35.8 (-1.0) | 57.0 (+2.0) | 85.2 (+3.6) |
| `lambda_calendar_2` | 27.2 (-1.8) | 35.7 (-1.1) | 54.5 (-0.4) | 82.2 (+0.5) |
| `lambda_calendar_3` | 26.6 (-2.4) | 35.1 (-1.7) | 54.6 (-0.4) | 84.8 (+3.2) |
| `window_52` | 38.5 (+9.5) | 50.9 (+14.1) | 80.8 (+25.9) | 115.1 (+33.5) |
| `oversample_2` | 27.5 (-1.5) | 36.5 (-0.3) | 53.5 (-1.5) | 81.6 (+0.0) |
| `oversample_3` | 26.7 (-2.3) | 35.4 (-1.4) | 53.9 (-1.0) | 84.3 (+2.7) |
| `oversample_5` | 26.2 (-2.8) | 34.6 (-2.2) | 55.0 (+0.1) | 83.6 (+2.0) |

## Configs

- `baseline__seed42` (baseline): stock config; delta {"xgboost.random_state": 42}; run 2026-10-15T15:53:41+00:00
- `lambda_1.5__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 42}; run 2026-10-15T15:57:07+00:00
- `lambda_2__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 42}; run 2026-10-15T16:00:31+00:00
- `lambda_3__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 42}; run 2026-10-15T16:06:28+00:00
- `lambda_5__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 42}; run 2026-10-15T16:29:32+00:00
- `lambda_calendar_2__seed42` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 42}; run 2026-10-15T16:52:39+00:00
- `lambda_calendar_3__seed42` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 42}; run 2026-10-15T16:56:33+00:00
- `window_52__seed42` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 42}; run 2026-10-15T16:58:10+00:00
- `oversample_2__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-15T17:01:50+00:00
- `oversample_3__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-15T17:05:42+00:00
- `oversample_5__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-15T17:09:42+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name covers different dates in different schemes, so never compare across schemes.
