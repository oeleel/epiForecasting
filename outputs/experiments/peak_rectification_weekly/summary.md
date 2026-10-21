# peak_rectification: results

Generated 2026-10-21T21:41:52+00:00 from `log.jsonl` (13 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row (of the same seed, when several seeds ran).

Phase scheme: `adiga_2026_10`.

Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 35 cutoff(s) at stride 1: 2025-10-04, 2025-10-11, 2025-10-18, 2025-10-25, 2025-11-01, 2025-11-08, 2025-11-15, 2025-11-22, 2025-11-29, 2025-12-06, 2025-12-13, 2025-12-20, 2025-12-27, 2026-01-03, 2026-01-10, 2026-01-17, 2026-01-24, 2026-01-31, 2026-02-07, 2026-02-14, 2026-02-21, 2026-02-28, 2026-03-07, 2026-03-14, 2026-03-21, 2026-03-28, 2026-04-04, 2026-04-11, 2026-04-18, 2026-04-25, 2026-05-02, 2026-05-09, 2026-05-16, 2026-05-23, 2026-05-30. Excluded from scoring: US. Seeds: 42. Git 2d9ade65e4, data cache 2026-10-15T17:36:37+00:00 (sha256 c91d9d81b8ab).

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline__seed42` | 119.73 | 53.01 | -24.5 | +0.00 | +0.0% | no | 759.3 |
| lambda | `lambda_1.5__seed42` | 126.25 | 54.37 | 2.2 | +6.52 | -5.5% | no | 804.5 |
| lambda | `lambda_2__seed42` | 131.75 | 55.62 | 22.7 | +12.02 | -10.0% | no | 1074.1 |
| lambda | `lambda_3__seed42` | 136.99 | 56.99 | 37.7 | +17.26 | -14.4% | no | 808.6 |
| lambda | `lambda_5__seed42` | 146.64 | 59.19 | 56.0 | +26.91 | -22.5% | no | 843.7 |
| lambda_calendar | `lambda_calendar_2__seed42` | 121.73 | 53.34 | 13.0 | +2.00 | -1.7% | no | 789.8 |
| lambda_calendar | `lambda_calendar_3__seed42` | 123.63 | 53.72 | 30.4 | +3.90 | -3.3% | no | 836.9 |
| window | `window_12__seed42` | 197.25 | 99.79 | -96.0 | +77.52 | -64.8% | no | 207.2 |
| window | `window_26__seed42` | 181.63 | 92.26 | -155.0 | +61.90 | -51.7% | no | 253.5 |
| window | `window_52__seed42` | 163.61 | 67.35 | -26.3 | +43.88 | -36.6% | no | 322.8 |
| oversample | `oversample_2__seed42` | 119.61 | 52.77 | 0.3 | -0.12 | +0.1% | yes | 860.4 |
| oversample | `oversample_3__seed42` | 122.47 | 53.58 | 20.9 | +2.74 | -2.3% | no | 934.4 |
| oversample | `oversample_5__seed42` | 124.70 | 54.00 | 35.9 | +4.97 | -4.2% | no | 1053.2 |

## Peak WIS by horizon

Peak-phase WIS on each horizon's forecasts alone (`by_phase_horizon`), mean over seeds; delta vs `baseline`, and with several seeds the min..max of the same-seed paired deltas. `n/a` when the phase is absent from that horizon's forecasts or the row was logged before phase x horizon was recorded.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 71.1 (+0.0) | 104.6 (+0.0) | 141.1 (+0.0) | 162.2 (+0.0) |
| `lambda_1.5` | 71.2 (+0.1) | 111.2 (+6.6) | 152.6 (+11.5) | 170.0 (+7.8) |
| `lambda_2` | 69.9 (-1.2) | 114.3 (+9.8) | 158.8 (+17.8) | 183.9 (+21.8) |
| `lambda_3` | 69.5 (-1.6) | 117.4 (+12.8) | 166.8 (+25.7) | 194.3 (+32.1) |
| `lambda_5` | 69.1 (-2.0) | 123.4 (+18.8) | 179.7 (+38.6) | 214.3 (+52.2) |
| `lambda_calendar_2` | 66.9 (-4.2) | 104.2 (-0.4) | 144.1 (+3.0) | 171.8 (+9.6) |
| `lambda_calendar_3` | 64.8 (-6.3) | 102.0 (-2.6) | 144.4 (+3.3) | 183.4 (+21.2) |
| `window_12` | 130.3 (+59.2) | 181.7 (+77.1) | 220.9 (+79.9) | 256.1 (+93.9) |
| `window_26` | 128.2 (+57.1) | 175.8 (+71.2) | 199.0 (+57.9) | 223.5 (+61.3) |
| `window_52` | 95.5 (+24.4) | 146.3 (+41.7) | 197.2 (+56.2) | 215.4 (+53.2) |
| `oversample_2` | 67.3 (-3.8) | 104.0 (-0.6) | 140.0 (-1.1) | 167.1 (+4.9) |
| `oversample_3` | 66.4 (-4.7) | 103.1 (-1.5) | 139.9 (-1.2) | 180.5 (+18.3) |
| `oversample_5` | 63.5 (-7.7) | 99.7 (-4.9) | 146.6 (+5.5) | 189.1 (+26.9) |

## Overall WIS by horizon

All scored phases (`by_horizon`), mean over seeds; delta vs `baseline`. The pooled WIS above is the mean of these over horizons, so it can hide an arm that helps h1/h2 and hurts h3/h4.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 32.5 (+0.0) | 47.6 (+0.0) | 61.5 (+0.0) | 70.4 (+0.0) |
| `lambda_1.5` | 32.4 (-0.1) | 48.8 (+1.2) | 64.0 (+2.5) | 72.2 (+1.8) |
| `lambda_2` | 31.5 (-1.1) | 49.5 (+2.0) | 65.7 (+4.2) | 75.7 (+5.3) |
| `lambda_3` | 31.2 (-1.3) | 50.1 (+2.5) | 68.2 (+6.7) | 78.4 (+8.0) |
| `lambda_5` | 31.0 (-1.6) | 51.8 (+4.2) | 71.0 (+9.5) | 82.9 (+12.5) |
| `lambda_calendar_2` | 31.3 (-1.3) | 47.1 (-0.5) | 61.6 (+0.1) | 73.3 (+2.9) |
| `lambda_calendar_3` | 30.4 (-2.1) | 46.0 (-1.5) | 61.6 (+0.1) | 76.8 (+6.4) |
| `window_12` | 66.8 (+34.3) | 90.0 (+42.4) | 110.6 (+49.1) | 131.7 (+61.3) |
| `window_26` | 64.4 (+31.9) | 85.5 (+37.9) | 100.5 (+39.0) | 118.6 (+48.2) |
| `window_52` | 41.4 (+8.9) | 61.6 (+14.0) | 79.1 (+17.5) | 87.4 (+17.0) |
| `oversample_2` | 31.3 (-1.3) | 47.2 (-0.4) | 60.7 (-0.8) | 71.9 (+1.5) |
| `oversample_3` | 31.0 (-1.5) | 46.6 (-1.0) | 60.7 (-0.9) | 76.1 (+5.7) |
| `oversample_5` | 29.9 (-2.6) | 45.5 (-2.1) | 62.4 (+0.8) | 78.2 (+7.8) |

## Configs

- `baseline__seed42` (baseline): stock config; delta {"xgboost.random_state": 42}; run 2026-10-21T18:52:44+00:00
- `lambda_1.5__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 42}; run 2026-10-21T19:06:09+00:00
- `lambda_2__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 42}; run 2026-10-21T19:46:40+00:00
- `lambda_3__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 42}; run 2026-10-21T20:00:09+00:00
- `lambda_5__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 42}; run 2026-10-21T20:14:13+00:00
- `lambda_calendar_2__seed42` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 42}; run 2026-10-21T20:27:23+00:00
- `lambda_calendar_3__seed42` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 42}; run 2026-10-21T20:41:20+00:00
- `window_12__seed42` (window): fit only on feature rows from the last 12 weeks before each cutoff; delta {"data.train_window_weeks": 12, "xgboost.random_state": 42}; run 2026-10-21T20:44:47+00:00
- `window_26__seed42` (window): fit only on feature rows from the last 26 weeks before each cutoff; delta {"data.train_window_weeks": 26, "xgboost.random_state": 42}; run 2026-10-21T20:49:01+00:00
- `window_52__seed42` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 42}; run 2026-10-21T20:54:23+00:00
- `oversample_2__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-21T21:08:44+00:00
- `oversample_3__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-21T21:24:18+00:00
- `oversample_5__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-21T21:41:52+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name covers different dates in different schemes, so never compare across schemes.
