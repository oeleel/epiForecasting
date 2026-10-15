# peak_rectification: results

Generated 2026-10-15T17:09:49+00:00 from `log.jsonl` (11 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row (of the same seed, when several seeds ran).

Phase scheme: `adiga_2026_10`.

Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoff(s) at stride 4: 2025-10-04, 2025-11-01, 2025-11-29, 2025-12-27, 2026-01-24, 2026-02-21, 2026-03-21, 2026-04-18, 2026-05-16. Excluded from scoring: US. Seeds: 43. Git 7bee4a9573, data cache 2026-10-15T15:23:56+00:00 (sha256 c91d9d81b8ab).

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline__seed43` | 107.27 | 50.46 | -46.0 | +0.00 | +0.0% | no | 195.0 |
| lambda | `lambda_1.5__seed43` | 108.97 | 50.68 | -24.0 | +1.70 | -1.6% | no | 206.5 |
| lambda | `lambda_2__seed43` | 110.11 | 50.71 | -4.6 | +2.84 | -2.6% | no | 205.2 |
| lambda | `lambda_3__seed43` | 110.20 | 50.73 | 15.0 | +2.93 | -2.7% | no | 297.3 |
| lambda | `lambda_5__seed43` | 111.50 | 50.74 | 25.7 | +4.23 | -3.9% | no | 439.9 |
| lambda_calendar | `lambda_calendar_2__seed43` | 106.80 | 49.93 | -13.0 | -0.47 | +0.4% | yes | 445.4 |
| lambda_calendar | `lambda_calendar_3__seed43` | 107.17 | 50.06 | 4.5 | -0.10 | +0.1% | yes | 239.4 |
| window | `window_52__seed43` | 180.25 | 72.63 | 94.3 | +72.98 | -68.0% | no | 95.9 |
| oversample | `oversample_2__seed43` | 107.49 | 50.55 | -18.2 | +0.22 | -0.2% | no | 219.3 |
| oversample | `oversample_3__seed43` | 108.82 | 50.85 | -3.5 | +1.55 | -1.4% | no | 228.2 |
| oversample | `oversample_5__seed43` | 107.44 | 50.34 | 6.0 | +0.17 | -0.2% | no | 241.2 |

## Peak WIS by horizon

Peak-phase WIS on each horizon's forecasts alone (`by_phase_horizon`), mean over seeds; delta vs `baseline`, and with several seeds the min..max of the same-seed paired deltas. `n/a` when the phase is absent from that horizon's forecasts or the row was logged before phase x horizon was recorded.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 67.0 (+0.0) | 103.9 (+0.0) | 128.9 (+0.0) | 149.4 (+0.0) |
| `lambda_1.5` | 67.6 (+0.6) | 99.4 (-4.5) | 133.6 (+4.7) | 156.0 (+6.6) |
| `lambda_2` | 63.9 (-3.1) | 96.9 (-7.0) | 139.9 (+11.0) | 162.9 (+13.5) |
| `lambda_3` | 59.6 (-7.4) | 95.6 (-8.3) | 146.0 (+17.1) | 165.0 (+15.5) |
| `lambda_5` | 57.1 (-9.9) | 94.6 (-9.3) | 154.7 (+25.8) | 166.8 (+17.4) |
| `lambda_calendar_2` | 62.2 (-4.8) | 94.1 (-9.8) | 130.6 (+1.8) | 162.5 (+13.1) |
| `lambda_calendar_3` | 60.6 (-6.4) | 90.2 (-13.7) | 130.9 (+2.0) | 170.2 (+20.8) |
| `window_52` | 91.8 (+24.8) | 155.2 (+51.2) | 243.3 (+114.4) | 275.0 (+125.6) |
| `oversample_2` | 63.9 (-3.1) | 97.8 (-6.1) | 129.9 (+1.1) | 160.1 (+10.7) |
| `oversample_3` | 63.6 (-3.4) | 91.5 (-12.4) | 131.2 (+2.3) | 171.7 (+22.2) |
| `oversample_5` | 58.8 (-8.2) | 89.9 (-14.0) | 134.3 (+5.4) | 171.1 (+21.7) |

## Overall WIS by horizon

All scored phases (`by_horizon`), mean over seeds; delta vs `baseline`. The pooled WIS above is the mean of these over horizons, so it can hide an arm that helps h1/h2 and hurts h3/h4.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 28.5 (+0.0) | 37.8 (+0.0) | 54.4 (+0.0) | 81.1 (+0.0) |
| `lambda_1.5` | 28.7 (+0.2) | 37.0 (-0.8) | 54.7 (+0.3) | 82.3 (+1.2) |
| `lambda_2` | 27.4 (-1.1) | 36.3 (-1.5) | 55.7 (+1.3) | 83.4 (+2.3) |
| `lambda_3` | 25.8 (-2.8) | 36.1 (-1.7) | 57.7 (+3.4) | 83.3 (+2.2) |
| `lambda_5` | 24.9 (-3.6) | 35.9 (-1.9) | 59.2 (+4.9) | 82.9 (+1.8) |
| `lambda_calendar_2` | 26.9 (-1.6) | 36.1 (-1.7) | 54.5 (+0.1) | 82.2 (+1.1) |
| `lambda_calendar_3` | 26.4 (-2.1) | 35.2 (-2.6) | 54.9 (+0.5) | 83.7 (+2.6) |
| `window_52` | 37.5 (+9.0) | 51.6 (+13.8) | 83.6 (+29.2) | 117.8 (+36.7) |
| `oversample_2` | 27.6 (-0.9) | 37.0 (-0.8) | 54.8 (+0.4) | 82.8 (+1.6) |
| `oversample_3` | 27.5 (-1.0) | 35.7 (-2.1) | 54.9 (+0.5) | 85.2 (+4.1) |
| `oversample_5` | 25.8 (-2.7) | 35.3 (-2.5) | 55.7 (+1.3) | 84.5 (+3.4) |

## Configs

- `baseline__seed43` (baseline): stock config; delta {"xgboost.random_state": 43}; run 2026-10-15T15:53:40+00:00
- `lambda_1.5__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 43}; run 2026-10-15T15:57:07+00:00
- `lambda_2__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 43}; run 2026-10-15T16:00:32+00:00
- `lambda_3__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 43}; run 2026-10-15T16:06:37+00:00
- `lambda_5__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 43}; run 2026-10-15T16:29:51+00:00
- `lambda_calendar_2__seed43` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 43}; run 2026-10-15T16:52:45+00:00
- `lambda_calendar_3__seed43` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 43}; run 2026-10-15T16:56:44+00:00
- `window_52__seed43` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 43}; run 2026-10-15T16:58:20+00:00
- `oversample_2__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-10-15T17:01:59+00:00
- `oversample_3__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-10-15T17:05:47+00:00
- `oversample_5__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-10-15T17:09:49+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name covers different dates in different schemes, so never compare across schemes.
