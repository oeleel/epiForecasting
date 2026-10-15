# peak_rectification: results

Generated 2026-10-15T17:09:45+00:00 from `log.jsonl` (11 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row (of the same seed, when several seeds ran).

Phase scheme: `adiga_2026_10`.

Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoff(s) at stride 4: 2025-10-04, 2025-11-01, 2025-11-29, 2025-12-27, 2026-01-24, 2026-02-21, 2026-03-21, 2026-04-18, 2026-05-16. Excluded from scoring: US. Seeds: 44. Git 7bee4a9573, data cache 2026-10-15T15:23:56+00:00 (sha256 c91d9d81b8ab).

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline__seed44` | 106.05 | 49.95 | -40.5 | +0.00 | +0.0% | no | 197.0 |
| lambda | `lambda_1.5__seed44` | 107.69 | 50.32 | -17.8 | +1.64 | -1.6% | no | 203.7 |
| lambda | `lambda_2__seed44` | 108.10 | 50.29 | -7.0 | +2.05 | -1.9% | no | 202.5 |
| lambda | `lambda_3__seed44` | 109.02 | 50.03 | 11.1 | +2.97 | -2.8% | no | 297.4 |
| lambda | `lambda_5__seed44` | 110.63 | 50.13 | 20.5 | +4.58 | -4.3% | no | 429.5 |
| lambda_calendar | `lambda_calendar_2__seed44` | 103.09 | 48.95 | -12.1 | -2.96 | +2.8% | yes | 455.2 |
| lambda_calendar | `lambda_calendar_3__seed44` | 106.04 | 50.10 | 9.9 | -0.01 | +0.0% | yes | 235.6 |
| window | `window_52__seed44` | 172.16 | 71.30 | 72.4 | +66.11 | -62.3% | no | 97.3 |
| oversample | `oversample_2__seed44` | 105.77 | 50.05 | -13.4 | -0.28 | +0.3% | yes | 217.4 |
| oversample | `oversample_3__seed44` | 104.87 | 49.85 | -6.0 | -1.18 | +1.1% | yes | 233.1 |
| oversample | `oversample_5__seed44` | 107.38 | 50.35 | 17.1 | +1.33 | -1.2% | no | 240.7 |

## Peak WIS by horizon

Peak-phase WIS on each horizon's forecasts alone (`by_phase_horizon`), mean over seeds; delta vs `baseline`, and with several seeds the min..max of the same-seed paired deltas. `n/a` when the phase is absent from that horizon's forecasts or the row was logged before phase x horizon was recorded.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 68.4 (+0.0) | 99.9 (+0.0) | 128.1 (+0.0) | 146.7 (+0.0) |
| `lambda_1.5` | 63.9 (-4.4) | 98.0 (-1.9) | 133.6 (+5.5) | 157.1 (+10.5) |
| `lambda_2` | 61.9 (-6.4) | 96.4 (-3.4) | 137.3 (+9.2) | 159.8 (+13.1) |
| `lambda_3` | 60.4 (-8.0) | 93.5 (-6.3) | 138.6 (+10.5) | 167.8 (+21.2) |
| `lambda_5` | 57.9 (-10.5) | 92.0 (-7.8) | 143.5 (+15.4) | 175.5 (+28.9) |
| `lambda_calendar_2` | 59.8 (-8.6) | 91.0 (-8.9) | 127.9 (-0.2) | 155.3 (+8.6) |
| `lambda_calendar_3` | 60.7 (-7.7) | 87.3 (-12.6) | 132.0 (+3.8) | 166.8 (+20.2) |
| `window_52` | 91.5 (+23.1) | 159.3 (+59.4) | 218.1 (+90.0) | 260.1 (+113.5) |
| `oversample_2` | 63.4 (-5.0) | 91.1 (-8.7) | 129.0 (+0.8) | 160.8 (+14.1) |
| `oversample_3` | 60.5 (-7.8) | 90.0 (-9.9) | 127.4 (-0.7) | 163.7 (+17.1) |
| `oversample_5` | 59.4 (-9.0) | 88.3 (-11.5) | 134.3 (+6.2) | 171.5 (+24.8) |

## Overall WIS by horizon

All scored phases (`by_horizon`), mean over seeds; delta vs `baseline`. The pooled WIS above is the mean of these over horizons, so it can hide an arm that helps h1/h2 and hurts h3/h4.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 29.1 (+0.0) | 36.9 (+0.0) | 54.5 (+0.0) | 79.3 (+0.0) |
| `lambda_1.5` | 27.6 (-1.5) | 36.4 (-0.5) | 55.1 (+0.6) | 82.1 (+2.8) |
| `lambda_2` | 26.8 (-2.3) | 36.1 (-0.8) | 55.5 (+0.9) | 82.9 (+3.6) |
| `lambda_3` | 26.0 (-3.1) | 35.1 (-1.8) | 55.9 (+1.4) | 83.1 (+3.8) |
| `lambda_5` | 25.4 (-3.7) | 35.1 (-1.8) | 56.3 (+1.8) | 83.8 (+4.5) |
| `lambda_calendar_2` | 26.2 (-2.9) | 35.1 (-1.8) | 53.7 (-0.9) | 80.8 (+1.5) |
| `lambda_calendar_3` | 26.6 (-2.5) | 34.5 (-2.5) | 55.4 (+0.9) | 84.0 (+4.7) |
| `window_52` | 37.5 (+8.4) | 52.5 (+15.6) | 79.2 (+24.7) | 116.0 (+36.7) |
| `oversample_2` | 27.6 (-1.5) | 35.2 (-1.7) | 54.5 (-0.0) | 83.0 (+3.7) |
| `oversample_3` | 26.6 (-2.5) | 35.3 (-1.6) | 54.3 (-0.2) | 83.3 (+4.0) |
| `oversample_5` | 26.1 (-3.0) | 35.2 (-1.7) | 56.5 (+2.0) | 83.6 (+4.3) |

## Configs

- `baseline__seed44` (baseline): stock config; delta {"xgboost.random_state": 44}; run 2026-10-15T15:53:42+00:00
- `lambda_1.5__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 44}; run 2026-10-15T15:57:06+00:00
- `lambda_2__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 44}; run 2026-10-15T16:00:29+00:00
- `lambda_3__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 44}; run 2026-10-15T16:06:34+00:00
- `lambda_5__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 44}; run 2026-10-15T16:29:38+00:00
- `lambda_calendar_2__seed44` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 44}; run 2026-10-15T16:52:41+00:00
- `lambda_calendar_3__seed44` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 44}; run 2026-10-15T16:56:36+00:00
- `window_52__seed44` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 44}; run 2026-10-15T16:58:14+00:00
- `oversample_2__seed44` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 44}; run 2026-10-15T17:01:51+00:00
- `oversample_3__seed44` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 44}; run 2026-10-15T17:05:44+00:00
- `oversample_5__seed44` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 44}; run 2026-10-15T17:09:45+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name covers different dates in different schemes, so never compare across schemes.
