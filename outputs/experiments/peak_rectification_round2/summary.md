# peak_rectification: results

Generated 2026-10-15T17:09:51+00:00 from `log.jsonl` (33 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row (of the same seed, when several seeds ran).

Phase scheme: `adiga_2026_10`.

Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoff(s) at stride 4: 2025-10-04, 2025-11-01, 2025-11-29, 2025-12-27, 2026-01-24, 2026-02-21, 2026-03-21, 2026-04-18, 2026-05-16. Excluded from scoring: US. Seeds: 42, 43, 44. Git 7bee4a9573, data cache 2026-10-15T15:23:56+00:00 (sha256 c91d9d81b8ab).

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline__seed42` | 108.53 | 50.59 | -38.2 | +0.00 | +0.0% | no | 195.7 |
| baseline | `baseline__seed43` | 107.27 | 50.46 | -46.0 | +0.00 | +0.0% | no | 195.0 |
| baseline | `baseline__seed44` | 106.05 | 49.95 | -40.5 | +0.00 | +0.0% | no | 197.0 |
| lambda | `lambda_1.5__seed42` | 109.88 | 50.79 | -13.5 | +1.35 | -1.2% | no | 205.9 |
| lambda | `lambda_2__seed42` | 109.96 | 50.67 | 2.1 | +1.43 | -1.3% | no | 203.4 |
| lambda | `lambda_3__seed42` | 106.87 | 49.71 | 11.3 | -1.66 | +1.5% | yes | 290.0 |
| lambda | `lambda_5__seed42` | 111.70 | 50.82 | 20.7 | +3.17 | -2.9% | no | 428.8 |
| lambda | `lambda_1.5__seed43` | 108.97 | 50.68 | -24.0 | +1.70 | -1.6% | no | 206.5 |
| lambda | `lambda_2__seed43` | 110.11 | 50.71 | -4.6 | +2.84 | -2.6% | no | 205.2 |
| lambda | `lambda_3__seed43` | 110.20 | 50.73 | 15.0 | +2.93 | -2.7% | no | 297.3 |
| lambda | `lambda_5__seed43` | 111.50 | 50.74 | 25.7 | +4.23 | -3.9% | no | 439.9 |
| lambda | `lambda_1.5__seed44` | 107.69 | 50.32 | -17.8 | +1.64 | -1.6% | no | 203.7 |
| lambda | `lambda_2__seed44` | 108.10 | 50.29 | -7.0 | +2.05 | -1.9% | no | 202.5 |
| lambda | `lambda_3__seed44` | 109.02 | 50.03 | 11.1 | +2.97 | -2.8% | no | 297.4 |
| lambda | `lambda_5__seed44` | 110.63 | 50.13 | 20.5 | +4.58 | -4.3% | no | 429.5 |
| lambda_calendar | `lambda_calendar_2__seed42` | 106.39 | 49.90 | -10.9 | -2.14 | +2.0% | yes | 459.4 |
| lambda_calendar | `lambda_calendar_3__seed42` | 107.66 | 50.28 | 2.0 | -0.87 | +0.8% | yes | 234.5 |
| lambda_calendar | `lambda_calendar_2__seed43` | 106.80 | 49.93 | -13.0 | -0.47 | +0.4% | yes | 445.4 |
| lambda_calendar | `lambda_calendar_3__seed43` | 107.17 | 50.06 | 4.5 | -0.10 | +0.1% | yes | 239.4 |
| lambda_calendar | `lambda_calendar_2__seed44` | 103.09 | 48.95 | -12.1 | -2.96 | +2.8% | yes | 455.2 |
| lambda_calendar | `lambda_calendar_3__seed44` | 106.04 | 50.10 | 9.9 | -0.01 | +0.0% | yes | 235.6 |
| window | `window_52__seed42` | 174.95 | 71.33 | 77.7 | +66.42 | -61.2% | no | 96.3 |
| window | `window_52__seed43` | 180.25 | 72.63 | 94.3 | +72.98 | -68.0% | no | 95.9 |
| window | `window_52__seed44` | 172.16 | 71.30 | 72.4 | +66.11 | -62.3% | no | 97.3 |
| oversample | `oversample_2__seed42` | 106.16 | 49.77 | -26.5 | -2.37 | +2.2% | yes | 220.3 |
| oversample | `oversample_3__seed42` | 106.16 | 50.08 | -12.0 | -2.37 | +2.2% | yes | 231.7 |
| oversample | `oversample_5__seed42` | 106.78 | 49.89 | 5.5 | -1.75 | +1.6% | yes | 240.0 |
| oversample | `oversample_2__seed43` | 107.49 | 50.55 | -18.2 | +0.22 | -0.2% | no | 219.3 |
| oversample | `oversample_3__seed43` | 108.82 | 50.85 | -3.5 | +1.55 | -1.4% | no | 228.2 |
| oversample | `oversample_5__seed43` | 107.44 | 50.34 | 6.0 | +0.17 | -0.2% | no | 241.2 |
| oversample | `oversample_2__seed44` | 105.77 | 50.05 | -13.4 | -0.28 | +0.3% | yes | 217.4 |
| oversample | `oversample_3__seed44` | 104.87 | 49.85 | -6.0 | -1.18 | +1.1% | yes | 233.1 |
| oversample | `oversample_5__seed44` | 107.38 | 50.35 | 17.1 | +1.33 | -1.2% | no | 240.7 |

## Across seeds

Mean and sample sd over seeds. `delta vs baseline mean` compares means; `paired deltas` are config minus baseline at the same seed, the comparison that cancels seed noise.

| config | seeds | peak WIS mean | sd | overall WIS mean | sd | delta peak WIS vs baseline mean | paired peak WIS deltas (seed: delta) |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 3 | 107.28 | 1.24 | 50.33 | 0.34 | +0.00 | 42: +0.00, 43: +0.00, 44: +0.00 |
| `lambda_1.5` | 3 | 108.85 | 1.10 | 50.60 | 0.25 | +1.56 | 42: +1.35, 43: +1.70, 44: +1.64 |
| `lambda_2` | 3 | 109.39 | 1.12 | 50.56 | 0.23 | +2.11 | 42: +1.43, 43: +2.84, 44: +2.05 |
| `lambda_3` | 3 | 108.70 | 1.69 | 50.16 | 0.52 | +1.41 | 42: -1.66, 43: +2.93, 44: +2.97 |
| `lambda_5` | 3 | 111.28 | 0.57 | 50.56 | 0.38 | +3.99 | 42: +3.17, 43: +4.23, 44: +4.58 |
| `lambda_calendar_2` | 3 | 105.43 | 2.03 | 49.59 | 0.56 | -1.86 | 42: -2.14, 43: -0.47, 44: -2.96 |
| `lambda_calendar_3` | 3 | 106.96 | 0.83 | 50.15 | 0.12 | -0.33 | 42: -0.87, 43: -0.10, 44: -0.01 |
| `window_52` | 3 | 175.79 | 4.11 | 71.75 | 0.76 | +68.50 | 42: +66.42, 43: +72.98, 44: +66.11 |
| `oversample_2` | 3 | 106.47 | 0.90 | 50.12 | 0.40 | -0.81 | 42: -2.37, 43: +0.22, 44: -0.28 |
| `oversample_3` | 3 | 106.62 | 2.01 | 50.26 | 0.52 | -0.67 | 42: -2.37, 43: +1.55, 44: -1.18 |
| `oversample_5` | 3 | 107.20 | 0.36 | 50.19 | 0.26 | -0.08 | 42: -1.75, 43: +0.17, 44: +1.33 |

## Peak WIS by horizon

Peak-phase WIS on each horizon's forecasts alone (`by_phase_horizon`), mean over seeds; delta vs `baseline`, and with several seeds the min..max of the same-seed paired deltas. `n/a` when the phase is absent from that horizon's forecasts or the row was logged before phase x horizon was recorded.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 67.9 (+0.0; paired +0.0..+0.0) | 101.1 (+0.0; paired +0.0..+0.0) | 129.7 (+0.0; paired +0.0..+0.0) | 150.1 (+0.0; paired +0.0..+0.0) |
| `lambda_1.5` | 66.2 (-1.7; paired -4.4..+0.6) | 98.6 (-2.4; paired -4.5..-0.9) | 134.1 (+4.4; paired +3.2..+5.5) | 157.7 (+7.5; paired +5.6..+10.5) |
| `lambda_2` | 62.7 (-5.2; paired -6.4..-3.1) | 96.4 (-4.6; paired -7.0..-3.4) | 139.2 (+9.5; paired +8.4..+11.0) | 162.5 (+12.3; paired +10.4..+13.5) |
| `lambda_3` | 59.9 (-8.0; paired -8.6..-7.4) | 93.6 (-7.4; paired -8.3..-6.3) | 141.3 (+11.6; paired +7.3..+17.1) | 164.3 (+14.1; paired +5.7..+21.2) |
| `lambda_5` | 57.7 (-10.3; paired -10.5..-9.9) | 93.3 (-7.7; paired -9.3..-6.1) | 147.5 (+17.8; paired +12.2..+25.8) | 173.4 (+23.3; paired +17.4..+28.9) |
| `lambda_calendar_2` | 61.6 (-6.3; paired -8.6..-4.8) | 92.8 (-8.2; paired -9.8..-6.0) | 129.8 (+0.1; paired -1.2..+1.8) | 159.3 (+9.2; paired +5.8..+13.1) |
| `lambda_calendar_3` | 60.7 (-7.2; paired -7.7..-6.4) | 89.1 (-12.0; paired -13.7..-9.8) | 130.9 (+1.2; paired -2.4..+3.8) | 170.2 (+20.1; paired +19.4..+20.8) |
| `window_52` | 92.5 (+24.6; paired +23.1..+26.0) | 156.3 (+55.2; paired +51.2..+59.4) | 231.2 (+101.4; paired +90.0..+114.4) | 264.8 (+114.7; paired +104.9..+125.6) |
| `oversample_2` | 63.7 (-4.3; paired -5.0..-3.1) | 95.1 (-6.0; paired -8.7..-3.1) | 128.6 (-1.1; paired -5.1..+1.1) | 159.9 (+9.8; paired +4.7..+14.1) |
| `oversample_3` | 61.8 (-6.2; paired -7.8..-3.4) | 90.5 (-10.6; paired -12.4..-9.3) | 128.2 (-1.6; paired -6.2..+2.3) | 168.5 (+18.4; paired +15.8..+22.2) |
| `oversample_5` | 59.4 (-8.5; paired -9.0..-8.2) | 88.0 (-13.1; paired -14.0..-11.5) | 133.4 (+3.7; paired -0.6..+6.2) | 171.9 (+21.8; paired +18.8..+24.8) |

## Overall WIS by horizon

All scored phases (`by_horizon`), mean over seeds; delta vs `baseline`. The pooled WIS above is the mean of these over horizons, so it can hide an arm that helps h1/h2 and hurts h3/h4.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 28.9 (+0.0; paired +0.0..+0.0) | 37.2 (+0.0; paired +0.0..+0.0) | 54.6 (+0.0; paired +0.0..+0.0) | 80.7 (+0.0; paired +0.0..+0.0) |
| `lambda_1.5` | 28.3 (-0.6; paired -1.5..+0.2) | 36.6 (-0.6; paired -0.8..-0.4) | 54.9 (+0.3; paired +0.1..+0.6) | 82.5 (+1.9; paired +1.2..+2.8) |
| `lambda_2` | 27.0 (-1.9; paired -2.3..-1.1) | 36.1 (-1.1; paired -1.5..-0.8) | 55.7 (+1.1; paired +0.9..+1.3) | 83.4 (+2.7; paired +2.2..+3.6) |
| `lambda_3` | 25.9 (-2.9; paired -3.1..-2.8) | 35.4 (-1.8; paired -1.8..-1.7) | 56.5 (+1.9; paired +0.9..+3.4) | 82.8 (+2.1; paired +0.3..+3.8) |
| `lambda_5` | 25.2 (-3.7; paired -3.7..-3.6) | 35.6 (-1.6; paired -1.9..-1.0) | 57.5 (+2.9; paired +1.8..+4.9) | 84.0 (+3.3; paired +1.8..+4.5) |
| `lambda_calendar_2` | 26.8 (-2.1; paired -2.9..-1.6) | 35.6 (-1.5; paired -1.8..-1.1) | 54.2 (-0.4; paired -0.9..+0.1) | 81.7 (+1.1; paired +0.5..+1.5) |
| `lambda_calendar_3` | 26.5 (-2.4; paired -2.5..-2.1) | 34.9 (-2.2; paired -2.6..-1.7) | 55.0 (+0.4; paired -0.4..+0.9) | 84.2 (+3.5; paired +2.6..+4.7) |
| `window_52` | 37.8 (+8.9; paired +8.4..+9.5) | 51.7 (+14.5; paired +13.8..+15.6) | 81.2 (+26.6; paired +24.7..+29.2) | 116.3 (+35.6; paired +33.5..+36.7) |
| `oversample_2` | 27.6 (-1.3; paired -1.5..-0.9) | 36.2 (-0.9; paired -1.7..-0.3) | 54.2 (-0.4; paired -1.5..+0.4) | 82.5 (+1.8; paired +0.0..+3.7) |
| `oversample_3` | 26.9 (-2.0; paired -2.5..-1.0) | 35.5 (-1.7; paired -2.1..-1.4) | 54.4 (-0.2; paired -1.0..+0.5) | 84.3 (+3.6; paired +2.7..+4.1) |
| `oversample_5` | 26.1 (-2.8; paired -3.0..-2.7) | 35.1 (-2.1; paired -2.5..-1.7) | 55.7 (+1.1; paired +0.1..+2.0) | 83.9 (+3.3; paired +2.0..+4.3) |

## Configs

- `baseline__seed42` (baseline): stock config; delta {"xgboost.random_state": 42}; run 2026-10-15T15:53:41+00:00
- `baseline__seed43` (baseline): stock config; delta {"xgboost.random_state": 43}; run 2026-10-15T15:53:40+00:00
- `baseline__seed44` (baseline): stock config; delta {"xgboost.random_state": 44}; run 2026-10-15T15:53:42+00:00
- `lambda_1.5__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 42}; run 2026-10-15T15:57:07+00:00
- `lambda_2__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 42}; run 2026-10-15T16:00:31+00:00
- `lambda_3__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 42}; run 2026-10-15T16:06:28+00:00
- `lambda_5__seed42` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 42}; run 2026-10-15T16:29:32+00:00
- `lambda_1.5__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 43}; run 2026-10-15T15:57:07+00:00
- `lambda_2__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 43}; run 2026-10-15T16:00:32+00:00
- `lambda_3__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 43}; run 2026-10-15T16:06:37+00:00
- `lambda_5__seed43` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 43}; run 2026-10-15T16:29:51+00:00
- `lambda_1.5__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}, "xgboost.random_state": 44}; run 2026-10-15T15:57:06+00:00
- `lambda_2__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}, "xgboost.random_state": 44}; run 2026-10-15T16:00:29+00:00
- `lambda_3__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}, "xgboost.random_state": 44}; run 2026-10-15T16:06:34+00:00
- `lambda_5__seed44` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}, "xgboost.random_state": 44}; run 2026-10-15T16:29:38+00:00
- `lambda_calendar_2__seed42` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 42}; run 2026-10-15T16:52:39+00:00
- `lambda_calendar_3__seed42` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 42}; run 2026-10-15T16:56:33+00:00
- `lambda_calendar_2__seed43` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 43}; run 2026-10-15T16:52:45+00:00
- `lambda_calendar_3__seed43` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 43}; run 2026-10-15T16:56:44+00:00
- `lambda_calendar_2__seed44` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 2x; delta {"sample_weights.by_phase.peak": 2.0, "xgboost.random_state": 44}; run 2026-10-15T16:52:41+00:00
- `lambda_calendar_3__seed44` (lambda_calendar): peak-phase rows by target date (origin + h weeks) under adiga_2026_10 weighted 3x; delta {"sample_weights.by_phase.peak": 3.0, "xgboost.random_state": 44}; run 2026-10-15T16:56:36+00:00
- `window_52__seed42` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 42}; run 2026-10-15T16:58:10+00:00
- `window_52__seed43` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 43}; run 2026-10-15T16:58:20+00:00
- `window_52__seed44` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52, "xgboost.random_state": 44}; run 2026-10-15T16:58:14+00:00
- `oversample_2__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-15T17:01:50+00:00
- `oversample_3__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-15T17:05:42+00:00
- `oversample_5__seed42` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 42}; run 2026-10-15T17:09:42+00:00
- `oversample_2__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-10-15T17:01:59+00:00
- `oversample_3__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-10-15T17:05:47+00:00
- `oversample_5__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-10-15T17:09:49+00:00
- `oversample_2__seed44` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 2x (1 extra copy each); delta {"data.oversample": {"factor": 2, "phase": "peak"}, "xgboost.random_state": 44}; run 2026-10-15T17:01:51+00:00
- `oversample_3__seed44` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 3x (2 extra copies each); delta {"data.oversample": {"factor": 3, "phase": "peak"}, "xgboost.random_state": 44}; run 2026-10-15T17:05:44+00:00
- `oversample_5__seed44` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 44}; run 2026-10-15T17:09:45+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name covers different dates in different schemes, so never compare across schemes.
