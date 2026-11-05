# peak_rectification: results

Generated 2026-11-05T05:52:37+00:00 from `log.jsonl` (2 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row (of the same seed, when several seeds ran).

Phase scheme: `adiga_2026_10`.

Split: train from 2022-02-05, eval 2024-10-01..2025-05-31, 35 cutoff(s) at stride 1: 2024-10-05, 2024-10-12, 2024-10-19, 2024-10-26, 2024-11-02, 2024-11-09, 2024-11-16, 2024-11-23, 2024-11-30, 2024-12-07, 2024-12-14, 2024-12-21, 2024-12-28, 2025-01-04, 2025-01-11, 2025-01-18, 2025-01-25, 2025-02-01, 2025-02-08, 2025-02-15, 2025-02-22, 2025-03-01, 2025-03-08, 2025-03-15, 2025-03-22, 2025-03-29, 2025-04-05, 2025-04-12, 2025-04-19, 2025-04-26, 2025-05-03, 2025-05-10, 2025-05-17, 2025-05-24, 2025-05-31. Excluded from scoring: US. Seeds: 43. Git 9eb9a76387, data cache 2026-10-15T17:36:37+00:00 (sha256 c91d9d81b8ab).

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline__seed43` | 359.81 | 117.86 | -436.5 | +0.00 | +0.0% | no | 596.5 |
| oversample | `oversample_5__seed43` | 369.88 | 121.29 | -397.5 | +10.07 | -2.8% | no | 768.9 |

## Peak WIS by horizon

Peak-phase WIS on each horizon's forecasts alone (`by_phase_horizon`), mean over seeds; delta vs `baseline`, and with several seeds the min..max of the same-seed paired deltas. `n/a` when the phase is absent from that horizon's forecasts or the row was logged before phase x horizon was recorded.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 217.4 (+0.0) | 308.4 (+0.0) | 413.1 (+0.0) | 500.3 (+0.0) |
| `oversample_5` | 194.3 (-23.1) | 334.2 (+25.8) | 431.1 (+18.0) | 519.9 (+19.6) |

## Overall WIS by horizon

All scored phases (`by_horizon`), mean over seeds; delta vs `baseline`. The pooled WIS above is the mean of these over horizons, so it can hide an arm that helps h1/h2 and hurts h3/h4.

| config | h1 (delta) | h2 (delta) | h3 (delta) | h4 (delta) |
|---|---:|---:|---:|---:|
| `baseline` | 72.7 (+0.0) | 102.7 (+0.0) | 136.8 (+0.0) | 159.3 (+0.0) |
| `oversample_5` | 64.6 (-8.1) | 108.2 (+5.5) | 145.4 (+8.6) | 167.1 (+7.7) |

## Configs

- `baseline__seed43` (baseline): stock config; delta {"xgboost.random_state": 43}; run 2026-11-05T05:39:48+00:00
- `oversample_5__seed43` (oversample): peak-phase training rows by target date under adiga_2026_10 fitted 5x (4 extra copies each); delta {"data.oversample": {"factor": 5, "phase": "peak"}, "xgboost.random_state": 43}; run 2026-11-05T05:52:37+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name covers different dates in different schemes, so never compare across schemes.
