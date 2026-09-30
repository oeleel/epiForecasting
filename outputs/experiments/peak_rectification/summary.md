# peak_rectification: results

Generated 2026-09-30T20:24:23+00:00 from `log.jsonl` (10 configs). Do not edit by hand; rerun the script or pass `--summary-only`.

Target: peak WIS (lower is better). Guard: overall WIS. Every delta is vs this run's own `baseline` row.

Split: train from 2022-02-05, eval 2025-10-01..2026-05-31, 9 cutoff(s) at stride 4: 2025-10-04, 2025-11-01, 2025-11-29, 2025-12-27, 2026-01-24, 2026-02-21, 2026-03-21, 2026-04-18, 2026-05-16. Excluded from scoring: US. Git 13be50b0de (dirty), data cache 2026-09-30T18:44:41+00:00.

| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |
|---|---|---:|---:|---:|---:|---:|:---:|---:|
| baseline | `baseline` | 121.01 | 50.05 | -58.8 | +0.00 | +0.0% | no | 478.7 |
| lambda | `lambda_1.5` | 123.74 | 50.92 | -35.7 | +2.73 | -2.3% | no | 393.8 |
| lambda | `lambda_2` | 123.54 | 50.64 | -17.9 | +2.53 | -2.1% | no | 294.1 |
| lambda | `lambda_3` | 123.71 | 50.40 | -6.4 | +2.70 | -2.2% | no | 393.4 |
| lambda | `lambda_5` | 123.87 | 50.53 | 7.9 | +2.86 | -2.4% | no | 358.2 |
| lambda_calendar | `lambda_calendar_2` | 123.70 | 50.60 | -29.8 | +2.69 | -2.2% | no | 314.6 |
| lambda_calendar | `lambda_calendar_3` | 122.66 | 50.21 | -6.2 | +1.65 | -1.4% | no | 265.3 |
| window | `window_12` | 238.26 | 96.02 | -193.0 | +117.25 | -96.9% | no | 77.8 |
| window | `window_26` | 240.24 | 92.16 | -188.2 | +119.23 | -98.5% | no | 87.9 |
| window | `window_52` | 197.99 | 71.85 | 67.5 | +76.98 | -63.6% | no | 192.1 |

## Configs

- `baseline` (baseline): stock config; run 2026-09-30T20:10:48+00:00
- `lambda_1.5` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 1.5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 1.5}}; run 2026-09-30T20:17:23+00:00
- `lambda_2` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 2x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 2.0}}; run 2026-09-30T20:22:17+00:00
- `lambda_3` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 3x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}}; run 2026-09-30T20:18:13+00:00
- `lambda_5` (lambda): approaching-peak rows (K=6 weeks before the season max) weighted 5x; delta {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}}; run 2026-09-30T20:24:12+00:00
- `lambda_calendar_2` (lambda_calendar): calendar peak-phase rows (Dec-Jan) weighted 2x; delta {"sample_weights.by_phase.peak": 2.0}; run 2026-09-30T20:17:04+00:00
- `lambda_calendar_3` (lambda_calendar): calendar peak-phase rows (Dec-Jan) weighted 3x; delta {"sample_weights.by_phase.peak": 3.0}; run 2026-09-30T20:21:29+00:00
- `window_12` (window): fit only on feature rows from the last 12 weeks before each cutoff; delta {"data.train_window_weeks": 12}; run 2026-09-30T20:10:03+00:00
- `window_26` (window): fit only on feature rows from the last 26 weeks before each cutoff; delta {"data.train_window_weeks": 26}; run 2026-09-30T20:11:31+00:00
- `window_52` (window): fit only on feature rows from the last 52 weeks before each cutoff; delta {"data.train_window_weeks": 52}; run 2026-09-30T20:15:00+00:00

## Reading this table

- `n/a` means the metric is absent from that run's forecasts (for example a cutoff subset with no peak-phase target dates), so no reward was computed.
- `better` uses `agent.orchestrator._is_better` on the target metric only; check the overall WIS column before calling an arm a win.
- Phases are calendar phases (`agent.phase_evaluator.PhaseEvaluator.PHASE_MAP`), the same labelling every other number in this repo uses.
