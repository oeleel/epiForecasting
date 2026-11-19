# TSci vs our framework - XGBoost on the flu data

Rows scored: 60 (location, cutoff, horizon) keys forecast by both frameworks; 5 locations, cutoffs 2025-12-27, 2026-01-24, 2026-02-21. Metric: MAE of the point forecast (weekly hospital admissions). `change_pct` below zero means the framework improved on its own baseline.

## Overall

| framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|
| ours | 60 | 435.5 | 465.4 | +6.9% |
| tsci | 60 | 497.2 | 437.6 | -12.0% |

## By cutoff

| cutoff_date | framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|---|
| 2025-12-27 | ours | 20 | 490.2 | 473.0 | -3.5% |
| 2025-12-27 | tsci | 20 | 828.7 | 830.8 | +0.3% |
| 2026-01-24 | ours | 20 | 636.8 | 636.8 | +0.0% |
| 2026-01-24 | tsci | 20 | 362.5 | 361.8 | -0.2% |
| 2026-02-21 | ours | 20 | 179.6 | 286.4 | +59.5% |
| 2026-02-21 | tsci | 20 | 300.5 | 120.2 | -60.0% |

## By horizon

| horizon | framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|---|
| 1 | ours | 15 | 413.0 | 379.1 | -8.2% |
| 1 | tsci | 15 | 225.6 | 441.6 | +95.7% |
| 2 | ours | 15 | 457.6 | 423.1 | -7.5% |
| 2 | tsci | 15 | 397.5 | 467.6 | +17.6% |
| 3 | ours | 15 | 447.1 | 487.4 | +9.0% |
| 3 | tsci | 15 | 669.4 | 454.0 | -32.2% |
| 4 | ours | 15 | 424.3 | 572.0 | +34.8% |
| 4 | tsci | 15 | 696.4 | 387.3 | -44.4% |

## Our framework on every location it forecast (not comparable with TSci)

| framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|
| ours | 624 | 136.3 | 167.0 | +22.5% |
