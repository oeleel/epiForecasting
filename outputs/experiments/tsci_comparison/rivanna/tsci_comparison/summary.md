# TSci vs our framework - XGBoost on the flu data

Rows scored: 60 (location, cutoff, horizon) keys forecast by both frameworks; 5 locations, cutoffs 2025-11-29, 2025-12-27, 2026-01-24. Metric: MAE of the point forecast (weekly hospital admissions). `change_pct` below zero means the framework improved on its own baseline.

## Overall

| framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|
| ours | 60 | 541.7 | 473.5 | -12.6% |
| tsci | 60 | 644.4 | 683.3 | +6.0% |

## By cutoff

| cutoff_date | framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|---|
| 2025-11-29 | ours | 20 | 498.2 | 367.3 | -26.3% |
| 2025-11-29 | tsci | 20 | 741.9 | 879.2 | +18.5% |
| 2025-12-27 | ours | 20 | 490.2 | 490.2 | +0.0% |
| 2025-12-27 | tsci | 20 | 828.7 | 673.6 | -18.7% |
| 2026-01-24 | ours | 20 | 636.8 | 563.1 | -11.6% |
| 2026-01-24 | tsci | 20 | 362.5 | 497.2 | +37.2% |

## By horizon

| horizon | framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|---|
| 1 | ours | 15 | 389.7 | 370.3 | -5.0% |
| 1 | tsci | 15 | 195.2 | 254.2 | +30.2% |
| 2 | ours | 15 | 481.7 | 475.5 | -1.3% |
| 2 | tsci | 15 | 447.8 | 478.9 | +6.9% |
| 3 | ours | 15 | 610.9 | 487.4 | -20.2% |
| 3 | tsci | 15 | 851.8 | 864.9 | +1.5% |
| 4 | ours | 15 | 684.5 | 560.8 | -18.1% |
| 4 | tsci | 15 | 1082.7 | 1135.4 | +4.9% |

## Our framework on every location it forecast (not comparable with TSci)

| framework | n | baseline_mae | final_mae | change_pct |
|---|---|---|---|---|
| ours | 624 | 172.8 | 158.8 | -8.1% |
