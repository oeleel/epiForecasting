# Model Improvement Summary

*Generated: 2026-02-10 06:44:16*

## Overview

- **Baseline MAPE**: 59.00%
- **Final MAPE**: 35.10%
- **MAPE Reduction**: 23.9 percentage points
- **Relative Improvement**: 40.5%

---

## Overall Metrics by Step

| Step | Name | MAPE (%) | MAE | RMSE | MAPE Reduction (pp) | Relative Impr. |
|------|------|----------|-----|------|---------------------|----------------|
| 0 | Baseline | 59.00 | N/A | N/A | 0.0 | 0.0% |
| 2 | Target Transformation | 55.00 | 635.6 | 3601.1 | 4.0 | 6.8% |
| 3 | Regularization | 54.55 | 628.5 | 3583.3 | 4.5 | 7.5% |
| 4 | Feature Engineering | 53.30 | 627.9 | 3581.5 | 5.7 | 9.7% |
| 5 | Expanded Validation | 36.07 | 444.0 | 2456.2 | 22.9 | 38.9% |
| 6 | Quantile Regression | 35.10 | 449.8 | 2456.7 | 23.9 | 40.5% |
| 7 | Location Clustering | 35.10 | 449.8 | 2456.7 | 23.9 | 40.5% |

---

## Per-Horizon MAPE by Step

| Step | Name | H1 (%) | H2 (%) | H3 (%) | H4 (%) |
|------|------|--------|--------|--------|--------|
| 0 | Baseline | N/A | N/A | N/A | N/A |
| 2 | Target Transformation | 37.4 | 51.3 | 61.8 | 69.2 |
| 3 | Regularization | 37.4 | 50.2 | 63.4 | 66.9 |
| 4 | Feature Engineering | 36.2 | 48.6 | 62.4 | 65.7 |
| 5 | Expanded Validation | 23.0 | 30.8 | 38.3 | 52.2 |
| 6 | Quantile Regression | 25.4 | 30.3 | 36.6 | 48.1 |
| 7 | Location Clustering | 26.3 | 35.1 | 39.6 | 48.8 |

---

## Train/Validation Gap Ratio by Step

| Step | Name | Gap Ratio (H1) | Status |
|------|------|----------------|--------|
| 0 | Baseline | 39.90 | Overfitting |
| 2 | Target Transformation | N/A | - |
| 3 | Regularization | 1.30 | Excellent |
| 4 | Feature Engineering | 1.32 | Excellent |
| 5 | Expanded Validation | 1.31 | Excellent |
| 6 | Quantile Regression | N/A | - |
| 7 | Location Clustering | N/A | - |

---

## Incremental Improvements

| Step | Name | Incremental Improvement |
|------|------|-------------------------|
| 2 | Target Transformation | +6.78% |
| 3 | Regularization | +0.82% |
| 4 | Feature Engineering | +2.29% |
| 5 | Expanded Validation | +32.32% |
| 6 | Quantile Regression | +2.69% |
| 7 | Location Clustering | 0.00% |

---

## Key Findings

### Most Impactful Step: **Step 5 - Expanded Validation**
- Incremental Improvement: **32.32%**

### Summary
- Started at **59.00%** MAPE (baseline)
- Ended at **35.10%** MAPE
- **MAPE Reduction: 23.9 percentage points**
- Relative Improvement: 40.5%
