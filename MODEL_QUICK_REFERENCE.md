# Flu Forecasting Model - Quick Reference

**Model:** XGBoost Direct Forecast Ensemble | **Version:** 2.0 | **Updated:** Jan 2025

---

## Model at a Glance

| | |
|---|---|
| **Architecture** | 4 separate XGBoost models (one per forecast horizon) |
| **Forecast Range** | 1-4 weeks ahead |
| **Locations** | All US states + territories |
| **Features** | 59 engineered features |
| **Training Cutoff** | November 2, 2024 |

---

## Performance (Nov 2024 - Apr 2025)

| Metric | Overall | Week 1 | Week 2 | Week 3 | Week 4 |
|--------|---------|--------|--------|--------|--------|
| **MAPE** | 59.0% | 51.4% | 51.1% | 63.8% | 70.9% |
| **MAE** | 515 | 400 | 504 | 558 | 599 |

⚠️ **Context:** 2024-2025 season was ~2-3x more severe than any previous season in training data.

---

## Key Features (59 total)

| Category | Count | Examples |
|----------|-------|----------|
| **Lag values** | 7 | 1-4 week, 8, 12, 52-week lags |
| **Rolling stats** | 9 | 4/8-week mean, std, min, max |
| **Temporal** | 11 | Week/month cyclical encoding |
| **US national** | 8 | National lags & rolling stats |
| **Year-over-year** | 3 | YoY ratio, % change, difference |
| **Rate of change** | 5 | WoW change, acceleration, momentum |
| **Season phase** | 4 | Onset/peak/decline indicators |
| **Other** | 6 | Rate features, interactions |

---

## Why Direct Forecasting?

| Recursive (Old) | Direct (Current) |
|-----------------|------------------|
| 1 model, iterated 4x | 4 specialized models |
| Errors accumulate | Independent predictions |
| Week 4 MAPE: ~130% | Week 4 MAPE: ~71% |

---

## Post-Prediction Constraints

**Floor Constraint:** Predictions can't drop below 30% of last known value
- Prevents unrealistic drops during severe seasons
- Improved MAPE by ~1%

---

## Limitations

1. **Unprecedented seasons** - Cannot predict events exceeding all historical data
2. **No external data** - No weather, vaccination, or mobility data
3. **Reporting lag** - Real-world performance depends on data availability

---

## Quick Start

```python
from feature_engineering import FeatureEngineer
from direct_forecast import DirectForecastEnsemble

# Prepare data
features_df = FeatureEngineer().create_all_features(data)

# Train & predict
ensemble = DirectForecastEnsemble(forecast_horizon=4)
ensemble.train(features_df)
forecasts = ensemble.generate_forecasts(features_df, cutoff_date="2024-11-02")
```

---

## Files

```
├── direct_forecast.py      # Main forecasting code
├── feature_engineering.py  # 59 feature creation
├── config.py               # Hyperparameters
└── outputs/
    ├── forecasts_improved_direct.csv
    └── forecast_plots/*_improved.png
```

---

*Full documentation: `XGBOOST_MODEL_DOCUMENTATION.md`*
