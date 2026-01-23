# XGBoost Model Documentation

note: add shap dependency plots for marginal improvement quantified per parameter

1/23:

* [ ] create visualizations using matplotlib, or something similar, not plotly because too complicated. A function that you give a location and date (optional) and it takes those and returns a visualization of the forecast for all points in the date range for 1-4 weeks forecast

- [X] create a way to store the input data in the repository, to review with aniruddha
- [ ] Visual inspection of forecast (e.g. week by week during the increases, make sure it is not flat a substantial way into the increase)
- [ ] for data split, retrain for the recent weeks data, organize the data splitting (may not be beneficial to have 80/20 split, play with ratio and training vs. testing data)
- [ ] find way to finetune/retune model based on most recent data
  - [ ] could also retrain model, look into this
- [ ] in the future, probabilistic forecasts, if time look into
- [ ] there is also a vector-to-vector mapping for xgboost, outputs all 4 weeks forecasts at once
  - [ ] three different model types
    - [ ] recursive
    - [ ] vector-to-vector
    - [ ] direct forecasting (independent models per week)
  - [ ] agentic idea for parameter choosing and optimization with neurosymbolic ai, making an ai system for forecasting with foundation models
    - [ ] TimeGPT is an example of a foundation model
    - [ ] agentic ai will find the rules that the neurosymbolic ai will need to follow, will be useful for finding different rules between fields (finance, influenza, etc)

## Flu Hospitalization Forecasting Model

**Last Updated:** January 2025
**Model Version:** 2.0 (Direct Forecast Ensemble)
**Training Cutoff:** 2024-11-02

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [Model Architecture](#model-architecture)
3. [Hyperparameters](#hyperparameters)
4. [Feature Engineering](#feature-engineering)
5. [Training Process](#training-process)
6. [Prediction Methodology](#prediction-methodology)
7. [Model Performance](#model-performance)
8. [Post-Prediction Constraints](#post-prediction-constraints)
9. [Model Limitations](#model-limitations)
10. [Usage Guidelines](#usage-guidelines)

---

## Model Overview

### Purpose

The XGBoost model is designed to forecast influenza hospitalization counts at the state and national level for 1-4 weeks ahead. The model uses historical hospitalization data, temporal patterns, and spatial context to generate accurate short-term forecasts.

### Model Type

- **Algorithm:** XGBoost (eXtreme Gradient Boosting)
- **Architecture:** Direct Forecast Ensemble (4 separate models)
- **Task:** Regression (predicting continuous hospitalization counts)
- **Objective Function:** Squared Error (`reg:squarederror`)
- **Ensemble Method:** Gradient Boosting with Decision Trees

### Key Characteristics

- **Multi-location:** Unified models for all US states and territories
- **Multi-horizon:** Separate model trained for each forecast horizon (1-4 weeks)
- **Direct Forecasting:** Avoids error accumulation from recursive methods
- **Time-aware:** Incorporates temporal patterns, seasonality, and year-over-year comparisons
- **Spatial-aware:** Uses national-level context for state predictions
- **Floor Constraint:** Prevents extreme under-predictions during anomalous seasons

---

## Model Architecture

### Direct Forecast Ensemble

Unlike traditional recursive forecasting (which accumulates errors), this model uses a **Direct Forecast Ensemble** approach:

```
┌─────────────────────────────────────────────────────────────┐
│                 Direct Forecast Ensemble                     │
├─────────────────────────────────────────────────────────────┤
│  Model 1 (Horizon 1)  →  Predicts 1 week ahead              │
│  Model 2 (Horizon 2)  →  Predicts 2 weeks ahead             │
│  Model 3 (Horizon 3)  →  Predicts 3 weeks ahead             │
│  Model 4 (Horizon 4)  →  Predicts 4 weeks ahead             │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**

- No error accumulation between horizons
- Each model optimized for its specific forecast horizon
- Longer horizon models can learn different patterns than short-term models

**XGBoost Configuration (per model):**

- **Base Learners:** Decision trees (regression trees)
- **Ensemble Size:** Up to 1,275 trees (with early stopping)
- **Tree Depth:** Maximum depth of 5 levels
- **Learning Strategy:** Gradient descent optimization

### Model Components

1. **Feature Preprocessing**

   - Categorical encoding: Location names encoded using LabelEncoder
   - Missing value handling: Forward fill for lag features, zero fill for others
   - Feature scaling: Not required (tree-based models are scale-invariant)
2. **Training Process**

   - Each horizon model trained with shifted targets
   - Validation-based early stopping (50 rounds patience)
   - 80/20 train-validation split on historical data
3. **Prediction Process**

   - Direct prediction using horizon-specific model
   - Floor constraint applied post-prediction
   - Location-specific predictions using categorical features

---

## Hyperparameters

### Current Configuration

The model uses the following hyperparameters, optimized via Optuna:

```python
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',           # Regression with squared error loss
    'max_depth': 5,                            # Maximum tree depth
    'learning_rate': 0.049483671454615456,     # Shrinkage rate (eta)
    'n_estimators': 1275,                      # Maximum number of trees
    'subsample': 0.9691511174148248,           # Row sampling ratio
    'colsample_bytree': 0.8844127158628735,    # Column sampling ratio per tree
    'random_state': 42,                        # Random seed for reproducibility
    'early_stopping_rounds': 50                # Early stopping patience
}
```

### Monotonic Constraints

Applied to enforce sensible relationships between features and predictions:

```python
MONOTONIC_FEATURES = {
    'value_lag_1': 1,              # Higher recent values → higher predictions
    'value_lag_2': 1,
    'value_rolling_mean_4': 1,     # Higher recent average → higher predictions
    'value_rolling_mean_8': 1,
    'us_total_lag_1': 1,           # Higher national values → higher predictions
    'us_total_rolling_mean_4': 1,
}
```

### Hyperparameter Details

| Parameter                 | Value | Purpose                                                  |
| ------------------------- | ----- | -------------------------------------------------------- |
| `max_depth`             | 5     | Controls tree complexity; balances fit vs generalization |
| `learning_rate`         | 0.049 | Small steps = more stable training, requires more trees  |
| `n_estimators`          | 1275  | Max trees; early stopping typically stops before         |
| `subsample`             | 0.969 | 96.9% of rows per tree; reduces overfitting              |
| `colsample_bytree`      | 0.884 | 88.4% of features per tree; adds diversity               |
| `early_stopping_rounds` | 50    | Stops if no improvement for 50 rounds                    |

---

## Feature Engineering

### Feature Summary

The model uses **59 features** across 9 categories:

| Category            | Features | Purpose                                              |
| ------------------- | -------- | ---------------------------------------------------- |
| Temporal            | 11       | Seasonal patterns (week/month/day cyclical encoding) |
| Lag                 | 7        | Recent historical values (1-52 week lags)            |
| Rolling Statistics  | 9        | Trends and variability over 4/8 week windows         |
| US National Context | 8        | National-level patterns for state predictions        |
| Year-over-Year      | 3        | Comparison to same period last year                  |
| Rate of Change      | 5        | Momentum and acceleration metrics                    |
| Season Phase        | 4        | Flu season stage indicators                          |
| Rate-based          | 4        | Per-capita rate features                             |
| Interaction         | 2        | Cross-feature relationships                          |

### Feature Categories Detail

#### 1. Temporal Features (11 features)

```
year, month, week_of_year, day_of_year
week_sin, week_cos, month_sin, month_cos, day_sin, day_cos
```

**Purpose:** Capture seasonal flu patterns; cyclical encoding ensures December and January are "close" in the feature space.

#### 2. Lag Features (7 features)

```
value_lag_1, value_lag_2, value_lag_3, value_lag_4
value_lag_8, value_lag_12, value_lag_52
```

**Purpose:** Recent activity is most predictive; 52-week lag enables year-over-year comparisons.

#### 3. Rolling Statistics (9 features)

```
value_rolling_mean_4, value_rolling_mean_8
value_rolling_std_4, value_rolling_std_8
value_rolling_min_4, value_rolling_min_8
value_rolling_max_4, value_rolling_max_8
value_trend_4w
```

**Purpose:** Capture trends, volatility, and recent range of activity.

#### 4. US National Context (8 features)

```
us_total_lag_1, us_total_lag_2, us_total_lag_3, us_total_lag_4
us_total_rolling_mean_4, us_total_rolling_mean_8
us_total_rolling_std_4, us_total_rolling_std_8
```

**Purpose:** States often follow national trends; provides context when state data is sparse.

#### 5. Year-over-Year Features (3 features)

```
yoy_ratio, yoy_pct_change, yoy_diff
```

**Purpose:** Compare current values to same week last year; helps identify unusual season severity.

#### 6. Rate of Change Features (5 features)

```
wow_change, wow_pct_change, acceleration
momentum_4w, momentum_4w_pct
```

**Purpose:** Capture velocity and acceleration of hospitalization trends.

#### 7. Season Phase Features (4 features)

```
is_flu_season, season_phase
season_phase_sin, season_phase_cos
```

**Purpose:** Encode flu season stage (onset/peak/decline/off-season).

#### 8. Season Severity Features (2 features)

```
season_severity_ratio, recent_severity_ratio
```

**Purpose:** Compare current season cumulative activity to previous year; helps adjust for unprecedented seasons.

#### 9. Interaction & Other Features (4 features)

```
location_name (encoded), weekly_rate
recent_vs_historical, seasonal_deviation
```

**Purpose:** Location-specific learning and normalized population metrics.

---

## Training Process

### Data Preparation

1. **Data Source:** CDC FluSight repository (`target-hospital-admissions.csv`)
2. **Temporal Cutoff:** All data up to cutoff date (prevents data leakage)
3. **Target Shifting:** For horizon h, target = value at time t+h

### Training Procedure

For each horizon (1-4 weeks):

1. **Prepare Data**

   - Shift target column by horizon weeks
   - Remove rows where shifted target is NaN
2. **Split Data**

   - Training: First 80% of data (by time)
   - Validation: Last 20% of data
3. **Train Model**

   - Fit XGBoost with early stopping
   - Monitor validation MAE
   - Stop if no improvement for 50 rounds

### Training Results (Current Model)

| Horizon | Train MAE | Val MAE |
| ------- | --------- | ------- |
| Week 1  | 6.84      | 272.80  |
| Week 2  | 8.76      | 294.77  |
| Week 3  | 12.48     | 323.01  |
| Week 4  | 13.99     | 338.67  |

---

## Prediction Methodology

### Direct Forecasting Process

```
Input: Historical data up to cutoff_date

For each location:
    1. Get last available data row
    2. Prepare features from last row
  
    For each horizon (1, 2, 3, 4):
        3. Use horizon-specific model to predict
        4. Apply floor constraint (if enabled)
        5. Ensure prediction ≥ 0
        6. Store forecast
    
Output: DataFrame with forecasts for all locations and horizons
```

### Key Differences from Recursive Forecasting

| Aspect                | Recursive             | Direct (Current)        |
| --------------------- | --------------------- | ----------------------- |
| Models                | 1 model, iterated     | 4 separate models       |
| Error propagation     | Errors compound       | Independent per horizon |
| Feature updates       | Predictions feed back | No feedback needed      |
| Computational cost    | Lower                 | Higher (4x training)    |
| Long-horizon accuracy | Degrades rapidly      | More stable             |

---

## Model Performance

### Overall Metrics (Nov 2024 - Apr 2025 Evaluation)

| Metric         | Value                   |
| -------------- | ----------------------- |
| **MAPE** | 59.03%                  |
| **MAE**  | 515.04 hospitalizations |
| **MSE**  | 11,664,738              |
| **RMSE** | 3,415.66                |

### Performance by Horizon

| Horizon | MAE    | MAPE   |
| ------- | ------ | ------ |
| Week 1  | 399.66 | 51.37% |
| Week 2  | 504.08 | 51.06% |
| Week 3  | 557.81 | 63.79% |
| Week 4  | 598.58 | 70.90% |

### Performance Context

The 2024-2025 flu season was **unusually severe** (~2-3x higher hospitalizations than previous seasons), which poses challenges for any model trained on historical data. Key observations:

- Week 1 predictions are most accurate
- Accuracy degrades with horizon (expected)
- Model under-predicts during peak season due to unprecedented severity
- Floor constraint helps prevent extreme under-predictions during decline

---

## Post-Prediction Constraints

### Floor Constraint

To prevent unrealistically low predictions (especially during anomalous seasons), a floor constraint is applied:

```python
# Floor decays with horizon
horizon_floor_ratio = 0.3 * (1 - (horizon - 1) * 0.05)
# Week 1: 30%, Week 2: 28.5%, Week 3: 27%, Week 4: 25.5%

floor_value = previous_value * horizon_floor_ratio
prediction = max(prediction, floor_value)
```

**Purpose:** Ensures predictions don't drop more than ~70-75% from the previous known value, which would be unrealistic for a single week's change.

**Impact:** Improved MAPE from ~60% to ~59% by preventing extreme under-predictions during the severe 2024-2025 season.

---

## Model Limitations

### 1. Unprecedented Seasons

- Model trained on historical patterns
- Cannot predict "black swan" events that exceed all training data
- 2024-2025 season was ~2-3x more severe than any in training data

### 2. No External Data

- Relies solely on hospitalization history
- Does not include: weather, vaccination rates, mobility data, viral surveillance

### 3. Location Generalization

- Single set of models for all locations
- May not capture location-specific nuances perfectly

### 4. Forecast Horizon

- Limited to 4 weeks ahead
- Longer horizons would require different approach

### 5. Real-time Constraints

- Requires recent data to be available
- Data reporting lags may affect real-world performance

---

## Usage Guidelines

### Running Forecasts

```python
from data_loader import FluDataLoader
from feature_engineering import FeatureEngineer
from direct_forecast import DirectForecastEnsemble

# Load and prepare data
loader = FluDataLoader()
data = loader.load_and_preprocess("2024-11-02")

engineer = FeatureEngineer()
features_df = engineer.create_all_features(data)

# Train models
ensemble = DirectForecastEnsemble(forecast_horizon=4)
ensemble.train(features_df, validation_split=0.2)

# Generate forecasts
forecasts = ensemble.generate_forecasts(
    data=features_df,
    cutoff_date="2024-11-02",
    use_floor_constraint=True,
    floor_ratio=0.3
)
```

### Best Practices

1. **Regular Retraining:** Retrain weekly with new data
2. **Monitor Performance:** Track forecast errors over time
3. **Use Confidence Intervals:** Account for uncertainty in predictions
4. **Compare to Baselines:** Check against naive forecasts (last week's value)

### File Structure

```
epiForecasting/
├── config.py                    # Hyperparameters and settings
├── data_loader.py               # CDC data loading
├── feature_engineering.py       # All 59 features
├── model.py                     # XGBoost wrapper
├── direct_forecast.py           # Direct ensemble implementation
├── train.py                     # Training utilities
├── predict.py                   # Recursive forecasting (legacy)
├── evaluate.py                  # Evaluation metrics
├── models/                      # Saved models
└── outputs/                     # Forecasts and plots
```

---

## Technical Specifications

### Dependencies

- **XGBoost:** >= 1.7.0
- **Pandas:** >= 1.5.0
- **NumPy:** >= 1.21.0
- **Scikit-learn:** >= 1.1.0

### Model Files

```
models/
├── flu_model_enhanced_2024-11-02.json           # Enhanced model
└── flu_model_enhanced_2024-11-02_metadata.json  # Model metadata
```

---

## Summary

| Aspect                     | Details                                                |
| -------------------------- | ------------------------------------------------------ |
| **Architecture**     | Direct Forecast Ensemble (4 XGBoost models)            |
| **Features**         | 59 features across 9 categories                        |
| **Training Data**    | Historical flu hospitalizations through Nov 2024       |
| **Forecast Horizon** | 1-4 weeks ahead                                        |
| **Key Innovation**   | Direct forecasting avoids recursive error accumulation |
| **Performance**      | MAPE ~59%, MAE ~515 (on severe 2024-2025 season)       |

---

**Document Version:** 2.0
**Last Updated:** January 2025
**Maintained By:** Flu Forecasting Team
