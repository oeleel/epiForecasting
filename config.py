"""
Configuration file for XGBoost flu forecasting pipeline
"""

# Data configuration
CDC_DATA_URL = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv"
DEFAULT_CUTOFF_DATE = "2024-11-02"
MIN_TRAINING_WEEKS = 10

# Feature engineering configuration
LAG_FEATURES = [1, 2, 3, 4, 8, 12, 52]  # Weeks to lag (52 = year-over-year)
ROLLING_WINDOWS = [4, 8]  # Rolling window sizes
US_LAG_FEATURES = [1, 2, 3, 4]  # US total lags

# Monotonic constraints for XGBoost (1 = increasing, -1 = decreasing, 0 = none)
# Applied to features where we expect a consistent directional relationship
MONOTONIC_FEATURES = {
    'value_lag_1': 1,
    'value_lag_2': 1,
    'value_rolling_mean_4': 1,
    'value_rolling_mean_8': 1,
    'us_total_lag_1': 1,
    'us_total_rolling_mean_4': 1,
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.049483671454615456,
    'n_estimators': 1275,
    'subsample': 0.9691511174148248,
    'colsample_bytree': 0.8844127158628735,
    'random_state': 42,
    'early_stopping_rounds': 50
}

# Validation configuration
VALIDATION_CUTOFFS = [
    "2024-09-01",
    "2024-10-01", 
    "2024-11-02"
]

# Forecasting configuration
FORECAST_HORIZON = 4  # weeks ahead to predict

# Optuna optimization configuration
OPTUNA_N_TRIALS = 50  # Default number of trials for hyperparameter optimization
OPTUNA_STUDY_NAME = None  # None = auto-generate based on timestamp

