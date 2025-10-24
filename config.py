"""
Configuration file for XGBoost flu forecasting pipeline
"""

# Data configuration
CDC_DATA_URL = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv"
DEFAULT_CUTOFF_DATE = "2024-11-02"
MIN_TRAINING_WEEKS = 10

# Feature engineering configuration
LAG_FEATURES = [1, 2, 3, 4, 8, 12]  # Weeks to lag
ROLLING_WINDOWS = [4, 8]  # Rolling window sizes
US_LAG_FEATURES = [1, 2, 3, 4]  # US total lags

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
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

