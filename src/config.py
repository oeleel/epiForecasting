"""
Configuration file for XGBoost flu forecasting pipeline
"""

# Data configuration
# Data is fetched from the FluSight GitHub repository and cached locally.
# Use `python -m src.data_loader update` to fetch the latest data.
# Repository: https://github.com/cdcepi/FluSight-forecast-hub
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

# XGBoost hyperparameters - Version 1 (original, overfitting)
# Train MAE ~7-14, Val MAE ~273-339 (40x gap)
XGBOOST_PARAMS_V1 = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.049483671454615456,
    'n_estimators': 1275,
    'subsample': 0.9691511174148248,
    'colsample_bytree': 0.8844127158628735,
    'random_state': 42,
    'early_stopping_rounds': 50
}

# XGBoost hyperparameters - Version 2 (regularized to reduce overfitting)
# Reduced max_depth, increased regularization (alpha, lambda, gamma)
# Reduced subsample and colsample for more conservative trees
XGBOOST_PARAMS_V2 = {
    'objective': 'reg:squarederror',
    'max_depth': 3,                    # Reduced from 5 to limit tree complexity
    'learning_rate': 0.05,             # Similar to V1
    'n_estimators': 800,               # Reduced from 1275
    'subsample': 0.7,                  # Reduced from 0.97 for regularization
    'colsample_bytree': 0.6,           # Reduced from 0.88 for regularization
    'min_child_weight': 30,            # Added: requires more samples per leaf
    'reg_alpha': 1.0,                  # Added: L1 regularization
    'reg_lambda': 5.0,                 # Added: L2 regularization
    'gamma': 1.0,                      # Added: minimum loss reduction for split
    'random_state': 42,
    'early_stopping_rounds': 50
}

# Active hyperparameters (switch between versions here)
XGBOOST_PARAMS = XGBOOST_PARAMS_V2

# Validation configuration - Legacy (3 dates, single season onset)
VALIDATION_CUTOFFS_LEGACY = [
    "2024-09-01",
    "2024-10-01",
    "2024-11-02"
]

# Expanded validation cutoffs covering multiple seasons and phases
# Generates monthly cutoffs from October through March for each flu season
def generate_expanded_cutoffs():
    """Generate validation cutoffs covering multiple seasons and phases."""
    cutoffs = []
    # Flu seasons available in data: 2022-23, 2023-24, 2024-25
    seasons = [
        (2022, 2023),  # Season 2022-23
        (2023, 2024),  # Season 2023-24
        (2024, 2025),  # Season 2024-25 (partial)
    ]

    for start_year, end_year in seasons:
        # Onset phase: October, November (weeks 40-48)
        cutoffs.append(f"{start_year}-10-15")  # Mid-October
        cutoffs.append(f"{start_year}-11-15")  # Mid-November

        # Peak phase: December, January (weeks 49-52, 1-4)
        cutoffs.append(f"{start_year}-12-15")  # Mid-December
        cutoffs.append(f"{end_year}-01-15")    # Mid-January

        # Decline phase: February, March (weeks 5-16)
        cutoffs.append(f"{end_year}-02-15")    # Mid-February
        cutoffs.append(f"{end_year}-03-15")    # Mid-March

    return cutoffs

VALIDATION_CUTOFFS_EXPANDED = generate_expanded_cutoffs()

# Active validation cutoffs (use expanded by default)
VALIDATION_CUTOFFS = VALIDATION_CUTOFFS_EXPANDED

# Forecasting configuration
FORECAST_HORIZON = 4  # weeks ahead to predict

# Target transformation configuration
# Options: "raw" (direct hospitalization counts), "ratio" (value(t+h)/value(t)), "log" (log1p transform)
# Log mode shows best performance (55% MAPE vs 59% baseline, 65% raw, 87% ratio)
TARGET_MODE = "log"

# Ratio mode configuration
RATIO_CLIP_MIN = 0.05  # Minimum ratio (5% of current value)
RATIO_CLIP_MAX = 10.0  # Maximum ratio (10x current value)
RATIO_MIN_DENOMINATOR = 5  # If current value < this, use raw prediction instead

# Optuna optimization configuration
OPTUNA_N_TRIALS = 100  # Number of trials for hyperparameter optimization
OPTUNA_STUDY_NAME = None  # None = auto-generate based on timestamp
OPTUNA_USE_PRUNING = True  # Use MedianPruner for early stopping of poor trials

# XGBoost hyperparameters - Version 3 (Optuna optimized with expanded validation)
# Will be populated after running Optuna optimization
XGBOOST_PARAMS_V3 = None  # Placeholder - set after optimization

# Feature engineering version tracking
# v2: Added trend features, removed low-importance features
# v3: Further pruned based on updated SHAP analysis
FEATURE_VERSION = "v3"

# Features removed based on initial SHAP analysis (< 1% of top feature)
FEATURES_REMOVED_V2 = [
    'is_flu_season',       # Max SHAP 0.065, avg rank 55.2
    'season_phase_cos',    # Max SHAP 0.085, avg rank 55.0
    'month_sin',           # Max SHAP 0.098, avg rank 53.5
    'season_phase',        # Max SHAP 0.149, avg rank 52.2
    'yoy_pct_change',      # Max SHAP 0.190, avg rank 52.5
    'seasonal_deviation',  # Max SHAP 0.268, avg rank 50.5 (interaction feature)
    'yoy_ratio',           # Max SHAP 0.374, avg rank 48.2
]

# Additional features removed in v3 based on updated SHAP analysis (Feb 2026)
# These v2-added features showed low/zero importance
FEATURES_REMOVED_V3 = [
    'surge_indicator',              # Zero SHAP across all horizons, avg rank 57.0
    'consecutive_increase_weeks',   # Very low importance, avg rank 52.75
    'consecutive_decrease_weeks',   # Very low importance, avg rank 49.0
    'linear_trend_slope_3w',        # Redundant with 4w version, avg rank 37.5
]

# Combined list for current version
FEATURES_REMOVED = FEATURES_REMOVED_V2 + FEATURES_REMOVED_V3

# New trend-capturing features added in v2 (kept in v3)
FEATURES_ADDED = [
    'linear_trend_slope_4w',         # Slope of 4-week linear fit - KEEP (avg rank 10.0)
    'value_to_rolling_mean_ratio',   # Current vs rolling mean - KEEP (avg rank 6.25, top performer!)
    'historical_percentile',         # Where current value sits - KEEP (avg rank 16.0)
    'ratio_to_historical_max',       # Current / historical max - KEEP (avg rank 25.25)
]

# Features that were in v2 FEATURES_ADDED but removed in v3
FEATURES_REMOVED_FROM_V2_ADDED = [
    'linear_trend_slope_3w',        # Redundant with 4w
    'consecutive_increase_weeks',   # Low importance
    'consecutive_decrease_weeks',   # Low importance
    'surge_indicator',              # Zero importance
]

# Quantile regression configuration
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]  # Prediction interval quantiles
ENABLE_QUANTILE_FORECASTS = True  # Toggle quantile forecasting on/off

# Location clustering configuration
ENABLE_LOCATION_CLUSTERING = True  # Toggle cluster-specific models on/off
N_LOCATION_CLUSTERS = 5  # Number of clusters for location grouping
MIN_CLUSTER_SIZE = 3  # Minimum locations per cluster (smaller clusters get merged)

