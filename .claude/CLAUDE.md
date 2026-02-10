# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Epidemiological forecasting system for US influenza hospitalizations using XGBoost. Predicts 1-4 weeks ahead for all US states and territories using CDC FluSight data.

## Commands

### Setup
```bash
pip install -r documentation/requirements.txt
```

### Run Pipeline
```bash
# Full pipeline (train + forecast)
python pipeline.py --cutoff-date 2024-11-02

# Specific locations only
python pipeline.py --cutoff-date 2024-11-02 --locations US 06 12 48

# With confidence intervals
python pipeline.py --cutoff-date 2024-11-02 --confidence

# Evaluation only
python pipeline.py --cutoff-date 2024-11-02 --evaluate-only
```

### Export Model Inputs
```bash
python3 export_model_inputs.py --cutoff-date 2024-12-01
```

### Hyperparameter Optimization
```bash
python optimize.py --n-trials 50
```

## Architecture

### Direct Forecast Ensemble (Key Innovation)
The system uses 4 independent XGBoost models, one for each forecast horizon (1-4 weeks ahead). This avoids error accumulation from recursive forecasting where a single model iterates on its own predictions.

### Data Pipeline Flow
```
CDC API → FluDataLoader → FeatureEngineer (59 features) → DirectForecastEnsemble → Forecasts
```

### Core Modules
- `data_loader.py` - `FluDataLoader`: fetches CDC data, applies temporal cutoffs
- `feature_engineering.py` - `FeatureEngineer`: creates 59 engineered features (lags, rolling stats, temporal patterns, national context, YoY comparisons)
- `direct_forecast.py` - `DirectForecastEnsemble`: 4 horizon-specific XGBoost models
- `config.py` - Central configuration for hyperparameters and feature settings
- `pipeline.py` - `FluForecastingPipeline`: end-to-end orchestration with CLI

### Feature Categories (59 total)
- Lag features: 1-4 week, 8, 12, 52-week lags
- Rolling statistics: 4/8-week mean, std, min, max
- Temporal: Week/month cyclical encoding
- National context: US-level aggregates as features
- Year-over-year: Ratio, % change, difference
- Rate of change: WoW change, acceleration, momentum
- Season phase: Onset/peak/decline indicators

### Post-Prediction Constraints
Floor constraint ensures predictions can't drop below 30% of last known value to prevent unrealistic drops during severe seasons.

## Directory Structure

- `models/` - Saved XGBoost model artifacts (.json + metadata)
- `outputs/` - Forecasts, evaluation results, plots
- `data/processed_features/` - Exported feature datasets
- `documentation/` - Full model documentation and requirements

## Configuration

All hyperparameters are centralized in `config.py`:
- `XGBOOST_PARAMS` - Model hyperparameters (Optuna-tuned)
- `LAG_FEATURES`, `ROLLING_WINDOWS` - Feature engineering settings
- `VALIDATION_CUTOFFS` - Walk-forward validation dates
- `MONOTONIC_FEATURES` - Constraints for specific features

## Data Source

CDC FluSight Repository: weekly influenza hospitalization data fetched from `CDC_DATA_URL` in config.py.
