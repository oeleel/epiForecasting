# Flu Hospitalization Forecasting Pipeline

An XGBoost-based time series forecasting pipeline for predicting influenza hospitalizations at the state level using CDC FluSight data.

## Overview

This pipeline provides a complete solution for forecasting flu hospitalizations with the following capabilities:

- **Data Management**: Automated data fetching from CDC FluSight repository with temporal cutoffs
- **Feature Engineering**: Comprehensive feature creation including lag features, rolling statistics, and temporal patterns
- **Model Training**: XGBoost-based models with walk-forward validation
- **Forecasting**: 4-week ahead predictions using recursive forecasting
- **Evaluation**: Multiple metrics and visualization tools
- **Pipeline Orchestration**: End-to-end workflow management

## Features

### Data Processing
- Fetches data from CDC FluSight repository (updated weekly)
- Applies temporal cutoffs to prevent data leakage
- Handles missing values and data validation
- Supports state-level and national-level forecasting

### Feature Engineering
- **Lag Features**: 1, 2, 3, 4, 8, and 12-week lags
- **Rolling Statistics**: 4 and 8-week rolling means, standard deviations, min/max
- **Temporal Features**: Cyclical encoding of week, month, and day of year
- **National Context**: US-level features for additional context
- **Trend Analysis**: Recent vs historical comparisons

### Model Architecture
- **XGBoost Regression**: Tree-based model for non-linear pattern capture
- **Recursive Forecasting**: Multi-step ahead predictions using previous predictions
- **State-Specific Learning**: Unified model with location as categorical feature
- **Walk-Forward Validation**: Multiple temporal splits for robust evaluation

### Forecasting Capabilities
- **4-Week Ahead Predictions**: Short to medium-term forecasting
- **Confidence Intervals**: Bootstrap-based uncertainty quantification
- **Multiple Locations**: State-level and national predictions
- **Weekly Updates**: Model retraining with new data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd epiForecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from pipeline import FluForecastingPipeline

# Initialize pipeline
pipeline = FluForecastingPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(
    cutoff_date="2024-11-02",
    locations=['US', '06', '12', '48'],  # US, CA, FL, TX
    retrain=True,
    generate_forecasts=True
)
```

### Command Line Usage

```bash
# Run full pipeline
python pipeline.py --cutoff-date 2024-11-02 --locations US 06 12 48

# Generate forecasts only (using existing model)
python pipeline.py --cutoff-date 2024-11-02 --no-retrain

# Generate forecasts with confidence intervals
python pipeline.py --cutoff-date 2024-11-02 --confidence

# Run evaluation only
python pipeline.py --cutoff-date 2024-11-02 --evaluate-only
```

### Jupyter Notebook Demo

```bash
jupyter notebook flu_forecasting_demo.ipynb
```

## Pipeline Components

### 1. Data Loading (`data_loader.py`)
- Fetches data from CDC FluSight repository
- Applies temporal cutoffs
- Validates data quality
- Provides data summaries

### 2. Feature Engineering (`feature_engineering.py`)
- Creates lag and rolling features
- Implements cyclical encoding for seasonality
- Handles missing values
- Generates interaction features

### 3. Model Training (`train.py`)
- Walk-forward validation
- Hyperparameter management
- Model selection
- Performance tracking

### 4. Forecasting (`predict.py`)
- Recursive prediction generation
- Confidence interval calculation
- Multiple location support
- Output formatting

### 5. Evaluation (`evaluate.py`)
- Multiple metrics (MAE, RMSE, MAPE, SMAPE)
- Horizon-specific analysis
- Location-specific analysis
- Visualization tools

### 6. Pipeline Orchestration (`pipeline.py`)
- End-to-end workflow management
- Command-line interface
- Result aggregation
- Report generation

## Configuration

Edit `config.py` to customize:

- **Data settings**: URL, cutoff dates, minimum training weeks
- **Feature engineering**: Lag periods, rolling windows
- **Model parameters**: XGBoost hyperparameters
- **Validation**: Cutoff dates for walk-forward validation
- **Forecasting**: Prediction horizon

## Output Files

The pipeline generates several output files:

- **Forecasts**: `forecasts_YYYY-MM-DD_YYYYMMDD_HHMMSS.csv`
- **Models**: `flu_model_YYYY-MM-DD.json` and metadata
- **Feature Importance**: `feature_importance_YYYY-MM-DD.csv`
- **Evaluation Results**: `evaluation_YYYY-MM-DD_YYYYMMDD_HHMMSS.json`
- **Pipeline Results**: `pipeline_results_YYYY-MM-DD_YYYYMMDD_HHMMSS.json`
- **Forecast Reports**: `forecasts_YYYY-MM-DD_YYYYMMDD_HHMMSS_report.md`

## Model Performance

The pipeline uses walk-forward validation with multiple temporal splits:

- **Training Periods**: Up to specified cutoff dates
- **Validation Periods**: Next 4 weeks after cutoff
- **Metrics**: MAE, RMSE, MAPE calculated per split
- **Aggregation**: Mean and standard deviation across splits

## Weekly Updates

To update the model with new data:

```python
# Update with new cutoff date
pipeline.update_with_new_data("2024-11-09")
```

Or via command line:

```bash
python pipeline.py --cutoff-date 2024-11-09
```

## Data Requirements

- **Source**: CDC FluSight repository
- **Format**: CSV with columns: date, location, location_name, value, weekly_rate
- **Frequency**: Weekly data
- **Coverage**: All US states and territories
- **Update Schedule**: Weekly (typically Saturdays)

## Limitations

- **Prediction Horizon**: Limited to 4 weeks ahead
- **Data Dependency**: Requires consistent weekly data updates
- **Location Coverage**: Limited to locations in CDC dataset
- **Seasonal Patterns**: Model learns from historical patterns only

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CDC FluSight team for providing the data
- XGBoost developers for the machine learning framework
- Contributors to the open-source data science ecosystem

## Support

For questions or issues:

1. Check the documentation
2. Review the example notebook
3. Open an issue on GitHub
4. Contact the development team

## Changelog

### Version 1.0.0
- Initial release
- XGBoost-based forecasting
- Walk-forward validation
- 4-week ahead predictions
- State-level forecasting
- Confidence intervals
- Complete pipeline orchestration