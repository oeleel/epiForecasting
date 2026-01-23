# Model Input Data

This directory contains the processed feature data that is fed into the XGBoost influenza forecasting model.

## Contents

- `processed_features/` - Processed feature datasets with all engineered features
- `feature_dictionary.csv` - Documentation of all features (name, category, description, type, unit)

## What is This Data?

The model input data consists of **54+ engineered features** created from raw CDC influenza hospitalization data. These features capture:

- **Historical patterns** (lag values, rolling statistics)
- **Temporal patterns** (seasonal cycles, time-of-year effects)
- **National context** (US-level trends)
- **Year-over-year comparisons** (how current season compares to last year)
- **Rate of change** (momentum, acceleration)
- **Season phase** (onset/peak/decline indicators)
- **Season severity** (how severe current season is vs historical)

These features transform raw hospitalization counts into rich signals that help the model make accurate 1-4 week forecasts.

## File Format

### Processed Features CSV

Each file in `processed_features/` contains:

- **Metadata columns:**
  - `date` - Week ending date (Saturday)
  - `location` - US state/territory code (e.g., "CA", "US")
  - `location_name` - Full location name (e.g., "California")
  - `value` - Weekly influenza hospitalizations (target variable)
  - `weekly_rate` - Per-capita hospitalization rate

- **Engineered features (54+):**
  - See `feature_dictionary.csv` for complete descriptions

### Metadata JSON

Each processed features file has an accompanying metadata JSON with:
- Export date and cutoff date
- Data summary statistics
- Feature categories and counts
- Configuration parameters
- Missing value information

## How to Use This Data

### 1. Inspect the Features

```python
import pandas as pd

# Load processed features
df = pd.read_csv('data/processed_features/model_inputs_2024-11-02.csv')

# View feature columns
print(df.columns)

# Check a specific location
ca_data = df[df['location'] == 'CA']
print(ca_data.head())
```

### 2. Understand the Features

```python
# Load feature dictionary
feature_dict = pd.read_csv('data/feature_dictionary.csv')

# View all features by category
print(feature_dict.groupby('category')['feature_name'].count())

# Look up a specific feature
lag_features = feature_dict[feature_dict['category'] == 'Lag Features']
print(lag_features)
```

### 3. Analyze Feature Distributions

```python
import matplotlib.pyplot as plt

# Example: Plot lag features over time for a state
ca_data = df[df['location'] == 'CA'].sort_values('date')

plt.figure(figsize=(12, 6))
plt.plot(ca_data['date'], ca_data['value_lag_1'], label='1 week ago')
plt.plot(ca_data['date'], ca_data['value_lag_4'], label='4 weeks ago')
plt.plot(ca_data['date'], ca_data['value_lag_52'], label='52 weeks ago (YoY)')
plt.legend()
plt.title('California: Lag Features Over Time')
plt.show()
```

## Regenerating the Data

To create a new dataset with a different cutoff date:

```bash
# Export features with default cutoff (2024-11-02)
python3 export_model_inputs.py

# Export features with custom cutoff
python3 export_model_inputs.py --cutoff-date 2024-12-01

# Export to custom directory
python3 export_model_inputs.py --cutoff-date 2024-12-01 --output data/processed_features/

# Only generate feature dictionary (no data export)
python3 export_model_inputs.py --feature-dict-only
```

### Command-line Options

- `--cutoff-date YYYY-MM-DD` - Data cutoff date (default: 2024-11-02)
- `--output DIR` - Output directory (default: data/processed_features)
- `--no-metadata` - Skip metadata JSON generation
- `--feature-dict-only` - Only generate feature dictionary

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| **Lag Features** | 7 | Historical values from 1-52 weeks ago |
| **Rolling Statistics** | 9 | Moving averages, std dev, min/max over 4-8 week windows |
| **Temporal Features** | 10 | Seasonal cycles (week, month, day-of-year encodings) |
| **US National Context** | 8 | National-level lags and rolling statistics |
| **Year-over-Year** | 3 | Comparisons to same week last year |
| **Rate of Change** | 5 | Week-over-week changes, momentum, acceleration |
| **Season Phase** | 4 | Flu season timing indicators (onset/peak/decline) |
| **Season Severity** | 2 | Current season severity vs historical |
| **Rate Features** | 4 | Per-capita rate-based features |
| **Interaction Features** | 2 | Combined signals (recent vs historical) |

## Data Quality Notes

### Missing Values

- Early records may have missing lag features (e.g., `value_lag_52` requires 52 weeks of history)
- Missing values are forward-filled within each location
- Final NaN values are filled with 0

### Temporal Cutoff

All data respects the specified cutoff date to prevent data leakage. The model only "sees" data that would have been available at the cutoff date.

### Locations

Data includes:
- 50 US states
- District of Columbia
- US territories (e.g., Puerto Rico, US Virgin Islands)
- US national total (location code: "US")

## Related Documentation

- `../documentation/XGBOOST_MODEL_DOCUMENTATION.md` - Full model documentation
- `../documentation/MODEL_QUICK_REFERENCE.md` - Quick reference guide
- `../feature_engineering.py` - Source code for feature creation
- `feature_dictionary.csv` - Complete feature descriptions

## Data Source

Raw data is sourced from the CDC FluSight forecast hub:
- URL: https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv
- Updated: Weekly
- Coverage: All US states and territories

## Questions?

If you have questions about the data or features, please refer to:
1. `feature_dictionary.csv` for feature definitions
2. `metadata_*.json` files for data statistics
3. The model documentation in `../documentation/`

## Version History

- **v1.0** (Jan 2025) - Initial export functionality with 54+ features
  - Comprehensive feature dictionary
  - Metadata generation
  - Command-line interface for easy regeneration
