# Feature Engineering Explanation for Flu Forecasting

## 🧠 Overview: Why Feature Engineering Matters

Feature engineering transforms raw data into meaningful predictors that help the XGBoost model understand patterns in flu hospitalizations. The goal is to capture:
- **Temporal patterns** (seasonality, trends)
- **Historical context** (recent activity, momentum)
- **Spatial context** (national vs. local patterns)
- **Interactions** between different factors

## 📊 Feature Categories

### 1. **Temporal Features** (Seasonal Patterns)

**Purpose**: Capture seasonal flu patterns and cyclical behavior

**Features Created**:
- `year`: Year (2022, 2023, 2024)
- `month`: Month (1-12)
- `week_of_year`: Week number (1-52/53)
- `day_of_year`: Day of year (1-365/366)

**Cyclical Encoding** (Most Important):
- `week_sin` / `week_cos`: Converts week of year to sine/cosine
- `month_sin` / `month_cos`: Converts month to sine/cosine  
- `day_sin` / `day_cos`: Converts day of year to sine/cosine

**Why Cyclical Encoding?**
- Raw month values (1, 2, 3, ..., 12) suggest December (12) is far from January (1)
- But December and January are actually close in time
- Sine/cosine encoding makes December and January close in value
- Helps model understand that flu season typically peaks in winter months

**Example**:
```
Month 1 (January):  month_sin = 0.5,  month_cos = 0.87
Month 12 (December): month_sin = -0.5, month_cos = 0.87
```
Both have similar cosine values, showing they're close in the seasonal cycle.

### 2. **Lag Features** (Recent History)

**Purpose**: Capture recent flu activity that influences future predictions

**Features Created**:
- `value_lag_1`: Flu hospitalizations 1 week ago
- `value_lag_2`: Flu hospitalizations 2 weeks ago
- `value_lag_3`: Flu hospitalizations 3 weeks ago
- `value_lag_4`: Flu hospitalizations 4 weeks ago
- `value_lag_8`: Flu hospitalizations 8 weeks ago
- `value_lag_12`: Flu hospitalizations 12 weeks ago

**Why These Lags?**
- **1-4 weeks**: Recent activity (most predictive)
- **8 weeks**: Medium-term memory (2 months ago)
- **12 weeks**: Seasonal memory (3 months ago, different season)

**Example**:
```
If current week is 2024-11-02:
- value_lag_1 = flu hospitalizations on 2024-10-26
- value_lag_2 = flu hospitalizations on 2024-10-19
- value_lag_4 = flu hospitalizations on 2024-10-05
```

### 3. **Rolling Features** (Trends and Variability)

**Purpose**: Capture trends, averages, and variability over time windows

**Features Created**:
- `value_rolling_mean_4`: Average flu activity over last 4 weeks
- `value_rolling_mean_8`: Average flu activity over last 8 weeks
- `value_rolling_std_4`: Variability in flu activity over last 4 weeks
- `value_rolling_std_8`: Variability in flu activity over last 8 weeks
- `value_rolling_min_4`: Minimum flu activity over last 4 weeks
- `value_rolling_max_4`: Maximum flu activity over last 4 weeks
- `value_rolling_min_8`: Minimum flu activity over last 8 weeks
- `value_rolling_max_8`: Maximum flu activity over last 8 weeks
- `value_trend_4w`: Current value minus 4-week average

**Why These Windows?**
- **4 weeks**: Short-term trend (1 month)
- **8 weeks**: Medium-term trend (2 months)

**Example**:
```
If current week has 20 hospitalizations:
- value_rolling_mean_4 = (15 + 18 + 22 + 20) / 4 = 18.75
- value_trend_4w = 20 - 18.75 = 1.25 (above average)
```

### 4. **US-Level Features** (National Context)

**Purpose**: Provide national context that influences local predictions

**Features Created**:
- `us_total_lag_1`: US total flu hospitalizations 1 week ago
- `us_total_lag_2`: US total flu hospitalizations 2 weeks ago
- `us_total_lag_3`: US total flu hospitalizations 3 weeks ago
- `us_total_lag_4`: US total flu hospitalizations 4 weeks ago
- `us_total_rolling_mean_4`: US average over last 4 weeks
- `us_total_rolling_mean_8`: US average over last 8 weeks
- `us_total_rolling_std_4`: US variability over last 4 weeks
- `us_total_rolling_std_8`: US variability over last 8 weeks

**Why National Context?**
- Flu spreads across states
- National trends influence local predictions
- Provides additional signal when local data is sparse

**Example**:
```
If California has 100 hospitalizations:
- us_total_lag_1 = 5000 (US total last week)
- us_total_rolling_mean_4 = 4800 (US average over 4 weeks)
- This shows if national flu activity is rising or falling
```

### 5. **Interaction Features** (Combined Effects)

**Purpose**: Capture interactions between different factors

**Features Created**:
- `recent_vs_historical`: 4-week average / 8-week average
- Location × seasonal patterns (handled by XGBoost)

**Why Interactions?**
- Some states have different seasonal patterns
- Recent vs. historical comparison shows momentum
- Helps model understand state-specific behaviors

**Example**:
```
If 4-week average = 20 and 8-week average = 15:
- recent_vs_historical = 20/15 = 1.33
- This means recent activity is 33% higher than historical average
- Suggests increasing flu activity
```

## 🔧 How Features Work Together

### **Feature Importance Hierarchy**:
1. **Lag features** (most important): Recent activity is the strongest predictor
2. **Rolling features**: Trends and momentum
3. **Temporal features**: Seasonal patterns
4. **US features**: National context
5. **Interaction features**: Combined effects

### **Feature Engineering Process**:
1. **Temporal**: Extract seasonal patterns
2. **Lag**: Add recent history
3. **Rolling**: Add trends and variability
4. **US**: Add national context
5. **Interactions**: Combine different signals
6. **Missing values**: Handle gaps in data

## 📈 Practical Examples

### **Example 1: High Flu Activity Prediction**
```
Current week: 2024-11-02
- value_lag_1 = 25 (high last week)
- value_rolling_mean_4 = 22 (above average trend)
- month = 11 (November, flu season)
- us_total_lag_1 = 5000 (high national activity)
→ Model predicts: High flu activity (25+ hospitalizations)
```

### **Example 2: Low Flu Activity Prediction**
```
Current week: 2024-07-15
- value_lag_1 = 5 (low last week)
- value_rolling_mean_4 = 6 (low trend)
- month = 7 (July, off-season)
- us_total_lag_1 = 1000 (low national activity)
→ Model predicts: Low flu activity (5-10 hospitalizations)
```

### **Example 3: Rising Trend Prediction**
```
Current week: 2024-10-15
- value_lag_1 = 15 (moderate last week)
- value_trend_4w = 3 (above 4-week average)
- recent_vs_historical = 1.2 (20% above historical)
- month = 10 (October, flu season starting)
→ Model predicts: Increasing flu activity (15-20 hospitalizations)
```

## 🎯 Why This Feature Engineering Works

### **Captures Multiple Time Scales**:
- **Short-term**: 1-4 week lags and rolling windows
- **Medium-term**: 8-12 week lags and rolling windows
- **Long-term**: Seasonal patterns and yearly cycles

### **Handles Missing Data**:
- Forward filling for lag features
- Rolling windows with minimum periods
- Graceful handling of sparse data

### **Provides Context**:
- Local history (lag and rolling features)
- National context (US features)
- Seasonal context (temporal features)
- Combined effects (interaction features)

### **Works with XGBoost**:
- XGBoost can learn complex interactions between features
- Handles both numerical and categorical features
- Robust to missing values and outliers
- Can identify which features are most important

## 🔍 Feature Selection and Importance

The model automatically learns which features are most important:
- **High importance**: Recent activity (lag_1, lag_2)
- **Medium importance**: Trends (rolling features)
- **Low importance**: Long-term lags, some temporal features

This helps identify what drives flu predictions most effectively.

## 📊 Summary

The feature engineering creates a rich set of predictors that capture:
- **When**: Seasonal and temporal patterns
- **What**: Recent and historical flu activity
- **Where**: Local and national context
- **How**: Trends, variability, and interactions

This comprehensive feature set enables the XGBoost model to make accurate flu hospitalization predictions by understanding the complex patterns in the data.
