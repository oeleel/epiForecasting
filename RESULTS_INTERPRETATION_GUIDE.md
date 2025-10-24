# Flu Forecasting Results Interpretation Guide

## 📊 Understanding Your Forecast Results

This guide explains how to interpret the outputs from the flu hospitalization forecasting pipeline.

## 🗂️ Output Files Generated

The pipeline creates several files in the `outputs/` directory:

### 1. **Forecast Files** (`forecasts_YYYY-MM-DD_YYYYMMDD_HHMMSS.csv`)
**What it contains:** Your actual predictions for flu hospitalizations
**Key columns:**
- `date`: The date you're predicting for
- `location`: State/territory code (e.g., "US", "CA", "TX")
- `location_name`: Full location name (e.g., "California", "Texas")
- `prediction_horizon`: How many weeks ahead (1, 2, 3, or 4)
- `predicted_value`: **Your forecast number** (hospitalizations per 100,000 people)
- `pred_lower_95` / `pred_upper_95`: 95% confidence interval bounds (if available)

### 2. **Feature Importance** (`feature_importance_YYYY-MM-DD.csv`)
**What it contains:** Which factors the model considers most important
**Key columns:**
- `feature`: Name of the feature (e.g., "lag_1_weekly_rate", "rolling_mean_4w")
- `importance`: How important this feature is (higher = more important)

### 3. **Evaluation Results** (`evaluation_YYYY-MM-DD_YYYYMMDD_HHMMSS.json`)
**What it contains:** How well the model performed on historical data
**Key metrics:**
- `mae`: Mean Absolute Error (lower is better)
- `rmse`: Root Mean Square Error (lower is better)
- `mape`: Mean Absolute Percentage Error (lower is better)
- `r2`: R-squared (closer to 1.0 is better)

## 📈 How to Read Your Forecast Numbers

### **Your Forecast Numbers Mean:**
- **`predicted_value`**: Expected flu hospitalizations per 100,000 people
- **Example**: If `predicted_value = 15.2`, it means the model predicts 15.2 flu hospitalizations per 100,000 people for that location and date

### **Confidence Intervals:**
- **`pred_lower_95`**: 95% chance the actual value will be above this number
- **`pred_upper_95`**: 95% chance the actual value will be below this number
- **Example**: If predicted_value = 15.2, pred_lower_95 = 12.1, pred_upper_95 = 18.3
  - This means there's a 95% chance the actual value will be between 12.1 and 18.3

### **Prediction Horizons:**
- **1-week ahead**: Predictions for next week
- **2-week ahead**: Predictions for 2 weeks from now
- **3-week ahead**: Predictions for 3 weeks from now
- **4-week ahead**: Predictions for 4 weeks from now

## 🎯 Model Performance Metrics Explained

### **Overall Performance:**
- **MAE (Mean Absolute Error)**: Average difference between predictions and actual values
  - Example: MAE = 0.15 means predictions are off by 0.15 on average
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
  - Example: RMSE = 0.25 means larger errors are penalized more
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
  - Example: MAPE = 12.5% means predictions are off by 12.5% on average
- **R² (R-squared)**: How well the model explains the data
  - Example: R² = 0.75 means the model explains 75% of the variation

### **Performance by Location:**
Shows how well the model performs for each state/territory:
- Some locations may have better predictions than others
- This helps identify where the model is most reliable

### **Performance by Forecast Horizon:**
Shows how accuracy changes with prediction distance:
- 1-week ahead: Usually most accurate
- 4-week ahead: Usually least accurate (more uncertainty)

## 🔍 Feature Importance Interpretation

### **Most Important Features:**
- **Lag features**: Recent flu activity (most predictive)
- **Rolling statistics**: Trends over time
- **Temporal features**: Seasonal patterns
- **National context**: Overall US flu activity

### **How to Read:**
- Higher importance = more predictive
- Features with importance > 0.1 are typically very important
- Features with importance < 0.01 are usually not very predictive

## 📊 Practical Example

Let's say you get these results for California:

```
Date: 2024-11-09, Location: CA, Horizon: 1-week
Predicted Value: 18.5
Confidence Interval: 15.2 - 22.1
```

**Interpretation:**
- California is predicted to have 18.5 flu hospitalizations per 100,000 people next week
- There's a 95% chance the actual value will be between 15.2 and 22.1
- This is a 1-week ahead prediction (most reliable)

## ⚠️ Important Limitations

### **What the Model Can't Predict:**
- Sudden outbreaks or unusual events
- Changes in healthcare behavior
- New flu strains not in historical data
- External factors (vaccination campaigns, policy changes)

### **Uncertainty Increases With:**
- Longer prediction horizons (4-week is less reliable than 1-week)
- Locations with less historical data
- Unusual seasonal patterns

## 🎯 How to Use Your Forecasts

### **For Public Health Planning:**
- Use 1-2 week ahead predictions for immediate planning
- Use 3-4 week ahead predictions for general trend awareness
- Always consider confidence intervals for risk assessment

### **For Decision Making:**
- High confidence intervals = more uncertainty
- Low confidence intervals = more reliable predictions
- Compare with historical patterns for context

## 📈 Trend Analysis

### **Increasing Trends:**
- If predictions increase from 1-week to 4-week ahead
- May indicate growing flu activity
- Consider preventive measures

### **Decreasing Trends:**
- If predictions decrease from 1-week to 4-week ahead
- May indicate declining flu activity
- Good news for healthcare systems

### **Stable Trends:**
- If predictions remain similar across horizons
- May indicate steady flu activity
- Continue current monitoring

## 🔄 Model Updates

### **When to Retrain:**
- New data becomes available (weekly)
- Significant changes in flu patterns
- Model performance degrades

### **How to Update:**
```python
# Update with new cutoff date
pipeline.update_with_new_data("2024-11-09")
```

## 📞 Getting Help

If you need help interpreting specific results:
1. Check the evaluation metrics for overall model quality
2. Look at feature importance to understand what drives predictions
3. Compare confidence intervals across locations and horizons
4. Consider historical context and seasonal patterns

Remember: These are statistical predictions based on historical patterns, not guarantees of future outcomes!
