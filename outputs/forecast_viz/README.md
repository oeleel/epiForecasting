# Week-by-Week Forecast Visualizations

This directory contains interactive HTML visualizations for inspecting forecasts week-by-week to verify they properly capture increases and aren't inappropriately flat.

## Files

### Interactive Horizon Visualizations

- `horizon_1_week_interactive.html` - 1 week ahead forecasts with location dropdown
- `horizon_2_week_interactive.html` - 2 weeks ahead forecasts with location dropdown
- `horizon_3_week_interactive.html` - 3 weeks ahead forecasts with location dropdown
- `horizon_4_week_interactive.html` - 4 weeks ahead forecasts with location dropdown

### Combined Dashboard

- `all_horizons_dashboard_US.html` - All 4 horizons for US national data in one view

### Summary Statistics

- `forecast_quality_summary.csv` - Summary metrics by horizon including flat forecast detection

## How to Use

### Opening Visualizations

Simply open any HTML file in your web browser:
- Double-click the file, or
- Right-click → Open With → Browser, or
- Drag and drop into browser window

### Interactive Features

#### Location Dropdown (in individual horizon files)
- **Location:** Top-left of the plot, above the title
- **Default:** Shows "US (US)" by default
- Click the dropdown to see all 53 locations sorted alphabetically by name
- Select any location (e.g., "California (06)", "New York (36)")
- The plot instantly updates to show that location's forecasts and actuals
- Format: "Location Name (Code)" for easy identification
- **Note:** The combined dashboard (`all_horizons_dashboard_US.html`) shows only US and has no dropdown

#### Hover Tooltips
- Hover over any point to see:
  - Exact forecast and actual values
  - Cutoff date (when forecast was made)
  - Forecast error
  - Percentage error

#### Color Coding
- **Black line**: Actual hospitalization values
- **Blue dashed line**: Forecast values
- **Marker colors**:
  - 🟢 Green: Forecast captures direction correctly
  - 🟠 Orange: Forecast underestimates growth (increases but not enough)
  - 🔴 Red: Flat forecast during significant increase (BAD - what we're checking for!)
  - ⚫ Gray: No actual data available yet

#### Flat Forecast Indicators
- Red **X** markers indicate weeks where:
  - Forecast changed < 5% 
  - BUT actual values increased > 20%
  - This is the key issue we're looking for!

### Interpreting Results

#### What to Look For

1. **Flat Forecasts During Growth**
   - Look for red X markers during October-December (onset period)
   - These indicate the model failed to capture increases

2. **Underestimated Growth**
   - Orange markers show forecasts that increased but not enough
   - Common during rapid growth phases

3. **Correct Direction**
   - Green markers show good forecast performance
   - Forecast trend matches actual trend

#### Summary Statistics

View `forecast_quality_summary.csv` for:
- **flat_during_increase**: Count of flat forecasts during increases
- **flat_pct**: Percentage of forecasts that were inappropriately flat
- **underestimated_growth**: Count of forecasts that underestimated growth
- **correct_direction**: Count of forecasts with correct trend direction
- **correct_direction_pct**: Percentage with correct trend
- **mean_abs_error**: Average absolute forecast error
- **mean_pct_error**: Average percentage error

#### Current Results Summary

| Horizon | Flat During Increase | Underestimated Growth | Correct Direction |
|---------|---------------------|----------------------|-------------------|
| 1 week  | 0 (0%)              | 0                    | 0 (0%)            |
| 2 weeks | 35 (2.5%)           | 173                  | 996 (72%)         |
| 3 weeks | 30 (2.2%)           | 144                  | 790 (57%)         |
| 4 weeks | 46 (3.3%)           | 159                  | 906 (66%)         |

**Key Findings:**
- Horizon 1 has no actuals available yet (forecasts are for future dates)
- Horizons 2-4 show some flat forecasts during increases (2-3%)
- Direction accuracy decreases with longer horizons (72% → 57%)
- 4-week horizon has most flat forecasts (46 instances)

## Regenerating Visualizations

To create new visualizations:

```bash
# All horizons for all locations
python3 visualize_forecasts_weekly.py

# Specific location
python3 visualize_forecasts_weekly.py --location US

# Specific horizon only
python3 visualize_forecasts_weekly.py --horizon 2

# Open in browser automatically
python3 visualize_forecasts_weekly.py --open-browser

# Combine options
python3 visualize_forecasts_weekly.py --location CA --horizon 3 --open-browser
```

## Technical Details

### Flat Forecast Detection Criteria

A forecast is flagged as "flat during increase" when:
- Forecast week-over-week change < 5%
- AND Actual week-over-week change > 20%

This catches periods where the model should have predicted growth but remained relatively flat.

### Underestimated Growth Criteria

A forecast is flagged as "underestimated growth" when:
- Both forecast and actual show increases (positive change)
- BUT forecast increase is < 50% of actual increase

### Data Sources

- **Forecasts**: `outputs/forecasts_improved_direct.csv`
- **Actuals**: Downloaded from CDC FluSight hub via `data_loader.py`

## Questions or Issues?

If you notice systematic issues in the visualizations:
1. Check which locations/horizons are most affected
2. Look at specific time periods (onset vs peak vs decline)
3. Consider whether model needs retraining or parameter adjustment
4. Review the forecast generation logic in `direct_forecast.py`
