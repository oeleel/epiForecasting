"""
Feature engineering for influenza hospitalization forecasting

Version: v2 - Optimized based on SHAP analysis
- Removed 7 low-importance features (< 1% of top feature)
- Added 8 new trend-capturing features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from src import config


class FeatureEngineer:
    """Handles feature engineering for flu forecasting"""

    def __init__(self, lag_features: List[int] = config.LAG_FEATURES,
                 rolling_windows: List[int] = config.ROLLING_WINDOWS,
                 us_lag_features: List[int] = config.US_LAG_FEATURES,
                 feature_version: str = None):
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.us_lag_features = us_lag_features
        self.feature_version = feature_version or getattr(config, 'FEATURE_VERSION', 'v1')
        self.features_removed = getattr(config, 'FEATURES_REMOVED', [])
        self.features_added = getattr(config, 'FEATURES_ADDED', [])
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date column

        Args:
            data: DataFrame with 'date' column

        Returns:
            DataFrame with added temporal features
        """
        df = data.copy()

        # Extract temporal components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['day_of_year'] = df['date'].dt.dayofyear

        # Cyclical encoding for seasonality
        # Week of year (1-52/53) -> sin/cos (PROTECTED - foundational features)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        # Month (1-12) -> sin/cos
        # month_sin is REMOVED in v2 if in features_removed (low importance)
        if 'month_sin' not in self.features_removed:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Day of year (1-365/366) -> sin/cos (PROTECTED - foundational features)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        return df
    
    def create_lag_features(self, data: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Create lag features for each location
        
        Args:
            data: DataFrame with 'location' and target_col columns
            target_col: Name of the target column
            
        Returns:
            DataFrame with added lag features
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)
        
        # Create lag features for each location
        for lag in self.lag_features:
            lag_col = f'{target_col}_lag_{lag}'
            df[lag_col] = df.groupby('location')[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Create rolling statistics for each location
        
        Args:
            data: DataFrame with 'location' and target_col columns
            target_col: Name of the target column
            
        Returns:
            DataFrame with added rolling features
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)
        
        # Create rolling features for each location
        for window in self.rolling_windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby('location')[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling standard deviation
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby('location')[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            
            # Rolling min and max
            df[f'{target_col}_rolling_min_{window}'] = (
                df.groupby('location')[target_col]
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_max_{window}'] = (
                df.groupby('location')[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(0, drop=True)
            )
        
        # Recent trend: difference from 4-week average
        if 4 in self.rolling_windows:
            df[f'{target_col}_trend_4w'] = (
                df[target_col] - df[f'{target_col}_rolling_mean_4']
            )
        
        return df
    
    def create_us_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create US-level features for context
        
        Args:
            data: DataFrame with 'location' and 'value' columns
            
        Returns:
            DataFrame with added US-level features
        """
        df = data.copy()
        
        # Calculate US total for each date
        us_data = df[df['location'] == 'US'].copy()
        if len(us_data) == 0:
            print("Warning: No US data found, skipping US features")
            return df
        
        # Create lag features for US total
        us_data = us_data.sort_values('date').reset_index(drop=True)
        for lag in self.us_lag_features:
            lag_col = f'us_total_lag_{lag}'
            us_data[lag_col] = us_data['value'].shift(lag)
        
        # Create rolling features for US total
        for window in self.rolling_windows:
            us_data[f'us_total_rolling_mean_{window}'] = (
                us_data['value'].rolling(window=window, min_periods=1).mean()
            )
            us_data[f'us_total_rolling_std_{window}'] = (
                us_data['value'].rolling(window=window, min_periods=1).std()
            )
        
        # Merge US features back to main dataframe
        us_features = ['date'] + [col for col in us_data.columns 
                                 if col.startswith('us_total_')]
        us_data_subset = us_data[us_features]
        
        df = df.merge(us_data_subset, on='date', how='left')
        
        return df
    
    def create_yoy_features(self, data: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Add year-over-year comparison features

        Args:
            data: DataFrame with 'location' and target_col columns
            target_col: Name of the target column

        Returns:
            DataFrame with added YoY features
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        # YoY change ratio (current / last year) - REMOVED in v2 if in features_removed
        if f'{target_col}_lag_52' in df.columns:
            if 'yoy_ratio' not in self.features_removed:
                df['yoy_ratio'] = df[target_col] / (df[f'{target_col}_lag_52'] + 1)

            # YoY percentage change - REMOVED in v2 if in features_removed
            if 'yoy_pct_change' not in self.features_removed:
                df['yoy_pct_change'] = (df[target_col] - df[f'{target_col}_lag_52']) / (df[f'{target_col}_lag_52'] + 1)

            # YoY difference (absolute change) - KEPT (high importance for H3/H4)
            df['yoy_diff'] = df[target_col] - df[f'{target_col}_lag_52']

        return df
    
    def create_season_severity_features(self, data: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Create features that capture how severe the current season is compared to previous seasons.
        This helps the model adjust predictions for unprecedented seasons.
        
        Args:
            data: DataFrame with 'location', 'date', and target_col columns
            target_col: Name of the target column
            
        Returns:
            DataFrame with added season severity features
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)
        
        # Calculate cumulative sum within each flu season (Oct-Apr)
        # First, identify flu season: Oct year X to Apr year X+1
        df['flu_year'] = df['date'].apply(
            lambda x: x.year if x.month >= 10 else x.year - 1
        )
        
        # Cumulative hospitalizations within each season per location
        df['season_cumsum'] = df.groupby(['location', 'flu_year'])[target_col].cumsum()
        
        # Same point last season (52-week lag of cumsum)
        df['season_cumsum_lag_52'] = df.groupby('location')['season_cumsum'].shift(52)
        
        # Season severity ratio: how much worse/better is this season vs last year
        df['season_severity_ratio'] = df['season_cumsum'] / (df['season_cumsum_lag_52'] + 1)
        df['season_severity_ratio'] = df['season_severity_ratio'].clip(0, 10)  # Cap extreme values
        
        # Rolling 4-week severity (recent trend comparison)
        rolling_4 = df.groupby('location')[target_col].rolling(4, min_periods=1).sum().reset_index(0, drop=True)
        rolling_4_lag_52 = df.groupby('location')[target_col].shift(52).rolling(4, min_periods=1).sum().reset_index(0, drop=True)
        df['recent_severity_ratio'] = rolling_4 / (rolling_4_lag_52 + 1)
        df['recent_severity_ratio'] = df['recent_severity_ratio'].clip(0, 10)
        
        # Drop intermediate columns
        df = df.drop(columns=['flu_year', 'season_cumsum', 'season_cumsum_lag_52'], errors='ignore')
        
        return df
    
    def create_rate_of_change_features(self, data: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Create rate of change and momentum features
        
        Args:
            data: DataFrame with 'location' and target_col columns
            target_col: Name of the target column
            
        Returns:
            DataFrame with added rate of change features
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)
        
        # Week-over-week change (absolute)
        df['wow_change'] = df.groupby('location')[target_col].diff(1)
        
        # Week-over-week percentage change
        df['wow_pct_change'] = df.groupby('location')[target_col].pct_change(1, fill_method=None)
        # Cap extreme values
        df['wow_pct_change'] = df['wow_pct_change'].clip(-10, 10)
        
        # Acceleration (change in change - second derivative)
        df['acceleration'] = df.groupby('location')['wow_change'].diff(1)
        
        # 4-week momentum (change over 4 weeks)
        df['momentum_4w'] = df.groupby('location')[target_col].diff(4)
        
        # 4-week percentage momentum
        shifted_4 = df.groupby('location')[target_col].shift(4)
        df['momentum_4w_pct'] = (df[target_col] - shifted_4) / (shifted_4 + 1)
        df['momentum_4w_pct'] = df['momentum_4w_pct'].clip(-10, 10)
        
        return df
    
    def create_season_phase_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create flu season phase indicator features

        Args:
            data: DataFrame with 'month' and 'week_of_year' columns

        Returns:
            DataFrame with added season phase features
        """
        df = data.copy()

        # Binary: is it flu season? (October - April) - REMOVED in v2 if in features_removed
        if 'is_flu_season' not in self.features_removed:
            df['is_flu_season'] = df['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

        # Season phase encoding (0=off-season, 1=onset, 2=peak, 3=decline)
        # Need to compute this even if not keeping it, for season_phase_sin
        week = df['week_of_year']
        season_phase_values = pd.Series(0, index=df.index)  # Default: off-season

        # Onset: weeks 40-48 (October - late November)
        season_phase_values[(week >= 40) & (week <= 48)] = 1

        # Peak: weeks 49-52 and 1-4 (December - January)
        season_phase_values[(week >= 49) | (week <= 4)] = 2

        # Decline: weeks 5-16 (February - April)
        season_phase_values[(week >= 5) & (week <= 16)] = 3

        # Only add season_phase if not in removed features
        if 'season_phase' not in self.features_removed:
            df['season_phase'] = season_phase_values

        # Cyclical encoding of season phase
        # season_phase_sin is kept (inconsistent but useful for H4)
        df['season_phase_sin'] = np.sin(2 * np.pi * season_phase_values / 4)

        # season_phase_cos is REMOVED in v2 if in features_removed
        if 'season_phase_cos' not in self.features_removed:
            df['season_phase_cos'] = np.cos(2 * np.pi * season_phase_values / 4)

        return df
    
    def create_rate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on per-capita rates (weekly_rate)
        
        Args:
            data: DataFrame with 'weekly_rate' column
            
        Returns:
            DataFrame with added rate-based features
        """
        df = data.copy()
        
        if 'weekly_rate' not in df.columns:
            return df
        
        df = df.sort_values(['location', 'date']).reset_index(drop=True)
        
        # Lag features on rate
        df['rate_lag_1'] = df.groupby('location')['weekly_rate'].shift(1)
        df['rate_lag_4'] = df.groupby('location')['weekly_rate'].shift(4)
        
        # Rolling features on rate
        df['rate_rolling_mean_4'] = (
            df.groupby('location')['weekly_rate']
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Rate trend
        df['rate_trend_4w'] = df['weekly_rate'] - df['rate_rolling_mean_4']
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different components

        Args:
            data: DataFrame with various features

        Returns:
            DataFrame with added interaction features
        """
        df = data.copy()

        # State-specific seasonal patterns
        if 'week_sin' in df.columns and 'location' in df.columns:
            # This will be handled by XGBoost's ability to learn interactions
            # between location (categorical) and temporal features
            pass

        # Recent vs historical comparison
        if 'value_rolling_mean_4' in df.columns and 'value_rolling_mean_8' in df.columns:
            df['recent_vs_historical'] = (
                df['value_rolling_mean_4'] / (df['value_rolling_mean_8'] + 1e-8)
            )

        # Seasonality-adjusted recent trend (REMOVED in v2 - low SHAP importance)
        # Only create if not in removed features list
        if 'seasonal_deviation' not in self.features_removed:
            if 'value_lag_52' in df.columns and 'value_lag_1' in df.columns:
                df['seasonal_deviation'] = (df['value_lag_1'] - df['value_lag_52']) / (df['value_lag_52'] + 1)
                df['seasonal_deviation'] = df['seasonal_deviation'].clip(-10, 10)

        return df

    def create_trend_features(self, data: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Create advanced trend-capturing features (new in v2).

        These features help the model better capture trend direction, acceleration,
        and unprecedented situations.

        Args:
            data: DataFrame with 'location', 'date', and target_col columns
            target_col: Name of the target column

        Returns:
            DataFrame with added trend features
        """
        df = data.copy()
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        # 1. Linear trend slope over 4 weeks
        # Uses least-squares fit to capture direction and steepness of recent trend
        def calc_slope(values):
            if len(values) < 2 or values.isna().all():
                return 0.0
            valid = values.dropna()
            if len(valid) < 2:
                return 0.0
            x = np.arange(len(valid))
            try:
                slope, _, _, _, _ = stats.linregress(x, valid)
                return slope if np.isfinite(slope) else 0.0
            except:
                return 0.0

        df['linear_trend_slope_4w'] = (
            df.groupby('location')[target_col]
            .rolling(window=4, min_periods=2)
            .apply(calc_slope, raw=False)
            .reset_index(0, drop=True)
        )

        # 2. Linear trend slope over 3 weeks (REMOVED in v3 - redundant with 4w)
        if 'linear_trend_slope_3w' not in self.features_removed:
            df['linear_trend_slope_3w'] = (
                df.groupby('location')[target_col]
                .rolling(window=3, min_periods=2)
                .apply(calc_slope, raw=False)
                .reset_index(0, drop=True)
            )

        # 3. Value to rolling mean ratio (acceleration indicator)
        # > 1 means accelerating, < 1 means decelerating
        rolling_mean_4 = (
            df.groupby('location')[target_col]
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df['value_to_rolling_mean_ratio'] = df[target_col] / (rolling_mean_4 + 1e-6)
        df['value_to_rolling_mean_ratio'] = df['value_to_rolling_mean_ratio'].clip(0.1, 10.0)

        # 4. Consecutive increase weeks (REMOVED in v3 - low importance)
        # Count of consecutive weeks where value increased
        if 'consecutive_increase_weeks' not in self.features_removed:
            def count_consecutive_increases(group):
                values = group[target_col].values
                result = np.zeros(len(values))
                count = 0
                for i in range(1, len(values)):
                    if values[i] > values[i-1]:
                        count += 1
                    else:
                        count = 0
                    result[i] = count
                return pd.Series(result, index=group.index)

            df['consecutive_increase_weeks'] = df.groupby('location', group_keys=False).apply(
                count_consecutive_increases
            )

        # 5. Consecutive decrease weeks (REMOVED in v3 - low importance)
        if 'consecutive_decrease_weeks' not in self.features_removed:
            def count_consecutive_decreases(group):
                values = group[target_col].values
                result = np.zeros(len(values))
                count = 0
                for i in range(1, len(values)):
                    if values[i] < values[i-1]:
                        count += 1
                    else:
                        count = 0
                    result[i] = count
                return pd.Series(result, index=group.index)

            df['consecutive_decrease_weeks'] = df.groupby('location', group_keys=False).apply(
                count_consecutive_decreases
            )

        # 6. Historical percentile
        # Where current value sits in the historical distribution for this location
        def calc_percentile(group):
            values = group[target_col].values
            result = np.zeros(len(values))
            for i in range(len(values)):
                # Only use data up to current point (no leakage)
                historical = values[:i+1]
                if len(historical) > 1:
                    result[i] = stats.percentileofscore(historical, values[i], kind='rank')
                else:
                    result[i] = 50.0  # Default to median for first observation
            return pd.Series(result, index=group.index)

        df['historical_percentile'] = df.groupby('location', group_keys=False).apply(
            calc_percentile
        )

        # 7. Ratio to historical max
        # When > 1.0, we're in unprecedented territory
        def calc_ratio_to_max(group):
            values = group[target_col].values
            result = np.zeros(len(values))
            for i in range(len(values)):
                # Only use data up to current point (no leakage)
                historical_max = np.nanmax(values[:i+1])
                if historical_max > 0:
                    result[i] = values[i] / historical_max
                else:
                    result[i] = 1.0
            return pd.Series(result, index=group.index)

        df['ratio_to_historical_max'] = df.groupby('location', group_keys=False).apply(
            calc_ratio_to_max
        )

        # 8. Surge indicator (REMOVED in v3 - zero importance)
        # Binary flag: 1 if wow_change > 2 std dev above mean for that location
        if 'surge_indicator' not in self.features_removed:
            if 'wow_change' not in df.columns:
                df['wow_change'] = df.groupby('location')[target_col].diff(1)

            def calc_surge_indicator(group):
                wow = group['wow_change'].values
                result = np.zeros(len(wow))
                for i in range(len(wow)):
                    # Only use data up to current point (no leakage)
                    historical_wow = wow[:i+1]
                    valid = historical_wow[~np.isnan(historical_wow)]
                    if len(valid) > 2:
                        mean_wow = np.mean(valid)
                        std_wow = np.std(valid)
                        if std_wow > 0 and not np.isnan(wow[i]):
                            if wow[i] > mean_wow + 2 * std_wow:
                                result[i] = 1
                return pd.Series(result, index=group.index)

            df['surge_indicator'] = df.groupby('location', group_keys=False).apply(
                calc_surge_indicator
            )

        return df
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            data: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        df = data.copy()

        # Forward fill missing values within each location
        df = df.sort_values(['location', 'date']).reset_index(drop=True)

        # Forward fill for lag and rolling features
        lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        for col in lag_cols:
            df[col] = df.groupby('location')[col].ffill()

        # Forward fill for rate-of-change features
        roc_cols = ['wow_change', 'wow_pct_change', 'acceleration', 'momentum_4w', 'momentum_4w_pct',
                    'yoy_ratio', 'yoy_pct_change', 'yoy_diff', 'seasonal_deviation',
                    'rate_lag_1', 'rate_lag_4', 'rate_rolling_mean_4', 'rate_trend_4w',
                    'season_severity_ratio', 'recent_severity_ratio']
        for col in roc_cols:
            if col in df.columns:
                df[col] = df.groupby('location')[col].ffill()

        # Forward fill for v2 trend features
        trend_cols = ['linear_trend_slope_4w', 'linear_trend_slope_3w', 'value_to_rolling_mean_ratio',
                      'consecutive_increase_weeks', 'consecutive_decrease_weeks',
                      'historical_percentile', 'ratio_to_historical_max', 'surge_indicator']
        for col in trend_cols:
            if col in df.columns:
                df[col] = df.groupby('location')[col].ffill()

        # Fill remaining missing values with 0 for value columns
        value_cols = [col for col in df.columns if 'value' in col or 'us_total' in col]
        for col in value_cols:
            df[col] = df[col].fillna(0)

        # Fill remaining numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)

        return df
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in the correct order

        Args:
            data: Raw DataFrame with 'date', 'location', 'value' columns

        Returns:
            DataFrame with all engineered features
        """
        print(f"Feature engineering version: {self.feature_version}")
        if self.features_removed:
            print(f"Features to skip: {len(self.features_removed)}")

        print("Creating temporal features...")
        df = self.create_temporal_features(data)

        print("Creating lag features...")
        df = self.create_lag_features(df)

        print("Creating rolling features...")
        df = self.create_rolling_features(df)

        print("Creating YoY features...")
        df = self.create_yoy_features(df)

        print("Creating season severity features...")
        df = self.create_season_severity_features(df)

        print("Creating rate-of-change features...")
        df = self.create_rate_of_change_features(df)

        print("Creating season phase features...")
        df = self.create_season_phase_features(df)

        print("Creating rate-based features...")
        df = self.create_rate_features(df)

        print("Creating US-level features...")
        df = self.create_us_features(df)

        print("Creating interaction features...")
        df = self.create_interaction_features(df)

        # NEW in v2: Add trend-capturing features
        if self.feature_version in ('v2', 'v3'):
            print("Creating trend features...")
            df = self.create_trend_features(df)

        print("Handling missing values...")
        df = self.handle_missing_values(df)

        # Remove rows where target is missing (needed for lag features)
        df = df.dropna(subset=['value']).reset_index(drop=True)

        # Count and display feature information
        feature_cols = self.get_feature_columns(df)
        print(f"Feature engineering completed. Final shape: {df.shape}")
        print(f"Total features: {len(feature_cols)}")
        if self.features_removed:
            print(f"Features removed (v2): {self.features_removed}")
        if self.feature_version in ('v2', 'v3'):
            added_in_df = [f for f in self.features_added if f in df.columns]
            print(f"Features added: {added_in_df}")

        return df
    
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding target and metadata)
        
        Args:
            data: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['date', 'location', 'location_name', 'value', 'weekly_rate']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        return feature_cols


def main():
    """Example usage of feature engineering"""
    from data_loader import FluDataLoader
    
    # Load data
    loader = FluDataLoader()
    data = loader.load_and_preprocess("2024-11-02")
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(data)
    
    # Display feature information
    feature_cols = engineer.get_feature_columns(features_df)
    print(f"\nCreated {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Show sample of features
    print(f"\nSample of engineered features:")
    print(features_df[['date', 'location', 'value'] + feature_cols[:10]].head())


if __name__ == "__main__":
    main()

