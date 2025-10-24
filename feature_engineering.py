"""
Feature engineering for influenza hospitalization forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import config


class FeatureEngineer:
    """Handles feature engineering for flu forecasting"""
    
    def __init__(self, lag_features: List[int] = config.LAG_FEATURES,
                 rolling_windows: List[int] = config.ROLLING_WINDOWS,
                 us_lag_features: List[int] = config.US_LAG_FEATURES):
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.us_lag_features = us_lag_features
    
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
        # Week of year (1-52/53) -> sin/cos
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Month (1-12) -> sin/cos
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year (1-365/366) -> sin/cos
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
            df[col] = df.groupby('location')[col].fillna(method='ffill')
        
        # Fill remaining missing values with 0 for value columns
        value_cols = [col for col in df.columns if 'value' in col or 'us_total' in col]
        for col in value_cols:
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
        print("Creating temporal features...")
        df = self.create_temporal_features(data)
        
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        print("Creating rolling features...")
        df = self.create_rolling_features(df)
        
        print("Creating US-level features...")
        df = self.create_us_features(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Remove rows where target is missing (needed for lag features)
        df = df.dropna(subset=['value']).reset_index(drop=True)
        
        print(f"Feature engineering completed. Final shape: {df.shape}")
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

