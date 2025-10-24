"""
Data loader for influenza hospitalization data from CDC FluSight repository
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Optional, Tuple
import config


class FluDataLoader:
    """Handles loading and preprocessing of flu hospitalization data"""
    
    def __init__(self, data_url: str = config.CDC_DATA_URL):
        self.data_url = data_url
        self.data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from CDC repository"""
        try:
            response = requests.get(self.data_url)
            response.raise_for_status()
            
            # Read CSV from URL
            from io import StringIO
            data = pd.read_csv(StringIO(response.text))
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date and location for consistent ordering
            data = data.sort_values(['date', 'location']).reset_index(drop=True)
            
            self.data = data
            return data
            
        except Exception as e:
            raise Exception(f"Failed to fetch data from {self.data_url}: {str(e)}")
    
    def apply_temporal_cutoff(self, cutoff_date: str) -> pd.DataFrame:
        """
        Apply temporal cutoff to prevent data leakage
        
        Args:
            cutoff_date: Date string in YYYY-MM-DD format
            
        Returns:
            Filtered DataFrame with data only up to cutoff date
        """
        if self.data is None:
            self.fetch_data()
        
        cutoff_dt = pd.to_datetime(cutoff_date)
        
        # Filter data up to cutoff date (inclusive)
        filtered_data = self.data[self.data['date'] <= cutoff_dt].copy()
        
        print(f"Data filtered to cutoff date {cutoff_date}")
        print(f"Date range: {filtered_data['date'].min()} to {filtered_data['date'].max()}")
        print(f"Number of records: {len(filtered_data)}")
        print(f"Number of locations: {filtered_data['location'].nunique()}")
        
        return filtered_data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and check for issues
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, raises exception otherwise
        """
        # Check for required columns
        required_cols = ['date', 'location', 'location_name', 'value', 'weekly_rate']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values in critical columns
        critical_cols = ['date', 'location', 'value']
        for col in critical_cols:
            if data[col].isnull().any():
                print(f"Warning: Missing values found in column '{col}'")
        
        # Check date consistency
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            raise ValueError("Date column is not in datetime format")
        
        # Check for negative values
        if (data['value'] < 0).any():
            print("Warning: Negative values found in 'value' column")
        
        print("Data validation completed successfully")
        return True
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get summary statistics of the data
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'date_range': (data['date'].min(), data['date'].max()),
            'total_records': len(data),
            'unique_locations': data['location'].nunique(),
            'location_list': sorted(data['location'].unique().tolist()),
            'total_hospitalizations': data['value'].sum(),
            'avg_weekly_hospitalizations': data['value'].mean(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        return summary
    
    def load_and_preprocess(self, cutoff_date: str = config.DEFAULT_CUTOFF_DATE) -> pd.DataFrame:
        """
        Complete data loading and preprocessing pipeline
        
        Args:
            cutoff_date: Date string in YYYY-MM-DD format
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        # Fetch data
        data = self.fetch_data()
        
        # Apply temporal cutoff
        data = self.apply_temporal_cutoff(cutoff_date)
        
        # Validate data
        self.validate_data(data)
        
        # Get summary
        summary = self.get_data_summary(data)
        print("\nData Summary:")
        for key, value in summary.items():
            if key != 'location_list':  # Don't print full location list
                print(f"{key}: {value}")
        
        return data


def main():
    """Example usage of the data loader"""
    loader = FluDataLoader()
    
    # Load data with Nov 2, 2024 cutoff
    data = loader.load_and_preprocess("2024-11-02")
    
    # Display sample of data
    print("\nSample data:")
    print(data.head(10))
    
    # Check data by location
    print("\nData by location (first 5 locations):")
    for location in data['location'].unique()[:5]:
        loc_data = data[data['location'] == location]
        print(f"Location {location}: {len(loc_data)} records, "
              f"date range {loc_data['date'].min()} to {loc_data['date'].max()}")


if __name__ == "__main__":
    main()

