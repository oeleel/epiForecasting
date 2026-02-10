"""
Data loader for influenza hospitalization data from CDC FluSight repository

Fetches data directly from the FluSight GitHub repository and caches locally.
The repository updates weekly, so use update_data() to get the latest.

Repository: https://github.com/cdcepi/FluSight-forecast-hub
"""

import pandas as pd
import requests
import os
import json
from datetime import datetime
from typing import Optional, Tuple, Dict
from pathlib import Path
from src import config


class FluDataLoader:
    """
    Handles loading and preprocessing of flu hospitalization data.

    Data is fetched from the CDC FluSight GitHub repository and cached locally.
    Use update_data() to refresh the local cache with the latest data.
    """

    # FluSight GitHub repository URLs
    FLUSIGHT_REPO = "cdcepi/FluSight-forecast-hub"
    FLUSIGHT_BRANCH = "main"
    TARGET_DATA_PATH = "target-data/target-hospital-admissions.csv"

    # Raw GitHub content URL
    RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"

    # GitHub API URL for commit info
    API_URL_TEMPLATE = "https://api.github.com/repos/{repo}/commits?path={path}&per_page=1"

    def __init__(self,
                 cache_dir: str = "data/raw",
                 cache_filename: str = "flusight_hospital_admissions.csv",
                 metadata_filename: str = "flusight_metadata.json"):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to store cached data
            cache_filename: Filename for cached CSV data
            metadata_filename: Filename for cache metadata (timestamps, etc.)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / cache_filename
        self.metadata_file = self.cache_dir / metadata_filename
        self.data = None

        # Build URLs
        self.data_url = self.RAW_URL_TEMPLATE.format(
            repo=self.FLUSIGHT_REPO,
            branch=self.FLUSIGHT_BRANCH,
            path=self.TARGET_DATA_PATH
        )
        self.api_url = self.API_URL_TEMPLATE.format(
            repo=self.FLUSIGHT_REPO,
            path=self.TARGET_DATA_PATH
        )

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_metadata(self) -> Dict:
        """
        Get metadata about the cached data.

        Returns:
            Dictionary with cache metadata (last_updated, source, etc.)
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache_metadata(self, metadata: Dict) -> None:
        """Save cache metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_remote_last_updated(self) -> Optional[str]:
        """
        Check when the remote data was last updated on GitHub.

        Returns:
            ISO format datetime string of last commit, or None if unavailable
        """
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            commits = response.json()
            if commits:
                return commits[0]['commit']['committer']['date']
        except Exception as e:
            print(f"Could not check remote update time: {e}")
        return None

    def is_cache_stale(self, max_age_hours: int = 168) -> bool:
        """
        Check if the local cache is stale.

        Args:
            max_age_hours: Maximum age in hours before cache is considered stale
                          Default is 168 (1 week) since FluSight updates weekly

        Returns:
            True if cache is stale or doesn't exist, False otherwise
        """
        metadata = self.get_cache_metadata()
        if not metadata or 'last_fetched' not in metadata:
            return True

        try:
            last_fetched = datetime.fromisoformat(metadata['last_fetched'].replace('Z', '+00:00'))
            age_hours = (datetime.now(last_fetched.tzinfo) - last_fetched).total_seconds() / 3600
            return age_hours > max_age_hours
        except Exception:
            return True

    def fetch_from_github(self, force: bool = False) -> pd.DataFrame:
        """
        Fetch the latest data from the FluSight GitHub repository.

        Args:
            force: If True, fetch even if cache is fresh

        Returns:
            DataFrame with hospitalization data
        """
        print(f"Fetching data from FluSight GitHub repository...")
        print(f"URL: {self.data_url}")

        try:
            response = requests.get(self.data_url, timeout=60)
            response.raise_for_status()

            # Parse CSV
            from io import StringIO
            data = pd.read_csv(StringIO(response.text))

            # Convert date column
            data['date'] = pd.to_datetime(data['date'])

            # Sort by date and location
            data = data.sort_values(['date', 'location']).reset_index(drop=True)

            # Save to cache
            data.to_csv(self.cache_file, index=False)

            # Get remote commit info
            remote_updated = self.get_remote_last_updated()

            # Save metadata
            metadata = {
                'last_fetched': datetime.utcnow().isoformat() + 'Z',
                'remote_last_updated': remote_updated,
                'source_url': self.data_url,
                'repository': self.FLUSIGHT_REPO,
                'records': len(data),
                'date_range': {
                    'min': data['date'].min().isoformat(),
                    'max': data['date'].max().isoformat()
                },
                'locations': data['location'].nunique()
            }
            self._save_cache_metadata(metadata)

            print(f"Successfully fetched {len(data)} records")
            print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
            print(f"Locations: {data['location'].nunique()}")
            print(f"Data cached to: {self.cache_file}")

            self.data = data
            return data

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch data from GitHub: {e}")

    def load_from_cache(self) -> pd.DataFrame:
        """
        Load data from local cache.

        Returns:
            DataFrame with hospitalization data

        Raises:
            FileNotFoundError: If cache doesn't exist
        """
        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"No cached data found at {self.cache_file}. "
                "Run update_data() to fetch from GitHub."
            )

        print(f"Loading data from cache: {self.cache_file}")
        data = pd.read_csv(self.cache_file)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(['date', 'location']).reset_index(drop=True)

        # Show cache info
        metadata = self.get_cache_metadata()
        if metadata:
            last_fetched = metadata.get('last_fetched', 'unknown')
            print(f"Cache last updated: {last_fetched}")

        self.data = data
        return data

    def fetch_data(self, use_cache: bool = True, max_cache_age_hours: int = 168) -> pd.DataFrame:
        """
        Fetch data, using cache if available and fresh.

        Args:
            use_cache: If True, use cached data if available and not stale
            max_cache_age_hours: Maximum cache age before fetching fresh data

        Returns:
            DataFrame with hospitalization data
        """
        # Try to use cache if requested
        if use_cache and self.cache_file.exists():
            if not self.is_cache_stale(max_cache_age_hours):
                return self.load_from_cache()
            else:
                print("Cache is stale, fetching fresh data...")

        # Fetch from GitHub
        try:
            return self.fetch_from_github()
        except ConnectionError as e:
            # Fall back to cache if available
            if self.cache_file.exists():
                print(f"Warning: {e}")
                print("Falling back to cached data...")
                return self.load_from_cache()
            raise

    def update_data(self) -> pd.DataFrame:
        """
        Force update data from the FluSight GitHub repository.

        This is the main method users should call to get fresh data.

        Returns:
            DataFrame with the latest hospitalization data
        """
        print("=" * 60)
        print("UPDATING FLU HOSPITALIZATION DATA")
        print("=" * 60)
        print(f"Source: {self.FLUSIGHT_REPO}")
        print(f"File: {self.TARGET_DATA_PATH}")
        print("=" * 60)

        # Check current cache status
        metadata = self.get_cache_metadata()
        if metadata:
            print(f"\nCurrent cache:")
            print(f"  Last fetched: {metadata.get('last_fetched', 'N/A')}")
            print(f"  Records: {metadata.get('records', 'N/A')}")
            date_range = metadata.get('date_range', {})
            print(f"  Date range: {date_range.get('min', 'N/A')} to {date_range.get('max', 'N/A')}")

        print("\nFetching latest data...")
        data = self.fetch_from_github(force=True)

        print("\n" + "=" * 60)
        print("UPDATE COMPLETE")
        print("=" * 60)

        return data

    def get_data_status(self) -> Dict:
        """
        Get the current status of the data cache.

        Returns:
            Dictionary with cache status information
        """
        status = {
            'cache_exists': self.cache_file.exists(),
            'cache_path': str(self.cache_file),
            'is_stale': self.is_cache_stale(),
        }

        if self.cache_file.exists():
            metadata = self.get_cache_metadata()
            status.update(metadata)

        return status

    def apply_temporal_cutoff(self, cutoff_date: str) -> pd.DataFrame:
        """
        Apply temporal cutoff to prevent data leakage.

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
        Validate data quality and check for issues.

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
        Get summary statistics of the data.

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

    def load_and_preprocess(self, cutoff_date: str = None) -> pd.DataFrame:
        """
        Complete data loading and preprocessing pipeline.

        Args:
            cutoff_date: Date string in YYYY-MM-DD format (default from config)

        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        if cutoff_date is None:
            cutoff_date = config.DEFAULT_CUTOFF_DATE

        # Fetch data (uses cache if fresh)
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
                print(f"  {key}: {value}")

        return data


def update_data():
    """
    CLI function to update the local data cache from FluSight GitHub.

    This can be called directly:
        python -m src.data_loader update
    """
    loader = FluDataLoader()
    loader.update_data()


def show_status():
    """
    CLI function to show the current data cache status.

    This can be called directly:
        python -m src.data_loader status
    """
    loader = FluDataLoader()
    status = loader.get_data_status()

    print("=" * 60)
    print("FLU DATA CACHE STATUS")
    print("=" * 60)

    if status['cache_exists']:
        print(f"Cache file: {status['cache_path']}")
        print(f"Last fetched: {status.get('last_fetched', 'N/A')}")
        print(f"Remote last updated: {status.get('remote_last_updated', 'N/A')}")
        print(f"Records: {status.get('records', 'N/A')}")
        date_range = status.get('date_range', {})
        print(f"Date range: {date_range.get('min', 'N/A')} to {date_range.get('max', 'N/A')}")
        print(f"Locations: {status.get('locations', 'N/A')}")
        print(f"Cache stale: {'Yes' if status['is_stale'] else 'No'}")
    else:
        print(f"No cache found at: {status['cache_path']}")
        print("Run 'python -m src.data_loader update' to fetch data.")

    print("=" * 60)


def main():
    """Main entry point for CLI usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data_loader <command>")
        print("")
        print("Commands:")
        print("  update  - Fetch latest data from FluSight GitHub repository")
        print("  status  - Show current data cache status")
        print("  demo    - Run demo showing data loading")
        return

    command = sys.argv[1].lower()

    if command == 'update':
        update_data()
    elif command == 'status':
        show_status()
    elif command == 'demo':
        # Demo usage
        loader = FluDataLoader()
        data = loader.load_and_preprocess("2024-11-02")
        print("\nSample data:")
        print(data.head(10))
    else:
        print(f"Unknown command: {command}")
        print("Use 'update', 'status', or 'demo'")


if __name__ == "__main__":
    main()
