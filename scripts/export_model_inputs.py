"""
Export processed model input features for sharing with research colleagues.

This script generates and saves the complete processed feature dataset (59 engineered features)
that is fed into the XGBoost forecasting model.
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.data_loader import FluDataLoader
from src.feature_engineering import FeatureEngineer
from src import config


class ModelInputExporter:
    """Export and document model input features"""
    
    def __init__(self, output_dir: str = "data/processed_features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = FluDataLoader()
        self.engineer = FeatureEngineer()
        
    def get_feature_descriptions(self) -> Dict[str, Dict]:
        """
        Generate comprehensive feature descriptions for all 59 features.
        
        Returns:
            Dictionary mapping feature names to their metadata
        """
        descriptions = {}
        
        # Lag features
        for lag in config.LAG_FEATURES:
            descriptions[f'value_lag_{lag}'] = {
                'category': 'Lag Features',
                'description': f'Hospitalization count {lag} weeks ago',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
        
        # Rolling statistics
        for window in config.ROLLING_WINDOWS:
            descriptions[f'value_rolling_mean_{window}'] = {
                'category': 'Rolling Statistics',
                'description': f'{window}-week rolling mean of hospitalizations',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
            descriptions[f'value_rolling_std_{window}'] = {
                'category': 'Rolling Statistics',
                'description': f'{window}-week rolling standard deviation of hospitalizations',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
            descriptions[f'value_rolling_min_{window}'] = {
                'category': 'Rolling Statistics',
                'description': f'{window}-week rolling minimum hospitalizations',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
            descriptions[f'value_rolling_max_{window}'] = {
                'category': 'Rolling Statistics',
                'description': f'{window}-week rolling maximum hospitalizations',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
        
        # Trend feature
        descriptions['value_trend_4w'] = {
            'category': 'Rolling Statistics',
            'description': 'Deviation from 4-week rolling average (current - mean)',
            'type': 'numeric',
            'unit': 'hospitalizations'
        }
        
        # Temporal features
        descriptions['year'] = {
            'category': 'Temporal Features',
            'description': 'Year',
            'type': 'numeric',
            'unit': 'year'
        }
        descriptions['month'] = {
            'category': 'Temporal Features',
            'description': 'Month (1-12)',
            'type': 'numeric',
            'unit': 'month'
        }
        descriptions['week_of_year'] = {
            'category': 'Temporal Features',
            'description': 'ISO week number (1-52/53)',
            'type': 'numeric',
            'unit': 'week'
        }
        descriptions['day_of_year'] = {
            'category': 'Temporal Features',
            'description': 'Day of year (1-365/366)',
            'type': 'numeric',
            'unit': 'day'
        }
        descriptions['week_sin'] = {
            'category': 'Temporal Features',
            'description': 'Sine encoding of week of year (seasonal cycle)',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        descriptions['week_cos'] = {
            'category': 'Temporal Features',
            'description': 'Cosine encoding of week of year (seasonal cycle)',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        descriptions['month_sin'] = {
            'category': 'Temporal Features',
            'description': 'Sine encoding of month (seasonal cycle)',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        descriptions['month_cos'] = {
            'category': 'Temporal Features',
            'description': 'Cosine encoding of month (seasonal cycle)',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        descriptions['day_sin'] = {
            'category': 'Temporal Features',
            'description': 'Sine encoding of day of year (seasonal cycle)',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        descriptions['day_cos'] = {
            'category': 'Temporal Features',
            'description': 'Cosine encoding of day of year (seasonal cycle)',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        
        # US national features
        for lag in config.US_LAG_FEATURES:
            descriptions[f'us_total_lag_{lag}'] = {
                'category': 'US National Context',
                'description': f'US national total hospitalizations {lag} weeks ago',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
        
        for window in config.ROLLING_WINDOWS:
            descriptions[f'us_total_rolling_mean_{window}'] = {
                'category': 'US National Context',
                'description': f'{window}-week rolling mean of US national hospitalizations',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
            descriptions[f'us_total_rolling_std_{window}'] = {
                'category': 'US National Context',
                'description': f'{window}-week rolling std of US national hospitalizations',
                'type': 'numeric',
                'unit': 'hospitalizations'
            }
        
        # Year-over-year features
        descriptions['yoy_ratio'] = {
            'category': 'Year-over-Year',
            'description': 'Current hospitalizations / same week last year',
            'type': 'numeric',
            'unit': 'ratio'
        }
        descriptions['yoy_pct_change'] = {
            'category': 'Year-over-Year',
            'description': 'Percentage change vs same week last year',
            'type': 'numeric',
            'unit': 'percentage'
        }
        descriptions['yoy_diff'] = {
            'category': 'Year-over-Year',
            'description': 'Absolute difference vs same week last year',
            'type': 'numeric',
            'unit': 'hospitalizations'
        }
        
        # Season severity features
        descriptions['season_severity_ratio'] = {
            'category': 'Season Severity',
            'description': 'Cumulative season hospitalizations vs same point last season',
            'type': 'numeric',
            'unit': 'ratio'
        }
        descriptions['recent_severity_ratio'] = {
            'category': 'Season Severity',
            'description': 'Recent 4-week severity vs same period last year',
            'type': 'numeric',
            'unit': 'ratio'
        }
        
        # Rate of change features
        descriptions['wow_change'] = {
            'category': 'Rate of Change',
            'description': 'Week-over-week absolute change in hospitalizations',
            'type': 'numeric',
            'unit': 'hospitalizations'
        }
        descriptions['wow_pct_change'] = {
            'category': 'Rate of Change',
            'description': 'Week-over-week percentage change',
            'type': 'numeric',
            'unit': 'percentage'
        }
        descriptions['acceleration'] = {
            'category': 'Rate of Change',
            'description': 'Change in week-over-week change (second derivative)',
            'type': 'numeric',
            'unit': 'hospitalizations'
        }
        descriptions['momentum_4w'] = {
            'category': 'Rate of Change',
            'description': '4-week momentum (absolute change over 4 weeks)',
            'type': 'numeric',
            'unit': 'hospitalizations'
        }
        descriptions['momentum_4w_pct'] = {
            'category': 'Rate of Change',
            'description': '4-week momentum (percentage change over 4 weeks)',
            'type': 'numeric',
            'unit': 'percentage'
        }
        
        # Season phase features
        descriptions['is_flu_season'] = {
            'category': 'Season Phase',
            'description': 'Binary indicator: 1 if October-April, 0 otherwise',
            'type': 'binary',
            'unit': 'boolean'
        }
        descriptions['season_phase'] = {
            'category': 'Season Phase',
            'description': 'Season phase: 0=off-season, 1=onset, 2=peak, 3=decline',
            'type': 'categorical',
            'unit': 'phase_code'
        }
        descriptions['season_phase_sin'] = {
            'category': 'Season Phase',
            'description': 'Sine encoding of season phase',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        descriptions['season_phase_cos'] = {
            'category': 'Season Phase',
            'description': 'Cosine encoding of season phase',
            'type': 'numeric',
            'unit': 'dimensionless'
        }
        
        # Rate-based features
        descriptions['rate_lag_1'] = {
            'category': 'Rate Features',
            'description': 'Per-capita hospitalization rate 1 week ago',
            'type': 'numeric',
            'unit': 'rate_per_100k'
        }
        descriptions['rate_lag_4'] = {
            'category': 'Rate Features',
            'description': 'Per-capita hospitalization rate 4 weeks ago',
            'type': 'numeric',
            'unit': 'rate_per_100k'
        }
        descriptions['rate_rolling_mean_4'] = {
            'category': 'Rate Features',
            'description': '4-week rolling mean of per-capita rate',
            'type': 'numeric',
            'unit': 'rate_per_100k'
        }
        descriptions['rate_trend_4w'] = {
            'category': 'Rate Features',
            'description': 'Deviation from 4-week rolling mean rate',
            'type': 'numeric',
            'unit': 'rate_per_100k'
        }
        
        # Interaction features
        descriptions['recent_vs_historical'] = {
            'category': 'Interaction Features',
            'description': 'Ratio of 4-week mean to 8-week mean',
            'type': 'numeric',
            'unit': 'ratio'
        }
        descriptions['seasonal_deviation'] = {
            'category': 'Interaction Features',
            'description': 'Normalized comparison to same week last year',
            'type': 'numeric',
            'unit': 'ratio'
        }
        
        return descriptions
    
    def export_features(self, cutoff_date: str = config.DEFAULT_CUTOFF_DATE,
                       include_metadata: bool = True) -> Dict[str, Path]:
        """
        Export processed features to CSV file.
        
        Args:
            cutoff_date: Date cutoff for data (YYYY-MM-DD format)
            include_metadata: Whether to generate metadata JSON
            
        Returns:
            Dictionary with paths to created files
        """
        print(f"Loading data with cutoff date: {cutoff_date}")
        
        # Load raw data
        raw_data = self.loader.load_and_preprocess(cutoff_date)
        
        # Create features
        print("\nCreating features...")
        features_df = self.engineer.create_all_features(raw_data)
        
        # Generate output filename
        cutoff_str = cutoff_date.replace("-", "")
        output_csv = self.output_dir / f"model_inputs_{cutoff_date}.csv"
        
        # Save features
        print(f"\nSaving processed features to: {output_csv}")
        features_df.to_csv(output_csv, index=False)
        
        created_files = {'features': output_csv}
        
        # Generate metadata
        if include_metadata:
            metadata = self.generate_metadata(features_df, cutoff_date, raw_data)
            metadata_file = self.output_dir / f"metadata_{cutoff_date}.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"Saved metadata to: {metadata_file}")
            created_files['metadata'] = metadata_file
        
        print(f"\nExport completed successfully!")
        print(f"Total records: {len(features_df)}")
        print(f"Total features: {len(self.engineer.get_feature_columns(features_df))}")
        
        return created_files
    
    def generate_metadata(self, features_df: pd.DataFrame, 
                         cutoff_date: str, raw_data: pd.DataFrame) -> Dict:
        """
        Generate metadata about the exported features.
        
        Args:
            features_df: DataFrame with processed features
            cutoff_date: Data cutoff date
            raw_data: Original raw data
            
        Returns:
            Dictionary with metadata
        """
        feature_cols = self.engineer.get_feature_columns(features_df)
        
        # Calculate statistics by category
        feature_descriptions = self.get_feature_descriptions()
        categories = {}
        for feat in feature_cols:
            if feat in feature_descriptions:
                cat = feature_descriptions[feat]['category']
                categories[cat] = categories.get(cat, 0) + 1
        
        metadata = {
            'export_info': {
                'export_date': datetime.now().isoformat(),
                'cutoff_date': cutoff_date,
                'script_version': '1.0'
            },
            'data_summary': {
                'total_records': len(features_df),
                'total_features': len(feature_cols),
                'date_range': {
                    'start': features_df['date'].min().isoformat(),
                    'end': features_df['date'].max().isoformat()
                },
                'locations': {
                    'count': features_df['location'].nunique(),
                    'list': sorted(features_df['location'].unique().tolist())
                },
                'total_hospitalizations': float(raw_data['value'].sum()),
                'avg_weekly_hospitalizations': float(raw_data['value'].mean())
            },
            'feature_categories': categories,
            'feature_list': feature_cols,
            'configuration': {
                'lag_features': config.LAG_FEATURES,
                'rolling_windows': config.ROLLING_WINDOWS,
                'us_lag_features': config.US_LAG_FEATURES
            },
            'missing_values': {
                col: int(features_df[col].isnull().sum()) 
                for col in feature_cols if features_df[col].isnull().sum() > 0
            }
        }
        
        return metadata
    
    def export_feature_dictionary(self) -> Path:
        """
        Export a CSV file documenting all features.
        
        Returns:
            Path to the feature dictionary CSV
        """
        descriptions = self.get_feature_descriptions()
        
        # Convert to DataFrame
        dict_data = []
        for feature_name, info in sorted(descriptions.items()):
            dict_data.append({
                'feature_name': feature_name,
                'category': info['category'],
                'description': info['description'],
                'type': info['type'],
                'unit': info['unit']
            })
        
        dict_df = pd.DataFrame(dict_data)
        
        # Save to data directory (not processed_features, at data root)
        output_path = self.output_dir.parent / "feature_dictionary.csv"
        dict_df.to_csv(output_path, index=False)
        
        print(f"\nFeature dictionary saved to: {output_path}")
        print(f"Total features documented: {len(dict_df)}")
        
        # Print category summary
        print("\nFeature categories:")
        for category, count in dict_df['category'].value_counts().items():
            print(f"  {category}: {count} features")
        
        return output_path


def main():
    """Command-line interface for exporting model inputs"""
    parser = argparse.ArgumentParser(
        description='Export processed model input features for research sharing'
    )
    parser.add_argument(
        '--cutoff-date',
        type=str,
        default=config.DEFAULT_CUTOFF_DATE,
        help=f'Data cutoff date in YYYY-MM-DD format (default: {config.DEFAULT_CUTOFF_DATE})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_features',
        help='Output directory for processed features (default: data/processed_features)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip metadata generation'
    )
    parser.add_argument(
        '--feature-dict-only',
        action='store_true',
        help='Only generate feature dictionary, skip data export'
    )
    
    args = parser.parse_args()
    
    exporter = ModelInputExporter(output_dir=args.output)
    
    if args.feature_dict_only:
        print("Generating feature dictionary only...")
        exporter.export_feature_dictionary()
    else:
        print("=" * 70)
        print("Model Input Data Export")
        print("=" * 70)
        
        # Export features
        files = exporter.export_features(
            cutoff_date=args.cutoff_date,
            include_metadata=not args.no_metadata
        )
        
        # Also export feature dictionary
        print("\nGenerating feature dictionary...")
        dict_file = exporter.export_feature_dictionary()
        files['dictionary'] = dict_file
        
        print("\n" + "=" * 70)
        print("Export Complete!")
        print("=" * 70)
        print("\nCreated files:")
        for file_type, filepath in files.items():
            print(f"  {file_type}: {filepath}")


if __name__ == "__main__":
    main()
