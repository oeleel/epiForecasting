"""
Location Clustering for Flu Forecasting

Groups similar locations (states) based on their flu hospitalization patterns
to enable cluster-specific models that train on more homogeneous data.

Clustering is based on:
- Peak hospitalization levels
- Peak timing
- Season duration
- Volatility
- Total seasonal burden
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

from src import config


class LocationClusterer:
    """
    Clusters locations (states) based on their flu hospitalization patterns.
    """

    def __init__(self, n_clusters: int = None):
        """
        Initialize the location clusterer.

        Args:
            n_clusters: Number of clusters (default from config)
        """
        self.n_clusters = n_clusters or getattr(config, 'N_LOCATION_CLUSTERS', 5)
        self.min_cluster_size = getattr(config, 'MIN_CLUSTER_SIZE', 3)
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.location_to_cluster = {}
        self.cluster_profiles = {}
        self.clustering_method = None

    def _identify_seasons(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify flu seasons in the data.
        A flu season runs from week 40 (October) to week 20 (May) of the next year.

        Returns DataFrame with season labels added.
        """
        df = data.copy()
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['year'] = df['date'].dt.year

        # Assign season: if week >= 40, season starts this year; otherwise it started last year
        df['season'] = df.apply(
            lambda row: f"{row['year']}-{row['year']+1}" if row['week'] >= 40
                       else f"{row['year']-1}-{row['year']}",
            axis=1
        )

        return df

    def compute_location_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a profile vector for each location based on historical patterns.

        Profile features:
        - mean_peak_value: Average peak hospitalization count across seasons
        - mean_peak_week: Average week of peak (timing)
        - mean_season_duration: Average number of weeks with significant activity
        - mean_peak_rate: Population-adjusted peak rate (if available)
        - volatility: Variance of week-over-week changes
        - mean_season_total: Average total hospitalizations per season

        Args:
            data: DataFrame with date, location, value columns

        Returns:
            DataFrame with location profiles (one row per location)
        """
        df = self._identify_seasons(data)
        profiles = []

        for location in df['location'].unique():
            loc_data = df[df['location'] == location].sort_values('date')

            if len(loc_data) < 20:  # Need sufficient data
                continue

            # Compute per-season statistics
            season_stats = []
            for season in loc_data['season'].unique():
                season_data = loc_data[loc_data['season'] == season]
                if len(season_data) < 10:  # Need at least 10 weeks
                    continue

                # Peak value and timing
                peak_idx = season_data['value'].idxmax()
                peak_value = season_data.loc[peak_idx, 'value']
                peak_week = season_data.loc[peak_idx, 'week']

                # Season duration (weeks with value > 10% of peak)
                threshold = peak_value * 0.1
                active_weeks = (season_data['value'] >= threshold).sum()

                # Total hospitalizations
                season_total = season_data['value'].sum()

                # Peak rate (if weekly_rate is available)
                if 'weekly_rate' in season_data.columns:
                    peak_rate = season_data['weekly_rate'].max()
                else:
                    peak_rate = np.nan

                season_stats.append({
                    'peak_value': peak_value,
                    'peak_week': peak_week,
                    'duration': active_weeks,
                    'season_total': season_total,
                    'peak_rate': peak_rate
                })

            if not season_stats:
                continue

            season_df = pd.DataFrame(season_stats)

            # Compute volatility (variance of week-over-week changes)
            loc_data = loc_data.sort_values('date')
            wow_changes = loc_data['value'].diff().dropna()
            volatility = wow_changes.var() if len(wow_changes) > 0 else 0

            # Aggregate across seasons
            profile = {
                'location': location,
                'mean_peak_value': season_df['peak_value'].mean(),
                'std_peak_value': season_df['peak_value'].std(),
                'mean_peak_week': season_df['peak_week'].mean(),
                'mean_season_duration': season_df['duration'].mean(),
                'mean_peak_rate': season_df['peak_rate'].mean() if not season_df['peak_rate'].isna().all() else 0,
                'volatility': volatility,
                'mean_season_total': season_df['season_total'].mean(),
                'n_seasons': len(season_df)
            }
            profiles.append(profile)

        profiles_df = pd.DataFrame(profiles)

        # Fill NaN values with 0
        profiles_df = profiles_df.fillna(0)

        return profiles_df

    def cluster_locations(self, profiles: pd.DataFrame,
                         n_clusters: Optional[int] = None) -> Dict[str, int]:
        """
        Cluster locations based on their profiles.

        Tries both KMeans and Hierarchical clustering, picks the one with
        better silhouette score.

        Args:
            profiles: DataFrame with location profiles
            n_clusters: Number of clusters (default from init)

        Returns:
            Dictionary mapping location code to cluster ID
        """
        n_clusters = n_clusters or self.n_clusters
        self.profiles_df = profiles.copy()

        # Extract features for clustering (exclude location column)
        feature_cols = ['mean_peak_value', 'mean_peak_week', 'mean_season_duration',
                       'mean_peak_rate', 'volatility', 'mean_season_total']
        X = profiles[feature_cols].values

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Try KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

        # Try Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)

        # Pick the better one
        if kmeans_silhouette >= hierarchical_silhouette:
            labels = kmeans_labels
            self.clustering_model = kmeans
            self.clustering_method = 'kmeans'
            best_silhouette = kmeans_silhouette
        else:
            labels = hierarchical_labels
            self.clustering_model = hierarchical
            self.clustering_method = 'hierarchical'
            best_silhouette = hierarchical_silhouette

        print(f"Selected {self.clustering_method} clustering "
              f"(silhouette: {best_silhouette:.3f})")
        print(f"  KMeans silhouette: {kmeans_silhouette:.3f}")
        print(f"  Hierarchical silhouette: {hierarchical_silhouette:.3f}")

        # Create location to cluster mapping
        profiles = profiles.copy()
        profiles['cluster'] = labels

        self.location_to_cluster = dict(zip(profiles['location'], profiles['cluster']))

        # Handle small clusters
        self._merge_small_clusters(profiles, X_scaled)

        # Compute cluster profiles
        self._compute_cluster_profiles(profiles)

        return self.location_to_cluster

    def _merge_small_clusters(self, profiles: pd.DataFrame, X_scaled: np.ndarray):
        """
        Merge clusters with fewer than min_cluster_size locations into the nearest cluster.
        """
        cluster_sizes = profiles.groupby('cluster').size()
        small_clusters = cluster_sizes[cluster_sizes < self.min_cluster_size].index.tolist()

        if not small_clusters:
            return

        print(f"Merging {len(small_clusters)} small clusters...")

        # Compute cluster centroids
        cluster_centroids = {}
        for cluster_id in profiles['cluster'].unique():
            cluster_mask = profiles['cluster'] == cluster_id
            cluster_centroids[cluster_id] = X_scaled[cluster_mask.values].mean(axis=0)

        # For each small cluster, find the nearest large cluster
        for small_cluster in small_clusters:
            small_centroid = cluster_centroids[small_cluster]

            # Find nearest large cluster
            min_dist = np.inf
            nearest_cluster = None
            for cluster_id, centroid in cluster_centroids.items():
                if cluster_id in small_clusters:
                    continue
                dist = np.linalg.norm(small_centroid - centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = cluster_id

            if nearest_cluster is not None:
                # Merge small cluster into nearest cluster
                for loc, cluster in self.location_to_cluster.items():
                    if cluster == small_cluster:
                        self.location_to_cluster[loc] = nearest_cluster
                print(f"  Merged cluster {small_cluster} into cluster {nearest_cluster}")

    def _compute_cluster_profiles(self, profiles: pd.DataFrame):
        """
        Compute summary statistics for each cluster.
        """
        profiles = profiles.copy()
        profiles['cluster'] = profiles['location'].map(self.location_to_cluster)

        for cluster_id in profiles['cluster'].unique():
            cluster_data = profiles[profiles['cluster'] == cluster_id]

            self.cluster_profiles[int(cluster_id)] = {
                'n_locations': len(cluster_data),
                'locations': cluster_data['location'].tolist(),
                'mean_peak_value': float(cluster_data['mean_peak_value'].mean()),
                'mean_volatility': float(cluster_data['volatility'].mean()),
                'mean_season_total': float(cluster_data['mean_season_total'].mean())
            }

    def get_cluster_description(self, cluster_id: int) -> str:
        """
        Generate a human-readable description of a cluster.
        """
        if cluster_id not in self.cluster_profiles:
            return "Unknown cluster"

        profile = self.cluster_profiles[cluster_id]
        n_locs = profile['n_locations']
        peak = profile['mean_peak_value']
        total = profile['mean_season_total']

        # Categorize by size
        if peak > 1000:
            size_desc = "Very high volume"
        elif peak > 500:
            size_desc = "High volume"
        elif peak > 100:
            size_desc = "Medium volume"
        else:
            size_desc = "Low volume"

        return f"{size_desc} ({n_locs} locations, avg peak: {peak:.0f})"

    def save_results(self, output_dir: str = "outputs/location_clusters") -> str:
        """
        Save clustering results and visualizations.

        Args:
            output_dir: Directory to save results

        Returns:
            Path to output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save cluster assignments
        assignments = pd.DataFrame([
            {'location': loc, 'cluster': cluster}
            for loc, cluster in self.location_to_cluster.items()
        ])
        assignments.to_csv(os.path.join(output_dir, 'cluster_assignments.csv'), index=False)

        # Save cluster profiles
        profiles_summary = []
        for cluster_id, profile in self.cluster_profiles.items():
            profiles_summary.append({
                'cluster': cluster_id,
                'n_locations': profile['n_locations'],
                'locations': ', '.join(profile['locations'][:10]) + ('...' if profile['n_locations'] > 10 else ''),
                'description': self.get_cluster_description(cluster_id),
                'mean_peak_value': profile['mean_peak_value'],
                'mean_season_total': profile['mean_season_total']
            })

        profiles_df = pd.DataFrame(profiles_summary)
        profiles_df.to_csv(os.path.join(output_dir, 'cluster_profiles.csv'), index=False)

        # Save full results as JSON
        results = {
            'clustering_method': self.clustering_method,
            'n_clusters': len(self.cluster_profiles),
            'timestamp': datetime.now().isoformat(),
            'cluster_summary': {
                str(k): {
                    'n_locations': v['n_locations'],
                    'locations': v['locations'],
                    'description': self.get_cluster_description(k)
                }
                for k, v in self.cluster_profiles.items()
            },
            'location_to_cluster': {k: int(v) for k, v in self.location_to_cluster.items()}
        }

        with open(os.path.join(output_dir, 'clustering_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Create visualization
        self._create_visualization(output_dir)

        print(f"Clustering results saved to: {output_dir}")
        return output_dir

    def _create_visualization(self, output_dir: str):
        """
        Create PCA visualization of clusters.
        """
        if self.profiles_df is None or len(self.profiles_df) == 0:
            return

        feature_cols = ['mean_peak_value', 'mean_peak_week', 'mean_season_duration',
                       'mean_peak_rate', 'volatility', 'mean_season_total']
        X = self.profiles_df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Get cluster assignments
        clusters = [self.location_to_cluster.get(loc, -1)
                   for loc in self.profiles_df['location']]

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                             cmap='tab10', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')

        # Add location labels
        for i, loc in enumerate(self.profiles_df['location']):
            plt.annotate(loc, (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=8, alpha=0.7)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Location Clusters (PCA Visualization)')
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def get_cluster_for_location(self, location: str) -> int:
        """
        Get the cluster ID for a given location.

        Args:
            location: Location code

        Returns:
            Cluster ID (defaults to 0 if location not found)
        """
        return self.location_to_cluster.get(location, 0)

    def get_locations_in_cluster(self, cluster_id: int) -> List[str]:
        """
        Get all locations in a given cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List of location codes
        """
        return [loc for loc, c in self.location_to_cluster.items() if c == cluster_id]

    def print_cluster_summary(self):
        """
        Print a summary of the clustering results.
        """
        print("\n" + "=" * 70)
        print("LOCATION CLUSTERING SUMMARY")
        print("=" * 70)
        print(f"Method: {self.clustering_method}")
        print(f"Number of clusters: {len(self.cluster_profiles)}")

        for cluster_id in sorted(self.cluster_profiles.keys()):
            profile = self.cluster_profiles[cluster_id]
            desc = self.get_cluster_description(cluster_id)
            locations = profile['locations']

            print(f"\nCluster {cluster_id}: {desc}")
            print(f"  Locations ({profile['n_locations']}): {', '.join(locations[:8])}"
                  + ("..." if len(locations) > 8 else ""))

        print("=" * 70)


def main():
    """Example usage of LocationClusterer."""
    from data_loader import FluDataLoader

    print("Loading data...")
    loader = FluDataLoader()
    data = loader.fetch_data()

    print("\nComputing location profiles...")
    clusterer = LocationClusterer(n_clusters=5)
    profiles = clusterer.compute_location_profiles(data)
    print(f"Computed profiles for {len(profiles)} locations")

    print("\nClustering locations...")
    location_to_cluster = clusterer.cluster_locations(profiles)

    print("\nSaving results...")
    clusterer.save_results()

    clusterer.print_cluster_summary()


if __name__ == "__main__":
    main()
