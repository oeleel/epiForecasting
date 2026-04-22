"""Tests for feature group toggling (wired from config through FeatureEngineer)."""

import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import FeatureEngineer
from src.config import FEATURE_GROUPS


def _make_minimal_data(weeks=30):
    """Build minimal data that FeatureEngineer.create_all_features can consume."""
    dates = pd.date_range("2023-01-07", periods=weeks, freq="7D")
    rows = []
    for d in dates:
        rows.append({
            "date": d,
            "location": "01",
            "location_name": "Alabama",
            "value": 50.0 + np.random.default_rng(42).normal(0, 10),
            "weekly_rate": 1.0,
        })
    return pd.DataFrame(rows)


class TestFeatureToggle:

    def test_no_groups_enabled_creates_all_features(self):
        """When groups_enabled is None, all features are created (default)."""
        data = _make_minimal_data()
        eng = FeatureEngineer(groups_enabled=None)
        df = eng.create_all_features(data)
        cols = eng.get_feature_columns(df)
        # Should have lag features
        assert any(c.startswith("value_lag_") for c in cols)

    def test_disable_lag_removes_lag_columns(self):
        """Disabling the 'lag' group should remove value_lag_* columns."""
        data = _make_minimal_data()
        groups = {name: True for name in FEATURE_GROUPS}
        groups["lag"] = False

        eng = FeatureEngineer(groups_enabled=groups)
        df = eng.create_all_features(data)
        cols = eng.get_feature_columns(df)
        lag_cols = [c for c in cols if c.startswith("value_lag_")]
        assert len(lag_cols) == 0, f"Expected no lag columns, got: {lag_cols}"

    def test_disable_rolling_removes_rolling_columns(self):
        """Disabling the 'rolling' group should remove value_rolling_* columns."""
        data = _make_minimal_data()
        groups = {name: True for name in FEATURE_GROUPS}
        groups["rolling"] = False

        eng = FeatureEngineer(groups_enabled=groups)
        df = eng.create_all_features(data)
        cols = eng.get_feature_columns(df)
        rolling_cols = [c for c in cols if c.startswith("value_rolling_")]
        assert len(rolling_cols) == 0, f"Expected no rolling columns, got: {rolling_cols}"

    def test_disable_national_context_removes_us_columns(self):
        """Disabling 'national_context' should remove us_total_* and us_lag_* columns."""
        data = _make_minimal_data()
        groups = {name: True for name in FEATURE_GROUPS}
        groups["national_context"] = False

        eng = FeatureEngineer(groups_enabled=groups)
        df = eng.create_all_features(data)
        cols = eng.get_feature_columns(df)
        us_cols = [c for c in cols if c.startswith("us_total_") or c.startswith("us_lag_")]
        assert len(us_cols) == 0, f"Expected no US-level columns, got: {us_cols}"

    def test_all_enabled_matches_default(self):
        """All groups enabled should produce the same features as None."""
        data = _make_minimal_data()

        eng_default = FeatureEngineer(groups_enabled=None)
        df_default = eng_default.create_all_features(data)

        groups = {name: True for name in FEATURE_GROUPS}
        eng_all = FeatureEngineer(groups_enabled=groups)
        df_all = eng_all.create_all_features(data)

        assert set(df_default.columns) == set(df_all.columns)
