"""Tests for the Data Quality Agent (agent.data_quality)."""

import numpy as np
import pandas as pd
import pytest

from agent.data_quality import DataQualityChecker, DataQualityReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    locations=("01",),
    start="2023-01-07",
    weeks=20,
    base_value=100.0,
):
    """Build a minimal weekly hospitalization DataFrame."""
    rows = []
    dates = pd.date_range(start, periods=weeks, freq="7D")
    for loc in locations:
        for d in dates:
            rows.append({
                "date": d,
                "location": loc,
                "location_name": f"State_{loc}",
                "value": base_value + np.random.default_rng(42).normal(0, 5),
                "weekly_rate": 1.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataQualityChecker:

    def test_clean_data_no_issues(self):
        df = _make_data(weeks=20)
        checker = DataQualityChecker("2023-06-01")
        report = checker.run(df)
        # Clean synthetic data should have no critical issues
        assert report.critical_count == 0

    def test_empty_data_is_critical(self):
        df = _make_data(start="2025-01-01", weeks=5)
        checker = DataQualityChecker("2024-01-01")  # cutoff before data
        report = checker.run(df)
        assert report.critical_count == 1
        assert report.issues[0].check == "empty_data"

    def test_missing_weeks_detected(self):
        df = _make_data(weeks=10, start="2023-01-07")
        # Remove 3 weeks to create a gap
        df = df[~df["date"].isin(pd.to_datetime(["2023-02-11", "2023-02-18", "2023-02-25"]))]
        checker = DataQualityChecker("2023-04-01")
        report = checker.run(df)
        gap_issues = [i for i in report.issues if i.check == "missing_weeks"]
        assert len(gap_issues) >= 1
        assert "gap" in gap_issues[0].detail

    def test_null_values_detected(self):
        df = _make_data(weeks=10)
        df.loc[df.index[3], "value"] = np.nan
        df.loc[df.index[5], "value"] = np.nan
        checker = DataQualityChecker("2023-06-01")
        report = checker.run(df)
        null_issues = [i for i in report.issues if i.check == "missing_value"]
        assert len(null_issues) >= 1

    def test_spike_detected(self):
        # Use many weeks so the outlier doesn't dominate the std
        df = _make_data(weeks=100, base_value=50.0)
        # Inject a massive spike in the middle
        df.loc[df.index[50], "value"] = 5000.0
        checker = DataQualityChecker("2025-01-01")
        report = checker.run(df)
        spike_issues = [i for i in report.issues if i.check == "spike"]
        assert len(spike_issues) >= 1

    def test_recent_completeness_missing_locations(self):
        df = _make_data(locations=("01", "02"), weeks=20)
        # Remove location 02's recent data
        cutoff = "2023-05-20"
        cutoff_dt = pd.to_datetime(cutoff)
        recent_start = cutoff_dt - pd.Timedelta(7 * 8, unit="D")
        mask = (df["location"] == "02") & (df["date"] >= recent_start)
        df = df[~mask]
        checker = DataQualityChecker(cutoff)
        report = checker.run(df)
        recent_issues = [i for i in report.issues if i.check == "recent_completeness"]
        assert len(recent_issues) >= 1

    def test_report_to_dict_serializable(self):
        """Report.to_dict() must be JSON-serializable."""
        import json
        df = _make_data(weeks=10)
        checker = DataQualityChecker("2023-06-01")
        report = checker.run(df)
        d = report.to_dict()
        # Should not raise
        json.dumps(d, default=str)

    def test_summary_stats_populated(self):
        df = _make_data(weeks=20)
        checker = DataQualityChecker("2023-06-01")
        report = checker.run(df)
        assert "total_rows" in report.summary_stats
        assert "n_locations" in report.summary_stats
        assert report.summary_stats["n_locations"] == 1
