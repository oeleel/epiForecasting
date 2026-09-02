"""Data Quality Agent — pre-training checks on raw CDC FluSight data.

Runs before the improvement loop to catch issues that would silently
degrade forecast quality: missing weeks, outliers, reporting gaps,
sudden jumps that suggest data revisions or artifacts.

The agent has two modes:
    1. Deterministic checks (always run, no LLM needed)
    2. LLM summary (optional, interprets the findings in context)

Usage:
    python -m agent check-data --cutoff-date 2024-11-02
    python -m agent check-data --cutoff-date 2024-11-02 --dry-run  # no LLM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.adapters.flu_forecast import FluForecastAdapter


# ---------------------------------------------------------------------------
# Check result types
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    """A single data quality issue found by a check."""
    check: str          # which check found it
    severity: str       # "info" | "warning" | "critical"
    location: Optional[str] = None
    date: Optional[str] = None
    detail: str = ""


@dataclass
class DataQualityReport:
    """Aggregated output of all checks."""
    cutoff_date: str
    n_locations: int
    date_range: Dict[str, str]
    n_rows: int
    issues: List[Issue] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cutoff_date": self.cutoff_date,
            "n_locations": self.n_locations,
            "date_range": self.date_range,
            "n_rows": self.n_rows,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "check": i.check,
                    "severity": i.severity,
                    "location": i.location,
                    "date": i.date,
                    "detail": i.detail,
                }
                for i in self.issues
            ],
            "summary_stats": self.summary_stats,
        }


# ---------------------------------------------------------------------------
# Data Quality Checker (deterministic — no LLM)
# ---------------------------------------------------------------------------

class DataQualityChecker:
    """Runs deterministic quality checks on raw CDC hospitalization data.

    Each check_* method appends Issues to self.issues. The run() method
    orchestrates all checks and returns a DataQualityReport.
    """

    # Weeks of data we expect per location (since CDC started reporting).
    # If a location has fewer than this fraction of the expected weeks,
    # flag it.
    COVERAGE_THRESHOLD = 0.90

    # A week-over-week change larger than this multiple of the location's
    # standard deviation is flagged as a potential outlier/revision.
    SPIKE_ZSCORE = 4.0

    # Minimum admissions value — values below this in non-summer months
    # during active seasons are suspicious (potential reporting lag).
    MIN_EXPECTED_VALUE_ACTIVE = 0.0  # 0 itself is suspicious during season

    # Recent weeks to check more carefully (relative to cutoff)
    RECENT_WINDOW_WEEKS = 8

    def __init__(self, cutoff_date: str):
        self.cutoff_date = cutoff_date
        self.cutoff_dt = pd.to_datetime(cutoff_date)
        self.issues: List[Issue] = []

    def run(self, data: pd.DataFrame) -> DataQualityReport:
        """Run all checks and return a report."""
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Only check data up to cutoff
        df = df[df["date"] <= self.cutoff_dt].copy()

        if df.empty:
            report = DataQualityReport(
                cutoff_date=self.cutoff_date,
                n_locations=0,
                date_range={"min": "", "max": ""},
                n_rows=0,
            )
            report.issues.append(Issue(
                check="empty_data",
                severity="critical",
                detail=f"No data found at or before cutoff {self.cutoff_date}",
            ))
            return report

        self.issues = []

        # Run all checks
        self.check_missing_weeks(df)
        self.check_missing_values(df)
        self.check_spikes(df)
        self.check_zero_reporting(df)
        self.check_recent_completeness(df)
        self.check_location_coverage(df)

        # Summary stats
        summary_stats = self._compute_summary(df)

        return DataQualityReport(
            cutoff_date=self.cutoff_date,
            n_locations=df["location"].nunique(),
            date_range={
                "min": str(df["date"].min().date()),
                "max": str(df["date"].max().date()),
            },
            n_rows=len(df),
            issues=list(self.issues),
            summary_stats=summary_stats,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_missing_weeks(self, df: pd.DataFrame) -> None:
        """Flag locations that have gaps in their weekly time series."""
        for loc, group in df.groupby("location"):
            dates = group["date"].sort_values()
            diffs = dates.diff().dt.days.dropna()
            # Normal cadence is 7 days. Anything > 10 is a gap.
            gaps = diffs[diffs > 10]
            for idx in gaps.index:
                gap_start = dates.loc[idx - 1] if (idx - 1) in dates.index else dates.iloc[0]
                gap_end = dates.loc[idx]
                gap_weeks = int(diffs.loc[idx] / 7)
                loc_name = group["location_name"].iloc[0] if "location_name" in group.columns else loc
                self.issues.append(Issue(
                    check="missing_weeks",
                    severity="warning" if gap_weeks <= 2 else "critical",
                    location=str(loc_name),
                    date=str(gap_start.date()),
                    detail=f"{gap_weeks}-week gap ({gap_start.date()} to {gap_end.date()})",
                ))

    def check_missing_values(self, df: pd.DataFrame) -> None:
        """Flag locations that have null hospitalization values."""
        nulls = df[df["value"].isna()]
        if nulls.empty:
            return

        for loc, group in nulls.groupby("location"):
            loc_name = group["location_name"].iloc[0] if "location_name" in group.columns else loc
            dates = sorted(group["date"])
            self.issues.append(Issue(
                check="missing_value",
                severity="warning" if len(group) <= 4 else "critical",
                location=str(loc_name),
                date=str(dates[0].date()),
                detail=f"{len(group)} null values "
                       f"({dates[0].date()} to {dates[-1].date()})",
            ))

    def check_spikes(self, df: pd.DataFrame) -> None:
        """Flag week-over-week changes that are statistical outliers.

        Large spikes can indicate data revisions, backfill dumps, or
        reporting artifacts rather than real epidemiological signal.
        """
        for loc, group in df.groupby("location"):
            group = group.sort_values("date")
            values = group["value"].dropna()
            if len(values) < 10:
                continue

            diff = values.diff().dropna()
            if diff.std() == 0:
                continue

            zscore = (diff - diff.mean()) / diff.std()
            outliers = zscore[zscore.abs() > self.SPIKE_ZSCORE]

            loc_name = group["location_name"].iloc[0] if "location_name" in group.columns else loc
            sorted_idx = values.index.tolist()
            for idx in outliers.index:
                row = group.loc[idx]
                pos = sorted_idx.index(idx)
                prev_val = values.iloc[pos - 1] if pos > 0 else None
                cur_val = values.loc[idx]
                if prev_val is not None:
                    change = cur_val - prev_val
                    detail = (f"Value jumped {change:+.0f} (z={zscore.loc[idx]:.1f}), "
                              f"{prev_val:.0f} -> {cur_val:.0f}")
                else:
                    detail = f"Value={cur_val:.0f} (z={zscore.loc[idx]:.1f})"
                self.issues.append(Issue(
                    check="spike",
                    severity="warning",
                    location=str(loc_name),
                    date=str(row["date"].date()),
                    detail=detail,
                ))

    def check_zero_reporting(self, df: pd.DataFrame) -> None:
        """Flag locations reporting zero during active flu months (Oct-Apr)."""
        active_months = {10, 11, 12, 1, 2, 3, 4}
        active = df[df["date"].dt.month.isin(active_months)]
        zeros = active[(active["value"] == 0) | (active["value"].isna())]

        # Group consecutive zeros by location to avoid flooding with issues
        for loc, group in zeros.groupby("location"):
            if len(group) < 2:
                continue
            loc_name = group["location_name"].iloc[0] if "location_name" in group.columns else loc
            dates = sorted(group["date"])
            self.issues.append(Issue(
                check="zero_reporting",
                severity="warning",
                location=str(loc_name),
                date=str(dates[0].date()),
                detail=f"{len(group)} zero/null values during active months "
                       f"({dates[0].date()} to {dates[-1].date()})",
            ))

    def check_recent_completeness(self, df: pd.DataFrame) -> None:
        """Check that the most recent weeks before cutoff have data for all locations."""
        recent_start = self.cutoff_dt - pd.Timedelta(7 * (self.RECENT_WINDOW_WEEKS), unit="D")
        recent = df[df["date"] >= recent_start]

        if recent.empty:
            self.issues.append(Issue(
                check="recent_completeness",
                severity="critical",
                detail=f"No data in the {self.RECENT_WINDOW_WEEKS} weeks before cutoff",
            ))
            return

        all_locations = set(df["location"].unique())
        recent_locations = set(recent["location"].unique())
        missing = all_locations - recent_locations

        if missing:
            # Map FIPS to names
            name_map = dict(zip(df["location"].astype(str), df["location_name"]))
            missing_names = [name_map.get(str(loc), str(loc)) for loc in sorted(missing)]
            self.issues.append(Issue(
                check="recent_completeness",
                severity="critical",
                detail=f"{len(missing)} locations have no data in the last "
                       f"{self.RECENT_WINDOW_WEEKS} weeks: {', '.join(missing_names[:10])}"
                       f"{'...' if len(missing_names) > 10 else ''}",
            ))

        # Check for the most recent week specifically
        max_date = recent["date"].max()
        latest_week = recent[recent["date"] == max_date]
        latest_locations = set(latest_week["location"].unique())
        missing_latest = all_locations - latest_locations

        if missing_latest and len(missing_latest) > 3:
            self.issues.append(Issue(
                check="recent_completeness",
                severity="warning",
                detail=f"{len(missing_latest)} locations missing from the latest "
                       f"reporting week ({max_date.date()})",
            ))

    def check_location_coverage(self, df: pd.DataFrame) -> None:
        """Flag locations with significantly fewer data points than expected."""
        weeks_per_loc = df.groupby("location")["date"].nunique()
        expected = weeks_per_loc.max()

        low_coverage = weeks_per_loc[weeks_per_loc < expected * self.COVERAGE_THRESHOLD]
        if low_coverage.empty:
            return

        name_map = dict(zip(df["location"].astype(str), df["location_name"]))
        for loc, n_weeks in low_coverage.items():
            pct = n_weeks / expected * 100
            loc_name = name_map.get(str(loc), str(loc))
            self.issues.append(Issue(
                check="location_coverage",
                severity="warning" if pct > 75 else "critical",
                location=str(loc_name),
                detail=f"Only {n_weeks}/{expected} weeks of data ({pct:.0f}% coverage)",
            ))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute high-level summary statistics for the report."""
        recent_start = self.cutoff_dt - pd.Timedelta(7 * (self.RECENT_WINDOW_WEEKS), unit="D")
        recent = df[df["date"] >= recent_start]

        return {
            "total_rows": len(df),
            "n_locations": df["location"].nunique(),
            "n_weeks": df["date"].nunique(),
            "null_values": int(df["value"].isna().sum()),
            "zero_values": int((df["value"] == 0).sum()),
            "recent_mean": round(float(recent["value"].mean()), 1) if not recent.empty else None,
            "recent_median": round(float(recent["value"].median()), 1) if not recent.empty else None,
            "recent_max": round(float(recent["value"].max()), 1) if not recent.empty else None,
        }


# ---------------------------------------------------------------------------
# LLM prompt for interpreting findings
# ---------------------------------------------------------------------------

DATA_QUALITY_PROMPT = """\
{domain_context}

## Data Quality Check Results

**Cutoff date**: {cutoff_date}
**Data range**: {date_range}
**Locations**: {n_locations}
**Total rows**: {n_rows}

### Summary Statistics
{summary_stats}

### Issues Found
- Critical: {critical_count}
- Warnings: {warning_count}

{issues_block}

## Your Task

You are a data quality analyst reviewing CDC FluSight hospitalization data
before it is used to train a forecasting model. Based on the issues above:

1. **Assessment** (2-3 sentences): Is this data safe to train on? Are the
   issues likely to affect forecast quality?

2. **Key concerns** (bullet points): Which issues are most likely to cause
   problems during training or forecasting? Be specific about which
   locations or time periods are affected.

3. **Recommendations** (bullet points): What should the researcher do?
   Options include: proceed as-is, exclude specific locations, wait for
   data updates, or investigate specific anomalies.

Keep it concise and actionable. If the data looks clean, say so.
"""


def format_data_quality_prompt(report: DataQualityReport, domain_context: str) -> str:
    """Build the LLM prompt from a DataQualityReport."""
    # Summary stats block
    stats = report.summary_stats
    stats_lines = [f"- {k}: {v}" for k, v in stats.items()]
    stats_block = "\n".join(stats_lines)

    # Issues block
    if not report.issues:
        issues_block = "No issues found — data appears clean."
    else:
        issues_lines = []
        # Group by severity
        for severity in ["critical", "warning", "info"]:
            group = [i for i in report.issues if i.severity == severity]
            if not group:
                continue
            issues_lines.append(f"**{severity.upper()}:**")
            for issue in group[:15]:  # Cap to avoid huge prompts
                loc = f" [{issue.location}]" if issue.location else ""
                date = f" ({issue.date})" if issue.date else ""
                issues_lines.append(f"- [{issue.check}]{loc}{date}: {issue.detail}")
            if len(group) > 15:
                issues_lines.append(f"  ... and {len(group) - 15} more {severity} issues")
            issues_lines.append("")
        issues_block = "\n".join(issues_lines)

    return DATA_QUALITY_PROMPT.format(
        domain_context=domain_context,
        cutoff_date=report.cutoff_date,
        date_range=f"{report.date_range['min']} to {report.date_range['max']}",
        n_locations=report.n_locations,
        n_rows=report.n_rows,
        summary_stats=stats_block,
        critical_count=report.critical_count,
        warning_count=report.warning_count,
        issues_block=issues_block,
    )
