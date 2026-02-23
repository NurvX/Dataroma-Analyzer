#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - CSV Validation Ledger System

Comprehensive validation framework for CSV output files.
Implements invariant checking, schema validation, numeric reconciliation,
and cross-file contradiction detection.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity levels for validation violations."""

    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"


@dataclass
class ValidationViolation:
    """Represents a single validation violation."""

    file_path: str
    severity: Severity
    invariant_type: str
    violation_description: str
    sample_tickers: List[str] = field(default_factory=list)
    sample_values: str = ""
    suspected_code_path: str = ""
    remediation_proposal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "file_path": self.file_path,
            "severity": self.severity.value,
            "invariant_type": self.invariant_type,
            "violation_description": self.violation_description,
            "sample_tickers": ", ".join(self.sample_tickers[:3]),
            "sample_values": self.sample_values,
            "suspected_code_path": self.suspected_code_path,
            "remediation_proposal": self.remediation_proposal,
        }


class CSVValidator:
    """
    Comprehensive CSV validation framework.

    Validates schema, periods, numeric reconciliation, boolean flags,
    and preview list integrity across all analysis CSV files.
    """

    # Required metadata columns for all analysis files
    REQUIRED_METADATA_COLS = ["_analysis_type", "_generated"]

    # Optional metadata columns (context-dependent)
    OPTIONAL_METADATA_COLS = ["_periods", "_window_quarters"]

    # Column naming convention patterns
    PCT_PATTERN = re.compile(r".*_pct$|.*_percent$|.*_percentage$")
    RATIO_PATTERN = re.compile(r".*_ratio$")

    # Preview column patterns (columns that show truncated lists)
    PREVIEW_PATTERNS = [
        r"^top_managers$",
        r"^buying_managers$",
        r"^selling_managers$",
        r"^active_managers$",
    ]

    # Period format patterns
    PERIOD_PATTERN = re.compile(r"^Q[1-4]\s+\d{4}$")

    # Stale threshold: number of quarters before data is considered stale
    STALE_QUARTERS_THRESHOLD = 3

    # Filter criterion columns - expected to be 100% True by design
    # These columns exist because the file filters FOR that criterion
    FILTER_CRITERION_COLUMNS: Dict[str, List[str]] = {
        "52_week_high_sells.csv": ["near_52w_high"],
        "52_week_low_buys.csv": ["near_52w_low"],
        "stock_life_cycles.csv": ["currently_held"],
    }

    # Files exempt from period validation (contain intentional multi-year historical data)
    PERIOD_EXEMPT_FILES: Set[str] = {
        "position_building_timeline.csv",  # 20+ years of history (advanced/)
        "sector_rotation_patterns.csv",  # multi-year sector data (advanced/)
        "quarterly_activity_timeline.csv",  # 18+ years (historical/)
        "stock_life_cycles.csv",  # full lifecycle (historical/)
    }

    # Columns that reference historical periods (should not be checked against _periods window)
    # These columns intentionally contain periods outside the recent analysis window
    HISTORICAL_PERIOD_COLS: Set[str] = {
        "first_buy_period",  # first ever buy, can be decades ago
        "last_buy_period",  # can reference any historical period
        "latest_period",  # can reference periods outside recent window for advanced analysis
        "flip_quarter",  # position flip can be historical
    }

    def __init__(self, analysis_dir: str = "analysis") -> None:
        """
        Initialize the CSV validator.

        Args:
            analysis_dir: Path to the analysis output directory
        """
        self.analysis_dir = Path(analysis_dir)
        self.violations: List[ValidationViolation] = []
        self._current_quarter: Optional[str] = None
        self._all_quarters: List[str] = []

    def _get_current_quarter(self) -> str:
        """Determine the current quarter based on today's date."""
        if self._current_quarter:
            return self._current_quarter

        today = datetime.now()
        quarter = (today.month - 1) // 3 + 1
        self._current_quarter = f"Q{quarter} {today.year}"
        return self._current_quarter

    def _parse_quarter(self, period: str) -> Tuple[int, int]:
        """
        Parse a quarter string into (year, quarter) tuple.

        Args:
            period: Quarter string like "Q3 2025"

        Returns:
            Tuple of (year, quarter_number)
        """
        match = self.PERIOD_PATTERN.match(str(period).strip())
        if not match:
            return (0, 0)

        parts = period.strip().split()
        quarter = int(parts[0][1])
        year = int(parts[1])
        return (year, quarter)

    def _quarters_between(self, period1: str, period2: str) -> int:
        """
        Calculate the number of quarters between two periods.

        Args:
            period1: Earlier period (e.g., "Q1 2024")
            period2: Later period (e.g., "Q3 2025")

        Returns:
            Number of quarters between the periods
        """
        y1, q1 = self._parse_quarter(period1)
        y2, q2 = self._parse_quarter(period2)

        if y1 == 0 or y2 == 0:
            return 0

        return (y2 - y1) * 4 + (q2 - q1)

    def _get_relative_path(self, file_path: Path) -> str:
        """Get path relative to analysis directory for cleaner output."""
        try:
            return str(file_path.relative_to(self.analysis_dir))
        except ValueError:
            return str(file_path)

    def _add_violation(
        self,
        file_path: Path,
        severity: Severity,
        invariant_type: str,
        description: str,
        sample_tickers: Optional[List[str]] = None,
        sample_values: str = "",
        code_path: str = "",
        remediation: str = "",
    ) -> None:
        """Add a validation violation to the list."""
        self.violations.append(
            ValidationViolation(
                file_path=self._get_relative_path(file_path),
                severity=severity,
                invariant_type=invariant_type,
                violation_description=description,
                sample_tickers=sample_tickers or [],
                sample_values=sample_values,
                suspected_code_path=code_path,
                remediation_proposal=remediation,
            )
        )

    def enumerate_csv_files(self) -> List[Path]:
        """
        Enumerate all CSV files in analysis directories.

        Returns:
            List of paths to CSV files in current/, advanced/, historical/
        """
        csv_files: List[Path] = []

        subdirs = ["current", "advanced", "historical"]

        for subdir in subdirs:
            subdir_path = self.analysis_dir / subdir
            if subdir_path.exists():
                csv_files.extend(subdir_path.glob("*.csv"))

        logger.info(f"Found {len(csv_files)} CSV files to validate")
        return csv_files

    def validate_schema(self, file_path: Path, df: pd.DataFrame) -> None:
        """
        Validate schema requirements for a CSV file.

        Checks:
        - Required metadata columns exist
        - Column naming conventions (pct vs ratio)
        - Preview columns have corresponding count columns
        """
        rel_path = self._get_relative_path(file_path)

        # Check for required metadata columns
        missing_required = [
            col for col in self.REQUIRED_METADATA_COLS if col not in df.columns
        ]

        if missing_required:
            self._add_violation(
                file_path,
                Severity.MAJOR,
                "schema_metadata",
                f"Missing required metadata columns: {missing_required}",
                code_path="lib/analysis/base_analyzer.py",
                remediation="Add metadata columns in analyzer's save method",
            )

        # Check column naming conventions - mixed pct/percent/percentage
        pct_cols = [col for col in df.columns if "percent" in col.lower()]
        ratio_cols = [col for col in df.columns if "ratio" in col.lower()]

        # Warn about inconsistent naming (prefer _pct suffix)
        inconsistent_pct = [
            col
            for col in pct_cols
            if not col.endswith("_pct") and not col.startswith("_")
        ]

        if inconsistent_pct:
            self._add_violation(
                file_path,
                Severity.MINOR,
                "schema_naming",
                f"Inconsistent percentage column naming: {inconsistent_pct}. Prefer _pct suffix.",
                code_path="lib/utils/csv_formatter.py",
                remediation="Standardize to _pct suffix in CSVFormatter.standardize_columns()",
            )

        # Check preview columns have managers_shown
        for pattern in self.PREVIEW_PATTERNS:
            matching_cols = [col for col in df.columns if re.match(pattern, col)]
            if matching_cols and "managers_shown" not in df.columns:
                self._add_violation(
                    file_path,
                    Severity.MINOR,
                    "schema_preview",
                    f"Preview column '{matching_cols[0]}' exists but 'managers_shown' is missing",
                    code_path="lib/analysis/holdings_analyzer.py",
                    remediation="Add managers_shown column when truncating manager lists",
                )

    def validate_periods(self, file_path: Path, df: pd.DataFrame) -> None:
        """
        Validate period/window consistency.

        Checks:
        - _periods metadata matches actual period content
        - Current files don't have stale periods
        - Period columns are within declared window

        Exemptions:
        - Files in PERIOD_EXEMPT_FILES are skipped entirely (multi-year historical data)
        - Columns in HISTORICAL_PERIOD_COLS are not checked against _periods window
        """
        rel_path = self._get_relative_path(file_path)
        is_current = "current" in rel_path
        file_name = file_path.name

        # Skip period validation entirely for exempt files (multi-year historical data)
        if file_name in self.PERIOD_EXEMPT_FILES:
            logger.debug(f"Skipping period validation for exempt file: {file_name}")
            return

        # Check _periods metadata vs actual data
        if "_periods" in df.columns:
            declared_periods_raw = df["_periods"].iloc[0] if len(df) > 0 else ""

            if pd.notna(declared_periods_raw):
                declared_periods_str = str(declared_periods_raw)
                declared_periods = [
                    p.strip() for p in declared_periods_str.split(",")
                ]

                # Check for period columns in data
                # Exclude columns that intentionally reference historical periods
                period_cols = [
                    col
                    for col in df.columns
                    if col in ["period", "periods", "latest_period", "last_buy_period"]
                    and col not in self.HISTORICAL_PERIOD_COLS
                ]

                for pcol in period_cols:
                    if pcol == "periods":
                        # This is a list column, extract all periods
                        all_data_periods: Set[str] = set()
                        for val in df[pcol].dropna():
                            for p in str(val).split(","):
                                stripped = p.strip()
                                if self.PERIOD_PATTERN.match(stripped):
                                    all_data_periods.add(stripped)

                        # Check if data periods are outside declared window
                        undeclared = all_data_periods - set(declared_periods)
                        if undeclared and len(undeclared) > 2:
                            sample_tickers = (
                                df["ticker"].head(3).tolist()
                                if "ticker" in df.columns
                                else []
                            )
                            self._add_violation(
                                file_path,
                                Severity.MAJOR,
                                "period_mismatch",
                                f"Data contains periods not in _periods metadata: {list(undeclared)[:3]}",
                                sample_tickers=sample_tickers,
                                sample_values=f"_periods: {declared_periods_str}",
                                code_path="lib/analysis/base_analyzer.py",
                                remediation="Update _periods metadata to include all data periods",
                            )
                    else:
                        # Single period column
                        data_periods = set(
                            df[pcol].dropna().apply(lambda x: str(x).strip()).unique()
                        )
                        valid_data_periods = {
                            p for p in data_periods if self.PERIOD_PATTERN.match(p)
                        }

                        if valid_data_periods:
                            undeclared = valid_data_periods - set(declared_periods)
                            if undeclared:
                                sample_tickers = (
                                    df["ticker"].head(3).tolist()
                                    if "ticker" in df.columns
                                    else []
                                )
                                self._add_violation(
                                    file_path,
                                    Severity.MINOR,
                                    "period_mismatch",
                                    f"Column '{pcol}' contains periods outside declared window: {undeclared}",
                                    sample_tickers=sample_tickers,
                                    code_path="lib/analysis/base_analyzer.py",
                                    remediation="Verify period filtering logic",
                                )

        # Check for stale periods in current/ files
        if is_current and "_window_quarters" in df.columns:
            window = df["_window_quarters"].iloc[0] if len(df) > 0 else None

            if pd.notna(window):
                try:
                    window_int = int(window)
                    current_q = self._get_current_quarter()

                    # Check period columns for staleness
                    if "period" in df.columns:
                        for idx, row in df.iterrows():
                            period_val = row.get("period")
                            if pd.notna(period_val) and self.PERIOD_PATTERN.match(
                                str(period_val).strip()
                            ):
                                quarters_ago = self._quarters_between(
                                    str(period_val).strip(), current_q
                                )
                                if quarters_ago > self.STALE_QUARTERS_THRESHOLD:
                                    ticker = row.get("ticker", "Unknown")
                                    self._add_violation(
                                        file_path,
                                        Severity.MAJOR,
                                        "period_stale",
                                        f"Current file contains stale data ({quarters_ago} quarters old)",
                                        sample_tickers=[str(ticker)],
                                        sample_values=f"period: {period_val}",
                                        code_path="lib/analysis/orchestrator.py",
                                        remediation="Filter out data older than window_quarters",
                                    )
                                    break  # Report once per file
                except (ValueError, TypeError):
                    pass

    def validate_numeric_reconciliation(
        self, file_path: Path, df: pd.DataFrame
    ) -> None:
        """
        Validate numeric field reconciliation.

        Checks:
        - net_flow = buy_actions - sell_actions (where applicable)
        - active_managers <= manager_count
        - managers_shown matches preview truncation
        """
        rel_path = self._get_relative_path(file_path)

        # Check net_activity = buy_add_actions - sell_reduce_actions
        has_net = "net_activity" in df.columns
        has_buy_add = "buy_add_actions" in df.columns
        has_sell_reduce = "sell_reduce_actions" in df.columns

        if has_net and has_buy_add and has_sell_reduce:
            mismatches = []
            for idx, row in df.iterrows():
                net = row.get("net_activity", 0)
                buy_add = row.get("buy_add_actions", 0)
                sell_reduce = row.get("sell_reduce_actions", 0)

                if pd.notna(net) and pd.notna(buy_add) and pd.notna(sell_reduce):
                    expected_net = buy_add - sell_reduce
                    if abs(net - expected_net) > 0.01:  # Allow small float tolerance
                        ticker = row.get("ticker", f"row_{idx}")
                        mismatches.append(
                            (
                                ticker,
                                f"net={net}, expected={expected_net} (buy_add={buy_add}, sell_reduce={sell_reduce})",
                            )
                        )

            if mismatches:
                sample_tickers = [m[0] for m in mismatches[:3]]
                sample_vals = "; ".join([m[1] for m in mismatches[:3]])
                self._add_violation(
                    file_path,
                    Severity.CRITICAL,
                    "numeric_reconciliation",
                    f"net_activity != buy_add_actions - sell_reduce_actions ({len(mismatches)} rows)",
                    sample_tickers=sample_tickers,
                    sample_values=sample_vals,
                    code_path="lib/analysis/advanced_analyzer.py",
                    remediation="Fix net_activity calculation formula",
                )

        # Check buy_count + add_count + reduce_count + sell_count consistency
        count_cols = ["buy_count", "add_count", "reduce_count", "sell_count"]
        if all(col in df.columns for col in count_cols):
            # If total_actions exists, verify it
            if "total_actions" in df.columns:
                mismatches = []
                for idx, row in df.iterrows():
                    total = row.get("total_actions", 0)
                    computed = sum(row.get(col, 0) or 0 for col in count_cols)

                    if pd.notna(total) and abs(total - computed) > 0.01:
                        ticker = row.get("ticker", f"row_{idx}")
                        mismatches.append((ticker, f"total={total}, computed={computed}"))

                if mismatches:
                    sample_tickers = [m[0] for m in mismatches[:3]]
                    self._add_violation(
                        file_path,
                        Severity.CRITICAL,
                        "numeric_reconciliation",
                        f"total_actions != sum of action counts ({len(mismatches)} rows)",
                        sample_tickers=sample_tickers,
                        sample_values="; ".join([m[1] for m in mismatches[:3]]),
                        code_path="lib/analysis/historical_analyzer.py",
                        remediation="Fix total_actions aggregation",
                    )

        # Check active_managers <= manager_count
        if "active_managers" in df.columns and "manager_count" in df.columns:
            violations = df[
                (df["active_managers"] > df["manager_count"])
                & df["active_managers"].notna()
                & df["manager_count"].notna()
            ]

            if len(violations) > 0:
                sample_tickers = (
                    violations["ticker"].head(3).tolist()
                    if "ticker" in violations.columns
                    else []
                )
                self._add_violation(
                    file_path,
                    Severity.CRITICAL,
                    "numeric_constraint",
                    f"active_managers > manager_count ({len(violations)} rows)",
                    sample_tickers=sample_tickers,
                    code_path="lib/analysis/holdings_analyzer.py",
                    remediation="Verify active_managers is subset of all managers",
                )

        # Check managers_shown matches preview truncation
        if "managers_shown" in df.columns and "top_managers" in df.columns:
            for idx, row in df.iterrows():
                shown = row.get("managers_shown")
                top_mgrs = row.get("top_managers", "")

                if pd.notna(shown) and pd.notna(top_mgrs) and top_mgrs:
                    # Count managers in preview (handle "+N more" suffix)
                    mgr_list = str(top_mgrs).split(",")
                    actual_shown = len(
                        [m for m in mgr_list if not m.strip().startswith("+")]
                    )

                    if int(shown) != actual_shown:
                        ticker = row.get("ticker", f"row_{idx}")
                        self._add_violation(
                            file_path,
                            Severity.MINOR,
                            "preview_count",
                            f"managers_shown ({shown}) != actual preview count ({actual_shown})",
                            sample_tickers=[str(ticker)],
                            sample_values=f"top_managers: {top_mgrs[:50]}...",
                            code_path="lib/utils/csv_formatter.py",
                            remediation="Sync managers_shown with format_manager_list()",
                        )
                        break  # Report once per file

    def validate_boolean_flags(self, file_path: Path, df: pd.DataFrame) -> None:
        """
        Validate boolean flag integrity.

        Checks:
        - near_52w_low and near_52w_high are mutually exclusive
        - Flags are not degenerate (>90% same value)

        Exemptions:
        - Filter criterion columns that are expected to be 100% True by design
        - Integer columns are not treated as boolean (avoids false positives)
        """
        rel_path = self._get_relative_path(file_path)
        filename = file_path.name

        # Get filter criterion columns for this file (expected 100% True)
        filter_criterion_cols = self.FILTER_CRITERION_COLUMNS.get(filename, [])

        # Check mutual exclusivity of 52-week flags
        if "near_52w_low" in df.columns and "near_52w_high" in df.columns:
            # Convert to boolean
            low_flags = df["near_52w_low"].map(
                lambda x: x in [True, "True", 1, "1"] if pd.notna(x) else False
            )
            high_flags = df["near_52w_high"].map(
                lambda x: x in [True, "True", 1, "1"] if pd.notna(x) else False
            )

            both_true = (low_flags & high_flags).sum()
            if both_true > 0:
                # Find sample tickers
                both_mask = low_flags & high_flags
                sample_tickers = (
                    df.loc[both_mask, "ticker"].head(3).tolist()
                    if "ticker" in df.columns
                    else []
                )
                self._add_violation(
                    file_path,
                    Severity.CRITICAL,
                    "boolean_mutual_exclusion",
                    f"near_52w_low AND near_52w_high both True ({both_true} rows)",
                    sample_tickers=sample_tickers,
                    code_path="lib/utils/calculations.py",
                    remediation="Fix 52-week position thresholds to be mutually exclusive",
                )

        # Check for degenerate boolean flags
        for col in df.columns:
            if col.startswith("_"):
                continue

            # Skip filter criterion columns - they are expected to be 100% True
            if col in filter_criterion_cols:
                continue

            # Only check actual boolean columns (dtype == bool)
            # This avoids false positives on integer columns that happen to be all non-zero
            if df[col].dtype != bool:
                continue

            unique_vals = df[col].dropna().unique()

            if len(unique_vals) > 0 and len(df) >= 10:
                # Check distribution of True/False values
                true_count = df[col].sum()
                true_pct = (true_count / len(df)) * 100

                # Flag if >90% True or <10% True (essentially useless filter)
                if true_pct > 90:
                    self._add_violation(
                        file_path,
                        Severity.MINOR,
                        "boolean_degenerate",
                        f"Column '{col}' is {true_pct:.1f}% True (provides no filtering value)",
                        remediation="Adjust threshold or remove column if not useful",
                    )
                elif true_pct < 10 and true_count > 0:
                    self._add_violation(
                        file_path,
                        Severity.MINOR,
                        "boolean_degenerate",
                        f"Column '{col}' is only {true_pct:.1f}% True (may be too restrictive)",
                        remediation="Review threshold calibration",
                    )

    def validate_preview_lists(self, file_path: Path, df: pd.DataFrame) -> None:
        """
        Validate preview list column integrity.

        Checks:
        - Preview columns have associated count columns
        - Preview count matches declared limit
        """
        rel_path = self._get_relative_path(file_path)

        # Find preview-style columns
        preview_cols = []
        for col in df.columns:
            for pattern in self.PREVIEW_PATTERNS:
                if re.match(pattern, col):
                    preview_cols.append(col)

        for preview_col in preview_cols:
            # Check for "+N more" pattern consistency
            has_more_pattern = False
            total_with_more = 0

            for idx, row in df.iterrows():
                val = row.get(preview_col, "")
                if pd.notna(val) and "+more" in str(val).lower().replace(" ", ""):
                    total_with_more += 1
                    has_more_pattern = True

            # If we have truncated lists, managers_shown should exist
            if has_more_pattern and "managers_shown" not in df.columns:
                self._add_violation(
                    file_path,
                    Severity.MINOR,
                    "preview_metadata",
                    f"Column '{preview_col}' has truncated values but no 'managers_shown' column",
                    code_path="lib/analysis/holdings_analyzer.py",
                    remediation="Add managers_shown to track preview truncation",
                )

            # Check if manager_count exists and is consistent
            if has_more_pattern and "manager_count" in df.columns:
                inconsistent = []
                for idx, row in df.iterrows():
                    val = str(row.get(preview_col, ""))
                    mgr_count = row.get("manager_count")

                    if "+more" in val.lower().replace(" ", "") and pd.notna(mgr_count):
                        # Extract the "+N more" number
                        match = re.search(r"\+(\d+)\s*more", val, re.IGNORECASE)
                        if match:
                            more_count = int(match.group(1))
                            shown_count = val.count(",")  # Rough estimate
                            expected_total = shown_count + more_count + 1

                            if abs(expected_total - mgr_count) > 1:
                                ticker = row.get("ticker", f"row_{idx}")
                                inconsistent.append(
                                    (ticker, f"shown+more={expected_total}, manager_count={mgr_count}")
                                )

                if inconsistent:
                    self._add_violation(
                        file_path,
                        Severity.MINOR,
                        "preview_count_mismatch",
                        f"Preview truncation math inconsistent ({len(inconsistent)} rows)",
                        sample_tickers=[i[0] for i in inconsistent[:3]],
                        sample_values="; ".join([i[1] for i in inconsistent[:3]]),
                        code_path="lib/utils/csv_formatter.py",
                        remediation="Verify format_manager_list() count logic",
                    )

    def validate_file(self, file_path: Path) -> int:
        """
        Run all validations on a single file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Number of violations found
        """
        initial_violations = len(self.violations)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self._add_violation(
                file_path,
                Severity.CRITICAL,
                "file_read",
                f"Failed to read CSV: {e}",
                remediation="Check file encoding and format",
            )
            return 1

        if df.empty:
            # Empty files are allowed but noted
            logger.debug(f"Empty file: {file_path.name}")
            return 0

        # Run all validations
        self.validate_schema(file_path, df)
        self.validate_periods(file_path, df)
        self.validate_numeric_reconciliation(file_path, df)
        self.validate_boolean_flags(file_path, df)
        self.validate_preview_lists(file_path, df)

        return len(self.violations) - initial_violations

    def validate_cross_file(self) -> None:
        """
        Validate cross-file contradictions and consistency.

        Checks:
        - Same ticker with contradictory flags across files
        - Period window consistency across related files
        """
        # Load relevant files for cross-checking
        current_dir = self.analysis_dir / "current"

        # Check 52-week files for contradictory positions
        low_file = current_dir / "52_week_low_buys.csv"
        high_file = current_dir / "52_week_high_sells.csv"

        if low_file.exists() and high_file.exists():
            try:
                low_df = pd.read_csv(low_file)
                high_df = pd.read_csv(high_file)

                if "ticker" in low_df.columns and "ticker" in high_df.columns:
                    overlap = set(low_df["ticker"]) & set(high_df["ticker"])

                    if overlap:
                        # Check if any have contradictory 52-week positions
                        for ticker in list(overlap)[:5]:
                            low_row = low_df[low_df["ticker"] == ticker].iloc[0]
                            high_row = high_df[high_df["ticker"] == ticker].iloc[0]

                            low_pos = low_row.get("52_week_position_pct", 50)
                            high_pos = high_row.get("52_week_position_pct", 50)

                            # Same ticker should have same position in both files
                            if pd.notna(low_pos) and pd.notna(high_pos):
                                if abs(float(low_pos) - float(high_pos)) > 1:
                                    self._add_violation(
                                        low_file,
                                        Severity.MINOR,
                                        "cross_file_position",
                                        f"Ticker {ticker} has different 52w position across files",
                                        sample_tickers=[ticker],
                                        sample_values=f"low_buys: {low_pos}%, high_sells: {high_pos}%",
                                        code_path="lib/analysis/price_analyzer.py",
                                        remediation="Ensure consistent price data source",
                                    )
                                    break
            except Exception as e:
                logger.warning(f"Error in cross-file validation: {e}")

        # Check _window_quarters consistency across current/ files
        current_windows: Dict[Path, int] = {}

        if current_dir.exists():
            for csv_file in current_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, nrows=1)
                    if "_window_quarters" in df.columns:
                        window = df["_window_quarters"].iloc[0]
                        if pd.notna(window):
                            current_windows[csv_file] = int(window)
                except Exception:
                    continue

            if current_windows:
                unique_windows = set(current_windows.values())
                if len(unique_windows) > 1:
                    files_by_window: Dict[int, List[str]] = {}
                    for f, w in current_windows.items():
                        files_by_window.setdefault(w, []).append(f.name)

                    self._add_violation(
                        current_dir / "(multiple files)",
                        Severity.MINOR,
                        "cross_file_window",
                        f"Inconsistent _window_quarters in current/ files: {dict(files_by_window)}",
                        code_path="lib/analysis/orchestrator.py",
                        remediation="Use consistent window across current/ analyses",
                    )

    def validate_all(self) -> pd.DataFrame:
        """
        Run all validations and return the validation ledger.

        Returns:
            DataFrame containing all validation violations
        """
        self.violations = []  # Reset

        # Enumerate and validate all files
        csv_files = self.enumerate_csv_files()

        for file_path in csv_files:
            logger.debug(f"Validating: {file_path.name}")
            self.validate_file(file_path)

        # Run cross-file validations
        self.validate_cross_file()

        # Create ledger DataFrame
        if self.violations:
            ledger = pd.DataFrame([v.to_dict() for v in self.violations])
        else:
            ledger = pd.DataFrame(
                columns=[
                    "file_path",
                    "severity",
                    "invariant_type",
                    "violation_description",
                    "sample_tickers",
                    "sample_values",
                    "suspected_code_path",
                    "remediation_proposal",
                ]
            )

        return ledger

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results.

        Returns:
            Dictionary with summary statistics
        """
        if not self.violations:
            self.validate_all()

        critical_count = sum(
            1 for v in self.violations if v.severity == Severity.CRITICAL
        )
        major_count = sum(1 for v in self.violations if v.severity == Severity.MAJOR)
        minor_count = sum(1 for v in self.violations if v.severity == Severity.MINOR)

        # Count files with violations
        files_with_violations = set(v.file_path for v in self.violations)
        total_files = len(self.enumerate_csv_files())
        passing_files = total_files - len(files_with_violations)

        # Group by invariant type
        by_type: Dict[str, int] = {}
        for v in self.violations:
            by_type[v.invariant_type] = by_type.get(v.invariant_type, 0) + 1

        return {
            "total_files_scanned": total_files,
            "files_passing": passing_files,
            "files_with_violations": len(files_with_violations),
            "total_violations": len(self.violations),
            "critical_count": critical_count,
            "major_count": major_count,
            "minor_count": minor_count,
            "by_invariant_type": by_type,
            "pass_rate": f"{(passing_files / total_files * 100):.1f}%"
            if total_files > 0
            else "N/A",
        }

    def print_summary(self) -> None:
        """Print a formatted summary report to console."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("CSV VALIDATION SUMMARY")
        print("=" * 70)

        print(f"\nFiles Scanned:     {summary['total_files_scanned']}")
        print(f"Files Passing:     {summary['files_passing']}")
        print(f"Files with Issues: {summary['files_with_violations']}")
        print(f"Pass Rate:         {summary['pass_rate']}")

        print("\n" + "-" * 40)
        print("VIOLATIONS BY SEVERITY")
        print("-" * 40)
        print(f"  Critical: {summary['critical_count']}")
        print(f"  Major:    {summary['major_count']}")
        print(f"  Minor:    {summary['minor_count']}")
        print(f"  Total:    {summary['total_violations']}")

        if summary["by_invariant_type"]:
            print("\n" + "-" * 40)
            print("VIOLATIONS BY TYPE")
            print("-" * 40)
            for inv_type, count in sorted(
                summary["by_invariant_type"].items(), key=lambda x: -x[1]
            ):
                print(f"  {inv_type}: {count}")

        # Show cross-file contradictions if any
        cross_file_violations = [
            v for v in self.violations if "cross_file" in v.invariant_type
        ]
        if cross_file_violations:
            print("\n" + "-" * 40)
            print("CROSS-FILE CONTRADICTIONS")
            print("-" * 40)
            for v in cross_file_violations:
                print(f"  - {v.violation_description}")

        print("\n" + "=" * 70)

        # Final status
        if summary["critical_count"] > 0:
            print("STATUS: FAILED (Critical issues found)")
        elif summary["major_count"] > 0:
            print("STATUS: WARNING (Major issues found)")
        elif summary["minor_count"] > 0:
            print("STATUS: PASSED with minor issues")
        else:
            print("STATUS: PASSED (All checks passed)")

        print("=" * 70 + "\n")


def main() -> int:
    """Main entry point for the CSV validator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate CSV files in the analysis directory"
    )
    parser.add_argument(
        "--analysis-dir",
        "-d",
        default="analysis",
        help="Path to analysis directory (default: analysis)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only show summary, no file-by-file output"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run validation
    validator = CSVValidator(analysis_dir=args.analysis_dir)
    validator.validate_all()

    # Print summary
    validator.print_summary()

    # Return non-zero exit code if critical issues found
    summary = validator.get_summary()
    if summary["critical_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
