#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Base Analyzer

Abstract base class providing common analysis functionality and patterns
for all analysis modules.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
import pandas as pd
from typing import Dict, List, Optional

from ..data.data_loader import DataLoader
from ..utils.formatters import DataFormatter
from ..utils.calculations import FinancialCalculations, ScoringUtils
import re


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analysis modules.

    Provides common functionality and enforces consistent interface
    across all analyzer implementations.
    """

    def __init__(self, data_loader: DataLoader):
        """Initialize analyzer with data loader."""
        self.data = data_loader
        self.formatter = DataFormatter()
        self.calc = FinancialCalculations()
        self.scoring = ScoringUtils()

        if not self.data.data_loaded:
            raise ValueError("Data must be loaded before creating analyzer")

        self._recent_quarters_cache = None

    @abstractmethod
    def analyze(self) -> pd.DataFrame:
        """
        Perform the core analysis.

        Returns:
            DataFrame with analysis results
        """

    def get_analysis_name(self) -> str:
        """Return the analysis name for file naming."""
        return self.__class__.__name__.replace("Analyzer", "").lower()

    def get_analysis_title(self) -> str:
        """Return the analysis title for display."""
        name = self.get_analysis_name()
        return name.replace("_", " ").title()

    def get_recent_quarters(self, num_quarters: int = 3) -> List[str]:
        """
        Get the most recent quarters from the data.

        Args:
            num_quarters: Number of recent quarters to return (default: 3)

        Returns:
            List of quarter strings (e.g., ["Q1 2025", "Q4 2024", "Q3 2024"])

        Raises:
            ValueError: If num_quarters is less than 1
        """
        if num_quarters < 1:
            raise ValueError(f"num_quarters must be >= 1, got {num_quarters}")

        if self._recent_quarters_cache and len(self._recent_quarters_cache) >= num_quarters:
            return self._recent_quarters_cache[:num_quarters]

        if self.data.history_df is None or self.data.history_df.empty:
            logging.warning("No history data available to determine recent quarters")
            return []

        all_quarters = self.data.history_df["period"].dropna().unique()

        quarter_data = []
        for quarter in all_quarters:
            match = re.match(r"Q(\d) (\d{4})", quarter)
            if match:
                q_num = int(match.group(1))
                year = int(match.group(2))
                quarter_data.append((year, q_num, quarter))

        quarter_data.sort(key=lambda x: (x[0], x[1]), reverse=True)

        recent_quarters = [q[2] for q in quarter_data[:num_quarters]]

        self._recent_quarters_cache = recent_quarters

        logging.info(f"Determined recent {num_quarters} quarters: {recent_quarters}")
        return recent_quarters

    def filter_recent_activities(self, df: pd.DataFrame, num_quarters: int = 3) -> pd.DataFrame:
        """
        Filter DataFrame to only include activities from recent quarters.

        Args:
            df: DataFrame with 'period' column
            num_quarters: Number of recent quarters to include

        Returns:
            Filtered DataFrame
        """
        recent_quarters = self.get_recent_quarters(num_quarters)
        if not recent_quarters:
            return df

        return df[df["period"].isin(recent_quarters)]

    def format_output(self, df: pd.DataFrame, apply_precision: bool = True) -> pd.DataFrame:
        """Apply consistent formatting to output DataFrame."""
        if df.empty:
            return df

        if apply_precision:
            df = self.formatter.apply_precision_formatting(df)

        return df

    def prepare_for_export(self, df: pd.DataFrame, clean_names: bool = True) -> pd.DataFrame:
        """Prepare DataFrame for export (CSV/Excel)."""
        return self.formatter.prepare_for_export(df, clean_names=clean_names)

    def add_metadata_columns(
        self,
        df: pd.DataFrame,
        window_quarters: int,
        periods: Optional[List[str]] = None,
        analysis_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add metadata columns to output DataFrame for analysis traceability.

        These columns document the analysis window and parameters used,
        helping users understand and compare different analysis outputs.

        Args:
            df: DataFrame to add metadata to
            window_quarters: Number of quarters used in the analysis window
            periods: List of specific period strings (e.g., ["Q1 2025", "Q2 2025"])
                    If None, will be auto-determined from get_recent_quarters()
            analysis_type: Optional description of the analysis type

        Returns:
            DataFrame with metadata columns appended
        """
        if df is None:
            return df

        df = df.copy()

        # Get periods if not provided
        if periods is None:
            periods = self.get_recent_quarters(window_quarters)

        # For empty DataFrames, still add metadata columns (for validation compliance)
        if df.empty:
            # Add empty metadata columns with proper structure
            pass  # Fall through to add metadata to empty DataFrame

        # Add metadata columns with underscore prefix
        df["_window_quarters"] = window_quarters
        df["_periods"] = ", ".join(periods) if periods else ""
        df["_generated"] = datetime.now().strftime("%Y-%m-%d")

        if analysis_type:
            df["_analysis_type"] = analysis_type

        return df

    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame contains required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if all columns exist, False otherwise
        """
        if df is None or df.empty:
            logging.warning(f"DataFrame is empty for {self.__class__.__name__}")
            return False

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing required columns in {self.__class__.__name__}: {missing_columns}")
            return False

        return True

    def filter_active_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out inactive/historical holdings (those with 0 value)."""
        if df is None or df.empty:
            return df

        if "value" in df.columns:
            active_df = df[df["value"] > 0].copy()
            logging.debug(f"Filtered to {len(active_df)} active positions from {len(df)} total")
            return active_df

        return df

    def add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common calculated fields to DataFrame using vectorized operations."""
        import numpy as np

        if df is None or df.empty:
            return df

        df = df.copy()

        # 52-week position calculation (vectorized)
        required_52w_cols = ["current_price", "52_week_low", "52_week_high"]
        if all(col in df.columns for col in required_52w_cols):
            current = df["current_price"]
            low = df["52_week_low"]
            high = df["52_week_high"]

            # Calculate range, handle invalid ranges
            price_range = high - low
            valid_range = price_range > 0

            # Default to 50% for invalid ranges, calculate for valid
            df["52_week_position_pct"] = np.where(
                valid_range,
                np.clip((current - low) / price_range * 100, 0, 100),
                50.0
            )

        # Price change percentage (vectorized)
        if "current_price" in df.columns and "reported_price" in df.columns:
            reported = df["reported_price"]
            current = df["current_price"]

            # Handle division by zero
            df["price_change_pct"] = np.where(
                reported > 0,
                ((current - reported) / reported) * 100,
                0.0
            )

        # Position value calculation (vectorized)
        if "shares" in df.columns and "current_price" in df.columns:
            shares = df["shares"].fillna(0)
            price = df["current_price"].fillna(0)

            # Only calculate for positive values
            df["current_position_value"] = np.where(
                (shares > 0) & (price > 0),
                shares * price,
                0.0
            )

        return df

    def get_manager_summary(self, manager_ids: pd.Series) -> str:
        """Get formatted summary of managers."""
        return self.data.get_manager_list(manager_ids)

    def get_activity_summary(self, activities: pd.Series) -> str:
        """Get formatted summary of activities."""
        return self.data.get_activity_summary(activities)

    def log_analysis_summary(self, df: pd.DataFrame, analysis_name: Optional[str] = None) -> None:
        """Log summary of analysis results."""
        name = analysis_name or self.get_analysis_name()

        if df.empty:
            logging.warning(f"{name} analysis returned no results")
        else:
            logging.info(f"{name} analysis completed: {len(df)} results")

            if "manager_count" in df.columns:
                avg_managers = df["manager_count"].mean()
                logging.info(f"  Average managers per stock: {avg_managers:.1f}")

            if "total_value" in df.columns:
                total_value = df["total_value"].sum()
                logging.info(f"  Total value analyzed: ${total_value:,.0f}")


class MultiAnalyzer(BaseAnalyzer):
    """
    Base class for analyzers that produce multiple related analyses.
    """

    @abstractmethod
    def analyze_all(self) -> Dict[str, pd.DataFrame]:
        """
        Perform all related analyses.

        Returns:
            Dictionary mapping analysis names to DataFrames
        """

    def analyze(self) -> pd.DataFrame:
        """Default implementation returns the first analysis."""
        results = self.analyze_all()
        if results:
            return list(results.values())[0]
        return pd.DataFrame()

    def format_all_outputs(self, results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply formatting to all analysis results."""
        formatted_results = {}
        for name, df in results.items():
            formatted_results[name] = self.format_output(df)
        return formatted_results
