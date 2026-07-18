#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Formatters

General formatting utilities for numbers, percentages, and currency.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

"""
Data formatting utilities for analysis outputs.
"""

import logging
import pandas as pd
import re
from typing import List, Optional


class DataFormatter:
    """Utility class for consistent data formatting across all analysis modules."""

    # Define precision rules for numeric formatting
    PRECISION_RULES = {
        # Price and percentage fields
        "price": 2,
        "current_price": 2,
        "reported_price": 2,
        "avg_portfolio_pct": 2,
        "max_portfolio_pct": 2,
        "portfolio_pct_std": 2,
        "gain_loss_pct": 2,
        "portfolio_percent": 2,
        "conviction_score": 2,
        "appeal_score": 2,
        "value_score": 1,
        "consensus_score": 2,
        # Financial ratios
        "pe_ratio": 2,
        "debt_to_equity": 2,
        "roe": 2,
        "current_ratio": 2,
        "gross_margin": 2,
        "beta": 2,
        "eps": 4,
        # Large numbers (no decimals)
        "market_cap": 0,
        "total_shares": 0,
        "shares": 0,
        "total_position_value": 2,
        "total_value": 2,
        "value": 2,
        # Count fields
        "manager_count": 0,
        "position_count": 0,
        "num_managers": 0,
        "buy_count": 0,
        "sell_count": 0,
    }

    @classmethod
    def apply_precision_formatting(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply consistent numeric formatting to DataFrame columns.

        Args:
            df: DataFrame to format

        Returns:
            Formatted DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # Apply formatting based on precision rules
        for col, decimals in cls.PRECISION_RULES.items():
            if col in df.columns:
                try:
                    # Convert to numeric first
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    # Round to specified decimals
                    df[col] = df[col].round(decimals)
                except Exception as e:
                    logging.warning(f"Could not format column {col}: {e}")

        return df

    @classmethod
    def format_market_cap(cls, market_cap: float) -> str:
        """Format market cap for display."""
        if pd.isna(market_cap) or market_cap <= 0:
            return "N/A"
        elif market_cap >= 1_000_000_000_000:  # Trillion
            return f"${market_cap / 1_000_000_000_000:.1f}T"
        elif market_cap >= 1_000_000_000:  # Billion
            return f"${market_cap / 1_000_000_000:.1f}B"
        elif market_cap >= 1_000_000:  # Million
            return f"${market_cap / 1_000_000:.1f}M"
        else:
            return f"${market_cap:,.0f}"

    @classmethod
    def categorize_market_cap(cls, market_cap: float) -> str:
        """Categorize market cap into standard buckets."""
        if pd.isna(market_cap) or market_cap <= 0:
            return "Unknown"
        elif market_cap < 300_000_000:  # < $300M
            return "Micro-Cap"
        elif market_cap < 2_000_000_000:  # < $2B
            return "Small-Cap"
        elif market_cap < 10_000_000_000:  # < $10B
            return "Mid-Cap"
        elif market_cap < 200_000_000_000:  # < $200B
            return "Large-Cap"
        else:
            return "Mega-Cap"

    @classmethod
    def format_percentage(cls, value: float, decimals: int = 2) -> str:
        """Format value as percentage."""
        if pd.isna(value):
            return "N/A"
        return f"{value:.{decimals}f}%"

    @classmethod
    def format_currency(cls, value: float, decimals: int = 2) -> str:
        """Format value as currency."""
        if pd.isna(value):
            return "N/A"
        if value >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"${value / 1_000:.1f}K"
        else:
            return f"${value:.{decimals}f}"

    @classmethod
    def clean_column_names(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names for better readability."""
        df = df.copy()

        # Column name replacements
        replacements = {
            "ticker": "Ticker",
            "stock": "Company",
            "manager_name": "Manager",
            "manager_names": "Managers",
            "total_value": "Total Value ($)",
            "avg_portfolio_pct": "Avg Portfolio %",
            "max_portfolio_pct": "Max Portfolio %",
            "manager_count": "# Managers",
            "conviction_score": "Conviction Score",
            "appeal_score": "Appeal Score",
            "value_score": "Value Score",
            "current_price": "Current Price ($)",
            "reported_price": "Reported Price ($)",
            "shares": "Shares",
            "total_shares": "Total Shares",
            "market_cap": "Market Cap",
            "pe_ratio": "P/E Ratio",
        }

        # Apply replacements
        df.columns = [replacements.get(col, col.replace("_", " ").title()) for col in df.columns]

        return df

    @classmethod
    def prepare_for_export(cls, df: pd.DataFrame, clean_names: bool = True) -> pd.DataFrame:
        """Prepare DataFrame for CSV export with formatting and cleaned names."""
        if df.empty:
            return df

        # Clean up duplicate and JSON columns first
        df = cls.remove_duplicate_columns(df)
        df = cls.flatten_json_columns(df)

        # Apply precision formatting
        df = cls.apply_precision_formatting(df)

        # Clean column names if requested
        if clean_names:
            df = cls.clean_column_names(df)

        return df

    @classmethod
    def remove_duplicate_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate columns created by DataFrame merges.

        Columns with .1, .2 suffixes (e.g., manager.1, ticker.1) are duplicates
        created when merging DataFrames with same-named columns.
        """
        if df.empty:
            return df

        df = df.copy()

        # Find columns with .N suffix pattern
        duplicate_cols = [col for col in df.columns if re.match(r".*\.\d+$", col)]

        if duplicate_cols:
            logging.debug(f"Removing duplicate columns: {duplicate_cols}")
            df = df.drop(columns=duplicate_cols)

        return df

    @classmethod
    def flatten_json_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten columns containing JSON/dict data to readable formats.

        Converts dict columns to either:
        - Comma-separated key:value pairs for simple dicts
        - Separate columns for known dict structures
        """
        if df.empty:
            return df

        df = df.copy()

        # Known JSON columns and how to flatten them
        json_column_handlers = {
            "action_breakdown": cls._flatten_action_breakdown,
            "next_action_breakdown": cls._flatten_action_breakdown,
            "crisis_details": cls._flatten_to_string,
            "manager_details": cls._flatten_to_string,
            "new_managers": cls._flatten_list_to_string,
            "recent_actions": cls._flatten_list_to_string,
        }

        for col in df.columns:
            if col in json_column_handlers:
                handler = json_column_handlers[col]
                df = handler(df, col)
            elif cls._is_json_column(df[col]):
                # Generic handling for unknown JSON columns
                df = cls._flatten_to_string(df, col)

        return df

    @classmethod
    def _is_json_column(cls, series: pd.Series) -> bool:
        """Check if a column contains JSON-like data (dicts or lists)."""
        sample = series.dropna().head(5).astype(str)
        for val in sample:
            if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
                return True
        return False

    @classmethod
    def _flatten_action_breakdown(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Flatten action_breakdown dict to separate columns."""
        if col not in df.columns:
            return df

        # Extract action counts to separate columns.
        # The analyzers may have already stringified the breakdown
        # (e.g. "Add: 3, Buy: 1"), so handle both dict and string forms
        # instead of silently writing zeros for strings.
        action_types = ["Buy", "Add", "Reduce", "Sell"]

        def extract_count(value, action):
            if isinstance(value, dict):
                return value.get(action, 0)
            if isinstance(value, str):
                match = re.search(rf"\b{action}\b\s*:\s*(\d+)", value)
                if match:
                    return int(match.group(1))
            return 0

        for action in action_types:
            new_col = f"{action.lower()}_count"
            if new_col not in df.columns:
                df[new_col] = df[col].apply(lambda x, a=action: extract_count(x, a))

        # Drop the original JSON column
        df = df.drop(columns=[col])

        return df

    @classmethod
    def _flatten_to_string(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Convert dict/list column to comma-separated string."""
        if col not in df.columns:
            return df

        def to_string(val):
            if isinstance(val, dict):
                return ", ".join([f"{k}:{v}" for k, v in val.items()])
            elif isinstance(val, list):
                return ", ".join(str(v) for v in val)
            return str(val) if pd.notna(val) else ""

        df[col] = df[col].apply(to_string)
        return df

    @classmethod
    def _flatten_list_to_string(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Convert list column to comma-separated string."""
        if col not in df.columns:
            return df

        def list_to_string(val):
            if isinstance(val, list):
                return ", ".join(str(v) for v in val)
            elif isinstance(val, set):
                return ", ".join(str(v) for v in sorted(val))
            elif pd.notna(val):
                # Handle string representations of lists like "['a', 'b', 'c']"
                str_val = str(val).strip()
                if str_val.startswith("[") and str_val.endswith("]"):
                    try:
                        import ast
                        parsed = ast.literal_eval(str_val)
                        if isinstance(parsed, (list, set, tuple)):
                            return ", ".join(str(v) for v in parsed)
                    except (ValueError, SyntaxError):
                        pass
                return str_val
            return ""

        df[col] = df[col].apply(list_to_string)
        return df

    @classmethod
    def clean_dataframe_for_csv(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive cleanup for CSV export.

        Removes duplicate columns, flattens JSON, applies precision formatting.
        Does NOT rename columns (keeps original names for programmatic use).
        """
        if df.empty:
            return df

        df = cls.remove_duplicate_columns(df)
        df = cls.flatten_json_columns(df)
        df = cls.apply_precision_formatting(df)

        # Drop artifact columns left behind by chained reset_index() calls
        artifact_cols = [c for c in df.columns if c in ("index", "level_0")]
        if artifact_cols:
            df = df.drop(columns=artifact_cols)

        return df
