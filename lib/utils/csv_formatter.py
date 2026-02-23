#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - CSV Formatter

Formats analysis results into clean CSV files with proper headers.
CSV formatting utilities for clean, consistent output with manager name mapping
and column standardization.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CSVFormatter:
    """Formats CSV files with clean manager names and consistent columns."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.manager_mapping = self._load_manager_mapping()

    def _load_manager_mapping(self) -> Dict[str, str]:
        """Load manager ID to name mapping."""
        mapping = {}

        try:
            with open(self.cache_dir / "json" / "managers.json", "r") as f:
                managers = json.load(f)

            for mgr in managers:
                mgr_id = mgr.get("id", mgr.get("manager_id", ""))
                name = mgr.get("name", "")
                if mgr_id and name:
                    # Clean the name - remove any "Updated" timestamps
                    clean_name = re.sub(r"\s+Updated\s+\d+\s+\w+\s+\d+", "", name).strip()
                    mapping[mgr_id] = clean_name
        except Exception as e:
            logger.warning(f"Could not load manager mapping: {e}")

        return mapping

    def clean_manager_name(self, name: str) -> str:
        """Remove 'Updated' timestamps from manager names."""
        if pd.isna(name) or not name:
            return ""
        # Remove "Updated DD Mon YYYY" pattern
        cleaned = re.sub(r"\s+Updated\s+\d+\s+\w+\s+\d+", "", str(name)).strip()
        # Also remove double spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def format_manager_list(self, managers_str: str, max_managers: int = 5) -> str:
        """Format manager list for readability, limiting to top N managers."""
        if pd.isna(managers_str) or not managers_str:
            return ""

        # Split by comma and clean each
        manager_entries = [m.strip() for m in str(managers_str).split(",")]
        clean_managers = []

        for entry in manager_entries:
            # First try to map if it's an ID
            mapped_name = self.manager_mapping.get(entry, entry)
            # Then clean any timestamps
            clean_name = self.clean_manager_name(mapped_name)
            if clean_name and clean_name not in clean_managers:  # Avoid duplicates
                clean_managers.append(clean_name)

        # Limit to max_managers for readability
        total_count = len(clean_managers)
        if total_count > max_managers:
            display_managers = clean_managers[:max_managers]
            remaining = total_count - max_managers
            display_managers.append(f"+{remaining} more")
            return ", ".join(display_managers)

        return ", ".join(clean_managers) if clean_managers else ""

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and order."""
        if df.empty:
            return df

        df = df.copy()

        # FIRST: Remove any existing duplicate columns (.1, .2 suffixes)
        duplicate_cols = [col for col in df.columns if re.match(r".*\.\d+$", col)]
        if duplicate_cols:
            logger.debug(f"Removing pre-existing duplicate columns: {duplicate_cols}")
            df = df.drop(columns=duplicate_cols)

        # Canonical column name mappings for standardization
        # These are direct renames (source -> target)
        column_mappings = {
            # Value standardization - use 'total_value' as canonical
            "value": "total_value",
            "current_total_value": "total_value",
            "remaining_value": "total_value",
            "current_value": "total_value",
            # Portfolio % standardization
            # Keep avg_portfolio_pct and max_portfolio_pct distinct (different aggregations)
            # But standardize 'portfolio_percent' to 'portfolio_pct'
            "portfolio_percent": "portfolio_pct",
            # Period standardization
            "quarter": "period",  # single period
            # Keep 'periods' for lists - intentionally different
        }

        # Apply column mappings
        rename_dict = {}
        for source_col, target_col in column_mappings.items():
            if source_col in df.columns and target_col not in df.columns:
                rename_dict[source_col] = target_col
            elif source_col in df.columns and target_col in df.columns:
                # Target already exists, drop the source to avoid duplicates
                logger.debug(f"Dropping '{source_col}' as '{target_col}' already exists")
                df = df.drop(columns=[source_col])

        if rename_dict:
            logger.debug(f"Renaming columns: {rename_dict}")
            df = df.rename(columns=rename_dict)

        # Define column groups that map to the same target (in preference order)
        # First item in each list is preferred and will become the target name
        column_groups = [
            # Manager columns -> "manager"
            (["manager_name", "manager_id", "manager"], "manager"),
            # Ticker columns -> "ticker"
            (["ticker", "stock", "symbol"], "ticker"),
            # Company name columns -> "company_name"
            (["company_name", "name", "company"], "company_name"),
            # Price columns -> "current_price"
            (["current_price", "price", "latest_price"], "current_price"),
            # Date columns -> "action_date" (only for actual activity dates, not first_reported_date)
            (["action_date", "activity_date"], "action_date"),
            # Action columns -> "action"
            (["action", "activity", "activity_type", "transaction"], "action"),
        ]

        # Process each group: keep only the first (most preferred) column that exists
        for source_cols, target_name in column_groups:
            existing_cols = [col for col in source_cols if col in df.columns]
            if len(existing_cols) > 0:
                # Keep the first (most preferred) existing column
                keep_col = existing_cols[0]
                # Drop all other columns in this group
                cols_to_drop = [col for col in existing_cols[1:]]
                if cols_to_drop:
                    logger.debug(f"Dropping redundant columns: {cols_to_drop}, keeping: {keep_col}")
                    df = df.drop(columns=cols_to_drop)
                # Rename to target if needed
                if keep_col != target_name:
                    df = df.rename(columns={keep_col: target_name})

        # FINAL: Remove any duplicate columns that might have been created
        duplicate_cols = [col for col in df.columns if re.match(r".*\.\d+$", col)]
        if duplicate_cols:
            logger.debug(f"Removing final duplicate columns: {duplicate_cols}")
            df = df.drop(columns=duplicate_cols)

        # Standard column order (put most important first)
        # Note: 'top_managers' is preferred over 'managers' as it clearly indicates truncation
        standard_order = [
            "ticker",
            "company_name",
            "action",
            "action_date",
            "current_price",
            "manager",
            "top_managers",
            "managers_shown",
            "manager_count",
            "managers",
            "total_value",
            "portfolio_pct",
            "avg_portfolio_pct",
        ]

        # Separate metadata columns (prefixed with _) - these go at the end
        metadata_cols = [col for col in df.columns if col.startswith("_")]
        regular_cols = [col for col in df.columns if not col.startswith("_")]

        # Get regular columns in preferred order
        ordered_cols = [col for col in standard_order if col in regular_cols]
        other_regular_cols = [col for col in regular_cols if col not in ordered_cols]

        # Final order: standard columns, other regular columns, then metadata columns at end
        final_order = ordered_cols + other_regular_cols + sorted(metadata_cols)

        return df[final_order]

    def truncate_long_text(self, text: str, max_length: int = 100) -> str:
        """Truncate long text values for readability."""
        if pd.isna(text) or not text:
            return ""
        text = str(text)
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def format_csv_file(self, filepath: Path) -> bool:
        """Format a single CSV file."""
        try:
            df = pd.read_csv(filepath)

            # Clean manager columns
            manager_cols = [
                "manager",
                "managers",
                "buying_managers",
                "selling_managers",
                "top_managers",
                "active_managers",
            ]

            for col in manager_cols:
                if col in df.columns:
                    if col == "manager":
                        # Single manager - just clean the name
                        df[col] = df[col].apply(self.clean_manager_name)
                    else:
                        # Multiple managers - format and limit count
                        df[col] = df[col].apply(self.format_manager_list)

            # Truncate long detail columns for readability
            detail_cols = [
                "crisis_details",
                "manager_details",
                "periods",
                "new_manager_names",
            ]

            for col in detail_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: self.truncate_long_text(x, 100))

            # Standardize column order
            df = self.standardize_columns(df)

            # Save formatted CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Formatted CSV: {filepath.name}")
            return True

        except Exception as e:
            logger.error(f"Error formatting {filepath}: {e}")
            return False

    def format_all_csvs(self, directory: Path) -> int:
        """Format all CSV files in a directory."""
        csv_files = list(directory.glob("**/*.csv"))
        formatted_count = 0

        logger.info(f"Formatting {len(csv_files)} CSV files in {directory}")

        for csv_file in csv_files:
            if self.format_csv_file(csv_file):
                formatted_count += 1

        logger.info(f"Successfully formatted {formatted_count}/{len(csv_files)} CSV files")
        return formatted_count
