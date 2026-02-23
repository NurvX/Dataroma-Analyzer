#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Calculations

Financial calculations including returns, momentum scores, and portfolio metrics.
Common financial calculations and utility functions for analysis modules.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

import pandas as pd
import re
from typing import Optional, Union


class FinancialCalculations:
    """Common financial calculations for analysis modules."""

    @staticmethod
    def calculate_position_value(shares: Union[int, float], price: Union[int, float]) -> float:
        """Calculate position value from shares and price."""
        if pd.isna(shares) or pd.isna(price) or shares <= 0 or price <= 0:
            return 0.0
        return float(shares) * float(price)

    @staticmethod
    def calculate_portfolio_percentage(position_value: float, total_portfolio_value: float) -> float:
        """Calculate what percentage a position represents of total portfolio."""
        if pd.isna(position_value) or pd.isna(total_portfolio_value) or total_portfolio_value <= 0:
            return 0.0
        return (position_value / total_portfolio_value) * 100

    @staticmethod
    def calculate_price_change_percentage(current_price: float, original_price: float) -> float:
        """Calculate percentage change between two prices."""
        if pd.isna(current_price) or pd.isna(original_price) or original_price <= 0:
            return 0.0
        return ((current_price - original_price) / original_price) * 100

    @staticmethod
    def calculate_conviction_score(portfolio_pct: float, manager_count: int) -> float:
        """Calculate conviction score based on portfolio percentage and manager count."""
        if pd.isna(portfolio_pct) or manager_count <= 0:
            return 0.0
        return portfolio_pct * manager_count

    @staticmethod
    def calculate_52_week_position(current_price: float, week_52_low: float, week_52_high: float) -> float:
        """Calculate where current price sits in 52-week range (0-100%)."""
        if any(pd.isna([current_price, week_52_low, week_52_high])) or week_52_high <= week_52_low:
            return 50.0  # Default to middle if data is invalid

        if current_price <= week_52_low:
            return 0.0
        elif current_price >= week_52_high:
            return 100.0
        else:
            return ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100

    @staticmethod
    def is_near_52_week_low(
        current_price: float, week_52_low: float, week_52_high: float, threshold_pct: float = 25.0
    ) -> bool:
        """Check if current price is in the bottom X% of the 52-week range.

        Args:
            current_price: Current stock price.
            week_52_low: 52-week low price.
            week_52_high: 52-week high price.
            threshold_pct: Percentage threshold for "near low" (default 25%).
                           A stock at 20% position is in the bottom 25%.

        Returns:
            True if price is in the bottom threshold_pct of the 52-week range.
        """
        if pd.isna(current_price) or pd.isna(week_52_low) or pd.isna(week_52_high):
            return False
        if week_52_high <= week_52_low:
            return False
        position_pct = (current_price - week_52_low) / (week_52_high - week_52_low) * 100
        return position_pct < threshold_pct

    @staticmethod
    def is_near_52_week_high(
        current_price: float, week_52_low: float, week_52_high: float, threshold_pct: float = 25.0
    ) -> bool:
        """Check if current price is in the top X% of the 52-week range.

        Args:
            current_price: Current stock price.
            week_52_low: 52-week low price.
            week_52_high: 52-week high price.
            threshold_pct: Percentage threshold for "near high" (default 25%).
                           A stock at 80% position is in the top 25%.

        Returns:
            True if price is in the top threshold_pct of the 52-week range.
        """
        if pd.isna(current_price) or pd.isna(week_52_low) or pd.isna(week_52_high):
            return False
        if week_52_high <= week_52_low:
            return False
        position_pct = (current_price - week_52_low) / (week_52_high - week_52_low) * 100
        return position_pct > (100 - threshold_pct)


class TextAnalysisUtils:
    """Utility functions for analyzing text data like activities."""

    @staticmethod
    def extract_percentage_change(action: str) -> Optional[float]:
        """Extract percentage change from action string.

        Returns:
            Positive value for Add actions (accumulation)
            Negative value for Reduce/Sell actions (distribution)
            None if no percentage found
        """
        if pd.isna(action):
            return None

        action_str = str(action)
        action_lower = action_str.lower()

        # Handle special cases for full exits
        if "sold all" in action_lower or "sell 100" in action_lower:
            return -100.0

        # Determine if this is a reduction/sell (negative) or add/buy (positive)
        is_reduction = "reduce" in action_lower or "sell" in action_lower

        # Extract percentage using regex
        match = re.search(r"([+-]?\d+\.?\d*)\s*%", action_str)
        if match:
            value = float(match.group(1))
            # Apply sign based on action type
            if is_reduction:
                return -abs(value)  # Reductions are always negative
            else:
                return abs(value)  # Adds are always positive

        # Handle cases without % symbol (e.g., "Reduce 30")
        if is_reduction:
            match = re.search(r"(?:reduce|sell)\s+(\d+\.?\d*)", action_lower)
            if match:
                return -abs(float(match.group(1)))

        return None

    @staticmethod
    def extract_action_type(action: str) -> str:
        """Extract action type from action string."""
        if pd.isna(action):
            return "Hold"

        action_lower = str(action).lower()

        if "sell" in action_lower or "sold" in action_lower or "exit" in action_lower:
            return "Sell"
        elif "new" in action_lower or "buy" in action_lower:
            return "Buy"
        elif "add" in action_lower or "+" in action:
            return "Add"
        elif "reduce" in action_lower or "-" in action:
            return "Reduce"
        else:
            return "Hold"

    @staticmethod
    def clean_company_name(company_name: str) -> str:
        """Clean and standardize company names."""
        if pd.isna(company_name):
            return ""

        name = str(company_name).strip()

        # Remove common prefixes/suffixes that might be inconsistent
        name = re.sub(r"^-\s*", "", name)  # Remove leading dash
        name = re.sub(r"\s+", " ", name)  # Multiple spaces to single space

        return name.strip()


class ScoringUtils:
    """Utility functions for creating various scoring algorithms."""

    @staticmethod
    def calculate_hidden_gem_score(
        manager_count: int,
        max_portfolio_pct: float,
        avg_portfolio_pct: float,
        recent_activity_score: float = 0,
        price_momentum_score: float = 0,
        manager_quality_score: float = 1.0,
    ) -> float:
        """
        Calculate sophisticated hidden gem score using multiple factors.

        Factors:
        1. Exclusivity (low manager count but high conviction)
        2. Conviction (high portfolio percentages)
        3. Recent activity (buying momentum)
        4. Price momentum (technical factors)
        5. Manager quality (track record weighting)
        """
        # Factor 1: Exclusivity score (inverse of manager count, but reward some managers)
        # Note: When manager_count <= 5, (5 - manager_count) is always >= 0, so max() is not needed
        exclusivity_score = (5 - manager_count) / 4 if manager_count <= 5 else 0

        # Factor 2: Conviction score (based on portfolio allocations)
        conviction_score = min(max_portfolio_pct / 10, 1.0) + (avg_portfolio_pct / 20)

        # Factor 3: Recent activity score (0-1, passed in)
        activity_score = min(recent_activity_score, 1.0)

        # Factor 4: Price momentum score (0-1, passed in)
        momentum_score = min(price_momentum_score, 1.0)

        # Factor 5: Manager quality multiplier (1.0+ based on track record)
        quality_multiplier = max(manager_quality_score, 0.5)

        # Weighted combination
        base_score = exclusivity_score * 0.3 + conviction_score * 0.4 + activity_score * 0.15 + momentum_score * 0.15

        # Apply manager quality multiplier
        final_score = base_score * quality_multiplier

        return round(final_score, 3)

    @staticmethod
    def calculate_appeal_score(
        manager_count: int,
        avg_portfolio_pct: float,
        recent_buy_count: int = 0,
        value_factor: float = 0,
        max_score: float = 10.0,
    ) -> float:
        """Calculate general stock appeal score (0-10)."""
        score = 0

        # Manager count factor (max 3 points)
        score += min((manager_count / 10) * 3, 3)

        # Portfolio percentage factor (max 2 points)
        score += min((avg_portfolio_pct / 5) * 2, 2)

        # Value factor (max 2 points)
        score += min(value_factor * 2, 2)

        # Recent activity factor (max 3 points)
        score += min((recent_buy_count / 20) * 3, 3)

        return round(min(score, max_score), 2)

    @staticmethod
    def calculate_manager_quality_score(
        manager_id: str,
        total_portfolio_value: float = 0,
        track_record_years: int = 0,
        performance_metrics: Optional[dict] = None,
    ) -> float:
        """Calculate manager quality score for weighting (1.0 = average, higher = better)."""
        # Default scores for well-known high-quality managers
        premium_managers = {
            "berkshire": 2.0,
            "bh": 2.0,
            "munger": 1.8,
            "cm": 1.8,
            "akre": 1.6,
            "value": 1.5,  # Li Lu
            "pershing": 1.4,  # Ackman
            "mohnish": 1.3,
        }

        # Check for exact match first, then partial match for manager IDs
        base_score = premium_managers.get(manager_id, None)
        if base_score is None:
            # Try partial matching for manager IDs (e.g., 'berkshire_hathaway' matches 'berkshire')
            manager_id_lower = manager_id.lower()
            for key, value in premium_managers.items():
                if key in manager_id_lower or manager_id_lower in key:
                    base_score = value
                    break
            if base_score is None:
                base_score = 1.0

        # Adjust based on portfolio size (larger = more credible)
        if total_portfolio_value > 10_000_000_000:  # $10B+
            base_score *= 1.2
        elif total_portfolio_value > 1_000_000_000:  # $1B+
            base_score *= 1.1

        return min(base_score, 2.5)  # Cap at 2.5x
