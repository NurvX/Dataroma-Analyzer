#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Constants

Centralized configuration for scoring weights, thresholds, and analysis parameters.
Eliminates magic numbers scattered across the codebase.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Weights used in scoring algorithms."""

    # Hidden Gem Score weights (must sum to 1.0)
    GEM_EXCLUSIVITY: float = 0.30
    GEM_CONVICTION: float = 0.25
    GEM_ACTIVITY: float = 0.20
    GEM_MOMENTUM: float = 0.15
    GEM_QUALITY: float = 0.10

    # Alternative hidden gem scoring (current implementation)
    GEM_EXCLUSIVITY_ALT: float = 0.30
    GEM_CONVICTION_ALT: float = 0.40
    GEM_ACTIVITY_ALT: float = 0.15
    GEM_MOMENTUM_ALT: float = 0.15

    # Appeal score component weights
    APPEAL_MANAGER_COUNT: float = 3.0
    APPEAL_PORTFOLIO_PCT: float = 2.0
    APPEAL_VALUE_FACTOR: float = 2.0
    APPEAL_RECENT_ACTIVITY: float = 3.0

    # Momentum score weights
    MOMENTUM_BUY_COUNT: float = 2.0
    MOMENTUM_HOLDER_COUNT: float = 1.0
    MOMENTUM_RECENCY_BONUS_RECENT: float = 5.0
    MOMENTUM_RECENCY_BONUS_MID: float = 4.0
    MOMENTUM_RECENCY_BONUS_OLD: float = 3.0

    # Contrarian score divisor
    CONTRARIAN_SCORE_DIVISOR: float = 10.0


@dataclass(frozen=True)
class PriceThresholds:
    """Price thresholds for analysis categories."""

    PENNY: float = 5.0
    LOW: float = 10.0
    MODERATE: float = 20.0
    MID: float = 50.0
    HIGH: float = 100.0

    # Price analysis thresholds
    HIGH_CONVICTION_MAX_PRICE: float = 25.0
    NEAR_52_WEEK_DEFAULT_THRESHOLD: float = 10.0
    NEAR_52_WEEK_VALUE_THRESHOLD: float = 15.0
    BOTTOM_RANGE_THRESHOLD: float = 25.0
    TOP_RANGE_THRESHOLD: float = 75.0

    # Value opportunity thresholds
    DISCOUNT_THRESHOLD: float = -2.0  # Down 2% from reported
    DEEP_DISCOUNT_THRESHOLD: float = -10.0


@dataclass(frozen=True)
class MarketCapThresholds:
    """Market capitalization thresholds for categorization."""

    MICRO: int = 300_000_000  # $300M
    SMALL: int = 2_000_000_000  # $2B
    MID: int = 10_000_000_000  # $10B
    LARGE: int = 200_000_000_000  # $200B

    # Manager quality portfolio size thresholds
    LARGE_PORTFOLIO: int = 10_000_000_000  # $10B
    MEDIUM_PORTFOLIO: int = 1_000_000_000  # $1B


@dataclass(frozen=True)
class AnalysisDefaults:
    """Default values for analysis parameters."""

    # Quarter analysis
    RECENT_QUARTERS: int = 3
    CONTRARIAN_QUARTERS: int = 2

    # Conviction thresholds
    MIN_CONVICTION_YEARS: int = 5
    HIGH_CONVICTION_PCT: float = 5.0
    MEANINGFUL_POSITION_PCT: float = 1.5
    HIDDEN_GEM_MIN_PCT: float = 2.0
    UNDER_RADAR_MIN_PCT: float = 3.0
    HIGH_CONCENTRATION_PCT: float = 3.0

    # Manager count thresholds
    MAX_GEM_MANAGERS: int = 4
    MAX_UNDER_RADAR_MANAGERS: int = 2
    MIN_MULTI_MANAGER: int = 5
    MIN_CONTRARIAN_MANAGERS: int = 3
    MIN_MOMENTUM_BUYS: int = 2
    MIN_SELLERS: int = 2

    # Scoring limits
    MAX_APPEAL_SCORE: float = 10.0
    MAX_MANAGER_QUALITY_SCORE: float = 2.5
    MIN_MANAGER_QUALITY_SCORE: float = 0.5

    # Display limits
    HIDDEN_GEMS_DISPLAY_LIMIT: int = 50
    TOP_HOLDINGS_DISPLAY_LIMIT: int = 50
    MOMENTUM_DISPLAY_LIMIT: int = 50
    VALUE_PLAYS_DISPLAY_LIMIT: int = 30
    CONTRARIAN_DISPLAY_LIMIT: int = 30
    UNDER_RADAR_DISPLAY_LIMIT: int = 40
    NEW_POSITIONS_DISPLAY_LIMIT: int = 100
    CONCENTRATION_DISPLAY_LIMIT: int = 100

    # Consistency threshold
    MIN_CONSISTENCY_SCORE: float = 0.3


@dataclass(frozen=True)
class ManagerQuality:
    """Quality scores for known managers."""

    # Premium manager multipliers (1.0 = average)
    BERKSHIRE: float = 2.0
    MUNGER: float = 1.8
    AKRE: float = 1.6
    LI_LU: float = 1.5
    ACKMAN: float = 1.4
    PABRAI: float = 1.3
    DEFAULT: float = 1.0

    # Portfolio size multipliers
    LARGE_PORTFOLIO_MULTIPLIER: float = 1.2
    MEDIUM_PORTFOLIO_MULTIPLIER: float = 1.1

    # Premium manager threshold
    PREMIUM_THRESHOLD: float = 1.2


@dataclass(frozen=True)
class DataValidation:
    """Thresholds for data validation."""

    # Sample sizes for validation
    VALIDATION_SAMPLE_SIZE: int = 100

    # Valid action types
    VALID_ACTIONS: tuple = ("Buy", "Sell", "Add", "Reduce", "Hold")

    # Required fields
    ACTIVITY_REQUIRED_FIELDS: tuple = ("ticker", "manager_id", "action_type", "period")
    HOLDING_REQUIRED_FIELDS: tuple = ("ticker", "manager_id", "value")
    MANAGER_REQUIRED_FIELDS: tuple = ("id", "name")

    # Corruption indicators
    CORRUPTED_SYMBOL: str = "\u2261"  # The ≡ symbol


# Regex patterns for performance (compiled once)
import re

QUARTER_PATTERN = re.compile(r"Q(\d)\s+(\d{4})")
NAME_CLEANING_PATTERN = re.compile(r"\s+Updated\s+\d{1,2}\s+\w+\s+\d{4}$")
PERCENTAGE_PATTERN = re.compile(r"([+-]?\d+\.?\d*)\s*%")
REDUCE_PATTERN = re.compile(r"Reduce\s+([+-]?\d+\.?\d*)")
LEADING_DASH_PATTERN = re.compile(r"^-\s*")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
