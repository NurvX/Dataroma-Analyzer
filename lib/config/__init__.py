#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Configuration Module

Contains constants, scoring weights, thresholds, and file contracts used across the analyzer.
"""

from .constants import (
    ScoringWeights,
    PriceThresholds,
    MarketCapThresholds,
    AnalysisDefaults,
    ManagerQuality,
    DataValidation,
)

from .file_contracts import (
    ViolationType,
    Violation,
    ColumnSpec,
    FileContract,
    validate_against_contract,
    CONTRACT_REGISTRY,
    get_contract,
    get_contracts_by_mode,
    validate_file,
    generate_contract_documentation,
    STANDARD_METADATA_COLUMNS,
)

__all__ = [
    # Constants
    "ScoringWeights",
    "PriceThresholds",
    "MarketCapThresholds",
    "AnalysisDefaults",
    "ManagerQuality",
    "DataValidation",
    # File contracts
    "ViolationType",
    "Violation",
    "ColumnSpec",
    "FileContract",
    "validate_against_contract",
    "CONTRACT_REGISTRY",
    "get_contract",
    "get_contracts_by_mode",
    "validate_file",
    "generate_contract_documentation",
    "STANDARD_METADATA_COLUMNS",
]
