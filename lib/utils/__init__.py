#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Utils Package

Utility functions for parsing, formatting, and calculations.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

"""Utility components."""

from .parsers import DataromaParser
from .logging_config import setup_logging, get_logger
from .cross_file_validator import CrossFileValidator
from .csv_validator import CSVValidator, Severity, ValidationViolation

__all__ = [
    "DataromaParser",
    "setup_logging",
    "get_logger",
    "CrossFileValidator",
    "CSVValidator",
    "Severity",
    "ValidationViolation",
]
