#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Logging Configuration

Centralized logging configuration for consistent log formatting and levels
across all modules.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Compact format for CLI output
COMPACT_FORMAT = "%(levelname)s: %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = DEFAULT_FORMAT,
    use_compact: bool = False,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path to write logs to
        format_string: Format string for log messages
        use_compact: Use compact format suitable for CLI output
    """
    if use_compact:
        format_string = COMPACT_FORMAT

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured Logger instance
    """
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """Set the logging level for the root logger.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logging.getLogger().setLevel(level)


def enable_debug() -> None:
    """Enable debug-level logging."""
    set_level(logging.DEBUG)


def enable_quiet() -> None:
    """Enable quiet mode (only warnings and errors)."""
    set_level(logging.WARNING)
