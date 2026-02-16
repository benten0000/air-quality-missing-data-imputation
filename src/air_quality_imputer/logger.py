"""Logging configuration for the air quality imputation project.

This module provides a centralized logging setup with consistent formatting
across all modules in the project.
"""

import logging
import sys


def setup_logger(name: str = "air_quality_imputer", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a logger for the project.

    Args:
        name: Name of the logger. Defaults to "air_quality_imputer".
        level: Logging level. Defaults to logging.INFO.

    Returns:
        Configured logger instance with consistent formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter with timestamp, name, level, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


# Module-level logger instance
logger = setup_logger()
