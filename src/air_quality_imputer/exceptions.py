"""Custom exceptions for the air quality imputation project.

This module defines exception classes used throughout the project to handle
specific error conditions in a consistent and descriptive manner.
"""


class ModelBuildError(Exception):
    """Exception raised when model building fails.

    This exception is raised when there is an error during model initialization,
    architecture configuration, or any other aspect of model construction.
    """

    pass


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails.

    This exception is raised when there is an error loading or processing
    datasets, including file I/O errors, parsing errors, or data format issues.
    """

    pass


class TrackingError(Exception):
    """Exception raised when MLflow tracking operations fail.

    This exception is raised when there is an error with MLflow tracking,
    such as experiment creation, run initialization, parameter logging,
    or metric recording failures.
    """

    pass


class ValidationError(Exception):
    """Exception raised when input validation fails.

    This exception is raised when input parameters or data fail validation
    checks, such as out-of-range values, incorrect types, or other
    constraint violations.
    """

    pass
