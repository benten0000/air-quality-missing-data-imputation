"""Abstract base class for imputation models."""

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseImputer(ABC):
    """Abstract base class for all imputation models.

    This class defines the interface that all imputation models must implement.
    It provides methods for fitting models, performing imputation, and managing
    model state for serialization and deserialization.
    """

    @abstractmethod
    def fit(
        self,
        dataset: dict[str, np.ndarray],
        **kwargs,
    ) -> dict[str, np.ndarray] | None:
        """Fit the imputation model on training data.

        Args:
            dataset: Dictionary containing training data with keys such as 'X'.
                The 'X' key should contain the input data as a numpy array.
            **kwargs: Additional model-specific parameters for training.

        Returns:
            Dictionary containing training metrics (e.g., loss values),
            or None if no metrics are available.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def impute(
        self,
        dataset: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """Impute missing values in the dataset.

        Args:
            dataset: Dictionary containing data to impute with keys such as 'X'.
                The 'X' key should contain the input data with missing values
                (represented as NaN) as a numpy array.

        Returns:
            Imputed data as a numpy array with the same shape as the input,
            or None if imputation fails.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the model state for serialization.

        This method should return a dictionary containing all model parameters
        and state information needed to restore the model later.

        Returns:
            Dictionary mapping parameter names to their values as torch tensors.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the model state from a dictionary.

        Args:
            state_dict: Dictionary containing model parameters and state,
                as returned by `state_dict()`.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
