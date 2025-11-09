"""Base model class for all ensemble models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class BaseModel(ABC):
    """Abstract base class for all models in the ensemble.

    This class defines the interface that all models must implement.

    Attributes:
        model: The underlying model instance.
        params: Model hyperparameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base model.

        Args:
            **kwargs: Model-specific hyperparameters.
        """
        self.model: Optional[Any] = None
        self.params: Dict[str, Any] = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Train the model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Test features of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            Dictionary containing evaluation metrics.
        """
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
        }

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        return self.params

    def set_params(self, **params: Any) -> "BaseModel":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            Self for method chaining.
        """
        self.params.update(params)
        return self
