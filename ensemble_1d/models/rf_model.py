"""Random Forest model implementation."""

from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ensemble_1d.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest ensemble model.

    A robust ensemble method using multiple decision trees.

    Args:
        n_estimators: Number of trees. Default: 100.
        max_depth: Maximum tree depth. Default: None.
        min_samples_split: Minimum samples to split. Default: 2.
        min_samples_leaf: Minimum samples per leaf. Default: 1.
        max_features: Number of features for best split. Default: 'sqrt'.
        **kwargs: Additional scikit-learn parameters.

    Example:
        >>> model = RandomForestModel(n_estimators=200, max_depth=10)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        **kwargs: Any,
    ) -> None:
        """Initialize Random Forest model."""
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **kwargs,
        )
        self.model: Optional[RandomForestClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train the Random Forest model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).

        Raises:
            ValueError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, path)

    def load(self, path: str) -> "RandomForestModel":
        """Load model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            Self for method chaining.
        """
        self.model = joblib.load(path)
        return self
