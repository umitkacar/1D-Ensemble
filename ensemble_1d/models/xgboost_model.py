"""XGBoost model implementation."""

from typing import Any, Optional

import numpy as np
import xgboost as xgb

from ensemble_1d.models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model.

    A powerful gradient boosting framework that uses tree-based learning.

    Args:
        n_estimators: Number of boosting rounds. Default: 100.
        learning_rate: Step size shrinkage. Default: 0.1.
        max_depth: Maximum tree depth. Default: 6.
        subsample: Fraction of samples for training. Default: 0.8.
        colsample_bytree: Fraction of features for training. Default: 0.8.
        **kwargs: Additional XGBoost parameters.

    Example:
        >>> model = XGBoostModel(n_estimators=100, learning_rate=0.1)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """Initialize XGBoost model."""
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs,
        )
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """Train the XGBoost model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.model = xgb.XGBClassifier(**self.params)
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
        self.model.save_model(path)

    def load(self, path: str) -> "XGBoostModel":
        """Load model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            Self for method chaining.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        return self
