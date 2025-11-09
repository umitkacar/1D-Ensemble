"""Ensemble model combining multiple base models."""

import sys
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from ensemble_1d.models.base import BaseModel


class EnsembleModel:
    """Ensemble model that combines predictions from multiple models.

    Supports various fusion methods including voting, weighted averaging,
    and stacking.

    Args:
        models: List of base models to ensemble.
        fusion_method: Method to combine predictions. Options: 'voting',
            'weighted', 'average'. Default: 'voting'.
        weights: Optional weights for weighted fusion. Must sum to 1.

    Example:
        >>> from ensemble_1d import XGBoostModel, PyTorchModel, EnsembleModel
        >>> models = [XGBoostModel(), PyTorchModel()]
        >>> ensemble = EnsembleModel(models=models, fusion_method='voting')
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        models: List[BaseModel],
        fusion_method: Literal["voting", "weighted", "average"] = "voting",
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize ensemble model."""
        self.models = models
        self.fusion_method = fusion_method

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1")
            self.weights = np.array(weights)
        else:
            self.weights = np.ones(len(models)) / len(models)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Train all models in the ensemble.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        if self.fusion_method == "voting":
            return self._voting_predict(X)
        if self.fusion_method in ["weighted", "average"]:
            return self._weighted_predict(X)
        raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Hard voting prediction.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        predictions = np.array([model.predict(X) for model in self.models])
        # Majority voting
        voted_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions,
        )
        return voted_predictions

    def _weighted_predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted prediction based on probability estimates.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        probas = np.array([model.predict_proba(X) for model in self.models])
        # Weighted average of probabilities
        weighted_proba = np.average(probas, axis=0, weights=self.weights)
        return np.argmax(weighted_proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance.

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

        # Add individual model metrics
        for i, model in enumerate(self.models):
            model_metrics = model.evaluate(X, y)
            metrics[f"model_{i}_accuracy"] = model_metrics["accuracy"]

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get ensemble parameters.

        Returns:
            Dictionary of ensemble parameters.
        """
        return {
            "fusion_method": self.fusion_method,
            "weights": self.weights.tolist(),
            "num_models": len(self.models),
        }
