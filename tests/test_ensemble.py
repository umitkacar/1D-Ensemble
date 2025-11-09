"""Unit tests for ensemble functionality."""

import numpy as np
import pytest

from ensemble_1d.fusion import EnsembleModel
from ensemble_1d.models import PyTorchModel, RandomForestModel, XGBoostModel


class TestEnsembleModel:
    """Test ensemble model implementation."""

    def test_initialization(self):
        """Test ensemble initialization."""
        models = [
            XGBoostModel(n_estimators=10),
            RandomForestModel(n_estimators=10),
        ]
        ensemble = EnsembleModel(models=models, fusion_method="voting")
        assert len(ensemble.models) == 2
        assert ensemble.fusion_method == "voting"

    def test_invalid_weights(self):
        """Test invalid weights raise error."""
        models = [XGBoostModel(), RandomForestModel()]

        with pytest.raises(ValueError, match="must sum to 1"):
            EnsembleModel(models=models, weights=[0.5, 0.3])

        with pytest.raises(ValueError, match="must match"):
            EnsembleModel(models=models, weights=[1.0])

    def test_voting_fusion(self, sample_data):
        """Test voting fusion method."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            XGBoostModel(n_estimators=10),
            RandomForestModel(n_estimators=10),
        ]
        ensemble = EnsembleModel(models=models, fusion_method="voting")
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_weighted_fusion(self, sample_data):
        """Test weighted fusion method."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            XGBoostModel(n_estimators=10),
            RandomForestModel(n_estimators=10),
        ]
        weights = [0.6, 0.4]
        ensemble = EnsembleModel(models=models, fusion_method="weighted", weights=weights)
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_predict_proba(self, sample_data):
        """Test ensemble probability prediction."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            XGBoostModel(n_estimators=10),
            RandomForestModel(n_estimators=10),
        ]
        ensemble = EnsembleModel(models=models)
        ensemble.fit(X_train, y_train)

        probas = ensemble.predict_proba(X_test)
        assert probas.shape[0] == len(y_test)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_evaluate(self, sample_data):
        """Test ensemble evaluation."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            XGBoostModel(n_estimators=10),
            RandomForestModel(n_estimators=10),
        ]
        ensemble = EnsembleModel(models=models)
        ensemble.fit(X_train, y_train)

        metrics = ensemble.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "model_0_accuracy" in metrics
        assert "model_1_accuracy" in metrics

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_ensemble(self, sample_data):
        """Test full ensemble with all models."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            XGBoostModel(n_estimators=10),
            RandomForestModel(n_estimators=10),
            PyTorchModel(hidden_size=32, num_epochs=3, batch_size=16),
        ]
        ensemble = EnsembleModel(models=models, fusion_method="weighted")
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        metrics = ensemble.evaluate(X_test, y_test)

        assert len(predictions) == len(y_test)
        assert metrics["accuracy"] > 0.0
