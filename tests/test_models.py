"""Unit tests for model implementations."""

import numpy as np
import pytest

from ensemble_1d.models import PyTorchModel, RandomForestModel, XGBoostModel


class TestXGBoostModel:
    """Test XGBoost model implementation."""

    def test_initialization(self):
        """Test model initialization."""
        model = XGBoostModel(n_estimators=50, learning_rate=0.05)
        assert model.params["n_estimators"] == 50
        assert model.params["learning_rate"] == 0.05

    def test_fit_predict(self, sample_data):
        """Test model training and prediction."""
        X_train, X_test, y_train, y_test = sample_data

        model = XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert predictions.dtype in [np.int32, np.int64]

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_data

        model = XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape[0] == len(y_test)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_data

        model = XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestRandomForestModel:
    """Test Random Forest model implementation."""

    def test_initialization(self):
        """Test model initialization."""
        model = RandomForestModel(n_estimators=50, max_depth=5)
        assert model.params["n_estimators"] == 50
        assert model.params["max_depth"] == 5

    def test_fit_predict(self, sample_data):
        """Test model training and prediction."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestModel(n_estimators=10)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestModel(n_estimators=10)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape[0] == len(y_test)
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestPyTorchModel:
    """Test PyTorch model implementation."""

    def test_initialization(self):
        """Test model initialization."""
        model = PyTorchModel(hidden_size=64, num_epochs=5)
        assert model.params["hidden_size"] == 64
        assert model.params["num_epochs"] == 5

    @pytest.mark.slow
    def test_fit_predict(self, sample_data):
        """Test model training and prediction."""
        X_train, X_test, y_train, y_test = sample_data

        model = PyTorchModel(hidden_size=32, num_epochs=5, batch_size=16)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    @pytest.mark.slow
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_data

        model = PyTorchModel(hidden_size=32, num_epochs=5, batch_size=16)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape[0] == len(y_test)
        assert np.allclose(probas.sum(axis=1), 1.0, rtol=0.01)

    @pytest.mark.gpu
    def test_gpu_training(self, binary_data):
        """Test GPU training if available."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        X, y = binary_data
        model = PyTorchModel(hidden_size=16, num_epochs=3, device="cuda")
        model.fit(X, y)

        assert model.device == "cuda"
