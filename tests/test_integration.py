"""Integration tests for end-to-end workflows."""

import pytest

from ensemble_1d import EnsembleModel, PyTorchModel, RandomForestModel, XGBoostModel


@pytest.mark.integration
class TestEndToEnd:
    """Test complete workflows."""

    def test_quickstart_example(self, sample_data):
        """Test the quickstart example workflow."""
        X_train, X_test, y_train, y_test = sample_data

        # Initialize models
        models = [
            XGBoostModel(n_estimators=10, learning_rate=0.1),
            RandomForestModel(n_estimators=10, max_depth=5),
        ]

        # Create ensemble
        ensemble = EnsembleModel(models=models, fusion_method="voting")

        # Train
        ensemble.fit(X_train, y_train)

        # Predict
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)

        # Evaluate
        metrics = ensemble.evaluate(X_test, y_test)

        # Assertions
        assert len(predictions) == len(y_test)
        assert probabilities.shape[0] == len(y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "f1_score" in metrics

    @pytest.mark.slow
    def test_advanced_workflow(self, sample_data):
        """Test advanced workflow with custom weights."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            XGBoostModel(n_estimators=20, max_depth=5),
            RandomForestModel(n_estimators=20, max_depth=8),
            PyTorchModel(hidden_size=64, num_epochs=5),
        ]

        weights = [0.4, 0.3, 0.3]
        ensemble = EnsembleModel(models=models, fusion_method="weighted", weights=weights)

        ensemble.fit(X_train, y_train)
        metrics = ensemble.evaluate(X_test, y_test)

        assert metrics["accuracy"] >= 0.5  # Should be better than random
