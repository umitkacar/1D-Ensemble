"""
🚀 Quick Start Example for 1D-Ensemble
========================================

This example demonstrates the basic usage of the 1D-Ensemble framework.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ensemble_1d import EnsembleModel, PyTorchModel, RandomForestModel, XGBoostModel


def main() -> None:
    """Run quick start example."""
    print("🎯 1D-Ensemble Quick Start Example")
    print("=" * 50)

    # Generate synthetic dataset
    print("\n📊 Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Initialize models
    print("\n🤖 Initializing models...")
    models = [
        XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=5),
        PyTorchModel(hidden_size=128, num_epochs=20, batch_size=32),
        RandomForestModel(n_estimators=200, max_depth=10),
    ]

    # Create ensemble
    print("🔗 Creating ensemble with voting fusion...")
    ensemble = EnsembleModel(models=models, fusion_method="voting")

    # Train ensemble
    print("🏋️  Training ensemble...")
    ensemble.fit(X_train, y_train)
    print("✅ Training complete!")

    # Evaluate
    print("\n📈 Evaluating performance...")
    metrics = ensemble.evaluate(X_test, y_test)

    print("\n🏆 Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    # Individual model performance
    print("\n📊 Individual Model Performance:")
    for i in range(len(models)):
        acc = metrics.get(f"model_{i}_accuracy", 0)
        print(f"  Model {i}: {acc:.4f}")

    # Make predictions
    print("\n🔮 Making predictions on new data...")
    predictions = ensemble.predict(X_test[:5])
    probabilities = ensemble.predict_proba(X_test[:5])

    print("\nSample Predictions:")
    for i in range(5):
        print(f"  Sample {i}: Predicted={predictions[i]}, True={y_test[i]}")
        print(f"    Probabilities: {probabilities[i]}")

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
