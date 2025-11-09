"""Pytest configuration and fixtures."""

import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Generate sample classification dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def binary_data():
    """Generate sample binary classification dataset.

    Returns:
        tuple: (X, y)
    """
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42,
    )
    return X, y


@pytest.fixture
def random_state():
    """Return a fixed random state for reproducibility.

    Returns:
        int: Random state seed
    """
    return 42
