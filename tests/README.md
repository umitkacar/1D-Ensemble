# 🧪 Tests

Comprehensive test suite for the 1D-Ensemble framework.

## 📋 Test Structure

```
tests/
├── conftest.py           # Pytest fixtures and configuration
├── test_models.py        # Unit tests for individual models
├── test_ensemble.py      # Unit tests for ensemble functionality
└── test_integration.py   # Integration tests for workflows
```

## 🚀 Running Tests

### Basic Usage

```bash
# Run all tests
hatch run test

# Run with coverage
hatch run test-cov

# Run in parallel
hatch run test-parallel

# Run verbose
hatch run test-verbose
```

### Advanced Usage

```bash
# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestXGBoostModel

# Run specific test
pytest tests/test_models.py::TestXGBoostModel::test_initialization

# Run tests matching pattern
pytest -k "test_predict"

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "unit"      # Run only unit tests
pytest -m "integration"  # Run only integration tests
```

## 🏷️ Test Markers

- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow-running tests
- `gpu`: Tests requiring GPU

## 📊 Coverage

```bash
# Generate coverage report
hatch run test-cov

# View HTML coverage report
open htmlcov/index.html
```

## ✅ Pre-commit Testing

Tests are automatically run via pre-commit hooks:

```bash
# Install pre-commit hooks
hatch run pre-commit-install

# Run manually
hatch run pre-commit-run
```

## 🎯 Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from ensemble_1d import XGBoostModel


def test_model_initialization():
    """Test model initialization."""
    model = XGBoostModel(n_estimators=100)
    assert model.params["n_estimators"] == 100


@pytest.mark.slow
def test_model_training(sample_data):
    """Test model training."""
    X_train, X_test, y_train, y_test = sample_data
    model = XGBoostModel()
    model.fit(X_train, y_train)
    assert model.model is not None
```

## 📈 Test Coverage Goals

- Overall coverage: **>80%**
- Critical paths: **>95%**
- Docstring coverage: **>80%**
