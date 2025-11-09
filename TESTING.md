# 🧪 Testing Guide

## Quick Test

The package includes a comprehensive test script:

```bash
python test_package.py
```

This tests:
- ✅ Package imports
- ✅ Type annotations
- ✅ Basic model functionality
- ✅ Ensemble operations

## Running Full Test Suite

### With Pytest

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=ensemble_1d --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests in parallel
pytest -n auto
```

### With Hatch

```bash
# Run tests
hatch run test

# With coverage
hatch run test-cov

# Parallel execution
hatch run test-parallel
```

## Test Requirements

### Minimum Requirements

The package works with minimal dependencies:
- numpy
- scikit-learn
- joblib

### Full Requirements

For all features:
- torch (for PyTorchModel)
- xgboost (for XGBoostModel)
- All packages in requirements.txt

## Known Issues

### PyTorch Installation

PyTorch is large (~2GB). If you don't need PyTorchModel:
```bash
pip install ensemble-1d  # Skips torch
```

To use PyTorchModel:
```bash
pip install ensemble-1d torch
```

### NumPy Version

We pin `numpy<2.0.0` due to compatibility with ML libraries.
This is intentional and ensures stability.

## Continuous Integration

The project uses GitHub Actions for:
- ✅ Python 3.8-3.12 testing
- ✅ Code quality checks (black, ruff, mypy)
- ✅ Security scanning (bandit)
- ✅ Coverage reporting

## Test Coverage Goals

- Overall: >80%
- Critical paths: >95%
- Type coverage: 100%

## Troubleshooting

### Import Errors

If you see import errors, ensure dependencies are installed:

```bash
pip install -e ".[all]"  # Install everything
```

### Test Failures

1. Check Python version (>=3.8 required)
2. Ensure all dependencies installed
3. Try running individual test files
4. Check for conflicting package versions

### Performance Tests

For benchmark tests:
```bash
pytest tests/test_models.py::test_benchmark -v
```

## Contributing Tests

When adding features:
1. Add unit tests in `tests/test_*.py`
2. Ensure >80% coverage
3. Add integration tests if needed
4. Run `hatch run all-checks` before committing
