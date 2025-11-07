# 📚 Examples

This directory contains example scripts demonstrating various features of the 1D-Ensemble framework.

## 🚀 Quick Start Examples

### Basic Usage

Run the quick start example:

```bash
python examples/quickstart.py
```

This demonstrates:
- Creating an ensemble with multiple models
- Training on synthetic data
- Evaluating performance
- Making predictions

## 📖 Available Examples

| Example | Description | Features |
|---------|-------------|----------|
| `quickstart.py` | Basic ensemble usage | XGBoost, PyTorch, Random Forest |
| `hyperparameter_tuning.py` | Optuna optimization | Automated hyperparameter search |
| `model_export.py` | Model serialization | Save/load models, ONNX export |
| `custom_fusion.py` | Custom fusion methods | Advanced ensemble techniques |

## 🎯 Running Examples

All examples can be run directly:

```bash
# Run any example
python examples/<example_name>.py
```

## 📝 Requirements

Make sure you have installed the package with all dependencies:

```bash
pip install -e ".[all]"
```

## 💡 Learn More

- Check the [documentation](../docs/) for detailed guides
- Explore [notebooks](../notebooks/) for interactive examples
- Read the [API reference](../docs/api/) for detailed information
