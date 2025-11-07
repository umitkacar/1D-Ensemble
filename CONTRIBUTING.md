# 🤝 Contributing to 1D-Ensemble

First off, thank you for considering contributing to 1D-Ensemble! 🎉

It's people like you that make 1D-Ensemble such a great tool. We welcome contributions from everyone, whether it's:

- 🐛 Reporting a bug
- 💬 Discussing the current state of the code
- 🔧 Submitting a fix
- 💡 Proposing new features
- 📝 Improving documentation
- 🎨 Enhancing visualizations

## 🚀 Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### 📋 Pull Request Process

1. **Fork the repo** and create your branch from `main`.
2. **Make your changes** with clear, concise commits.
3. **Add tests** if you've added code that should be tested.
4. **Update documentation** if you've changed APIs or added features.
5. **Ensure the test suite passes** (`pytest`).
6. **Make sure your code lints** (`black`, `ruff`, `mypy`).
7. **Issue that pull request!**

### 🔀 Git Workflow

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/1D-Ensemble.git
cd 1D-Ensemble

# Create a new branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git add .
git commit -m "Add some amazing feature"

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request on GitHub
```

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA for GPU support

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/1D-Ensemble.git
cd 1D-Ensemble

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,viz,mlops,deploy,explainability]"

# Install pre-commit hooks
pre-commit install
```

## ✅ Code Quality Standards

We maintain high code quality standards. Please ensure your contributions meet these requirements:

### 🎨 Code Formatting

We use **Black** for code formatting:

```bash
# Format code
black .

# Check formatting
black --check .
```

### 🔍 Linting

We use **Ruff** for fast Python linting:

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

### 🏷️ Type Checking

We use **MyPy** for static type checking:

```bash
# Run type checker
mypy ensemble_1d/ --ignore-missing-imports
```

### 🧪 Testing

We aim for >80% code coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=ensemble_1d --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## 📝 Coding Conventions

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for all function signatures
- Write docstrings in Google style format
- Keep functions focused and under 50 lines when possible
- Use meaningful variable and function names

### Example Function

```python
from typing import List, Tuple
import numpy as np


def train_ensemble_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1
) -> Tuple[object, dict]:
    """Train an ensemble model with specified parameters.

    Args:
        X_train: Training features of shape (n_samples, n_features).
        y_train: Training labels of shape (n_samples,).
        n_estimators: Number of estimators in the ensemble.
        learning_rate: Learning rate for gradient boosting.

    Returns:
        A tuple containing:
            - Trained model object
            - Dictionary of training metrics

    Raises:
        ValueError: If X_train and y_train have mismatched shapes.

    Example:
        >>> X = np.random.rand(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> model, metrics = train_ensemble_model(X, y)
    """
    # Implementation here
    pass
```

## 🐛 Bug Reports

Great bug reports tend to have:

- ✅ A quick summary and/or background
- ✅ Steps to reproduce (be specific!)
- ✅ What you expected would happen
- ✅ What actually happens
- ✅ Notes (possibly including why you think this might be happening, or things you tried that didn't work)

### Bug Report Template

```markdown
**Description:**
A clear and concise description of the bug.

**To Reproduce:**
Steps to reproduce the behavior:
1. Import module '...'
2. Call function '...'
3. See error

**Expected behavior:**
What you expected to happen.

**Actual behavior:**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10]
- Package version: [e.g., 1.0.0]

**Additional context:**
Any other context about the problem.
```

## 💡 Feature Requests

We love feature requests! Please provide:

- ✅ Clear use case
- ✅ Expected behavior
- ✅ Potential implementation approach
- ✅ Examples of similar features in other projects (if any)

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like:**
What you want to happen.

**Describe alternatives you've considered:**
Other solutions you've thought about.

**Additional context:**
Any other context, screenshots, or examples.
```

## 📚 Documentation

Good documentation is crucial! When contributing:

- Update README.md if needed
- Add docstrings to all public functions and classes
- Include usage examples in docstrings
- Update or create relevant notebook examples

## 🎯 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```bash
feat(models): add XGBoost hyperparameter tuning

Implemented Optuna-based hyperparameter optimization for XGBoost
models with cross-validation support.

Closes #123
```

```bash
fix(pytorch): resolve CUDA memory leak in training loop

Fixed memory accumulation issue by properly clearing gradients
and moving tensors to CPU after each batch.

Fixes #456
```

## 🔄 Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## 📊 Performance Benchmarks

If your PR affects performance:

- Include benchmark results
- Compare with baseline
- Document any trade-offs

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## 📞 Getting Help

- 💬 Open a [GitHub Discussion](https://github.com/umitkacar/1D-Ensemble/discussions)
- 🐛 Report bugs via [GitHub Issues](https://github.com/umitkacar/1D-Ensemble/issues)
- 📧 Email maintainers for private concerns

## 🎉 Recognition

Contributors will be:

- Listed in our README
- Mentioned in release notes
- Added to our Contributors page

## 📜 Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<div align="center">

### 🌟 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

**Happy Coding! 🚀**

</div>
