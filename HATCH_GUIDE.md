# 🚀 Hatch Development Guide

Complete guide for using Hatch with the 1D-Ensemble project.

## 📦 What is Hatch?

Hatch is a modern Python project manager that handles:
- ✅ Environment management
- ✅ Package building
- ✅ Version management
- ✅ Script execution
- ✅ Testing workflows

## 🛠️ Installation

```bash
# Install Hatch
pip install hatch

# Or with pipx (recommended)
pipx install hatch
```

## 🎯 Quick Start

```bash
# Create default environment and install dependencies
hatch env create

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Format code
hatch run format

# Lint code
hatch run lint

# Type check
hatch run type-check

# Run all quality checks
hatch run quality
```

## 🌍 Hatch Environments

### Default Environment

```bash
# Enter the default environment shell
hatch shell

# Run command in default environment
hatch run <command>
```

### Test Environment

```bash
# Run tests with specific Python version
hatch run test.py3.11:test

# Run tests across all Python versions
hatch run test:test
```

### Documentation Environment

```bash
# Build docs
hatch run docs:build

# Serve docs locally
hatch run docs:serve
```

## 📜 Available Scripts

All scripts are defined in `pyproject.toml` under `[tool.hatch.envs.default.scripts]`:

### Testing Scripts

| Command | Description |
|---------|-------------|
| `hatch run test` | Run all tests |
| `hatch run test-cov` | Run tests with coverage report |
| `hatch run test-parallel` | Run tests in parallel (faster) |
| `hatch run test-verbose` | Run tests with verbose output |

### Code Quality Scripts

| Command | Description |
|---------|-------------|
| `hatch run lint` | Run Ruff linter |
| `hatch run lint-fix` | Auto-fix linting issues |
| `hatch run format` | Format code with Black |
| `hatch run format-check` | Check code formatting |
| `hatch run type-check` | Run MyPy type checking |
| `hatch run quality` | Run all quality checks |
| `hatch run all-checks` | Run everything (format, lint, type, test) |

### Coverage Scripts

| Command | Description |
|---------|-------------|
| `hatch run coverage-report` | Show coverage report |
| `hatch run coverage-html` | Generate HTML coverage report |
| `hatch run coverage-erase` | Erase coverage data |

### Pre-commit Scripts

| Command | Description |
|---------|-------------|
| `hatch run pre-commit-install` | Install pre-commit hooks |
| `hatch run pre-commit-run` | Run pre-commit on all files |

## 🏗️ Building & Publishing

### Build Package

```bash
# Build both wheel and source distribution
hatch build

# Build wheel only
hatch build --target wheel

# Build source distribution only
hatch build --target sdist

# Clean before building
hatch clean
hatch build
```

### Version Management

```bash
# Show current version
hatch version

# Bump version
hatch version patch  # 1.0.0 -> 1.0.1
hatch version minor  # 1.0.0 -> 1.1.0
hatch version major  # 1.0.0 -> 2.0.0

# Set specific version
hatch version "1.2.3"
```

## 🧪 Advanced Testing

### Run Tests with Markers

```bash
# Run only unit tests
hatch run test -m unit

# Skip slow tests
hatch run test -m "not slow"

# Run specific test file
hatch run test tests/test_models.py

# Run specific test
hatch run test tests/test_models.py::TestXGBoostModel::test_fit
```

### Run Tests Across Python Versions

```bash
# Test with all configured Python versions
hatch run test:test

# Test with specific Python version
hatch run test.py3.11:test
hatch run test.py3.9:test
```

### Parallel Testing

```bash
# Run tests in parallel (uses pytest-xdist)
hatch run test-parallel

# Specify number of workers
hatch run pytest -n 4
```

## 🎨 Code Quality Workflow

### Complete Quality Check

```bash
# Run all quality checks at once
hatch run all-checks
```

This runs:
1. ✅ Black formatting check
2. ✅ Ruff linting
3. ✅ MyPy type checking
4. ✅ Pytest with coverage

### Individual Checks

```bash
# Check formatting (doesn't modify files)
hatch run format-check

# Format code
hatch run format

# Lint and auto-fix
hatch run lint-fix

# Type check
hatch run type-check
```

## 🔗 Pre-commit Integration

```bash
# Install hooks (run once)
hatch run pre-commit-install

# Manually run all hooks
hatch run pre-commit-run

# Pre-commit will run automatically on git commit
git commit -m "Your message"
```

## 🐳 Docker Integration

```bash
# Build Docker image
docker build -t ensemble-1d:latest .

# Or use Makefile
make docker-build

# Run with docker-compose
make docker-compose-up
```

## 💡 Pro Tips

### 1. Shell vs Run

```bash
# Enter environment shell (interactive)
hatch shell
> pytest
> python examples/quickstart.py
> exit

# Run single command (non-interactive)
hatch run test
```

### 2. Environment Management

```bash
# List all environments
hatch env show

# Remove all environments
hatch env prune

# Create specific environment
hatch env create test

# Remove specific environment
hatch env remove test
```

### 3. Matrix Testing

The `pyproject.toml` defines a test matrix for Python 3.8-3.12:

```bash
# Run tests on all Python versions
hatch run test:test
```

### 4. Custom Scripts

Add your own scripts to `pyproject.toml`:

```toml
[tool.hatch.envs.default.scripts]
my-script = "python my_script.py {args}"
```

Then run:

```bash
hatch run my-script --arg1 value1
```

## 🆚 Hatch vs Other Tools

| Feature | Hatch | Poetry | Pipenv |
|---------|-------|--------|--------|
| Build backend | ✅ | ✅ | ❌ |
| Environment management | ✅ | ✅ | ✅ |
| Script runner | ✅ | ❌ | ❌ |
| Version management | ✅ | ✅ | ❌ |
| Python version matrix | ✅ | ❌ | ❌ |
| Speed | ⚡⚡⚡ | ⚡⚡ | ⚡ |

## 📚 Resources

- [Official Hatch Documentation](https://hatch.pypa.io/)
- [PyPA Packaging Guide](https://packaging.python.org/)
- [Project pyproject.toml](pyproject.toml)

## 🔧 Troubleshooting

### Environment Issues

```bash
# Remove all environments and recreate
hatch env prune
hatch env create
```

### Missing Dependencies

```bash
# Sync environment with pyproject.toml
hatch env remove default
hatch env create
```

### Test Failures

```bash
# Run tests verbosely to see details
hatch run test-verbose

# Run specific failing test
hatch run test tests/test_models.py::test_name -vv
```

---

**Happy Hacking with Hatch! 🚀**
