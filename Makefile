# ============================================
# Makefile for 1D-Ensemble
# ============================================
# Ultra-modern development workflow automation

.PHONY: help install test lint format clean docs

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)🚀 1D-Ensemble Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ==========================================
# Installation
# ==========================================
install: ## Install package in development mode
	@echo "$(BLUE)📦 Installing package...$(NC)"
	hatch env create

install-all: ## Install with all dependencies
	@echo "$(BLUE)📦 Installing package with all dependencies...$(NC)"
	pip install -e ".[all]"

# ==========================================
# Testing
# ==========================================
test: ## Run tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	hatch run test

test-cov: ## Run tests with coverage
	@echo "$(BLUE)📊 Running tests with coverage...$(NC)"
	hatch run test-cov

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)⚡ Running tests in parallel...$(NC)"
	hatch run test-parallel

test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)📣 Running tests verbosely...$(NC)"
	hatch run test-verbose

coverage-html: ## Generate HTML coverage report
	@echo "$(BLUE)📈 Generating coverage report...$(NC)"
	hatch run coverage-html
	@echo "$(GREEN)✅ Coverage report: htmlcov/index.html$(NC)"

# ==========================================
# Code Quality
# ==========================================
lint: ## Run linting
	@echo "$(BLUE)🔍 Linting code...$(NC)"
	hatch run lint

lint-fix: ## Fix linting issues
	@echo "$(BLUE)🔧 Fixing linting issues...$(NC)"
	hatch run lint-fix

format: ## Format code with black
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	hatch run format

format-check: ## Check code formatting
	@echo "$(BLUE)🎨 Checking code formatting...$(NC)"
	hatch run format-check

type-check: ## Run type checking
	@echo "$(BLUE)🏷️  Type checking...$(NC)"
	hatch run type-check

quality: ## Run all quality checks
	@echo "$(BLUE)✨ Running all quality checks...$(NC)"
	hatch run quality

all-checks: ## Run all checks (format, lint, type, test)
	@echo "$(BLUE)🎯 Running all checks...$(NC)"
	hatch run all-checks

# ==========================================
# Pre-commit
# ==========================================
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)🔗 Installing pre-commit hooks...$(NC)"
	hatch run pre-commit-install

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)🏃 Running pre-commit...$(NC)"
	hatch run pre-commit-run

# ==========================================
# Cleanup
# ==========================================
clean: ## Clean build artifacts
	@echo "$(BLUE)🧹 Cleaning...$(NC)"
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean everything including environments
	@echo "$(YELLOW)⚠️  Removing all virtual environments...$(NC)"
	rm -rf .venv/ venv/
	hatch env prune

# ==========================================
# Build
# ==========================================
build: clean ## Build package
	@echo "$(BLUE)🏗️  Building package...$(NC)"
	hatch build

build-wheel: ## Build wheel only
	@echo "$(BLUE)🏗️  Building wheel...$(NC)"
	hatch build --target wheel

build-sdist: ## Build source distribution
	@echo "$(BLUE)🏗️  Building source distribution...$(NC)"
	hatch build --target sdist

# ==========================================
# Documentation
# ==========================================
docs-build: ## Build documentation
	@echo "$(BLUE)📚 Building documentation...$(NC)"
	hatch run docs:build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)🌐 Serving documentation...$(NC)"
	hatch run docs:serve

# ==========================================
# Development
# ==========================================
dev: install pre-commit-install ## Setup development environment
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

run-example: ## Run quick start example
	@echo "$(BLUE)🚀 Running example...$(NC)"
	python examples/quickstart.py

# ==========================================
# Docker
# ==========================================
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	docker build -t ensemble-1d:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	docker run -p 8501:8501 ensemble-1d:latest

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)🐳 Starting services...$(NC)"
	docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "$(BLUE)🛑 Stopping services...$(NC)"
	docker-compose down

# ==========================================
# Release
# ==========================================
version: ## Show current version
	@echo "$(BLUE)📌 Current version:$(NC)"
	@python -c "from ensemble_1d import __version__; print(__version__)"

publish-test: build ## Publish to TestPyPI
	@echo "$(YELLOW)⚠️  Publishing to TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	@echo "$(YELLOW)⚠️  Publishing to PyPI...$(NC)"
	twine upload dist/*
