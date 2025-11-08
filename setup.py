"""Setup script for 1D-Ensemble package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ensemble-1d",
    version="1.0.0",
    author="Umit Kacar",
    author_email="umit@example.com",
    description="Modern Machine Learning Ensemble Framework for 1D Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/umitkacar/1D-Ensemble",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
        "typing-extensions>=4.0.0;python_version<'3.10'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "ruff>=0.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "viz": [
            "plotly>=5.17.0",
            "seaborn>=0.12.0",
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
        ],
        "mlops": [
            "mlflow>=2.8.0",
            "wandb>=0.15.0",
            "optuna>=3.4.0",
            "ray[tune]>=2.7.0",
        ],
        "deploy": [
            "onnx>=1.15.0",
            "onnxruntime>=1.16.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
)
