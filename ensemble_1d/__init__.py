"""
🚀 1D-Ensemble: Modern Machine Learning Framework
==================================================

A production-ready ensemble learning framework for 1D data with support for
XGBoost, PyTorchT, and Scikit-learn models.

Example:
    >>> from ensemble_1d import EnsembleModel
    >>> from ensemble_1d.models import XGBoostModel, RandomForestModel
    >>> models = [XGBoostModel(), RandomForestModel()]
    >>> ensemble = EnsembleModel(models=models)
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "Umit Kacar"
__email__ = "umit@example.com"
__license__ = "MIT"

# Lazy imports to avoid heavy dependencies on package import
__all__ = [
    "BaseModel",
    "XGBoostModel",
    "PyTorchModel",
    "RandomForestModel",
    "EnsembleModel",
    "__version__",
]


# Use __getattr__ for lazy loading
def __getattr__(name: str):  # type: ignore[misc]
    """Lazy import for heavy dependencies."""
    if name == "BaseModel":
        from ensemble_1d.models.base import BaseModel

        return BaseModel
    if name == "XGBoostModel":
        from ensemble_1d.models.xgboost_model import XGBoostModel

        return XGBoostModel
    if name == "PyTorchModel":
        from ensemble_1d.models.pytorch_model import PyTorchModel

        return PyTorchModel
    if name == "RandomForestModel":
        from ensemble_1d.models.rf_model import RandomForestModel

        return RandomForestModel
    if name == "EnsembleModel":
        from ensemble_1d.fusion.ensemble import EnsembleModel

        return EnsembleModel
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
