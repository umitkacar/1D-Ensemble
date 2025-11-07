"""
🚀 1D-Ensemble: Modern Machine Learning Framework
==================================================

A production-ready ensemble learning framework for 1D data with support for
XGBoost, PyTorch, and Scikit-learn models.

Example:
    >>> from ensemble_1d import EnsembleModel, XGBoostModel, PyTorchModel
    >>> models = [XGBoostModel(), PyTorchModel()]
    >>> ensemble = EnsembleModel(models=models)
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "Umit Kacar"
__email__ = "umit@example.com"
__license__ = "MIT"

# Import main classes
from ensemble_1d.models.base import BaseModel
from ensemble_1d.models.xgboost_model import XGBoostModel
from ensemble_1d.models.pytorch_model import PyTorchModel
from ensemble_1d.models.rf_model import RandomForestModel
from ensemble_1d.fusion.ensemble import EnsembleModel

__all__ = [
    "BaseModel",
    "XGBoostModel",
    "PyTorchModel",
    "RandomForestModel",
    "EnsembleModel",
    "__version__",
]
