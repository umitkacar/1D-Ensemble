"""Model implementations for 1D-Ensemble."""

# Lazy imports to avoid requiring all dependencies
__all__ = [
    "BaseModel",
    "PyTorchModel",
    "RandomForestModel",
    "XGBoostModel",
]


def __getattr__(name: str):  # type: ignore[misc]
    """Lazy import for model classes."""
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
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
