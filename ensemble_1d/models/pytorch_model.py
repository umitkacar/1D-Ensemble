"""PyTorch neural network model implementation."""

from typing import Any, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ensemble_1d.models.base import BaseModel


class NeuralNet(nn.Module):
    """Feed-forward neural network.

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden units.
        num_classes: Number of output classes.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
    ) -> None:
        """Initialize neural network."""
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(hidden_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output logits.
        """
        return self.network(x)


class PyTorchModel(BaseModel):
    """PyTorch neural network model.

    A flexible deep learning model using PyTorch.

    Args:
        hidden_size: Number of hidden units. Default: 128.
        num_layers: Number of hidden layers. Default: 2.
        learning_rate: Learning rate for optimizer. Default: 0.001.
        batch_size: Batch size for training. Default: 32.
        num_epochs: Number of training epochs. Default: 10.
        device: Device to use ('cuda' or 'cpu'). Default: auto-detect.
        **kwargs: Additional parameters.

    Example:
        >>> model = PyTorchModel(hidden_size=128, num_epochs=20)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 10,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PyTorch model."""
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            **kwargs,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[NeuralNet] = None
        self.num_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchModel":
        """Train the PyTorch model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        input_size = X.shape[1]
        self.num_classes = len(np.unique(y))

        # Initialize model
        self.model = NeuralNet(
            input_size=input_size,
            hidden_size=self.params["hidden_size"],
            num_classes=self.num_classes,
            num_layers=self.params["num_layers"],
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        # Training loop
        self.model.train()
        for epoch in range(self.params["num_epochs"]):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).

        Raises:
            ValueError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, input_size: int) -> "PyTorchModel":
        """Load model from disk.

        Args:
            path: File path to load the model from.
            input_size: Number of input features.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If num_classes is not set.
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be set before loading")

        self.model = NeuralNet(
            input_size=input_size,
            hidden_size=self.params["hidden_size"],
            num_classes=self.num_classes,
            num_layers=self.params["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(torch.load(path))
        return self
