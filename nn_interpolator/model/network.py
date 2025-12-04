"""
Neural network model for 5D interpolation.

This module provides a fully connected neural network architecture
designed for interpolating 5-dimensional numerical data.
"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative of ReLU activation function."""
    return (x > 0).astype(np.float64)


class Interpolator5D:
    """
    Neural network for interpolating 5D numerical datasets.
    
    This class implements a fully connected neural network with configurable
    hidden layers for learning continuous mappings from 5D input space to
    target output space.
    
    Attributes:
        input_dim: Number of input dimensions (fixed at 5).
        output_dim: Number of output dimensions.
        hidden_layers: List of hidden layer sizes.
        weights: List of weight matrices for each layer.
        biases: List of bias vectors for each layer.
    
    Example:
        >>> model = Interpolator5D(output_dim=1, hidden_layers=[64, 32])
        >>> predictions = model.forward(X)
    """
    
    def __init__(
        self,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the neural network.
        
        Args:
            output_dim: Number of output dimensions.
            hidden_layers: List of hidden layer sizes. Defaults to [64, 64, 32].
            random_state: Random seed for weight initialization.
        """
        self.input_dim = 5
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 64, 32]
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.weights: List[NDArray[np.float64]] = []
        self.biases: List[NDArray[np.float64]] = []
        
        self._initialize_weights()
        
        # Cache for backpropagation
        self._activations: List[NDArray[np.float64]] = []
        self._pre_activations: List[NDArray[np.float64]] = []
    
    def _initialize_weights(self) -> None:
        """Initialize weights using He initialization."""
        layer_sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            std = np.sqrt(2.0 / layer_sizes[i])
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std
            bias = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weight.astype(np.float64))
            self.biases.append(bias.astype(np.float64))
    
    def forward(
        self,
        X: NDArray[np.float64],
        training: bool = False
    ) -> NDArray[np.float64]:
        """
        Forward pass through the network.
        
        Args:
            X: Input data with shape (n_samples, 5).
            training: If True, cache activations for backpropagation.
        
        Returns:
            Network output with shape (n_samples, output_dim).
        """
        current = X.astype(np.float64)
        
        if training:
            self._activations = [current]
            self._pre_activations = []
        
        # Hidden layers with ReLU activation
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            if training:
                self._pre_activations.append(z)
            current = relu(z)
            if training:
                self._activations.append(current)
        
        # Output layer (linear activation)
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        if training:
            self._pre_activations.append(z)
            self._activations.append(z)
        
        return z
    
    def backward(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        learning_rate: float = 0.001
    ) -> float:
        """
        Backward pass (backpropagation) and weight update.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values from forward pass.
            learning_rate: Learning rate for gradient descent.
        
        Returns:
            Mean squared error loss.
        """
        n_samples = y_true.shape[0]
        
        # Compute loss
        loss = np.mean((y_pred - y_true) ** 2)
        
        # Compute gradients
        delta = 2 * (y_pred - y_true) / n_samples
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dW = np.dot(self._activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Gradient for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * relu_derivative(self._pre_activations[i - 1])
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
        
        return float(loss)
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Make predictions for input data.
        
        Args:
            X: Input data with shape (n_samples, 5).
        
        Returns:
            Predictions with shape (n_samples, output_dim).
        """
        return self.forward(X, training=False)
    
    def get_params(self) -> dict:
        """
        Get model parameters for saving.
        
        Returns:
            Dictionary containing model configuration and weights.
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
    
    def set_params(self, params: dict) -> "Interpolator5D":
        """
        Load model parameters.
        
        Args:
            params: Dictionary containing model configuration and weights.
        
        Returns:
            Self for method chaining.
        """
        self.input_dim = params["input_dim"]
        self.output_dim = params["output_dim"]
        self.hidden_layers = params["hidden_layers"]
        self.weights = [np.array(w, dtype=np.float64) for w in params["weights"]]
        self.biases = [np.array(b, dtype=np.float64) for b in params["biases"]]
        return self
    
    @classmethod
    def from_params(cls, params: dict) -> "Interpolator5D":
        """
        Create a model from saved parameters.
        
        Args:
            params: Dictionary containing model configuration and weights.
        
        Returns:
            New Interpolator5D instance with loaded parameters.
        """
        model = cls(
            output_dim=params["output_dim"],
            hidden_layers=params["hidden_layers"]
        )
        model.set_params(params)
        return model
