"""Tests for the Interpolator5D model."""

import numpy as np
import pytest

from nn_interpolator.model.network import Interpolator5D


class TestInterpolator5D:
    """Test suite for Interpolator5D model."""

    def test_initialization(self):
        """Test model initialization."""
        model = Interpolator5D(output_dim=1, hidden_layers=[64, 32])

        assert model.input_dim == 5
        assert model.output_dim == 1
        assert model.hidden_layers == [64, 32]
        assert len(model.weights) == 3  # 2 hidden + 1 output
        assert len(model.biases) == 3

    def test_forward_pass(self):
        """Test forward pass through the network."""
        model = Interpolator5D(output_dim=1, hidden_layers=[64, 32], random_state=42)
        X = np.random.rand(10, 5)

        output = model.forward(X)

        assert output.shape == (10, 1)

    def test_predict(self):
        """Test prediction method."""
        model = Interpolator5D(output_dim=1, hidden_layers=[32, 16], random_state=42)
        X = np.random.rand(5, 5)

        predictions = model.predict(X)

        assert predictions.shape == (5, 1)

    def test_backward_pass(self):
        """Test backward pass updates weights."""
        model = Interpolator5D(output_dim=1, hidden_layers=[16], random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.rand(10, 1)

        # Get initial weights
        initial_weights = [w.copy() for w in model.weights]

        # Forward and backward pass
        y_pred = model.forward(X, training=True)
        loss = model.backward(y, y_pred, learning_rate=0.1)

        # Weights should have changed
        for i, (w_init, w_new) in enumerate(zip(initial_weights, model.weights)):
            assert not np.allclose(w_init, w_new), f"Layer {i} weights unchanged"

        assert isinstance(loss, float)
        assert loss > 0

    def test_training_reduces_loss(self):
        """Test that training reduces the loss."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.sum(X, axis=1, keepdims=True)  # Simple linear target

        model = Interpolator5D(output_dim=1, hidden_layers=[32, 16], random_state=42)

        # Initial prediction and loss
        y_pred_initial = model.forward(X, training=True)
        loss_initial = model.backward(y, y_pred_initial, learning_rate=0.01)

        # Train for a few iterations
        for _ in range(100):
            y_pred = model.forward(X, training=True)
            loss = model.backward(y, y_pred, learning_rate=0.01)

        # Loss should be lower
        assert loss < loss_initial

    def test_get_set_params(self):
        """Test saving and loading model parameters."""
        model = Interpolator5D(output_dim=2, hidden_layers=[32, 16], random_state=42)
        X = np.random.rand(5, 5)

        # Get predictions with original model
        pred_original = model.predict(X)

        # Save and load params
        params = model.get_params()
        new_model = Interpolator5D.from_params(params)

        # Predictions should be identical
        pred_loaded = new_model.predict(X)

        assert np.allclose(pred_original, pred_loaded)

    def test_multi_output(self):
        """Test model with multiple outputs."""
        model = Interpolator5D(output_dim=3, hidden_layers=[32], random_state=42)
        X = np.random.rand(10, 5)

        output = model.predict(X)

        assert output.shape == (10, 3)
