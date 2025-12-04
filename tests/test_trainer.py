"""Tests for the Trainer class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nn_interpolator.data.loader import DataLoader
from nn_interpolator.model.network import Interpolator5D
from nn_interpolator.training.trainer import Trainer


class TestTrainer:
    """Test suite for Trainer."""

    def test_fit(self):
        """Test model training."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.sum(X, axis=1)  # Simple target

        model = Interpolator5D(output_dim=1, hidden_layers=[32, 16], random_state=42)
        trainer = Trainer(model)

        history = trainer.fit(X, y, epochs=10, batch_size=16, learning_rate=0.01, verbose=False)

        assert "loss" in history
        assert len(history["loss"]) == 10

    def test_fit_with_validation(self):
        """Test training with validation data."""
        np.random.seed(42)
        X_train = np.random.rand(80, 5)
        y_train = np.sum(X_train, axis=1)
        X_val = np.random.rand(20, 5)
        y_val = np.sum(X_val, axis=1)

        model = Interpolator5D(output_dim=1, hidden_layers=[32], random_state=42)
        trainer = Trainer(model)

        history = trainer.fit(
            X_train, y_train, X_val, y_val, epochs=10, batch_size=16, verbose=False
        )

        assert "loss" in history
        assert "val_loss" in history
        assert len(history["val_loss"]) == 10

    def test_evaluate(self):
        """Test model evaluation."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.sum(X, axis=1)

        model = Interpolator5D(output_dim=1, hidden_layers=[32, 16], random_state=42)
        trainer = Trainer(model)

        # Train briefly
        trainer.fit(X, y, epochs=50, batch_size=16, learning_rate=0.01, verbose=False)

        # Evaluate
        metrics = trainer.evaluate(X, y)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0

    def test_save_load_model(self):
        """Test saving and loading a trained model."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.sum(X, axis=1)

        # Create and train model
        loader = DataLoader()
        loader.load_from_arrays(X, y)
        loader.normalize()
        train_X, train_y, val_X, val_y = loader.split(test_size=0.2, random_state=42)

        model = Interpolator5D(output_dim=1, hidden_layers=[32], random_state=42)
        trainer = Trainer(model)
        trainer.fit(train_X, train_y, val_X, val_y, epochs=10, verbose=False)

        # Get predictions
        pred_original = model.predict(val_X)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            trainer.save_model(model_path, loader)

            # Load model
            loaded_trainer, norm_params = Trainer.load_model(model_path)

            # Predictions should be identical
            pred_loaded = loaded_trainer.model.predict(val_X)

            assert np.allclose(pred_original, pred_loaded)
            assert norm_params is not None

    def test_training_loss_decreases(self):
        """Test that training loss decreases over epochs."""
        np.random.seed(42)
        X = np.random.rand(200, 5)
        y = np.sin(np.sum(X, axis=1))  # Non-linear target

        model = Interpolator5D(output_dim=1, hidden_layers=[64, 32], random_state=42)
        trainer = Trainer(model)

        history = trainer.fit(X, y, epochs=100, batch_size=32, learning_rate=0.01, verbose=False)

        # Loss should decrease
        initial_loss = np.mean(history["loss"][:5])
        final_loss = np.mean(history["loss"][-5:])

        assert final_loss < initial_loss
