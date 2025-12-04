"""Tests for the Predictor class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nn_interpolator.data.loader import DataLoader
from nn_interpolator.model.network import Interpolator5D
from nn_interpolator.training.trainer import Trainer
from nn_interpolator.inference.predictor import Predictor


class TestPredictor:
    """Test suite for Predictor."""

    @pytest.fixture
    def trained_model_path(self):
        """Create a trained model and return its path."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.sum(X, axis=1)

        loader = DataLoader()
        loader.load_from_arrays(X, y)
        loader.normalize()
        train_X, train_y, val_X, val_y = loader.split(test_size=0.2, random_state=42)

        model = Interpolator5D(output_dim=1, hidden_layers=[32], random_state=42)
        trainer = Trainer(model)
        trainer.fit(train_X, train_y, val_X, val_y, epochs=10, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            trainer.save_model(model_path, loader)
            yield model_path

    def test_load_predictor(self, trained_model_path):
        """Test loading a predictor from file."""
        predictor = Predictor.load(trained_model_path)

        assert predictor.model is not None
        assert predictor.data_loader is not None

    def test_predict(self, trained_model_path):
        """Test making predictions."""
        predictor = Predictor.load(trained_model_path)
        X = np.random.rand(10, 5)

        predictions = predictor.predict(X)

        assert predictions.shape == (10, 1)

    def test_predict_single_sample(self, trained_model_path):
        """Test prediction for single sample."""
        predictor = Predictor.load(trained_model_path)
        X = np.random.rand(5)  # 1D array

        predictions = predictor.predict(X)

        assert predictions.shape == (1, 1)

    def test_predict_single(self, trained_model_path):
        """Test predict_single method."""
        predictor = Predictor.load(trained_model_path)

        result = predictor.predict_single(0.1, 0.2, 0.3, 0.4, 0.5)

        assert isinstance(result, float)

    def test_batch_predict(self, trained_model_path):
        """Test batch prediction."""
        predictor = Predictor.load(trained_model_path)
        X = np.random.rand(100, 5)

        predictions = predictor.batch_predict(X, batch_size=10)

        assert predictions.shape == (100, 1)

    def test_invalid_dimensions(self, trained_model_path):
        """Test that invalid input dimensions raise an error."""
        predictor = Predictor.load(trained_model_path)
        X = np.random.rand(10, 3)  # Wrong dimensions

        with pytest.raises(ValueError, match="must have 5 dimensions"):
            predictor.predict(X)
