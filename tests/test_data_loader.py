"""Tests for the DataLoader class."""

import numpy as np
import pytest

from nn_interpolator.data.loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader."""

    def test_load_from_arrays(self):
        """Test loading data from numpy arrays."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        loader = DataLoader()
        loader.load_from_arrays(X, y)

        assert loader.data is not None
        assert loader.targets is not None
        assert loader.data.shape == (100, 5)
        assert loader.targets.shape == (100, 1)

    def test_load_from_arrays_invalid_dimensions(self):
        """Test that loading data with wrong dimensions raises an error."""
        X = np.random.rand(100, 3)  # Wrong number of dimensions
        y = np.random.rand(100)

        loader = DataLoader()
        with pytest.raises(ValueError, match="must have shape"):
            loader.load_from_arrays(X, y)

    def test_normalize(self):
        """Test data normalization."""
        X = np.random.rand(100, 5) * 100  # Large scale
        y = np.random.rand(100) * 1000

        loader = DataLoader()
        loader.load_from_arrays(X, y)
        loader.normalize()

        # Check that data is approximately normalized
        assert np.abs(np.mean(loader.data)) < 1e-10
        assert np.abs(np.std(loader.data) - 1.0) < 0.1

    def test_split(self):
        """Test data splitting."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        loader = DataLoader()
        loader.load_from_arrays(X, y)
        train_X, train_y, val_X, val_y = loader.split(test_size=0.2, random_state=42)

        assert train_X.shape[0] == 80
        assert val_X.shape[0] == 20
        assert train_y.shape[0] == 80
        assert val_y.shape[0] == 20

    def test_normalize_input(self):
        """Test normalizing new input data."""
        X = np.random.rand(100, 5) * 100
        y = np.random.rand(100)

        loader = DataLoader()
        loader.load_from_arrays(X, y)
        loader.normalize()

        # Normalize new data
        X_new = np.random.rand(10, 5) * 100
        X_norm = loader.normalize_input(X_new)

        assert X_norm.shape == (10, 5)

    def test_denormalize_targets(self):
        """Test denormalizing target values."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100) * 1000

        loader = DataLoader()
        loader.load_from_arrays(X, y)
        original_mean = np.mean(y)
        loader.normalize()

        # Create some normalized predictions
        y_norm = np.zeros((10, 1))
        y_denorm = loader.denormalize_targets(y_norm)

        # Should be close to original mean
        assert np.allclose(y_denorm, original_mean, atol=1.0)

    def test_get_set_normalization_params(self):
        """Test saving and loading normalization parameters."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        loader = DataLoader()
        loader.load_from_arrays(X, y)
        loader.normalize()

        params = loader.get_normalization_params()

        # Create new loader and set params
        new_loader = DataLoader()
        new_loader.set_normalization_params(params)

        # Should have same normalization behavior
        X_test = np.random.rand(5, 5)
        X_norm1 = loader.normalize_input(X_test)
        X_norm2 = new_loader.normalize_input(X_test)

        assert np.allclose(X_norm1, X_norm2)
