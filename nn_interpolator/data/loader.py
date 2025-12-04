"""
Data loading module for 5D numerical datasets.

This module provides functionality to load, preprocess, and prepare
5-dimensional numerical data for neural network training.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray


class DataLoader:
    """
    DataLoader for 5D numerical datasets.
    
    This class handles loading, preprocessing, and splitting of 5D numerical
    data for training neural network interpolation models.
    
    Attributes:
        data: The loaded input data with shape (n_samples, 5).
        targets: The target values with shape (n_samples, output_dim).
        train_data: Training input data after splitting.
        train_targets: Training targets after splitting.
        val_data: Validation input data after splitting.
        val_targets: Validation targets after splitting.
    
    Example:
        >>> loader = DataLoader()
        >>> loader.load_from_arrays(X, y)
        >>> train_X, train_y, val_X, val_y = loader.split(test_size=0.2)
    """
    
    def __init__(self) -> None:
        """Initialize the DataLoader."""
        self.data: Optional[NDArray[np.float64]] = None
        self.targets: Optional[NDArray[np.float64]] = None
        self.train_data: Optional[NDArray[np.float64]] = None
        self.train_targets: Optional[NDArray[np.float64]] = None
        self.val_data: Optional[NDArray[np.float64]] = None
        self.val_targets: Optional[NDArray[np.float64]] = None
        self._mean: Optional[NDArray[np.float64]] = None
        self._std: Optional[NDArray[np.float64]] = None
        self._target_mean: Optional[NDArray[np.float64]] = None
        self._target_std: Optional[NDArray[np.float64]] = None
    
    def load_from_arrays(
        self,
        data: NDArray[np.float64],
        targets: NDArray[np.float64]
    ) -> "DataLoader":
        """
        Load data from numpy arrays.
        
        Args:
            data: Input data array with shape (n_samples, 5).
            targets: Target values array with shape (n_samples,) or (n_samples, output_dim).
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If data doesn't have 5 input dimensions.
        """
        if data.ndim != 2 or data.shape[1] != 5:
            raise ValueError(f"Data must have shape (n_samples, 5), got {data.shape}")
        
        self.data = data.astype(np.float64)
        self.targets = targets.astype(np.float64)
        
        if self.targets.ndim == 1:
            self.targets = self.targets.reshape(-1, 1)
        
        return self
    
    def load_from_csv(self, filepath: str, target_columns: int = 1) -> "DataLoader":
        """
        Load data from a CSV file.
        
        The CSV file should have 5 input columns followed by target column(s).
        
        Args:
            filepath: Path to the CSV file.
            target_columns: Number of target columns (default: 1).
        
        Returns:
            Self for method chaining.
        """
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        self.data = data[:, :5].astype(np.float64)
        self.targets = data[:, 5:5 + target_columns].astype(np.float64)
        
        if self.targets.ndim == 1:
            self.targets = self.targets.reshape(-1, 1)
        
        return self
    
    def normalize(self, fit: bool = True) -> "DataLoader":
        """
        Normalize the data using z-score normalization.
        
        Args:
            fit: If True, compute mean and std from current data.
                 If False, use previously computed values.
        
        Returns:
            Self for method chaining.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_arrays or load_from_csv first.")
        
        if fit:
            self._mean = np.mean(self.data, axis=0)
            self._std = np.std(self.data, axis=0)
            self._std[self._std == 0] = 1.0  # Avoid division by zero
            
            self._target_mean = np.mean(self.targets, axis=0)
            self._target_std = np.std(self.targets, axis=0)
            self._target_std[self._target_std == 0] = 1.0
        
        self.data = (self.data - self._mean) / self._std
        self.targets = (self.targets - self._target_mean) / self._target_std
        
        return self
    
    def denormalize_targets(
        self,
        targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Denormalize target values back to original scale.
        
        Args:
            targets: Normalized target values.
        
        Returns:
            Denormalized target values.
        """
        if self._target_mean is None or self._target_std is None:
            return targets
        return targets * self._target_std + self._target_mean
    
    def normalize_input(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize input data using stored statistics.
        
        Args:
            data: Input data to normalize.
        
        Returns:
            Normalized input data.
        """
        if self._mean is None or self._std is None:
            return data
        return (data - self._mean) / self._std
    
    def split(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64]
    ]:
        """
        Split data into training and validation sets.
        
        Args:
            test_size: Fraction of data to use for validation (0.0 to 1.0).
            shuffle: Whether to shuffle data before splitting.
            random_state: Random seed for reproducibility.
        
        Returns:
            Tuple of (train_data, train_targets, val_data, val_targets).
        """
        if self.data is None or self.targets is None:
            raise ValueError("No data loaded.")
        
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        
        split_idx = int(n_samples * (1 - test_size))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        self.train_data = self.data[train_indices]
        self.train_targets = self.targets[train_indices]
        self.val_data = self.data[val_indices]
        self.val_targets = self.targets[val_indices]
        
        return (
            self.train_data,
            self.train_targets,
            self.val_data,
            self.val_targets
        )
    
    def get_normalization_params(self) -> dict:
        """
        Get normalization parameters for saving/loading.
        
        Returns:
            Dictionary containing normalization statistics.
        """
        return {
            "input_mean": self._mean.tolist() if self._mean is not None else None,
            "input_std": self._std.tolist() if self._std is not None else None,
            "target_mean": self._target_mean.tolist() if self._target_mean is not None else None,
            "target_std": self._target_std.tolist() if self._target_std is not None else None,
        }
    
    def set_normalization_params(self, params: dict) -> "DataLoader":
        """
        Set normalization parameters from saved values.
        
        Args:
            params: Dictionary containing normalization statistics.
        
        Returns:
            Self for method chaining.
        """
        if params.get("input_mean") is not None:
            self._mean = np.array(params["input_mean"])
        if params.get("input_std") is not None:
            self._std = np.array(params["input_std"])
        if params.get("target_mean") is not None:
            self._target_mean = np.array(params["target_mean"])
        if params.get("target_std") is not None:
            self._target_std = np.array(params["target_std"])
        return self
