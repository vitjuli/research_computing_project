"""
Inference module for making predictions with trained models.

This module provides functionality for loading trained models and
making predictions on new 5D input data.
"""

import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from numpy.typing import NDArray

from nn_interpolator.model.network import Interpolator5D
from nn_interpolator.data.loader import DataLoader


class Predictor:
    """
    Predictor for making inferences with trained Interpolator5D models.
    
    This class handles model loading and prediction, including proper
    data normalization and denormalization.
    
    Attributes:
        model: The loaded Interpolator5D model.
        data_loader: DataLoader with normalization parameters.
    
    Example:
        >>> predictor = Predictor.load("model.json")
        >>> predictions = predictor.predict(X_new)
    """
    
    def __init__(
        self,
        model: Interpolator5D,
        normalization_params: Optional[dict] = None
    ) -> None:
        """
        Initialize the predictor.
        
        Args:
            model: Trained Interpolator5D model.
            normalization_params: Optional normalization parameters.
        """
        self.model = model
        self.data_loader = DataLoader()
        
        if normalization_params is not None:
            self.data_loader.set_normalization_params(normalization_params)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Predictor":
        """
        Load a predictor from a saved model file.
        
        Args:
            filepath: Path to the saved model JSON file.
        
        Returns:
            New Predictor instance with loaded model.
        """
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            save_dict = json.load(f)
        
        model = Interpolator5D.from_params(save_dict["model"])
        normalization = save_dict.get("normalization")
        
        return cls(model, normalization)
    
    def predict(
        self,
        X: NDArray[np.float64],
        denormalize: bool = True
    ) -> NDArray[np.float64]:
        """
        Make predictions for input data.
        
        Args:
            X: Input data with shape (n_samples, 5) or (5,) for single sample.
            denormalize: Whether to denormalize the output.
        
        Returns:
            Predictions with shape (n_samples, output_dim).
        """
        # Handle single sample input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Validate input dimensions
        if X.shape[1] != 5:
            raise ValueError(f"Input must have 5 dimensions, got {X.shape[1]}")
        
        # Normalize input
        X_norm = self.data_loader.normalize_input(X)
        
        # Make prediction
        y_pred = self.model.predict(X_norm)
        
        # Denormalize output
        if denormalize:
            y_pred = self.data_loader.denormalize_targets(y_pred)
        
        return y_pred
    
    def predict_single(
        self,
        x1: float,
        x2: float,
        x3: float,
        x4: float,
        x5: float,
        denormalize: bool = True
    ) -> Union[float, List[float]]:
        """
        Make a prediction for a single 5D point.
        
        Args:
            x1, x2, x3, x4, x5: The 5 input dimensions.
            denormalize: Whether to denormalize the output.
        
        Returns:
            Predicted value(s) as float or list of floats.
        """
        X = np.array([[x1, x2, x3, x4, x5]], dtype=np.float64)
        y_pred = self.predict(X, denormalize=denormalize)
        
        if y_pred.shape[1] == 1:
            return float(y_pred[0, 0])
        return y_pred[0].tolist()
    
    def batch_predict(
        self,
        X: NDArray[np.float64],
        batch_size: int = 1000,
        denormalize: bool = True
    ) -> NDArray[np.float64]:
        """
        Make predictions in batches for large datasets.
        
        Args:
            X: Input data with shape (n_samples, 5).
            batch_size: Number of samples per batch.
            denormalize: Whether to denormalize the output.
        
        Returns:
            Predictions with shape (n_samples, output_dim).
        """
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            batch_pred = self.predict(batch, denormalize=denormalize)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)
