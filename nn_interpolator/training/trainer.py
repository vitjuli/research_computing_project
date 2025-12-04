"""
Training module for neural network models.

This module provides functionality for training the Interpolator5D model
with various optimization options and training callbacks.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from nn_interpolator.model.network import Interpolator5D
from nn_interpolator.data.loader import DataLoader


class Trainer:
    """
    Trainer for Interpolator5D neural network models.
    
    This class handles the training loop, validation, and model checkpointing
    for 5D interpolation models.
    
    Attributes:
        model: The Interpolator5D model to train.
        history: Dictionary containing training history (loss, val_loss).
    
    Example:
        >>> trainer = Trainer(model)
        >>> history = trainer.fit(train_X, train_y, val_X, val_y, epochs=100)
    """
    
    def __init__(self, model: Interpolator5D) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: The Interpolator5D model to train.
        """
        self.model = model
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
        }
    
    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: Optional[NDArray[np.float64]] = None,
        y_val: Optional[NDArray[np.float64]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True,
        callbacks: Optional[List[Callable[[int, Dict[str, float]], None]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training input data with shape (n_samples, 5).
            y_train: Training targets with shape (n_samples, output_dim).
            X_val: Validation input data (optional).
            y_val: Validation targets (optional).
            epochs: Number of training epochs.
            batch_size: Mini-batch size for training.
            learning_rate: Learning rate for optimization.
            verbose: Whether to print training progress.
            callbacks: List of callback functions called after each epoch.
        
        Returns:
            Dictionary containing training history.
        """
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.model.forward(X_batch, training=True)
                
                # Backward pass and weight update
                batch_loss = self.model.backward(y_batch, y_pred, learning_rate)
                epoch_losses.append(batch_loss)
            
            # Record training loss
            train_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(train_loss)
            
            # Compute validation loss
            val_loss = None
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
                self.history["val_loss"].append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.6f}"
                print(msg)
            
            # Call callbacks
            if callbacks:
                metrics = {"loss": train_loss, "val_loss": val_loss}
                for callback in callbacks:
                    callback(epoch, metrics)
        
        return self.history
    
    def evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Input data with shape (n_samples, 5).
            y: True target values.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.model.predict(X)
        
        mse = float(np.mean((y_pred - y) ** 2))
        mae = float(np.mean(np.abs(y_pred - y)))
        rmse = float(np.sqrt(mse))
        
        # R-squared score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    
    def save_model(
        self,
        filepath: Union[str, Path],
        data_loader: Optional[DataLoader] = None
    ) -> None:
        """
        Save the model to a JSON file.
        
        Args:
            filepath: Path to save the model.
            data_loader: Optional DataLoader to save normalization parameters.
        """
        filepath = Path(filepath)
        
        save_dict = {
            "model": self.model.get_params(),
            "history": self.history,
        }
        
        if data_loader is not None:
            save_dict["normalization"] = data_loader.get_normalization_params()
        
        with open(filepath, "w") as f:
            json.dump(save_dict, f)
    
    @classmethod
    def load_model(
        cls,
        filepath: Union[str, Path]
    ) -> Tuple["Trainer", Optional[dict]]:
        """
        Load a model from a JSON file.
        
        Args:
            filepath: Path to the saved model.
        
        Returns:
            Tuple of (Trainer, normalization_params).
        """
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            save_dict = json.load(f)
        
        model = Interpolator5D.from_params(save_dict["model"])
        trainer = cls(model)
        trainer.history = save_dict.get("history", {"loss": [], "val_loss": []})
        
        normalization = save_dict.get("normalization")
        
        return trainer, normalization
