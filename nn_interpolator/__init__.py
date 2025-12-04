"""
NN Interpolator - A neural network package for 5D numerical dataset interpolation.

This package provides tools for loading data, training models, and making predictions
on 5-dimensional numerical datasets.
"""

__version__ = "0.1.0"

from nn_interpolator.data.loader import DataLoader
from nn_interpolator.model.network import Interpolator5D
from nn_interpolator.training.trainer import Trainer
from nn_interpolator.inference.predictor import Predictor

__all__ = ["DataLoader", "Interpolator5D", "Trainer", "Predictor", "__version__"]
