"""
FastAPI application for NN Interpolator.

This module provides REST endpoints for training and prediction
with the Interpolator5D neural network model.
"""

import io
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nn_interpolator import DataLoader, Interpolator5D, Predictor, Trainer

app = FastAPI(
    title="NN Interpolator API",
    description="REST API for training and querying 5D neural network interpolation models",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for models
MODELS_DIR = Path("/tmp/nn_models")
MODELS_DIR.mkdir(exist_ok=True)
models_registry: Dict[str, dict] = {}


# Pydantic models for API
class TrainingConfig(BaseModel):
    """Configuration for model training."""

    hidden_layers: List[int] = Field(
        default=[64, 64, 32], description="List of hidden layer sizes"
    )
    epochs: int = Field(default=100, ge=1, le=10000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size for training")
    learning_rate: float = Field(
        default=0.001, gt=0, le=1.0, description="Learning rate"
    )
    test_size: float = Field(
        default=0.2, gt=0, lt=1.0, description="Fraction of data for validation"
    )
    normalize: bool = Field(default=True, description="Whether to normalize data")


class TrainingData(BaseModel):
    """Training data input."""

    X: List[List[float]] = Field(..., description="Input data with 5 features per sample")
    y: List[float] = Field(..., description="Target values")
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class PredictionInput(BaseModel):
    """Input for prediction."""

    X: List[List[float]] = Field(
        ..., description="Input data with 5 features per sample"
    )


class SinglePredictionInput(BaseModel):
    """Input for single point prediction."""

    x1: float = Field(..., description="First dimension value")
    x2: float = Field(..., description="Second dimension value")
    x3: float = Field(..., description="Third dimension value")
    x4: float = Field(..., description="Fourth dimension value")
    x5: float = Field(..., description="Fifth dimension value")


class ModelInfo(BaseModel):
    """Model information response."""

    model_id: str
    hidden_layers: List[int]
    output_dim: int
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None


class TrainingResponse(BaseModel):
    """Response from training endpoint."""

    model_id: str
    message: str
    final_loss: float
    final_val_loss: Optional[float] = None
    history: Dict[str, List[float]]


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""

    predictions: List[List[float]]


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "NN Interpolator API",
        "version": "0.1.0",
        "description": "REST API for 5D neural network interpolation",
    }


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/train", response_model=TrainingResponse)
async def train_model(data: TrainingData) -> TrainingResponse:
    """
    Train a new interpolation model.

    Args:
        data: Training data and configuration.

    Returns:
        Training response with model ID and training history.
    """
    # Validate input data
    X = np.array(data.X, dtype=np.float64)
    y = np.array(data.y, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 5:
        raise HTTPException(
            status_code=400,
            detail=f"Input X must have shape (n_samples, 5), got {X.shape}",
        )

    if len(X) != len(y):
        raise HTTPException(
            status_code=400,
            detail=f"X and y must have same length, got {len(X)} and {len(y)}",
        )

    # Create data loader
    loader = DataLoader()
    loader.load_from_arrays(X, y)

    if data.config.normalize:
        loader.normalize()

    train_X, train_y, val_X, val_y = loader.split(
        test_size=data.config.test_size, random_state=42
    )

    # Create and train model
    model = Interpolator5D(
        output_dim=1, hidden_layers=data.config.hidden_layers, random_state=42
    )
    trainer = Trainer(model)

    history = trainer.fit(
        train_X,
        train_y,
        val_X,
        val_y,
        epochs=data.config.epochs,
        batch_size=data.config.batch_size,
        learning_rate=data.config.learning_rate,
        verbose=False,
    )

    # Save model
    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.json"
    trainer.save_model(model_path, loader)

    # Register model
    models_registry[model_id] = {
        "path": str(model_path),
        "hidden_layers": data.config.hidden_layers,
        "output_dim": 1,
        "training_loss": history["loss"][-1] if history["loss"] else None,
        "validation_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }

    return TrainingResponse(
        model_id=model_id,
        message="Model trained successfully",
        final_loss=history["loss"][-1] if history["loss"] else 0.0,
        final_val_loss=history["val_loss"][-1] if history["val_loss"] else None,
        history=history,
    )


@app.post("/train/csv", response_model=TrainingResponse)
async def train_from_csv(
    file: UploadFile = File(...),
    hidden_layers: str = "64,64,32",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    test_size: float = 0.2,
    normalize: bool = True,
) -> TrainingResponse:
    """
    Train a model from an uploaded CSV file.

    The CSV should have 5 input columns and 1 target column.

    Args:
        file: CSV file with training data.
        hidden_layers: Comma-separated list of hidden layer sizes.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimization.
        test_size: Fraction of data for validation.
        normalize: Whether to normalize data.

    Returns:
        Training response with model ID and training history.
    """
    # Parse hidden layers
    try:
        layers = [int(x.strip()) for x in hidden_layers.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hidden_layers format")

    # Read CSV file
    try:
        contents = await file.read()
        data = np.loadtxt(io.StringIO(contents.decode("utf-8")), delimiter=",", skiprows=1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    if data.shape[1] < 6:
        raise HTTPException(
            status_code=400, detail="CSV must have at least 6 columns (5 inputs + 1 target)"
        )

    X = data[:, :5]
    y = data[:, 5]

    # Create training data and delegate to main train endpoint
    training_data = TrainingData(
        X=X.tolist(),
        y=y.tolist(),
        config=TrainingConfig(
            hidden_layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            test_size=test_size,
            normalize=normalize,
        ),
    )

    return await train_model(training_data)


@app.get("/models", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    """
    List all trained models.

    Returns:
        List of model information.
    """
    return [
        ModelInfo(
            model_id=model_id,
            hidden_layers=info["hidden_layers"],
            output_dim=info["output_dim"],
            training_loss=info.get("training_loss"),
            validation_loss=info.get("validation_loss"),
        )
        for model_id, info in models_registry.items()
    ]


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """
    Get information about a specific model.

    Args:
        model_id: The model identifier.

    Returns:
        Model information.
    """
    if model_id not in models_registry:
        raise HTTPException(status_code=404, detail="Model not found")

    info = models_registry[model_id]
    return ModelInfo(
        model_id=model_id,
        hidden_layers=info["hidden_layers"],
        output_dim=info["output_dim"],
        training_loss=info.get("training_loss"),
        validation_loss=info.get("validation_loss"),
    )


@app.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, data: PredictionInput) -> PredictionResponse:
    """
    Make predictions with a trained model.

    Args:
        model_id: The model identifier.
        data: Input data for prediction.

    Returns:
        Predictions for input data.
    """
    if model_id not in models_registry:
        raise HTTPException(status_code=404, detail="Model not found")

    # Load model
    model_path = models_registry[model_id]["path"]
    predictor = Predictor.load(model_path)

    # Validate and make predictions
    X = np.array(data.X, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 5:
        raise HTTPException(
            status_code=400,
            detail=f"Input X must have shape (n_samples, 5), got {X.shape}",
        )

    predictions = predictor.predict(X)

    return PredictionResponse(predictions=predictions.tolist())


@app.post("/models/{model_id}/predict/single")
async def predict_single(model_id: str, data: SinglePredictionInput) -> dict:
    """
    Make a prediction for a single 5D point.

    Args:
        model_id: The model identifier.
        data: Single input point.

    Returns:
        Prediction for the input point.
    """
    if model_id not in models_registry:
        raise HTTPException(status_code=404, detail="Model not found")

    # Load model
    model_path = models_registry[model_id]["path"]
    predictor = Predictor.load(model_path)

    # Make prediction
    result = predictor.predict_single(data.x1, data.x2, data.x3, data.x4, data.x5)

    return {"prediction": result}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict:
    """
    Delete a trained model.

    Args:
        model_id: The model identifier.

    Returns:
        Deletion confirmation.
    """
    if model_id not in models_registry:
        raise HTTPException(status_code=404, detail="Model not found")

    # Delete model file
    model_path = Path(models_registry[model_id]["path"])
    if model_path.exists():
        model_path.unlink()

    # Remove from registry
    del models_registry[model_id]

    return {"message": f"Model {model_id} deleted successfully"}
