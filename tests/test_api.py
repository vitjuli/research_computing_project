"""Tests for the FastAPI backend."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "NN Interpolator API"

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_train_model(self, client):
        """Test model training endpoint."""
        np.random.seed(42)
        X = np.random.rand(100, 5).tolist()
        y = [sum(x) for x in X]

        response = client.post(
            "/train",
            json={
                "X": X,
                "y": y,
                "config": {
                    "hidden_layers": [32, 16],
                    "epochs": 10,
                    "batch_size": 16,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "final_loss" in data
        assert "history" in data

    def test_list_models(self, client):
        """Test listing models."""
        response = client.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_train_and_predict(self, client):
        """Test training and then predicting."""
        # Train a model
        np.random.seed(42)
        X = np.random.rand(100, 5).tolist()
        y = [sum(x) for x in X]

        train_response = client.post(
            "/train",
            json={
                "X": X,
                "y": y,
                "config": {"hidden_layers": [32], "epochs": 20},
            },
        )

        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        # Get model info
        info_response = client.get(f"/models/{model_id}")
        assert info_response.status_code == 200

        # Make prediction
        X_test = np.random.rand(5, 5).tolist()
        predict_response = client.post(
            f"/models/{model_id}/predict",
            json={"X": X_test},
        )

        assert predict_response.status_code == 200
        predictions = predict_response.json()["predictions"]
        assert len(predictions) == 5

    def test_predict_single(self, client):
        """Test single point prediction."""
        # First train a model
        np.random.seed(42)
        X = np.random.rand(50, 5).tolist()
        y = [sum(x) for x in X]

        train_response = client.post(
            "/train",
            json={"X": X, "y": y, "config": {"epochs": 10}},
        )
        model_id = train_response.json()["model_id"]

        # Make single prediction
        response = client.post(
            f"/models/{model_id}/predict/single",
            json={"x1": 0.1, "x2": 0.2, "x3": 0.3, "x4": 0.4, "x5": 0.5},
        )

        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_delete_model(self, client):
        """Test model deletion."""
        # Train a model
        np.random.seed(42)
        X = np.random.rand(50, 5).tolist()
        y = [sum(x) for x in X]

        train_response = client.post(
            "/train",
            json={"X": X, "y": y, "config": {"epochs": 5}},
        )
        model_id = train_response.json()["model_id"]

        # Delete model
        delete_response = client.delete(f"/models/{model_id}")
        assert delete_response.status_code == 200

        # Model should no longer exist
        get_response = client.get(f"/models/{model_id}")
        assert get_response.status_code == 404

    def test_predict_nonexistent_model(self, client):
        """Test prediction with nonexistent model."""
        response = client.post(
            "/models/nonexistent/predict",
            json={"X": [[0.1, 0.2, 0.3, 0.4, 0.5]]},
        )
        assert response.status_code == 404

    def test_train_invalid_data(self, client):
        """Test training with invalid data."""
        response = client.post(
            "/train",
            json={
                "X": [[0.1, 0.2, 0.3]],  # Wrong dimensions
                "y": [0.5],
            },
        )
        assert response.status_code == 400
