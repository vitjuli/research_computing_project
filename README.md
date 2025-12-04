# NN Interpolator

A full-stack system for learning and serving neural network models that can interpolate any 5D numerical dataset.

## Features

- **Python Package (`nn_interpolator`)**: Data loading, model training, and inference for 5D numerical datasets
- **FastAPI Backend**: REST API endpoints for training and prediction
- **Next.js Frontend**: Interactive UI for training and querying models
- **Documentation**: Sphinx-based ReadTheDocs HTML documentation
- **Docker Support**: Docker Compose for containerized deployment

## Project Structure

```
research_computing_project/
├── nn_interpolator/          # Python package
│   ├── data/                 # Data loading module
│   ├── model/                # Neural network model
│   ├── training/             # Training utilities
│   └── inference/            # Prediction/inference
├── api/                      # FastAPI backend
├── frontend/                 # Next.js frontend
├── docs/                     # Sphinx documentation
├── tests/                    # Python tests
├── docker-compose.yml        # Docker composition
├── Dockerfile.backend        # Backend container
├── Dockerfile.frontend       # Frontend container
└── pyproject.toml           # Python project configuration
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vitjuli/research_computing_project.git
cd research_computing_project

# Install the Python package
pip install -e .

# For development (includes testing and linting tools)
pip install -e ".[dev]"

# For API development
pip install -e ".[api]"

# For documentation building
pip install -e ".[docs]"
```

### Using the Python Package

```python
import numpy as np
from nn_interpolator import DataLoader, Interpolator5D, Trainer, Predictor

# Generate sample 5D data
X = np.random.rand(1000, 5)
y = np.sin(np.sum(X, axis=1))

# Load and preprocess data
loader = DataLoader()
loader.load_from_arrays(X, y)
loader.normalize()
train_X, train_y, val_X, val_y = loader.split(test_size=0.2)

# Create and train model
model = Interpolator5D(output_dim=1, hidden_layers=[64, 64, 32])
trainer = Trainer(model)
history = trainer.fit(train_X, train_y, val_X, val_y, epochs=100)

# Save and load model
trainer.save_model("model.json", loader)
predictor = Predictor.load("model.json")

# Make predictions
predictions = predictor.predict(np.random.rand(10, 5))
```

### Running the API Server

```bash
# Install API dependencies
pip install -e ".[api]"

# Start the server
uvicorn api.main:app --reload

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Running the Frontend

```bash
cd frontend
npm install
npm run dev

# Frontend will be available at http://localhost:3000
```

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Backend API: http://localhost:8000
# Frontend: http://localhost:3000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/train` | POST | Train a new model |
| `/train/csv` | POST | Train from CSV file |
| `/models` | GET | List all models |
| `/models/{id}` | GET | Get model info |
| `/models/{id}/predict` | POST | Batch prediction |
| `/models/{id}/predict/single` | POST | Single point prediction |
| `/models/{id}` | DELETE | Delete a model |

## Building Documentation

```bash
cd docs
pip install -e "../[docs]"
make html

# Documentation will be in docs/build/html/
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Iuliia Vitiugova (Юлия Витюгова)