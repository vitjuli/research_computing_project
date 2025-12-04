Quick Start Guide
=================

This guide will help you get started with NN Interpolator.

Loading Data
------------

The ``DataLoader`` class handles data loading and preprocessing:

.. code-block:: python

   from nn_interpolator import DataLoader
   import numpy as np

   # Create sample 5D data
   X = np.random.rand(1000, 5)  # 1000 samples, 5 features
   y = np.sum(X, axis=1)  # Target values

   # Load data
   loader = DataLoader()
   loader.load_from_arrays(X, y)

   # Normalize data (recommended)
   loader.normalize()

   # Split into training and validation sets
   train_X, train_y, val_X, val_y = loader.split(test_size=0.2)

Loading from CSV
~~~~~~~~~~~~~~~~

.. code-block:: python

   loader = DataLoader()
   loader.load_from_csv("data.csv")

The CSV should have 5 input columns followed by target column(s).

Training a Model
----------------

Create and train an interpolation model:

.. code-block:: python

   from nn_interpolator import Interpolator5D, Trainer

   # Create model with custom architecture
   model = Interpolator5D(
       output_dim=1,
       hidden_layers=[64, 64, 32]
   )

   # Create trainer
   trainer = Trainer(model)

   # Train the model
   history = trainer.fit(
       train_X, train_y,
       X_val=val_X, y_val=val_y,
       epochs=100,
       batch_size=32,
       learning_rate=0.001
   )

   # Evaluate
   metrics = trainer.evaluate(val_X, val_y)
   print(f"MSE: {metrics['mse']:.6f}")
   print(f"R²: {metrics['r2']:.4f}")

Making Predictions
------------------

Use the ``Predictor`` class for inference:

.. code-block:: python

   from nn_interpolator import Predictor

   # Save trained model
   trainer.save_model("model.json", loader)

   # Load for prediction
   predictor = Predictor.load("model.json")

   # Predict on new data
   X_new = np.random.rand(10, 5)
   predictions = predictor.predict(X_new)

   # Single point prediction
   result = predictor.predict_single(0.1, 0.2, 0.3, 0.4, 0.5)

Using the REST API
------------------

Start the API server:

.. code-block:: bash

   uvicorn api.main:app --reload

Train a model via API:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/train",
       json={
           "X": X.tolist(),
           "y": y.tolist(),
           "config": {
               "hidden_layers": [64, 32],
               "epochs": 100
           }
       }
   )
   model_id = response.json()["model_id"]

Make predictions:

.. code-block:: python

   response = requests.post(
       f"http://localhost:8000/models/{model_id}/predict",
       json={"X": X_new.tolist()}
   )
   predictions = response.json()["predictions"]
