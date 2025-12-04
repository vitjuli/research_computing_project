NN Interpolator Documentation
==============================

Welcome to the NN Interpolator documentation. This package provides a neural network
solution for interpolating 5-dimensional numerical datasets.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Features
--------

- **Data Loading**: Flexible data loading from numpy arrays or CSV files
- **Neural Network Model**: Configurable fully connected network for 5D interpolation
- **Training**: Easy-to-use training with validation and callbacks
- **Inference**: Fast prediction with batch support
- **REST API**: FastAPI backend for model training and prediction
- **Web Interface**: Next.js frontend for interactive model management

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from nn_interpolator import DataLoader, Interpolator5D, Trainer, Predictor

   # Generate sample data
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

   # Save model
   trainer.save_model("model.json", loader)

   # Load and predict
   predictor = Predictor.load("model.json")
   predictions = predictor.predict(np.random.rand(10, 5))

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
