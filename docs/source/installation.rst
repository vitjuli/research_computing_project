Installation
============

Requirements
------------

- Python 3.9 or higher
- NumPy >= 1.21.0

Installation from Source
------------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/vitjuli/research_computing_project.git
   cd research_computing_project
   pip install -e .

Optional Dependencies
---------------------

For development (testing, linting):

.. code-block:: bash

   pip install -e ".[dev]"

For the FastAPI backend:

.. code-block:: bash

   pip install -e ".[api]"

For building documentation:

.. code-block:: bash

   pip install -e ".[docs]"

Docker Installation
-------------------

You can also run the entire stack using Docker Compose:

.. code-block:: bash

   docker-compose up --build

This will start:

- Backend API on port 8000
- Frontend on port 3000
