Installation
============

aimz requires **Python 3.11 or higher** and is available via `PyPI <https://pypi.org/project/aimz/>`_ and `conda-forge <https://anaconda.org/conda-forge/aimz>`_.


Quick Install
-------------

.. tab-set::

   .. tab-item:: CPU (default)

      .. code-block:: sh

         pip install -U aimz

      Installs aimz with CPU-only JAX — no GPU drivers required.

   .. tab-item:: GPU (CUDA 13)

      .. code-block:: sh

         pip install -U "aimz[gpu]"

      Installs aimz and ``jax[cuda13]`` for NVIDIA GPU acceleration.

      .. warning::

         GPU support is not available on Windows due to JAX limitations.
         Use **Linux** or **WSL2** with a compatible NVIDIA GPU and CUDA drivers.

   .. tab-item:: conda

      .. code-block:: sh

         conda install conda-forge::aimz

      Installs the CPU version from conda-forge.
      GPU support via conda is not yet available — use pip for GPU installs.


Optional Extras
---------------

aimz ships several optional dependency groups that you can install with the
``pip install "aimz[<extra>]"`` syntax. Combine multiple extras with commas,
e.g. ``pip install "aimz[mlflow,docs]"``.

.. tab-set::

   .. tab-item:: mlflow

      .. code-block:: sh

         pip install -U "aimz[mlflow]"

      Adds `MLflow <https://mlflow.org/>`_ integration for experiment tracking and model logging.

   .. tab-item:: docs

      .. code-block:: sh

         pip install -U "aimz[docs]"

      Installs everything needed to build the documentation locally.

   .. tab-item:: dev

      .. code-block:: sh

         pip install -U "aimz[dev]"

      Development tools for linting, formatting, and testing.


Install from Source
-------------------

For the latest unreleased changes, install directly from GitHub:

.. tab-set::

   .. tab-item:: Latest (CPU)

      .. code-block:: sh

         pip install aimz@git+https://github.com/markean/aimz.git

   .. tab-item:: Latest (GPU)

      .. code-block:: sh

         pip install "aimz[gpu]@git+https://github.com/markean/aimz.git"

   .. tab-item:: Specific release

      .. code-block:: sh

         # Replace <tag> with the desired release tag
         pip install aimz@git+https://github.com/markean/aimz.git@<tag>

   .. tab-item:: Editable (development)

      .. code-block:: sh

         git clone https://github.com/markean/aimz.git
         cd aimz
         pip install -e ".[dev]"

      Installs in editable mode with development dependencies.
