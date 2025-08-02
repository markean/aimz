# Installation

**aimz** requires Python 3.11 or higher. It is available via [PyPI](https://pypi.org/project/aimz/) and [conda-forge](https://anaconda.org/conda-forge/aimz).

## Install with pip

**CPU (default):**

  ```sh
  pip install -U aimz
  ```

**GPU (NVIDIA, CUDA 12):**

  ```sh
  pip install -U "aimz[gpu]"
  ```

!!! warning
    GPU support is not available on Windows due to dependence on JAX. For GPU acceleration, use Linux or WSL2 with a compatible NVIDIA GPU and CUDA drivers.

## Install with conda

```sh
conda install conda-forge::aimz
```
