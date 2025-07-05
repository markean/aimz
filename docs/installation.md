# Installation

!!! warning

    Due to its dependence on JAX and NumPyro, GPU support is **not available on Windows**. 
    For GPU acceleration, please use Linux or WSL2 (Windows Subsystem for Linux 2) with a compatible NVIDIA GPU and CUDA drivers. 

**aimz** is available on [PyPI](https://pypi.org/project/aimz/) for both CPU and GPU environments. Install or update using `pip`:

CPU (default):
```sh
pip install -U aimz
```

GPU (NVIDIA, CUDA 12):
```sh
pip install -U "aimz[gpu]"
```
!!! note

    This installs `jax[cuda12]` with the version specified by the package. However, to ensure you have the latest compatible version of JAX with CUDA 12, it is recommended to update JAX separately after installation:
    ```sh
    pip install -U "jax[cuda12]"
    ```
    Refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for up-to-date compatibility and driver requirements.


