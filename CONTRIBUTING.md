# Contributing
Thank you for your interest in contributing to **aims**. Whether you're fixing a bug, adding a feature, or improving documentation, we welcome your help.


## Development Guide
1. **Clone the repository**
   ```bash
   git clone https://github.com/markean/aims.git
   cd aims
   ```

2. **Set up your environment**

   Python 3.11 or higher is required. We use [uv](https://docs.astral.sh/uv/) as the package and environment manager. You can create a virtual environment and install dependencies as follows:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ."[dev]"
   ```

3. **Run pre-commit hooks**

   We use [`pre-commit`](https://pre-commit.com/) checks to automatically check code formatting and quality before commits. To run all hooks manually:

   ```bash
   pre-commit run --all-files
   ```

4. **Code Style and Documentation**

   - Code style is enforced using [`ruff`](https://docs.astral.sh/ruff/), configured via `pyproject.toml`.
   - Use clear, consistent docstrings following the [Google style guide](https://google.github.io/styleguide/pyguide.html).
   - To lint your code:
     ```bash
     ruff check .
     ```

5. **Run tests**

   - Add unit tests for new functionality in the `tests/` directory.
   - Run tests locally with `pytest` before submitting a PR.


## Need Help?
If you run into any issues, feel free to open an issue or reach out to the maintainers listed in `pyproject.toml`.
