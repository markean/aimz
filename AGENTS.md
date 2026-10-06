# AGENTS.md

Rules for working in this repository that are not enforced by tooling. For aimz behavior and usage, read the docs in `docs/source/` or the generated `llms.txt`.

## What this is

- aimz wraps a user-written NumPyro model (the kernel) in `ImpactModel`. It ships no models, templates, or modeling components, and does not explain NumPyro; the docs describe aimz behavior only.
- The public API consists of `aimz.__all__`, `aimz.model.KernelSpec`, `aimz.utils.data`, and the documented functions in `aimz.mlflow`. Names beginning with `_` are private.

## Commands

- Tests: `pytest tests`. The suite runs on three host CPU devices, configured in `tests/conftest.py`. To run one method's tests: `pytest tests/test_<method>.py`.
- Lint and format: `ruff check .` and `ruff format --check .`; Ruff selects all rules.
- Types: `ty check aimz`; keep it clean. Tests and notebooks are not checked.
- Docs: `make -C docs html` with the `docs` extra. `.. jupyter-execute::` cells run at build time, so keep them small and offline.

## Git and releases

- Branch from `dev`; pull requests target `dev`. `main` is release-only.
- Do not commit or push unless asked; leave changes in the working tree.
- Do not run `uv lock` or `uv sync`, and do not edit `uv.lock`, unless the task is a dependency change.
- Do not bump the version. The release commit sets it in `pyproject.toml` and `uv.lock` and dates the changelog entry.

## Code

- Prefer minimal, library-native changes: use standard library over new dependencies; do not add runtime dependencies unless strictly needed.
- Misuse that still yields a meaningful result should warn. Raise only when no meaningful result can be produced.
- Validation is metadata-only (for example, names and shapes); never materialize data solely for validation.
- Inline simple logic; do not introduce small one-off helper functions.
- Use per-file ignores in `pyproject.toml` instead of `# noqa`.
- Leave one blank line before a function's final `return` when logic precedes it.
- Add the Apache header to every new `.py` file.
- Kernel validation is based on signature and trace. Kernels wrapped in an effect handler or a `functools.wraps` decorator must remain accepted.

## Tests

- Add a test only when it covers lines or branches not already covered by existing tests.
- Add it as one workflow test in the matching `tests/test_<method>.py`, rather than one assertion or test per bug.
- Test files are named `test_*.py`.
- Tests that write files use the autouse fixture that changes into `tmp_path`, so nothing should be written into the repository.

## Changelog and docs

- Every user-visible change gets one bullet under `## Unreleased` in `CHANGELOG.md`, under Added, Changed, Removed, or Fixed.
- Write changelog entries in terms of behavior ("now ..., instead of ..."), use MyST roles such as ``{meth}`~aimz.ImpactModel.predict` ``, end with the issue link, and avoid specific numeric values.
- Docs-only changes do not get a changelog entry.
- Add every new public object to `docs/source/api/*.rst`; that list also feeds `llms.txt`.
- The skill in `aimz/.agents/skills/aimz/` ships in the wheel and tells agents how to use aimz. When a change alters behavior it describes, update it in the same change.
- In reStructuredText, use one sentence per line.
- Use Google-style docstrings with `Args`, `Returns`, and `Raises`; document warnings under `Warns`.
