# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `.train_on_batch()` and `.fit_on_batch()` methods to `ImpactModel` (@markean, [#15](https://github.com/markean/aimz/issues/15)).
- Custom `ArrayDataset` class for handling data in `ImpactModel`, removing the need for the `jax-dataloader` dependency (@markean, [#14](https://github.com/markean/aimz/issues/14)).
- GitHub Pages documentation site (@markean, [#10](https://github.com/markean/aimz/issues/10)).
- Installation instructions in the documentation site (@markean, [#10](https://github.com/markean/aimz/issues/10)).
- `ArrayLoader` class supports `shuffle` and `drop_last` parameters for epoch training for `.fit()` (@markean, [#15](https://github.com/markean/aimz/issues/15)).

### Changed

- Adopted `jax.typing` module for improved type hints.
- Removed unnecessary JAX array type conversion in `ImpactModel` methods.
- Updated `.fit()`, `.train_on_batch()`, and `.fit_on_batch()` to train the model using the internal SVI state, continuing from the last state if available (@markean, [#15](https://github.com/markean/aimz/issues/15)).

### Removed

- Removed `jax-dataloader` dependency (@markean, [#14](https://github.com/markean/aimz/issues/14)).

## [v0.1.0](https://github.com/markean/aimz/releases/tag/v0.1.0) - 2025-06-27

### Added
- Initial public release.
