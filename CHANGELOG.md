# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- `ImpactModel` methods `.predict()`, `.predict_on_batch()`, `.log_likelihood()`, and `.estimate_effect()` now return outputs as xarray DataTree instead of ArviZ InferenceData. Dimension names now follow the `dim_N` convention instead of the previous `dimN` style (@markean, [#49](https://github.com/markean/aimz/issues/49)).
- `.fit()`, `.fit_on_batch()`, and `.train_on_batch()` methods in `ImpactModel` now check for `"/"` in kernel site names to ensure compatibility with xarray DataTree (@markean, [#49](https://github.com/markean/aimz/issues/49)).
- `.predict()` in `ImpactModel` now checks for available posterior samples before falling back to `.predict_on_batch()`.
- `ArrayLoader` validates that `batch_size` is a positive integer.

### Removed

- Removed the `arviz` dependency (@markean, [#49](https://github.com/markean/aimz/issues/49)).

## [v0.3.2](https://github.com/markean/aimz/releases/tag/v0.3.2) - 2025-08-13

### Changed

- Updated `.predict()` and `.predict_on_batch()` to check for available posterior samples before returning outputs. This prevents errors when posterior samples are not defined based on the model specification.

## [v0.3.1](https://github.com/markean/aimz/releases/tag/v0.3.1) - 2025-08-02

### Fixed

- `ArrayDataset` and `ArrayLoader` now preserve the order in which input arrays are provided, ensuring consistent input mapping in methods like `.predict()` and `.log_likelihood()` (@markean, [#43](https://github.com/markean/aimz/issues/43)).

## [v0.3.0](https://github.com/markean/aimz/releases/tag/v0.3.0) - 2025-07-18

### Changed

- `ImpactModel` initialization parameter `vi` has been renamed to `inference` for compatibility with MCMC in future releases (@markean, [#36](https://github.com/markean/aimz/issues/36)).
- `ImpactModel` now supports `ArrayLoader` for both input and output data (@markean, [#24](https://github.com/markean/aimz/issues/24)).
- Renamed the posterior sample attribute of `ImpactModel` from `.posterior_samples_` to `.posterior`, which is now initialized to `None` (@markean, [#25](https://github.com/markean/aimz/issues/25)).
- `ArrayLoader` and `ArrayDataset` no longer require the `torch` dependency. `ArrayDataset` now accepts only named arrays, and `ArrayLoader` yields tuples of a dictionary and a padding integer (@markean, [#26](https://github.com/markean/aimz/issues/26)).

### Removed

- Removed the `torch` dependency (@markean, [#26](https://github.com/markean/aimz/issues/26)).

## [v0.2.0](https://github.com/markean/aimz/releases/tag/v0.2.0) - 2025-07-10

### Added

- `.train_on_batch()` and `.fit_on_batch()` methods to `ImpactModel` (@markean, [#15](https://github.com/markean/aimz/issues/15)).
- Custom `ArrayDataset` class for handling data in `ImpactModel`, removing the need for the `jax-dataloader` dependency (@markean, [#14](https://github.com/markean/aimz/issues/14)).
- GitHub Pages documentation site (@markean, [#10](https://github.com/markean/aimz/issues/10)).
- Installation instructions in the documentation site (@markean, [#10](https://github.com/markean/aimz/issues/10)).
- `ArrayLoader` class supports `shuffle` and `drop_last` parameters for epoch training for `.fit()` (@markean, [#15](https://github.com/markean/aimz/issues/15)).

### Changed

- Adopted `jax.typing` module for improved type hints.
- Removed unnecessary JAX array type conversion in `ImpactModel` methods.
- The `.fit()` method now uses epoch-based (minibatch) training (@markean, [#15](https://github.com/markean/aimz/issues/15)).
- Updated `.fit()`, `.train_on_batch()`, and `.fit_on_batch()` to train the model using the internal SVI state, continuing from the last state if available (@markean, [#15](https://github.com/markean/aimz/issues/15)).

### Removed

- Removed the `jax-dataloader` dependency (@markean, [#14](https://github.com/markean/aimz/issues/14)).
- Removed the `guide` property, as it is part of the `vi` property.

## [v0.1.0](https://github.com/markean/aimz/releases/tag/v0.1.0) - 2025-06-27

### Added

- Initial public release.
