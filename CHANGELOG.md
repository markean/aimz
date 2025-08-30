# Changelog

All notable changes to this project will be documented in this file and are best viewed on the [Changelog](https://aimz.readthedocs.io/en/latest/changelog.html) page.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added a `return_sites` parameter to the {meth}`~aimz.model.ImpactModel.predict` and {meth}`~aimz.model.ImpactModel.predict_on_batch` methods in {class}`~aimz.model.ImpactModel`, allowing users to specify which sites to include in the output ([#55](https://github.com/markean/aimz/issues/55)).

### Changed

- Switched documentation build system from MkDocs to Sphinx and ReadTheDocs ([https://aimz.readthedocs.io](https://aimz.readthedocs.io)).
- Added input `X` validation to {meth}`~aimz.model.ImpactModel.sample_prior_predictive` ([#65](https://github.com/markean/aimz/issues/65)).

### Fixed

- Fixed incompatibility with Zarr when models output arrays in `bfloat16` by automatically promoting them to `float32` before saving ([#57](https://github.com/markean/aimz/issues/57)).
- Enhanced data array validation to preserve device placement for JAX arrays ([#53](https://github.com/markean/aimz/issues/53)).
- Fixed the error message in {meth}`~aimz.model.ImpactModel.sample_posterior_predictive` when ``self.param_output`` is passed as an argument, which previously incorrectly referenced {meth}`~aimz.model.ImpactModel.sample_prior_predictive` ([#65](https://github.com/markean/aimz/issues/65)).

## [v0.4.0](https://github.com/markean/aimz/releases/tag/v0.4.0) - 2025-08-18

### Added

- Support for NumPyro MCMC in {class}`~aimz.model.ImpactModel`, including {meth}`~aimz.model.ImpactModel.fit_on_batch`, {meth}`~aimz.model.ImpactModel.sample`, and {meth}`~aimz.model.ImpactModel.set_posterior_sample` methods ([#35](https://github.com/markean/aimz/issues/35)).

### Changed

- {class}`~aimz.model.ImpactModel` methods {meth}`~aimz.model.ImpactModel.predict`, {meth}`~aimz.model.ImpactModel.predict_on_batch`, {meth}`~aimz.model.ImpactModel.log_likelihood`, and {meth}`~aimz.model.ImpactModel.estimate_effect` now return outputs as xarray DataTree instead of ArviZ InferenceData. Dimension names now follow the `dim_N` convention instead of the previous `dimN` style ([#49](https://github.com/markean/aimz/issues/49)).
- {meth}`~aimz.model.ImpactModel.fit`, {meth}`~aimz.model.ImpactModel.fit_on_batch`, and {meth}`~aimz.model.ImpactModel.train_on_batch` methods in {class}`~aimz.model.ImpactModel` now check for `"/"` in kernel site names to ensure compatibility with xarray DataTree ([#49](https://github.com/markean/aimz/issues/49)).

### Removed

- Removed the `arviz` dependency ([#49](https://github.com/markean/aimz/issues/49)).

### Fixed

- {meth}`~aimz.model.ImpactModel.predict` in {class}`~aimz.model.ImpactModel` now checks for available posterior samples before falling back to {meth}`~aimz.model.ImpactModel.predict_on_batch`.
- {class}`~aimz.utils.data.ArrayLoader` validates that `batch_size` is a positive integer.

## [v0.3.2](https://github.com/markean/aimz/releases/tag/v0.3.2) - 2025-08-13

### Changed

- Updated {meth}`~aimz.model.ImpactModel.predict` and {meth}`~aimz.model.ImpactModel.predict_on_batch` to check for available posterior samples before returning outputs. This prevents errors when posterior samples are not defined based on the model specification.

## [v0.3.1](https://github.com/markean/aimz/releases/tag/v0.3.1) - 2025-08-02

### Fixed

- {class}`~aimz.utils.data.ArrayDataset` and {class}`~aimz.utils.data.ArrayLoader` now preserve the order in which input arrays are provided, ensuring consistent input mapping in methods like {meth}`~aimz.model.ImpactModel.predict` and {meth}`~aimz.model.ImpactModel.log_likelihood` ([#43](https://github.com/markean/aimz/issues/43)).

## [v0.3.0](https://github.com/markean/aimz/releases/tag/v0.3.0) - 2025-07-18

### Changed

- {class}`~aimz.model.ImpactModel` initialization parameter `vi` has been renamed to `inference` for compatibility with MCMC in future releases ([#36](https://github.com/markean/aimz/issues/36)).
- {class}`~aimz.model.ImpactModel` now supports {class}`~aimz.utils.data.ArrayLoader` for both input and output data ([#24](https://github.com/markean/aimz/issues/24)).
- Renamed the posterior sample attribute of {class}`~aimz.model.ImpactModel` from {attr}`~aimz.model.ImpactModel.posterior_samples_` to {attr}`~aimz.model.ImpactModel.posterior`, which is now initialized to `None` ([#25](https://github.com/markean/aimz/issues/25)).
- {class}`~aimz.utils.data.ArrayLoader` and {class}`~aimz.utils.data.ArrayDataset` no longer require the `torch` dependency. {class}`~aimz.utils.data.ArrayDataset` now accepts only named arrays, and {class}`~aimz.utils.data.ArrayLoader` yields tuples of a dictionary and a padding integer ([#26](https://github.com/markean/aimz/issues/26)).

### Removed

- Removed the `torch` dependency ([#26](https://github.com/markean/aimz/issues/26)).

## [v0.2.0](https://github.com/markean/aimz/releases/tag/v0.2.0) - 2025-07-10

### Added

- {meth}`~aimz.model.ImpactModel.train_on_batch` and {meth}`~aimz.model.ImpactModel.fit_on_batch` methods to {class}`~aimz.model.ImpactModel` ([#15](https://github.com/markean/aimz/issues/15)).
- Custom {class}`~aimz.utils.data.ArrayDataset` class for handling data in {class}`~aimz.model.ImpactModel`, removing the need for the `jax-dataloader` dependency ([#14](https://github.com/markean/aimz/issues/14)).
- GitHub Pages documentation site ([#10](https://github.com/markean/aimz/issues/10)).
- Installation instructions in the documentation site ([#10](https://github.com/markean/aimz/issues/10)).
- {class}`~aimz.utils.data.ArrayLoader` class supports `shuffle` parameter for epoch training for {meth}`~aimz.model.ImpactModel.fit` ([#15](https://github.com/markean/aimz/issues/15)).

### Changed

- Adopted {mod}`jax.typing` module for improved type hints.
- Removed unnecessary JAX array type conversion in {class}`~aimz.model.ImpactModel` methods.
- The {meth}`~aimz.model.ImpactModel.fit` method now uses epoch-based (minibatch) training ([#15](https://github.com/markean/aimz/issues/15)).
- Updated {meth}`~aimz.model.ImpactModel.fit`, {meth}`~aimz.model.ImpactModel.train_on_batch`, and {meth}`~aimz.model.ImpactModel.fit_on_batch` to train the model using the internal SVI state, continuing from the last state if available ([#15](https://github.com/markean/aimz/issues/15)).

### Removed

- Removed the `jax-dataloader` dependency ([#14](https://github.com/markean/aimz/issues/14)).
- Removed the `guide` property, as it is part of the `vi` property.

## [v0.1.0](https://github.com/markean/aimz/releases/tag/v0.1.0) - 2025-06-27

### Added

- Initial public release.
