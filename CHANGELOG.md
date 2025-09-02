# Changelog

All notable changes to this project will be documented in this file and are best viewed on the [Changelog](https://aimz.readthedocs.io/en/latest/changelog.html) page.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Methods in {class}`~aimz.ImpactModel` now automatically determine the `batch_size` if it is not provided, based on the input data and number of samples([#70](https://github.com/markean/aimz/issues/70)).

## [v0.5.0](https://github.com/markean/aimz/releases/tag/v0.5.0) - 2025-09-01

### Added

- Added a `return_sites` parameter to the {meth}`~aimz.ImpactModel.predict` and {meth}`~aimz.ImpactModel.predict_on_batch` methods in {class}`~aimz.ImpactModel`, allowing users to specify which sites to include in the output ([#55](https://github.com/markean/aimz/issues/55)).
- {meth}`~aimz.ImpactModel.sample_prior_predictive_on_batch`, replacing {meth}`~aimz.ImpactModel.sample_prior_predictive` ([#67](https://github.com/markean/aimz/issues/67)).
- {meth}`~aimz.ImpactModel.sample_posterior_predictive_on_batch`, replacing {meth}`~aimz.ImpactModel.sample_posterior_predictive` ([#67](https://github.com/markean/aimz/issues/67)).

### Changed

- Switched documentation build system from MkDocs to Sphinx and ReadTheDocs ([https://aimz.readthedocs.io](https://aimz.readthedocs.io)).
- Added input `X` validation to {meth}`~aimz.ImpactModel.sample_prior_predictive` ([#65](https://github.com/markean/aimz/issues/65)).
- Exposed {class}`~aimz.ImpactModel` at the top-level package, allowing `from aimz import ImpactModel` ([#67](https://github.com/markean/aimz/issues/67)).
- {meth}`~aimz.ImpactModel.sample_prior_predictive` now returns a {class}`xarray.DataTree` instead of a dictionary, and writes outputs to files like the other methods ([#67](https://github.com/markean/aimz/issues/67)).
- {meth}`~aimz.ImpactModel.sample_posterior_predictive` is now an alias of {meth}`~aimz.ImpactModel.predict` and returns a {class}`xarray.DataTree` ([#67](https://github.com/markean/aimz/issues/67)).

### Fixed

- Enhanced data array validation to preserve device placement for JAX arrays ([#53](https://github.com/markean/aimz/issues/53)).
- Fixed incompatibility with Zarr when models output arrays in `bfloat16` by automatically promoting them to `float32` before saving ([#57](https://github.com/markean/aimz/issues/57)).
- Fixed the error message in {meth}`~aimz.ImpactModel.sample_posterior_predictive` when ``self.param_output`` is passed as an argument, which previously incorrectly referenced {meth}`~aimz.ImpactModel.sample_prior_predictive` ([#65](https://github.com/markean/aimz/issues/65)).

## [v0.4.0](https://github.com/markean/aimz/releases/tag/v0.4.0) - 2025-08-18

### Added

- Support for NumPyro MCMC in {class}`~aimz.ImpactModel`, including {meth}`~aimz.ImpactModel.fit_on_batch`, {meth}`~aimz.ImpactModel.sample`, and {meth}`~aimz.ImpactModel.set_posterior_sample` methods ([#35](https://github.com/markean/aimz/issues/35)).

### Changed

- {class}`~aimz.ImpactModel` methods {meth}`~aimz.ImpactModel.predict`, {meth}`~aimz.ImpactModel.predict_on_batch`, {meth}`~aimz.ImpactModel.log_likelihood`, and {meth}`~aimz.ImpactModel.estimate_effect` now return outputs as xarray DataTree instead of ArviZ InferenceData. Dimension names now follow the `dim_N` convention instead of the previous `dimN` style ([#49](https://github.com/markean/aimz/issues/49)).
- {meth}`~aimz.ImpactModel.fit`, {meth}`~aimz.ImpactModel.fit_on_batch`, and {meth}`~aimz.ImpactModel.train_on_batch` methods in {class}`~aimz.ImpactModel` now check for `"/"` in kernel site names to ensure compatibility with xarray DataTree ([#49](https://github.com/markean/aimz/issues/49)).

### Removed

- Removed the `arviz` dependency ([#49](https://github.com/markean/aimz/issues/49)).

### Fixed

- {meth}`~aimz.ImpactModel.predict` in {class}`~aimz.ImpactModel` now checks for available posterior samples before falling back to {meth}`~aimz.ImpactModel.predict_on_batch`.
- {class}`~aimz.utils.data.ArrayLoader` validates that `batch_size` is a positive integer.

## [v0.3.2](https://github.com/markean/aimz/releases/tag/v0.3.2) - 2025-08-13

### Changed

- Updated {meth}`~aimz.ImpactModel.predict` and {meth}`~aimz.ImpactModel.predict_on_batch` to check for available posterior samples before returning outputs. This prevents errors when posterior samples are not defined based on the model specification.

## [v0.3.1](https://github.com/markean/aimz/releases/tag/v0.3.1) - 2025-08-02

### Fixed

- {class}`~aimz.utils.data.ArrayDataset` and {class}`~aimz.utils.data.ArrayLoader` now preserve the order in which input arrays are provided, ensuring consistent input mapping in methods like {meth}`~aimz.ImpactModel.predict` and {meth}`~aimz.ImpactModel.log_likelihood` ([#43](https://github.com/markean/aimz/issues/43)).

## [v0.3.0](https://github.com/markean/aimz/releases/tag/v0.3.0) - 2025-07-18

### Changed

- {class}`~aimz.ImpactModel` initialization parameter `vi` has been renamed to `inference` for compatibility with MCMC in future releases ([#36](https://github.com/markean/aimz/issues/36)).
- {class}`~aimz.ImpactModel` now supports {class}`~aimz.utils.data.ArrayLoader` for both input and output data ([#24](https://github.com/markean/aimz/issues/24)).
- Renamed the posterior sample attribute of {class}`~aimz.ImpactModel` from {attr}`~aimz.ImpactModel.posterior_samples_` to {attr}`~aimz.ImpactModel.posterior`, which is now initialized to `None` ([#25](https://github.com/markean/aimz/issues/25)).
- {class}`~aimz.utils.data.ArrayLoader` and {class}`~aimz.utils.data.ArrayDataset` no longer require the `torch` dependency. {class}`~aimz.utils.data.ArrayDataset` now accepts only named arrays, and {class}`~aimz.utils.data.ArrayLoader` yields tuples of a dictionary and a padding integer ([#26](https://github.com/markean/aimz/issues/26)).

### Removed

- Removed the `torch` dependency ([#26](https://github.com/markean/aimz/issues/26)).

## [v0.2.0](https://github.com/markean/aimz/releases/tag/v0.2.0) - 2025-07-10

### Added

- {meth}`~aimz.ImpactModel.train_on_batch` and {meth}`~aimz.ImpactModel.fit_on_batch` methods to {class}`~aimz.ImpactModel` ([#15](https://github.com/markean/aimz/issues/15)).
- Custom {class}`~aimz.utils.data.ArrayDataset` class for handling data in {class}`~aimz.ImpactModel`, removing the need for the `jax-dataloader` dependency ([#14](https://github.com/markean/aimz/issues/14)).
- GitHub Pages documentation site ([#10](https://github.com/markean/aimz/issues/10)).
- Installation instructions in the documentation site ([#10](https://github.com/markean/aimz/issues/10)).
- {class}`~aimz.utils.data.ArrayLoader` class supports `shuffle` parameter for epoch training for {meth}`~aimz.ImpactModel.fit` ([#15](https://github.com/markean/aimz/issues/15)).

### Changed

- Adopted {mod}`jax.typing` module for improved type hints.
- Removed unnecessary JAX array type conversion in {class}`~aimz.ImpactModel` methods.
- The {meth}`~aimz.ImpactModel.fit` method now uses epoch-based (minibatch) training ([#15](https://github.com/markean/aimz/issues/15)).
- Updated {meth}`~aimz.ImpactModel.fit`, {meth}`~aimz.ImpactModel.train_on_batch`, and {meth}`~aimz.ImpactModel.fit_on_batch` to train the model using the internal SVI state, continuing from the last state if available ([#15](https://github.com/markean/aimz/issues/15)).

### Removed

- Removed the `jax-dataloader` dependency ([#14](https://github.com/markean/aimz/issues/14)).
- Removed the `guide` property, as it is part of the `vi` property.

## [v0.1.0](https://github.com/markean/aimz/releases/tag/v0.1.0) - 2025-06-27

### Added

- Initial public release.
