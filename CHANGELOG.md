# Changelog

## Unreleased
### Added
- `.train_on_batch()` and `.fit_on_batch()` methods to `ImpactModel` (@markean, [#15](https://github.com/markean/aimz/issues/15)).
- Custom `ArrayDataset` class for handling data in `ImpactModel`, removing the need for the `jax-dataloader` dependency (@markean, [#14](https://github.com/markean/aimz/issues/14)).
- GitHub Pages documentation site (@markean, [#10](https://github.com/markean/aimz/issues/10)).
- Installation instructions (@markean, [#10](https://github.com/markean/aimz/issues/10)).

### Changed
- Adopted `jax.typing` module for improved type hints.
- Removed unnecessary JAX array type conversion in `ImpactModel` methods.
- Updated `.fit()`, `.train_on_batch()`, and `.fit_on_batch()` to train the model using the internal SVI state, continuing from the last state if available (@markean, [#15](https://github.com/markean/aimz/issues/15)).

### Removed
- Removed `jax-dataloader` dependency (@markean, [#14](https://github.com/markean/aimz/issues/14)).

## 0.1.0 (2025-06-27)
- Initial public release.