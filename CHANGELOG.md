# Changelog

All notable changes to this project will be documented in this file and are best viewed on the [Changelog](https://aimz.readthedocs.io/latest/changelog.html) page.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.13.0](https://github.com/markean/aimz/releases/tag/v0.13.0) - 2026-07-03

### Added

- {meth}`~aimz.ImpactModel.predict`, {meth}`~aimz.ImpactModel.log_likelihood`, {meth}`~aimz.ImpactModel.sample_prior_predictive`, and {meth}`~aimz.ImpactModel.sample_posterior_predictive` now accept a `shard_axis` argument selecting the multi-device sharding strategy. The default `"obs"` shards the observation axis of the input across devices and replicates the posterior (the previous behavior). The new `"draw"` shards the drawn samples across devices in chunks of `batch_size` draws while holding the whole input resident. This has no sharding effect on a single device ([#224](https://github.com/markean/aimz/issues/224)).

### Changed

- {meth}`~aimz.ImpactModel.predict`, {meth}`~aimz.ImpactModel.log_likelihood`, and {meth}`~aimz.ImpactModel.sample_prior_predictive` now place the conditioning samples on devices once instead of re-transferring and re-replicating them on every batch. For the data-parallel path of {meth}`~aimz.ImpactModel.predict` and {meth}`~aimz.ImpactModel.log_likelihood` the placed posterior is cached across calls (keyed by device sharding and rebuilt whenever the posterior is replaced); {meth}`~aimz.ImpactModel.sample_prior_predictive` places its per-call prior samples once before the batch loop. This is a no-op on a single device and affects performance only, not results ([#219](https://github.com/markean/aimz/issues/219)).
- When an observation-aligned posterior sample shape is detected as incompatible with the default `shard_axis="obs"`, {meth}`~aimz.ImpactModel.predict` now warns and reruns under the new `shard_axis="draw"` scheme instead of falling back to the in-memory {meth}`~aimz.ImpactModel.predict_on_batch`, keeping results streamed to disk and memory-bounded. {meth}`~aimz.ImpactModel.log_likelihood`, which previously failed with a raw shape-broadcasting error for such posteriors, behaves the same way ([#224](https://github.com/markean/aimz/issues/224)).
- The auto-computed `batch_size` budget is now denominated in bytes rather than a fixed element count. The element cap is derived at call time by dividing the byte budget by the output dtype's item size ([#227](https://github.com/markean/aimz/issues/227)).

### Fixed

- Keyword arguments to the disk-backed methods whose values are strings or mappings are no longer mistaken for array inputs. Previously such a value was routed into the observation-axis batching pipeline and raised a misleading leading-axis error ([#229](https://github.com/markean/aimz/issues/229)).
- Requesting an unknown `return_sites` name from predictive methods now raises a `ValueError` naming the unknown site(s), instead of silently returning an empty or partially populated result ([#231](https://github.com/markean/aimz/issues/231)).
- {meth}`~aimz.ImpactModel.estimate_effect` now raises a `ValueError` when the selected predictive group is missing from the baseline output, instead of a `KeyError` from the underlying subtraction ([#233](https://github.com/markean/aimz/issues/233)).
- Interrupting a disk-backed run ({meth}`~aimz.ImpactModel.predict`, {meth}`~aimz.ImpactModel.log_likelihood`, {meth}`~aimz.ImpactModel.sample_prior_predictive`) with `Ctrl-C` during the write phase no longer hangs; the interrupt now exits and the partially-written output directory is removed ([#235](https://github.com/markean/aimz/issues/235)).
- On multi-device hosts, calling a disk-backed method again with a different number of array or keyword arguments no longer fails with a sharding structure error ([#237](https://github.com/markean/aimz/issues/237)).
- {meth}`~aimz.ImpactModel.predict` / {meth}`~aimz.ImpactModel.predict_on_batch` interventions (and {meth}`~aimz.ImpactModel.estimate_effect`) no longer reuse one likelihood-noise draw across all posterior draws, fixing under-dispersed intervention predictions and intervals ([#239](https://github.com/markean/aimz/issues/239)).
- {meth}`~aimz.ImpactModel.estimate_effect` now warns when the baseline and intervention scenarios have different dimension sizes, instead of silently computing the effect on their overlap ([#241](https://github.com/markean/aimz/issues/241)).
- A background writer-thread failure while streaming results to disk now raises instead of being swallowed; previously disk-backed methods could return a silently zero-filled or truncated result ([#243](https://github.com/markean/aimz/issues/243)).
- {meth}`~aimz.ImpactModel.sample_prior_predictive` under the default `shard_axis="obs"` now forwards non-array keyword arguments to the model when drawing prior samples; previously they were dropped, so those samples used the kernel's default values (silently wrong) or raised a `TypeError` for a required argument ([#245](https://github.com/markean/aimz/issues/245)).
- The disk-backed methods now work with an array input when a custom `param_input`/`param_output` is set; previously the streamed dataset was keyed by the literal names `X`/`y` ([#247](https://github.com/markean/aimz/issues/247)).
- A user-built {class}`~aimz.utils.data.ArrayLoader` whose dataset carries array fields beyond the model input/output now works with the disk-backed methods, binding each field to the kernel parameter of the same name; previously such a loader failed during tracing with an error, and array arguments were matched by position rather than name ([#249](https://github.com/markean/aimz/issues/249)).
- {meth}`~aimz.ImpactModel.log_likelihood` now raises a ValueError when called with an array `X` but no `y`, instead of a `KeyError: 'y'` ([#254](https://github.com/markean/aimz/issues/254)).
- The disk-backed methods now raise a clear `TypeError` when `X` is neither an array-like nor a data loader (e.g. a Python list), instead of an `UnboundLocalError` from an incomplete internal type check ([#256](https://github.com/markean/aimz/issues/256)).
- A disk-backed method that fails before the write phase no longer leaves an orphaned subdirectory behind; the partially-created subdirectory is removed and the original error is re-raised ([#258](https://github.com/markean/aimz/issues/258)).
- A model restored from disk with `pickle`/`cloudpickle` (e.g. through the `mlflow` integration) is now re-registered for class-level cleanup, so {meth}`~aimz.ImpactModel.cleanup_models` removes its temporary directory; previously only directly constructed models were tracked ([#260](https://github.com/markean/aimz/issues/260)).
- Saving an aimz model with an input example (via {func}`~aimz.mlflow.save_model`, {func}`~aimz.mlflow.log_model`, or {func}`~aimz.mlflow.autolog`) no longer bakes an ephemeral temporary `output_dir` and a fixed `batch_size` into the logged model signature; MLflow replayed those machine-local defaults on reload/serving, which broke prediction on a different machine and forced tiny batches ([#262](https://github.com/markean/aimz/issues/262)).
- {func}`~aimz.mlflow.autolog` no longer crashes when a `rng_key` is passed to {meth}`~aimz.ImpactModel.fit` / {meth}`~aimz.ImpactModel.fit_on_batch` — the typed PRNG key was routed into the logged input example and raised, so MLflow silently skipped logging the model artifact. It also no longer leaks the observed label into the logged input example and signature when a custom `param_output` is used ([#262](https://github.com/markean/aimz/issues/262)).

## [v0.12.0](https://github.com/markean/aimz/releases/tag/v0.12.0) - 2026-05-23

### Changed

- Input arrays of any shape with at least one dimension are now accepted; the leading axis is treated as the sample axis. Previously, `X` was required to be 2D and `y` 1D, and `y` with shape `(n, 1)` triggered a `DataConversionWarning` from `scikit-learn` ([#199](https://github.com/markean/aimz/issues/199)).
- {meth}`~aimz.ImpactModel.log_likelihood` now evaluates the kernel directly at each posterior draw, mirroring the per-draw pattern used by predictive sampling. The seeded kernel is also constructed once before the per-batch loop, so each batch reuses the same cached compilation ([#202](https://github.com/markean/aimz/issues/202)).
- Writer-thread queue sizing used when streaming batched outputs now adapts to available host memory and the per-batch output size ([#208](https://github.com/markean/aimz/issues/208)).
- Disk-backed methods ({meth}`~aimz.ImpactModel.predict`, {meth}`~aimz.ImpactModel.sample_prior_predictive`, {meth}`~aimz.ImpactModel.log_likelihood`) now preallocate each site's Zarr array and write each batch into a fixed slice, replacing the previous per-batch append. This avoids repeated Zarr resizing and is faster when the batch size is small and the number of batches is large. As a consequence, every return site must emit an axis-1 size equal to the input batch size; kernels with incompatible return sites raise `NotImplementedError` ([#213](https://github.com/markean/aimz/issues/213)).

### Fixed

- Writer-thread startup errors while opening Zarr output groups are now reported through the existing writer error path and queued items are drained before shutdown, preventing the main thread from waiting indefinitely when a background writer fails before consuming its queue ([#210](https://github.com/markean/aimz/issues/210)).

### Removed

- The `scikit-learn` dependency ([#199](https://github.com/markean/aimz/issues/199)).

## [v0.11.0](https://github.com/markean/aimz/releases/tag/v0.11.0) - 2026-04-29

### Changed

- Removed package-level logging configuration from `aimz/__init__.py`. `aimz` no longer sets a log level, attaches a `StreamHandler(sys.stdout)`, or calls `logging.captureWarnings(True)` on import; the `aimz` logger now only has a `logging.NullHandler()` attached.
Configuring handlers, levels, and warnings capture is the responsibility of the application.
Log messages emitted by {class}`~aimz.ImpactModel` were also refined—trailing ellipses were removed and posterior sampling now reports the number of samples being drawn—and the output-directory cleanup notice raised when {meth}`~aimz.ImpactModel.predict_on_batch` and {meth}`~aimz.ImpactModel.log_likelihood` encounter an error is now logged at the `warning` level (previously `debug`) ([#192](https://github.com/markean/aimz/issues/192)).

### Fixed

- Fixed {meth}`~aimz.ImpactModel.sample_prior_predictive` failing on multi-device meshes with `ValueError: in_specs ... does not match the specs of the input ... @obs`. The probe batch used to trace the kernel and draw global prior samples is now built with `batch_size=1` and `device=None`, preventing JAX's sharding-in-types from propagating the `obs` mesh axis onto global (non-batched) sample sites that are later passed as the replicated `samples` argument to the {func}`jax.shard_map` sampler ([#194](https://github.com/markean/aimz/issues/194)).

## [v0.10.0](https://github.com/markean/aimz/releases/tag/v0.10.0) - 2026-04-17

### Added

- Added support for Python 3.14.
- {meth}`~aimz.ImpactModel.estimate_effect` now accepts an `on_batch` keyword argument.
When ``True``, predictions are dispatched through {meth}`~aimz.ImpactModel.predict_on_batch` and any raw `dict` results are automatically converted to {class}`xarray.DataTree` ([#180](https://github.com/markean/aimz/issues/180)).

### Changed

- {class}`~aimz.utils.data.ArrayDataset` now employs NumPy-based indexing in {class}`~aimz.utils.data.ArrayLoader` instead of triggering JAX tracing on each batch ([#168](https://github.com/markean/aimz/issues/168)).
- Changed the default value of `to_jax` in {class}`~aimz.utils.data.ArrayDataset` from `True` to `False` to avoid redundant conversion ([#170](https://github.com/markean/aimz/issues/170)).

### Fixed

- Fixed auto-computed `batch_size` rounding down to zero on multi-device setups when `MAX_ELEMENTS // num_samples` is smaller than the number of devices ([#172](https://github.com/markean/aimz/issues/172)).
- {meth}`~aimz.utils.data.ArrayLoader.pad_array` now pads with NumPy when given NumPy arrays, avoiding premature device transfers, and skips padding entirely when `n_pad` is zero ([#174](https://github.com/markean/aimz/issues/174)).

## [v0.9.1](https://github.com/markean/aimz/releases/tag/v0.9.1) - 2025-12-08

### Fixed

- Fixed {func}`jax.shard_map` closure error for sharded `rng_key` in parallelism methods when using JAX 0.8 and newer versions ([#140](https://github.com/markean/aimz/issues/140)).

## [v0.9.0](https://github.com/markean/aimz/releases/tag/v0.9.0) - 2025-11-16

### Added

- Added the class method {meth}`~aimz.ImpactModel.cleanup_models` to clean up temporary directories for all active model instances ([#136](https://github.com/markean/aimz/issues/136)).

### Changed

- The output subdirectory naming convention has changed from using only a timestamp to the pattern `<timestamp>_<caller_name>/`, where `<caller_name>` is the name of the method that triggered the write operation ([#138](https://github.com/markean/aimz/issues/138)).
- Lowered the logging level for exceptions during temporary directory cleanup from `exception` to `debug` to reduce console noise.

## [v0.8.1](https://github.com/markean/aimz/releases/tag/v0.8.1) - 2025-10-23

### Changed

- The minimum required versions are: Dask 2025.7, JAX 0.8, and Xarray 2025.7.
- Replaced deprecated {mod}`jax.experimental.shard_map.shard_map` with {func}`jax.shard_map` to ensure compatibility with JAX 0.8 and newer versions ([#128](https://github.com/markean/aimz/issues/128)).
- Logging exception messages are displayed before the writer thread is shut down, providing a more immediate response for {meth}`~aimz.ImpactModel.predict` and {meth}`~aimz.ImpactModel.log_likelihood`, especially when interrupted by the keyboard ([#130](https://github.com/markean/aimz/issues/130)).

## [v0.8.0](https://github.com/markean/aimz/releases/tag/v0.8.0) - 2025-10-14

### Added

- Extended MLflow autologging to support the {meth}`~aimz.ImpactModel.fit_on_batch` method ([#119](https://github.com/markean/aimz/issues/119)).
- Added __str__ and __repr__ methods to the {class}~aimz.ImpactModel ([#118](https://github.com/markean/aimz/issues/118)).
- {class}`~aimz.model.KernelSpec` now includes a `sample_sites` attribute listing all stochastic sample sites in the model kernel ([#125](https://github.com/markean/aimz/issues/125)).

## [v0.7.0](https://github.com/markean/aimz/releases/tag/v0.7.0) - 2025-09-29

### Added

- `output_dir` attribute to the root and group nodes of {class}`xarray.DataTree` objects returned by {meth}`~aimz.ImpactModel.sample_prior_predictive`, {meth}`~aimz.ImpactModel.predict`, and {meth}`~aimz.ImpactModel.log_likelihood`, specifying the directory where results are saved ([#85](https://github.com/markean/aimz/issues/85)).
- Introduced the public {class}`~aimz.model.KernelSpec` dataclass and the {attr}`~aimz.ImpactModel.kernel_spec` attribute on {class}`~aimz.ImpactModel`.
This exposes a lazily-built, cached structural specification of the user kernel (fields: ``traced``, ``return_sites``, ``output_observed``) so training and predictive methods avoid redundant model tracing ([#98](https://github.com/markean/aimz/issues/98)).
- When available, an `output_dir` attribute is added to the root node of {class}`xarray.DataTree` object returned by {meth}`~aimz.ImpactModel.estimate_effect`, specifying the directory where results are saved ([#110](https://github.com/markean/aimz/issues/110)).

### Changed

- All `tqdm` progress bars now use `dynamic_ncols=True` to adjust column width dynamically ([#93](https://github.com/markean/aimz/issues/93)).
- {meth}`~aimz.ImpactModel.fit_on_batch`, {meth}`~aimz.ImpactModel.sample_prior_predictive_on_batch`, {meth}`~aimz.ImpactModel.sample_prior_predictive`, and {meth}`~aimz.ImpactModel.train_on_batch` now reuse the cached {attr}`~aimz.ImpactModel.kernel_spec` and avoid redundant model tracing ([#98](https://github.com/markean/aimz/issues/98)).
- {meth}`~aimz.ImpactModel.set_posterior_sample` no longer accepts a `return_sites` parameter; downstream methods can now set it explicitly ([#100](https://github.com/markean/aimz/issues/100)).
- {meth}`~aimz.ImpactModel.set_posterior_sample` now raises an error when an empty posterior dictionary (`{}`) is provided ([#101](https://github.com/markean/aimz/issues/101)).
- {meth}`~aimz.ImpactModel.sample_prior_predictive_on_batch` and {meth}`~aimz.ImpactModel.sample_prior_predictive` now include posterior samples in the returned results if available ([#103](https://github.com/markean/aimz/issues/103)).
- {meth}`~aimz.ImpactModel.sample_prior_predictive_on_batch`, {meth}`~aimz.ImpactModel.sample_prior_predictive`, {meth}`~aimz.ImpactModel.sample`, {meth}`~aimz.ImpactModel.sample_posterior_predictive_on_batch`, {meth}`~aimz.ImpactModel.sample_posterior_predictive`, {meth}`~aimz.ImpactModel.predict_on_batch`, and {meth}`~aimz.ImpactModel.predict` can now accept a single `str` or an iterable of `str` values for the `return_sites` parameter ([#107](https://github.com/markean/aimz/issues/107)).
- {meth}`~aimz.ImpactModel.sample_prior_predictive_on_batch` returns the default output site along with deterministic sites when `return_sites` is not specified, to be consistent with the behavior of other sampling methods ([#108](https://github.com/markean/aimz/issues/108)).
- {meth}`~aimz.ImpactModel.estimate_effect` returns a `posterior` group node in the {class}`xarray.DataTree` object when posterior samples are available, to be consistent with other methods ([#110](https://github.com/markean/aimz/issues/110)).
- Subdirectories under {attr}`~aimz.ImpactModel.temp_dir` now include microseconds in their names to avoid duplicates and file-exists errors ([#110](https://github.com/markean/aimz/issues/110)).

### Fixed

- Methods in {class}`~aimz.ImpactModel` no longer include an empty `posterior` data variable in root node of the returned {class}`xarray.DataTree` when no posterior samples are available ([#91](https://github.com/markean/aimz/issues/91)).

## [v0.6.0](https://github.com/markean/aimz/releases/tag/v0.6.0) - 2025-09-14

### Added

- {meth}`~aimz.ImpactModel.sample_prior_predictive_on_batch`, {meth}`~aimz.ImpactModel.sample`, {meth}`~aimz.ImpactModel.sample_posterior_predictive_on_batch`, and {meth}`~aimz.ImpactModel.predict_on_batch` methods in {class}`~aimz.ImpactModel` now support a `return_datatree` parameter. When set to `True` (by default), results are returned as an {class}`xarray.DataTree`; otherwise, a `dict` is returned ([#74](https://github.com/markean/aimz/issues/74)).
- MLflow integration for {class}`~aimz.ImpactModel` ([#71](https://github.com/markean/aimz/issues/71)).

### Changed

- Methods in {class}`~aimz.ImpactModel` now automatically determine the `batch_size` if it is not provided, based on the input data and number of samples ([#70](https://github.com/markean/aimz/issues/70)).
- {meth}`~aimz.ImpactModel.sample_posterior_predictive_on_batch` and {meth}`~aimz.ImpactModel.sample_posterior_predictive` no longer accept the `in_sample` argument. Results are now always written to the `posterior_predictive` group.

### Removed

- Removed the `tqdm` dependency ([#80](https://github.com/markean/aimz/issues/80)).

### Fixed

- Methods in {class}`~aimz.ImpactModel` now handle empty posterior dictionaries (`{}`) gracefully instead of failing when no posterior samples are available ([#76](https://github.com/markean/aimz/issues/76)).

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
- Fixed the error message in {meth}`~aimz.ImpactModel.sample_posterior_predictive` when `self.param_output` is passed as an argument, which previously incorrectly referenced {meth}`~aimz.ImpactModel.sample_prior_predictive` ([#65](https://github.com/markean/aimz/issues/65)).

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
