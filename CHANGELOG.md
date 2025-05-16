# Changelog

All notable changes to `speclib` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## \[0.1.0-beta.1] - 2024-05-16

### Added

* `Spectrum` class to load and manipulate stellar model spectra.
* Support for multiple model grids: PHOENIX, DRIFT-PHOENIX, SPHINX, NextGen-solar, and MPS-Atlas.
* `resample()` and `regularize()` methods using `pysynphot` for consistent flux interpolation.
* `set_spectral_resolution()` method with Gaussian convolution.
* `bin()` method and `BinnedSpectrum` class for filter binning.
* `SpectralGrid` and `BinnedSpectralGrid` classes for fast interpolation across stellar parameters.
* `Filter`, `SED`, and `SEDGrid` classes for synthetic photometry.
* `apply_filter()` and `mag_to_flux()` utility functions.

### Changed

* Migrated from legacy `tool.poetry.*` layout to PEP 621-compliant `[project]` table.
* Included filter response curves via `importlib.resources`.
* Tests pass under Python 3.9 and 3.11.

### Known Issues

* DRIFT-PHOENIX, SPHINX, and MPS-Atlas models require local caching.
* No automated validation for model grid completeness.
* No test coverage for all interpolation edge cases.

---

For earlier development notes, see internal documentation or Git commit history.
