# Changelog

All notable changes to `speclib` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- Automatically trigger NewEra tarball download when loading a wavelength array for the first time.
  `load_newera_wavelength_array()` now calls `download_newera_grid()` if the expected `.txt` file is missing.
- Corrected the metallicity list for the MPS-Atlas grid in `utils.py` where `-0.95 - 0.9` was accidentally
  combined into a single entry.

### Added
- Example SPHINX spectra under `tests/data/sphinx/` for offline testing.
- Added `trilinear_interpolation` utility function and implemented it in `core.py` to improve readability and modularity.


### Changed
- SPHINX loader now reads the wavelength array directly from each cached
  spectrum file, removing the dependency on a separate wavelength file.
- Tests patch the SPHINX grid to use these local spectra so no network
  access is required when running the suite (see issue #27).

---

## [0.1.0-beta.4] - 2025-05-30

### Added
- Support for PHOENIX NewEra model grids (`newera_gaia`, `newera_jwst`, `newera_lowres`) via `Spectrum.from_grid()`
- Utility functions `load_newera_wavelength_array()` and `load_newera_flux_array()` in `utils.py` to handle NewEra file structure
- Automatic detection of wavelength grid parameters from NewEra file headers
- Warning when `alpha ≠ 0.0` is requested, as non-zero alpha values may not be reliably supported yet

### Changed
- Improved error messaging for missing NewEra files to aid debugging and model grid validation

---

## [0.1.0-beta.3] - 2025-05-28

### Added
- Option to disable interpolation in `SpectralGrid`, `BinnedSpectralGrid`, and `SEDGrid` via `interpolate=False`.
- Tests for nearest-neighbor retrieval.
- Test to ensure `speclib.__version__` matches the version in `pyproject.toml`.

### Changed
- `__version__` is now read from `pyproject.toml` using `importlib.metadata`.

---

## [0.1.0-beta.2] – 2025-05-23

### Added
- ✅ Support for loading **PHOENIX NewEra model spectra** via `Spectrum.from_grid(model_grid="newera")`
- 📥 Utility function `download_newera_grid()` to selectively download subsets of the NewEra model grid
- 🎯 Filtering options for `teff_range`, `logg_range`, `feh_range`, and `alpha_range` in `download_newera_grid()`
- ⚠️ Warning raised when attempting to download the full grid (~4.5 TB)

### Changed
- 🔧 Improved verbosity and error handling for model downloads

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
