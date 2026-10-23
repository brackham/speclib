# Changelog

All notable changes to `speclib` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0b13] - 2026-10-23

### Added

- Added an offline `InstrumentProfile` registry with `get_profile()`,
  `list_instrument_profiles()`, and 55 HST/JWST profiles. Bundled resolving-power
  curves, coverage definitions, source manifests, and reproducible conversion
  scripts expose scientific provenance through `InstrumentProfile.provenance`.
- Added 44 JWST/NIRSpec BOTS/S1600A1 profiles: nine SUB2048 configurations and
  35 configurations for SUB1024A (except PRISM), SUB1024B, SUB512, and SUB512S.
  They share official STScI wavelength-dependent resolving-power curves and use
  subarray-specific nominal coverage, including disconnected intervals.
  Short NIRSpec identifiers select SUB2048.
- Added HST/STIS G430L and G750L profiles for centered point sources through
  52X2, HST/WFC3 IR G102 and G141 profiles for unresolved sources at field
  center, and separate JWST/NIRISS SOSS CLEAR/SUBSTRIP256 orders 1 and 2.
  SOSS order 2 uses conservative 0.6–1.0 µm coverage; orders are not stitched.
- Added four JWST/NIRCam Module A Grism R first-order time-series profiles
  (F277W, F322W2, F356W, F444W), with a shared wavelength-dependent nominal
  resolving-power prescription and filter-specific coverage. F277W's blue and
  F444W's red output limits preserve model support for convolution padding.
- Added a JWST/MIRI P750L/SLITLESSPRISM time-series profile with nominal
  wavelength-dependent design resolving power, conservative 5.5–12 µm output,
  and 5–14 µm model support. These output limits are modeling choices rather
  than empirical instrument cutoffs.
- Added `Spectrum.observed_by()` and `SpectralGrid.observed_by()` for nominal
  Gaussian resolution matching and wavelength coverage. They retain native
  samples by default or accept explicit wavelengths or
  `samples_per_resolution_element`. Continuous source spectra are convolved
  with padding before detector gaps and unread subarray regions are excluded.
  Profiles describe nominal resolution and coverage, not exposure-specific
  LSFs; input widths must be negligible, and throughput, noise, slit losses,
  non-Gaussian LSFs, and deconvolution are not modeled. Source and calibration
  coverage must include convolution padding; neither is extrapolated.
- Added explicit `valid_intervals` coverage and gap-aware `resample()`, `bin()`,
  and inclusive `select_wavelength()` with copied metadata. Resampling and
  binning reject bins crossing gaps or sampled coverage boundaries; broadening,
  regularization, and repeated instrument matching reject discontinuous inputs.
  Inherited specutils slicing, arithmetic, and serialization do not enforce
  these coverage safeguards.
- Added Smitha et al. (2025) G2V, K0V, and M0V quiet, spot, penumbral, and
  umbral spectra through `Spectrum.from_smitha2025()`, with checksum-validated
  Edmond caching and `download_smitha2025_spectra()` for prefetching one or all
  tables. Products are discrete, without interpolation. Fluxes are interpreted
  as emergent surface flux; the release does not specify an air/vacuum
  convention, so the native wavelength coordinate is retained. Nonuniform
  native sampling can limit resolution-degradation requests.
- Added the Kostogryz et al. (2026) angle-dependent specific-intensity library
  (`kostogryz2026`) from Edmond, with checksum-validated caching and
  `download_kostogryz2026_spectra()`. Exact selection supports the published
  F3, G2, K0, K4, M0, M2, and M4 models and their native metallicities, five
  magnetic states (`hydro`, `ssd`, `B100G`, `B200G`, `B300G`), and ten native
  disk positions (`mu=0.1` through `1.0`). No interpolation, disk integration,
  or limb-darkening fitting is provided; spectra use nonuniform, low-resolution
  ODF sampling ([#80](https://github.com/brackham/speclib/pull/80)).
- Added `SpecificIntensitySpectrum`, `SpecificIntensityGrid`, and
  `BinnedSpecificIntensitySpectrum`. `from_library()` loads one native angle
  or all angles for a model and magnetic state; `SpecificIntensityGrid.at_mu()`
  selects an exact native angle. Individual spectra support wavelength
  selection, slicing, resampling, regularization, Gaussian broadening at constant
  wavelength FWHM, constant R, or wavelength-dependent R, and binning while
  preserving per-steradian units, metadata, and source provenance.
- Added first-class disk-integrated MPS-ATLAS Set 1 and Set 2 support, with
  `mps-atlas` aliasing Set 1. `download_mps_atlas_grid()` and automatic loading
  acquire checksum-validated Edmond v3 archives and read spectra directly from
  set-specific ZIP caches. Exact, nearest-available, and trilinear retrieval
  work through `Spectrum`, `SpectralGrid`, and `BinnedSpectralGrid`; sets are
  never mixed, and missing corners or incompatible wavelength axes raise errors.
  Upstream irradiance at 1 AU is converted to surface flux; the archives'
  center-to-limb intensity products are not exposed.
- Added SPHINX I V4 loading through `Spectrum.from_grid()`, `SpectralGrid`, and
  `BinnedSpectralGrid`, with `download_sphinx_grid()` and checksum-validated
  Zenodo downloads. An explicit `co_ratio` selects a fixed native C/O slice;
  exact, nearest-available, and trilinear retrieval use actual model inventory,
  with errors for missing interpolation corners. C/O is not interpolated.
- Added `set_variable_resolving_power()` to `Spectrum` and `SpectralGrid` for
  wavelength-dependent Gaussian broadening with unchanged output sampling.
  Sampled R values are interpolated linearly and must cover the full input
  axis without extrapolation. The prescription assumes smoothly varying R and
  negligible intrinsic widths, with sampling, work, and memory safeguards.
- Added a Python 3.12 installation smoke-test workflow that builds wheel and
  source distributions, installs the wheel and dependencies in a clean
  environment, runs `pip check`, compares installed runtime requirements with
  `pyproject.toml`, and verifies imports using isolated Python.

### Changed

- **Spectrum base class and dependency migration:** `speclib.Spectrum` now
  subclasses `specutils.Spectrum` instead of `specutils.Spectrum1D`, requiring
  `specutils>=2,<3` instead of `>=1.9.1,<2`. Update dependency constraints and
  code relying on the old base class or `Spectrum1D` type checks. Speclib's
  convenience API and last-spectral-axis defaults remain compatible; upstream
  arithmetic may return plain `specutils.Spectrum` objects without Speclib's
  convenience methods.
- **Breaking metallicity API changes:** standardized native selection on
  `metallicity`, `metallicity_bds`, and `metallicity_range` across spectra,
  grids, photometry, and model helpers. Replace generic `feh=` on NewEra,
  SPHINX, and MPS-ATLAS with `metallicity=` or native `mh=`; `feh=` now raises
  `ValueError` for these [M/H] grids. PHOENIX-ACES retains warning-free `feh=`.
  Replace removed `feh_bds`, `feh_range`, `fehs`, and `grid_fehs` names with
  `metallicity_bds`, `metallicity_range`, `metallicities`, and
  `grid_metallicities`. Existing positional order and numeric selection are
  preserved; no [Fe/H]-to-[M/H] conversion is performed. DRIFT-PHOENIX and
  NextGen retain legacy coordinate behavior apart from shared mechanical renames.
- Metallicities, bounds, and ranges reject explicit `None`; omit optional
  arguments to retain defaults. Multiple coordinate forms raise `ValueError`
  even when equal, and spectral-grid constructors reject scalar metallicity
  overrides in favor of `metallicity_bds`. Native `mh_bds`/`mh_range` aliases
  are available for [M/H] grids where applicable. NewEra helpers' `z` and
  `zscale` keywords remain accepted with deprecation warnings; use `metallicity`.
- **Breaking `SpectralGrid.get_spectrum()` return change:** in beta 12 this
  deprecated alias of `get_flux()` returned an Astropy flux `Quantity` and
  emitted `DeprecationWarning`. It now returns an independent Speclib `Spectrum`
  with copied wavelength, flux, and metadata, including instrument provenance
  and detector coverage, without a warning. Call `get_flux()` to retain the old
  quantity-returning behavior, or access `.flux` on the returned spectrum.
  `BinnedSpectralGrid.get_spectrum()` still returns a flux `Quantity`.
- Added `metallicity_type` metadata (`"feh"` for PHOENIX-ACES, `"mh"` for
  NewEra, SPHINX, MPS-ATLAS, and Kostogryz et al. 2026), plus numeric coordinate
  metadata for spectra, SEDs, and grid axes. Smitha et al. 2025 has no metallicity
  axis or metallicity metadata fields.
- NewEra V3 now targets FDR release 3.5 (record 18108) instead of release 3.4
  (record 17935), with explicit supplemental HSR availability from record 17936
  and inventory-provided download URLs. There is no silent fallback to 3.4.
- The regular NewEra temperature backbone is now 2300–7000 K by 100 K and
  7200–12000 K by 200 K. Sparse native special/benchmark models remain
  available for exact retrieval without becoming interpolation planes.
- NewEra grid construction skips missing native tuples, and HSR subset
  downloads filter the actual inventory rather than Cartesian axis products.
- MPS-ATLAS loading replaces the legacy manually populated, undifferentiated
  cache with set-specific archives. Replace old files directly under
  `mps-atlas` using `download_mps_atlas_grid()`.
- `Spectrum.resample()` now preserves an independent copy of input metadata,
  including Smitha et al. source provenance.

### Fixed

- Fixed reduced NewEra off-grid interpolation failing on nonexistent rounded
  metallicity filenames. `Spectrum.from_grid()` loads only required native
  models and reports missing corners clearly; `SpectralGrid` retains its
  nearest-neighbor fallback for missing corners
  ([#88](https://github.com/brackham/speclib/issues/88)).
- `find_bounds()` now selects true interpolation brackets on nonuniform grids,
  preventing spectrum interpolation from using two points on the same side
  of a requested coordinate.
- MPS-ATLAS wavelength-axis normalization now sorts wavelengths and flux
  together, preserving their pairing and returning strictly increasing
  coordinates. Only exact duplicates with consistent fluxes are collapsed;
  conflicting duplicates raise an error.

### Documentation

- Reorganized Read the Docs around installation, Quickstart, task-based user
  guides, scientific references, and a generated API reference, with integrated
  citation, changelog, and contributing information.
- Added five executable, rendered, and downloadable MyST-NB tutorials:
  Quickstart, spectral grids, spectral resolution, synthetic photometry, and
  instrument profiles.
- Added Smitha et al. (2025), Kostogryz et al. (2026), MPS-ATLAS, and SPHINX
  library references covering native sampling, units, parameter selection,
  caching, scientific limitations, metadata, and citations; updated NewEra
  Gaia/JWST/Low-Res product comparisons and interpolation guidance.
- Added generated instrument configuration/source tables, scientific assumptions
  and limitations, and offline provenance references. Reproducible calibration
  conversion instructions, detailed investigations, and the contributor
  provenance checklist are retained in `scripts/instruments/README.md`.
  WFC3/UVIS G280 remains intentionally unsupported because reviewed wavelength,
  sensitivity, and image-core calibrations do not sufficiently constrain
  along-dispersion spectral FWHM; follow-up is tracked in
  [#99](https://github.com/brackham/speclib/issues/99).
- Improved installation and update instructions for GitHub, local checkouts,
  and existing installations, including reinstalling after source updates to
  refresh dependency metadata. Expanded contribution guidance and added bug,
  feature, and model-library issue forms and a pull-request template.


## [0.1.0b12] - 2026-08-21

### Added

- Added constant spectral-resolving-power convolution with
  `set_spectral_resolving_power(resolving_power)` on both `Spectrum` and
  `SpectralGrid`, alongside matching
  `set_spectral_resolution(delta_lambda)` APIs.
- Added validation for under-sampled requests and excessive temporary-grid
  memory use.

### Changed

- Constant spectral-resolution convolution now correctly applies a fixed
  Gaussian wavelength FWHM, while constant resolving power is evaluated on a
  uniform log-wavelength grid.
- Both convolution modes support nonuniform wavelength grids internally while
  preserving the original wavelength sampling, and `SpectralGrid` no longer
  calls `regularize()` implicitly.

### Testing

- Added quantitative regression tests for constant wavelength FWHM, constant
  resolving power, nonuniform sampling, API consistency, and input validation.

### Documentation

- Documented the revised resolution APIs, numerical conventions, sampling
  requirements, flux treatment, and current limitations.

### Related issues

- Closes #36: Correct spectral-resolution handling and add constant resolving
  power


## [0.1.0b11] - 2026-01-28

### Added

- Added `download_newera_grid()` convenience helper that downloads and optionally
  refreshes NewEra model grids (`newera_gaia`, `newera_jwst`, `newera_lowres`) while
  clearing outdated files when `overwrite=True`.
- Documented NewEra tarball caching with on-demand extraction in `README.md`.

### Changed

- NewEra grid downloads now cache tarballs without unpacking by default; loaders
  extract only the required `.txt` file on demand.
- `download_newera_grid()` now supports `extract="all"` for users who want to
  explicitly unpack the full grid.
- NewEra tar extraction now resolves the requested member directly (with cached
  name lookups) to avoid repeated full scans of tar members.

### Documentation

- Added citation metadata for the software and Rackham & de Wit (2024).
- Added Zenodo creator/contributor metadata distinguishing Benjamin V. Rackham as creator and Zachory K. Berta-Thompson as contributor.
- Added contributor-credit policy documentation.

### Related issues

- Closes #61: Avoid unpacking full NewEra tarballs by default; extract only required files


## [0.1.0b10] - 2025-10-27

### Fixed

- Updated tox configuration to test only supported Python versions (3.11–3.13). All tests now pass cleanly across these environments.
- **SpectralGrid** now clips user-specified temperature, gravity, and metallicity bounds to the available model grid range, emitting a `UserWarning` when truncation occurs to prevent zero-size array errors during initialization.
- **SpectralGrid.get_flux(interp=True)** now always returns a **1-D flux array** aligned with `self.wavelength`.
  * Previously, for NewEra grids (`newera_jwst`, `newera_gaia`, `newera_lowres`), the interpolated flux was returned as shape `(1, N)` instead of `(N,)`, causing downstream shape mismatches.
  * Implemented normalization via `np.atleast_1d(np.squeeze(...))` and validation of the resulting dimension.
  * Added explicit docstring clarification that `get_flux` returns a 1-D vector.
- **NewEra grids** now respect the `interpolate` flag in both `Spectrum.from_grid` and `SpectralGrid.get_flux`, restoring parity with PHOENIX interpolation behavior.

### Added

- **Unit test:** `test_spectral_grid_clips_bounds_to_available_grid` validating that out-of-range bounds are truncated with warnings while still loading the expected spectra.
- **Unit test:** `test_newera_flux_shape_is_1d` in `tests/test_interpolation_toggle.py` to ensure both interpolated and nearest fluxes are 1-D for NewEra grids.
- **DummyInterpolator** helper class to simplify isolated shape tests.

### Verified

- All model grids (`phoenix`, `newera_jwst`, `newera_gaia`, `newera_lowres`) now pass shape consistency checks.
- Regression test confirms matching wavelength/flux lengths and no extra leading dimension.
- Verified with full test suite: `poetry run pytest` and `poetry run tox` both pass.

### Related issues

- Closes #40: Interpolation does not work with newera_jwst grid
- Closes #51: 🐛 Bug: Some model grids return flux with the wrong shape (extra leading dimension) when using `SpectralGrid.get_flux(interp=True)`*.
- Closes #57: 🐛 Bug: Inconsistent interpolation behavior for NewEra grids in speclib


## [0.1.0b9] - 2025-10-24

### Added

- `docs/requirements.txt` specifying Sphinx dependencies (`myst-parser`, `furo`).

### Changed

- Renamed `SpectralGrid.get_spectrum()` to `SpectralGrid.get_flux()` for clearer semantics and consistency across the API.
- Updated `.readthedocs.yml` to install documentation requirements explicitly, ensuring Read the Docs builds succeed without missing extensions.
- Updated NewEra helpers to use the V3.4 release (record 17935).

### Deprecated

- `SpectralGrid.get_spectrum()` remains available as a deprecated alias that raises a `DeprecationWarning` and will be removed in a future release.

### Fixed

- Documentation build failures on Read the Docs caused by missing `myst_parser` import errors.
- Improved local build reproducibility using `sphinx-build -b html docs _build/html`.

### Testing

- Updated `tests/test_interpolation_toggle.py` to cover the new method name and deprecated alias.


## [0.1.0b8] – 2025-10-20

### Added

- Allow environment variable `SPECLIB_LIBRARY_PATH` to override the default `~/.speclib/libraries` path.
- Add `speclib.utils.set_library_root()` helper to otherwise change the path.
- Documentation updates and regression tests for custom cache paths.


## [0.1.0b7] - 2025-08-15

### Changed

- ✨ Switched from `pysynphot` to `synphot` throughout the codebase, including `Spectrum.resample()` and filter application routines. ([#41](https://github.com/brackham/speclib/pull/41))
- 🧹 Removed compatibility workarounds for `pysynphot` imports and deprecated `warnings.catch_warnings()` block in `core.py`.
- 🔧 Updated `README.md` to reference `synphot` instead of `pysynphot`.
- 🔧 Relaxed Python version constraint to `>=3.11,<3.14`(previously `<3.12`) to support Python 3.12 and upcoming 3.13.
- ⬆️ Upgraded `astropy` from `6.0.1` to `7.1.0` and relaxed the version constraint to `>=6.0,<8.0`.
- ⬆️ Upgraded `matplotlib` from `3.7.5` to `3.10.5`, with updated constraint to `>=3.8,<4.0`.
- 🔧 Updated version constraints for `specutils`, `synphot`, `numpy`, `scipy`, `pooch`, and `tqdm` to reflect compatibility with newer Python and package versions.
- 📦 Synced `poetry.lock` to reflect updated and removed packages.

### Fixed

- 🐛 Replaced deprecated `np.trapz` calls with `np.trapezoid` to avoid warnings in NumPy 2.0+.

### Development

- 🧪 Removed deprecated warning filters in `pytest.ini` related to `pysynphot`.
- ✅ Confirmed compatibility with Python 3.11+ and current dependency stack.


## [0.1.0b6] – 2025-07-25

### Changed

- Relaxed `astropy` version constraint to support `astropy>=6.0.0,<8.0.0` (previously `<6.0.0`), enabling compatibility with newer releases.
- Confirmed full test suite passes with `astropy 6.x`.

### Testing

- Verified flux conservation after resampling spectra with `Spectrum.resample()`.
- Manually tested interpolation behavior in `Spectrum.from_grid()` and `SpectralGrid.get_flux()`.


## [0.1.0b5] - 2025-06-26

### Fixed

- Automatically trigger NewEra tarball download when loading a wavelength array for the first time.
  `load_newera_wavelength_array()` now calls `download_newera_grid()` if the expected `.txt` file is missing.
- Corrected the metallicity list for the MPS-Atlas grid in `utils.py` where `-0.95 - 0.9` was accidentally
  combined into a single entry.
- Fixed a bug in `SpectralGrid` that caused interpolation to fail with sparsely sampled NewEra grids. Now uses `NearestNDInterpolator` to support these cases. ([#34](https://github.com/brackham/speclib/pull/34))

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

## [0.1.0-beta.1] - 2024-05-16

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
