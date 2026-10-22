# Reproducing NIRSpec calibration assets

From the repository root, using the Poetry development environment:

```sh
poetry run python scripts/instruments/convert_nirspec.py --download --source-dir /tmp/nirspec-fits
# Subsequent conversions can be fully offline:
poetry run python scripts/instruments/convert_nirspec.py --source-dir /tmp/nirspec-fits --output-dir /tmp/nirspec-csv
```

`sources.json` in `src/speclib/data/instruments` is the authoritative record of
exact original URLs/filenames, source/output checksums, source definition,
FITS headers/history and conversion details. The original links are the files
listed under **Dispersion curves** on the
[STScI NIRSpec dispersers page](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters).
Each profile exposes its record offline as `get_profile(mode).provenance`.
The converter rejects changed files, wavelength units, row counts, nonphysical/nonmonotonic
values, and R inconsistent with the documented 2.2-pixel resolution definition
by more than 1%. This tolerance accommodates independent interpolation of R and
dispersion in the updated high-resolution tables; it does not claim 1% physical
accuracy. All original wavelength/R float32 values are promoted to float64 and
written with 17 significant digits. Source extension 1 supplies `WAVELENGTH`
(micrometers), `DLDS` (micrometers/pixel), and dimensionless `R`. The CSV contains
only `wavelength_um,resolving_power`; R is copied, never recomputed from DLDS.
No points are dropped or invented. The files declare `GROUND`/`MODELING`, not
measured in-flight LSFs; the headers do not explicitly flag air/vacuum wavelengths.
Speclib treats them as vacuum and adopts the nominal element as Gaussian FWHM.

`profiles.toml` transcribes all regular-subarray columns of
[BOTS Table 1](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps),
including disconnected intervals and physical detector gaps. Coverage is approximate and independent of the
dispersion-table domain. Both configuration and source tables in Sphinx are
generated from these records by `docs/_instrument_tables.py`.
Review any future source changes, update the pins deliberately, reconvert, and
run `poetry run pytest tests/test_instruments.py tests/test_instrument_data.py`.
To compare with local official FITS files as well, set
`SPECLIB_NIRSPEC_SOURCE_DIR=/tmp/nirspec-fits` for that test run. Ordinary tests
never download data. Compare regenerated CSV bytes with each `output_sha256`
in the manifest before replacing assets. Confirm redistribution terms again for
new sources; the present review and attribution are recorded in `NOTICE.md`.

Explicit configuration identifiers include the observing mode, slit and
subarray. SUB1024A (except PRISM), SUB1024B, SUB512 and SUB512S have separate
registry entries sharing each disperser's unchanged CSV. SUB512/SUB512S have
identical tabulated intervals but different spatial heights. The original
SUB2048 short identifiers and explicit aliases remain unchanged. Shared
coverage provenance, geometry and physical detector gaps are recorded in
`sources.json`, separately from the original resolution-data retrieval date.
No reconversion is needed for the subarray extension. Include
`tests/test_nirspec_subarrays.py` when checking its coverage contracts.

## Numerical support and assumptions

The runtime convolution uses source-centered log-wavelength Gaussians from the
existing resolution engine. For each piecewise-linear R calibration cell, its
minimum endpoint R bounds the largest four-sigma radius throughout that cell.
First bound all source centers whose wings can reach the requested detector
interval, then retain their full support to avoid domain-edge normalization
bias. This includes source light inside output detector gaps, without assuming
that kernel reach is monotonic. Input samples bracketing the envelope must also
be calibration-covered. Full PRISM needs approximately 0.570080–5.355472 µm plus
bracketing samples, so a suitably sampled source ending at 5.5 µm is adequate.

The original FITS timestamps do not capture later table updates. The manifest
retains both timestamps and history, including G235H/G395H updates announced on
2024-06-24. Bundled NOTICE.md records attribution and the reviewed STScI
redistribution policy; the numerical R columns are preserved unchanged. Some
internal intermediate files and the header-cited technical note have no verified
public URL; the linked FITS files, not those intermediates, are the reproducible
inputs. Attachment version numbers are not formal calibration release versions.

## Reproducing Phase 2: STIS, WFC3 and NIRISS/SOSS

```sh
poetry run python scripts/instruments/convert_phase2.py --download --source-dir /tmp/stis-lsf
poetry run python scripts/instruments/convert_phase2.py --source-dir /tmp/stis-lsf --output-dir /tmp/phase2-csv
SPECLIB_STIS_SOURCE_DIR=/tmp/stis-lsf poetry run pytest tests/test_phase2_instruments.py
```

`phase2_sources.json` is the canonical record of source links, byte pins, derived
LSF widths, supplementary handbook anchors, equations, coefficients, domains,
processing and redistribution review. The script validates input checksums,
aperture/wavelength headers, array dimensions, finite ordered offsets and
nonnegative amplitudes before deriving STIS half-height widths. It linearly
interpolates pixel FWHM between anchors, holds endpoint FWHM outside them,
converts with the mean dispersion and samples R at 0.001 µm spacing with exact
anchors. Runtime interpolation remains linear in R. Those endpoint extensions
are explicit approximations, including over padding, rather than extrapolation
of an empirical calibration. G750L uses the 7000 Å model LSF plus the handbook's
5500 and 10000 Å widths. The less-than-0.1-pixel differences from rounded
handbook values are retained, not tuned away.

WFC3 uses `R(lambda) = R_ref * lambda / lambda_ref`, a constant wavelength FWHM
anchored to Table 8.1, not an arbitrary two-pixel width. SOSS uses separate
measured order dispersions from Albert et al. (2023), Table 2, and the paper's
two-pixel convention, yielding a linear R curve for each order. Two endpoint
samples reproduce these analytic equations exactly. The selected padding domain
is a declared part of each nominal prescription, not a measured validity claim.
No numerical optical-model derivative or visit-specific correction is inferred.
No network calls occur without `--download`; WFC3/SOSS reproduction needs no
external data files. Output checksums are verified before writing. Changes to
inputs or scientific choices require explicit review and deliberate pin updates.

SOSS uses CLEAR/SUBSTRIP256, with independent ordinary output spectra. Order 2
practical coverage follows the current STScI calibration-status advice against
using wavelengths above 1 µm; the manifest also records the wider geometric
range. Orders are not stitched or decontaminated. STIS assumes the prime CCD
settings, unbinned pixels and centered point sources through 52X2. Unsupported
apertures/subarrays do not fall back to the defaults. See the generated tables
and source discussion in `docs/user_guide/instrument_provenance.rst`.


## Reproducing Phase 3A: NIRCam Module A Grism R time series

```sh
poetry run python scripts/instruments/convert_nircam.py --output-dir /tmp/nircam-csv
poetry run pytest tests/test_nircam_instruments.py
```

`nircam_sources.json` records the original Greene et al. (2017), Section 3.2
coefficients in ascending order, wavelength units, physical interpretation,
source URLs and output checksum. Reproduction is entirely offline; the
published equation is the input, with no invented calibration FITS file.
The converter samples 2601 wavelengths from 2.4 to 5.0 um at 0.001 um spacing,
writes float64 values with 17 significant digits, and rejects changed output
bytes before writing. It never updates pins. Runtime linearly interpolates R;
the maximum fractional interpolation error against direct polynomial evaluation
is below 4e-8. That is separate from the original few-percent optical-model fit
and numerical Gaussian convolution accuracy.

All four profiles share `nircam_grismr.csv`. Only first-order Module A Grism R
time-series configurations are supported. `profiles.toml` records adopted
output intervals: the original 5% filter limits for F322W2/F356W, conservative
inward cuts to 2.410 um for F277W and 4.985 um for F444W. These cuts preserve
convolution support within the published 2.4-5.0 um model domain, without
polynomial extrapolation or endpoint extension. The manifest retains all
original filter edges and the separate approximate 5.03 um TSO detector cutoff.
Real input padding and calibration-covered bracketing samples remain required.
See the scientific reference for second-order/neighbor contamination caveats.


## Reproducing Phase 3B: MIRI P750L/SLITLESSPRISM

```sh
poetry run python scripts/instruments/convert_miri.py --output-dir /tmp/miri-csv
poetry run pytest tests/test_miri_instruments.py
```

`miri_sources.json` records STScI's approximate 5/7.5/10 um reference values
and explicit linear trend, and Kendrew et al. (2015), Section 2 / Figure 3's
linear design curve through 14 um. The adopted nominal equation is
`R = 24*(lambda/um)-80` on 5-14 um. This is a design approximation, not an
in-flight slitless Gaussian LSF calibration. The original numerical values and
trend are the inputs; no source calibration file is invented. Inspected CRDS
SPECWCS files are audit evidence only and are not required for reproduction.

The converter writes 901 float64 wavelength/R samples in 0.01 um steps, using
17 significant digits, and rejects changed output checksums without updating
pins. The rows evaluate a nominal model; they are not measured calibration
points. Linear R interpolation is exact to roundoff. Dense cells keep the
existing conservative padding envelope local, especially near the blue end.
No network is used. Model support is wider than the adopted 5.5-12 um output;
real source padding and calibration-covered bracketing samples remain required.
The manifest and scientific reference explain the blue numerical cut, foldover,
red throughput limit, unsupported Cycle 6 IP/IPS placements, physical width
interpretation, audit URLs/checksums and redistribution review.
