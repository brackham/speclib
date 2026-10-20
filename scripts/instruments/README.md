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

`profiles.toml` transcribes the nine SUB2048 rows of
[BOTS Table 1](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps),
including three detector gaps. Coverage is approximate and independent of the
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
subarray. Only SUB2048 is bundled in phase 1. Further subarrays can receive
separate registry entries sharing a disperser's existing CSV, without changing
the numerical code or falling back to an incorrect configuration.

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
