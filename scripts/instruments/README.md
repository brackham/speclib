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


## Profile development standards

Every bundled profile must connect its numerical prescription to authoritative,
clickable sources. Keep the record small and specific to the source type; do not
invent a file, version, precision or measurement pedigree to fill a field.
This checklist applies to bundled STIS, WFC3, NIRISS, NIRSpec, NIRCam and MIRI LRS
profiles and future profiles.

### Minimum record

* **Configuration:** use `mode` and `configuration` for telescope, instrument,
  disperser/filter and relevant slit/aperture/subarray/order. Do not silently
  substitute a different setup.
* **Resolution source:** identify the responsible institution/team where known,
  the source kind (nominal prediction, dispersion model, measured LSF, analytic
  prescription, or another stated kind), and an authoritative documentation,
  dataset or publication URL. `source`/`source_url` give the human summary;
  the manifest records the original dataset URL or equation reference.
* **Coverage source:** use `coverage_source_url` for the separate authority
  supporting all valid intervals/gaps. Document approximations, configuration
  dependence and any limits chosen by Speclib rather than the source.
* **Identification:** `calibration_version` must identify the selected source
  or snapshot. Record available filenames, versions, dates and stable identifiers
  in the manifest. State unknowns explicitly; retrieval dates and FITS creation
  dates must not be presented as calibration release versions.
* **Physical meaning:** specify the supplied width/LSF definition, units,
  illumination/sampling convention and measurement or prediction pedigree.
  `resolution_definition` states how Speclib interprets it; a sampled table
  (`quality='tabulated'`) need not be an empirical measurement.
* **Processing:** record original columns, units and selection, conversions,
  interpolation, fitted coefficients/adopted constants and approximations.
  Separate source processing from Speclib's conversion and runtime evaluation.
  Use `assumptions` for the supported scientific limitations.
* **Reproducibility:** supply a conversion script and source/output checksums
  for distributed data files. For an analytic prescription, record the equation,
  coefficients, units, validity domain and DOI/section instead; no fictitious
  source file or checksum is required. For measured LSF data, retain their
  normalization and observational conditions before any R/Gaussian reduction.
* **Redistribution:** cite the applicable policy/license URL, attribution and
  review basis. Flag missing permission or restrictions before bundling data.

### One authoritative record per fact

Use the existing `InstrumentProfile` fields and a JSON manifest. A profile's
optional `provenance_ref="manifest.json#source_key"` selects a record bundled
under `speclib/data/instruments`. The public `profile.provenance` property
combines the manifest's optional `defaults`, `retrieved` date and the selected
`sources` entry into a detached dictionary, offline. Custom user profiles may
omit this locator; bundled profiles must provide a complete record.

Use descriptive flat record keys, following NIRSpec: `institution`,
`source_kind`, `documentation_url`/`publication_url`, `source_definition`,
`processing` and `reproduction_script`. File sources additionally have
`url`, `filename`, `sha256`, `output_file` and `output_sha256`; analytic
sources can instead have `equation` and `coefficients`. Source-specific
header fields are optional. Record output checksums whenever distributing a
derived numerical table, including one sampled from an analytic prescription.
A manifest can share defaults among related sources;
different instruments can use separate manifests. No new provenance class or
unrelated fields are needed.

Keep profile configuration/coverage in the TOML registry and original-data facts
in the manifest. Reference shared source records rather than copying citations
into each filter entry. Generate documentation tables from those records, as
`docs/_instrument_tables.py` does for NIRSpec. Every scientific citation must
render as a hyperlink, including original-data links distinct from descriptive
pages. Use persistent identifiers when available; describe retrieval by filename
when attachment URLs are unstable.

### Review and reproduction checklist

1. Open every source and check its actual contents, configuration and definition;
   record unavailable intermediates or uncertain authorship/version information.
1. Reproduce output files without network access from pinned inputs. Preserve
   raw values unless a documented transformation is scientifically necessary.
1. Check numerical integrity once, plus representative line widths, boundaries,
   gaps, sampling, metadata ownership and grid consistency. Keep extensive
   independent scientific campaigns separate from routine regression tests.
1. Inspect `get_profile(mode).provenance` and the rendered configuration/source
   tables. Confirm links, offline wheel/sdist access, attribution and installation.
1. Build Sphinx with warnings as errors and execute the lightweight tutorials;
   add concise user examples. Change the changelog only when requested.

For bundled curves, follow the [reproduction procedures](README.md) and the
[scientific reference](../../docs/user_guide/instrument_provenance.rst). The source
manifests are downloadable from that reference; reproducing data must not silently
replace checksum pins when upstream attachments change.


## Calibration source review, October 2026

Detailed investigation notes retained for maintainers. Current supported
configurations, assumptions and generated tables are in the
[public scientific reference](../../docs/user_guide/instrument_provenance.rst).

### WFC3/UVIS G280: conditional deferral

The 2026-10-22 review did not establish a sufficiently supported
wavelength-dependent spectral FWHM prescription for a bundled G280 profile.
The intended configuration was a point source in positive first order near
the center of UVIS chip 2. No G280 identifier, alias, curve or adopted output
interval is registered.

[Handbook Table 8.1](https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview)
quotes R approximately 70 at 300 nm, based on a Gaussian image core measured
perpendicular to dispersion. This implies a reference width of about 4.29 nm;
it does not determine its wavelength dependence. A two-pixel element at the
listed mean dispersion of 1.3 nm/pixel would instead imply R approximately 115.
Neither a fixed pixel width nor a constant wavelength FWHM is adopted.

[ISR 2017-20](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-20.pdf)
and [ISR 2020-09](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2020/WFC3-ISR-2020-09.pdf)
calibrate field-dependent traces and wavelength solutions. Their centroid
residuals are wavelength errors, not instrumental linewidths. The official
[G280 reference files](https://www.stsci.edu/hst/instrumentation/wfc3/documentation/grism-resources/wfc3-g280-calibrations)
reviewed here include chip-specific V2.5/V3.0 configuration files and 2021
positive/negative first-order sensitivity tables. These supply coordinates,
dispersion and sensitivity, without a calibrated spectral-width relation.

[ST-ECF TIR 2009-04](https://www.stecf.org/documents/isr/WFC3_TIR-2009-04.pdf), Section 3,
measures wavelength-dependent cross-dispersion Gaussian cores in ground tests.
The monochromator spots are elongated along dispersion. Transferring the
transverse widths to spectral FWHM would require an additional optical or
empirical justification; direct-imaging widths do not supply that relation.

The [G280 handbook discussion](https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-2-slitless-spectroscopy-with-the-uvis-g280-grism)
distinguishes the higher-throughput positive order, favors chip 2 for a single
UV target, and describes changing dispersion and higher-order overlap beyond
about 400 nm. Nominal 190–800 nm coverage therefore does not establish a useful
single-order Gaussian-matching interval or resolution support for padding.

A future profile needs a published spectral Gaussian-width prescription or
independent point-source line-width/optical-model constraints at several
wavelengths for the selected chip, position and order, including support beyond
the output boundaries. G280's inclusion in Issue #91 was conditional on suitable
data. Its deferral leaves the 55 supported profiles unchanged. No G280 source
files or publication figures are redistributed with the package.
Future G280 calibration and profile work is tracked in
[Issue #99](https://github.com/brackham/speclib/issues/99), separately from
the initial instrument-profile implementation in
[Issue #91](https://github.com/brackham/speclib/issues/91).

### Original resolving-power data

The [STScI NIRSpec dispersers page](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters)
distributes seven dispersion/resolution FITS tables used by the ETC. Its
instrument reference is [Jakobsen et al. (2022), A&A 661, A80](https://doi.org/10.1051/0004-6361/202142663)
([author preprint](https://arxiv.org/abs/2202.03305)). Speclib copies the linked
tables, not values digitized or inferred from that publication.

(See the generated `sources` table in the public instrument reference.)

NASA/STScI is the distributor. All seven headers identify `SYSTEM=MODELING`
and `PEDIGREE=GROUND`. G235H/G395H name Pierre Ferruit as author; the other
headers name a generation script, not a complete calibration-team author list.
These are nominal dispersion-derived prescriptions, not empirical in-flight
or BOTS point-source LSF measurements. `quality='tabulated'` identifies the
representation and does not imply empirical accuracy.

The source table contains `WAVELENGTH` (micrometers), `DLDS` (micrometers
per pixel) and dimensionless `R` in binary-table extension 1. STScI defines
a 2.2-pixel resolution element, approximately
$R=\lambda/(2.2\,\mathrm{DLDS})$, for typically fully illuminated
IFU/MOS/200 mas fixed-slit apertures. This is not a measured Gaussian FWHM for
the wide S1600A1 slit. The FITS wavelength unit does not explicitly flag air
versus vacuum; Speclib treats the coordinates as vacuum without conversion.

### Identification and processing

`calibration_version` labels the selected Speclib source snapshot, retrieved
2026-10-17. There is no stated formal calibration release version for these
attachments. Full SHA-256 pins identify their bytes. FITS creation timestamps
above are not last-revision dates: G235H/G395H retain 2016 timestamps, but their
histories record interpolation by G. M. Wahlgren on 2024-05-16, and the linked
STScI page announces their update on 2024-06-24. Attachment version `1` is not
a calibration version.

The [conversion script](convert_nirspec.py)
checks each original checksum, timestamp, row count, units, ordering and numerical
validity, then copies every `WAVELENGTH`/`R` pair. Source float32 values are
promoted to float64 and written with 17 significant digits to
`wavelength_um,resolving_power` CSVs. Conversion does not fit, interpolate,
thin, or recompute R. Runtime evaluation linearly interpolates the published R
column in wavelength and never extrapolates it.

The original G235H/G395H R and dispersion columns are separately interpolated;
their maximum deviations from the 2.2-pixel ratio are about 0.403% and 0.836%.
The converter's 1% consistency check accommodates this source property; it is
not a physical-accuracy claim. Published R values remain untouched.

Some original intermediate sources cannot be independently retrieved: five
`LITREF` values name internal as-built files; G235H/G395H name the technical
note `ESA-JWST-SCI-NRS-TN-2016-024`, for which no public copy was located.
These are header identifiers, not unverifiable citations substituted for the
data. The directly linked official FITS tables are the reproducible inputs.
If attachment URLs move, locate the original filenames on the linked STScI
page and require the same checksums before accepting replacements.

### Coverage and Gaussian approximation

The interval limits and gaps come separately from [BOTS wavelength ranges and gaps, Table 1](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps).
Speclib transcribes these approximate S1600A1 limits in `profiles.toml`.
They are not derived from the R tables, and they vary with the observation's
wavelength solution and grating-wheel position. The regular SUB2048, SUB1024A,
SUB1024B, SUB512 and SUB512S columns were checked on 2026-10-22; the complete
registry-generated matrix is in the [scientific reference](../../docs/user_guide/instrument_provenance.rst). PRISM/CLEAR is
unavailable for SUB1024A. Multistripe configurations are not implemented.

The [detector subarray reference, Tables 1–2](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-detectors/nirspec-detector-subarray-mode)
distinguishes SUB1024A (blue NRS1/red NRS2) from SUB1024B (red NRS1/blue NRS2).
SUB512 and SUB512S have the same 512-pixel spectral extent and Table 1 coverage,
but spatial heights of 32 and 16 pixels. SUB512S has no unilluminated spatial
pixels in the trace. PRISM is recommended for these smaller subarrays; the
coverage table and BOTS template also permit partial grating coverage.

Physical detector gaps are 1.31–1.35 µm for G140H (outside the F070LP output),
2.20–2.27 µm for G235H and 3.72–3.82 µm for G395H. The omitted regions between
subarray intervals can be much larger; they must not be interpreted as a wider
physical detector gap. All omitted wavelengths remain absent from the output.
Source light from these regions can still contribute wings before cropping.

The seven unchanged R tables and original/output checksums are shared across
subarrays. Selecting fewer detector columns or spatial rows does not establish
a different instrumental R prescription. Speclib retains the nominal 2.2-pixel
definition, including its existing wide-slit/point-source limitation, without
inventing a geometry correction. All new interval edges support the existing
conservative convolution envelope within the original calibration domain.
Real input padding and calibration-covered bracketing samples are still
required; unsupported boundaries raise errors rather than extrapolate R.

Speclib adopts the tabulated $\lambda/R$ as **Gaussian FWHM** using the
existing source-centered log-wavelength convolution. This is a Speclib modeling
choice, not a claim that the source supplies Gaussian LSF measurements. It
assumes negligible intrinsic model widths and applies no deconvolution.
There is no full non-Gaussian LSF, throughput, detector response/pixel geometry,
noise, systematics, or observation-dependent resolution/wavelength solution.
Output sampling is separate: synphot's bin integration and sparse samples can
change apparent FWHMs. See [usage guide](../../docs/user_guide/instrument_profiles.rst) for sampling guidance.

### Programmatic records and redistribution

```python
from speclib import get_profile

profile = get_profile("jwst/nirspec/g395h/f290lp")
record = profile.provenance  # Fresh dictionary; no network request.
print(record["url"], record["filename"], record["sha256"])
print(record["fits_date"], record["history"], record["processing"])
print(profile.coverage_source_url, profile.resolution_definition)
print(record["coverage_retrieved"], record["subarray_source_url"])
print(record["subarrays"]["SUB512S"], record["nominal_detector_gap_um"])

```
The [packaged manifest](../../src/speclib/data/instruments/sources.json)
is authoritative for original/output filenames and checksums, source headers,
URLs and conversion details. `profiles.toml` is authoritative for profile
configuration, coverage and adopted assumptions. Published tables are generated
from these records. Both files and the curves ship in wheels and sdists.

The [STScI content-use policy](https://www.stsci.edu/copyright) permits use
under its public-domain-like NASA/STScI terms unless otherwise stated, requests
acknowledgement, and identifies exceptions for third-party restrictions. Review
of the source page and all seven headers found no contrary notice. This is the
redistribution basis, not a dataset-specific license or endorsement; Speclib's
code license does not relicense the source data. Attribution and the review date
are in [NOTICE.md](../../src/speclib/data/instruments/NOTICE.md).

### STIS, WFC3 and NIRISS prescriptions

The [Phase 2 manifest](../../src/speclib/data/instruments/phase2_sources.json)
records the original sources, equations, coefficients, calibration domains and
output checksums. These profiles use the same offline representation and
Gaussian engine as NIRSpec. Their configuration and practical coverage are in
`profiles.toml`; the tables below are generated from those records.

(See the generated `phase2_sources` table in the public instrument reference.)

#### STIS: model LSFs reduced to Gaussian widths

The [STScI spectral-resolution page](https://www.stsci.edu/hst/instrumentation/stis/performance/spectral-resolution)
provides model LSF text files, with relative pixel offset and four aperture
columns. The converter selects `52x2.0` and finds the half-maximum crossings
on each side of the global peak by linear interpolation. It preserves the source
amplitudes while measuring the width; no normalization, Gaussian fit or
second-moment width is substituted. These are **model point-source LSFs**, not
measurements of a particular observation. The files have no formal calibration
version; exact SHA-256 pins identify the retrieved source bytes.
The [STIS handbook's LSF introduction](https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-6-line-spread-functions)
states that internal-lamp and on-orbit observations confirmed and revised the
original pre-flight models. The public text files do not identify which
individual revisions they contain.

(See the generated `stis_widths` table in the public instrument reference.)

The derived widths differ from the rounded [handbook Table 13.41](https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-6-line-spread-functions/first-order-line-spread-functions)
values by less than 0.1 pixel. The percentage differences above are a source
consistency check, not a physical-accuracy estimate. In particular, the blue
G430L difference is not established to arise from rounding alone; the
unversioned files do not resolve the discrepancy. The converter retains the
LSF-derived widths without tuning them to the handbook.
There is only one downloadable G750L LSF wavelength on the source page. Its
additional handbook anchors, shown above, provide the wavelength dependence;
G430L uses both downloadable wavelengths. Pixel FWHM is interpolated linearly
between anchors and held constant beyond the outer anchors. This endpoint
extension over the complete grating coverage and padding is an explicit nominal
approximation, not additional calibration data. Average dispersion converts
pixel FWHM to wavelength FWHM. A dense sampled R curve retains each anchor;
runtime interpolation is in R, as for every profile.

Coverage comes separately from the complete spectral ranges for the prime CCD
grating settings in the linked grating reference pages. It does not promise
observation-specific pixel centers. A centered point source through 52X2 is
assumed; a slit-filling source has a much broader LSF. Gaussian matching omits
the original asymmetric wings, slit losses and G750L fringing. The tables are
`quality='derived'` because Speclib reduces LSFs and interpolates their widths.

#### WFC3: reference resolving power with constant wavelength FWHM

[WFC3 Instrument Handbook Table 8.1](https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview)
supplies first-order coverage, mean dispersions and resolving powers at reference
wavelengths. Speclib holds the reference wavelength FWHM constant, so R increases
linearly with wavelength. It does not assume a two-pixel FWHM: the effective
pixel width follows from the reference wavelength, R and dispersion. The
handbook defines R from the Gaussian image core measured perpendicular to
dispersion. The prescription assumes an unresolved source at nominal field
center and ignores morphology, scan geometry and field-dependent dispersion.
It is `quality='nominal'`, not a measured wavelength-dependent calibration.
The two endpoint samples reproduce the linear R equation exactly, including
its declared padding domain.

#### NIRISS/SOSS: independent order-specific nominal curves

[Albert et al. (2023), PASP 135, 075001](https://doi.org/10.1088/1538-3873/acd7a3)
([author preprint](https://arxiv.org/abs/2306.04572)), Table 2 and Section 5.3,
supply separate CV3 average dispersions and the two-pixel Nyquist resolution
convention. Speclib uses the measured dispersions, rather than the parenthesized
optical-model values. Each order's R increases with wavelength under this
constant-average-dispersion approximation. This reproduces the paper's nominal
first-order check near its reference wavelength, but does not reconstruct the
full derivative of an in-flight wavelength solution. Individual LED LSF widths
in Figure 6 vary around the adopted two-pixel convention. The curves are
`quality='nominal'`; no publication figure is digitized or redistributed.

The [GR700XD documentation](https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-gr700xd-grism)
provides the rounded first-order coverage and the CLEAR/SUBSTRIP256 setup.
For order 2, the separate [calibration-status page](https://jwst-docs.stsci.edu/jwst-calibration-status/niriss-calibration-status/niriss-soss-calibration-status)
(updated June 2026) advises against using wavelengths above 1 µm because the
signal is low and photometric calibration is uncertain. Speclib therefore uses
a shorter practical interval than the geometric range. The manifest retains
both the paper's geometric limits and this coverage decision. This review
informs useful coverage; it is not an in-flight resolution calibration. Visit
wavelength shifts, order contamination and extraction corrections are outside
this approximation. The two orders remain independent ordinary spectra.

#### Reproduction and attribution

The [Phase 2 converter](convert_phase2.py)
reproduces all six outputs offline using checksum-pinned STIS inputs and the
manifest's numerical prescriptions. It rejects changed source and output bytes;
it never rewrites pins. See the [reproduction procedure](README.md). Analytic sources require no fictitious
input filename or checksum; their equations and coefficients are the inputs.
All derived CSVs have output checksums. Domains include explicit nominal
padding, with no runtime extrapolation. Wavelength labels do not explicitly
identify air versus vacuum; Speclib treats these approximate coordinates as
vacuum without conversion.

The STIS source headers and STScI pages were reviewed against the [STScI content-use policy](https://www.stsci.edu/copyright). No contrary restriction
was found on the selected numerical LSFs. WFC3 and SOSS CSVs are original
evaluations of cited numerical prescriptions; publication text and figures are
not bundled. NASA/STScI and the SOSS instrument team are acknowledged in
[NOTICE.md](../../src/speclib/data/instruments/NOTICE.md). No NASA,
STScI or author endorsement is implied.


### NIRCam: Module A Grism R time-series prediction

(See the generated `nircam_sources` table in the public instrument reference.)

[Greene et al. (2017), Section 3.2 / Figure 2](https://doi.org/10.1117/1.JATIS.3.3.035001) give the polynomial reproduced in
the [scientific reference](../../docs/user_guide/instrument_provenance.rst). It fits predicted point-source FWHM resolving power
to within a few percent. The model uses a simulated monochromatic PSF, scaled
by 1.1 along dispersion for the pupil beam factor, then sampled in LW pixels
with nominal dispersion 10 Å/pixel. Pixel sampling limits the short-wavelength
resolution; diffraction matters increasingly at longer wavelengths. This is
not a measured in-flight LSF or a constant R=1600 prescription.

The [STScI grisms reference](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-grisms),
Figure 1, repeats the coefficients. Table 3 gives R/C dispersions of
1.004/1.006 nm/pixel with field variations up to 2%. The paper reports all four
grism dispersions within 1% of their nominal design. Neither source gives
separate R/C or module polynomials; identical visit-specific LSFs are not
established. Speclib supports only the actual LW time-series configuration:
**Module A Grism R, first order**, for an isolated point source. Its four filters
share one nominal curve; there is no field-, visit- or subarray-specific PSF
correction. The different standard placements for LW+SW imaging and LW+SW
spectroscopy do not have separately calibrated profiles here.

#### Coverage and conservative edge cuts

Table 4 supplies the four filters' 5% transmission limits, recorded separately
in the [NIRCam manifest](../../src/speclib/data/instruments/nircam_sources.json).
The [grism TSO page](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series)
specifies Module A Grism R and full detector columns in the TSO subarrays.
F444W's standard placement loses wavelengths above approximately 5.03 µm,
before the filter's 5.084 µm transmission edge. Observation-specific usable
coverage also depends on extraction, signal, contamination and saturation.

The published resolution curve spans 2.4–5.0 µm. Speclib **does not extend it**
to the wider filter limits or for padding. Instead, output starts at 2.410 µm
for F277W and ends at 4.985 µm for F444W. These rounded inward cuts retain
support for the contributing source kernels and their normalization, plus
bracketing input samples. F322W2/F356W require no cuts. The registry's adopted
intervals are therefore narrower than the original filter limits for two modes.
Real source padding inside the calibration domain is still required; supplying
only the output interval, or too coarse a bracketing grid, fails safely.

#### Processing, accuracy and attribution

The [converter](convert_nircam.py) evaluates
the documented coefficients at 2601 points from 2.4 to 5.0 µm in 0.001 µm steps.
CSV serialization uses 17 significant digits. Runtime linearly interpolates R
without extrapolation. A dense comparison with direct polynomial evaluation
has maximum fractional interpolation error below $4\times10^{-8}$.
This represents the polynomial accurately; it does not improve its physical
accuracy. The model's few-percent fit, interpolation error, numerical Gaussian
convolution error and apparent broadening from output-bin integration are
separate quantities. No coefficient is adjusted to compensate for convolution.

The manifest records the coefficients, units, source interpretation, original
publication URL, support, boundary policy, reproduction command/script and
output SHA-256. It is an analytic prescription, so there is no source calibration
FITS file or empirical LSF dataset. All records and the shared CSV load offline.
See the [reproduction procedure](README.md).

Current STScI Figure 3 notes faint second-order spectra with F277W/F322W2;
Greene Section 3.1 discusses F322W2 only. Visibility and overlap depend on source
placement and geometry. The [TSO strategies](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-strategies/nircam-time-series-observation-recommended-strategies)
also describe neighboring-source spectral overlap and orientation constraints.
Gaussian FWHM matching supplies no contamination removal, throughput, exact
pixel sampling, non-Gaussian LSF or finite-input-resolution correction.

The bundled CSV is Speclib's numerical evaluation of the cited factual
coefficients. Acknowledgement of Greene et al., NASA/STScI and the NIRCam team,
and the reviewed [STScI content-use policy](https://www.stsci.edu/copyright)
are recorded in [NOTICE.md](../../src/speclib/data/instruments/NOTICE.md).
No publication text, figure or simulated PSF is redistributed, and no publisher
material is claimed to be public domain or relicensed under Speclib's license.


### MIRI: P750L slitless time-series design approximation

(See the generated `miri_sources` table in the public instrument reference.)

[STScI Filters and Dispersers](https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers) explicitly describes
a linear rise from R approximately 40 at 5 µm to 160 at 10 µm, also giving
R approximately 100 at 7.5 µm in the [LRS mode description](https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-low-resolution-spectroscopy).
[Kendrew et al. (2015), Section 2 / Figure 3](https://doi.org/10.1086/682255)
([author preprint](https://arxiv.org/abs/1512.03000)) shows the linear **as-designed**
curve through 14 µm. Speclib evaluates $R=24(\lambda/\mu\mathrm{m})-80$
on 5–14 µm. Values above 10 µm follow the published nominal design trend,
not additional measured anchors. The approximate inputs supply no quantitative
physical accuracy bound or universal pixel FWHM. Adopting $\lambda/R$
as Gaussian FWHM is a Speclib modeling assumption, not a source LSF measurement.

Kendrew Section 3.3 reports ground monochromator measurements over only 7–8 µm:
R = 95.3 ± 0.6 at 7.5 µm after removing the monochromator linewidth, without
correcting increasing PSF size. This is not an in-flight SLITLESSPRISM uncertainty
or a reason to tune the design curve. [Bouwman et al. (2023), Section 1.3](https://doi.org/10.1088/1538-3873/acbc49) repeats nominal slitless resolution values;
its R=50 spectrophotometric performance uses binned data and supplies no full
measured slitless resolving-power curve. Gaussian convolution validation cannot
establish physical agreement with MIRI's actual LSF.

The [pipeline SPECWCS reference description](https://jwst-pipeline.readthedocs.io/en/stable/jwst/references_general/specwcs_reffile.html)
and inspected slitless FITS files provide wavelength coordinates and detector
pixel geometry, not spectral LSF widths. Differentiating a wavelength solution
alone does not determine resolving power. The [MIRI calibration-status page](https://jwst-docs.stsci.edu/jwst-calibration-status/miri-calibration-status/miri-lrs-calibration-status) records wavelength and photometric updates,
including separate slit/slitless wavelength calibrations and pointing shifts;
these are not resolving-power calibrations. The prism's shared optics do not
justify substituting a fixed-slit dispersion relation: field distortion differs.
No MRS resolution data are used.

#### Coverage, subarrays and reproduction

The [known-issues coverage discussion](https://jwst-docs.stsci.edu/known-issues/miri-known-issues/miri-lrs-known-issues) distinguishes
nominal 5–14 µm coverage, source-dependent blue foldover (sometimes compromising
wavelengths up to 5.2 µm), and weak red signal that can shorten useful coverage
to 12–13 µm. Slitless lacks the fixed-slit filter blocking light below 4.5 µm.
Speclib chooses 5.5–12 µm output, with 5–14 µm model support for convolution.
The blue cut is an additional numerical safety margin, not a universal
contamination threshold. A wider red default would retain real optical coverage
but less reliable practical signal; 12 µm is recommended for this nominal profile.
These cuts neither certify an extraction nor simulate throughput or foldover.

Only conventional SLITLESSPRISM is supported. The [Cycle 6 subarray table](https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-subarrays) lists SLITLESSPRISM (416×72), IP (384×68)
and IPS (256×52) at different detector locations. IP/IPS synchronize readout
with the 390 Hz EMI; IPS sacrifices red wavelength coverage and background.
The [LRS TSO page](https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-time-series-observations/miri-lrs-tsos) supports these observing options,
but separate coverage/LSF equivalence is not established here. IP/IPS therefore
fail lookup instead of inheriting the conventional profile. Pointing, extraction,
detector scattering and subarray changes remain physical limitations.

The [manifest](../../src/speclib/data/instruments/miri_sources.json)
records the equation, approximate published anchors, model-domain justification,
inspected-reference checksums (audit only), coverage decisions and retrieval date
2026-10-22. There is no invented source calibration file for this model.
The [converter](convert_miri.py) reproduces
901 samples spaced by 0.01 µm, verifies the pinned output SHA-256 and never
updates pins. Dense cells keep padding bounds local; linear interpolation
reproduces the adopted equation to floating-point roundoff. All rows are model
evaluations, not measured intermediate values. No calibration coefficient or
convolution algorithm is changed to compensate for numerical broadening.
See the [offline reproduction instructions](README.md).
The CSV and manifest ship in both wheels and sdists.

The CSV is an original numerical evaluation of factual approximate values and
the published linear trend. NASA/STScI, the MIRI team, Kendrew et al. and Bouwman
et al. are acknowledged under the reviewed [STScI content-use policy](https://www.stsci.edu/copyright); no publication text, figure or CRDS file is
redistributed or relicensed. See [NOTICE.md](../../src/speclib/data/instruments/NOTICE.md).
