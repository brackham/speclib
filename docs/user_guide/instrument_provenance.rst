Scientific provenance of instrument profiles
============================================

Three ingredients have distinct sources: the numerical resolving-power curve,
the valid detector coverage, and Speclib's approximation to the LSF. Practical
examples are in :doc:`instrument_profiles`; contributor requirements are in
:doc:`../development/instrument_profiles`.

Original resolving-power data
-----------------------------

The `STScI NIRSpec dispersers page
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters>`_
distributes seven dispersion/resolution FITS tables used by the ETC. Its
instrument reference is `Jakobsen et al. (2022), A&A 661, A80
<https://doi.org/10.1051/0004-6361/202142663>`_
(`author preprint <https://arxiv.org/abs/2202.03305>`_). Speclib copies the linked
tables, not values digitized or inferred from that publication.

.. include:: ../_build/instrument_tables/sources.rst

NASA/STScI is the distributor. All seven headers identify ``SYSTEM=MODELING``
and ``PEDIGREE=GROUND``. G235H/G395H name Pierre Ferruit as author; the other
headers name a generation script, not a complete calibration-team author list.
These are nominal dispersion-derived prescriptions, not empirical in-flight
or BOTS point-source LSF measurements. ``quality='tabulated'`` identifies the
representation and does not imply empirical accuracy.

The source table contains ``WAVELENGTH`` (micrometers), ``DLDS`` (micrometers
per pixel) and dimensionless ``R`` in binary-table extension 1. STScI defines
a 2.2-pixel resolution element, approximately
:math:`R=\lambda/(2.2\,\mathrm{DLDS})`, for typically fully illuminated
IFU/MOS/200 mas fixed-slit apertures. This is not a measured Gaussian FWHM for
the wide S1600A1 slit. The FITS wavelength unit does not explicitly flag air
versus vacuum; Speclib treats the coordinates as vacuum without conversion.

Identification and processing
-----------------------------

``calibration_version`` labels the selected Speclib source snapshot, retrieved
2026-10-17. There is no stated formal calibration release version for these
attachments. Full SHA-256 pins identify their bytes. FITS creation timestamps
above are not last-revision dates: G235H/G395H retain 2016 timestamps, but their
histories record interpolation by G. M. Wahlgren on 2024-05-16, and the linked
STScI page announces their update on 2024-06-24. Attachment version ``1`` is not
a calibration version.

The :download:`conversion script <../../scripts/instruments/convert_nirspec.py>`
checks each original checksum, timestamp, row count, units, ordering and numerical
validity, then copies every ``WAVELENGTH``/``R`` pair. Source float32 values are
promoted to float64 and written with 17 significant digits to
``wavelength_um,resolving_power`` CSVs. Conversion does not fit, interpolate,
thin, or recompute R. Runtime evaluation linearly interpolates the published R
column in wavelength and never extrapolates it.

The original G235H/G395H R and dispersion columns are separately interpolated;
their maximum deviations from the 2.2-pixel ratio are about 0.403% and 0.836%.
The converter's 1% consistency check accommodates this source property; it is
not a physical-accuracy claim. Published R values remain untouched.

Some original intermediate sources cannot be independently retrieved: five
``LITREF`` values name internal as-built files; G235H/G395H name the technical
note ``ESA-JWST-SCI-NRS-TN-2016-024``, for which no public copy was located.
These are header identifiers, not unverifiable citations substituted for the
data. The directly linked official FITS tables are the reproducible inputs.
If attachment URLs move, locate the original filenames on the linked STScI
page and require the same checksums before accepting replacements.

Coverage and Gaussian approximation
-----------------------------------

The interval limits and gaps come separately from `BOTS wavelength ranges and
gaps, Table 1
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps>`_.
Speclib transcribes these approximate S1600A1 limits in ``profiles.toml``.
They are not derived from the R tables, and they vary with the observation's
wavelength solution and grating-wheel position. The regular SUB2048, SUB1024A,
SUB1024B, SUB512 and SUB512S columns were checked on 2026-10-22; the complete
registry-generated matrix is in :doc:`instrument_profiles`. PRISM/CLEAR is
unavailable for SUB1024A. Multistripe configurations are not implemented.

The `detector subarray reference, Tables 1–2
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-detectors/nirspec-detector-subarray-mode>`_
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

Speclib adopts the tabulated :math:`\lambda/R` as **Gaussian FWHM** using the
existing source-centered log-wavelength convolution. This is a Speclib modeling
choice, not a claim that the source supplies Gaussian LSF measurements. It
assumes negligible intrinsic model widths and applies no deconvolution.
There is no full non-Gaussian LSF, throughput, detector response/pixel geometry,
noise, systematics, or observation-dependent resolution/wavelength solution.
Output sampling is separate: synphot's bin integration and sparse samples can
change apparent FWHMs. See :doc:`instrument_profiles` for sampling guidance.

Programmatic records and redistribution
---------------------------------------

.. code-block:: python

   from speclib import get_profile

   profile = get_profile("jwst/nirspec/g395h/f290lp")
   record = profile.provenance  # Fresh dictionary; no network request.
   print(record["url"], record["filename"], record["sha256"])
   print(record["fits_date"], record["history"], record["processing"])
   print(profile.coverage_source_url, profile.resolution_definition)
   print(record["coverage_retrieved"], record["subarray_source_url"])
   print(record["subarrays"]["SUB512S"], record["nominal_detector_gap_um"])

The :download:`packaged manifest <../../src/speclib/data/instruments/sources.json>`
is authoritative for original/output filenames and checksums, source headers,
URLs and conversion details. ``profiles.toml`` is authoritative for profile
configuration, coverage and adopted assumptions. Published tables are generated
from these records. Both files and the curves ship in wheels and sdists.

The `STScI content-use policy <https://www.stsci.edu/copyright>`_ permits use
under its public-domain-like NASA/STScI terms unless otherwise stated, requests
acknowledgement, and identifies exceptions for third-party restrictions. Review
of the source page and all seven headers found no contrary notice. This is the
redistribution basis, not a dataset-specific license or endorsement; Speclib's
code license does not relicense the source data. Attribution and the review date
are in :download:`NOTICE.md <../../src/speclib/data/instruments/NOTICE.md>`.

STIS, WFC3 and NIRISS prescriptions
-----------------------------------

The :download:`Phase 2 manifest <../../src/speclib/data/instruments/phase2_sources.json>`
records the original sources, equations, coefficients, calibration domains and
output checksums. These profiles use the same offline representation and
Gaussian engine as NIRSpec. Their configuration and practical coverage are in
``profiles.toml``; the tables below are generated from those records.

.. include:: ../_build/instrument_tables/phase2_sources.rst

STIS: model LSFs reduced to Gaussian widths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `STScI spectral-resolution page
<https://www.stsci.edu/hst/instrumentation/stis/performance/spectral-resolution>`_
provides model LSF text files, with relative pixel offset and four aperture
columns. The converter selects ``52x2.0`` and finds the half-maximum crossings
on each side of the global peak by linear interpolation. It preserves the source
amplitudes while measuring the width; no normalization, Gaussian fit or
second-moment width is substituted. These are **model point-source LSFs**, not
measurements of a particular observation. The files have no formal calibration
version; exact SHA-256 pins identify the retrieved source bytes.
The `STIS handbook's LSF introduction
<https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-6-line-spread-functions>`_
states that internal-lamp and on-orbit observations confirmed and revised the
original pre-flight models. The public text files do not identify which
individual revisions they contain.

.. include:: ../_build/instrument_tables/stis_widths.rst

The derived widths differ from the rounded `handbook Table 13.41
<https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-6-line-spread-functions/first-order-line-spread-functions>`_
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
``quality='derived'`` because Speclib reduces LSFs and interpolates their widths.

WFC3: reference resolving power with constant wavelength FWHM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`WFC3 Instrument Handbook Table 8.1
<https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview>`_
supplies first-order coverage, mean dispersions and resolving powers at reference
wavelengths. Speclib holds the reference wavelength FWHM constant, so R increases
linearly with wavelength. It does not assume a two-pixel FWHM: the effective
pixel width follows from the reference wavelength, R and dispersion. The
handbook defines R from the Gaussian image core measured perpendicular to
dispersion. The prescription assumes an unresolved source at nominal field
center and ignores morphology, scan geometry and field-dependent dispersion.
It is ``quality='nominal'``, not a measured wavelength-dependent calibration.
The two endpoint samples reproduce the linear R equation exactly, including
its declared padding domain.

NIRISS/SOSS: independent order-specific nominal curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Albert et al. (2023), PASP 135, 075001
<https://doi.org/10.1088/1538-3873/acd7a3>`_
(`author preprint <https://arxiv.org/abs/2306.04572>`__), Table 2 and Section 5.3,
supply separate CV3 average dispersions and the two-pixel Nyquist resolution
convention. Speclib uses the measured dispersions, rather than the parenthesized
optical-model values. Each order's R increases with wavelength under this
constant-average-dispersion approximation. This reproduces the paper's nominal
first-order check near its reference wavelength, but does not reconstruct the
full derivative of an in-flight wavelength solution. Individual LED LSF widths
in Figure 6 vary around the adopted two-pixel convention. The curves are
``quality='nominal'``; no publication figure is digitized or redistributed.

The `GR700XD documentation
<https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-gr700xd-grism>`_
provides the rounded first-order coverage and the CLEAR/SUBSTRIP256 setup.
For order 2, the separate `calibration-status page
<https://jwst-docs.stsci.edu/jwst-calibration-status/niriss-calibration-status/niriss-soss-calibration-status>`_
(updated June 2026) advises against using wavelengths above 1 µm because the
signal is low and photometric calibration is uncertain. Speclib therefore uses
a shorter practical interval than the geometric range. The manifest retains
both the paper's geometric limits and this coverage decision. This review
informs useful coverage; it is not an in-flight resolution calibration. Visit
wavelength shifts, order contamination and extraction corrections are outside
this approximation. The two orders remain independent ordinary spectra.

Reproduction and attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :download:`Phase 2 converter <../../scripts/instruments/convert_phase2.py>`
reproduces all six outputs offline using checksum-pinned STIS inputs and the
manifest's numerical prescriptions. It rejects changed source and output bytes;
it never rewrites pins. See the :download:`reproduction procedure
<../../scripts/instruments/README.md>`. Analytic sources require no fictitious
input filename or checksum; their equations and coefficients are the inputs.
All derived CSVs have output checksums. Domains include explicit nominal
padding, with no runtime extrapolation. Wavelength labels do not explicitly
identify air versus vacuum; Speclib treats these approximate coordinates as
vacuum without conversion.

The STIS source headers and STScI pages were reviewed against the `STScI
content-use policy <https://www.stsci.edu/copyright>`_. No contrary restriction
was found on the selected numerical LSFs. WFC3 and SOSS CSVs are original
evaluations of cited numerical prescriptions; publication text and figures are
not bundled. NASA/STScI and the SOSS instrument team are acknowledged in
:download:`NOTICE.md <../../src/speclib/data/instruments/NOTICE.md>`. No NASA,
STScI or author endorsement is implied.


NIRCam: Module A Grism R time-series prediction
-----------------------------------------------

.. include:: ../_build/instrument_tables/nircam_sources.rst

`Greene et al. (2017), Section 3.2 / Figure 2
<https://doi.org/10.1117/1.JATIS.3.3.035001>`_ give the polynomial reproduced in
:doc:`instrument_profiles`. It fits predicted point-source FWHM resolving power
to within a few percent. The model uses a simulated monochromatic PSF, scaled
by 1.1 along dispersion for the pupil beam factor, then sampled in LW pixels
with nominal dispersion 10 Å/pixel. Pixel sampling limits the short-wavelength
resolution; diffraction matters increasingly at longer wavelengths. This is
not a measured in-flight LSF or a constant R=1600 prescription.

The `STScI grisms reference
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-grisms>`_,
Figure 1, repeats the coefficients. Table 3 gives R/C dispersions of
1.004/1.006 nm/pixel with field variations up to 2%. The paper reports all four
grism dispersions within 1% of their nominal design. Neither source gives
separate R/C or module polynomials; identical visit-specific LSFs are not
established. Speclib supports only the actual LW time-series configuration:
**Module A Grism R, first order**, for an isolated point source. Its four filters
share one nominal curve; there is no field-, visit- or subarray-specific PSF
correction. The different standard placements for LW+SW imaging and LW+SW
spectroscopy do not have separately calibrated profiles here.

Coverage and conservative edge cuts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Table 4 supplies the four filters' 5% transmission limits, recorded separately
in the :download:`NIRCam manifest <../../src/speclib/data/instruments/nircam_sources.json>`.
The `grism TSO page
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series>`_
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

Processing, accuracy and attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :download:`converter <../../scripts/instruments/convert_nircam.py>` evaluates
the documented coefficients at 2601 points from 2.4 to 5.0 µm in 0.001 µm steps.
CSV serialization uses 17 significant digits. Runtime linearly interpolates R
without extrapolation. A dense comparison with direct polynomial evaluation
has maximum fractional interpolation error below :math:`4\times10^{-8}`.
This represents the polynomial accurately; it does not improve its physical
accuracy. The model's few-percent fit, interpolation error, numerical Gaussian
convolution error and apparent broadening from output-bin integration are
separate quantities. No coefficient is adjusted to compensate for convolution.

The manifest records the coefficients, units, source interpretation, original
publication URL, support, boundary policy, reproduction command/script and
output SHA-256. It is an analytic prescription, so there is no source calibration
FITS file or empirical LSF dataset. All records and the shared CSV load offline.
See the :download:`reproduction procedure <../../scripts/instruments/README.md>`.

Current STScI Figure 3 notes faint second-order spectra with F277W/F322W2;
Greene Section 3.1 discusses F322W2 only. Visibility and overlap depend on source
placement and geometry. The `TSO strategies
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-strategies/nircam-time-series-observation-recommended-strategies>`_
also describe neighboring-source spectral overlap and orientation constraints.
Gaussian FWHM matching supplies no contamination removal, throughput, exact
pixel sampling, non-Gaussian LSF or finite-input-resolution correction.

The bundled CSV is Speclib's numerical evaluation of the cited factual
coefficients. Acknowledgement of Greene et al., NASA/STScI and the NIRCam team,
and the reviewed `STScI content-use policy <https://www.stsci.edu/copyright>`_
are recorded in :download:`NOTICE.md <../../src/speclib/data/instruments/NOTICE.md>`.
No publication text, figure or simulated PSF is redistributed, and no publisher
material is claimed to be public domain or relicensed under Speclib's license.


MIRI: P750L slitless time-series design approximation
-----------------------------------------------------

.. include:: ../_build/instrument_tables/miri_sources.rst

`STScI Filters and Dispersers <https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers>`_ explicitly describes
a linear rise from R approximately 40 at 5 µm to 160 at 10 µm, also giving
R approximately 100 at 7.5 µm in the `LRS mode description <https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-low-resolution-spectroscopy>`_.
`Kendrew et al. (2015), Section 2 / Figure 3 <https://doi.org/10.1086/682255>`_
(`author preprint <https://arxiv.org/abs/1512.03000>`__) shows the linear **as-designed**
curve through 14 µm. Speclib evaluates :math:`R=24(\lambda/\mu\mathrm{m})-80`
on 5–14 µm. Values above 10 µm follow the published nominal design trend,
not additional measured anchors. The approximate inputs supply no quantitative
physical accuracy bound or universal pixel FWHM. Adopting :math:`\lambda/R`
as Gaussian FWHM is a Speclib modeling assumption, not a source LSF measurement.

Kendrew Section 3.3 reports ground monochromator measurements over only 7–8 µm:
R = 95.3 ± 0.6 at 7.5 µm after removing the monochromator linewidth, without
correcting increasing PSF size. This is not an in-flight SLITLESSPRISM uncertainty
or a reason to tune the design curve. `Bouwman et al. (2023), Section 1.3
<https://doi.org/10.1088/1538-3873/acbc49>`_ repeats nominal slitless resolution values;
its R=50 spectrophotometric performance uses binned data and supplies no full
measured slitless resolving-power curve. Gaussian convolution validation cannot
establish physical agreement with MIRI's actual LSF.

The `pipeline SPECWCS reference description <https://jwst-pipeline.readthedocs.io/en/stable/jwst/references_general/specwcs_reffile.html>`_
and inspected slitless FITS files provide wavelength coordinates and detector
pixel geometry, not spectral LSF widths. Differentiating a wavelength solution
alone does not determine resolving power. The `MIRI calibration-status page
<https://jwst-docs.stsci.edu/jwst-calibration-status/miri-calibration-status/miri-lrs-calibration-status>`_ records wavelength and photometric updates,
including separate slit/slitless wavelength calibrations and pointing shifts;
these are not resolving-power calibrations. The prism's shared optics do not
justify substituting a fixed-slit dispersion relation: field distortion differs.
No MRS resolution data are used.

Coverage, subarrays and reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `known-issues coverage discussion <https://jwst-docs.stsci.edu/known-issues/miri-known-issues/miri-lrs-known-issues>`_ distinguishes
nominal 5–14 µm coverage, source-dependent blue foldover (sometimes compromising
wavelengths up to 5.2 µm), and weak red signal that can shorten useful coverage
to 12–13 µm. Slitless lacks the fixed-slit filter blocking light below 4.5 µm.
Speclib chooses 5.5–12 µm output, with 5–14 µm model support for convolution.
The blue cut is an additional numerical safety margin, not a universal
contamination threshold. A wider red default would retain real optical coverage
but less reliable practical signal; 12 µm is recommended for this nominal profile.
These cuts neither certify an extraction nor simulate throughput or foldover.

Only conventional SLITLESSPRISM is supported. The `Cycle 6 subarray table
<https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-subarrays>`_ lists SLITLESSPRISM (416×72), IP (384×68)
and IPS (256×52) at different detector locations. IP/IPS synchronize readout
with the 390 Hz EMI; IPS sacrifices red wavelength coverage and background.
The `LRS TSO page <https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-time-series-observations/miri-lrs-tsos>`_ supports these observing options,
but separate coverage/LSF equivalence is not established here. IP/IPS therefore
fail lookup instead of inheriting the conventional profile. Pointing, extraction,
detector scattering and subarray changes remain physical limitations.

The :download:`manifest <../../src/speclib/data/instruments/miri_sources.json>`
records the equation, approximate published anchors, model-domain justification,
inspected-reference checksums (audit only), coverage decisions and retrieval date
2026-10-22. There is no invented source calibration file for this model.
The :download:`converter <../../scripts/instruments/convert_miri.py>` reproduces
901 samples spaced by 0.01 µm, verifies the pinned output SHA-256 and never
updates pins. Dense cells keep padding bounds local; linear interpolation
reproduces the adopted equation to floating-point roundoff. All rows are model
evaluations, not measured intermediate values. No calibration coefficient or
convolution algorithm is changed to compensate for numerical broadening.
See the :download:`offline reproduction instructions <../../scripts/instruments/README.md>`.
The CSV and manifest ship in both wheels and sdists.

The CSV is an original numerical evaluation of factual approximate values and
the published linear trend. NASA/STScI, the MIRI team, Kendrew et al. and Bouwman
et al. are acknowledged under the reviewed `STScI content-use policy
<https://www.stsci.edu/copyright>`_; no publication text, figure or CRDS file is
redistributed or relicensed. See :download:`NOTICE.md <../../src/speclib/data/instruments/NOTICE.md>`.
