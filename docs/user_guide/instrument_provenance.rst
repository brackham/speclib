Instrument reference
======================

Profiles combine a resolving-power prescription, detector coverage, and a
Gaussian line-spread approximation. These have different sources and
uncertainties. Use :doc:`instrument_profiles` for examples and sampling
choices, or the :doc:`API reference <../api/instruments>` for parameters.

All profiles adopt :math:`\lambda/R` as Gaussian FWHM, assume negligible
intrinsic source widths, and linearly interpolate tabulated R without
extrapolation. Wavelengths are treated as vacuum. For sources with an
unspecified wavelength convention, the original coordinates are retained
and assumed to be vacuum.
``quality`` classifies the prescription as tabulated, derived, or nominal;
scientific uncertainties depend on the source. Observation-specific LSFs
require additional calibration.

NIRSpec BOTS
--------------

Short identifiers such as ``jwst/nirspec/g395h/f290lp`` select
**BOTS/S1600A1/SUB2048**. Other regular subarrays use
``jwst/nirspec/bots/s1600a1/<subarray>/<disperser>/<filter>``.
PRISM/CLEAR is unavailable with SUB1024A; multistripe subarrays are unsupported.

.. include:: ../_build/instrument_tables/modes.rst

Coverage follows approximate `STScI BOTS Table 1
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps>`_
limits. Semicolons separate disconnected intervals. Missing coverage includes
unread subarray columns as well as physical detector gaps. The physical gaps
are 1.31–1.35 µm for G140H (outside F070LP output), 2.20–2.27 µm for G235H,
and 3.72–3.82 µm for G395H. Actual wavelength limits depend on the observation.

The `subarray reference
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-detectors/nirspec-detector-subarray-mode>`_
distinguishes SUB1024A's outer blue NRS1/red NRS2 portions from SUB1024B's
inner portions. SUB512/SUB512S share spectral coverage but have spatial
heights of 32/16 pixels; SUB512S has no unilluminated spatial pixels in the
trace. These small subarrays are intended especially for PRISM. Use the ETC
to assess partial grating coverage and extraction limits for an observation.

Resolving-power sources
~~~~~~~~~~~~~~~~~~~~~~~~~

The seven official FITS tables are linked from the `STScI dispersers page
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters>`_.
The instrument is described by `Jakobsen et al. (2022)
<https://doi.org/10.1051/0004-6361/202142663>`_.

.. include:: ../_build/instrument_tables/sources.rst

The tables give nominal resolving powers from ground/modeling prescriptions.
Their 2.2-pixel resolution element is approximately
:math:`R=\lambda/(2.2\,\mathrm{DLDS})` for typically fully illuminated
IFU/MOS/200 mas fixed-slit apertures. Adopting it as Gaussian FWHM for
S1600A1 is a modeling approximation. All subarrays share the same disperser
curve, without an illumination or extraction correction.

``speclib`` copies every original wavelength/R pair without fitting or
recomputing R. The snapshot was retrieved on 2026-10-17; source checksums
identify it precisely. G235H/G395H retain 2016 FITS creation timestamps
and include updates from 2024 in their history. No formal
calibration release version is supplied. The manifest retains header history
and conversion details.

STIS, WFC3/IR, and NIRISS/SOSS
--------------------------------

STIS short names select the prime CCD setting, unbinned pixels, and a centered
point source through **52X2**; explicit aliases include
``hst/stis/ccd/52x2/g430l``. WFC3/IR profiles describe first-order unresolved
sources at nominal field center. SOSS defaults to **CLEAR/SUBSTRIP256**, with
aliases such as ``jwst/niriss/soss/substrip256/order1``. Other apertures,
field-specific configurations, and subarrays are unsupported.

.. include:: ../_build/instrument_tables/phase2_modes.rst

.. include:: ../_build/instrument_tables/phase2_sources.rst

STIS
~~~~~~

Widths come from the half-maximum crossings of the `STScI model LSFs
<https://www.stsci.edu/hst/instrumentation/stis/performance/spectral-resolution>`_,
with supplementary `handbook anchors
<https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-6-line-spread-functions/first-order-line-spread-functions>`_
for G750L. Pixel FWHM is linearly interpolated between anchors and held
constant beyond them, including padding. Mean dispersion converts it to
wavelength FWHM. This endpoint extension is a nominal approximation.

.. include:: ../_build/instrument_tables/stis_widths.rst

Differences from rounded handbook widths are below 0.1 pixel. These compare
the two sources; physical accuracy requires independent calibration. The
blue G430L discrepancy remains unexplained in the unversioned files. The LSF-derived widths are retained unchanged. Gaussian
matching omits asymmetric wings, slit losses, and G750L fringing. Slit-filling
sources have broader LSFs than the assumed point source.

WFC3/IR
~~~~~~~~~

`Handbook Table 8.1
<https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview>`_
provides reference R and mean dispersion. The profiles hold the reference
wavelength FWHM constant, so R increases linearly with wavelength.
The pixel width follows from the reference wavelength, R, and dispersion. The handbook defines R from the Gaussian image
core perpendicular to dispersion; source morphology, scans, and field-dependent
dispersion are omitted. The declared padding domain uses the same nominal
constant-FWHM prescription.

NIRISS/SOSS
~~~~~~~~~~~~~

`Albert et al. (2023), Table 2 and Section 5.3
<https://doi.org/10.1088/1538-3873/acd7a3>`_ supply measured CV3 average
dispersions and a two-pixel resolution convention. Each order uses its own
constant average dispersion, giving R proportional to wavelength.
These nominal curves approximate each order with a single dispersion value;
local in-flight wavelength solutions and LSFs require additional calibration.

Order 1 follows the rounded `GR700XD coverage
<https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-gr700xd-grism>`_.
Order 2 is restricted to wavelengths below 1 µm following `STScI calibration
advice <https://jwst-docs.stsci.edu/jwst-calibration-status/niriss-calibration-status/niriss-soss-calibration-status>`_
(June 2026): redder signal is low and photometric calibration uncertain.
The manifest retains the wider geometric range. Orders remain separate;
visit shifts, contamination, and extraction corrections are omitted.

NIRCam grism time series
--------------------------

The short ``jwst/nircam/grismr/<filter>`` names select
**Module A, Grism R, first order**, also identified as
``jwst/nircam/tsgrism/modulea/grismr/<filter>``. Grism C, Module B,
wide-field slitless spectroscopy, and short-wavelength DHS are unsupported.

.. include:: ../_build/instrument_tables/nircam_modes.rst

.. include:: ../_build/instrument_tables/nircam_sources.rst

`Greene et al. (2017), Section 3.2
<https://doi.org/10.1117/1.JATIS.3.3.035001>`_ give the point-source optical
prediction, also reproduced in the `STScI grisms reference
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-grisms>`_:

.. math::

   R(x) \simeq 3.35x^4 - 41.9x^3 + 95.5x^2 + 536x - 240,
   \qquad x=\lambda/(1\,\mu\mathrm{m}).

The fit represents the optical prediction to within a few percent.
All four filters share this predicted point-source curve, without
field-, visit-, or subarray-specific corrections.

The polynomial's domain is **2.4–5.0 µm**. F277W's output starts at 2.410 µm
and F444W's ends at 4.985 µm to leave convolution padding inside that domain.
These ``speclib`` output limits leave room for convolution within the
polynomial domain. The table also lists the official 5% filter-transmission
limits, which F322W2/F356W retain. F444W also has a `TSO detector cutoff
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series>`_
near 5.03 µm, beyond the adopted interval. No throughput is applied.

Faint second orders can occur with F277W/F322W2, and neighboring spectra can
overlap. See the `observing strategies
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-strategies/nircam-time-series-observation-recommended-strategies>`_.
Contamination removal and detector sampling require separate modeling.

MIRI LRS slitless time series
-------------------------------

``jwst/miri/lrs/slitless`` selects **P750L/SLITLESSPRISM**, an isolated point
source at the conventional time-series placement. The canonical identifier
is ``jwst/miri/lrs/slitless/slitlessprism``. Fixed slit, WFSS, MRS, and
SLITLESSPRISM_IP/IPS are unsupported.

.. include:: ../_build/instrument_tables/miri_modes.rst

.. include:: ../_build/instrument_tables/miri_sources.rst

The nominal equation is :math:`R=24(\lambda/\mu\mathrm{m})-80` on 5–14 µm.
It follows `STScI's linear description
<https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers>`_
(R approximately 40, 100, 160 at 5, 7.5, 10 µm) and the design curve in
`Kendrew et al. (2015), Figure 3 <https://doi.org/10.1086/682255>`_.
This design approximation has no quantitative physical accuracy bound.
An in-flight slitless LSF requires a spectral-width calibration in addition
to a wavelength solution.

The **5.5–12 µm** output is narrower than the 5–14 µm model support. The
blue cut leaves convolution padding above the model boundary; the red cut
follows commonly useful coverage as throughput falls. Some observations are
useful to 13–14 µm. `Slitless foldover
<https://jwst-docs.stsci.edu/known-issues/miri-known-issues/miri-lrs-known-issues>`_
below about 4.5 µm can affect wavelengths up to about 5.2 µm for some sources.
Useful coverage depends on the source and extraction. Foldover contamination
requires separate correction.

The `IP/IPS subarrays
<https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-subarrays>`_
have different placements; IPS loses red coverage. Their coverage and LSF
require separate calibration before profiles can be provided.
Pointing, extraction, and detector scattering remain limitations.

.. _g280-calibration-status:

Why WFC3/UVIS G280 is unavailable
-----------------------------------

The `handbook <https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview>`_
quotes R approximately 70 at 300 nm from a cross-dispersion Gaussian image
core. The reviewed `calibration files
<https://www.stsci.edu/hst/instrumentation/wfc3/documentation/grism-resources/wfc3-g280-calibrations>`_
provide traces, wavelength solutions, and sensitivities. A G280 profile also
requires a supported wavelength-dependent spectral FWHM.
No G280 alias or output interval is registered. Further calibration work is
tracked in `Issue #99 <https://github.com/brackham/speclib/issues/99>`_.

Provenance records
--------------------

Inspect each profile's record offline:

.. code-block:: python

   from speclib import get_profile

   profile = get_profile("jwst/nirspec/g395h/f290lp")
   print(profile.resolution_definition, profile.coverage_source_url)
   print(profile.provenance)

The returned dictionary contains source URLs, identifiers, checksums,
processing details, and assumptions. Analytic prescriptions record the
equations and coefficients used to generate their curves. Custom profiles
without a provenance reference return an empty dictionary.

Download the packaged records for reproducibility:

* :download:`Profile registry <../../src/speclib/data/instruments/profiles.toml>`:
  configurations, coverage, and assumptions used to generate the tables above.
* :download:`NIRSpec manifest <../../src/speclib/data/instruments/sources.json>`.
* :download:`STIS, WFC3, and SOSS manifest <../../src/speclib/data/instruments/phase2_sources.json>`.
* :download:`NIRCam manifest <../../src/speclib/data/instruments/nircam_sources.json>`.
* :download:`MIRI manifest <../../src/speclib/data/instruments/miri_sources.json>`.

Manifests identify original inputs and derived outputs with checksums, and
record nominal padding domains. The registry, manifests, and curves ship
with the package; no runtime CRDS, Pandeia, or JWST pipeline is needed.
Conversion scripts and detailed calibration investigations are in the
`repository's maintainer notes
<https://github.com/brackham/speclib/tree/main/scripts/instruments>`_.

Attribution and redistribution review are recorded in
:download:`NOTICE.md <../../src/speclib/data/instruments/NOTICE.md>` under the
`STScI content-use policy <https://www.stsci.edu/copyright>`_ and cited source
terms. The code license does not relicense source data or publication material.
