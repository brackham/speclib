Instrument profiles
===================

``observed_by()`` matches a model to an instrument's nominal wavelength-dependent
Gaussian resolving power and valid wavelength coverage. It is **resolution
matching, not a full instrument simulation**. Profiles and calibration curves
are bundled and available offline.

Usage and profile information
-----------------------------

.. code-block:: python

   import astropy.units as u
   import numpy as np
   from speclib import Spectrum, get_profile, list_instrument_profiles

   # A synthetic source with sufficient sampling and real convolution padding.
   wave = np.geomspace(0.5, 6.0, 30000) * u.um
   unit = u.erg / (u.s * u.cm**2 * u.AA)
   source = Spectrum(spectral_axis=wave, flux=np.ones(wave.size) * unit)

   profile = get_profile("JWST/NIRSpec/PRISM/CLEAR")  # Case-insensitive.
   observed = source.observed_by(profile)  # An ordinary Spectrum.
   print(observed.valid_intervals)
   print(profile.configuration, profile.calibration_version, profile.assumptions)
   print(profile.provenance["url"])        # Original STScI FITS file.
   print(profile.provenance["sha256"])     # Exact source bytes.
   print(profile.coverage_source_url)     # Separate coverage reference.
   print(list_instrument_profiles())

   sampled = source.observed_by(profile, samples_per_resolution_element=10)
   centers = np.linspace(0.65, 5.2, 300) * u.um
   resampled = source.observed_by(profile, wavelength=centers)

The two sampling options are mutually exclusive. Without either, original samples
within valid coverage are retained. Explicit centers must be strictly monotonic,
with at least two centers per requested interval; both centers and their implicit
bin edges must lie in valid coverage. Ascending and descending axes are supported.

The same method on :class:`~speclib.SpectralGrid` returns an ordinary independent
grid and reuses each convolution plan across its spectra. Both interfaces preserve
units and copied metadata; ``meta['instrument_profile']`` identifies the calibration.
``grid.get_flux()`` returns the interpolated flux Quantity;
``grid.get_spectrum()`` returns an independent ``Spectrum`` retaining instrument
provenance and coverage. For example, after matching a suitable high-resolution
grid, use ``observed_grid.get_spectrum(5750, 4.5, 0.0)``. See :doc:`spectral_grids`.

Detector gaps and supported operations
--------------------------------------

A detector gap is missing coverage in one spectrum, not an independent spectral
order. Arrays contain only valid samples, without artificial gap samples, NaNs,
masks or filler fluxes. ``valid_intervals`` is a defensive-copy wavelength quantity
of shape ``(n, 2)``, stored in ``meta['valid_intervals']``. It returns ``None`` for
unrestricted spectra; gaps are never inferred from nonuniform sampling.

Plot each valid interval separately so lines do not connect across detector gaps:

.. code-block:: python

   import matplotlib.pyplot as plt

   observed = source.observed_by("jwst/nirspec/g395h/f290lp")
   for lower, upper in observed.valid_intervals:
       selected = ((observed.spectral_axis >= lower)
                   & (observed.spectral_axis <= upper))
       plt.plot(observed.spectral_axis[selected].to_value(u.um),
                observed.flux[selected].value)

* ``resample()`` accepts one monotonic array spanning several intervals and
  integrates each independently. It rejects gap centers, gap-crossing implicit
  bins, and bins beyond available source samples, even with ``taper=True``.
  At least two requested centers per interval are needed to define bin edges.
* ``bin()`` accepts explicit bins wholly within valid, sampled coverage, with at
  least two source samples per bin. Gap-straddling bins fail even when their
  centers are valid. Binned results retain the union of their explicit bin extents.
* ``select_wavelength()`` clips the coverage and drops unrepresented intervals.
* Broadening, repeated instrument matching, and ``regularize()`` reject known
  discontinuous inputs. Use the continuous underlying model or select one valid
  interval first. Continuous broadening and Python ``copy.deepcopy()`` retain
  coverage. Coverage-aware resampling/binning reject masks and uncertainties.

Detector gaps are imposed **after** convolving the continuous underlying source;
light inside a gap can contribute wings on a detector. Missing source flux is
never reconstructed by interpolating across an explicit input gap.

Inherited specutils arithmetic, slicing, integration and serialization do not
understand these coverage limits. Prefer Speclib's ``select_wavelength()`` for
cropping; upstream slices can retain metadata broader than their samples.
Supported Speclib resampling/binning check actual sample bounds as well. Do not
assume upstream operations reconcile coverage or retain it after replacing
metadata. Manually removing the coverage declaration removes its safeguards.

Independent spectral orders may overlap and need explicit weighting/stitching.
That functionality is separate; no order collection or stitching is added here.
NIRISS/SOSS order identifiers each produce an ordinary independent spectrum.

Resolving power versus sampling
-------------------------------

Resolving power :math:`R=\lambda/\mathrm{FWHM}` describes the nominal line-spread
function. Sampling specifies output-coordinate spacing. Automatic spacing is
approximately :math:`\delta\lambda=\lambda/(N R(\lambda))`; these are **not
actual detector pixels**.

Both sampling options use the existing :meth:`~speclib.Spectrum.resample`
machinery: synphot integrates flux into implicit bins inferred from the centers.
Bin integration broadens narrow lines, and sparse half-height samples can bias
a measured FWHM. A coarsely sampled output therefore need not reproduce the
nominal width when measured directly from its returned values.

Representative synthetic-line maximum apparent R errors were 6.80%, 2.79%, 0.43%
and 0.36% at 3, 5, 10 and 20 centers per FWHM. These are examples, not universal
bounds; line placement relative to bins matters. Preserve fine native sampling
(the default), request more output samples, or fit a bin-integrated line model
for precise width measurements. Use :meth:`~speclib.Spectrum.bin` for explicit
observational bin widths. Calibration curves are never adjusted for sampling.

Supported NIRSpec modes
-----------------------

Identifiers are case-insensitive. The nine short identifiers, such as
``jwst/nirspec/g395h/f290lp``, continue to select **BOTS/S1600A1/SUB2048**.
Their explicit SUB2048 aliases remain available. The 35 additional canonical
identifiers use ``jwst/nirspec/bots/s1600a1/<subarray>/<disperser>/<filter>``:

.. code-block:: python

   observed = spectrum.observed_by(
       "jwst/nirspec/bots/s1600a1/sub1024a/g395h/f290lp"
   )

Use ``sub1024a``, ``sub1024b``, ``sub512`` or ``sub512s`` with a supported pair
below. PRISM/CLEAR with SUB1024A is unavailable and raises ``ValueError``.
Multistripe subarrays are not bundled.

.. include:: ../_build/instrument_tables/modes.rst

The table is generated from the registry using approximate
`STScI BOTS Table 1
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps>`_
coverage. Semicolons separate disconnected intervals; missing regions include
unread subarray pixels as well as physical detector gaps. SUB1024A samples the
blue part on NRS1 and red part on NRS2; SUB1024B samples their inner parts.
For medium gratings and PRISM the spectrum falls on NRS1.

SUB512 and SUB512S are distinct setups with identical nominal wavelength
intervals: their spatial heights are 32 and 16 pixels, respectively. These
smaller subarrays are intended especially for PRISM; STScI also describes
partial grating spectra of very bright sources. Tabulated coverage does not
imply that every configuration is recommended or optimal. All
subarrays reuse the same disperser's nominal R curve, without a spatial
extraction or slit-illumination correction. Native sampling remains the default;
convolution uses the continuous padded source before selecting valid intervals.
Useful detector/extraction bounds vary by observation; consult the ETC for
precise limits. See :doc:`instrument_provenance` for physical gaps, geometry
sources and the resolving-power assumptions.

Supported STIS, WFC3 and NIRISS modes
-------------------------------------

STIS shorthand identifiers select the prime CCD setting, a centered point source
and the **52X2 aperture**. Explicit aliases have the form
``hst/stis/ccd/52x2/g430l``. WFC3 profiles describe first-order unresolved sources
at nominal field center. SOSS shorthand identifiers select **CLEAR/SUBSTRIP256**;
explicit aliases include ``jwst/niriss/soss/substrip256/order1``. Unsupported
apertures, field-specific configurations and subarrays fail clearly.

.. include:: ../_build/instrument_tables/phase2_modes.rst

STIS uses FWHMs derived from model LSF tables, with supplementary handbook
anchors and explicit endpoint-width approximations. WFC3 and SOSS use nominal
analytic prescriptions with wavelength-dependent R. See
:doc:`instrument_provenance` for equations, definitions, direct data links,
padding assumptions and redistribution terms.

SOSS order 2 uses practical coverage restricted according to current STScI
calibration advice. Its geometric extent is wider; the excluded red region has
low signal and uncertain photometric calibration. Each order is processed
independently; their overlapping wavelength ranges do not imply stitching.

.. code-block:: python

   # The source must cover each profile plus real convolution padding.
   optical_wave = np.geomspace(0.25, 3.0, 40000) * u.um
   optical_source = Spectrum(spectral_axis=optical_wave,
                             flux=np.ones(optical_wave.size) * unit)
   stis = optical_source.observed_by("hst/stis/g430l")
   wfc3 = optical_source.observed_by("hst/wfc3/g141")
   order1 = optical_source.observed_by("jwst/niriss/soss/order1")
   order2 = optical_source.observed_by("jwst/niriss/soss/order2")

Supported NIRCam long-wavelength time series
--------------------------------------------

NIRCam profiles select **Module A, Grism R, first order**, using the canonical
identifier ``jwst/nircam/tsgrism/modulea/grismr/<filter>``. The shorter
``jwst/nircam/grismr/<filter>`` is an alias for that configuration. Grism C,
Module B, wide-field slitless spectroscopy and short-wavelength DHS profiles
are not supported.

.. include:: ../_build/instrument_tables/nircam_modes.rst

The shared, wavelength-dependent resolving power follows the point-source
optical prediction of `Greene et al. (2017), Section 3.2
<https://doi.org/10.1117/1.JATIS.3.3.035001>`_, also reproduced by
`STScI <https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-grisms>`_:

.. math::

   R(x) \simeq 3.35x^4 - 41.9x^3 + 95.5x^2 + 536x - 240,
   \qquad x = \lambda/(1\,\mu\mathrm{m}).

This is a **nominal prediction**, not a measured in-flight LSF. Speclib adopts
:math:`\lambda/R` as Gaussian FWHM. The polynomial is used only within its
published **2.4–5.0 µm** model domain. To retain safe convolution padding,
Speclib trims F277W's blue edge to **2.410 µm** and F444W's red edge to
**4.985 µm**. These are conservative Speclib output cuts, not official filter
edges. F322W2 and F356W retain the 5% filter-transmission limits.

The full filter limits are 2.395–3.179, 2.413–4.083, 3.100–4.041 and
3.835–5.084 µm, respectively. The `TSO documentation
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series>`_
also identifies a detector cutoff near 5.03 µm for F444W at the standard target
placement, outside Speclib's adopted interval. None of these limits promises
useful signal or an exact visit wavelength solution. See
:doc:`instrument_provenance` for the source and padding assumptions.

.. code-block:: python

   nircam = source.observed_by("jwst/nircam/grismr/f322w2")
   nircam_sampled = source.observed_by(
       "jwst/nircam/grismr/f444w", samples_per_resolution_element=5
   )
   # Identical interface for an existing SpectralGrid:
   # nircam_grid = grid.observed_by("jwst/nircam/grismr/f356w")

The source must extend beyond the nominal output interval for convolution.
Native sampling remains the default; explicit wavelengths and automatic spacing
use the same bin integration described above. The profiles do not multiply by
filter transmission. Faint second-order spectra can occur with F277W/F322W2,
and neighboring sources can contaminate slitless spectra. No order removal,
field-dependent PSF, exact detector sampling or extraction model is applied.
See the `TSO recommended strategies
<https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-strategies/nircam-time-series-observation-recommended-strategies>`_
and the :doc:`scientific reference <instrument_provenance>`.

Supported MIRI LRS slitless time series
---------------------------------------

``jwst/miri/lrs/slitless`` selects **P750L / SLITLESSPRISM**, an isolated
point source at the conventional time-series placement. The canonical identifier
includes the subarray. Fixed slit, MIRI WFSS, MRS and the Cycle 6
SLITLESSPRISM_IP/IPS subarrays are unsupported, with no implicit fallback.
The `new subarrays <https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-subarrays>`_ change placement and readout;
IPS also loses red coverage. Identical wavelength solutions/LSFs are not assumed.

.. include:: ../_build/instrument_tables/miri_modes.rst

The **nominal** prescription is :math:`R = 24\,(\lambda/\mu\mathrm{m})-80`,
giving approximately 40, 100 and 160 at 5, 7.5 and 10 µm. It follows
`STScI's explicit linear description <https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers>`_ and the
`Kendrew et al. (2015) design curve <https://arxiv.org/abs/1512.03000>`_ through 14 µm.
It is not a measured in-flight slitless Gaussian LSF. Speclib adopts
:math:`\lambda/R` as Gaussian FWHM; no universal pixel-width convention is inferred.

Output is conservatively **5.5–12 µm**, with model support **5–14 µm**.
The 5.5 µm cut leaves real convolution support above the 5 µm model boundary;
it is a Speclib numerical cut, not a measured optical edge. The 12 µm cut
follows commonly used coverage as prism throughput falls toward the nominal
14 µm red edge. Some spectra remain useful to 13–14 µm, but this default
prioritizes conservative matching. Useful limits depend on source and extraction.
`Slitless spectral foldover <https://jwst-docs.stsci.edu/known-issues/miri-known-issues/miri-lrs-known-issues>`_ below about 4.5 µm
has no fixed-slit blocking filter and can compromise the blue spectrum,
moving the useful limit toward 5.2 µm for some spectral shapes. Matching does
not correct contamination or guarantee useful observed data at any limit.

.. code-block:: python

   mid_ir_wave = np.linspace(5.0, 14.0, 45001) * u.um
   mid_ir = Spectrum(spectral_axis=mid_ir_wave,
                     flux=np.ones(mid_ir_wave.size) * unit)
   miri = mid_ir.observed_by("jwst/miri/lrs/slitless")
   miri_sampled = mid_ir.observed_by("jwst/miri/lrs/slitless",
                                     samples_per_resolution_element=10)
   # For a suitably covered grid:
   # miri_grid = grid.observed_by("jwst/miri/lrs/slitless")

Input must include real flux outside the output interval for broad low-R kernels.
A well-sampled 5–14 µm source is sufficient; input starting at 5.5 µm is not.
Default sampling retains input wavelengths without additional output-bin
integration. Coarse output bins can alter apparent widths and depths;
resolution matching and observational binning remain distinct. See
:doc:`instrument_provenance` for definitions, evidence and reproduction.

Calibration and limitations
---------------------------

The :doc:`scientific provenance reference <instrument_provenance>` separates
the original STScI dispersion-derived R tables, approximate BOTS detector
coverage, and Speclib's Gaussian LSF assumption. The **2.2-pixel source resolution
element is adopted as Gaussian FWHM**; it is not an empirical BOTS point-source
LSF. ``quality='tabulated'`` describes the source representation, not physical
precision. No observation-dependent illumination or resolution correction is made.

Convolution uses :doc:`resolution`, including linear interpolation of R and
source-centered log-wavelength Gaussians. Each requested interval needs real
model and calibration padding, with bracketing samples. Padding follows local
kernel support, including contributing source light outside detector limits;
flux and R are never extrapolated. A sufficiently sampled model extending to
5.5 micrometers supports full PRISM red coverage; loading only 0.6–5.3 micrometers
is insufficient. Each selected instrument interval requires its full padding.

Input sampling must meet the generic two-samples-per-FWHM minimum, but sampling
does not establish physical source resolution. Intrinsic widths must be negligible:
no deconvolution or finite-input-resolution correction is performed. Numeric
``meta['native_resolving_power']`` is checked when supplied; descriptive metadata
produces a warning. Unknown native resolution is the caller's responsibility.
Masks, uncertainties, nonfinite or multidimensional flux are unsupported for
instrument matching. There is no throughput, noise, detector response, slit-loss,
non-Gaussian LSF or observation-specific wavelength simulation.

Original file links, checksums, source headers, conversion history and
redistribution terms are accessible through ``profile.provenance`` and the
:doc:`provenance reference <instrument_provenance>`. No runtime CRDS, Pandeia
or JWST-pipeline dependency is needed.
