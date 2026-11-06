Instrument profiles
===================

Match a spectrum to an instrument's nominal resolving power and wavelength
coverage with :meth:`~speclib.Spectrum.observed_by`:

.. code-block:: python

   import astropy.units as u
   import numpy as np
   from speclib import Spectrum, get_profile, list_instrument_profiles

   wave = np.geomspace(0.5, 6.0, 30_000) * u.um
   flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
   source = Spectrum(spectral_axis=wave, flux=np.ones(wave.size) * flux_unit)
   observed = source.observed_by("jwst/nirspec/prism/clear")
   print(observed.valid_intervals)

Profiles are bundled and work offline. They apply nominal Gaussian
broadening. Throughput, noise, slit losses, and non-Gaussian LSF structure
require separate modeling.
Use a source with much finer intrinsic resolution than the target and enough
wavelength coverage for convolution beyond the output limits.

Choosing a profile
------------------

.. code-block:: python

   print(list_instrument_profiles())
   profile = get_profile("jwst/nirspec/prism/clear")
   print(profile.configuration, profile.assumptions)
   print(profile.valid_intervals)

Identifiers are case-insensitive. The 55 profiles cover STIS, WFC3/IR,
NIRISS/SOSS, NIRSpec/BOTS, NIRCam grism time series, and MIRI/LRS slitless.
The :doc:`instrument reference <instrument_provenance>` lists all supported
configurations, wavelength ranges, source citations, and limitations.
WFC3/UVIS G280 is :ref:`unavailable pending a suitable resolution calibration
<g280-calibration-status>`.

For example, the short NIRSpec names select BOTS/S1600A1/SUB2048. Select
another supported subarray explicitly:

.. code-block:: python

   observed = source.observed_by(
       "jwst/nirspec/bots/s1600a1/sub1024a/g395h/f290lp"
   )

Sampling the output
-------------------

The default retains source samples within valid coverage. To choose a new
axis, supply either a sampling density or wavelength centers:

.. code-block:: python

   sampled = source.observed_by(profile, samples_per_resolution_element=10)
   centers = np.linspace(0.65, 5.2, 300) * u.um
   resampled = source.observed_by(profile, wavelength=centers)

Automatic spacing is approximately :math:`\delta\lambda=\lambda/(N R)`,
giving a chosen number of wavelength samples per nominal resolution element.
Use explicit wavelengths or bins to represent detector sampling. Explicit centers must be monotonic, with at
least two per requested interval. Their implied bin edges must also lie
within valid coverage.

With explicit ``wavelength``, ``valid_intervals`` on the result describes the
coverage actually returned (the outer implicit bin edges within each profile
interval that contains requested centers), not the nominal profile.
``meta["instrument_profile"]["valid_intervals_um"]`` retains the nominal
bounds.

Some profiles define wider **extended** coverage that is accepted only for
explicit ``wavelength`` requests; the default and generated sampling always use
the recommended ``valid_intervals``. The extended bounds are available as
:attr:`~speclib.instruments.InstrumentProfile.extended_intervals` (``None`` when
a profile has none) and in ``meta["instrument_profile"]["extended_intervals_um"]``.
Extended coverage widens only the accepted output wavelengths: the source and
calibration must still supply the full convolution padding. For example, the
order-1 wavelengths of a NIRISS/SOSS time series can be passed directly:

.. code-block:: python

   observed = grid.observed_by(
       "jwst/niriss/soss/order1", wavelength=observed_wavelength,  # 0.860-2.813 um
   )

See :doc:`instrument_provenance` for the SOSS limits and caveats.

Untrimmed observational wavelengths
-----------------------------------

Observed wavelength arrays often extend beyond an instrument's supported
coverage. By default (``on_invalid="raise"``) any unsupported bin raises
``ValueError``. With ``on_invalid="nan"``, supported bins are modeled and the
rest are returned as ``NaN``, with one ``UserWarning`` per call naming the
mode, the number of excluded bins and the reason:

.. code-block:: python

   observed = grid.observed_by(
       "jwst/niriss/soss/order2", wavelength=observed_wavelength,
       on_invalid="nan",
   )
   good = observed.valid_mask            # shared by every model in the grid
   model = observed.get_spectrum(5750, 4.5, metallicity=0.0).flux[good]
   data, error = flux[good], flux_error[good]    # same mask for observations

The result has exactly the requested ``wavelength`` (length, order, and
descending orientation). ``valid_mask`` marks bins with valid predictions; it
is derived from ``valid_intervals``, not from the flux values. Select
observations with the same mask before fitting, because ``NaN`` is not a
physical flux of zero. ``valid_intervals`` covers only modeled bins, so it never
spans unsupported regions or detector gaps, and
``meta["instrument_profile"]["requested_bins"]`` and ``["valid_bins"]`` record
the counts. If no bin is supported, ``ValueError`` is raised instead of an
all-``NaN`` result. The default may change to ``"nan"`` before the first stable
release.

A bin is supported only if its complete implicit resampling bin lies in one
profile interval (extended coverage included), not across a detector gap, and
the resolving-power calibration covers its convolution padding. Bin edges are
computed from the **entire** requested array before any bin is excluded, so each
retained bin keeps the width it has in the original array. Do not trim the
wavelength array yourself: recomputing edges from the retained centers changes
the outermost supported bins. A center inside coverage can still be unsupported
if its edge is not.

This concerns the **instrument**. A source model that does not cover the
supported bins plus convolution padding, or that is too coarsely sampled,
is a different problem: it is never converted to ``NaN`` and still raises
``ValueError``. Neither flux nor calibration is extrapolated. In particular,
the outermost supported bins need source flux beyond the instrument limit; for
example, for 300 uniformly spaced centers from 0.500 to 1.409 µm, SOSS order 2
supports bins above 0.6 µm, but the first supported bin needs source coverage
down to about 0.5986 µm, including a bracketing sample. Use a source
that extends a little past the instrument limits, or drop the edge bin.

In regression tests with smooth and structured spectra, supported bins agree
with strict calculations using the same bin edges and convolution window to
round-off. Changing the outer coverage changes the adaptive convolution grid
and can produce differences of a few parts in :math:`10^6` in these examples,
including between two strict calls. These differences arise from convolution
discretization rather than ``NaN`` handling; they are not a general accuracy
bound for arbitrary spectra or sampling.

``NaN`` bins are not data. Convolution, ``regularize()`` and repeated
instrument matching reject spectra containing them. ``resample()`` and
``bin()`` use only supported samples and reject requested bins outside
``valid_intervals``. Use ``select_wavelength()`` to crop to the supported
range first when needed.

Both options use :meth:`~speclib.Spectrum.resample`, which integrates flux
into bins. Coarse bins and sparse samples can increase apparent line widths.
Retain fine sampling or fit a bin-integrated model when measuring widths;
use :meth:`~speclib.Spectrum.bin` for explicit observational bin widths.

Working with detector gaps
--------------------------

``valid_intervals`` is a wavelength quantity of shape ``(n, 2)``; ``None``
means unrestricted coverage. The arrays contain only valid samples, with no
filler values in gaps. Plot each interval separately:

.. code-block:: python

   import matplotlib.pyplot as plt

   observed = source.observed_by("jwst/nirspec/g395h/f290lp")
   for lower, upper in observed.valid_intervals:
       selected = ((observed.wavelength >= lower)
                   & (observed.wavelength <= upper))
       plt.plot(observed.wavelength[selected].to_value(u.um),
                observed.flux[selected].value)

``resample()`` and ``bin()`` reject bins crossing a gap or extending beyond
sampled coverage. ``bin()`` needs at least two source samples per bin;
``resample()`` needs at least two centers per requested interval.
``select_wavelength()`` crops both samples and coverage. Broadening,
``regularize()``, and repeated instrument matching reject discontinuous
inputs; use the continuous source or select one interval first.

The continuous source is convolved before detector gaps are imposed, so
source light within an output gap can contribute wings on a detector.
Missing input flux is not reconstructed across a declared gap. Inherited
specutils operations, including slicing, arithmetic, and serialization, do
not account for these coverage limits; use ``select_wavelength()`` to crop.
Removing or replacing coverage metadata removes its safeguards.

Spectral orders are separate from detector gaps. SOSS orders each return an
independent spectrum; overlapping orders are not stitched or decontaminated.

Applying profiles to grids
--------------------------

For a high-resolution grid covering the profile and its convolution padding:

.. code-block:: python

   observed_grid = grid.observed_by("jwst/nirspec/g395h/f290lp")
   spectrum = observed_grid.get_spectrum(5750, 4.5, metallicity=0.0)

The result is an independent :class:`~speclib.SpectralGrid`.
``get_spectrum()`` retains coverage and instrument provenance;
``get_flux()`` returns only the flux quantity. See :doc:`spectral_grids`.

Source requirements
-------------------

Input sampling needs at least two samples per target FWHM in every interval.
The source must also have much finer intrinsic resolution than the target.
Broadening assumes negligible input line widths and applies no deconvolution.
Numeric ``meta['native_resolving_power']`` is checked when present;
descriptive values produce a warning. With no value, check the model's
native resolution yourself.

Each requested output interval needs real source and calibration coverage
for its full convolution padding, including bracketing samples. Neither flux
nor resolving power is extrapolated. With the default sampling or
``samples_per_resolution_element``, that is the whole profile interval.
With explicit ``wavelength``, only the requested output region is produced, so
the source needs just that region's outer implicit bin edges plus local
convolution padding (about four Gaussian sigma of the local resolution
element, plus the support of the kernels that contribute) and bracketing
samples. A model covering 0.64–0.93 µm can therefore be observed through
``jwst/niriss/soss/order2`` at 0.665–0.897 µm, although the full order
cannot be observed from it. Intervals with no requested centers are not
needed. For full-profile output, a well-sampled source
extending to 5.5 µm provides the padding for full PRISM red coverage at
5.3 µm. A well-sampled 5–14 µm source supports the MIRI 5.5–12 µm output.

Instrument matching requires finite, one-dimensional flux without masks or
uncertainties. Coverage-aware resampling and binning also reject masks and
uncertainties. See :doc:`resolution` for the broadening conventions and the
:doc:`instrument tutorial <../tutorials/instrument_profiles>` for plots,
sampling comparisons, and examples with each instrument.
