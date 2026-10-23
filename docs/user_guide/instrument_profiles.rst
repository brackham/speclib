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
nor resolving power is extrapolated. For example, a well-sampled source
extending to 5.5 µm provides the padding for full PRISM red coverage at
5.3 µm. A well-sampled 5–14 µm source supports the MIRI 5.5–12 µm output.

Instrument matching requires finite, one-dimensional flux without masks or
uncertainties. Coverage-aware resampling and binning also reject masks and
uncertainties. See :doc:`resolution` for the broadening conventions and the
:doc:`instrument tutorial <../tutorials/instrument_profiles>` for plots,
sampling comparisons, and examples with each instrument.
