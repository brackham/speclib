Spectra
=======

Create a spectrum from wavelength and flux-density quantities:

.. code-block:: python

   import astropy.units as u
   import numpy as np
   from speclib import Spectrum

   wavelength = np.linspace(5000, 5100, 1001) * u.AA
   flux = np.ones(wavelength.size) * u.erg / (u.s * u.cm**2 * u.AA)
   spectrum = Spectrum(spectral_axis=wavelength, flux=flux)
   sampled = spectrum.resample(np.linspace(5010, 5090, 81) * u.AA)

:class:`~speclib.Spectrum` extends :class:`specutils.Spectrum`.
``spectrum.wavelength`` and ``spectrum.flux`` retain their units.
``speclib`` processing methods operate on one-dimensional spectra.
For multidimensional containers, the spectral dimension defaults to last;
see the :class:`~speclib.Spectrum` constructor for overrides.

Loading models
--------------

:meth:`~speclib.Spectrum.from_grid` returns ascending wavelengths in Å and
flux density in ``erg / (s cm2 Å)``. Unit conversion preserves the source's
air/vacuum convention. Model surface fluxes are not scaled by stellar radius
or distance. See :doc:`model_libraries` for loading examples and
:doc:`../models/index` for each product's conventions.

``wl_min`` and ``wl_max`` crop during loading, excluding samples equal to the
endpoints. ``wavelength=`` then resamples the cropped spectrum.

Common operations
-----------------

These methods return new objects:

* :meth:`~speclib.Spectrum.resample` integrates flux into bins inferred from
  new wavelength centers. ``taper=True`` permits zero throughput beyond the
  source endpoints; the default does not taper.
* :meth:`~speclib.Spectrum.regularize` resamples onto a linear wavelength
  axis. Without ``delta_lambda``, it uses the smallest input interval.
* :meth:`~speclib.Spectrum.bin` returns a :class:`~speclib.BinnedSpectrum`
  with explicit centers, widths, limits, and mean flux densities.
* :doc:`Resolution methods <resolution>` broaden spectra on the input axis.
* :meth:`~speclib.Spectrum.select_wavelength` crops a spectrum and any
  declared detector coverage.

Resampling, cropping, and broadening preserve the ``speclib`` class.
Inherited operations follow specutils behavior; arithmetic can return a
``specutils.Spectrum`` without the additional convenience methods.
For spectra with detector gaps, see :doc:`instrument_profiles`.
