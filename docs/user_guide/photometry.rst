Synthetic photometry
====================

Filters and responses
---------------------

:class:`~speclib.Filter` loads a packaged response curve plus effective
wavelength, bandwidth, flux zeropoint, and zeropoint uncertainty. The response
is a dimensionless :class:`~speclib.Spectrum` tabulated in Å. Accepted
names include 2MASS J/H/Ks, Bessell UBVRI, Gaia G/G_BP/G_RP, Kepler, Spitzer
channels 1--4, and TESS; use the exact names stored in the filter table.

``Filter.resample(wavelength)`` mutates ``filter.response`` in place. This is
important when reusing one Filter instance with spectra on different axes.

Applying a response
-------------------

:func:`~speclib.apply_filter` multiplies spectrum flux density by the response,
integrates over the spectrum wavelength samples, and divides by the tabulated
filter bandwidth. If array shapes differ, it first resamples the response
onto the spectrum wavelength axis (with tapering) and therefore mutates the
filter object. The result is one flux-density quantity in the input spectrum's
flux unit.

The spectrum should cover the response band. The function does not check
coverage, propagate spectrum uncertainty, or implement detector-specific
photon-counting conventions; validate those choices for precision work.

SEDs and magnitude conversion
-----------------------------

:class:`~speclib.SED` applies a sequence of filters to a supplied spectrum and
stores effective wavelength, bandwidth, and flux arrays. ``SED.from_grid``
first loads one model with :meth:`speclib.Spectrum.from_grid`. The
``model_grid`` argument is forwarded, but there is no way in this classmethod
to supply SPHINX C/O or NewEra alpha, so its support for model grids is more
limited than that of ``Spectrum.from_grid``.

:class:`~speclib.SEDGrid` precomputes and trilinearly interpolates SEDs. It
currently accepts PHOENIX only and always interpolates; it cannot select the
nearest model instead.

:func:`~speclib.mag_to_flux` converts a magnitude using the filter zeropoint
and returns ``(mean, standard_deviation)`` from Monte Carlo samples of both
magnitude and zeropoint uncertainty. ``mag_err`` is interpreted in magnitudes.
The function currently uses NumPy's process-wide random generator and has no
seed argument; set ``numpy.random.seed`` immediately before the call when an
exactly reproducible draw is required.

See :doc:`../tutorials/synthetic_photometry` for an offline workflow.
