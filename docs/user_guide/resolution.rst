Resampling, binning, and spectral resolution
============================================

Use :meth:`speclib.Spectrum.resample` to change wavelength samples and
:meth:`speclib.Spectrum.bin` to measure mean flux density in explicit
intervals. The resolution methods below apply Gaussian broadening but retain
the exact input samples; resample afterward when detector sampling is also
required.

Spectral resolution and resolving power describe different broadening laws.
All three broadening operations return independent objects.

Constant wavelength resolution
------------------------------

:meth:`speclib.Spectrum.set_spectral_resolution` applies a Gaussian whose
full width at half maximum is constant in wavelength:

.. code-block:: python

   import astropy.units as u

   broadened = spectrum.set_spectral_resolution(5 * u.AA)

The resulting resolving power varies as

.. math::

   R(\lambda) = \frac{\lambda}{\Delta\lambda}.

Constant resolving power
------------------------

:meth:`speclib.Spectrum.set_spectral_resolving_power` applies constant
resolving power:

.. code-block:: python

   broadened = spectrum.set_spectral_resolving_power(2700)

The Gaussian width therefore varies as

.. math::

   \Delta\lambda(\lambda) = \frac{\lambda}{R}.

The convolution is evaluated on a temporary uniform log-wavelength grid. Since

.. math::

   F_\lambda\,d\lambda = \lambda F_\lambda\,d\ln\lambda,

the implementation convolves :math:`\lambda F_\lambda` and then divides by
wavelength. This is the appropriate measure for preserving
wavelength-integrated flux in the continuous limit on an unbounded domain.
Finite boundaries and numerical remapping limit exact conservation over the
returned interval.

The stationary Gaussian is applied to flux per logarithmic wavelength
interval. When expressed as :math:`F_\lambda`, an isolated feature therefore
has a log-normal-like profile with slight asymmetry in linear wavelength. This convention is intended primarily for ordinary
astronomical resolving powers, where the distinction is small.

Variable resolving power
------------------------

:meth:`speclib.Spectrum.set_variable_resolving_power` accepts sampled
wavelength and resolving-power arrays and linearly interpolates
:math:`R(\lambda)` between them:

.. code-block:: python

   curve_wavelength = [0.6, 1.0, 2.0, 5.3] * u.um
   curve_resolving_power = [30, 100, 200, 300]
   broadened = spectrum.set_variable_resolving_power(
       curve_wavelength,
       curve_resolving_power,
   )

The sampled curve must cover the full input wavelength range.
:math:`R(\lambda)` should vary smoothly across a local resolution element.
Rapid changes in the line-spread function require a calibrated
wavelength-dependent LSF.

Each input wavelength contributes a source-centered Gaussian with its local
resolving power. This conserves wavelength-integrated flux and uses the same
log-wavelength and flux conventions as constant resolving power. A constant
sampled curve reproduces
:meth:`~speclib.Spectrum.set_spectral_resolving_power` within numerical
tolerance.

Sampling and temporary grids
----------------------------

The input sampling must be fine enough to represent the requested result.
Every input interval must provide at least two samples per target FWHM. For
constant resolving power, this criterion is evaluated in log wavelength and
is locally equivalent to two samples across :math:`\lambda/R`.
Undersampled requests raise ``ValueError``.

Very fine input sampling or broad wavelength ranges can require large temporary
grids. Requests exceeding the work or memory limits raise ``ValueError``.

Edges, flux, and ancillary state
--------------------------------

Constant-width convolution extends the endpoint value beyond each boundary.
Variable-width kernels are truncated and renormalized at the returned
wavelength limits. These boundary treatments alter the line-spread function
near the endpoints, where line widths, centroids, and integrated flux can
differ from their interior behavior.

Resolution methods reject spectra with masks or uncertainties because their
propagation is not implemented.

All resolution operations assume that the intrinsic input line width is
negligible relative to the requested broadening. They do not deconvolve a
known input line-spread function. The line-spread function is Gaussian;
wavelength-dependent non-Gaussian profiles are not supported.

Spectral grids
--------------

:class:`speclib.SpectralGrid` provides the same three methods. They return new
grid objects without reloading model spectra or modifying the originals:

.. code-block:: python

   low_resolution_grid = grid.set_spectral_resolution(5 * u.AA)
   constant_r_grid = grid.set_spectral_resolving_power(2700)
   variable_r_grid = grid.set_variable_resolving_power(
       curve_wavelength,
       curve_resolving_power,
   )

The equivalent constructor options are ``spectral_resolution`` and
``spectral_resolving_power``. Only one may be supplied at a time.

The :doc:`../tutorials/spectral_resolution` notebook shows all three modes and
plots their effects on isolated features.
