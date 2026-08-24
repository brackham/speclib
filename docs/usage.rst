Usage
=====

Spectral resolution
-------------------

Spectral resolution and resolving power describe different broadening laws.
Both operations return new objects and retain the input wavelength sampling;
use :meth:`speclib.Spectrum.regularize` separately when a uniform output grid
is desired. Returned spectra and grids contain independent copies of mutable
axis and grid state.

Constant wavelength resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`speclib.Spectrum.set_spectral_resolution` applies a Gaussian whose
full width at half maximum is constant in wavelength:

.. code-block:: python

   import astropy.units as u

   broadened = spectrum.set_spectral_resolution(5 * u.AA)

The resulting resolving power varies as

.. math::

   R(\lambda) = \frac{\lambda}{\Delta\lambda}.

Constant resolving power
~~~~~~~~~~~~~~~~~~~~~~~~

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
wavelength-integrated flux in the continuous, unbounded-domain limit. Finite
boundaries and numerical remapping limit exact conservation over the returned
interval.

The stationary Gaussian is applied to flux per logarithmic wavelength
interval. When expressed as :math:`F_\lambda`, an isolated feature therefore
has a log-normal-like profile rather than a profile that is exactly symmetric
in linear wavelength. This convention is intended primarily for ordinary
astronomical resolving powers, where the distinction is small.

Variable resolving power
~~~~~~~~~~~~~~~~~~~~~~~~

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

The sampled curve must cover the full wavelength range of the input spectrum;
the method does not extrapolate. :math:`R(\lambda)` is intended to vary
smoothly relative to a local resolution element. A resolving-power curve by
itself is not sufficient to describe an instrument whose line-spread function
changes substantially across one local FWHM; such cases require a calibrated
wavelength-dependent LSF.

Each input wavelength contributes a source-centered Gaussian with its local
resolving power. This conserves wavelength-integrated flux and uses the same
log-wavelength and flux conventions as constant resolving power. A constant
sampled curve therefore reproduces
:meth:`speclib.Spectrum.set_spectral_resolving_power` within numerical
tolerance.

The returned spectrum retains the exact input wavelength axis. If detector or
extracted-spectrum sampling is also required, apply :meth:`~speclib.Spectrum.resample`
as a separate operation.

Sampling and temporary grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resolution and output sampling remain separate, but the input sampling must be
fine enough to represent the requested result. Every input interval must
provide at least two samples per target FWHM. For constant resolving power,
this criterion is evaluated in log wavelength and is equivalent locally to
requiring two samples across :math:`\lambda/R`. Under-sampled requests raise a
``ValueError``.

Constant-width operations use an internal uniform grid with at least ten
samples per Gaussian FWHM and no coarser spacing than the finest input
interval. Variable resolving power instead uses an adaptive grid whose local
spacing follows the requested FWHM. Before allocating either representation,
speclib estimates its work and memory requirements and rejects impractical
operations. This avoids combining global sampling set by the narrowest kernel
with support set by the broadest kernel.

Edges and ancillary state
~~~~~~~~~~~~~~~~~~~~~~~~~

Constant-width convolution extends the endpoint value beyond each boundary.
Variable-width kernels are truncated and renormalized at the returned
wavelength limits. No missing spectrum is reconstructed outside the observed
range, and edge line-spread functions should not be interpreted as complete
instrument profiles. Interior resolution and centroid guarantees do not apply
near a boundary.

Masks and uncertainties require scientifically explicit propagation rules.
Because those rules are outside the scope of these methods, spectra containing
either are rejected instead of silently dropping or copying unsupported state.

All resolution operations assume that the intrinsic input line width is negligible
relative to the requested broadening. They do not deconvolve a known input
line-spread function. Variable resolving power assumes a Gaussian line-spread
function; wavelength-dependent non-Gaussian profiles are not supported.

Spectral grids
~~~~~~~~~~~~~~

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
