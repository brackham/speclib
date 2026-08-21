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

Sampling and temporary grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resolution and output sampling remain separate, but the input sampling must be
fine enough to represent the requested result. Every input interval must
provide at least two samples per target FWHM. For constant resolving power,
this criterion is evaluated in log wavelength and is equivalent locally to
requiring two samples across :math:`\lambda/R`. Under-sampled requests raise a
``ValueError``.

The internal uniform grid has at least ten samples per Gaussian FWHM and is no
coarser than the finest input interval. Before allocating it, speclib estimates
a conservative working set of eight float64 arrays per temporary point and
rejects operations exceeding 512 MiB. This prevents pathological sampling or
resolution requests from causing accidental unbounded allocations.

Edges and ancillary state
~~~~~~~~~~~~~~~~~~~~~~~~~

The convolution extends the endpoint value beyond each boundary. No missing
spectrum is reconstructed outside the observed range. Features within four
Gaussian standard deviations of a boundary can therefore lose flux and have a
truncated line-spread function; the interior flux and resolution guarantees do
not apply there.

Masks and uncertainties require scientifically explicit propagation rules.
Because those rules are outside the scope of these methods, spectra containing
either are rejected instead of silently dropping or copying unsupported state.

Both operations assume that the intrinsic input line width is negligible
relative to the requested broadening. They do not deconvolve a known input
line-spread function, and wavelength-dependent resolving power is not supported.

Spectral grids
~~~~~~~~~~~~~~

:class:`speclib.SpectralGrid` provides the same two methods. They return new
grid objects without reloading model spectra or modifying the originals:

.. code-block:: python

   low_resolution_grid = grid.set_spectral_resolution(5 * u.AA)
   constant_r_grid = grid.set_spectral_resolving_power(2700)

The equivalent constructor options are ``spectral_resolution`` and
``spectral_resolving_power``. Only one may be supplied at a time.
