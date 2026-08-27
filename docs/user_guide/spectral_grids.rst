Spectral grids
==============

A :class:`~speclib.SpectralGrid` preloads model spectra over selected
:math:`T_\mathrm{eff}`, :math:`\log g`, and metallicity axes. This costs memory
up front but makes repeated retrieval inexpensive. Every stored spectrum is
put onto one wavelength axis; passing ``wavelength=`` resamples each model as
it is loaded.

Constructing and inspecting a grid
----------------------------------

.. code-block:: python

   from speclib import SpectralGrid

   grid = SpectralGrid(
       teff_bds=(3000, 3200),
       logg_bds=(4.0, 5.0),
       feh_bds=(-0.5, 0.5),
       model_grid="sphinx",
       co_ratio=0.5,
   )

``grid.grid_teffs``, ``grid.grid_loggs``, and ``grid.grid_fehs`` describe the
declared/available axes for that library selection. ``grid.teffs``,
``grid.loggs``, and ``grid.fehs`` are the values loaded within the aligned
bounds. ``grid.points`` lists combinations actually loaded, which is the more
informative property for an incomplete grid. ``grid.wavelength`` and
``grid.unit`` describe the common spectral data; ``grid.data`` has shape
``(number_of_loaded_models, number_of_wavelength_samples)``.

Bounds are aligned outward to grid points. Values outside a library's
declared minimum or maximum are clipped and emit ``UserWarning``. Reversed
bounds or bounds that do not contain exactly two elements raise ``ValueError``.
This constructor clipping does not authorize extrapolation:
:meth:`~speclib.SpectralGrid.get_flux` raises
``ValueError`` for a request outside the loaded bounds.

Retrieval
---------

.. code-block:: python

   from speclib import Spectrum

   flux = grid.get_flux(3100, 4.5, 0.0, interpolate=True)
   spectrum = Spectrum(spectral_axis=grid.wavelength, flux=flux)

``get_flux`` returns a one-dimensional Astropy quantity aligned with
``grid.wavelength``; it does not return a ``Spectrum``. Setting
``interpolate=False`` selects the nearest loaded model. The older
``get_spectrum`` name is a deprecated alias for ``get_flux`` and emits a
``DeprecationWarning``.

Resolution methods operate on every stored spectrum, return independent grid
objects, and reuse one convolution plan. Constructor keywords
``spectral_resolution`` and ``spectral_resolving_power`` provide the constant
width and constant-R operations respectively; specify at most one. Variable
resolving power is available as a method after construction.

Binned grids
------------

:class:`~speclib.BinnedSpectralGrid` loads and bins each model into supplied
``center`` and ``width`` arrays. Its retrieval method is still named
``get_spectrum`` and returns a flux quantity. This class shares much of the
interpolation behavior across three axes but does not use the newer clipping
helper; requests outside its declared parameter axes can therefore fail during
construction rather than being clipped. Prefer ``SpectralGrid`` unless
precomputed bins materially reduce repeated work.
