Spectral grids
==============

Load a parameter range once for repeated interpolation:

.. code-block:: python

   from speclib import SpectralGrid

   grid = SpectralGrid(
       teff_bds=(3000, 3200),
       logg_bds=(4.0, 5.0),
       metallicity_bds=(-0.5, 0.5),
       model_grid="sphinx",
       co_ratio=0.5,
   )
   spectrum = grid.get_spectrum(3100, 4.5, metallicity=0.0)
   flux = grid.get_flux(3100, 4.5, metallicity=0.0)

Download SPHINX first with :func:`~speclib.download_sphinx_grid`.
A grid holds all selected models in memory on one wavelength axis.
Passing ``wavelength=`` resamples each model during loading.

Retrieving spectra
------------------

``get_spectrum()`` returns an independent :class:`~speclib.Spectrum` with
wavelength, flux, and metadata, including any instrument provenance and
coverage. ``get_flux()`` returns only a one-dimensional Astropy quantity
aligned with ``grid.wavelength``.

Both interpolate by default. Use ``interpolate=False`` for the nearest
loaded model. Missing corners and boundary behavior depend on the library;
see :doc:`interpolation`.

Inspecting a grid
-----------------

``grid.teffs``, ``grid.loggs``, and ``grid.metallicities`` contain the loaded
axis values; the corresponding ``grid_*`` attributes describe the library's
available axes. ``grid.points`` lists combinations actually loaded, which
matters for incomplete grids. ``grid.data`` has shape
``(number_of_models, number_of_wavelength_samples)``; ``grid.unit`` gives its
flux unit. ``grid.metallicity_type`` identifies native [Fe/H] or [M/H].

Constructor bounds expand outward to bracket grid points. Bounds beyond the
library range are clipped with a warning; reversed or malformed bounds raise
``ValueError``. Retrieval outside the loaded bounds also raises ``ValueError``;
retrieval stays within those bounds.

Broadening grids
----------------

The :doc:`resolution` methods return new grids without reloading models.
Constructor options ``spectral_resolution`` and ``spectral_resolving_power``
apply constant width or constant R; choose at most one. Variable R is
available as a method after construction. Use :doc:`instrument_profiles`
for instrument-specific matching.

Binned grids
------------

:class:`~speclib.BinnedSpectralGrid` precomputes explicit bins for each model.
Its ``get_spectrum()`` returns a flux quantity; centers and widths belong to
the grid. Constructor bounds must lie within the library's parameter range;
out-of-range bounds can fail during loading. Prefer ``SpectralGrid`` unless
precomputed bins save substantial repeated work.

The :doc:`spectral-grids notebook <../tutorials/spectral_grids>` compares
exact, nearest, and interpolated spectra using a small synthetic grid.
