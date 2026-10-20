Spectral grids and binned grids
===============================

.. currentmodule:: speclib

SpectralGrid
------------

``get_flux()`` returns an interpolated Astropy Quantity. ``get_spectrum()``
uses the same parameter selection and returns an independent Speclib Spectrum,
including wavelength, flux, instrument provenance and coverage metadata.

.. autoclass:: SpectralGrid
   :members: valid_intervals, get_flux, get_spectrum, set_spectral_resolution, set_spectral_resolving_power, set_variable_resolving_power, observed_by

BinnedSpectralGrid
------------------

This separate class retains its flux-Quantity ``get_spectrum()`` interface;
explicit bin centers and widths belong to the grid.

.. autoclass:: BinnedSpectralGrid
   :members: get_spectrum
