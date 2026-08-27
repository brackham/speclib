Quickstart
==========

This self-contained workflow takes about five minutes and does not download a
stellar model archive. The :doc:`tutorials/quickstart` notebook contains the
same workflow with plots and can be downloaded from :doc:`tutorials/index`.

Create a spectrum
-----------------

Build a toy spectrum whose wavelength and flux carry Astropy units:

.. code-block:: python

   import astropy.units as u
   import numpy as np
   import speclib
   from speclib import Filter, Spectrum, apply_filter

   print(speclib.__version__)

   wavelength = np.linspace(10_000, 14_000, 8001) * u.AA
   continuum = np.ones(wavelength.size)
   line_1 = 0.25 * np.exp(-0.5 * ((wavelength.value - 11_500) / 2.0) ** 2)
   line_2 = 0.15 * np.exp(-0.5 * ((wavelength.value - 12_800) / 3.0) ** 2)
   flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
   spectrum = Spectrum(
       spectral_axis=wavelength,
       flux=(continuum - line_1 - line_2) * flux_unit,
   )

``spectrum.wavelength`` and ``spectrum.flux`` are Astropy quantities. The
usual ``Spectrum1D`` attributes and slicing behavior remain available.

Change resolution and sampling
------------------------------

Broadening and resampling are distinct operations and both return new objects:

.. code-block:: python

   broadened = spectrum.set_spectral_resolving_power(1000)
   output_wavelength = np.linspace(10_100, 13_900, 381) * u.AA
   sampled = broadened.resample(output_wavelength)

The original ``spectrum`` is unchanged. The first operation applies a
constant Gaussian resolving power :math:`R=\lambda/\Delta\lambda` while
retaining the input axis; the second changes the wavelength samples. Read
:doc:`user_guide/resolution` before applying a requested resolution to real
data.

Compute a band-integrated flux density
--------------------------------------

Packaged filter curves can be applied directly:

.. code-block:: python

   j_band = Filter("2MASS J")
   j_flux = apply_filter(spectrum, j_band)
   print(j_flux)

``apply_filter`` resamples the filter response to the spectrum when their
array shapes differ, integrates ``flux * response`` over wavelength, and
divides by the tabulated filter bandwidth. See
:doc:`user_guide/photometry` for interpretation and limitations.

Load real model data when ready
-------------------------------

The public call for loading a model is concise, but may initiate a download:

.. code-block:: python

   from speclib import Spectrum

   model = Spectrum.from_grid(
       teff=3000,
       logg=4.5,
       feh=0.0,
       co_ratio=0.5,
       model_grid="sphinx",
       interpolate=False,
   )

SPHINX must first be installed with :func:`~speclib.download_sphinx_grid`.
Choose a product using :doc:`models/index`, then use
:doc:`user_guide/spectral_grids` for efficient repeated interpolation.
