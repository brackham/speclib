Quickstart
==========

Broaden a synthetic absorption line and measure a J-band flux density.
This example runs without downloading model data. The
:doc:`notebook <tutorials/quickstart>` adds plots and is
:download:`available to download <tutorials/quickstart.ipynb>`.

Create a spectrum
-----------------

Here we use a simple spectrum with a flat continuum and a Gaussian line to
illustrate the workflow. Wavelength and flux are Astropy quantities.

.. code-block:: python

   import astropy.units as u
   import numpy as np
   from speclib import Filter, Spectrum, apply_filter

   wavelength = np.linspace(10_000, 15_000, 10_001) * u.AA
   line = 0.25 * np.exp(-0.5 * ((wavelength.value - 11_500) / 2.0) ** 2)
   flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
   spectrum = Spectrum(spectral_axis=wavelength, flux=(1 - line) * flux_unit)

Broaden and resample
--------------------

.. code-block:: python

   broadened = spectrum.set_spectral_resolving_power(1000)
   sampled = broadened.resample(np.linspace(10_100, 14_900, 481) * u.AA)

Broadening applies Gaussian resolving power :math:`R=\lambda/\mathrm{FWHM}`
and retains the input wavelengths. Resampling changes the samples and
integrates flux into their implied bins. Both return new spectra.
The line becomes shallower and wider. See :doc:`user_guide/resolution` for
input-resolution requirements and edge effects.

Calculate a band flux
---------------------

.. code-block:: python

   j_flux = apply_filter(spectrum, Filter("2MASS J"))
   print(j_flux)

The result is close to the continuum flux density. ``apply_filter`` integrates
flux times response over wavelength and divides by the tabulated bandwidth.
The spectrum covers the full J-band response; see
:doc:`user_guide/photometry` for normalization and uncertainty limitations.

To load a stellar model, choose a :doc:`model library <models/index>` and
follow :doc:`user_guide/model_libraries`. For repeated interpolation, use
:doc:`user_guide/spectral_grids`.
