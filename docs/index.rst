speclib documentation
=====================

``speclib`` provides tools for loading, interpolating, resampling,
binning, broadening, and synthesizing photometry from stellar model spectra.
It currently supports PHOENIX ACES (`Husser et al. 2013
<https://doi.org/10.1051/0004-6361/201219058>`_), MPS-ATLAS (`Kostogryz et
al. 2023 <https://doi.org/10.3847/2515-5172/acc180>`_), SPHINX (`Iyer et al.
2023 <https://doi.org/10.3847/1538-4357/acabc2>`_), and PHOENIX/1D NewEra spectra
(`Hauschildt et al. 2025
<https://doi.org/10.1051/0004-6361/202554171>`_).

The package combines :class:`~speclib.Spectrum` objects (an extension of
``specutils.Spectrum1D``) with :class:`~speclib.SpectralGrid`
collections. It keeps wavelength and flux units explicit and documents the
interpolation, sampling, and boundary assumptions that affect scientific use.

Install the current development release with:

.. code-block:: console

   python -m pip install git+https://github.com/brackham/speclib.git

A minimal spectrum that requires no download is only a few lines:

.. code-block:: python

   import astropy.units as u
   import numpy as np
   from speclib import Spectrum

   wavelength = np.linspace(5000, 5100, 1001) * u.AA
   flux = np.ones(wavelength.size) * u.erg / (u.s * u.cm**2 * u.AA)
   spectrum = Spectrum(spectral_axis=wavelength, flux=flux)
   sampled = spectrum.resample(np.linspace(5010, 5090, 81) * u.AA)

Start with the :doc:`quickstart`, work through the downloadable
:doc:`tutorials/index`, or go directly to the :doc:`api/index`. Before
downloading a model grid, read :doc:`models/index` for coverage, archive size,
cache behavior, citations, and limitations.

New users
   :doc:`installation` · :doc:`quickstart` · :doc:`tutorials/index`

Scientific reference
   :doc:`user_guide/index` · :doc:`models/index` · :doc:`api/index`

Project links
-------------

`GitHub <https://github.com/brackham/speclib>`_ · :doc:`citation` ·
:doc:`changelog` · :doc:`contributing`

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   quickstart
   user_guide/index
   tutorials/index
   models/index
   api/index
   citation
   changelog
   contributing
