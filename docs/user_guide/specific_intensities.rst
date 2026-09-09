Specific intensity
==================

Spectral flux density :math:`F_\lambda` is an angular integral of specific
intensity. :math:`I_\lambda(\mu)` retains the viewing direction and therefore
has a per-steradian unit. Here :math:`\mu=\cos\theta`, with :math:`\mu=1` at
disk center.

:class:`~speclib.SpecificIntensitySpectrum` represents one spectrum at one
:math:`\mu`. :class:`~speclib.SpecificIntensityGrid` collects the native
:math:`\mu` spectra for one stellar model and magnetic state.

Loading and selecting a disk position
-------------------------------------

.. code-block:: python

   from speclib import SpecificIntensityGrid

   grid = SpecificIntensityGrid.from_library(
       "kostogryz2026",
       model="G2",
       metallicity=0.0,
       magnetic_state="ssd",
   )

   print(grid.mu)
   spec = grid.at_mu(0.5)

``grid.mu`` lists the available disk positions. ``at_mu`` requires an exact
native value and does not interpolate.

Working with a spectrum
-----------------------

Specific-intensity spectra support wavelength selection, resampling,
regularization, convolution to lower resolution, and binning. These operations
preserve the specific-intensity class, metadata, and ``sr^-1`` unit.

.. code-block:: python

   import astropy.units as u

   optical = spec.select_wavelength(4000 * u.AA, 8000 * u.AA)
   broadened = optical.set_spectral_resolving_power(100)

The :doc:`../models/kostogryz2026` page lists the stellar models, metallicities,
and magnetic states. The spectra are available only at their native
:math:`\mu` values and use nonuniform, low-resolution ODF sampling. They are
specific intensities, not disk-integrated flux spectra.
