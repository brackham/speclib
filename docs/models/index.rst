Model libraries
===============

Model spectra are external scientific data products, not bundled package
examples. Each family has its own physical assumptions, coordinate coverage,
sampling, completeness, citation, and storage cost. Read the matching page
before selecting a product or ``model_grid`` value.

.. toctree::
   :maxdepth: 1

   phoenix
   mps_atlas
   kostogryz2026
   smitha2025
   sphinx
   newera

Quick selection
---------------

* Use :doc:`phoenix` for the established PHOENIX-ACES-AGSS-COND-2011
  high-resolution stellar library, with individual FITS files downloaded as
  needed.
* Use :doc:`mps_atlas` for a dense FGK grid and broad SED work on 1221
  nonuniform ODF intervals; choose its abundance/mixing-length Set 1 or Set 2
  explicitly when that distinction matters.
* Use :doc:`kostogryz2026` for center-to-limb specific intensities from 3D
  MURaM simulations at ten native disk positions.
* Use :doc:`smitha2025` for discrete quiet, spot, penumbral, and umbral spectra
  from 3D MHD simulations of G2V, K0V, and M0V stars. These products are not
  interpolated or exposed through ``SpectralGrid``.
* Use :doc:`sphinx` for the low-resolution M-dwarf grid with an explicit C/O
  dimension and a comparatively small single archive.
* Use :doc:`newera` for the newer PHOENIX/1D LTE atmospheres and choose its
  Gaia, JWST, or Low-Res reduced product by wavelength coverage and sampling.

Flux libraries return :class:`~speclib.Spectrum`. The Kostogryz center-to-limb
library returns :class:`~speclib.SpecificIntensitySpectrum`, preserving the
per-steradian intensity unit.
