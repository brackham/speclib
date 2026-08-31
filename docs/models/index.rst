Model libraries
===============

Model spectra are external scientific data products, not bundled package
examples. Each family has its own physical assumptions, coordinate coverage,
sampling, completeness, citation, and storage cost. Read the matching page
before selecting a ``model_grid`` value.

.. toctree::
   :maxdepth: 1

   phoenix
   mps_atlas
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
* Use :doc:`sphinx` for the low-resolution M-dwarf grid with an explicit C/O
  dimension and a comparatively small single archive.
* Use :doc:`newera` for the newer PHOENIX/1D LTE atmospheres and choose its
  Gaia, JWST, or Low-Res reduced product by wavelength coverage and sampling.

All loaders return :class:`~speclib.Spectrum` in Å and
``erg / (s cm2 Å)`` after reading the format used by each product.
This unit conversion does not alter a product's wavelength convention, and a
common output format does not make the underlying model assumptions
interchangeable.
