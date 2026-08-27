SPHINX
======

Product
-------

``model_grid="sphinx"`` reads **SPHINX I V4**, the cloud-free, low-resolution
M-dwarf grid from `Zenodo record 11392341
<https://doi.org/10.5281/zenodo.11392341>`_. The associated scientific paper is
`Iyer et al. (2023) <https://doi.org/10.3847/1538-4357/acabc2>`_. The archive
describes one-dimensional, self-consistent radiative-convective,
thermochemical-equilibrium models with mixing-length parameter 1.

Spectra cover 0.1--20 micrometers at approximately :math:`R=250`. Files store
wavelength in micrometers and flux in ``W / (m2 m)``. ``speclib`` converts the
units to Å and ``erg / (s cm2 Å)``; it does not otherwise
transform the wavelength coordinate.

Parameters and incompleteness
-----------------------------

Repository metadata declares filename values spanning:

* :math:`T_\mathrm{eff}` = 2000--4000 K in 100 K steps;
* :math:`\log g` values 4.0, 4.2, 4.25, 4.5, 4.7, 4.75, 5.0, 5.2, 5.25,
  and 5.5 (cgs);
* ``logZ`` values -1.0--+1.0 in 0.25 dex steps, passed through the API as
  ``feh``;
* C/O = 0.3, 0.5, 0.7, or 0.9.

The archive is sparse: these distinct values do not occur in every
combination, and the actual axes for a selected C/O slice are derived from the
extracted filenames rather than assumed from those nominal ranges.
Spectrum filenames follow
``Teff_<teff>_logg_<logg>_logZ_<logZ>_CtoO_<C/O>.txt``; their numeric
formatting is significant when indexing an exact model.
``speclib`` indexes exact parameter quadruples at runtime. A
:class:`~speclib.SpectralGrid` fixes one C/O value and interpolates only Teff,
log g, and ``logZ``. It raises ``ValueError`` if a requested trilinear
interpolation lacks any required corner. Nearest retrieval chooses an actual
available combination rather than an impossible tuple assembled independently
along each axis.

Download and use
----------------

.. code-block:: python

   from speclib import Spectrum, download_sphinx_grid

   download_sphinx_grid()
   spectrum = Spectrum.from_grid(
       3000, 4.5, 0.0,
       model_grid="sphinx",
       co_ratio=0.5,
       interpolate=False,
   )

:func:`~speclib.download_sphinx_grid` downloads the 160.7 MB
``SPHINX_MAY2024_CLOUDFREE_UPDATED.tar.gz`` archive, verifies the checksum
published by Zenodo, keeps the archive under ``<library root>/sphinx``, and
safely extracts only recognized spectrum files. ``overwrite=True`` clears the
existing SPHINX cache first.

C/O is mandatory even for exact retrieval. Passing an unavailable C/O value
raises ``ValueError`` and lists the values present in the extracted archive.
Different required corner spectra must also share exactly the same wavelength
axis.

For publication, the Zenodo record asks users to cite both the data DOI and
Iyer et al. (2023), in addition to :doc:`../citation` for ``speclib``.
