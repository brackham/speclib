Smitha et al. (2025) surface components
========================================

Product and scientific scope
----------------------------

``speclib`` supports the disk-integrated stellar surface-component spectra
from `Smitha et al. (2025) <https://doi.org/10.3847/2041-8213/ad9aaa>`_,
distributed in the `Edmond data release
<https://doi.org/10.17617/3.HS2EE6>`_. The authors simulated starspots on
G2V, K0V, and M0V dwarfs with the 3D radiative-magnetohydrodynamics MURaM
code and synthesized their spectra with MPS-ATLAS using ray-by-ray radiative
transfer.

These are 12 discrete source products: G2V, K0V, and M0V spectra for each of
the ``quiet``, ``spot``, ``penumbra``, and ``umbra`` surface components. They
are not another
:math:`T_\mathrm{eff}`--:math:`\log g`--metallicity atmosphere grid. Each
stellar type represents a separate simulation, and each file provides the
spatially averaged quiet region, combined spot, penumbra, and umbra. Neither
stellar types nor surface components are interpolated, and this library is not
available through :class:`~speclib.SpectralGrid`.

The released ``spot`` column is loaded directly. ``speclib`` does not attempt
to reconstruct it from the penumbra and umbra, because their spatial weights
are not part of the released spectral tables.

Available simulations and metadata
----------------------------------

The paper reports the following simulation gravities and component effective
temperatures. These temperatures are metadata describing the MHD products;
they are not coordinates accepted by the loader.

.. list-table:: Published Smitha et al. simulation properties
   :header-rows: 1
   :widths: 16 18 22 22 22

   * - Stellar type
     - :math:`\log g` (cgs)
     - Quiet :math:`T_\mathrm{eff}`
     - Penumbra :math:`T_\mathrm{eff}`
     - Umbra :math:`T_\mathrm{eff}`
   * - G2V
     - 4.438
     - 5810 K
     - 5102 K
     - 3819 K
   * - K0V
     - 4.609
     - 4965 K
     - 4410 K
     - 3401 K
   * - M0V
     - 4.826
     - 3696 K
     - 3609 K
     - 3399 K

The paper's table does not publish a combined-spot temperature, so
``meta["component_teff"]`` is ``None`` for ``component="spot"``. ``quiet``
and ``photosphere`` are equivalent public selectors and both produce the
canonical metadata value ``surface_component="quiet"``.

Wavelength sampling and flux convention
---------------------------------------

Each v1.0 spectrum file has 869 identical, strictly increasing wavelength
samples from 200.500 to 6010.001 nm. The sampling is nonuniform: adjacent
spacing ranges from about 0.4 to 20 nm, and
:math:`\lambda/\Delta\lambda` ranges from about 146 to 519. Smitha et al.
describe the synthesized spectra at approximately :math:`R=500`, but the
released tabulation is substantially more coarsely sampled than that over
parts of the spectrum. The quoted resolution is therefore distinct from the
sampling of the distributed files and should not be treated as a constant
measured or Gaussian line-spread function.

Use the actual wavelength array when resampling or degrading a spectrum.
Because ``speclib`` validates the input sampling before convolution,
:meth:`~speclib.Spectrum.set_spectral_resolving_power` may reject requests
that are under-sampled by this released grid.

The release and paper do not identify the wavelength coordinate as air or
vacuum. ``speclib`` therefore retains the released values without an
air/vacuum conversion and records this limitation in the spectrum metadata.

The text-file header describes the data as disk-integrated flux in
``erg s-1 sr-1 cm-2 nm-1``. The retained ``sr-1`` is inconsistent with an
angle-integrated flux. A live check of all three released files resolves the
physical convention numerically: integrating every quiet, penumbral, and
umbral spectrum over wavelength reproduces
:math:`\sigma T_\mathrm{eff}^4` for the published temperatures to within
0.25%. ``speclib`` consequently interprets the columns as emergent surface
:math:`F_\lambda` in ``erg / (s cm2 nm)`` after disk-angle integration. No
distance, radius, solid-angle, or contrast normalization is applied. The only
conversion is from per nm to the package convention
``erg / (s cm2 Å)``; wavelengths are returned in Å.

Retrieval and cache behavior
----------------------------

Retrieve an exact product with :meth:`speclib.Spectrum.from_smitha2025`:

.. code-block:: python

   from speclib import Spectrum

   quiet = Spectrum.from_smitha2025("K0V", component="quiet")
   spot = Spectrum.from_smitha2025("K0V", component="spot")
   penumbra = Spectrum.from_smitha2025("K0V", component="penumbra")
   umbra = Spectrum.from_smitha2025("K0V", component="umbra")

Here ``stellar_type`` and ``component`` are discrete selectors; no
interpolation is performed.

The returned objects are ordinary :class:`~speclib.Spectrum` instances, so
they can be resampled, binned, convolved to lower resolution, or used for
synthetic photometry. This provenance is user-accessible through ``spec.meta``.
Important keys include ``source_library``, ``stellar_type``,
``surface_component``, ``component_teff``, ``logg``, ``paper_doi``,
``data_doi``, ``dataset_version``, ``source_filename``, ``source_md5``,
``source_flux_unit``, the ``native_wavelength_*`` fields,
``native_resolving_power``, ``flux_definition``, and
``wavelength_convention``. Resampling and spectral-resolution operations
return independent copies of this metadata.

The first request downloads the selected stellar type automatically. To
prefetch one or all source files, use:

.. code-block:: python

   from speclib import download_smitha2025_spectra

   download_smitha2025_spectra("K0V")
   download_smitha2025_spectra()  # all three stellar types

Edmond v1.0 supplies three small spectral text files, one per stellar type,
rather than a large grid archive; each file contains all four components.
Files are cached under ``<library root>/smitha2025``. The helper resolves
current numeric file identifiers through the DOI-backed Edmond API, requires
dataset version 1.0 and the pinned filename, size, and published MD5, validates
cached files, and reuses them. ``overwrite=True`` refreshes the selected file
or all three files when no stellar type is supplied.

Citation
--------

For results using these spectra, cite `Smitha et al. (2025)
<https://doi.org/10.3847/2041-8213/ad9aaa>`_ and the `Edmond v1.0 data release
<https://doi.org/10.17617/3.HS2EE6>`_, in addition to the guidance on
:doc:`../citation` for citing ``speclib`` itself.
