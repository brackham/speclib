MPS-ATLAS
=========

Product and scope
-----------------

``speclib`` supports the disk-integrated spectra in release v3 of the
`MPS-ATLAS Library of Stellar Model Atmospheres and Spectra
<https://doi.org/10.17617/3.NJ56TR>`_. The library is described by
`Kostogryz et al. (2023) <https://doi.org/10.3847/2515-5172/acc180>`_ and was
computed with the code presented by `Witzke et al. (2021)
<https://doi.org/10.1051/0004-6361/202140275>`_. The atmosphere and
limb-darkening grid is described by `Kostogryz et al. (2022)
<https://doi.org/10.1051/0004-6361/202243722>`_.

This is a 1D LTE opacity-distribution-function (ODF) product intended for
stellar SEDs, broadband and spectrophotometric calculations, and
transit-adjacent applications that do not require resolved spectral lines.
Only ``mpsa_flux_spectra.dat``, the disk-integrated product, is exposed by the
current API. The upstream atmosphere structures and 24-position
center-to-limb intensities are not loaded.

Two discrete sets
-----------------

Set 1 and Set 2 use different abundance mixtures and mixing-length treatments.
Each spectrum or :class:`~speclib.SpectralGrid` uses one set for lookup,
nearest selection, and interpolation.

.. list-table:: MPS-ATLAS selectors
   :header-rows: 1
   :widths: 22 29 29 20

   * - Selector
     - Abundance mixture
     - Mixing-length treatment
     - v3 archive
   * - ``mps-atlas-set1``
     - Grevesse & Sauval (1998)
     - fixed :math:`\alpha_\mathrm{MLT}=1.25`
     - ``set1.zip``; 9,503,312,271 bytes
   * - ``mps-atlas-set2``
     - Asplund et al. (2009)
     - parameter-dependent values from Viani et al. (2018)
     - ``set2.zip``; 9,430,754,401 bytes
   * - ``mps-atlas``
     - Set 1
     - Set 1
     - alias of ``mps-atlas-set1``

``mps-atlas`` defaults to Set 1. Choose a set based on its abundance mixture
and mixing-length treatment. Metadata records the canonical selector and
``model_set``.

Parameter and wavelength grids
------------------------------

Both released sets declare 34,160 nominal combinations on these axes:

* :math:`T_\mathrm{eff}` = 3500--9000 K in 100 K steps;
* :math:`\log g` = 3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, and 5.0
  (cgs);
* [M/H] = -5.0--+1.5 across 61 values, with especially fine 0.05 dex
  sampling from -1.0 through +0.5.

The ``metallicity`` argument to :meth:`speclib.Spectrum.from_grid` selects
the upstream **[M/H]** coordinate for this family; ``mh`` is a native alias.
No abundance conversion is performed. Returned metadata records
``metallicity`` and ``metallicity_type="mh"``. Each archive is indexed from
its filenames to identify the available parameter combinations.

Every spectrum contains 1221 nonuniform Kurucz ODF intervals from about
9.09 nm to 160,000 nm (160 micrometers). Across much of the optical and
infrared the local interval spacing is roughly
:math:`\lambda/\Delta\lambda \sim 250`, and it is lower in parts of the UV.
This ratio describes wavelength spacing. The ODF intervals are broad, with
very coarse sampling at the infrared end. Their widths do not define a
calibrated Gaussian line-spread function. Use PHOENIX or an appropriate NewEra product when
line-resolved or substantially higher-sampling spectra are required.

A local wavelength reversal near 91 nm is sorted together with the flux during
loading. Conflicting duplicate wavelengths raise ``ValueError``; near but
distinct wavelengths are retained.

Units and physical conversion
-----------------------------

Source files tabulate wavelength in nm and disk-integrated
:math:`F_\nu` in ``erg s-1 cm-2 Hz-1`` using the release's irradiance
convention at 1 AU for a solar-radius source. ``speclib`` first applies the
documented geometric factor :math:`(\mathrm{AU}/R_\odot)^2`, then uses
Astropy's spectral-density equivalency to convert to surface
:math:`F_\lambda`. Returned spectra have ascending wavelengths in Å and flux
density in ``erg / (s cm2 Å)``, matching the package convention.

Download and cache behavior
---------------------------

Edmond distributes one archive per set.
The first request for a set therefore downloads about 9.5 GB. Check available
space and choose the library root before loading:

.. code-block:: python

   from speclib import download_mps_atlas_grid
   from speclib.utils import set_library_root

   set_library_root("/data/stellar-models")
   download_mps_atlas_grid("set2")

Archives are checked against v3 sizes and published MD5 hashes, and cached as
``<library root>/mps-atlas/set1/set1.zip`` or ``set2/set2.zip``. Requested
flux members are read directly from the ZIP; atmosphere and intensity members
are not extracted. ``overwrite=True`` refreshes only the selected set.
Corrupt caches raise an error and are removed. Old files directly under
``mps-atlas`` need replacing through the set-specific downloader.

Retrieval examples
------------------

High-level loading triggers the same acquisition automatically:

.. code-block:: python

   from speclib import Spectrum, SpectralGrid

   default_set1 = Spectrum.from_grid(
       5800, 4.5, metallicity=0.0, model_grid="mps-atlas"
   )
   explicit_set1 = Spectrum.from_grid(
       5800, 4.5, metallicity=0.0, model_grid="mps-atlas-set1"
   )
   set2 = Spectrum.from_grid(
       5800, 4.5, metallicity=0.0, model_grid="mps-atlas-set2"
   )

   grid = SpectralGrid(
       teff_bds=(5700, 5900),
       logg_bds=(4.4, 4.6),
       metallicity_bds=(-0.05, 0.05),
       model_grid="mps-atlas-set2",
   )

An exact request returns its archive member. With ``interpolate=False``, an
off-grid request selects the nearest actual combination in the selected set,
with distances scaled by the typical step of each axis. With
``interpolate=True``, interpolation is trilinear in Teff, log g, and [M/H]
only. All required corners must exist in that same set and share an identical
wavelength axis; otherwise loading raises ``ValueError``. Requests beyond the
selected archive's parameter range also raise ``ValueError``.

Center-to-limb data
-------------------

The archives also contain ``mpsa_intensity_spectra.dat`` at 24 disk positions
from :math:`\mu=1.0` to 0.01. These intensities are not loaded or integrated
by the current API.

For publication, cite Kostogryz et al. (2023), the Edmond data release,
Witzke et al. (2021), and Kostogryz et al. (2022), in addition to
:doc:`../citation` for ``speclib``.
