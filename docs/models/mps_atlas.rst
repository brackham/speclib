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

Set 1 and Set 2 are different scientific flavors. A spectrum or
:class:`~speclib.SpectralGrid` always fixes one set before exact lookup,
nearest selection, or interpolation; the sets are never blended.

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

The generic alias preserves the meaning of the former single-set
compatibility path and provides a convenient default. It does not imply that
Set 1 is scientifically preferable for every application. Loaded objects
record the canonical selector (``mps-atlas-set1`` or ``mps-atlas-set2``) and
the corresponding ``model_set`` value.

Parameter and wavelength grids
------------------------------

Both released sets declare 34,160 nominal combinations on these axes:

* :math:`T_\mathrm{eff}` = 3500--9000 K in 100 K steps;
* :math:`\log g` = 3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, and 5.0
  (cgs);
* [M/H] = -5.0--+1.5 across 61 values, with especially fine 0.05 dex
  sampling from -1.0 through +0.5.

The historical ``feh`` argument to :meth:`speclib.Spectrum.from_grid` selects
the upstream **[M/H]** coordinate for this family. Each archive is indexed
independently from its actual member names, so a nominal axis value is not
treated as proof that a particular combination exists.

Every spectrum contains 1221 nonuniform Kurucz ODF intervals from about
9.09 nm to 160,000 nm (160 micrometers). Across much of the optical and
infrared the local interval spacing is roughly
:math:`\lambda/\Delta\lambda \sim 250`, and it is lower in parts of the UV.
That ratio characterizes tabulation, not a measured or Gaussian resolving
power. The terminal infrared intervals are very coarse, and the enormous
nominal wavelength span should not be interpreted as uniform information
content or sensitivity. Use PHOENIX or an appropriate NewEra product when
line-resolved or substantially higher-sampling spectra are required.

The v3 flux files share a mostly ascending ODF coordinate sequence with one
local reversal near 91 nm. During loading, ``speclib`` stably sorts wavelength
and flux together into ascending order before interpolation or construction of
a :class:`~speclib.Spectrum`. Exact repeated coordinates, if encountered, are
collapsed only when their paired flux values agree to tight numerical
precision; conflicting duplicates raise ``ValueError``. Merely close
wavelengths remain distinct.

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

Edmond distributes whole-set archives rather than individual stellar models.
The first request for a set therefore downloads about 9.5 GB. Check available
space and choose the library root before loading:

.. code-block:: python

   from speclib import download_mps_atlas_grid
   from speclib.utils import set_library_root

   set_library_root("/data/stellar-models")
   download_mps_atlas_grid("set2")

The helper resolves the archive through the DOI-backed Edmond dataset API,
checks release metadata against the v3 size and published MD5, and lets pooch
verify the download. The cache layout is:

.. code-block:: text

   <library root>/mps-atlas/set1/set1.zip
   <library root>/mps-atlas/set2/set2.zip

The selected ZIP is retained and indexed separately. Requested flux members
are read directly from it; atmosphere and intensity members are not extracted.
Repeated requests reuse the archive and in-process index. Passing
``overwrite=True`` clears and refreshes only the selected set. A corrupt
cached archive is removed with an actionable error. Old manually arranged
files directly under ``mps-atlas`` are not silently interpreted as Set 1;
move to the new set-specific cache through the downloader.

Retrieval examples
------------------

High-level loading triggers the same acquisition automatically:

.. code-block:: python

   from speclib import Spectrum, SpectralGrid

   default_set1 = Spectrum.from_grid(
       5800, 4.5, 0.0, model_grid="mps-atlas"
   )
   explicit_set1 = Spectrum.from_grid(
       5800, 4.5, 0.0, model_grid="mps-atlas-set1"
   )
   set2 = Spectrum.from_grid(
       5800, 4.5, 0.0, model_grid="mps-atlas-set2"
   )

   grid = SpectralGrid(
       teff_bds=(5700, 5900),
       logg_bds=(4.4, 4.6),
       feh_bds=(-0.05, 0.05),
       model_grid="mps-atlas-set2",
   )

An exact request returns its archive member. With ``interpolate=False``, an
off-grid request selects the nearest actual combination in the selected set,
with distances scaled by the typical step of each axis. With
``interpolate=True``, interpolation is trilinear in Teff, log g, and [M/H]
only. All required corners must exist in that same set and share an identical
wavelength axis; otherwise loading raises ``ValueError``. Requests beyond the
selected archive's parameter range also raise ``ValueError`` rather than
extrapolating.

Center-to-limb data
-------------------

The v3 archives also contain ``mpsa_intensity_spectra.dat`` at 24 positions
from :math:`\mu=1.0` to 0.01. These specific intensities are valuable for
limb-darkening and stellar-disk calculations, but :math:`\mu` does not fit the
current three-parameter :class:`~speclib.SpectralGrid` model. They remain out
of scope pending a dedicated public API; loading disk-integrated spectra does
not extract or integrate them.

For publication, cite Kostogryz et al. (2023), the Edmond data release,
Witzke et al. (2021), and Kostogryz et al. (2022), in addition to
:doc:`../citation` for ``speclib``.
