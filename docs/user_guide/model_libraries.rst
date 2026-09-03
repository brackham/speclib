Model library workflows
=======================

The primary documented atmosphere grids are PHOENIX, MPS-ATLAS, SPHINX, and
PHOENIX/1D NewEra. The Smitha et al. (2025) surface-component spectra are a
separate discrete library. These products do not have identical coordinates,
units on disk, completeness, or download behavior. Use the dedicated
:doc:`../models/index` pages to choose a product and cite it.

The common loading interface is:

.. code-block:: python

   from speclib import Spectrum

   spectrum = Spectrum.from_grid(
       teff,
       logg,
       feh,
       model_grid="phoenix",
       interpolate=True,
   )

The three numeric coordinates are effective temperature in kelvin, base-10
surface gravity in cgs, and the library's metallicity coordinate. For SPHINX,
``feh`` selects the filename field called ``logZ`` and ``co_ratio`` is required.
For NewEra, ``feh`` selects the grid's [M/H]-like ``Z`` field and ``alpha`` is
an additional fixed selection. For MPS-ATLAS, ``feh`` selects [M/H], while
``mps-atlas-set1`` and ``mps-atlas-set2`` fix different abundance and
mixing-length assumptions, with ``mps-atlas`` defaulting to Set 1.
``speclib`` does not interpolate C/O, alpha, or between distinct model-library
families or flavors (for example, MPS-ATLAS Set 1 and Set 2).

The Smitha et al. products deliberately use a separate interface because they
are exact surface components from three individual 3D MHD simulations, not a
continuous atmosphere grid:

.. code-block:: python

   penumbra = Spectrum.from_smitha2025("K0V", component="penumbra")

``G2V``, ``K0V``, and ``M0V`` and the quiet, spot, penumbra, and umbra
components are never interpolated. See :doc:`../models/smitha2025`.

Additional accepted selectors
-----------------------------

``Spectrum.from_grid`` also accepts ``drift-phoenix`` and ``nextgen-solar`` as
compatibility paths that require users to arrange their caches. ``newera``
selects individual high-sampling-rate HDF5 files, whereas ``newera_gaia``,
``newera_jwst``, and ``newera_lowres`` select the reduced archives described
on :doc:`../models/newera`.
