Model library workflows
=======================

The primary documented families are PHOENIX, SPHINX, and PHOENIX/1D NewEra.
They do not have identical grids, units on disk, completeness, or download
behavior. Use the dedicated :doc:`../models/index` pages to choose a product
and cite it.

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
an additional fixed selection. ``speclib`` does not interpolate C/O or alpha.

Additional accepted selectors
-----------------------------

``Spectrum.from_grid`` also accepts ``drift-phoenix``, ``mps-atlas``, and
``nextgen-solar``. These are compatibility paths that require the user to
place files in the expected cache layout; there is no corresponding public
download helper. ``newera`` selects individual high-sampling-rate HDF5 files,
whereas ``newera_gaia``, ``newera_jwst``, and ``newera_lowres`` select the
reduced archives described on :doc:`../models/newera`. These compatibility
selectors are intentionally not presented as equally supported top-level
families.
