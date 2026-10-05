Model library workflows
=======================

The primary documented atmosphere grids are PHOENIX, MPS-ATLAS, SPHINX, and
PHOENIX/1D NewEra. The Smitha et al. (2025) surface-component spectra and
Kostogryz et al. (2026) center-to-limb specific intensities are separate
discrete libraries. These products do not have identical coordinates, units
on disk, completeness, or download behavior. Use the dedicated
:doc:`../models/index` pages to choose a product and cite it.

The common loading interface is:

.. code-block:: python

   from speclib import Spectrum

   spectrum = Spectrum.from_grid(
       teff,
       logg,
       metallicity=metallicity,
       model_grid="phoenix",
       interpolate=True,
   )

The three numeric coordinates are effective temperature in kelvin, base-10
surface gravity in cgs, and native metallicity.

``metallicity`` specifies the native metallicity coordinate of the selected
model library. SpecLib does not convert between ``[Fe/H]`` and ``[M/H]``.
Users comparing libraries must determine whether their abundance conventions
are scientifically equivalent for their application; alpha enhancement and
the adopted solar abundance mixture can matter.

.. list-table:: Native metallicity definitions
   :header-rows: 1
   :widths: 45 25 30

   * - Library / selector
     - Native coordinate
     - ``metallicity_type``
   * - PHOENIX-ACES (``phoenix``)
     - [Fe/H]
     - ``"feh"``
   * - NewEra (``newera``, ``newera_gaia``, ``newera_jwst``, ``newera_lowres``)
     - [M/H]
     - ``"mh"``
   * - SPHINX I (``sphinx``)
     - [M/H]; filename label ``logZ``
     - ``"mh"``
   * - MPS-ATLAS (``mps-atlas``, ``mps-atlas-set1``, ``mps-atlas-set2``)
     - [M/H]
     - ``"mh"``
   * - Kostogryz et al. (2026) (``kostogryz2026``)
     - [M/H]
     - ``"mh"``
   * - Smitha et al. (2025) (``smitha2025``)
     - No metallicity coordinate
     - Absent

For SPHINX, ``logZ`` is the upstream filename/storage label for [M/H], not
a third physical metallicity coordinate; ``co_ratio`` is required. NewEra's
filename label ``Z`` likewise stores [M/H], not a metal mass fraction;
``alpha`` selects a fixed additional slice.
For MPS-ATLAS, ``mps-atlas-set1`` and ``mps-atlas-set2`` fix different abundance and
mixing-length assumptions, with ``mps-atlas`` defaulting to Set 1.
``speclib`` does not interpolate C/O, alpha, or between distinct model-library
families or flavors (for example, MPS-ATLAS Set 1 and Set 2).

Native aliases
--------------

Prefer ``metallicity=`` in cross-library code. ``feh=`` is a scientifically
native alias for PHOENIX-ACES and emits no warning. ``mh=`` is a native alias
for NewEra (all selectors), SPHINX, and MPS-ATLAS. Passing ``mh=`` to PHOENIX-ACES
raises ``ValueError``. Supplying more than one coordinate keyword raises
``ValueError``, even when the values agree.

Passing ``feh=`` to NewEra, SPHINX, or MPS-ATLAS raises
``ValueError`` and directs the caller to ``metallicity=`` or ``mh=``.
No [Fe/H]-to-[M/H] conversion is performed. Existing positional arguments
retain their order and numeric meaning.

Use ``metallicity_bds`` for grid constructors and ``metallicity_range`` for
``utils.download_newera_hsr_subset``. ``mh_bds``/``mh_range`` are native aliases
on [M/H] grids. Mixing range/bounds forms also raises ``ValueError``.
The NewEra helpers' old ``z``/``zscale`` storage-name keywords are deprecated
aliases of ``metallicity``.

Returned spectra expose ``meta["metallicity"]`` and
``meta["metallicity_type"]``. The numeric coordinate describes the selected
model plane for exact/nearest retrieval, including existing filename rounding,
or the requested coordinate for interpolation. Grid objects expose
``metallicities``, ``grid_metallicities``, ``metallicity_bds``, and
``metallicity_type``; ``meta["metallicities"]`` records the loaded axis values.
SEDs inherit the spectrum metadata; SEDGrid supports PHOENIX only.
Smitha products have neither metallicity metadata field.

The Smitha et al. products deliberately use a separate interface because they
are exact surface components from three individual 3D MHD simulations, not a
continuous atmosphere grid:

.. code-block:: python

   penumbra = Spectrum.from_smitha2025("K0V", component="penumbra")

``G2V``, ``K0V``, and ``M0V`` and the quiet, spot, penumbra, and umbra
components are never interpolated. See :doc:`../models/smitha2025`.

Kostogryz et al. intensities use :class:`~speclib.SpecificIntensityGrid`
because they vary with disk position and are not disk-integrated fluxes. See
:doc:`specific_intensities` for the workflow and
:doc:`../models/kostogryz2026` for available models.

Additional accepted selectors
-----------------------------

``Spectrum.from_grid`` also accepts ``drift-phoenix`` and ``nextgen-solar`` as
compatibility paths that require users to arrange their caches. ``newera``
selects individual high-sampling-rate HDF5 files, whereas ``newera_gaia``,
``newera_jwst``, and ``newera_lowres`` select the reduced archives described
on :doc:`../models/newera`.
