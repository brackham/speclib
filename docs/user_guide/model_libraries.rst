Model library workflows
=======================

Load a model with :meth:`~speclib.Spectrum.from_grid`:

.. code-block:: python

   from speclib import Spectrum

   spectrum = Spectrum.from_grid(
       4000, 4.5, metallicity=0.0, model_grid="phoenix",
   )

This may download model files. Choose a product using :doc:`../models/index`,
which lists scientific assumptions, coverage, citations, and download sizes.

The three numeric coordinates are effective temperature in kelvin, base-10
surface gravity in cgs, and native metallicity.

``metallicity`` specifies the native metallicity coordinate of the selected
model library. Speclib does not convert between ``[Fe/H]`` and ``[M/H]``.
Alpha enhancement and solar abundance mixtures can also differ between libraries.

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

SPHINX filenames label [M/H] as ``logZ``; ``co_ratio`` is required.
NewEra filenames label [M/H] as ``Z``; ``alpha`` selects a fixed slice.
For MPS-ATLAS, ``mps-atlas-set1`` and ``mps-atlas-set2`` fix different abundance and
mixing-length assumptions, with ``mps-atlas`` defaulting to Set 1.
``speclib`` does not interpolate C/O, alpha, or between distinct model-library
families or flavors (for example, MPS-ATLAS Set 1 and Set 2).

Native aliases
--------------

Prefer ``metallicity=`` in cross-library code. ``feh=`` is a native
alias for PHOENIX-ACES. ``mh=`` is a native alias
for NewEra (all selectors), SPHINX, and MPS-ATLAS. Passing ``mh=`` to PHOENIX-ACES
raises ``ValueError``. Supplying more than one coordinate keyword raises
``ValueError``, even when the values agree.

Passing ``feh=`` to NewEra, SPHINX, or MPS-ATLAS raises
``ValueError`` and directs the caller to ``metallicity=`` or ``mh=``.
No [Fe/H]-to-[M/H] conversion is performed.
Use ``metallicity_bds`` for grid constructors and ``metallicity_range`` for
``utils.download_newera_hsr_subset``. ``mh_bds``/``mh_range`` are native aliases
on [M/H] grids. Mixing range/bounds forms also raises ``ValueError``.
The NewEra helpers' old ``z``/``zscale`` storage-name keywords are deprecated
aliases of ``metallicity``.

Returned spectra expose ``meta["metallicity"]`` and
``meta["metallicity_type"]``. The numeric coordinate describes the selected
model plane for exact/nearest retrieval, including filename rounding,
or the requested coordinate for interpolation. Grid objects expose
``metallicities``, ``grid_metallicities``, ``metallicity_bds``, and
``metallicity_type``; ``meta["metallicities"]`` records the loaded axis values.
SEDs inherit the spectrum metadata; SEDGrid supports PHOENIX only.
Smitha products have neither metallicity metadata field.

Select a Smitha et al. surface component by stellar type and component:

.. code-block:: python

   penumbra = Spectrum.from_smitha2025("K0V", component="penumbra")

``G2V``, ``K0V``, and ``M0V`` and the quiet, spot, penumbra, and umbra
components are never interpolated. See :doc:`../models/smitha2025`.

Load Kostogryz et al. intensities with :class:`~speclib.SpecificIntensityGrid`
to select spectra by disk position. See
:doc:`specific_intensities` for the workflow and
:doc:`../models/kostogryz2026` for available models.

Additional accepted selectors
-----------------------------

``Spectrum.from_grid`` also accepts ``drift-phoenix`` and ``nextgen-solar`` as
compatibility paths that require users to arrange their caches. ``newera``
selects individual high-sampling-rate HDF5 files. ``newera_gaia``,
``newera_jwst``, and ``newera_lowres`` select the reduced archives described
on :doc:`../models/newera`.
