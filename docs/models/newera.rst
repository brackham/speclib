PHOENIX/1D NewEra
=================

Family and version
------------------

NewEra is one PHOENIX/1D family with several derived spectral products.
``speclib`` targets the NewEra V3 spectra in **FDR release 3.5**,
`record 18108 <https://doi.org/10.25592/uhhfdm.18108>`_. V3 denotes the
model/spectral-product version; 3.5 denotes the repository release. The model
grid is described by `Hauschildt et al. (2025)
<https://doi.org/10.1051/0004-6361/202554171>`_. It contains LTE,
spherically symmetric atmospheres computed with updated atomic and molecular
line data.

The published overall coverage is 2300--12000 K, log g = 0.0--6.0, and
[M/H] = -4.0--+0.5. The paper describes a nominal temperature spacing of
100 K below 8000 K and 200 K above. The released V3 inventory instead has a
regular backbone of **2300--7000 K by 100 K, then 7200--12000 K by 200 K**;
``speclib`` uses this released sequence for interpolation/grid construction.
Log g spacing is 0.5 dex and metallicity spacing is 0.5 dex. For
-2.0 <= [M/H] <= 0.0, a subset has [alpha/Fe] from -0.2 to +1.2 in 0.2 dex
steps. The scientific grid is explicitly incomplete because some static
atmospheres are physically unavailable. Independent axis membership does not
establish that a model tuple exists.

Sparse native models at 3350, 5770, 6050, and 6060 K are available in the main
reduced products; HSR also includes the named 9602 K Vega model. Exact
``Spectrum.from_grid`` requests consult the selected product's inventory or
headers, including these special models. They are not added as global
interpolation planes.

Reduced flavors
---------------

All three selectors read the upstream Gaia-style text format: a header and
one flux row per model, with linear wavelength samples in nm and flux in
``W / (m2 nm)``. The following values come from the V3 files' own headers;
the step is a sampling interval, not a verified Gaussian FWHM or resolving
power.

.. list-table:: Which NewEra flavor should I use?
   :header-rows: 1
   :widths: 17 22 17 20 24

   * - Selector
     - Wavelength grid
     - V3 archive size
     - Intended use
     - Choose it when
   * - ``newera_gaia``
     - 300--1100 nm, 0.1 nm step (8001 samples)
     - 845.4 MB
     - Gaia DR4 spectral product
     - Your analysis is confined to the Gaia optical/near-IR range and you
       want the smallest archive.
   * - ``newera_jwst``
     - 600--28500 nm, 0.2 nm step (139501 samples)
     - 12.1 GB
     - JWST wavelength range and sampling product
     - You require broad near- to mid-infrared coverage, including JWST bands.
   * - ``newera_lowres``
     - 250--2500 nm, 0.01 nm step (225001 samples)
     - 18.2 GB
     - General reduced optical/near-IR spectra
     - You need denser linear sampling and UV-to-K-band coverage, but not the
       JWST product's long-wavelength reach.

The names describe upstream products; ``speclib`` does not apply an
instrument line-spread function when loading them. Do not equate the tabulated
step directly with physical resolution.

Cache and extraction
--------------------

.. code-block:: python

   from speclib import Spectrum, download_newera_grid

   download_newera_grid("newera_gaia")  # cache tarball; do not extract all
   spectrum = Spectrum.from_grid(
       4000, 4.5, metallicity=0.0,
       model_grid="newera_gaia",
       interpolate=False,
   )

Each flavor is cached in its own directory. By default
:func:`~speclib.download_newera_grid` retains the tarball without extraction.
The loader extracts the requested metallicity/alpha text member on demand.
For every NewEra selector, ``metallicity`` is native [M/H] and ``mh`` is a
native alias. No [Fe/H]-to-[M/H] conversion is performed. Returned spectra
record ``meta["metallicity"]`` and ``meta["metallicity_type"] = "mh"``.
The upstream ``Z`` filename label is not a metal mass fraction.

The HSR filter helper uses ``metallicity_range``:

.. code-block:: python

   from speclib import utils

   utils.download_newera_hsr_subset(
       teff_range=(4000, 4000),
       logg_range=(4.5, 4.5),
       metallicity_range=(-0.5, 0.0),
       alpha_range=(0.0, 0.0),
   )

``download_newera_grid`` downloads a whole reduced archive and does not
filter parameter ranges.
Use ``extract="all"`` only when the extra storage and extraction time are
intentional. ``overwrite=True`` clears that flavor's cache before fetching a
fresh archive.

The ``newera`` selector is different: it obtains individual high-sampling-rate
HDF5 models using the download links in ``list_of_available_NewEraV3_models.txt``
from record 18108 (HSR records 16738 and 17670). Native availability also
includes the explicit file catalog of the current
`additional-model record 17936 <https://doi.org/10.25592/uhhfdm.17936>`_. This
replaces the additional-model text list removed in release 3.5; there is no
fallback to an older repository release. Main V3 spectra take precedence
where the two inventories overlap.

The public ``download_newera_hsr_subset`` utility
is not exported at package root and its unconstrained collection is
approximately 4.5 TB. Prefer one of the reduced flavors unless a scientific
requirement specifically demands HSR data. This downloader filters actual
native tuples by inclusive parameter ranges, including special models.

HSR and reduced availability are distinct. The main Gaia/JWST/LowRes archives
share an alpha-zero model set; their archive loaders do not automatically
include the separate ``add001`` reduced files in record 18108. These supplemental
products add [M/H] = +0.5 combinations, not new temperature planes. The Gaia
archive filename retains the upstream ``v3.4`` suffix in release 3.5.
``SPECLIB_NEWERA_RECORD_ID`` still overrides the main release record; a custom
record uses its own inventory without merging the canonical supplemental
catalog. Existing unchanged V3 archives can be reused from the cache.

Interpolation and caveats
-------------------------

For reduced grids, interpolation fixes ``alpha`` and is trilinear in
Teff, log g, and metallicity when all corners exist; in a ``SpectralGrid``, a
missing corner falls back to nearest-neighbor evaluation.

Reduced grid loaders warn for nonzero alpha because alpha-enhanced reduced
products are not yet reliably supported. The upstream alpha coverage is only
a subset even within its stated metallicity range. Do not interpret a request
that merely lies on a declared axis as proof that a file exists.

For publication, cite Hauschildt et al. (2025) and the FDR data release, in
addition to :doc:`../citation` for ``speclib``.
