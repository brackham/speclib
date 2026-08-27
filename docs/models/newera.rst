PHOENIX/1D NewEra
=================

Family and version
------------------

NewEra is one PHOENIX/1D family with several derived spectral products.
``speclib`` targets the V3 spectra and V3.4 reduced archives in
`FDR Hamburg record 17935 <https://doi.org/10.25592/uhhfdm.17935>`_. The model
grid is described by `Hauschildt et al. (2025)
<https://doi.org/10.1051/0004-6361/202554171>`_. It contains LTE,
spherically symmetric atmospheres computed with updated atomic and molecular
line data.

The published overall coverage is 2300--12000 K, log g = 0.0--6.0, and
[M/H] = -4.0--+0.5. Temperature spacing is 100 K below 8000 K and 200 K above;
log g spacing is 0.5 dex and metallicity spacing is 0.5 dex. For
-2.0 <= [M/H] <= 0.0, a subset has [alpha/Fe] from -0.2 to +1.2 in 0.2 dex
steps. The scientific grid is explicitly incomplete because some static
atmospheres are physically unavailable.

Reduced flavors
---------------

All three selectors read the upstream Gaia-style text format: a header and
one flux row per model, with linear wavelength samples in nm and flux in
``W / (m2 nm)``. The following values come from the V3.4 files' own headers;
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
       4000, 4.5, 0.0,
       model_grid="newera_gaia",
       interpolate=False,
   )

Each flavor is cached in its own directory. By default
:func:`~speclib.download_newera_grid` retains the tarball without extraction.
The loader extracts the requested metallicity/alpha text member on demand.
Use ``extract="all"`` only when the extra storage and extraction time are
intentional. ``overwrite=True`` clears that flavor's cache before fetching a
fresh archive.

The ``newera`` selector is different: it obtains individual high-sampling-rate
HDF5 models from the V3 list. The public ``download_newera_hsr_subset`` utility
is not exported at package root and its unconstrained collection is
approximately 4.5 TB. Prefer one of the reduced flavors unless a scientific
requirement specifically demands HSR data.

Interpolation and caveats
-------------------------

``speclib`` currently declares a 100 K temperature axis all the way to
12000 K for these selectors, while the published grid uses 200 K spacing above
8000 K. Treat actual files—not the declared axis—as authoritative in that
regime. For reduced grids, interpolation fixes ``alpha`` and is trilinear in
Teff, log g, and metallicity when all corners exist; in a ``SpectralGrid``, a
missing corner falls back to nearest-neighbor evaluation.

Reduced grid loaders warn for nonzero alpha because alpha-enhanced reduced
products are not yet reliably supported. The upstream alpha coverage is only
a subset even within its stated metallicity range. Do not interpret a request
that merely lies on a declared axis as proof that a file exists.

For publication, cite Hauschildt et al. (2025) and the FDR data release, in
addition to :doc:`../citation` for ``speclib``.
