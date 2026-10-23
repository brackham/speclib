Installation
============

``speclib`` supports Python 3.11, 3.12, and 3.13. Install the current
development code from GitHub into an isolated environment:

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install git+https://github.com/brackham/speclib.git@main

For development, clone the repository and let Poetry create the environment:

.. code-block:: console

   git clone https://github.com/brackham/speclib.git
   cd speclib
   poetry env use python3.12
   poetry install

Alternatively, install a local checkout into your active environment with
``python -m pip install .`` from the repository root. Use
``python -m pip install -e .`` for an editable pip installation.

The main runtime dependencies include Astropy, specutils, synphot, NumPy,
h5py, and pooch. Matplotlib and Jupyter are development dependencies used by
the tutorials.

Updating an existing installation
---------------------------------

Pulling new code does not refresh installed dependency metadata, even for
an editable installation. After pulling dependency changes, rerun
``poetry install`` in a Poetry checkout, or ``python -m pip install .``
(``python -m pip install -e .`` for editable installs) in a pip checkout.
Updating ``specutils`` alone leaves ``speclib``'s recorded requirements unchanged.

To refresh a GitHub installation, including when the version number has
not changed, reinstall explicitly:

.. code-block:: console

   python -m pip install --force-reinstall git+https://github.com/brackham/speclib.git@main

This also reinstalls dependencies. Run these checks with the same Python
environment that runs your code (use ``poetry run python`` for Poetry):

.. code-block:: console

   python -m pip check
   python -c "from importlib.metadata import requires, version; print(version('speclib')); print('\n'.join(requires('speclib')))"

Current development code should declare ``specutils>=2,<3``. An older b12
installation may declare ``specutils>=1.9.1,<2``; reinstall from current
source to replace that metadata. Restart a running notebook kernel after
reinstalling.

Model data and the cache
------------------------

Constructing a :class:`~speclib.Spectrum` in memory needs no network access.
Loading a library model may fetch data from its upstream archive. By default,
downloaded files live under ``~/.speclib/libraries``. To keep large model data
on another volume, set the environment variable before importing ``speclib``:

.. code-block:: console

   export SPECLIB_LIBRARY_PATH=/data/stellar-models

For the current Python process, the equivalent configuration is:

.. code-block:: python

   from speclib.utils import set_library_root

   set_library_root("/data/stellar-models")

Calling ``set_library_root(None)`` restores selection from the environment
variable or default path. The selected path is returned by
:func:`speclib.utils.get_library_root`.

Data implications differ by model family:

* PHOENIX spectra and the shared wavelength array are downloaded on demand;
  :func:`~speclib.download_phoenix_grid` attempts the complete declared grid
  and can require substantial time and storage.
* :func:`~speclib.download_sphinx_grid` downloads a checksummed, approximately
  161 MB V4 archive, retains it, and extracts the spectrum text files.
* :func:`~speclib.download_mps_atlas_grid` downloads one checksummed v3 archive
  of about 9.5 GB into ``mps-atlas/set1`` or ``mps-atlas/set2``. The archive is
  retained and requested disk-integrated members are read directly from it.
* ``download_smitha2025_spectra("K0V")`` prefetches one stellar type, while
  ``download_smitha2025_spectra()`` prefetches all three. This release is three
  small spectral files rather than a large grid archive; they are cached in
  ``smitha2025`` and checked against the Edmond v1.0 sizes and MD5 hashes.
* Kostogryz intensity models are downloaded when first requested and cached
  under ``kostogryz2026``. Use ``download_kostogryz2026_spectra`` to prefetch
  one model or the full library.
* :func:`~speclib.download_newera_grid` caches one reduced-resolution archive
  (about 845 MB to 18.2 GB, depending on flavor). It does not extract the
  archive by default; loaders extract a requested metallicity file on demand.

See :doc:`models/index` before downloading. In particular, the separate
NewEra high-sampling-rate collection is approximately 4.5 TB and is not the
normal workflow for reduced products.
