Installation
============

``speclib`` supports Python 3.11, 3.12, and 3.13. Install the current
development release from GitHub into an isolated environment:

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install git+https://github.com/brackham/speclib.git

For development, clone the repository and let Poetry create the environment:

.. code-block:: console

   git clone https://github.com/brackham/speclib.git
   cd speclib
   poetry install

The main runtime dependencies include Astropy, specutils, synphot, NumPy,
h5py, and pooch. Matplotlib and Jupyter are development dependencies used by
the tutorials.

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
* :func:`~speclib.download_newera_grid` caches one reduced-resolution archive
  (about 845 MB to 18.2 GB, depending on flavor). It does not extract the
  archive by default; loaders extract a requested metallicity file on demand.

See :doc:`models/index` before downloading. In particular, the separate
NewEra high-sampling-rate collection is approximately 4.5 TB and is not the
normal workflow for reduced products.

Build the documentation
-----------------------

From a development checkout:

.. code-block:: console

   poetry run sphinx-build -W --keep-going -b html docs docs/_build/html

The notebooks are executed during this build. All committed tutorials are
offline and deliberately small.
