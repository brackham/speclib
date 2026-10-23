Installation
============

``speclib`` supports Python 3.11, 3.12, and 3.13. Install the current
development code from GitHub into an isolated environment:

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install git+https://github.com/brackham/speclib.git@main

For a local checkout, run ``python -m pip install .`` from the repository root
(or ``python -m pip install -e .`` for an editable installation).
See :doc:`contributing` for development with Poetry.

To run the tutorials locally, also install Matplotlib and Jupyter:

.. code-block:: console

   python -m pip install matplotlib notebook

Updating an existing installation
---------------------------------

To update from GitHub, run this in the environment where you use ``speclib``:

.. code-block:: console

   python -m pip install --force-reinstall git+https://github.com/brackham/speclib.git@main

For a local checkout using Poetry, pull the latest code and reinstall:

.. code-block:: console

   git pull
   poetry install

For a pip checkout, run ``python -m pip install .`` after pulling
(or ``python -m pip install -e .`` for an editable installation).
Restart any running notebook kernels after updating.

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

Downloads range from small individual files to archives of several gigabytes.
The :doc:`model-library reference <models/index>` lists download sizes, cache
behavior, and prefetch helpers for each library.
