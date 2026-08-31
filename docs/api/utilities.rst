Download and cache utilities
============================

Package-level download helpers
------------------------------

.. currentmodule:: speclib

.. autofunction:: download_phoenix_grid

.. autofunction:: download_sphinx_grid

.. autofunction:: download_mps_atlas_grid

.. autofunction:: download_newera_grid

.. autofunction:: download_file

Cache location
--------------

These two helpers are intentionally called through ``speclib.utils`` and are
documented because they define where users store model libraries.

.. currentmodule:: speclib.utils

.. autofunction:: get_library_root

.. autofunction:: set_library_root

Wavelength conversion
---------------------

.. autofunction:: vac2air

.. autofunction:: air2vac
