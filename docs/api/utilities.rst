Download and cache utilities
============================

Package-level download helpers
------------------------------

.. currentmodule:: speclib

.. autofunction:: download_phoenix_grid

.. autofunction:: download_sphinx_grid

.. autofunction:: download_mps_atlas_grid

.. autofunction:: download_smitha2025_spectra

.. autofunction:: download_kostogryz2026_spectra

.. autofunction:: download_newera_grid

.. autofunction:: download_file

Cache location
--------------

These two helpers are intentionally called through ``speclib.utils`` and are
documented because they define where users store model libraries.

.. currentmodule:: speclib.utils

.. autofunction:: get_library_root

.. autofunction:: set_library_root

Native model selection and filtering
------------------------------------

These helpers use ``metallicity`` or ``metallicity_range`` to select native
[M/H] directly, without an abundance conversion.

.. autofunction:: download_newera_hsr_subset

.. autofunction:: download_newera_file

.. autofunction:: load_newera_wavelength_array

.. autofunction:: load_newera_flux_array

.. autofunction:: load_sphinx_spectrum

.. autofunction:: load_mps_atlas_spectrum

Wavelength conversion
---------------------

.. autofunction:: vac2air

.. autofunction:: air2vac
