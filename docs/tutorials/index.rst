Tutorials and examples
======================

These notebooks are the source for both the rendered pages and downloads.
MyST-NB executes all four during the Sphinx build, so their displayed figures
and results track the source checkout. They use only data created in memory,
packaged filters, or a temporary explicitly synthetic grid; none accesses the
network or downloads a model archive.

.. list-table:: Notebook downloads
   :header-rows: 1

   * - Tutorial
     - What it demonstrates
     - Original notebook
   * - Quickstart
     - Create, inspect, broaden, resample, filter, and plot a spectrum.
     - :download:`quickstart.ipynb <quickstart.ipynb>`
   * - Spectral grids
     - Construct a temporary synthetic grid; inspect axes and combinations;
       compare exact, nearest, and interpolated flux; handle bounds.
     - :download:`spectral_grids.ipynb <spectral_grids.ipynb>`
   * - Spectral resolution
     - Compare constant wavelength FWHM, constant resolving power, and sampled
       wavelength-dependent resolving power.
     - :download:`spectral_resolution.ipynb <spectral_resolution.ipynb>`
   * - Synthetic photometry
     - Apply packaged filters, build an SED, and convert magnitude to flux with
       uncertainty.
     - :download:`synthetic_photometry.ipynb <synthetic_photometry.ipynb>`

.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart
   spectral_grids
   spectral_resolution
   synthetic_photometry
