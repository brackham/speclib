Spectra
=======

A :class:`~speclib.Spectrum` represents one flux-density array aligned with a
spectral coordinate. It subclasses :class:`specutils.Spectrum` (specutils 2.x),
so standard spectrum attributes and slicing are available; ``speclib`` adds model
loading, resampling, regularization, binning, and spectral broadening.

Units and model loading
-----------------------

Constructors accept ``spectral_axis`` and ``flux`` as quantities with units. A
spectrum returned by :meth:`speclib.Spectrum.from_grid` is sorted in ascending
wavelength and converted to Å and
``erg / (s cm2 Å)``, regardless of the source product's storage units.
The conversion does not change a source product's air/vacuum convention.
PHOENIX products are stellar surface flux densities tabulated at vacuum
wavelengths; no stellar radius or distance scaling is applied.

Use explicit keywords when constructing a spectrum:

.. code-block:: python

   spectrum = Spectrum(flux=flux, spectral_axis=wavelength)

For multidimensional containers, an explicit spectral coordinate defaults to
the last flux dimension, and FITS WCS inputs move the spectral dimension last.
These defaults preserve the existing ``speclib`` convention and can be
overridden with ``spectral_axis_index`` and ``move_spectral_axis``. The
``speclib`` processing methods operate on one-dimensional spectra.

``wl_min`` and ``wl_max`` crop during loading. Current implementation uses
strict interior samples, so samples equal to either supplied endpoint are
excluded. ``wavelength=`` then resamples the cropped spectrum.

Operations and object state
---------------------------

Resampling, regularization, wavelength selection, and broadening retain the
``speclib.Spectrum`` class. Inherited operations follow specutils behavior;
arithmetic operators can return an upstream ``specutils.Spectrum`` without
the additional ``speclib`` convenience methods.

The following methods return new objects and do not modify the input:

* :meth:`~speclib.Spectrum.resample` uses synphot to evaluate flux density on
  a wavelength axis with units supplied by the user. By default it does not
  taper outside the source range; request ``taper=True`` only when zero
  throughput beyond the endpoints is the intended boundary assumption.
* :meth:`~speclib.Spectrum.regularize` creates a linearly spaced wavelength
  axis, using the smallest input interval when ``delta_lambda`` is omitted,
  and calls ``resample``.
* :meth:`~speclib.Spectrum.bin` integrates each requested interval and returns
  a :class:`~speclib.BinnedSpectrum` with ``center``, ``width``, ``lower``,
  ``upper``, and mean flux-density arrays.
* The three resolution methods return a new spectrum on a copied input axis;
  see :doc:`resolution`.
