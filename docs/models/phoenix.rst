PHOENIX
=======

Product
-------

``model_grid="phoenix"`` uses the Göttingen
**PHOENIX-ACES-AGSS-COND-2011-HiRes** synthetic spectra described by
`Husser et al. (2013) <https://doi.org/10.1051/0004-6361/201219058>`_.
The published library contains spherical stellar atmospheres and vacuum
wavelength spectra from 500 Å to 5.5 micrometers. Its characteristic
native resolution is :math:`R=500{,}000` in the optical and near-infrared,
:math:`R=100{,}000` in the infrared, and 0.1 Å sampling in the UV.

The high-resolution FITS flux arrays contain stellar surface
:math:`F_\lambda` in ``erg / (s cm2 cm)`` and share a separately downloaded
wavelength array. ``speclib`` converts them to
``erg / (s cm2 Å)`` and Å. It does not apply radius/distance
dilution.

Parameter axes exposed by speclib
---------------------------------

The current selector declares:

* :math:`T_\mathrm{eff}` = 2300--7000 K in 100 K steps and 7200--12000 K in
  200 K steps;
* :math:`\log g` = 0.0--6.0 in 0.5 dex steps (cgs);
* [Fe/H] = -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0.

Some parameter combinations are missing, including some low-gravity models.
The ``phoenix`` loader does not expose the upstream alpha-enhanced subgrids.

Access and cache behavior
-------------------------

.. code-block:: python

   from speclib import Spectrum

   spectrum = Spectrum.from_grid(
       4000, 4.5, metallicity=0.0,
       model_grid="phoenix",
       interpolate=False,
   )

The wavelength file and requested spectrum are fetched from the Göttingen
PHOENIX server when absent and cached under ``<library root>/phoenix``.
Interpolated requests may fetch as many as eight corner spectra.
``metallicity`` is native [Fe/H]; ``feh`` remains a warning-free native alias.
Returned spectra record ``meta["metallicity"]`` and
``meta["metallicity_type"] = "feh"``. No abundance conversion is performed.
:func:`~speclib.download_phoenix_grid` loops over every combination declared
above and skips files that cannot be retrieved. For individual models,
``Spectrum.from_grid`` downloads files as needed.

The filename convention is
``lteTTTTT-G.G±Z.Z.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits``; solar metallicity
uses ``-0.0`` in upstream filenames.

Scientific cautions
-------------------

Interpolation in ``speclib`` is linear in flux across Teff, log g, and [Fe/H];
it is distinct from any upstream spectra that the PHOENIX authors themselves
interpolated during grid production. Missing corner files can prevent a
``speclib`` interpolation. Choose broadening much wider than the intrinsic
source lines: the resolution methods assume negligible input widths and
apply no deconvolution.

For publication, cite Husser et al. (2013) as the underlying library in
addition to :doc:`../citation`.
