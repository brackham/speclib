Interpolation and boundaries
============================

Interpolation is linear at each wavelength in the three axes
:math:`T_\mathrm{eff}`, :math:`\log g`, and metallicity. It assumes that the
corner spectra share a wavelength axis. C/O (SPHINX), alpha enhancement
(NewEra), and the selected MPS-ATLAS set identify fixed slices or flavors and
are not interpolated.

Exact, nearest, and interpolated requests
-----------------------------------------

An exact available combination returns the stored model. For an off-grid
request:

* ``interpolate=True`` (the default) uses trilinear interpolation when every
  necessary corner exists;
* ``interpolate=False`` returns a nearest available combination without
  interpolation.

For regular axes such as those used by PHOENIX, nearest selection is performed
independently on each of the three coordinates. SPHINX and MPS-ATLAS choose
the nearest *actual* combination after scaling each axis by its typical grid
step. MPS-ATLAS searches only the selected set's archive index. Ties follow
NumPy's first minimum and should not be treated as a scientific model
selection rule.

Incomplete grids
----------------

PHOENIX has missing models at some combinations; NewEra explicitly excludes physically
unavailable static models; SPHINX V4 and each MPS-ATLAS set are indexed from
exact filenames.

For SPHINX, interpolation that needs a missing corner raises ``ValueError``
and explains which corner is missing. Use nearest retrieval or choose another
point; ``speclib`` does not fill that hole. For the reduced NewEra selectors,
``SpectralGrid.get_flux`` catches a missing trilinear corner and falls back to
the prebuilt nearest-neighbor interpolator. Consequently,
``interpolate=True`` can produce a nearest spectrum rather than a linearly
interpolated spectrum in a sparse region. The fallback is not used for
SPHINX. MPS-ATLAS likewise raises ``ValueError`` for a missing corner or
incompatible corner wavelength grid and never consults the other set. Other
regular selectors propagate a missing key error.

Bounds and extrapolation
------------------------

``SpectralGrid`` aligns constructor bounds outward to available axis values.
Only portions beyond the global library range generate a warning; ordinary
off-grid interior bounds are silently expanded to bracket the request. A
retrieval outside the resulting ``teff_bds``, ``logg_bds``, or ``feh_bds``
raises ``ValueError``. No grid retrieval extrapolates beyond loaded bounds.

``Spectrum.from_grid`` does not run the constructor clipping step. MPS-ATLAS
explicitly rejects a value beyond the selected set's indexed axes; for other
families, a value outside an axis eventually fails while finding bounds or
loading a file.
Validate requested coordinates against the appropriate model page and, for an
incomplete library, against the actual available combinations.
