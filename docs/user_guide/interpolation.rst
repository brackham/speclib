Interpolation and boundaries
============================

Retrieve an interpolated or nearest model from an existing grid:

.. code-block:: python

   interpolated = grid.get_spectrum(3050, 4.25, metallicity=0.0)
   nearest = grid.get_spectrum(3050, 4.25, metallicity=0.0, interpolate=False)

Interpolation is linear at each wavelength in the three axes
:math:`T_\mathrm{eff}`, :math:`\log g`, and metallicity. It assumes that the
corner spectra share a wavelength axis. C/O (SPHINX), alpha enhancement
(NewEra), and the selected MPS-ATLAS set identify fixed slices or flavors and
are not interpolated.

``metallicity`` always means the selected library's native coordinate:
[Fe/H] for PHOENIX-ACES, [M/H] for NewEra, SPHINX, and MPS-ATLAS.
Speclib does not convert between [Fe/H] and [M/H].

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
NumPy's first minimum; choose between tied models using your scientific
criteria.

Incomplete grids
----------------

PHOENIX has missing models at some combinations; NewEra explicitly excludes physically
unavailable static models; SPHINX V4 and each MPS-ATLAS set are indexed from
exact filenames.

For SPHINX, interpolation that needs a missing corner raises ``ValueError``
and explains which corner is missing. Use nearest retrieval or choose another
point with all required corners. For the reduced NewEra selectors,
``Spectrum.from_grid`` loads only the required native regular-backbone models
and raises ``ValueError`` if a corner is missing. Interpolated requests use
the native corner models. Exact special models remain available. ``SpectralGrid.get_flux`` catches a missing trilinear corner and falls
back to the prebuilt nearest-neighbor interpolator. In a sparse region,
this fallback returns the nearest model even with ``interpolate=True``.
SPHINX and MPS-ATLAS raise ``ValueError`` for a missing corner; MPS-ATLAS
also requires compatible wavelength grids within the selected set. Other
regular selectors propagate a missing key error.

Bounds and extrapolation
------------------------

``SpectralGrid`` aligns constructor bounds outward to available axis values.
Only portions beyond the global library range generate a warning; interior
bounds expand to bracket the request. A
retrieval outside the resulting ``teff_bds``, ``logg_bds``, or ``metallicity_bds``
raises ``ValueError``. No grid retrieval extrapolates beyond loaded bounds.

``Spectrum.from_grid`` does not run the constructor clipping step. MPS-ATLAS
explicitly rejects a value beyond the selected set's indexed axes. Reduced
NewEra rejects metallicities outside its interpolation range with
``FileNotFoundError``, except when an exact one-decimal native lookup succeeds.
For Teff/log g and other families,
a value outside an axis can fail while finding bounds or loading a file.
Validate requested coordinates against the appropriate model page and, for an
incomplete library, against the actual available combinations.
