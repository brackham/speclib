Instrument profiles
===================

See :doc:`../user_guide/instrument_profiles` for scientific assumptions,
coverage, detector gaps, and the return types of ``observed_by()``.

.. autoclass:: speclib.InstrumentProfile
   :members: resolving_power_at, provenance

Public profile attributes include:

* ``mode`` and ``configuration``: canonical identifier and observing setup.
* ``wavelength``, ``resolving_power`` and ``valid_intervals``: immutable sampled
  calibration and detector coverage (wavelength quantities in micrometers).
* ``source``, ``source_url`` and ``coverage_source_url``: scientific attribution,
  resolving-power documentation and the separate coverage reference.
* ``calibration_version``, ``resolution_definition``, ``quality`` and
  ``assumptions``: source identification, adopted width convention and limitations.
* ``provenance_ref``: optional bundled ``manifest.json#key`` reference;
  ``provenance`` returns a detached dictionary containing original-data URLs,
  checksums, source identifiers and processing details. Unreferenced custom
  profiles return an empty dictionary.

.. autofunction:: speclib.get_profile

.. autofunction:: speclib.list_instrument_profiles

See :doc:`../user_guide/instrument_provenance` for the source-record specification
and :doc:`../development/instrument_profiles` for contributor requirements.

NIRCam shorthand ``jwst/nircam/grismr/<filter>`` selects the canonical
``jwst/nircam/tsgrism/modulea/grismr/<filter>`` setup. Grism C, Module B and
wide-field slitless configurations are unsupported.

Instrument matching returns ordinary spectra/grids with explicit
``valid_intervals`` metadata. See :attr:`speclib.Spectrum.valid_intervals` and
:attr:`speclib.SpectralGrid.valid_intervals`.

MIRI shorthand ``jwst/miri/lrs/slitless`` selects the canonical
``jwst/miri/lrs/slitless/slitlessprism`` profile: P750L, conventional
SLITLESSPRISM time series only. IP/IPS, fixed slit, WFSS and MRS are unsupported.
