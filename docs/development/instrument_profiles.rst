Instrument-profile provenance standard
======================================

Every bundled profile must connect its numerical prescription to authoritative,
clickable sources. Keep the record small and specific to the source type; do not
invent a file, version, precision or measurement pedigree to fill a field.
This checklist applies to bundled STIS, WFC3, NIRISS and NIRSpec profiles and
future NIRCam, MIRI and other profiles.

Minimum record
--------------

* **Configuration:** use ``mode`` and ``configuration`` for telescope, instrument,
  disperser/filter and relevant slit/aperture/subarray/order. Do not silently
  substitute a different setup.
* **Resolution source:** identify the responsible institution/team where known,
  the source kind (nominal prediction, dispersion model, measured LSF, analytic
  prescription, or another stated kind), and an authoritative documentation,
  dataset or publication URL. ``source``/``source_url`` give the human summary;
  the manifest records the original dataset URL or equation reference.
* **Coverage source:** use ``coverage_source_url`` for the separate authority
  supporting all valid intervals/gaps. Document approximations, configuration
  dependence and any limits chosen by Speclib rather than the source.
* **Identification:** ``calibration_version`` must identify the selected source
  or snapshot. Record available filenames, versions, dates and stable identifiers
  in the manifest. State unknowns explicitly; retrieval dates and FITS creation
  dates must not be presented as calibration release versions.
* **Physical meaning:** specify the supplied width/LSF definition, units,
  illumination/sampling convention and measurement or prediction pedigree.
  ``resolution_definition`` states how Speclib interprets it; a sampled table
  (``quality='tabulated'``) need not be an empirical measurement.
* **Processing:** record original columns, units and selection, conversions,
  interpolation, fitted coefficients/adopted constants and approximations.
  Separate source processing from Speclib's conversion and runtime evaluation.
  Use ``assumptions`` for the supported scientific limitations.
* **Reproducibility:** supply a conversion script and source/output checksums
  for distributed data files. For an analytic prescription, record the equation,
  coefficients, units, validity domain and DOI/section instead; no fictitious
  source file or checksum is required. For measured LSF data, retain their
  normalization and observational conditions before any R/Gaussian reduction.
* **Redistribution:** cite the applicable policy/license URL, attribution and
  review basis. Flag missing permission or restrictions before bundling data.

One authoritative record per fact
---------------------------------

Use the existing ``InstrumentProfile`` fields and a JSON manifest. A profile's
optional ``provenance_ref="manifest.json#source_key"`` selects a record bundled
under ``speclib/data/instruments``. The public ``profile.provenance`` property
combines the manifest's optional ``defaults``, ``retrieved`` date and the selected
``sources`` entry into a detached dictionary, offline. Custom user profiles may
omit this locator; bundled profiles must provide a complete record.

Use descriptive flat record keys, following NIRSpec: ``institution``,
``source_kind``, ``documentation_url``/``publication_url``, ``source_definition``,
``processing`` and ``reproduction_script``. File sources additionally have
``url``, ``filename``, ``sha256``, ``output_file`` and ``output_sha256``; analytic
sources can instead have ``equation`` and ``coefficients``. Source-specific
header fields are optional. Record output checksums whenever distributing a
derived numerical table, including one sampled from an analytic prescription.
A manifest can share defaults among related sources;
different instruments can use separate manifests. No new provenance class or
unrelated fields are needed.

Keep profile configuration/coverage in the TOML registry and original-data facts
in the manifest. Reference shared source records rather than copying citations
into each filter entry. Generate documentation tables from those records, as
``docs/_instrument_tables.py`` does for NIRSpec. Every scientific citation must
render as a hyperlink, including original-data links distinct from descriptive
pages. Use persistent identifiers when available; describe retrieval by filename
when attachment URLs are unstable.

Review and reproduction checklist
---------------------------------

#. Open every source and check its actual contents, configuration and definition;
   record unavailable intermediates or uncertain authorship/version information.
#. Reproduce output files without network access from pinned inputs. Preserve
   raw values unless a documented transformation is scientifically necessary.
#. Check numerical integrity once, plus representative line widths, boundaries,
   gaps, sampling, metadata ownership and grid consistency. Keep extensive
   independent scientific campaigns separate from routine regression tests.
#. Inspect ``get_profile(mode).provenance`` and the rendered configuration/source
   tables. Confirm links, offline wheel/sdist access, attribution and installation.
#. Build Sphinx with warnings as errors and execute the lightweight tutorials;
   add concise user examples. Change the changelog only when requested.

For bundled curves, follow the :download:`reproduction procedures
<../../scripts/instruments/README.md>` and the
:doc:`scientific reference <../user_guide/instrument_provenance>`. The source
manifests are downloadable from that reference; reproducing data must not silently
replace checksum pins when upstream attachments change.
