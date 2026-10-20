Scientific provenance of instrument profiles
============================================

Three ingredients have distinct sources: the numerical resolving-power curve,
the valid detector coverage, and Speclib's approximation to the LSF. Practical
examples are in :doc:`instrument_profiles`; contributor requirements are in
:doc:`../development/instrument_profiles`.

Original resolving-power data
-----------------------------

The `STScI NIRSpec dispersers page
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters>`_
distributes seven dispersion/resolution FITS tables used by the ETC. Its
instrument reference is `Jakobsen et al. (2022), A&A 661, A80
<https://doi.org/10.1051/0004-6361/202142663>`_
(`author preprint <https://arxiv.org/abs/2202.03305>`_). Speclib copies the linked
tables, not values digitized or inferred from that publication.

.. include:: ../_build/instrument_tables/sources.rst

NASA/STScI is the distributor. All seven headers identify ``SYSTEM=MODELING``
and ``PEDIGREE=GROUND``. G235H/G395H name Pierre Ferruit as author; the other
headers name a generation script, not a complete calibration-team author list.
These are nominal dispersion-derived prescriptions, not empirical in-flight
or BOTS point-source LSF measurements. ``quality='tabulated'`` identifies the
representation and does not imply empirical accuracy.

The source table contains ``WAVELENGTH`` (micrometers), ``DLDS`` (micrometers
per pixel) and dimensionless ``R`` in binary-table extension 1. STScI defines
a 2.2-pixel resolution element, approximately
:math:`R=\lambda/(2.2\,\mathrm{DLDS})`, for typically fully illuminated
IFU/MOS/200 mas fixed-slit apertures. This is not a measured Gaussian FWHM for
the wide S1600A1 slit. The FITS wavelength unit does not explicitly flag air
versus vacuum; Speclib treats the coordinates as vacuum without conversion.

Identification and processing
-----------------------------

``calibration_version`` labels the selected Speclib source snapshot, retrieved
2026-10-17. There is no stated formal calibration release version for these
attachments. Full SHA-256 pins identify their bytes. FITS creation timestamps
above are not last-revision dates: G235H/G395H retain 2016 timestamps, but their
histories record interpolation by G. M. Wahlgren on 2024-05-16, and the linked
STScI page announces their update on 2024-06-24. Attachment version ``1`` is not
a calibration version.

The :download:`conversion script <../../scripts/instruments/convert_nirspec.py>`
checks each original checksum, timestamp, row count, units, ordering and numerical
validity, then copies every ``WAVELENGTH``/``R`` pair. Source float32 values are
promoted to float64 and written with 17 significant digits to
``wavelength_um,resolving_power`` CSVs. Conversion does not fit, interpolate,
thin, or recompute R. Runtime evaluation linearly interpolates the published R
column in wavelength and never extrapolates it.

The original G235H/G395H R and dispersion columns are separately interpolated;
their maximum deviations from the 2.2-pixel ratio are about 0.403% and 0.836%.
The converter's 1% consistency check accommodates this source property; it is
not a physical-accuracy claim. Published R values remain untouched.

Some original intermediate sources cannot be independently retrieved: five
``LITREF`` values name internal as-built files; G235H/G395H name the technical
note ``ESA-JWST-SCI-NRS-TN-2016-024``, for which no public copy was located.
These are header identifiers, not unverifiable citations substituted for the
data. The directly linked official FITS tables are the reproducible inputs.
If attachment URLs move, locate the original filenames on the linked STScI
page and require the same checksums before accepting replacements.

Coverage and Gaussian approximation
-----------------------------------

The interval limits and gaps come separately from `BOTS wavelength ranges and
gaps, Table 1, SUB2048
<https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps>`_.
Speclib transcribes these approximate S1600A1 limits in ``profiles.toml``.
They are not derived from the R tables, and they vary with the observation's
wavelength solution and grating-wheel position. Other subarrays and multistripe
configurations on that page are not implemented.

Speclib adopts the tabulated :math:`\lambda/R` as **Gaussian FWHM** using the
existing source-centered log-wavelength convolution. This is a Speclib modeling
choice, not a claim that the source supplies Gaussian LSF measurements. It
assumes negligible intrinsic model widths and applies no deconvolution.
There is no full non-Gaussian LSF, throughput, detector response/pixel geometry,
noise, systematics, or observation-dependent resolution/wavelength solution.
Output sampling is separate: synphot's bin integration and sparse samples can
change apparent FWHMs. See :doc:`instrument_profiles` for sampling guidance.

Programmatic records and redistribution
---------------------------------------

.. code-block:: python

   from speclib import get_profile

   profile = get_profile("jwst/nirspec/g395h/f290lp")
   record = profile.provenance  # Fresh dictionary; no network request.
   print(record["url"], record["filename"], record["sha256"])
   print(record["fits_date"], record["history"], record["processing"])
   print(profile.coverage_source_url, profile.resolution_definition)

The :download:`packaged manifest <../../src/speclib/data/instruments/sources.json>`
is authoritative for original/output filenames and checksums, source headers,
URLs and conversion details. ``profiles.toml`` is authoritative for profile
configuration, coverage and adopted assumptions. Published tables are generated
from these records. Both files and the curves ship in wheels and sdists.

The `STScI content-use policy <https://www.stsci.edu/copyright>`_ permits use
under its public-domain-like NASA/STScI terms unless otherwise stated, requests
acknowledgement, and identifies exceptions for third-party restrictions. Review
of the source page and all seven headers found no contrary notice. This is the
redistribution basis, not a dataset-specific license or endorsement; Speclib's
code license does not relicense the source data. Attribution and the review date
are in :download:`NOTICE.md <../../src/speclib/data/instruments/NOTICE.md>`.
