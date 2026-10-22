# NIRSpec calibration attribution and redistribution

These numerical files reproduce every WAVELENGTH and R value from STScI's
public NIRSpec dispersion FITS tables, without fitting, thinning, or recalculating
R. They are not measured BOTS point-source line-spread functions.

Source distributor: NASA/STScI,
[NIRSpec Dispersers and Filters](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters).
Instrument reference: [Jakobsen et al. (2022), A&A 661, A80](https://doi.org/10.1051/0004-6361/202142663)
([author preprint](https://arxiv.org/abs/2202.03305)). The FITS headers identify
ground/modeling products; authorship/header identifiers are retained in the manifest.

Coverage: NASA/STScI,
[NIRSpec BOTS Wavelength Ranges and Gaps, Table 1](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps).

Retrieved 2026-10-17. Exact download URLs, SHA-256 checksums, FITS creation dates,
row counts, and modification history are in sources.json. JDox notes a
2024-06-24 update to G235H/G395H; those FITS files record interpolation on
2024-05-16 while retaining their original 2016 creation dates. An attachment
URL's version number alone is not a calibration version.

Redistribution basis: [STScI Institute Content Use Policy](https://www.stsci.edu/copyright)
(reviewed 2026-10-19). STScI asserts no copyright
on material prepared for NASA unless specifically stated and permits use as in
the public domain, requesting acknowledgement of NASA and STScI. The source
page and all seven FITS headers were inspected; no additional copyright or
redistribution restrictions are stated. The numerical conversions and coverage
metadata are redistributed on this basis, with this acknowledgement retained.
The policy includes exceptions for third-party restrictions; this review found
no contrary notice but does not supply a separate dataset-specific license grant.
No endorsement by NASA or STScI is implied. Speclib's code license does not
relicense third-party material.

# Phase 2: STIS, WFC3 and NIRISS/SOSS

Source review and retrieval: 2026-10-20. `phase2_sources.json` contains direct
links, source/output SHA-256 checksums where applicable, equations, coefficients
and processing assumptions. These files acknowledge NASA/STScI and the
NIRISS/SOSS instrument team; they imply no endorsement.

STIS numerical widths are reduced from the public model LSF tables on the
[STScI spectral-resolution page](https://www.stsci.edu/hst/instrumentation/stis/performance/spectral-resolution).
Only the 52x2.0 column is used. Supplementary widths and mean dispersions are
factual numerical parameters from the linked STIS handbook tables. The three
original text headers and the source pages contain no contrary copyright or
redistribution notice. The reviewed basis is the
[STScI content-use policy](https://www.stsci.edu/copyright) for NASA-prepared
material, with the same exceptions and attribution conditions described above.
These are model profiles, not measured in-flight LSFs; the original asymmetric
LSF wings are not reproduced by the Gaussian approximation.

WFC3 curves evaluate a nominal equation anchored to the numerical reference
resolving powers in
[WFC3 Handbook Table 8.1](https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview).
SOSS curves evaluate separate order-specific nominal equations using the
measured CV3 dispersions and two-pixel convention in
[Albert et al. (2023), PASP 135, 075001](https://doi.org/10.1088/1538-3873/acd7a3)
([author preprint](https://arxiv.org/abs/2306.04572)), Table 2 and Section 5.3.
The source prescriptions are acknowledged; these CSVs are original numerical
evaluations, with no publication text or figures redistributed. Useful SOSS
coverage also follows the
[STScI calibration-status guidance](https://jwst-docs.stsci.edu/jwst-calibration-status/niriss-calibration-status/niriss-soss-calibration-status).
Speclib's code license does not relicense third-party material.


# Phase 3A: NIRCam Module A Grism R time series

Reviewed 2026-10-21. `nircam_grismr.csv` is an original numerical evaluation of
the resolving-power polynomial in
[Greene et al. (2017), JATIS 3(3), 035001, Section 3.2 / Figure 2](https://doi.org/10.1117/1.JATIS.3.3.035001)
([original paper at STScI](https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JATIS_3_3_035001.pdf)),
also reproduced in
[STScI NIRCam Grisms, Figure 1](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-grisms).
Acknowledge Greene et al., NASA/STScI and the NIRCam instrument team.
The source is a nominal optical-model point-source prediction, not empirical
in-flight LSF data. No publication text, figure, simulated PSF or FITS file is
redistributed. The factual coefficients and units are the reproducible input;
`nircam_sources.json` identifies the output checksum and processing.

Coverage facts come from STScI Table 4 and the
[grism time-series documentation](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series).
Speclib's inward edge cuts preserve convolution support in 2.4-5.0 um and are
not new STScI calibration limits. The
[STScI content-use policy](https://www.stsci.edu/copyright) was reviewed;
the redistribution basis is original numerical evaluation of published factual
parameters with attribution. This does not assert that the SPIE publication
or figures are public domain or relicensed under Speclib's code license.
No author, NASA or STScI endorsement is implied.


## MIRI LRS slitless (reviewed 2026-10-22)

The MIRI CSV is Speclib's original evaluation of approximate factual resolving
powers and the explicit linear design trend published by NASA/STScI and
Kendrew et al. (2015), PASP 127, 623 (https://doi.org/10.1086/682255), also
consistent with the nominal slitless description in Bouwman et al. (2023),
PASP 135, 038002 (https://doi.org/10.1088/1538-3873/acbc49).
Acknowledge these authors, NASA/STScI and the MIRI instrument team.
The STScI content-use policy (https://www.stsci.edu/copyright) was reviewed;
no contrary restriction was found for these factual numerical descriptions.
No paper text, figures, CRDS calibration files or measured LSFs are distributed.
This does not relicense publisher material under Speclib's code license, claim
public-domain status for third-party material, or imply endorsement.
`miri_sources.json` records the precise interpretation and reproduction path.
