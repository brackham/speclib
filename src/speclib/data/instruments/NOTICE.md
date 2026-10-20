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
