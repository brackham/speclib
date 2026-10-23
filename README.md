# speclib

**Tools for working with stellar spectral libraries.**

`speclib` provides a lightweight Python interface for loading, manipulating, and analyzing stellar spectra and model grids. It includes utilities for photometric synthesis, spectral resampling, and SED construction using libraries such as PHOENIX.

Read the complete user guide, runnable tutorials, model library notes, and API
reference at **[speclib.readthedocs.io](https://speclib.readthedocs.io/)**.

---

## Installation

With Poetry (recommended):

```bash
git clone https://github.com/brackham/speclib.git
cd speclib
poetry env use python3.12
poetry install
```

Or with pip:

```bash
python -m pip install git+https://github.com/brackham/speclib.git@main
```

To install a local checkout with pip, run `python -m pip install .` from its
root. When dependencies change, reinstall `speclib` itself to refresh its
metadata; see [updating an existing installation](https://speclib.readthedocs.io/en/latest/installation.html#updating-an-existing-installation).

---

## Instrument resolution matching

On a loaded Speclib spectrum:

```python
observed = spectrum.observed_by("jwst/nirspec/g395h/f290lp")
```

The corresponding `SpectralGrid` method matches every model. Nine bundled
BOTS/S1600A1/SUB2048 profiles provide nominal wavelength-dependent resolution
and detector coverage offline. Native wavelength sampling is preserved by
default; detector gaps are explicit. This is Gaussian resolution matching, not
an observation simulation. See the [instrument-profile guide](docs/user_guide/instrument_profiles.rst)
and [scientific provenance](docs/user_guide/instrument_provenance.rst) for examples,
original calibration links and assumptions.

## Citation

If you use `speclib` in published work, please cite both:

1. the archived Zenodo release corresponding to the version of `speclib` used in your work ([doi:10.5281/zenodo.7868049](https://doi.org/10.5281/zenodo.7868049)), and
2. Rackham & de Wit (2024), [doi:10.3847/1538-3881/ad5833](https://doi.org/10.3847/1538-3881/ad5833).

The Zenodo citation provides a persistent software DOI, while Rackham & de Wit (2024) describes the scientific motivation and methodology. The contribution-credit policy is described in [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Requirements

* Python 3.11 to 3.13
* [astropy](https://www.astropy.org/)
* [specutils](https://specutils.readthedocs.io/)
* [synphot](https://synphot.readthedocs.io/)

---

## Quick example

```python
import astropy.units as u
import numpy as np

from speclib import Spectrum

wavelength = np.linspace(5000, 5100, 1001) * u.AA
flux = np.ones(wavelength.size) * u.erg / (u.s * u.cm**2 * u.AA)
spectrum = Spectrum(spectral_axis=wavelength, flux=flux)
sampled = spectrum.resample(np.linspace(5010, 5090, 81) * u.AA)
```

This example is entirely local. Loading a stellar-model spectrum may download
external data; see the [Quickstart](https://speclib.readthedocs.io/en/latest/quickstart.html)
and [model library guide](https://speclib.readthedocs.io/en/latest/user_guide/model_libraries.html)
before starting a large grid download.

### Custom library cache location

Downloaded spectral libraries are stored in `~/.speclib/libraries` by default.
Set the `SPECLIB_LIBRARY_PATH` environment variable or call
`speclib.utils.set_library_root("/path/to/cache")` to use a different location.

See the [model library reference](https://speclib.readthedocs.io/en/latest/models/index.html)
for supported spectra and their download, storage, and interpolation behavior.

---

## Contributing

Bug reports, documentation improvements, tests, and code contributions are welcome.
See the [contributor guide](CONTRIBUTING.md) for development commands and PR expectations,
or [open an issue](https://github.com/brackham/speclib/issues/new/choose) to discuss a bug,
feature, or spectral library.

---

## License

MIT © 2021–2026 Benjamin V. Rackham
