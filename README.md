# speclib

**Tools for working with stellar spectral libraries.**

`speclib` provides a lightweight Python interface for loading, manipulating, and analyzing stellar spectra and model grids. It includes utilities for photometric synthesis, spectral resampling, and SED construction using libraries such as PHOENIX.

---

## Installation

With Poetry (recommended):

```bash
git clone https://github.com/brackham/speclib.git
cd speclib
poetry install
```

Or with pip:

```bash
pip install git+https://github.com/brackham/speclib.git
```

---

## Requirements

* Python ≥ 3.11
* [astropy](https://www.astropy.org/)
* [specutils](https://specutils.readthedocs.io/)
* [pysynphot](https://pysynphot.readthedocs.io/)

---

## Example

```python
from speclib import Spectrum, Filter, apply_filter

spec = Spectrum.from_grid(teff=4000, logg=4.5, feh=0.0)
filt = Filter("2MASS J")
flux = apply_filter(spec, filt)

print(f"J-band flux: {flux:.2e}")
```

---

## License

MIT © 2025 Benjamin V. Rackham
