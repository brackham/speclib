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

## Example

```python
from speclib import Spectrum, Filter, apply_filter

spec = Spectrum.from_grid(teff=4000, logg=4.5, feh=0.0)
filt = Filter("2MASS J")
flux = apply_filter(spec, filt)

print(f"J-band flux: {flux:.2e}")
```

### Custom library cache location

Downloaded spectral libraries are stored in `~/.speclib/libraries` by default.
Set the ``SPECLIB_LIBRARY_PATH`` environment variable or call
``speclib.utils.set_library_root("/path/to/cache")`` to use a different
location.

### NewEra grid caching and extraction

NewEra grid downloads now keep the tarball cached without unpacking by default.
When a specific metallicity file is needed, it is extracted on demand from the
cached archive. To explicitly extract the full grid (for offline usage or bulk
access), use the ``extract="all"`` option:

```python
from speclib.utils import download_newera_grid

# Cache the tarball only (default behavior)
download_newera_grid("newera_jwst")

# Extract all members from the tarball
download_newera_grid("newera_jwst", extract="all")
```

### SPHINX I V4

Download and extract the SPHINX I V4 spectra from Zenodo record 11392341:

```python
from speclib import Spectrum, SpectralGrid, download_sphinx_grid

download_sphinx_grid()

spec = Spectrum.from_grid(
    teff=3000,
    logg=4.5,
    feh=0.0,
    co_ratio=0.5,
    model_grid="sphinx",
)

grid = SpectralGrid(
    teff_bds=(2800, 3200),
    logg_bds=(4.0, 5.0),
    feh_bds=(-0.5, 0.5),
    co_ratio=0.5,
    model_grid="sphinx",
)
```

SPHINX filenames call the metallicity parameter ``logZ``; it is selected
through speclib's existing ``feh`` argument. A C/O ratio must be supplied
explicitly and remains fixed within each three-dimensional ``SpectralGrid``.

---

## License

MIT © 2025 Benjamin V. Rackham
