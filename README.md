# speclib

[![DOI](https://zenodo.org/badge/339518965.svg)](https://doi.org/10.5281/zenodo.7868049)
[![Documentation Status](https://app.readthedocs.org/projects/speclib/badge/?version=latest)](https://speclib.readthedocs.io/en/latest/)
[![Installation smoke test](https://github.com/brackham/speclib/actions/workflows/installation.yml/badge.svg?branch=main)](https://github.com/brackham/speclib/actions/workflows/installation.yml)
[![License: MIT](https://img.shields.io/github/license/brackham/speclib)](https://github.com/brackham/speclib/blob/main/LICENSE)

**Tools for working with stellar spectral libraries.**

`speclib` is a Python package for accessing and working with stellar spectral libraries. It provides a consistent interface to stellar atmosphere models, along with tools for spectral interpolation, resolution matching, and synthetic photometry.

## Installation

`speclib` supports Python 3.11–3.13.

Install a specific release (e.g., `0.1.0b13`):

```bash
python -m pip install git+https://github.com/brackham/speclib.git@v0.1.0b13
```

Or install the latest development version from `main`:

```bash
python -m pip install git+https://github.com/brackham/speclib.git@main
```

The development version may contain changes not yet included in a release. See the [installation guide](https://speclib.readthedocs.io/en/latest/installation.html) for additional options.

## Documentation

Documentation, tutorials, model library descriptions, and the API reference are available at **[speclib.readthedocs.io](https://speclib.readthedocs.io/)**.

## Citation

If you use `speclib` in published work, please cite both:

1. The [Zenodo software archive](https://doi.org/10.5281/zenodo.7868049), preferably the DOI for the specific version used.
2. Rackham & de Wit (2024), [AJ, 168, 82](https://doi.org/10.3847/1538-3881/ad5833).

Please also cite the underlying stellar model libraries used in your analysis. See the [citation guide](https://speclib.readthedocs.io/en/latest/citation.html) for the appropriate references.

## License

`speclib` is distributed under the [MIT License](LICENSE).
