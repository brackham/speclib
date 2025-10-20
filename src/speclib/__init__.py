"""
speclib: Tools for working with stellar spectral libraries.
"""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import tomllib

try:
    __version__ = version("speclib")
except PackageNotFoundError:
    try:
        with open(Path(__file__).resolve().parents[2] / "pyproject.toml", "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    except Exception:
        __version__ = "unknown"

from .core import Spectrum, BinnedSpectrum, SpectralGrid, BinnedSpectralGrid
from .photometry import Filter, SED, SEDGrid, apply_filter, mag_to_flux
from .utils import download_file, download_phoenix_grid

__all__ = [
    "Spectrum",
    "BinnedSpectrum",
    "SpectralGrid",
    "BinnedSpectralGrid",
    "Filter",
    "SED",
    "SEDGrid",
    "apply_filter",
    "mag_to_flux",
    "download_file",
    "download_phoenix_grid",
]
