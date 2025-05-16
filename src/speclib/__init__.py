"""
speclib: Tools for working with stellar spectral libraries.
"""

__version__ = "0.1.0"

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
