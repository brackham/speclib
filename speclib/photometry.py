import astropy.units as u
import astropy.io as io
import numpy as np
from .main import Spectrum

__all__ = ["Filter"]


class Filter(object):
    """A filter."""

    def __init__(self, name):
        self.name = name
        data = self._get_filter_data(self.name)
        self.wl_eff = data["wl_eff"].quantity[0]
        self.bandwidth = data["bandwidth"].quantity[0]
        self.zeropoint_flux = data["zeropoint_flux"].quantity[0]
        self.zeropoint_flux_err = data["zeropoint_flux_err"].quantity[0]
        self.response_file = data["response_file"][0]
        self.response = self._load_response(self.response_file)
        self.resampled_response = None

    def _get_filter_data(self, name):
        filters = io.ascii.read("./data/filters/filters.ecsv")
        good = filters["name"] == name
        if not good.sum():
            raise ValueError(f"'{name}' not recognized.")
        data = filters[good]

        return data

    def _load_response(self, file):
        filter_dir = "./data/filters/"
        wave, trans = np.loadtxt(filter_dir + file, unpack=True)
        response = Spectrum(
            spectral_axis=wave * u.AA, flux=trans * u.dimensionless_unscaled
        )

        return response

    @u.quantity_input(wavelength=u.AA)
    def resample_response(self, wavelength):
        self.resampled_response = self.response.resample(wavelength)
