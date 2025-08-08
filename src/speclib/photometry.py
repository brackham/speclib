import astropy.units as u
import astropy.io as io
import numpy as np
from importlib.resources import files

from .core import Spectrum
from .utils import interpolate

__all__ = ["Filter", "SED", "SEDGrid", "apply_filter", "mag_to_flux"]


class Filter(object):
    """
    A photometric filter.

    Loads filter metadata and response curve from local ECSV and text files.

    Attributes
    ----------
    name : str
        Name of the filter.
    wl_eff : `~astropy.units.Quantity`
        Effective wavelength of the filter.
    bandwidth : `~astropy.units.Quantity`
        Bandwidth of the filter.
    zeropoint_flux : `~astropy.units.Quantity`
        Zeropoint flux of the filter.
    zeropoint_flux_err : `~astropy.units.Quantity`
        Error on the zeropoint flux.
    response : `~speclib.Spectrum`
        Filter response curve as a spectrum.
    """

    def __init__(self, name):
        self.name = name
        data = self._get_filter_data(self.name)
        self.wl_eff = data["wl_eff"].quantity[0]
        self.bandwidth = data["bandwidth"].quantity[0]
        self.zeropoint_flux = data["zeropoint_flux"].quantity[0]
        self.zeropoint_flux_err = data["zeropoint_flux_err"].quantity[0]
        response_file = data["response_file"][0]
        response_path = files("speclib.data.filters") / response_file
        self.response = self._load_response(response_path)

    def _get_filter_data(self, name):
        """Return the row of filter metadata corresponding to the given name."""
        filters_path = files("speclib.data.filters") / "filters.ecsv"
        filters = io.ascii.read(filters_path)
        good = filters["name"] == name
        if not good.sum():
            raise ValueError(f"'{name}' not recognized.")
        data = filters[good]

        return data

    def _load_response(self, response_file):
        """Load the filter response curve from a text file."""
        wave, trans = np.loadtxt(response_file, unpack=True)
        response = Spectrum(
            spectral_axis=wave * u.AA, flux=trans * u.dimensionless_unscaled
        )

        return response

    @u.quantity_input(wavelength=u.AA)
    def resample(self, wavelength, taper=True):
        """Resample the filter response curve onto a new wavelength grid."""
        self.response = self.response.resample(wavelength, taper=taper)


class SED(object):
    """
    A spectral energy distribution derived from a spectrum and a set of filters.

    Attributes
    ----------
    wavelength : `~astropy.units.Quantity`
        Effective wavelengths of the filters.
    bandwidth : `~astropy.units.Quantity`
        Bandwidths of the filters.
    flux : `~astropy.units.Quantity`
        Filtered fluxes.
    """

    def __init__(self, spec, filters, model_grid="phoenix"):
        wavelength = []
        bandwidth = []
        flux = []
        for filt in filters:
            wavelength.append(filt.wl_eff)
            bandwidth.append(filt.bandwidth)
            flux.append(apply_filter(spec, filt))

        # Hacky way to move units outside the arrays
        self.wavelength = np.array([f.value for f in wavelength]) * wavelength[0].unit
        self.bandwidth = np.array([f.value for f in bandwidth]) * bandwidth[0].unit
        self.flux = np.array([f.value for f in flux]) * flux[0].unit

    @classmethod
    def from_grid(self, teff, logg, feh, filters, model_grid="phoenix"):
        """
        Create an SED from a stellar model grid.

        Parameters
        ----------
        teff : float
            Effective temperature [K]
        logg : float
            Surface gravity [cgs]
        feh : float
            Metallicity [Fe/H]
        filters : list
            List of `~speclib.Filter` objects
        model_grid : str
            Name of model grid (default 'phoenix')

        Returns
        -------
        `~speclib.SED`
        """
        spec = Spectrum.from_grid(teff, logg, feh, model_grid=model_grid)
        sed = SED(spec, filters, model_grid)

        return sed


class SEDGrid(object):
    """
    A grid of SEDs for quick interpolation.

    Attributes
    ----------
    teff_bds : iterable
        The lower and upper bounds of the model temperatures to load.

    logg_bds : iterable
        The lower and upper bounds of the model logg values to load.

    feh_bds : iterable
        The lower and upper bounds of the model [Fe/H] to load.

    wavelength : `~astropy.units.Quantity`
        Effective wavelengths of the SED grid.

    bandwidth : `~astropy.units.Quantity`
        Bandwidths of the interpolated SED grid.

    fluxes : dict
        The fluxes of the model grid. Sorted by fluxes[teff][logg][feh].

    model_grid : str
        Name of the model grid. Only `phoenix` is currently supported.

    Methods
    -------
    get_SED(teff, logg, feh)
        Returns a SED for the given teff, logg, and feh.

    """

    def __init__(
        self,
        teff_bds,
        logg_bds,
        feh_bds,
        filters,
        model_grid="phoenix",
    ):
        """
        Parameters
        ----------
        teff_bds : iterable
            The lower and upper bounds of the model temperatures to load.

        logg_bds : iterable
            The lower and upper bounds of the model logg values to load.

        feh_bds : iterable
            The lower and upper bounds of the model [Fe/H] to load.

        filters : iterable
            An iterable of `~speclib.Filter` objects.

        model_grid : str, optional
            Name of the model grid. Only `phoenix` is currently supported.
        """
        # First check that the model_grid is valid.
        self.model_grid = model_grid.lower()

        if self.model_grid == "phoenix":
            # Grid of effective temperatures
            grid_teffs = np.append(
                np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)
            )

            # Grid of surface gravities
            grid_loggs = np.arange(0.0, 6.5, 0.5)

            # Grid of metallicities
            grid_fehs = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, +0.5, +1.0])
        else:
            raise NotImplementedError(
                f'"{model_grid}" model grid not found. '
                + "Only PHOENIX models are currently supported."
            )

        # Then ensure that the bounds given are valid.
        teff_bds = np.array(teff_bds)
        teff_bds = (
            grid_teffs[grid_teffs <= teff_bds.min()].max(),
            grid_teffs[grid_teffs >= teff_bds.max()].min(),
        )
        self.teff_bds = teff_bds

        logg_bds = np.array(logg_bds)
        logg_bds = (
            grid_loggs[grid_loggs <= logg_bds.min()].max(),
            grid_loggs[grid_loggs >= logg_bds.max()].min(),
        )
        self.logg_bds = logg_bds

        feh_bds = np.array(feh_bds)
        feh_bds = (
            grid_fehs[grid_fehs <= feh_bds.min()].max(),
            grid_fehs[grid_fehs >= feh_bds.max()].min(),
        )
        self.feh_bds = feh_bds

        # Define the values covered in the grid
        subset = np.logical_and(
            grid_teffs >= self.teff_bds[0], grid_teffs <= self.teff_bds[1]
        )
        self.teffs = grid_teffs[subset]

        subset = np.logical_and(
            grid_loggs >= self.logg_bds[0], grid_loggs <= self.logg_bds[1]
        )
        self.loggs = grid_loggs[subset]

        subset = np.logical_and(
            grid_fehs >= self.feh_bds[0], grid_fehs <= self.feh_bds[1]
        )
        self.fehs = grid_fehs[subset]

        # Load the fluxes
        fluxes = {}
        for teff in self.teffs:
            fluxes[teff] = {}
            for logg in self.loggs:
                fluxes[teff][logg] = {}
                for feh in self.fehs:
                    sed = SED.from_grid(teff, logg, feh, filters)

                    fluxes[teff][logg][feh] = sed.flux
        self.fluxes = fluxes

        # Save the wavelength array
        self.wavelength = sed.wavelength

        # Save the bandwidth array
        self.bandwidth = sed.bandwidth

    def get_SED(self, teff, logg, feh):
        """
        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            The interpolated flux array.
        """

        # First check that the values are within the grid
        teff_in_grid = self.teff_bds[0] <= teff <= self.teff_bds[1]
        logg_in_grid = self.logg_bds[0] <= logg <= self.logg_bds[1]
        feh_in_grid = self.feh_bds[0] <= feh <= self.feh_bds[1]

        booleans = [teff_in_grid, logg_in_grid, feh_in_grid]
        params = ["teff", "logg", "feh"]
        inputs = [teff, logg, feh]
        ranges = [self.teff_bds, self.logg_bds, self.feh_bds]

        if not all(booleans):
            message = "Input values are out of grid range.\n\n"
            for b, p, i, r in zip(booleans, params, inputs, ranges):
                if not b:
                    message += f"\tInput {p}: {i}. Valid range: {r}\n"
            raise ValueError(message)

        # Identify nearest values in grid
        flanking_teffs = (
            self.teffs[self.teffs <= teff].max(),
            self.teffs[self.teffs >= teff].min(),
        )
        flanking_loggs = (
            self.loggs[self.loggs <= logg].max(),
            self.loggs[self.loggs >= logg].min(),
        )
        flanking_fehs = (
            self.fehs[self.fehs <= feh].max(),
            self.fehs[self.fehs >= feh].min(),
        )

        # Define the points for interpolation
        params000 = (flanking_teffs[0], flanking_loggs[0], flanking_fehs[0])
        params100 = (flanking_teffs[1], flanking_loggs[0], flanking_fehs[0])
        params010 = (flanking_teffs[0], flanking_loggs[1], flanking_fehs[0])
        params110 = (flanking_teffs[1], flanking_loggs[1], flanking_fehs[0])
        params001 = (flanking_teffs[0], flanking_loggs[0], flanking_fehs[1])
        params101 = (flanking_teffs[1], flanking_loggs[0], flanking_fehs[1])
        params011 = (flanking_teffs[0], flanking_loggs[1], flanking_fehs[1])
        params111 = (flanking_teffs[1], flanking_loggs[1], flanking_fehs[1])

        # Interpolate trilinearly
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        if not params000 == params100:
            c000 = self.fluxes[params000[0]][params000[1]][params000[2]]
            c100 = self.fluxes[params100[0]][params100[1]][params100[2]]
            c00 = interpolate([c000, c100], flanking_teffs, teff)
        else:
            c00 = self.fluxes[params000[0]][params000[1]][params000[2]]

        if not params010 == params110:
            c010 = self.fluxes[params010[0]][params010[1]][params010[2]]
            c110 = self.fluxes[params110[0]][params110[1]][params110[2]]
            c10 = interpolate([c010, c110], flanking_teffs, teff)
        else:
            c10 = self.fluxes[params010[0]][params010[1]][params010[2]]

        if not params001 == params101:
            c001 = self.fluxes[params001[0]][params001[1]][params001[2]]
            c101 = self.fluxes[params101[0]][params101[1]][params101[2]]
            c01 = interpolate([c001, c101], flanking_teffs, teff)
        else:
            c01 = self.fluxes[params001[0]][params001[1]][params001[2]]

        if not params011 == params111:
            c011 = self.fluxes[params011[0]][params011[1]][params011[2]]
            c111 = self.fluxes[params111[0]][params111[1]][params111[2]]
            c11 = interpolate([c011, c111], flanking_teffs, teff)
        else:
            c11 = self.fluxes[params011[0]][params011[1]][params011[2]]

        if not params000 == params010:
            c0 = interpolate([c00, c10], flanking_loggs, logg)
            c1 = interpolate([c01, c11], flanking_loggs, logg)
        else:
            c0 = c00
            c1 = c01

        if not params000 == params001:
            flux = interpolate([c0, c1], flanking_fehs, feh)
        else:
            flux = c0

        return flux


def apply_filter(spec, filt):
    """
    Apply a photometric filter to a spectrum.

    Parameters
    ----------
    spec : `~speclib.Spectrum`
        The input spectrum.
    filt : `~speclib.Filter`
        The filter object.

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Filtered flux.
    """
    try:
        filtered_flux = spec.flux * filt.response.flux
    except ValueError:
        filt.resample(spec.wavelength)
        filtered_flux = spec.flux * filt.response.flux
    integrated_flux = np.trapezoid(filtered_flux, spec.wavelength) / filt.bandwidth

    return integrated_flux


def mag_to_flux(mag, filt, mag_err=0, nsamples=100000):
    """
    Convert magnitude to flux using a filter zeropoint and propagate uncertainty.

    Parameters
    ----------
    mag : float
        Apparent magnitude.
    filt : `~speclib.Filter`
        The filter.
    mag_err : float, optional
        Uncertainty on the magnitude.
    nsamples : int, optional
        Number of Monte Carlo samples (default 100,000).

    Returns
    -------
    mean : `~astropy.units.Quantity`
        Mean flux.
    std : `~astropy.units.Quantity`
        Standard deviation of the flux.
    """
    zp = filt.zeropoint_flux
    zp_err = filt.zeropoint_flux_err

    # Monte Carlo error propagation
    zp_dist = np.random.normal(zp.value, zp_err.value, size=nsamples) * zp.unit
    mag_dist = np.random.normal(mag, mag_err, size=nsamples)
    flux_dist = 10 ** (-mag_dist / 2.5) * zp_dist
    mean, std = np.mean(flux_dist), np.std(flux_dist)

    return mean, std
