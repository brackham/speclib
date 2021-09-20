import astropy.units as u
import astropy.io.fits as fits
import numpy as np
import os
import shutil
import urllib
from contextlib import closing
from specutils import Spectrum1D
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pysynphot as psp

__all__ = ["Spectrum", "BinnedSpectrum", "SpectralGrid", "BinnedSpectralGrid"]


def download_file(remote_path, local_path, verbose=True):
    """
    Download a file via ftp.
    """
    if verbose:
        print(f"> Downloading {remote_path}")
    with closing(urllib.request.urlopen(remote_path)) as r:
        with open(local_path, "wb") as f:
            shutil.copyfileobj(r, f)


def interpolate(fluxes, xlims, x):
    y0, y1 = fluxes
    x0, x1 = xlims
    w1 = (x - x0) / (x1 - x0)
    y = y0 * (1 - w1) + y1 * w1

    return y


def find_bounds(array, value):
    """
    Find and return the two nearest values in an array to a given value.
    """
    array = np.array(array)
    idxs = np.argsort(np.abs(array - value))[0:2]
    return np.sort(array[idxs])


def load_flux_array(fname, cache_dir, ftp_url):
    """
    Load a flux array.
    """
    # Look for a local file first
    flux_local_path = os.path.join(cache_dir, fname)
    try:
        flux = fits.getdata(flux_local_path)
    # If that doesn't work, download the remote file
    except FileNotFoundError:
        feh_folder = "Z" + fname[13:17]
        flux_remote_path = os.path.join(
            ftp_url, "HiResFITS/PHOENIX-ACES-AGSS-COND-2011", feh_folder, fname
        )
        download_file(flux_remote_path, flux_local_path)
        flux = fits.getdata(flux_local_path)

    return flux


class Spectrum(Spectrum1D):
    """
    A wrapper class for `~spectutils.spectrum1d.Spectrum1D`.

    New functionality added to load spectra from a model grid,
    resample spectra quickly using `~pysynphot`, and bin spectra.

    Methods
    -------
    from_grid(teff, logg, feh=0, wave=None, model_grid='phoenix')
        Load a model spectrum from a library.

    resample(wave)
        Resample a spectrum while conserving flux.

    bin(center, width)
        Bin a model spectrum within specified wavelength bins.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_grid(
        self,
        teff,
        logg,
        feh=0,
        wave=None,
        wl_min=None,
        wl_max=None,
        model_grid="phoenix",
    ):
        """
        Load a model spectrum from a library.

        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model.

        wave : `~astropy.units.Quantity`, optional
            Wavelengths of the interpolated spectrum.

        wl_min : `~astropy.units.Quantity`, optional
            Minimum wavelength of the model spectrum.

        wl_max : `~astropy.units.Quantity`, optional
            Maximium wavelength of the model spectrum.

        model_grid : str, optional
            Name of the model grid. Only `phoenix` is currently supported.

        Returns
        -------
        spec : `~speclib.Spectrum`
            A spectrum for the specified parameters.
        """

        if model_grid.lower() == "phoenix":
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".speclib/libraries/phoenix/"
            )
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            ftp_url = "ftp://phoenix.astro.physik.uni-goettingen.de"
            fname_str = (
                "lte{:05.0f}-{:0.2f}{:+0.1f}."
                + "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
            )

            # Grid of effective temperatures
            grid_teffs = np.append(
                np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)
            )

            # Grid of surface gravities
            grid_loggs = np.arange(0.0, 6.5, 0.5)

            # Grid of metallicities
            grid_fehs = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, +0.5, +1.0])

            # The convention of the PHOENIX model grids is that
            # [Fe/H] = 0.0 is written as a negative number.
            if feh == 0:
                feh = -0.0
        else:
            raise NotImplementedError(
                f'"{model_grid}" model grid not found. '
                + "Only PHOENIX models are currently supported."
            )

        # Load the wavelength array
        wave_local_path = os.path.join(
            cache_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        )
        try:
            wave_lib = fits.getdata(wave_local_path)
        except FileNotFoundError:
            wave_remote_path = os.path.join(
                ftp_url, "HiResFITS", "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
            )
            download_file(wave_remote_path, wave_local_path)
            wave_lib = fits.getdata(wave_local_path)

        teff_in_grid = teff in grid_teffs
        logg_in_grid = logg in grid_loggs
        feh_in_grid = feh in grid_fehs
        model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
        if not model_in_grid:
            if teff_in_grid:
                teff_bds = [teff, teff]
            else:
                teff_bds = find_bounds(grid_teffs, teff)
            if logg_in_grid:
                logg_bds = [logg, logg]
            else:
                logg_bds = find_bounds(grid_loggs, logg)
            if feh_in_grid:
                feh_bds = [feh, feh]
            else:
                feh_bds = find_bounds(grid_fehs, feh)

            fname000 = fname_str.format(teff_bds[0], logg_bds[0], feh_bds[0])
            fname100 = fname_str.format(teff_bds[1], logg_bds[0], feh_bds[0])
            fname010 = fname_str.format(teff_bds[0], logg_bds[1], feh_bds[0])
            fname110 = fname_str.format(teff_bds[1], logg_bds[1], feh_bds[0])
            fname001 = fname_str.format(teff_bds[0], logg_bds[0], feh_bds[1])
            fname101 = fname_str.format(teff_bds[1], logg_bds[0], feh_bds[1])
            fname011 = fname_str.format(teff_bds[0], logg_bds[1], feh_bds[1])
            fname111 = fname_str.format(teff_bds[1], logg_bds[1], feh_bds[1])

            if not fname000 == fname100:
                c000 = load_flux_array(fname000, cache_dir, ftp_url)
                c100 = load_flux_array(fname100, cache_dir, ftp_url)
                c00 = interpolate([c000, c100], teff_bds, teff)
            else:
                c00 = load_flux_array(fname000, cache_dir, ftp_url)

            if not fname010 == fname110:
                c010 = load_flux_array(fname010, cache_dir, ftp_url)
                c110 = load_flux_array(fname110, cache_dir, ftp_url)
                c10 = interpolate([c010, c110], teff_bds, teff)
            else:
                c10 = load_flux_array(fname010, cache_dir, ftp_url)

            if not fname001 == fname101:
                c001 = load_flux_array(fname001, cache_dir, ftp_url)
                c101 = load_flux_array(fname101, cache_dir, ftp_url)
                c01 = interpolate([c001, c101], teff_bds, teff)
            else:
                c01 = load_flux_array(fname001, cache_dir, ftp_url)

            if not fname011 == fname111:
                c011 = load_flux_array(fname011, cache_dir, ftp_url)
                c111 = load_flux_array(fname111, cache_dir, ftp_url)
                c11 = interpolate([c011, c111], teff_bds, teff)
            else:
                c11 = load_flux_array(fname011, cache_dir, ftp_url)

            if not fname000 == fname010:
                c0 = interpolate([c00, c10], logg_bds, logg)
                c1 = interpolate([c01, c11], logg_bds, logg)
            else:
                c0 = c00
                c1 = c01

            if not fname000 == fname001:
                flux = interpolate([c0, c1], feh_bds, feh)
            else:
                flux = c0

        elif model_in_grid:
            # Load the flux array
            fname = fname_str.format(teff, logg, feh)
            flux = load_flux_array(fname, cache_dir, ftp_url)

        # Load `~speclib.Spectrum` object
        conversion_factor = 1e-8  # erg/(s * cm^3) to erg/(s * cm^2 * Å)
        spec = Spectrum(
            spectral_axis=wave_lib * u.AA,
            flux=flux * conversion_factor * u.Unit("erg/(s * cm^2 * angstrom)"),
        )

        # Crop to wavelength min and max, if given
        if not all(v is None for v in [wl_min, wl_max]):
            if wl_min is None:
                wl_min = spec.wavelength.min()
            if wl_max is None:
                wl_max = spec.wavelength.max()
            mask = np.logical_or(spec.wavelength <= wl_min, spec.wavelength >= wl_max)
            spec = Spectrum(spectral_axis=spec.wavelength[~mask], flux=spec.flux[~mask])

        # Resample the spectrum to the desired wavelength array
        if wave is not None:
            spec = spec.resample(wave)

        return spec

    @u.quantity_input(wave=u.AA)
    def resample(self, wave):
        """
        Resample a spectrum while conserving flux.

        Parameters
        ----------
        wave : `~astropy.units.Quantity`
            A new wavelength axis. Unit must be specified.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
             A resampled spectrum.
        """
        # Convert wavelengths arrays to same unit
        wave_old = self.wavelength.to(u.AA).value
        wave_new = wave.to(u.AA).value
        waveunits = "angstrom"

        # The input value without a unit
        flux_old = self.flux.value

        # Make an observation object with pysynphot
        spectrum = psp.spectrum.ArraySourceSpectrum(wave=wave_old, flux=flux_old)
        throughput = np.ones(len(wave_old))
        filt = psp.spectrum.ArraySpectralElement(
            wave_old, throughput, waveunits=waveunits
        )
        obs = psp.observation.Observation(
            spectrum, filt, binset=wave_new, force="taper"
        )

        # Save the new binned flux array in a `~speclib.Spectrum` object
        spec_new = Spectrum(spectral_axis=wave, flux=obs.binflux * self.flux.unit)

        return spec_new

    @u.quantity_input(center=u.AA, width=u.AA)
    def bin(self, center, width):
        """
        Bin a model spectrum within specified wavelength bins.

        Parameters
        ----------
        center : `~astropy.units.Quantity`
            The centers of the wavelength bins.

        width : `~astropy.units.Quantity`
            The widths of the wavelength bins.

        Returns
        -------
        `~speclib.BinnedSpectrum`

        """
        wave = self.wavelength
        flux = self.flux
        binned_fluxes = []
        for cen, wid in zip(center, width):
            lower = cen - wid / 2.0
            upper = cen + wid / 2.0
            idx = np.where((wave >= lower) & (wave <= upper))

            # Adjust for bins that are slightly wider than the wavelength range
            # due to discretization of the wavelength grid
            scale_factor = (upper - lower) / (wave[idx][-1] - wave[idx][0])

            binned_flux = (
                scale_factor * np.trapz(flux[idx], wave[idx]) / (upper - lower)
            )
            binned_fluxes.append(binned_flux)
        binned_fluxes = u.Quantity(binned_fluxes)

        return BinnedSpectrum(center, width, binned_fluxes)


class BinnedSpectrum(object):
    """
    A binned spectrum.

    Attributes
    ----------
    center : `~astropy.units.Quantity`
        The centers of the wavelength bins.

    width : `~astropy.units.Quantity`
        The widths of the wavelength bins.

    lower : `~astropy.units.Quantity`
        The lower bounds of the wavelength bins.

    upper : `~astropy.units.Quantity`
        The upper bounds of the wavelength bins.

    flux : `~astropy.units.Quantity`
        The binned flux array.
    """

    @u.quantity_input(center=u.AA, width=u.AA)
    def __init__(self, center, width, flux):
        """
        Parameters
        ----------
        center : `~astropy.units.Quantity`
            The centers of the wavelength bins.

        width : `~astropy.units.Quantity`
            The widths of the wavelength bins.

        flux : iterable
            The binned flux array.
        """
        self.center = center
        self.width = width
        self.lower = center - width / 2.0
        self.upper = center + width / 2.0
        self.flux = flux


class SpectralGrid(object):
    """
    A grid of spectra for quick interpolation.

    Attributes
    ----------
    teff_bds : iterable
        The lower and upper bounds of the model temperatures to load.

    logg_bds : iterable
        The lower and upper bounds of the model logg values to load.

    feh_bds : iterable
        The lower and upper bounds of the model [Fe/H] to load.

    wave : `~astropy.units.Quantity`
        Wavelengths of the interpolated spectrum.

    fluxes : dict
        The fluxes of the model grid. Sorted by fluxes[teff][logg][feh].

    model_grid : str
        Name of the model grid. Only `phoenix` is currently supported.

    Methods
    -------
    get_spectrum(teff, logg, feh)
        Returns a binned spectrum for the given teff, logg, and feh.

    """

    def __init__(
        self, teff_bds, logg_bds, feh_bds, wave=None, model_grid="phoenix", **kwargs
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

        wavelength : `~astropy.units.Quantity`, optional
            Wavelengths of the interpolated spectrum.

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
                    spec = Spectrum.from_grid(teff, logg, feh, **kwargs)
                    # Resample the spectrum to the desired wavelength array
                    if wave is not None:
                        spec = spec.resample(wave)
                    fluxes[teff][logg][feh] = spec.flux
        self.fluxes = fluxes

        # Save the wavelength array
        self.wave = spec.wavelength

    def get_spectrum(self, teff, logg, feh):
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


class BinnedSpectralGrid(object):
    """
    A grid of binned spectra for quick interpolation.

    Attributes
    ----------
    teff_bds : iterable
        The lower and upper bounds of the model temperatures to load.

    logg_bds : iterable
        The lower and upper bounds of the model logg values to load.

    feh_bds : iterable
        The lower and upper bounds of the model [Fe/H] to load.

    center : `~astropy.units.Quantity`
        The centers of the wavelength bins.

    width : `~astropy.units.Quantity`
        The widths of the wavelength bins.

    lower : `~astropy.units.Quantity`
        The lower bounds of the wavelength bins.

    upper : `~astropy.units.Quantity`
        The upper bounds of the wavelength bins.

    fluxes : dict
        The fluxes of the model grid. Sorted by fluxes[teff][logg][feh].

    model_grid : str
        Name of the model grid. Only `phoenix` is currently supported.

    Methods
    -------
    get_spectrum(teff, logg, feh)
        Returns a binned spectrum for the given teff, logg, and feh.

    """

    def __init__(
        self, teff_bds, logg_bds, feh_bds, center, width, model_grid="phoenix", **kwargs
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

        center : `~astropy.units.Quantity`
            The centers of the wavelength bins.

        width : `~astropy.units.Quantity`
            The widths of the wavelength bins.

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
        self.center = center
        self.width = width
        self.lower = center - width / 2.0
        self.upper = center + width / 2.0

        fluxes = {}
        for teff in self.teffs:
            fluxes[teff] = {}
            for logg in self.loggs:
                fluxes[teff][logg] = {}
                for feh in self.fehs:
                    bs = Spectrum.from_grid(teff, logg, feh, **kwargs).bin(
                        center, width
                    )
                    fluxes[teff][logg][feh] = bs.flux
        self.fluxes = fluxes

    def get_spectrum(self, teff, logg, feh):
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
