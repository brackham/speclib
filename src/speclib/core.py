import astropy.units as u
import astropy.io.fits as fits
import numpy as np
import os
from pathlib import Path
import speclib.utils as utils
from specutils import Spectrum1D
from scipy.interpolate import NearestNDInterpolator

import warnings

import synphot as sp

__all__ = ["Spectrum", "BinnedSpectrum", "SpectralGrid", "BinnedSpectralGrid"]


class Spectrum(Spectrum1D):
    """
    A wrapper class for `~specutils.Spectrum1D` with extended functionality for
    working with stellar model spectra.

    This class adds capabilities to:
    - Load and interpolate spectra from various model grids
    - Resample spectra using `synphot` to conserve flux
    - Convolve spectra to lower resolution
    - Bin spectra into custom wavelength intervals

    Parameters
    ----------
    **kwargs : dict
        Arguments passed to the base `Spectrum1D` initializer.

    Methods
    -------
    from_grid(teff, logg, feh=0, wavelength=None, model_grid='phoenix')
        Load a model spectrum from a library.

    resample(wavelength)
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
        alpha=0.0,
        # CtoO=0.5,
        wavelength=None,
        wl_min=None,
        wl_max=None,
        model_grid="phoenix",
        interpolate=True,
        verbose=False,
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

        wavelength : `~astropy.units.Quantity`, optional
            Wavelengths of the interpolated spectrum.

        wl_min : `~astropy.units.Quantity`, optional
            Minimum wavelength of the model spectrum.

        wl_max : `~astropy.units.Quantity`, optional
            Maximium wavelength of the model spectrum.

        model_grid : str, optional
            Name of the model grid.

        verbose: bool, optional
            Print details for debugging.

        interpolate : bool, optional
            Whether to interpolate between grid points. If `True` (default), the spectrum
            will be trilinearly interpolated in (Teff, logg, [Fe/H]) space. If `False`,
            the nearest available grid point will be used without interpolation.

        Returns
        -------
        spec : `~speclib.Spectrum`
            A spectrum for the specified parameters.
        """
        # First check that the model_grid is valid.
        self.model_grid = model_grid.lower()
        if self.model_grid not in utils.VALID_MODELS:
            raise NotImplementedError(
                f'"{self.model_grid}" model grid not found. '
                + "Currently supported models are: "
                + str(utils.VALID_MODELS)
            )

        # Define grid points
        self.grid_points = utils.GRID_POINTS[self.model_grid]
        self.grid_teffs = self.grid_points["grid_teffs"]
        self.grid_loggs = self.grid_points["grid_loggs"]
        self.grid_fehs = self.grid_points["grid_fehs"]

        if self.model_grid == "phoenix":
            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg/(s * cm^3)")
            cache_dir = utils.get_library_root() / "phoenix"
            cache_dir.mkdir(parents=True, exist_ok=True)

            ftp_url = "ftp://phoenix.astro.physik.uni-goettingen.de"
            fname_str = (
                "lte{:05.0f}-{:0.2f}{:+0.1f}."
                + "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
            )

            # The convention of the PHOENIX model grids is that
            # [Fe/H] = 0.0 is written as a negative number.
            if feh == 0:
                feh = -0.0

            # Load the wavelength array
            wave_local_path = cache_dir / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
            try:
                wave_lib = fits.getdata(wave_local_path)
            except FileNotFoundError:
                wave_remote_path = os.path.join(
                    ftp_url, "HiResFITS", "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
                )
                utils.download_file(wave_remote_path, wave_local_path, verbose)
                wave_lib = fits.getdata(wave_local_path)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not interpolate and not model_in_grid:
                teff = utils.nearest(self.grid_teffs, teff)
                logg = utils.nearest(self.grid_loggs, logg)
                feh = utils.nearest(self.grid_fehs, feh)
                model_in_grid = True  # force nearest model retrieval
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = utils.load_flux_array(
                                fname, cache_dir, ftp_url
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the flux array
                fname = fname_str.format(teff, logg, feh)
                flux = utils.load_flux_array(fname, cache_dir, ftp_url)

        elif self.model_grid == "newera":
            import h5py

            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg / (s * cm^3)")
            cache_dir = utils.get_library_root() / "newera"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Ensure feh is float-compatible with naming convention
            if feh == 0:
                feh = -0.0

            # Define bounds
            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            alpha_in_grid = alpha in self.grid_points.get("grid_alphas", [0.0])
            use_alpha = feh >= -2.0 and feh <= 0.0 and alpha_in_grid

            def load_flux_from_h5(teff_, logg_, feh_, alpha_=0.0):
                local_path = utils.download_newera_file(
                    teff_, logg_, feh_, alpha_, cache_dir=cache_dir
                )
                with h5py.File(local_path, "r") as h5:
                    # Wavelengths in vacuum Ångströms
                    wl = h5["PHOENIX_SPECTRUM/wl"][:]

                    # Flux is log10(F_lambda) in erg / (s cm² cm)
                    log_flux = h5["PHOENIX_SPECTRUM/flux"][:]
                    flux_per_cm = 10**log_flux

                return wl, flux_per_cm

            if not model_in_grid and not interpolate:
                teff = utils.nearest(self.grid_teffs, teff)
                logg = utils.nearest(self.grid_loggs, logg)
                feh = utils.nearest(self.grid_fehs, feh)
                model_in_grid = True

            if not model_in_grid:
                teff_bds = utils.find_bounds(self.grid_teffs, teff)
                logg_bds = utils.find_bounds(self.grid_loggs, logg)
                feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                wave_lib = None
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            wl, flx = load_flux_from_h5(tt, gg, ff, alpha)
                            flux_dict[tt][gg][ff] = flx
                            if wave_lib is None:
                                wave_lib = wl

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            else:
                wave_lib, flux = load_flux_from_h5(teff, logg, feh, alpha)

        elif self.model_grid in ["newera_gaia", "newera_jwst", "newera_lowres"]:
            grid_name = self.model_grid
            lib_wave_unit = u.nm
            lib_flux_unit = u.W / (u.m**2 * u.nm)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            alpha_in_grid = alpha in self.grid_points.get("grid_alphas", [0.0])

            def load_flux(teff_, logg_, feh_, alpha_=0.0):
                return utils.load_newera_flux_array(
                    teff_, logg_, feh_, alpha_, grid_name
                )

            def load_wave(teff_, logg_, feh_, alpha_=0.0):
                return utils.load_newera_wavelength_array(
                    teff_, logg_, feh_, alpha_, grid_name
                )

            if not model_in_grid and not interpolate:
                teff = utils.nearest(self.grid_teffs, teff)
                logg = utils.nearest(self.grid_loggs, logg)
                feh = utils.nearest(self.grid_fehs, feh)
                model_in_grid = True

            if not model_in_grid:
                teff_bds = utils.find_bounds(self.grid_teffs, teff)
                logg_bds = utils.find_bounds(self.grid_loggs, logg)
                feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                wave_lib = None
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            if wave_lib is None:
                                wave_lib = load_wave(tt, gg, ff, alpha)
                            flux_dict[tt][gg][ff] = load_flux(tt, gg, ff, alpha)

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            else:
                wave_lib = load_wave(teff, logg, feh, alpha)
                flux = load_flux(teff, logg, feh, alpha)

        elif self.model_grid == "drift-phoenix":
            # Only works if the user has already cached the DRIFT-PHOENIX model grid
            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg/(s * cm^2 * angstrom)")
            cache_dir = utils.get_library_root() / "drift-phoenix"
            cache_dir.mkdir(parents=True, exist_ok=True)

            fname_str = "lte_{:4.0f}_{:0.1f}{:+0.1f}.7.dat.txt"

            # The convention of the DRIFT-PHOENIX model grids is that
            # [Fe/H] = 0.0 is written as a negative number.
            if feh == 0:
                feh = -0.0

            # Load the wavelength array
            wave_local_path = cache_dir / "lte_1000_3.0-0.0.7.dat.txt"
            wave_lib = np.loadtxt(wave_local_path, unpack=True, usecols=0)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = np.loadtxt(
                                cache_dir / fname, unpack=True, usecols=1
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the wavelength and flux arrays
                fname = fname_str.format(teff, logg, feh)
                wave_lib, flux = np.loadtxt(cache_dir / fname, unpack=True)

        elif self.model_grid == "nextgen-solar":
            # Only works if the user has already cached the NextGen model grid
            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg/(s * cm^2 * angstrom)")
            cache_dir = utils.get_library_root() / "nextgen-solar"
            cache_dir.mkdir(parents=True, exist_ok=True)

            fname_str = "lte{:05.0f}_{:+0.1f}_{:+.1f}_NextGen-solar.dat"

            # Load the wavelength array
            wave_local_path = cache_dir / "lte01600_+5.5_+0.0_NextGen-solar.dat"
            wave_lib = np.loadtxt(wave_local_path, unpack=True, usecols=0)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = np.loadtxt(
                                cache_dir / fname, unpack=True, usecols=1
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the wavelength and flux arrays
                fname = fname_str.format(teff, logg, feh)
                wave_lib, flux = np.loadtxt(cache_dir / fname, unpack=True)

        elif self.model_grid == "sphinx":
            # Only works if the user has already cached the SPHINX model grid
            lib_wave_unit = u.micron
            lib_flux_unit = u.Unit("W/(m^2 * m)")
            cache_dir = utils.get_library_root() / "sphinx"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # CtoO = 0.5  # Not varying this for now
            # fname_str = "Teff_{:04.1f}_logg_{:0.2f}_logZ_{:+0.2f}_CtoO_{:0.1f}_spectra.txt"
            fname_str = "Teff_{:04.1f}_logg_{:0.2f}_logZ_{:+0.2f}_CtoO_0.5_spectra.txt"

            def load_wave_flux(fname):
                path = cache_dir / fname
                wave, flux = np.loadtxt(path, unpack=True)
                return wave, flux

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = np.loadtxt(
                                cache_dir / fname, unpack=True, usecols=1
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the wavelength and flux arrays
                fname = fname_str.format(teff, logg, feh)
                wave_lib, flux = load_wave_flux(fname)

        elif self.model_grid == "mps-atlas":
            # Only works if the user has already cached the MPS-Atlas model grid
            lib_wave_unit = u.nm
            lib_flux_unit = (
                u.Unit("erg / (s * cm^2 * Hz^1)") * (u.AU / u.R_sun).cgs ** 2
            )
            cache_dir = utils.get_library_root() / "mps-atlas"
            cache_dir.mkdir(parents=True, exist_ok=True)
            fname_str = "MH{:+0.2f}/teff{:4.0f}/logg{:0.1f}/mpsa_flux_spectra.dat"

            # Load the wavelength array
            wave_local_path = (
                cache_dir / "MH+0.00/teff3500/logg3.0/mpsa_flux_spectra.dat"
            )
            wave_lib = np.loadtxt(wave_local_path, unpack=True, usecols=0)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(ff, tt, gg)
                            flux_dict[tt][gg][ff] = np.loadtxt(
                                cache_dir / fname, unpack=True, usecols=1
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the flux array
                fname = fname_str.format(feh, teff, logg)
                flux = np.loadtxt(cache_dir / fname, unpack=True, usecols=1)

        # Load `~speclib.Spectrum` object
        spec = Spectrum(
            spectral_axis=wave_lib * lib_wave_unit,
            flux=flux * lib_flux_unit,
        )
        # Ensure spectra are ordered correctly (problem for mps-atlas grid)
        idx_order = np.argsort(spec.wavelength)
        spec = Spectrum(
            spectral_axis=spec.wavelength[idx_order], flux=spec.flux[idx_order]
        )
        # Change to default units
        default_wave_unit = u.AA
        default_flux_unit = u.Unit("erg/(s * cm^2 * angstrom)")
        spec = Spectrum(
            spectral_axis=spec.spectral_axis.to(default_wave_unit),
            flux=spec.flux.to(default_flux_unit),
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
        if wavelength is not None:
            spec = spec.resample(wavelength)

        return spec

    @u.quantity_input(wavelength=u.AA)
    def resample(self, wavelength, taper=False):
        """
        Resample a spectrum while conserving flux.

        Parameters
        ----------
        wavelength : `~astropy.units.Quantity`
            A new wavelength axis. Unit must be specified.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
             A resampled spectrum.
        """
        if taper:
            force = "taper"
        else:
            force = None
        # Convert wavelengths arrays to same unit
        wave_old = self.wavelength.to(u.AA)
        wave_new = wavelength.to(u.AA)
        # waveunits = "angstrom"

        # The input value without a unit
        flux_old = self.flux.value

        # Make an observation object with synphot
        spectrum = sp.spectrum.SourceSpectrum(
            sp.models.Empirical1D, points=wave_old, lookup_table=flux_old
        )
        throughput = np.ones(len(wave_old)) * u.dimensionless_unscaled
        filt = sp.spectrum.SpectralElement(
            sp.models.Empirical1D,
            points=wave_old,
            lookup_table=throughput,
        )
        obs = sp.observation.Observation(spectrum, filt, binset=wave_new, force=force)

        # Save the new binned flux array in a `~speclib.Spectrum` object
        spec_new = Spectrum(
            spectral_axis=wavelength, flux=obs.binflux.value * self.flux.unit
        )

        return spec_new

    @u.quantity_input(delta_lambda=u.AA)
    def regularize(self, delta_lambda=None):
        """
        Resample a spectrum to a regularly spaced wavelength grid.

        Parameters
        ----------
        delta_lambda : `~astropy.units.Quantity`, optional
            The spacing of the new wavelength grid. Defaults to the smallest
            spacing in the orignal grid.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
            A resampled spectrum.
        """
        wl_min = self.wavelength.min()
        wl_max = self.wavelength.max()
        if delta_lambda is None:
            delta_lambda = np.diff(self.wavelength).min()
        n_points = int((wl_max.value - wl_min.value) / delta_lambda.value)
        regular_grid = (
            np.linspace(wl_min.value, wl_max.value, n_points) * delta_lambda.unit
        )
        spec_new = self.resample(regular_grid)

        return spec_new

    @u.quantity_input(spectral_resolution=u.AA)
    def set_spectral_resolution(self, spectral_resolution):
        """
        Set the spectral resolution by convolution with a Gaussian.

        Parameters
        ----------
        spectral_resolution : `~astropy.units.Quantity`
            The spectral resolution.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
            A spectrum with the desired spectral resolution.
        """
        from astropy.convolution import convolve, Gaussian1DKernel

        # # First, check if grid spacing is regular
        # delta_lambdas = np.unique(self.wavelength.diff())
        delta_lambda = np.unique(self.wavelength.diff()).min()

        kernel_size = (spectral_resolution / delta_lambda).value  # resolution elements
        kernel = Gaussian1DKernel(kernel_size)
        convolved_flux = convolve(self.flux, kernel, boundary="extend")
        spec_new = Spectrum(spectral_axis=self.wavelength, flux=convolved_flux)

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
        wavelength = self.wavelength
        flux = self.flux
        binned_fluxes = []
        for cen, wid in zip(center, width):
            lower = cen - wid / 2.0
            upper = cen + wid / 2.0
            idx = np.where((wavelength >= lower) & (wavelength <= upper))

            # Adjust for bins that are slightly wider than the wavelength range
            # due to discretization of the wavelength grid
            scale_factor = (upper - lower) / (wavelength[idx][-1] - wavelength[idx][0])

            binned_flux = (
                scale_factor
                * np.trapezoid(flux[idx], wavelength[idx])
                / (upper - lower)
            )
            binned_fluxes.append(binned_flux)
        binned_fluxes = u.Quantity(binned_fluxes)

        return BinnedSpectrum(center, width, binned_fluxes)


class BinnedSpectrum(object):
    """
    Represents a spectrum that has been binned into specified wavelength intervals.

    Useful for simulating photometric measurements or spectral channels.

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
    Represents a multi-dimensional grid of synthetic spectra from a model library.

    Provides fast access to preloaded spectra and supports trilinear interpolation
    in (Teff, logg, [Fe/H]) space.

    Attributes
    ----------
    teff_bds : iterable
        The lower and upper bounds of the model temperatures to load.

    logg_bds : iterable
        The lower and upper bounds of the model logg values to load.

    feh_bds : iterable
        The lower and upper bounds of the model [Fe/H] to load.

    wavelength : `~astropy.units.Quantity`
        Wavelengths of the interpolated spectrum.

    fluxes : dict
        The fluxes of the model grid. Sorted by fluxes[teff][logg][feh].

    model_grid : str
        Name of the model grid. Only `phoenix` is currently supported.

    Methods
    -------
    get_flux(teff, logg, feh, interpolate=True)
        Returns an interpolated flux array for the given teff, logg, and feh.

    """

    def _clip_bounds_to_grid(
        self,
        bounds,
        grid_values: np.ndarray,
        param_name: str,
    ) -> tuple[float, float]:
        """Clip the requested bounds to the available grid range.

        Parameters
        ----------
        bounds : iterable
            Two-element sequence specifying the requested lower and upper bound.
        grid_values : `numpy.ndarray`
            The discrete grid values available for the parameter.
        param_name : str
            Name of the parameter, used in warning messages.

        Returns
        -------
        tuple of float
            The bounds snapped to valid grid values.
        """

        values = np.asarray(bounds, dtype=float)
        if values.shape != (2,):
            raise ValueError(f"{param_name} must contain exactly two bounds.")

        lower = float(values[0])
        upper = float(values[1])
        if lower > upper:
            raise ValueError(
                f"{param_name} lower bound {lower} exceeds upper bound {upper}."
            )

        grid_min = float(np.min(grid_values))
        grid_max = float(np.max(grid_values))

        clipped_lower = float(np.clip(lower, grid_min, grid_max))
        clipped_upper = float(np.clip(upper, grid_min, grid_max))

        lower_candidates = grid_values[grid_values <= clipped_lower]
        if lower_candidates.size:
            aligned_lower = float(lower_candidates.max())
        else:
            aligned_lower = grid_min

        upper_candidates = grid_values[grid_values >= clipped_upper]
        if upper_candidates.size:
            aligned_upper = float(upper_candidates.min())
        else:
            aligned_upper = grid_max

        clipped_bounds = (aligned_lower, aligned_upper)

        if clipped_lower != lower or clipped_upper != upper:
            warnings.warn(
                f"{param_name} {(lower, upper)} truncated to valid range {clipped_bounds}",
                UserWarning,
            )

        return clipped_bounds

    def __init__(
        self,
        teff_bds,
        logg_bds,
        feh_bds,
        wavelength=None,
        spectral_resolution=None,
        model_grid="phoenix",
        **kwargs,
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

        spectral_resolution : `~astropy.units.Quantity`
            The spectral resolution.

        model_grid : str, optional
            Name of the model grid.
        """
        # First check that the model_grid is valid.
        self.model_grid = model_grid.lower()
        if self.model_grid not in utils.VALID_MODELS:
            raise NotImplementedError(
                f'"{self.model_grid}" model grid not found. '
                + "Currently supported models are: "
                + str(utils.VALID_MODELS)
            )

        # Define grid points
        self.grid_points = utils.GRID_POINTS[self.model_grid]
        self.grid_teffs = self.grid_points["grid_teffs"]
        self.grid_loggs = self.grid_points["grid_loggs"]
        self.grid_fehs = self.grid_points["grid_fehs"]

        # Then ensure that the bounds given are valid.
        self.teff_bds = self._clip_bounds_to_grid(teff_bds, self.grid_teffs, "teff_bds")
        self.logg_bds = self._clip_bounds_to_grid(logg_bds, self.grid_loggs, "logg_bds")
        self.feh_bds = self._clip_bounds_to_grid(feh_bds, self.grid_fehs, "feh_bds")

        # Define the values covered in the grid
        subset = np.logical_and(
            self.grid_teffs >= self.teff_bds[0], self.grid_teffs <= self.teff_bds[1]
        )
        self.teffs = self.grid_teffs[subset]

        subset = np.logical_and(
            self.grid_loggs >= self.logg_bds[0], self.grid_loggs <= self.logg_bds[1]
        )
        self.loggs = self.grid_loggs[subset]

        subset = np.logical_and(
            self.grid_fehs >= self.feh_bds[0], self.grid_fehs <= self.feh_bds[1]
        )
        self.fehs = self.grid_fehs[subset]

        # Load the fluxes
        fluxes = {}
        points = []
        data = []
        spec = None
        for teff in self.teffs:
            fluxes[teff] = {}
            for logg in self.loggs:
                fluxes[teff][logg] = {}
                for feh in self.fehs:
                    try:
                        spec = Spectrum.from_grid(
                            teff,
                            logg,
                            feh,
                            model_grid=self.model_grid,
                            **kwargs,
                        )
                    except ValueError:
                        # Skip combinations that do not exist in sparse grids
                        continue

                    # Set spectral resolution if specified
                    if spectral_resolution is not None:
                        spec = spec.regularize()
                        spec = spec.set_spectral_resolution(spectral_resolution)

                    # Resample the spectrum to the desired wavelength array
                    if wavelength is not None:
                        spec = spec.resample(wavelength)

                    fluxes[teff][logg][feh] = spec.flux
                    points.append([teff, logg, feh])
                    data.append(spec.flux.value)

        self.fluxes = fluxes

        if spec is not None:
            self.wavelength = spec.wavelength
            self.unit = spec.flux.unit
        else:
            self.wavelength = None
            self.unit = u.dimensionless_unscaled

        if points:
            self.points = np.array(points)
            self.data = np.vstack(data)
            self.interpolator = NearestNDInterpolator(self.points, self.data)
        else:
            self.points = np.empty((0, 3))
            self.data = np.empty((0,))
            self.interpolator = None

    def get_flux(self, teff, logg, feh, interpolate=True):
        """
        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model.

        interpolate : bool, optional
            Whether to interpolate between grid points. If `True` (default), the spectrum
            will be trilinearly interpolated in (Teff, logg, [Fe/H]) space. If `False`,
            the nearest available grid point will be used without interpolation.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            The interpolated flux array as a 1-D vector aligned to ``self.wavelength``.
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

        if self.model_grid in ["newera_gaia", "newera_jwst", "newera_lowres"]:
            if self.interpolator is None or not self.points.size:
                raise ValueError("SpectralGrid contains no spectra")

            if not interpolate:
                teff = utils.nearest(self.teffs, teff)
                logg = utils.nearest(self.loggs, logg)
                feh = utils.nearest(self.fehs, feh)
                flux = self.fluxes[teff][logg][feh]
            else:
                flux = utils.trilinear_interpolate(
                    self.fluxes,
                    (self.teffs, self.loggs, self.fehs),
                    (teff, logg, feh),
                )

            if not isinstance(flux, u.Quantity):
                flux = u.Quantity(flux, unit=self.unit, copy=False)
            else:
                flux = flux.to(self.unit)

            if flux.ndim != 1:
                raise ValueError(
                    "Interpolated flux has unexpected shape; expected a 1-D array"
                )

            return flux

        # If not interpolating, then just return the closest point in the grid.
        if not interpolate:
            teff = utils.nearest(self.teffs, teff)
            logg = utils.nearest(self.loggs, logg)
            feh = utils.nearest(self.fehs, feh)

            return self.fluxes[teff][logg][feh]

        # Otherwise, interpolate using the helper
        return utils.trilinear_interpolate(
            self.fluxes,
            (self.teffs, self.loggs, self.fehs),
            (teff, logg, feh),
        )

    def get_spectrum(self, teff, logg, feh, interpolate=True):
        """Deprecated alias for :meth:`get_flux`.

        .. deprecated:: 0.1.0
            Use :meth:`get_flux` instead.
        """

        warnings.warn(
            "SpectralGrid.get_spectrum is deprecated and will be removed in a "
            "future release. Use SpectralGrid.get_flux instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.get_flux(teff, logg, feh, interpolate=interpolate)


class BinnedSpectralGrid(object):
    """
    Represents a multi-dimensional grid of binned spectra from a model library.

    Supports trilinear interpolation over the parameter space.

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
        if self.model_grid not in utils.VALID_MODELS:
            raise NotImplementedError(
                f'"{self.model_grid}" model grid not found. '
                + "Currently supported models are: "
                + str(utils.VALID_MODELS)
            )

        # Define grid points
        self.grid_points = utils.GRID_POINTS[self.model_grid]
        self.grid_teffs = self.grid_points["grid_teffs"]
        self.grid_loggs = self.grid_points["grid_loggs"]
        self.grid_fehs = self.grid_points["grid_fehs"]

        # Then ensure that the bounds given are valid.
        teff_bds = np.array(teff_bds)
        teff_bds = (
            self.grid_teffs[self.grid_teffs <= teff_bds.min()].max(),
            self.grid_teffs[self.grid_teffs >= teff_bds.max()].min(),
        )
        self.teff_bds = teff_bds

        logg_bds = np.array(logg_bds)
        logg_bds = (
            self.grid_loggs[self.grid_loggs <= logg_bds.min()].max(),
            self.grid_loggs[self.grid_loggs >= logg_bds.max()].min(),
        )
        self.logg_bds = logg_bds

        feh_bds = np.array(feh_bds)
        feh_bds = (
            self.grid_fehs[self.grid_fehs <= feh_bds.min()].max(),
            self.grid_fehs[self.grid_fehs >= feh_bds.max()].min(),
        )
        self.feh_bds = feh_bds

        # Define the values covered in the grid
        subset = np.logical_and(
            self.grid_teffs >= self.teff_bds[0], self.grid_teffs <= self.teff_bds[1]
        )
        self.teffs = self.grid_teffs[subset]

        subset = np.logical_and(
            self.grid_loggs >= self.logg_bds[0], self.grid_loggs <= self.logg_bds[1]
        )
        self.loggs = self.grid_loggs[subset]

        subset = np.logical_and(
            self.grid_fehs >= self.feh_bds[0], self.grid_fehs <= self.feh_bds[1]
        )
        self.fehs = self.grid_fehs[subset]

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
                    bs = Spectrum.from_grid(
                        teff, logg, feh, model_grid=self.model_grid, **kwargs
                    ).bin(center, width)
                    fluxes[teff][logg][feh] = bs.flux
        self.fluxes = fluxes

    def get_spectrum(self, teff, logg, feh, interpolate=True):
        """
        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model.

        interpolate : bool, optional
            Whether to interpolate between grid points. If `True` (default), the spectrum
            will be trilinearly interpolated in (Teff, logg, [Fe/H]) space. If `False`,
            the nearest available grid point will be used without interpolation.

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

        # If not interpolating, then just return the closest point in the grid.
        if not interpolate:
            teff = utils.nearest(self.teffs, teff)
            logg = utils.nearest(self.loggs, logg)
            feh = utils.nearest(self.fehs, feh)

            return self.fluxes[teff][logg][feh]

        # Otherwise, interpolate using the helper
        return utils.trilinear_interpolate(
            self.fluxes,
            (self.teffs, self.loggs, self.fehs),
            (teff, logg, feh),
        )
