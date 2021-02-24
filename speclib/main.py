import astropy.units as u
import astropy.io.fits as fits
import numpy as np
import os
import shutil
import urllib
from contextlib import closing
from specutils import Spectrum1D

__all__ = ['load_spectrum']


def download_file(remote_path, local_path, verbose=True):
    """
    Download a file via ftp.
    """
    if verbose:
        print(f'> Downloading {remote_path}')
    with closing(urllib.request.urlopen(remote_path)) as r:
        with open(local_path, 'wb') as f:
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
    idxs = np.argsort(np.abs(array-value))[0:2]
    return np.sort(array[idxs])


def load_flux_array(fname, cache_dir, ftp_url):
    """
    Load a flux array.
    """
    # Look for a local file first
    flux_local_path = os.path.join(
        cache_dir,
        fname
    )
    try:
        flux = fits.getdata(flux_local_path)
    # If that doesn't work, download the remote file
    except FileNotFoundError:
        feh_folder = 'Z' + fname[13:17]
        flux_remote_path = os.path.join(
            ftp_url,
            'HiResFITS/PHOENIX-ACES-AGSS-COND-2011',
            feh_folder,
            fname
        )
        download_file(flux_remote_path, flux_local_path)
        flux = fits.getdata(flux_local_path)

    return flux


def load_spectrum(teff, logg, feh=0, model_grid='phoenix'):
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

    model_grid : str
        Name of the model grid. Only `phoenix` is currently supported.

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        A spectrum for the specified parameters.
    """

    if model_grid.lower() == 'phoenix':
        cache_dir = os.path.join(
            os.path.expanduser('~'),
            '.speclib/libraries/phoenix/'
        )
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        ftp_url = 'ftp://phoenix.astro.physik.uni-goettingen.de'
        fname_str = (
            'lte{:05.0f}-{:0.2f}{:+0.1f}.' +
            'PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        )

        # Grid of effective temperatures
        grid_teffs = np.append(
            np.arange(2300, 7100, 100),
            np.arange(7200, 12200, 200)
        )

        # Grid of surface gravities
        grid_loggs = np.arange(0.0, 6.5, 0.5)

        # Grid of metallicities
        grid_fehs = np.array([
            -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, +0.5, +1.0
        ])

        # The convention of the PHOENIX model grids is that
        # [Fe/H] = 0.0 is written as a negative number.
        if feh == 0:
            feh = -0.0
    else:
        raise NotImplementedError(
            f'"{model_grid}" model grid not found. ' +
            'Only PHOENIX models are currently supported.'
        )

    # Load the wavelength array
    wave_local_path = os.path.join(
        cache_dir,
        'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    )
    try:
        wave = fits.getdata(wave_local_path)
    except FileNotFoundError:
        wave_remote_path = os.path.join(
            ftp_url,
            'HiResFITS',
            'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
        )
        download_file(wave_remote_path, wave_local_path)
        wave = fits.getdata(wave_local_path)

    teff_in_grid = teff in grid_teffs
    logg_in_grid = logg in grid_loggs
    feh_in_grid = feh in grid_fehs
    model_in_grid = all([
        teff_in_grid,
        logg_in_grid,
        feh_in_grid
    ])
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
            print(fname000, fname001)
            flux = interpolate([c0, c1], feh_bds, feh)
        else:
            flux = c0

    elif model_in_grid:
        # Load the flux array
        fname = fname_str.format(teff, logg, feh)
        flux = load_flux_array(fname, cache_dir, ftp_url)

    spec = Spectrum1D(
        spectral_axis=wave * u.AA,
        flux=flux * u.Unit('erg/(s * cm^2 * angstrom)')
    )

    return spec
