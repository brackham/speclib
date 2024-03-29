import astropy.units as u
import astropy.io.fits as fits
import itertools
import numpy as np
import os
import shutil
import urllib
from astropy.io import fits
from contextlib import closing
from urllib.error import URLError

__all__ = [
    "download_file",
    "download_phoenix_grid",
    "find_bounds",
    "interpolate",
    "load_flux_array",
    "nearest",
    "vac2air",
    "air2vac"
]


def download_file(remote_path, local_path, verbose=True):
    """
    Download a file via ftp.
    """
    if verbose:
        print(f"> Downloading {remote_path}")
    with closing(urllib.request.urlopen(remote_path)) as r:
        with open(local_path, "wb") as f:
            shutil.copyfileobj(r, f)


def download_phoenix_grid(overwrite=False):
    # Define the remote and local paths
    ftp_url = "ftp://phoenix.astro.physik.uni-goettingen.de"
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".speclib/libraries/phoenix/"
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fname_str = (
        "lte{:05.0f}-{:0.2f}{:+0.1f}."
        + "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    )

    # Define the parameter space
    param_combos = list(itertools.product(*GRID_POINTS['phoenix'].values()))

    # Iterate through the parameter space
    for combo in param_combos:
        fname = fname_str.format(*combo)
        local_path = os.path.join(cache_dir, fname)
        feh_folder = "Z" + fname[13:17]
        remote_path = os.path.join(
            ftp_url, "HiResFITS/PHOENIX-ACES-AGSS-COND-2011", feh_folder, fname
        )
        # If overwriting, just go ahead and download the file
        if overwrite:
            download_file(remote_path, local_path)

        # Otherwise, skip files that already exist locally
        else:
            try:
                _ = fits.getdata(local_path)
                continue

            # If that doesn't work, download the remote file
            except FileNotFoundError:
                try:
                    download_file(remote_path, local_path)
                    continue

                # Some low-G models are missing, e.g., lte05400-0.00+1.0...
                except URLError:
                    continue


def find_bounds(array, value):
    """
    Find and return the two nearest values in an array to a given value.
    """
    array = np.array(array)
    idxs = np.argsort(np.abs(array - value))[0:2]

    return np.sort(array[idxs])



def interpolate(fluxes, xlims, x):
    y0, y1 = fluxes
    x0, x1 = xlims
    w1 = (x - x0) / (x1 - x0)
    y = y0 * (1 - w1) + y1 * w1

    return y


def nearest(array, value):
    """
    Return the nearst values in an array to a given value.
    """
    array = np.array(array)
    idx = np.argmin(np.abs(array - value))

    return array[idx]


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
        try:
            download_file(flux_remote_path, flux_local_path)
            flux = fits.getdata(flux_local_path)
        # Some low-G models are missing, e.g., lte05400-0.00+1.0...
        except URLError:
            flux = None

    return flux


@u.quantity_input(wl_vac=u.AA)
def vac2air(wl_vac):
    """
    Convert vacuum to air wavelength.

    See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion.

    Parameters
    ----------
    wl_vac : `~astropy.units.Quantity`
        Vacuum wavelength.

    Returns
    -------
    wl_air : `~astropy.units.Quantity`
        Air wavelength.

    See Also
    --------
    air2vac
        Convert air to vacuum wavelength.
    """
    # Wavelengths must be specified in Å
    orig_unit = wl_vac.unit
    wl_vac = wl_vac.to(u.AA).value

    s = 10.0 ** 4 / wl_vac
    n = 1 + 0.0000834254 + 0.02406147 / (130.0 - s ** 2) + 0.00015998 / (38.9 - s ** 2)
    wl_air = wl_vac / n

    # Convert back to original unit, if necessary
    wl_air = (wl_air * u.AA).to(orig_unit)

    return wl_air


@u.quantity_input(wl_air=u.AA)
def air2vac(wl_air):
    """
    Convert air to vacuum wavelength.

    See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion.

    Parameters
    ----------
     wl_air : `~astropy.units.Quantity`
        Air wavelength.

    Returns
    -------
    wl_vac : `~astropy.units.Quantity`
        Vacuum wavelength.

    See Also
    --------
    vac2air
        Convert vacuum to air wavelength.
    """
    # Wavelengths must be specified in Å
    orig_unit = wl_air.unit
    wl_air = wl_air.to(u.AA).value

    s = 10.0 ** 4 / wl_air
    n = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s ** 2)
        + 0.0001599740894897 / (38.92568793293 - s ** 2)
    )
    wl_vac = wl_air * n

    # Convert back to original unit, if necessary
    wl_vac = (wl_vac * u.AA).to(orig_unit)

    return wl_vac

VALID_MODELS = [
    'drift-phoenix',
    'mps-atlas',
    'nextgen-solar',
    'phoenix',
    'sphinx'
]

GRID_POINTS = {
    'drift-phoenix': {
        # Grid of effective temperatures
        'grid_teffs': np.arange(1000, 3100, 100),
        # Grid of surface gravities
        'grid_loggs': np.arange(3.0, 6.5, 0.5),
        # Grid of metallicities
        'grid_fehs': np.array([-0.6, -0.3, -0.0, 0.3]),
    },

    'mps-atlas': {
        # Grid of effective temperatures
        'grid_teffs': np.arange(3500, 9100, 100),
        # Grid of surface gravities
        'grid_loggs': np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0]),
        # Grid of metallicities
        'grid_fehs': np.array([
            -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9,
            -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.95 -0.9, -0.85,
            -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3,
            -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ]),
        },

    'nextgen-solar': {
        # Grid of effective temperatures
        'grid_teffs': np.append(np.arange(1600., 4000., 100), np.arange(4000., 10200., 200)),
        # Grid of surface gravities
        'grid_loggs': np.arange(3.5, 6.0, 0.5),
        # Grid of metallicities
        'grid_fehs': np.array([0.0]),
    },

    'phoenix': {
        # Grid of effective temperatures
        'grid_teffs': np.append(np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)),
        # Grid of surface gravities
        'grid_loggs': np.arange(0.0, 6.5, 0.5),
        # Grid of metallicities
        'grid_fehs': np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, +0.5, +1.0]),
    },

    'sphinx': {
        # Grid of effective temperatures
        'grid_teffs': np.arange(2000., 4100., 100),
        # Grid of surface gravities
        'grid_loggs': np.arange(4.0, 5.75, 0.25),
        # Grid of metallicities
        'grid_fehs': np.arange(-1, 1.25, 0.25),
        # # Grid of CtoOs
        # 'grid_CtoOs': np.array([0.3, 0.5, 0.7, 0.9]),
    },
}

