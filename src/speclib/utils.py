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
    "air2vac",
]


def download_newera_file(
    teff, logg, zscale, alpha_scale, cache_dir=None, verbose=False
):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".speclib/libraries/newera/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Format parameter strings
    teff_str = f"{teff:05.0f}"
    logg_str = f"{-logg:.2f}"  # Minus sign before logg
    feh_str = f"{zscale:+.1f}".replace("+0.0", "-0.0").replace(
        "-0.0", "-0.0"
    )  # Always show sign
    alpha_str = f".alpha={alpha_scale:+.1f}" if alpha_scale != 0.0 else ""

    # Compose filename
    fname = f"lte{teff_str}{logg_str}{feh_str}{alpha_str}.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"
    url = f"https://www.fdr.uni-hamburg.de/record/16738/files/{fname}?download=1"
    local_path = os.path.join(cache_dir, fname)

    if not os.path.exists(local_path):
        download_file(url, local_path, verbose=verbose)

    return local_path


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
    cache_dir = os.path.join(os.path.expanduser("~"), ".speclib/libraries/phoenix/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fname_str = (
        "lte{:05.0f}-{:0.2f}{:+0.1f}." + "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    )

    # Define the parameter space
    param_combos = list(itertools.product(*GRID_POINTS["phoenix"].values()))

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


def download_newera_grid(
    teff_range=None,
    logg_range=None,
    feh_range=None,
    alpha_range=None,
    overwrite=False,
    verbose=False,
):
    """
    Download a subset of the NewEra model grid from the FDR Hamburg repository.

    Parameters
    ----------
    teff_range : tuple or None
        (min, max) Teff range to include. If None, use full grid.
    logg_range : tuple or None
        (min, max) log(g) range to include. If None, use full grid.
    feh_range : tuple or None
        (min, max) [Fe/H] range to include. If None, use full grid.
    alpha_range : tuple or None
        (min, max) [alpha/Fe] range to include. If None, use full grid.
    overwrite : bool
        Whether to overwrite files that already exist locally.
    verbose : bool
        Whether to print download progress and errors.
    """
    import itertools
    import os
    import numpy as np
    import warnings

    cache_dir = os.path.join(os.path.expanduser("~"), ".speclib/libraries/newera/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Define grid step sizes
    delta_teff = 100
    delta_logg = 0.5
    delta_feh = 0.5
    delta_alpha = 0.2

    # Construct grid lists from ranges
    teffs = (
        np.arange(teff_range[0], teff_range[1] + 1, delta_teff)
        if teff_range
        else GRID_POINTS["newera"]["grid_teffs"]
    )
    loggs = (
        np.arange(logg_range[0], logg_range[1] + 0.001, delta_logg)
        if logg_range
        else GRID_POINTS["newera"]["grid_loggs"]
    )
    fehs = (
        np.arange(feh_range[0], feh_range[1] + 0.001, delta_feh)
        if feh_range
        else GRID_POINTS["newera"]["grid_fehs"]
    )
    alphas = (
        np.arange(alpha_range[0], alpha_range[1] + 0.001, delta_alpha)
        if alpha_range
        else GRID_POINTS["newera"]["grid_alphas"]
    )

    param_combos = list(itertools.product(teffs, loggs, fehs, alphas))

    # Warn if attempting to download the full grid
    if not any([teff_range, logg_range, feh_range, alpha_range]):
        warnings.warn(
            "Downloading the full NewEra model grid requires approximately 4.5 TB of disk space. Consider specifying parameter ranges."
        )

    for teff, logg, feh, alpha in param_combos:
        # Skip α-enriched models outside the valid [M/H] range
        if alpha != 0.0 and not (-2.0 <= feh <= 0.0):
            continue

        try:
            # Format filename using the validated helper
            teff_str = f"{teff:05.0f}"
            logg_str = f"{-logg:.2f}"
            feh_str = f"{feh:+.1f}".replace("+0.0", "-0.0")
            alpha_str = f".alpha={alpha:+.1f}" if alpha != 0.0 else ""
            fname = f"lte{teff_str}{logg_str}{feh_str}{alpha_str}.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"
            local_path = os.path.join(cache_dir, fname)

            if verbose:
                print(f"⬇ Downloading {fname}")

            if overwrite or not os.path.exists(local_path):
                url = f"https://www.fdr.uni-hamburg.de/record/16738/files/{fname}?download=1"
                download_file(url, local_path, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"⚠ Failed to download {fname}: {e}")


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

    s = 10.0**4 / wl_vac
    n = 1 + 0.0000834254 + 0.02406147 / (130.0 - s**2) + 0.00015998 / (38.9 - s**2)
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

    s = 10.0**4 / wl_air
    n = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s**2)
        + 0.0001599740894897 / (38.92568793293 - s**2)
    )
    wl_vac = wl_air * n

    # Convert back to original unit, if necessary
    wl_vac = (wl_vac * u.AA).to(orig_unit)

    return wl_vac


VALID_MODELS = [
    "drift-phoenix",
    "mps-atlas",
    "newera",
    "nextgen-solar",
    "phoenix",
    "sphinx",
]

GRID_POINTS = {
    "drift-phoenix": {
        # Grid of effective temperatures
        "grid_teffs": np.arange(1000, 3100, 100),
        # Grid of surface gravities
        "grid_loggs": np.arange(3.0, 6.5, 0.5),
        # Grid of metallicities
        "grid_fehs": np.array([-0.6, -0.3, -0.0, 0.3]),
    },
    "mps-atlas": {
        # Grid of effective temperatures
        "grid_teffs": np.arange(3500, 9100, 100),
        # Grid of surface gravities
        "grid_loggs": np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0]),
        # Grid of metallicities
        "grid_fehs": np.array(
            [
                -5.0,
                -4.5,
                -4.0,
                -3.5,
                -3.0,
                -2.5,
                -2.4,
                -2.3,
                -2.2,
                -2.1,
                -2.0,
                -1.9,
                -1.8,
                -1.7,
                -1.6,
                -1.5,
                -1.4,
                -1.3,
                -1.2,
                -1.1,
                -1.0,
                -0.95 - 0.9,
                -0.85,
                -0.8,
                -0.75,
                -0.7,
                -0.65,
                -0.6,
                -0.55,
                -0.5,
                -0.45,
                -0.4,
                -0.35,
                -0.3,
                -0.25,
                -0.2,
                -0.15,
                -0.1,
                -0.05,
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
            ]
        ),
    },
    "newera": {
        "grid_teffs": np.arange(2300, 12000 + 100, 100),  # 2300K to 12000K
        "grid_loggs": np.arange(0.0, 6.0 + 0.5, 0.5),  # 0.0 to 6.0
        "grid_fehs": np.array(
            [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
        ),
        # α-enhanced models only for -2.0 ≤ [M/H] ≤ 0.0
        "grid_alphas": np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
    },
    "nextgen-solar": {
        # Grid of effective temperatures
        "grid_teffs": np.append(
            np.arange(1600.0, 4000.0, 100), np.arange(4000.0, 10200.0, 200)
        ),
        # Grid of surface gravities
        "grid_loggs": np.arange(3.5, 6.0, 0.5),
        # Grid of metallicities
        "grid_fehs": np.array([0.0]),
    },
    "phoenix": {
        # Grid of effective temperatures
        "grid_teffs": np.append(
            np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)
        ),
        # Grid of surface gravities
        "grid_loggs": np.arange(0.0, 6.5, 0.5),
        # Grid of metallicities
        "grid_fehs": np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, +0.5, +1.0]),
    },
    "sphinx": {
        # Grid of effective temperatures
        "grid_teffs": np.arange(2000.0, 4100.0, 100),
        # Grid of surface gravities
        "grid_loggs": np.arange(4.0, 5.75, 0.25),
        # Grid of metallicities
        "grid_fehs": np.arange(-1, 1.25, 0.25),
        # # Grid of CtoOs
        # 'grid_CtoOs': np.array([0.3, 0.5, 0.7, 0.9]),
    },
}
