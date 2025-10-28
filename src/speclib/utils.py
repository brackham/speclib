import astropy.units as u
import astropy.io.fits as fits
import io
import itertools
import numpy as np
import os
import pooch
import re
import shutil
import tarfile
import urllib
import warnings
from astropy.io import fits
from contextlib import closing
from pathlib import Path
from urllib.error import URLError

__all__ = [
    "download_file",
    "download_phoenix_grid",
    "download_newera_grid",
    "get_newera_record_id",
    "load_newera_model_list",
    "find_bounds",
    "interpolate",
    "load_flux_array",
    "load_gaia_format_spectrum",
    "trilinear_interpolate",
    "nearest",
    "vac2air",
    "air2vac",
    "get_library_root",
    "set_library_root",
]


LIBRARY_ENVVAR = "SPECLIB_LIBRARY_PATH"
NEWERA_RECORD_ENVVAR = "SPECLIB_NEWERA_RECORD_ID"
NEWERA_DEFAULT_RECORD_ID = "17935"

NEWERA_INDEX_FILENAME = "list_of_available_NewEraV3_models.txt"
NEWERA_TARBALLS: dict[str, str] = {
    "newera_gaia": "PHOENIX-NewEraV3-GAIA-DR4_v3.4-SPECTRA.tar.gz",
    "newera_jwst": "PHOENIX-NewEraV3-JWST-SPECTRA.tar.gz",
    "newera_lowres": "PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz",
}

_LIBRARY_ROOT: Path | None = None
_NEWERA_INDEX_CACHE: dict[tuple[Path, str], dict] = {}

_NEWERA_MODEL_LINE_RE = re.compile(
    r"(?P<filename>"
    r"lte(?P<teff>\d{5})"
    r"(?P<logg_sign>[+-])(?P<logg>\d\.\d{2})"
    r"(?P<feh_sign>[+-])(?P<feh>\d\.\d)"
    r"(?:\.alpha=(?P<alpha_sign>[+-])(?P<alpha>\d\.\d))?"
    r"\.PHOENIX-NewEra(?:V[0-9.]+)?-ACES-COND-(?P<year>\d{4})\.HSR\.h5"
    r")",
    re.IGNORECASE,
)


def get_library_root() -> Path:
    """Return the directory where spectral libraries are cached."""
    if _LIBRARY_ROOT is not None:
        return _LIBRARY_ROOT
    env = os.environ.get(LIBRARY_ENVVAR)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".speclib" / "libraries"


def set_library_root(path: str | Path | None) -> Path:
    """Set a custom cache directory for spectral libraries.

    Passing ``None`` clears any previously set value and reverts to using the
    environment variable or default location.
    """
    global _LIBRARY_ROOT
    if path is None:
        _LIBRARY_ROOT = None
    else:
        _LIBRARY_ROOT = Path(path).expanduser()
    return get_library_root()


def get_newera_record_id() -> str:
    """Return the record ID hosting the NewEra V3.4 release."""

    return os.environ.get(NEWERA_RECORD_ENVVAR, NEWERA_DEFAULT_RECORD_ID)


def _get_newera_base_url(record_id: str | None = None) -> str:
    record = record_id or get_newera_record_id()
    return f"https://www.fdr.uni-hamburg.de/record/{record}/files"


def _get_newera_file_url(filename: str, record_id: str | None = None) -> str:
    return f"{_get_newera_base_url(record_id)}/{filename}?download=1"


def _normalize_newera_key(
    teff: float, logg: float, feh: float, alpha: float
) -> tuple[int, float, float, float]:
    teff_key = int(round(teff))
    logg_key = round(float(logg), 2)
    feh_key = round(float(feh), 1)
    alpha_key = round(float(alpha), 1)

    if logg_key == -0.0:
        logg_key = 0.0
    if feh_key == -0.0:
        feh_key = 0.0
    if alpha_key == -0.0:
        alpha_key = 0.0

    return teff_key, logg_key, feh_key, alpha_key


def _ensure_newera_index(cache_dir: Path, record_id: str) -> Path:
    """Ensure the NewEra index file is present locally, downloading if needed."""
    existing = sorted(cache_dir.glob("list_of_available_NewEra*.txt"), reverse=True)
    if existing:
        return existing[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    base_url = _get_newera_base_url(record_id)
    url = f"{base_url}/{NEWERA_INDEX_FILENAME}?download=1"

    try:
        pooch.retrieve(
            url=url,
            fname=NEWERA_INDEX_FILENAME,
            path=cache_dir,
            known_hash=None,
            progressbar=True,
        )
        return cache_dir / NEWERA_INDEX_FILENAME
    except Exception as exc:  # pragma: no cover - network dependent
        raise FileNotFoundError(f"Unable to download NewEra model list: {exc}")


def load_newera_model_list(
    *,
    library_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    record_id: str | None = None,
) -> dict:
    """Return metadata for available NewEra high-resolution spectra.

    Parameters
    ----------
    library_root : str or Path, optional
        Base library directory. If omitted, the configured library root is used.
    cache_dir : str or Path, optional
        Directory containing the HSR cache. Defaults to ``library_root / "newera"``.
    record_id : str, optional
        Override the FDR Hamburg record identifier. Defaults to :func:`get_newera_record_id`.

    Returns
    -------
    dict
        Dictionary with keys ``"entries"`` (mapping parameter tuples to filenames),
        ``"path"`` (the index file path), and ``"record_id"``.
    """

    if cache_dir is None:
        if library_root is None:
            base = get_library_root()
        else:
            base = Path(library_root).expanduser()
        cache_dir = base / "newera"
    else:
        cache_dir = Path(cache_dir).expanduser()

    record = record_id or get_newera_record_id()
    cache_key = (cache_dir.resolve(), record)
    if cache_key in _NEWERA_INDEX_CACHE:
        return _NEWERA_INDEX_CACHE[cache_key]

    index_path = _ensure_newera_index(cache_dir, record)

    entries: dict[tuple[int, float, float, float], str] = {}
    with open(index_path, "r") as handle:
        for line in handle:
            match = _NEWERA_MODEL_LINE_RE.search(line)
            if not match:
                continue

            teff = int(match.group("teff"))
            logg = -float(match.group("logg_sign") + match.group("logg"))
            feh = float(match.group("feh_sign") + match.group("feh"))

            alpha_sign = match.group("alpha_sign")
            if alpha_sign:
                alpha = float(alpha_sign + match.group("alpha"))
            else:
                alpha = 0.0

            key = _normalize_newera_key(teff, logg, feh, alpha)
            entries[key] = match.group("filename")

    result = {"entries": entries, "path": index_path, "record_id": record}
    _NEWERA_INDEX_CACHE[cache_key] = result
    return result


def _resolve_newera_tarball(
    grid_name: str, cache_dir: Path, record_id: str, overwrite: bool = False
) -> Path:
    """Ensure the NewEra tarball for the given grid is present locally, downloading if needed."""
    tarball_name = NEWERA_TARBALLS[grid_name]
    tar_path = cache_dir / tarball_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if tar_path.exists():
        if overwrite:
            print(f"🔁 Overwriting existing tarball: {tar_path.name}")
            tar_path.unlink()
        else:
            return tar_path

    url = _get_newera_file_url(tarball_name, record_id)
    try:
        print(f"⬇ Downloading {tarball_name} from {url}")
        path_str = pooch.retrieve(
            url=url,
            fname=tarball_name,
            path=cache_dir,
            known_hash=None,
            processor=None,
            progressbar=True,
        )
        return Path(path_str)
    except Exception as exc:  # pragma: no cover - network dependent
        raise FileNotFoundError(
            f"Unable to download NewEra archive for '{grid_name}': {exc}"
        )


def download_newera_file(
    teff, logg, zscale, alpha_scale, cache_dir=None, verbose=False
):
    if cache_dir is None:
        cache_dir = get_library_root() / "newera"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    record_id = get_newera_record_id()
    model_list = load_newera_model_list(cache_dir=cache_dir, record_id=record_id)
    key = _normalize_newera_key(teff, logg, zscale, alpha_scale)

    try:
        fname = model_list["entries"][key]
    except KeyError as exc:
        raise FileNotFoundError(
            "Requested NewEra model is not listed in the available V3.4 grid: "
            f"Teff={teff}, logg={logg}, [M/H]={zscale}, [alpha/Fe]={alpha_scale}"
        ) from exc

    local_path = cache_dir / fname

    if not local_path.exists():
        url = _get_newera_file_url(fname, record_id)
        download_file(url, local_path, verbose=verbose)

    return local_path


def _clear_directory(path: Path) -> None:
    """Remove all files and subdirectories within *path* without deleting *path* itself."""

    if not path.exists():
        return

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def download_newera_grid(
    grid_name: str, extract: bool = True, overwrite: bool = False
) -> Path:
    """
    Download and extract a NewEra tarball if not already cached.

    Parameters
    ----------
    grid_name : str
        One of "newera_gaia", "newera_lowres", or "newera_jwst".
    extract : bool, optional
        If True, extract the tarball after download. Default is True.
    overwrite : bool, optional
        If True, remove any existing files in the cache directory and re-download the
        tarball. Default is False.

    Returns
    -------
    Path
        Path to the extracted directory (e.g., ~/.speclib/libraries/newera_jwst).

    Note
    ----
    This function fetches reduced-resolution NewEra grids from the PHOENIX/1D
    NewEra **V3.4** release (record 17935) hosted by FDR Hamburg, suitable for
    most applications (e.g., forward modeling, calibration).
    """
    if grid_name not in NEWERA_TARBALLS:
        raise ValueError(
            f"Unknown grid_name '{grid_name}'. Must be one of {list(NEWERA_TARBALLS.keys())}"
        )

    # Cache location: ~/.speclib/libraries/{grid_name}/  (or custom path)
    cache_dir = get_library_root() / grid_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    record_id = get_newera_record_id()

    tarball_name = NEWERA_TARBALLS[grid_name]
    tar_path = cache_dir / tarball_name
    existing_before = tar_path.exists()

    if overwrite:
        print(f"🧹 Removing existing files in: {cache_dir}")
        _clear_directory(cache_dir)

    tar_path = _resolve_newera_tarball(
        grid_name, cache_dir, record_id, overwrite=overwrite
    )

    if existing_before and tar_path.exists() and not overwrite:
        print(f"✅ Using cached NewEra archive: {tar_path.name}")

    # Extract tarball if requested
    if extract:
        print(f"🗂 Extracting archive to: {cache_dir}")
        extract_missing_txt_files(tar_path, cache_dir)

    return cache_dir


def extract_missing_txt_files(tar_path: Path, extract_dir: Path) -> None:
    """
    Extract only missing .txt files from a tarball into extract_dir.

    Parameters
    ----------
    tar_path : Path
        Path to the tar archive.
    extract_dir : Path
        Directory to extract into.
    """
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".txt"):
                target_file = extract_dir / Path(member.name).name
                if not target_file.exists():
                    print(f"📦 Extracting: {member.name}")
                    tar.extract(member, path=extract_dir)


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
    cache_dir = get_library_root() / "phoenix"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname_str = (
        "lte{:05.0f}-{:0.2f}{:+0.1f}." + "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    )

    # Define the parameter space
    param_combos = list(itertools.product(*GRID_POINTS["phoenix"].values()))

    # Iterate through the parameter space
    for combo in param_combos:
        fname = fname_str.format(*combo)
        local_path = cache_dir / fname
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


def download_newera_hsr_subset(
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

    Warning
    -------
    This function accesses the **full-resolution NewEra HSR grid**, which totals ~4.5 TB.
    Use only when you need high-resolution spectra over specific parameter ranges.
    """
    import itertools
    import os
    import numpy as np
    import warnings

    record_id = get_newera_record_id()
    cache_dir = get_library_root() / "newera"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_list = load_newera_model_list(cache_dir=cache_dir, record_id=record_id)
    entries = model_list["entries"]

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

        key = _normalize_newera_key(teff, logg, feh, alpha)
        fname = entries.get(key)
        if not fname:
            continue

        local_path = cache_dir / fname

        if verbose:
            print(f"⬇ Downloading {fname}")

        if overwrite or not local_path.exists():
            url = _get_newera_file_url(fname, record_id)
            try:
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


def _flanking_vals(grid, value):
    grid = np.asarray(grid)
    lower = grid[grid <= value].max() if np.any(grid <= value) else grid.min()
    upper = grid[grid >= value].min() if np.any(grid >= value) else grid.max()
    return lower, upper


def trilinear_interpolate(fluxes, grid_axes, query_point):
    """Perform trilinear interpolation on a nested flux dictionary."""
    teff_grid, logg_grid, feh_grid = grid_axes
    teff, logg, feh = query_point

    t_bds = _flanking_vals(teff_grid, teff)
    g_bds = _flanking_vals(logg_grid, logg)
    f_bds = _flanking_vals(feh_grid, feh)

    c000 = fluxes[t_bds[0]][g_bds[0]][f_bds[0]]
    c100 = fluxes[t_bds[1]][g_bds[0]][f_bds[0]]
    c010 = fluxes[t_bds[0]][g_bds[1]][f_bds[0]]
    c110 = fluxes[t_bds[1]][g_bds[1]][f_bds[0]]
    c001 = fluxes[t_bds[0]][g_bds[0]][f_bds[1]]
    c101 = fluxes[t_bds[1]][g_bds[0]][f_bds[1]]
    c011 = fluxes[t_bds[0]][g_bds[1]][f_bds[1]]
    c111 = fluxes[t_bds[1]][g_bds[1]][f_bds[1]]

    if t_bds[0] != t_bds[1]:
        c00 = interpolate([c000, c100], t_bds, teff)
        c10 = interpolate([c010, c110], t_bds, teff)
        c01 = interpolate([c001, c101], t_bds, teff)
        c11 = interpolate([c011, c111], t_bds, teff)
    else:
        c00, c10, c01, c11 = c000, c010, c001, c011

    if g_bds[0] != g_bds[1]:
        c0 = interpolate([c00, c10], g_bds, logg)
        c1 = interpolate([c01, c11], g_bds, logg)
    else:
        c0, c1 = c00, c01

    if f_bds[0] != f_bds[1]:
        return interpolate([c0, c1], f_bds, feh)
    return c0


def load_flux_array(fname, cache_dir, ftp_url):
    """
    Load a flux array.
    """
    cache_dir = Path(cache_dir)
    flux_local_path = cache_dir / fname
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


def load_newera_wavelength_array(
    teff, logg, z, alpha=0.0, grid_name="newera_jwst", library_root=None
):
    """
    Load the wavelength array from a GAIA-format NewEra spectrum file,
    matching the given Teff and logg within a file specified by Z and alpha.

    Parameters
    ----------
    teff : float
        Effective temperature (K).
    logg : float
        Log surface gravity (dex).
    z : float
        Metallicity as mass fraction (e.g., 0.0).
    alpha : float, optional
        Alpha enhancement (e.g., 0.2).
    grid_name : str, optional
        One of "newera_gaia", "newera_jwst", or "newera_lowres".
    library_root : str or Path, optional
        Path to the base `.speclib/libraries/` directory.
        Defaults to ~/.speclib/libraries/.

    Returns
    -------
    np.ndarray
        Wavelength values in nanometers (unitless NumPy array).

    Raises
    ------
    FileNotFoundError
        If the expected file is missing.
    ValueError
        If a valid header is not found.
    """
    if not np.isclose(alpha, 0.0):
        warnings.warn(
            f"Alpha-enhanced models (alpha={alpha}) are not yet supported for grid '{grid_name}'. "
            "Behavior may be unreliable or fail.",
            UserWarning,
        )
    if library_root is None:
        library_root = get_library_root()
    else:
        library_root = Path(library_root)

    if grid_name not in ["newera_gaia", "newera_jwst", "newera_lowres"]:
        raise ValueError(f"Invalid grid_name '{grid_name}'")

    grid_dir = library_root / grid_name

    # Construct file name
    prefix = {
        "newera_gaia": "PHOENIX-NewEraV3-GAIA-DR4_v3.4-SPECTRA",
        "newera_jwst": "PHOENIX-NewEraV3-JWST-SPECTRA",
        "newera_lowres": "PHOENIX-NewEraV3-LowRes-SPECTRA",
    }[grid_name]

    # Format Z string: NewEra always uses Z-0.0 (not Z+0.0)
    z_str = "Z-0.0" if np.isclose(z, 0.0) else f"Z{z:+.1f}"

    # Format alpha string
    if np.isclose(alpha, 0.0):
        fname = f"{prefix}.{z_str}.txt"
    else:
        alpha_str = f"alpha={alpha:.1f}"
        fname = f"{prefix}.{z_str}.{alpha_str}.txt"

    filepath = grid_dir / fname

    # Trigger download and extraction if file is missing
    if not filepath.exists():
        download_newera_grid(grid_name)

    # Raise error if the file still doesn't exist.
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            header = np.loadtxt(io.StringIO(line), dtype="S41")
            try:
                header_teff = float(header[12])
                header_logg = float(header[13])
            except Exception:
                continue

            if np.isclose(header_teff, teff, atol=1.0) and np.isclose(
                header_logg, logg, atol=0.1
            ):
                res = float(header[7])
                wl_start = float(header[9])
                wl_end = float(header[10])
                nwl = int(header[8])
                wl = np.linspace(
                    wl_start, wl_end, num=int((wl_end - wl_start) / res) + 1
                )
                if wl.shape[0] != nwl:
                    raise ValueError(
                        f"Wavelength point mismatch: header says {nwl}, got {wl.shape[0]}"
                    )
                return wl

            # Skip the corresponding flux line
            f.readline()

    raise ValueError(f"No matching spectrum found in file for Teff={teff}, logg={logg}")


def load_newera_flux_array(
    teff, logg, z, alpha=0.0, grid_name="newera_jwst", library_root=None
):
    """
    Load a flux array from a bundled GAIA-format NewEra spectrum file,
    matching the given Teff and logg within a file specified by Z and alpha.

    Parameters
    ----------
    teff : float
        Effective temperature (K).
    logg : float
        Log surface gravity (dex).
    z : float
        Metallicity as mass fraction (e.g., 0.0).
    alpha : float, optional
        Alpha enhancement (e.g., 0.2).
    grid_name : str, optional
        One of "newera_gaia", "newera_jwst", or "newera_lowres".
    library_root : str or Path, optional
        Path to the base `.speclib/libraries/` directory.
        Defaults to ~/.speclib/libraries/.

    Returns
    -------
    np.ndarray
        Flux values (unitless NumPy array). Fluxes are in W/m^2/nm.

    Raises
    ------
    ValueError
        If no matching model is found in the file.
    """
    if library_root is None:
        library_root = get_library_root()
    else:
        library_root = Path(library_root)

    if grid_name not in ["newera_gaia", "newera_jwst", "newera_lowres"]:
        raise ValueError(f"Invalid grid_name '{grid_name}'")

    grid_dir = library_root / grid_name

    # Construct file name
    prefix = {
        "newera_gaia": "PHOENIX-NewEraV3-GAIA-DR4_v3.4-SPECTRA",
        "newera_jwst": "PHOENIX-NewEraV3-JWST-SPECTRA",
        "newera_lowres": "PHOENIX-NewEraV3-LowRes-SPECTRA",
    }[grid_name]

    # Format Z string: NewEra always uses Z-0.0 (not Z+0.0)
    z_str = "Z-0.0" if np.isclose(z, 0.0) else f"Z{z:+.1f}"

    # Format alpha string
    if np.isclose(alpha, 0.0):
        fname = f"{prefix}.{z_str}.txt"
    else:
        alpha_str = f"alpha={alpha:.1f}"
        fname = f"{prefix}.{z_str}.{alpha_str}.txt"

    filepath = grid_dir / fname

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            header = np.loadtxt(io.StringIO(line), dtype="S41")
            try:
                header_teff = float(header[12])
                header_logg = float(header[13])
            except Exception:
                continue

            flux_line = f.readline()
            if np.isclose(header_teff, teff, atol=1.0) and np.isclose(
                header_logg, logg, atol=0.1
            ):
                flux = np.loadtxt(io.StringIO(flux_line), unpack=True)
                return flux

    raise ValueError(f"No matching flux found in file for Teff={teff}, logg={logg}")


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
    "newera_gaia",
    "newera_jwst",
    "newera_lowres",
    "nextgen-solar",
    "phoenix",
    "sphinx",
]

# Shared grid values for all NewEra subtypes
newera_grid = {
    "grid_teffs": np.arange(2300, 12001, 100),
    "grid_loggs": np.arange(0.0, 6.1, 0.5),
    "grid_fehs": np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]),
    # α-enhanced models only for -2.0 ≤ [M/H] ≤ 0.0
    "grid_alphas": np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
}

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
                -0.95,
                -0.9,
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
    "newera": newera_grid,
    "newera_gaia": newera_grid,
    "newera_jwst": newera_grid,
    "newera_lowres": newera_grid,
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
