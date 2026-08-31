import astropy.units as u
import astropy.io.fits as fits
import io
import itertools
import json
import numpy as np
import os
import pooch
import re
import shutil
import tarfile
import tempfile
import urllib
import warnings
import zipfile
from astropy.io import fits
from contextlib import closing
from pathlib import Path, PurePosixPath
from urllib.error import URLError

__all__ = [
    "download_file",
    "download_phoenix_grid",
    "download_newera_grid",
    "download_mps_atlas_grid",
    "download_sphinx_grid",
    "get_newera_record_id",
    "load_newera_model_list",
    "load_mps_atlas_model_list",
    "load_mps_atlas_spectrum",
    "find_bounds",
    "interpolate",
    "load_flux_array",
    "load_gaia_format_spectrum",
    "load_sphinx_model_list",
    "load_sphinx_spectrum",
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

SPHINX_RECORD_ID = "11392341"
SPHINX_ARCHIVE_FILENAME = "SPHINX_MAY2024_CLOUDFREE_UPDATED.tar.gz"
SPHINX_ARCHIVE_HASH = "md5:63d8dd8da5bca66c2384ce3059a29e01"
SPHINX_ARCHIVE_URL = (
    f"https://zenodo.org/api/records/{SPHINX_RECORD_ID}/files/"
    f"{SPHINX_ARCHIVE_FILENAME}/content"
)

MPS_ATLAS_DATASET_DOI = "doi:10.17617/3.NJ56TR"
MPS_ATLAS_DATASET_API_URL = (
    "https://edmond.mpg.de/api/datasets/:persistentId/"
    f"?persistentId={MPS_ATLAS_DATASET_DOI}"
)
MPS_ATLAS_DATAFILE_URL = "https://edmond.mpg.de/api/access/datafile/{datafile_id}"
MPS_ATLAS_ARCHIVES: dict[str, dict[str, str | int]] = {
    "set1": {
        "filename": "set1.zip",
        "filesize": 9_503_312_271,
        "md5": "91130159a9c486a27f1d67e176ae4c8b",
    },
    "set2": {
        "filename": "set2.zip",
        "filesize": 9_430_754_401,
        "md5": "76b9cb4530acb96bc3da979d0e9ec119",
    },
}
MPS_ATLAS_SELECTORS = {
    "mps-atlas": "set1",
    "mps-atlas-set1": "set1",
    "mps-atlas-set2": "set2",
}

MPS_ATLAS_GRID_TEFFS = np.arange(3500.0, 9100.0, 100.0)
MPS_ATLAS_GRID_LOGGS = np.array(
    [3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0]
)
MPS_ATLAS_GRID_FEHS = np.array(
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
)

_LIBRARY_ROOT: Path | None = None
_NEWERA_INDEX_CACHE: dict[tuple[Path, str], dict] = {}
_NEWERA_TAR_MEMBER_CACHE: dict[Path, dict[str, str]] = {}
_SPHINX_INDEX_CACHE: dict[Path, dict] = {}
_MPS_ATLAS_INDEX_CACHE: dict[tuple[Path, str], dict] = {}
_MPS_ATLAS_WAVELENGTH_PLAN_CACHE: dict[tuple[Path, str, tuple[int, int]], dict] = {}

_MPS_ATLAS_DUPLICATE_FLUX_RTOL = 1e-12

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

_SPHINX_MODEL_FILENAME_RE = re.compile(
    r"^Teff_(?P<teff>\d+(?:\.\d+)?)"
    r"_logg_(?P<logg>\d+(?:\.\d+)?)"
    r"_logZ_(?P<metallicity>[+-]\d+(?:\.\d+)?)"
    r"_CtoO_(?P<co_ratio>\d+(?:\.\d+)?)\.txt$"
)

_MPS_ATLAS_FLUX_MEMBER_RE = re.compile(
    r"(?:^|/)MH(?P<metallicity>[+-]?\d+(?:\.\d+)?)"
    r"/teff(?P<teff>\d+)"
    r"/logg(?P<logg>[+-]?\d+(?:\.\d+)?)"
    r"/mpsa_flux_spectra\.dat$"
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


def normalize_mps_atlas_set(value: str) -> str:
    """Return ``"set1"`` or ``"set2"`` for an MPS-ATLAS selector."""

    normalized = str(value).lower()
    if normalized in MPS_ATLAS_ARCHIVES:
        return normalized
    try:
        return MPS_ATLAS_SELECTORS[normalized]
    except KeyError as exc:
        accepted = sorted([*MPS_ATLAS_ARCHIVES, *MPS_ATLAS_SELECTORS])
        raise ValueError(
            f"Unknown MPS-ATLAS set or selector '{value}'. "
            f"Expected one of {accepted}."
        ) from exc


def canonical_mps_atlas_selector(value: str) -> str:
    """Return the explicit public selector for an MPS-ATLAS set."""

    return f"mps-atlas-{normalize_mps_atlas_set(value)}"


def _mps_atlas_cache_dir(
    model_set: str, library_root: str | Path | None = None
) -> Path:
    root = (
        get_library_root()
        if library_root is None
        else Path(library_root).expanduser()
    )
    return root / "mps-atlas" / normalize_mps_atlas_set(model_set)


def _resolve_mps_atlas_archive_url(model_set: str) -> str:
    """Resolve a pinned MPS-ATLAS archive through the Edmond dataset API."""

    model_set = normalize_mps_atlas_set(model_set)
    expected = MPS_ATLAS_ARCHIVES[model_set]
    try:
        with urllib.request.urlopen(MPS_ATLAS_DATASET_API_URL) as response:
            payload = json.load(response)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Unable to resolve the official MPS-ATLAS Edmond dataset: "
            f"{exc}"
        ) from exc

    try:
        files = payload["data"]["latestVersion"]["files"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "The Edmond MPS-ATLAS dataset response did not contain a released "
            "file list."
        ) from exc

    for entry in files:
        data_file = entry.get("dataFile", {})
        if entry.get("label") != expected["filename"]:
            continue

        actual_size = data_file.get("filesize")
        actual_md5 = data_file.get("md5") or data_file.get("checksum", {}).get(
            "value"
        )
        if actual_md5 is not None:
            actual_md5 = str(actual_md5).lower()
        if actual_size != expected["filesize"] or actual_md5 != expected["md5"]:
            raise RuntimeError(
                f"The official {expected['filename']} metadata has changed from "
                "the release pinned by speclib. Review the new MPS-ATLAS release "
                "before updating its size or checksum."
            )
        try:
            datafile_id = int(data_file["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"The Edmond metadata for {expected['filename']} has no valid "
                "data-file identifier."
            ) from exc
        return MPS_ATLAS_DATAFILE_URL.format(datafile_id=datafile_id)

    raise RuntimeError(
        f"The official Edmond record does not list {expected['filename']}."
    )


def _mps_atlas_verification_path(archive_path: Path) -> Path:
    return archive_path.with_name(f".{archive_path.name}.verified-md5")


def _clear_mps_atlas_runtime_cache(cache_dir: Path, model_set: str) -> None:
    """Discard in-process index and wavelength plans for one selected set."""

    resolved_cache = cache_dir.resolve()
    _MPS_ATLAS_INDEX_CACHE.pop((resolved_cache, model_set), None)
    stale_plan_keys = [
        key
        for key in _MPS_ATLAS_WAVELENGTH_PLAN_CACHE
        if key[0].parent == resolved_cache and key[1] == model_set
    ]
    for key in stale_plan_keys:
        _MPS_ATLAS_WAVELENGTH_PLAN_CACHE.pop(key, None)


def _verify_cached_mps_atlas_archive(archive_path: Path, model_set: str) -> None:
    """Validate a cached archive once, then retain a checksum marker."""

    metadata = MPS_ATLAS_ARCHIVES[normalize_mps_atlas_set(model_set)]
    expected_size = int(metadata["filesize"])
    expected_md5 = str(metadata["md5"])
    archive_stat = archive_path.stat()
    actual_size = archive_stat.st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Cached MPS-ATLAS archive {archive_path} has size {actual_size} "
            f"bytes; expected {expected_size}. Use overwrite=True to refresh it."
        )

    marker_path = _mps_atlas_verification_path(archive_path)
    expected_marker = f"{expected_md5} {actual_size} {archive_stat.st_mtime_ns}"
    if marker_path.exists() and marker_path.read_text().strip() == expected_marker:
        return

    actual_md5 = pooch.file_hash(archive_path, alg="md5")
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Cached MPS-ATLAS archive {archive_path} failed its published MD5 "
            "checksum. Use overwrite=True to refresh it."
        )
    marker_path.write_text(f"{expected_marker}\n")


def download_mps_atlas_grid(
    model_set: str = "set1",
    overwrite: bool = False,
    library_root: str | Path | None = None,
) -> Path:
    """Download one official MPS-ATLAS model-set archive.

    Parameters
    ----------
    model_set : {"set1", "set2"}, optional
        Discrete MPS-ATLAS flavor. Public model selectors are also accepted;
        ``"mps-atlas"`` resolves to Set 1.
    overwrite : bool, optional
        Clear only the selected set's cache and download it again.
    library_root : str or Path, optional
        Base cache directory. Defaults to :func:`get_library_root`.

    Returns
    -------
    Path
        Selected set cache directory under ``mps-atlas/<set>``.

    Notes
    -----
    Edmond distributes each set as one approximately 9.5 GB ZIP archive.
    This helper caches the ZIP without eagerly extracting its members.
    """

    model_set = normalize_mps_atlas_set(model_set)
    metadata = MPS_ATLAS_ARCHIVES[model_set]
    cache_dir = _mps_atlas_cache_dir(model_set, library_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / str(metadata["filename"])

    if overwrite:
        _clear_directory(cache_dir)
        _clear_mps_atlas_runtime_cache(cache_dir, model_set)
    elif archive_path.exists():
        try:
            _verify_cached_mps_atlas_archive(archive_path, model_set)
        except ValueError as exc:
            archive_path.unlink(missing_ok=True)
            _mps_atlas_verification_path(archive_path).unlink(missing_ok=True)
            _clear_mps_atlas_runtime_cache(cache_dir, model_set)
            raise ValueError(
                f"{exc} The invalid cached archive was removed; retry to "
                "download a fresh copy."
            ) from exc
        return cache_dir

    required_bytes = int(metadata["filesize"])
    available_bytes = shutil.disk_usage(cache_dir).free
    if available_bytes < required_bytes:
        raise OSError(
            f"Downloading MPS-ATLAS {model_set} requires at least "
            f"{required_bytes / 1024**3:.2f} GiB, but only "
            f"{available_bytes / 1024**3:.2f} GiB is available in {cache_dir}."
        )

    url = _resolve_mps_atlas_archive_url(model_set)
    try:
        retrieved = Path(
            pooch.retrieve(
                url=url,
                fname=str(metadata["filename"]),
                path=cache_dir,
                known_hash=f"md5:{metadata['md5']}",
                processor=None,
                progressbar=True,
            )
        )
        if retrieved.stat().st_size != required_bytes:
            raise ValueError(
                f"Downloaded {metadata['filename']} has an unexpected size."
            )
    except Exception as exc:  # pragma: no cover - real download failure
        archive_path.unlink(missing_ok=True)
        _mps_atlas_verification_path(archive_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Unable to download MPS-ATLAS {model_set} from Edmond: {exc}"
        ) from exc

    archive_stat = archive_path.stat()
    marker = f"{metadata['md5']} {archive_stat.st_size} {archive_stat.st_mtime_ns}"
    _mps_atlas_verification_path(archive_path).write_text(f"{marker}\n")
    _clear_mps_atlas_runtime_cache(cache_dir, model_set)
    return cache_dir


def _normalize_mps_atlas_key(
    teff: float, logg: float, metallicity: float
) -> tuple[float, float, float]:
    values = [round(float(value), 8) for value in (teff, logg, metallicity)]
    return tuple(0.0 if value == -0.0 else value for value in values)


def load_mps_atlas_model_list(
    model_set: str = "set1",
    *,
    library_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    """Index exact disk-integrated models in one MPS-ATLAS set archive."""

    model_set = normalize_mps_atlas_set(model_set)
    if cache_dir is None:
        cache_dir = download_mps_atlas_grid(model_set, library_root=library_root)
    else:
        cache_dir = Path(cache_dir).expanduser()

    cache_dir = Path(cache_dir).expanduser()
    archive_path = cache_dir / str(MPS_ATLAS_ARCHIVES[model_set]["filename"])
    if not archive_path.exists():
        raise FileNotFoundError(
            f"MPS-ATLAS {model_set} archive not found at {archive_path}. "
            f"Run download_mps_atlas_grid('{model_set}') first."
        )
    cache_key = (cache_dir.resolve(), model_set)
    signature = (archive_path.stat().st_size, archive_path.stat().st_mtime_ns)
    cached = _MPS_ATLAS_INDEX_CACHE.get(cache_key)
    if cached is not None and cached["archive_signature"] == signature:
        return cached

    entries: dict[tuple[float, float, float], str] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                match = _MPS_ATLAS_FLUX_MEMBER_RE.search(info.filename)
                if match is None:
                    continue
                key = _normalize_mps_atlas_key(
                    match.group("teff"),
                    match.group("logg"),
                    match.group("metallicity"),
                )
                if key in entries and entries[key] != info.filename:
                    raise ValueError(
                        f"MPS-ATLAS {model_set} contains duplicate disk-integrated "
                        f"members for Teff={key[0]}, logg={key[1]}, [M/H]={key[2]}."
                    )
                entries[key] = info.filename
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"Cached MPS-ATLAS {model_set} archive is not a valid ZIP file: "
            f"{archive_path}."
        ) from exc

    if not entries:
        raise FileNotFoundError(
            f"No mpsa_flux_spectra.dat members were found in {archive_path}."
        )

    combinations = np.array(sorted(entries), dtype=float)
    result = {
        "model_set": model_set,
        "archive_path": archive_path,
        "archive_signature": signature,
        "entries": entries,
        "combinations": combinations,
        "grid_teffs": np.unique(combinations[:, 0]),
        "grid_loggs": np.unique(combinations[:, 1]),
        "grid_fehs": np.unique(combinations[:, 2]),
    }
    _MPS_ATLAS_INDEX_CACHE[cache_key] = result
    return result


def _validate_mps_atlas_spectrum_arrays(
    wavelength: np.ndarray, flux: np.ndarray, member_name: str
) -> None:
    """Validate raw or normalized MPS-ATLAS wavelength/flux arrays."""

    if wavelength.ndim != 1 or flux.ndim != 1:
        raise ValueError(
            f"MPS-ATLAS member {member_name} wavelength and flux must be "
            "one-dimensional."
        )
    if wavelength.shape != flux.shape:
        raise ValueError(
            f"MPS-ATLAS member {member_name} has inconsistent wavelength and "
            "flux lengths."
        )
    if wavelength.size == 0:
        raise ValueError(f"MPS-ATLAS member {member_name} is empty.")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError(
            f"MPS-ATLAS member {member_name} contains nonfinite or nonpositive "
            "wavelengths."
        )
    if not np.all(np.isfinite(flux)):
        raise ValueError(
            f"MPS-ATLAS member {member_name} contains nonfinite flux values."
        )


def _collapse_mps_atlas_duplicate_wavelengths(
    wavelength: np.ndarray, flux: np.ndarray, member_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse exact duplicate coordinates only when their fluxes agree."""

    unique_wavelength, first_indices, counts = np.unique(
        wavelength, return_index=True, return_counts=True
    )
    duplicate_groups = np.flatnonzero(counts > 1)
    for group_index in duplicate_groups:
        start = first_indices[group_index]
        stop = start + counts[group_index]
        duplicate_flux = flux[start:stop]
        if not np.allclose(
            duplicate_flux,
            duplicate_flux[0],
            rtol=_MPS_ATLAS_DUPLICATE_FLUX_RTOL,
            atol=0.0,
        ):
            raise ValueError(
                f"MPS-ATLAS member {member_name} contains duplicate wavelength "
                f"{unique_wavelength[group_index]} nm with materially different "
                "flux values; these samples cannot be collapsed safely."
            )

    keep = np.ones(wavelength.size, dtype=bool)
    for start, count in zip(first_indices, counts):
        if count > 1:
            keep[start + 1 : start + count] = False
    return wavelength[keep], flux[keep]


def _normalize_mps_atlas_spectral_axis(
    wavelength: np.ndarray,
    flux: np.ndarray,
    member_name: str,
    *,
    plan: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return paired MPS-ATLAS samples on a strictly increasing axis.

    The released spectra share a nonmonotonic ODF wavelength sequence. The
    flux samples remain paired to their coordinates while a stable sort puts
    that sequence into ascending order. Exact duplicate coordinates are only
    collapsed when their fluxes agree to a tight relative tolerance.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    _validate_mps_atlas_spectrum_arrays(wavelength, flux, member_name)

    if plan is not None and np.array_equal(wavelength, plan["raw_wavelength"]):
        order = plan["order"]
    else:
        order = np.argsort(wavelength, kind="stable")

    sorted_wavelength = wavelength[order]
    sorted_flux = flux[order]
    normalized_wavelength, normalized_flux = (
        _collapse_mps_atlas_duplicate_wavelengths(
            sorted_wavelength, sorted_flux, member_name
        )
    )
    _validate_mps_atlas_spectrum_arrays(
        normalized_wavelength, normalized_flux, member_name
    )
    if normalized_wavelength.size > 1 and not np.all(
        np.diff(normalized_wavelength) > 0
    ):
        raise ValueError(
            f"MPS-ATLAS member {member_name} could not be normalized to a "
            "strictly increasing wavelength grid."
        )

    if plan is None or not np.array_equal(wavelength, plan["raw_wavelength"]):
        plan = {
            "raw_wavelength": wavelength.copy(),
            "order": order,
            "normalized_wavelength": normalized_wavelength.copy(),
        }
    else:
        if not np.array_equal(
            normalized_wavelength, plan["normalized_wavelength"]
        ):
            raise ValueError(
                f"MPS-ATLAS member {member_name} does not match the cached "
                "normalized wavelength grid."
            )
        normalized_wavelength = plan["normalized_wavelength"]

    return normalized_wavelength, normalized_flux, plan


def _mps_atlas_irradiance_to_surface_flux(
    wavelength: u.Quantity, irradiance: u.Quantity
) -> u.Quantity:
    """Convert archive F_nu at 1 AU for R_sun to stellar-surface F_lambda."""

    geometric_scale = ((1 * u.au) / (1 * u.R_sun)).decompose() ** 2
    surface_flux_nu = irradiance * geometric_scale
    return surface_flux_nu.to(
        u.erg / (u.s * u.cm**2 * u.AA),
        equivalencies=u.spectral_density(wavelength),
    )


def load_mps_atlas_spectrum(
    teff: float,
    logg: float,
    metallicity: float,
    model_set: str = "set1",
    *,
    library_root: str | Path | None = None,
) -> tuple[u.Quantity, u.Quantity]:
    """Load one exact disk-integrated MPS-ATLAS surface spectrum."""

    model_set = normalize_mps_atlas_set(model_set)
    model_list = load_mps_atlas_model_list(model_set, library_root=library_root)
    key = _normalize_mps_atlas_key(teff, logg, metallicity)
    try:
        member_name = model_list["entries"][key]
    except KeyError as exc:
        raise ValueError(
            f"MPS-ATLAS {model_set} has no disk-integrated model for "
            f"Teff={teff}, logg={logg}, [M/H]={metallicity}."
        ) from exc

    try:
        with zipfile.ZipFile(model_list["archive_path"]) as archive:
            with archive.open(member_name) as source:
                data = np.loadtxt(source, skiprows=1, ndmin=2)
    except (KeyError, OSError, zipfile.BadZipFile, ValueError) as exc:
        raise ValueError(
            f"Unable to read MPS-ATLAS {model_set} member {member_name}: {exc}"
        ) from exc

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"MPS-ATLAS member {member_name} must contain wavelength and F_nu "
            "columns."
        )
    wavelength_values = np.asarray(data[:, 0], dtype=float)
    irradiance_values = np.asarray(data[:, 1], dtype=float)
    plan_key = (
        model_list["archive_path"].resolve(),
        model_set,
        model_list["archive_signature"],
    )
    cached_plan = _MPS_ATLAS_WAVELENGTH_PLAN_CACHE.get(plan_key)
    wavelength_values, irradiance_values, normalization_plan = (
        _normalize_mps_atlas_spectral_axis(
            wavelength_values,
            irradiance_values,
            member_name,
            plan=cached_plan,
        )
    )
    if cached_plan is None:
        _MPS_ATLAS_WAVELENGTH_PLAN_CACHE[plan_key] = normalization_plan

    wavelength = wavelength_values * u.nm
    irradiance = irradiance_values * (u.erg / (u.s * u.cm**2 * u.Hz))
    return wavelength, _mps_atlas_irradiance_to_surface_flux(
        wavelength, irradiance
    )


def _extract_sphinx_spectra(tar_path: Path, cache_dir: Path) -> None:
    """Safely extract only SPHINX spectrum files from the V4 tarball."""

    cache_root = cache_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not _SPHINX_MODEL_FILENAME_RE.fullmatch(member_path.name)
            ):
                continue

            target = cache_dir.joinpath(*member_path.parts)
            if not target.resolve().is_relative_to(cache_root):
                raise ValueError(f"Unsafe path in SPHINX archive: {member.name}")
            if target.exists():
                continue

            source = archive.extractfile(member)
            if source is None:  # pragma: no cover - guarded by member.isfile()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with source, tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as destination:
                    temporary_path = Path(destination.name)
                    shutil.copyfileobj(source, destination)
                os.replace(temporary_path, target)
            except BaseException:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise


def download_sphinx_grid(
    overwrite: bool = False,
    library_root: str | Path | None = None,
) -> Path:
    """Download and extract the SPHINX I V4 spectral grid.

    The V4 archive is fetched from Zenodo record 11392341 and verified against
    the checksum published by Zenodo. Only spectrum ``.txt`` files are
    extracted; the downloaded tarball remains in the cache.

    Parameters
    ----------
    overwrite : bool, optional
        Remove the existing SPHINX cache before downloading and extracting it.
    library_root : str or Path, optional
        Base cache directory. Defaults to :func:`get_library_root`.

    Returns
    -------
    Path
        The SPHINX cache directory.
    """

    root = get_library_root() if library_root is None else Path(library_root)
    cache_dir = root.expanduser() / "sphinx"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        _clear_directory(cache_dir)

    archive_path = Path(
        pooch.retrieve(
            url=SPHINX_ARCHIVE_URL,
            fname=SPHINX_ARCHIVE_FILENAME,
            path=cache_dir,
            known_hash=SPHINX_ARCHIVE_HASH,
            processor=None,
            progressbar=True,
        )
    )
    _extract_sphinx_spectra(archive_path, cache_dir)
    _SPHINX_INDEX_CACHE.pop(cache_dir.resolve(), None)
    return cache_dir


def _normalize_sphinx_key(
    teff: float, logg: float, metallicity: float, co_ratio: float
) -> tuple[float, float, float, float]:
    return tuple(
        round(float(value), 8) for value in (teff, logg, metallicity, co_ratio)
    )


def load_sphinx_model_list(
    *,
    library_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    """Index the exact parameter combinations in an extracted SPHINX I V4 grid."""

    if cache_dir is None:
        root = get_library_root() if library_root is None else Path(library_root)
        cache_dir = root.expanduser() / "sphinx"
    else:
        cache_dir = Path(cache_dir).expanduser()

    cache_key = cache_dir.resolve()
    cached = _SPHINX_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    entries: dict[tuple[float, float, float, float], Path] = {}
    for path in cache_dir.rglob("*.txt") if cache_dir.exists() else ():
        match = _SPHINX_MODEL_FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        key = _normalize_sphinx_key(
            match.group("teff"),
            match.group("logg"),
            match.group("metallicity"),
            match.group("co_ratio"),
        )
        entries[key] = path

    if not entries:
        raise FileNotFoundError(
            f"No extracted SPHINX I V4 spectra found in {cache_dir}. "
            "Run download_sphinx_grid() first."
        )

    combinations = np.array(sorted(entries), dtype=float)
    result = {
        "entries": entries,
        "combinations": combinations,
        "grid_teffs": np.unique(combinations[:, 0]),
        "grid_loggs": np.unique(combinations[:, 1]),
        "grid_fehs": np.unique(combinations[:, 2]),
        "grid_co_ratios": np.unique(combinations[:, 3]),
    }
    _SPHINX_INDEX_CACHE[cache_key] = result
    return result


def load_sphinx_spectrum(
    teff: float,
    logg: float,
    metallicity: float,
    co_ratio: float,
    *,
    library_root: str | Path | None = None,
) -> tuple[u.Quantity, u.Quantity]:
    """Load one exact SPHINX I V4 model spectrum with physical units."""

    model_list = load_sphinx_model_list(library_root=library_root)
    key = _normalize_sphinx_key(teff, logg, metallicity, co_ratio)
    try:
        path = model_list["entries"][key]
    except KeyError as exc:
        raise ValueError(
            "SPHINX I V4 has no model for "
            f"Teff={teff}, logg={logg}, logZ={metallicity}, C/O={co_ratio}."
        ) from exc

    wavelength, flux = np.loadtxt(path, unpack=True)
    return wavelength * u.micron, flux * (u.W / (u.m**2 * u.m))


def download_newera_grid(
    grid_name: str,
    extract: bool | str = False,
    overwrite: bool = False,
    library_root: str | Path | None = None,
) -> Path:
    """
    Download a NewEra tarball if not already cached.

    Parameters
    ----------
    grid_name : str
        One of "newera_gaia", "newera_lowres", or "newera_jwst".
    extract : bool or {"missing", "all"}, optional
        Controls extraction behavior. ``False`` (default) keeps the tarball cached
        without unpacking. ``True`` or ``"missing"`` extracts only missing ``.txt``
        members. ``"all"`` extracts every member in the archive.
    overwrite : bool, optional
        If True, remove any existing files in the cache directory and re-download the
        tarball. Default is False.
    library_root : str or Path, optional
        Base cache directory. Defaults to :func:`get_library_root`.

    Returns
    -------
    Path
        Path to the cached directory (e.g., ~/.speclib/libraries/newera_jwst).

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

    extract_mode = "none"
    if isinstance(extract, str):
        extract_mode = extract.lower()
    elif extract:
        extract_mode = "missing"

    if extract_mode not in {"none", "missing", "all"}:
        raise ValueError(
            "extract must be False, True, 'missing', or 'all' "
            f"(received: {extract})"
        )

    if library_root is None:
        library_root = get_library_root()
    else:
        library_root = Path(library_root)

    # Cache location: ~/.speclib/libraries/{grid_name}/  (or custom path)
    cache_dir = library_root / grid_name
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
    if extract_mode != "none":
        print(f"🗂 Extracting archive to: {cache_dir}")
        if extract_mode == "all":
            extract_all_members(tar_path, cache_dir)
        else:
            extract_missing_txt_files(tar_path, cache_dir)

    return cache_dir


def extract_member_from_tar(
    tar_path: Path, member_name: str, extract_dir: Path
) -> Path:
    """
    Extract a single tar member into extract_dir.

    Parameters
    ----------
    tar_path : Path
        Path to the tar archive.
    member_name : str
        The member filename to extract (basename match).
    extract_dir : Path
        Directory to extract into.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError:
            cached = _NEWERA_TAR_MEMBER_CACHE.get(tar_path)
            if cached is None:
                cached = {Path(name).name: name for name in tar.getnames()}
                _NEWERA_TAR_MEMBER_CACHE[tar_path] = cached
            full_name = cached.get(member_name)
            if full_name is None:
                raise FileNotFoundError(
                    f"Member '{member_name}' not found in archive: {tar_path}"
                )
            member = tar.getmember(full_name)

        print(f"📦 Extracting: {member.name}")
        tar.extract(member, path=extract_dir)
        return extract_dir / Path(member.name).name


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


def extract_all_members(tar_path: Path, extract_dir: Path) -> None:
    """
    Extract all tar members into extract_dir.

    Parameters
    ----------
    tar_path : Path
        Path to the tar archive.
    extract_dir : Path
        Directory to extract into.
    """
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            print(f"📦 Extracting: {member.name}")
            tar.extract(member, path=extract_dir)


def _ensure_newera_txt_file(
    grid_name: str, filename: str, grid_dir: Path, library_root: Path
) -> Path:
    tar_path = grid_dir / NEWERA_TARBALLS[grid_name]
    if not tar_path.exists():
        download_newera_grid(
            grid_name, extract=False, library_root=library_root, overwrite=False
        )

    if not tar_path.exists():
        raise FileNotFoundError(f"NewEra archive not found: {tar_path}")

    return extract_member_from_tar(tar_path, filename, grid_dir)


def download_file(remote_path, local_path, verbose=True):
    """Download one remote file to a local path.

    Parameters
    ----------
    remote_path : str or path-like
        URL understood by :mod:`urllib.request`.
    local_path : str or path-like
        Destination filename. Its parent directory must already exist.
    verbose : bool, optional
        Print the remote URL before downloading.
    """
    if verbose:
        print(f"> Downloading {remote_path}")
    with closing(urllib.request.urlopen(remote_path)) as r:
        with open(local_path, "wb") as f:
            shutil.copyfileobj(r, f)


def download_phoenix_grid(overwrite=False):
    """Download the declared PHOENIX grid into the configured cache.

    Parameters
    ----------
    overwrite : bool, optional
        Download every declared file even when a readable cached FITS file is
        present. Missing upstream combinations are skipped.

    Notes
    -----
    This is a bulk operation over the full parameter axes declared by
    ``speclib``. Individual spectra are downloaded on demand by
    :meth:`speclib.Spectrum.from_grid`, which is usually preferable.
    """
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
        If the expected file is missing and cannot be extracted from the cached
        tarball.
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

    if not filepath.exists():
        _ensure_newera_txt_file(grid_name, fname, grid_dir, library_root)

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
    FileNotFoundError
        If the expected file is missing and cannot be extracted from the cached
        tarball.
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
        _ensure_newera_txt_file(grid_name, fname, grid_dir, library_root)

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
    "mps-atlas-set1",
    "mps-atlas-set2",
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

mps_atlas_grid = {
    "grid_teffs": MPS_ATLAS_GRID_TEFFS,
    "grid_loggs": MPS_ATLAS_GRID_LOGGS,
    "grid_fehs": MPS_ATLAS_GRID_FEHS,
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
    # MPS-ATLAS availability is refined from the selected ZIP at runtime.
    "mps-atlas": mps_atlas_grid,
    "mps-atlas-set1": mps_atlas_grid,
    "mps-atlas-set2": mps_atlas_grid,
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
        # Distinct values found in SPHINX I V4 filenames. The exact available
        # combinations are indexed from extracted files at runtime.
        "grid_teffs": np.arange(2000.0, 4100.0, 100),
        "grid_loggs": np.array(
            [4.0, 4.2, 4.25, 4.5, 4.7, 4.75, 5.0, 5.2, 5.25, 5.5]
        ),
        "grid_fehs": np.arange(-1, 1.25, 0.25),
        "grid_co_ratios": np.array([0.3, 0.5, 0.7, 0.9]),
    },
}
