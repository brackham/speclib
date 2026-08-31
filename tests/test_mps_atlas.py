import hashlib
import io
import json
import shutil
import zipfile

import astropy.units as u
import numpy as np
import pytest

from speclib import download_mps_atlas_grid as public_download_mps_atlas_grid
from speclib import utils
from speclib.core import BinnedSpectralGrid, Spectrum, SpectralGrid


# Exact local ordering around the sole reversal in the v3 production archive.
WAVELENGTH_NM = np.array(
    [
        85.79684217280924,
        87.06691955221062,
        91.07195158078505,
        87.58558147727098,
        89.07737142457651,
        90.21800420659785,
        91.28453967606204,
    ]
)


def _spectrum_text(scale, wavelengths=WAVELENGTH_NM):
    rows = ["wavelength_nm fnu_at_1au"]
    rows.extend(
        f"{wavelength} {scale * value}"
        for wavelength, value in zip(
            wavelengths, np.arange(1.0, len(wavelengths) + 1.0)
        )
    )
    return "\n".join(rows) + "\n"


def _member_name(model_set, teff, logg, metallicity):
    return (
        f"{model_set}/MH{metallicity}/teff{teff:.0f}/logg{logg:.1f}/"
        "mpsa_flux_spectra.dat"
    )


def _write_archive(root, model_set, models):
    cache_dir = root / "mps-atlas" / model_set
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"{model_set}.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for (teff, logg, metallicity), scale in models.items():
            archive.writestr(
                _member_name(model_set, teff, logg, metallicity),
                _spectrum_text(scale),
            )
        archive.writestr(f"{model_set}/README.txt", "ignored")
    return cache_dir


def _cube_models(multiplier=1.0):
    models = {}
    for teff in (3500.0, 3600.0):
        for logg in (3.0, 3.5):
            for metallicity in (0.0, 0.1):
                scale = (
                    1.0
                    + (teff - 3500.0) / 100.0
                    + 2.0 * (logg - 3.0) / 0.5
                    + 4.0 * metallicity / 0.1
                )
                models[(teff, logg, metallicity)] = multiplier * scale
    return models


@pytest.fixture
def mps_atlas_cache(monkeypatch, tmp_path):
    _write_archive(tmp_path, "set1", _cube_models())
    _write_archive(tmp_path, "set2", _cube_models(multiplier=10.0))

    def use_cached_archive(model_set="set1", overwrite=False, library_root=None):
        del overwrite
        root = tmp_path if library_root is None else library_root
        return root / "mps-atlas" / utils.normalize_mps_atlas_set(model_set)

    monkeypatch.setattr(utils, "download_mps_atlas_grid", use_cached_archive)
    utils._MPS_ATLAS_INDEX_CACHE.clear()
    utils._MPS_ATLAS_WAVELENGTH_PLAN_CACHE.clear()
    utils.set_library_root(tmp_path)
    try:
        yield tmp_path
    finally:
        utils.set_library_root(None)
        utils._MPS_ATLAS_INDEX_CACHE.clear()
        utils._MPS_ATLAS_WAVELENGTH_PLAN_CACHE.clear()


def test_download_mps_atlas_grid_public_alias():
    assert public_download_mps_atlas_grid is utils.download_mps_atlas_grid


def test_resolve_mps_atlas_archive_uses_pinned_edmond_metadata(monkeypatch):
    metadata = utils.MPS_ATLAS_ARCHIVES["set2"]
    payload = {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "label": metadata["filename"],
                        "dataFile": {
                            "id": 12345,
                            "filesize": metadata["filesize"],
                            "md5": metadata["md5"],
                        },
                    }
                ]
            }
        }
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda url: Response(json.dumps(payload).encode()),
    )

    assert utils._resolve_mps_atlas_archive_url("set2") == (
        "https://edmond.mpg.de/api/access/datafile/12345"
    )


def test_resolve_mps_atlas_archive_rejects_changed_release(monkeypatch):
    payload = {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "label": "set1.zip",
                        "dataFile": {
                            "id": 1,
                            "filesize": 1,
                            "md5": "changed",
                        },
                    }
                ]
            }
        }
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda url: Response(json.dumps(payload).encode()),
    )
    with pytest.raises(RuntimeError, match="metadata has changed"):
        utils._resolve_mps_atlas_archive_url("set1")


def test_download_mps_atlas_grid_caches_sets_separately(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    source_cache = _write_archive(
        source_root, "set1", {(3500.0, 3.0, 0.0): 1.0}
    )
    source_archive = source_cache / "set1.zip"
    archive_bytes = source_archive.read_bytes()
    metadata = {
        "set1": {
            "filename": "set1.zip",
            "filesize": len(archive_bytes),
            "md5": hashlib.md5(archive_bytes).hexdigest(),
        },
        "set2": utils.MPS_ATLAS_ARCHIVES["set2"],
    }
    monkeypatch.setattr(utils, "MPS_ATLAS_ARCHIVES", metadata)
    monkeypatch.setattr(
        utils, "_resolve_mps_atlas_archive_url", lambda model_set: "mock://set1"
    )
    retrievals = []

    def fake_retrieve(**kwargs):
        retrievals.append(kwargs)
        target = kwargs["path"] / kwargs["fname"]
        shutil.copyfile(source_archive, target)
        return str(target)

    monkeypatch.setattr(utils.pooch, "retrieve", fake_retrieve)
    cache_root = tmp_path / "cache"

    result = utils.download_mps_atlas_grid("mps-atlas", library_root=cache_root)
    assert result == cache_root / "mps-atlas" / "set1"
    assert retrievals[0]["known_hash"] == f"md5:{metadata['set1']['md5']}"
    assert (result / ".set1.zip.verified-md5").exists()

    utils.download_mps_atlas_grid("set1", library_root=cache_root)
    assert len(retrievals) == 1

    stale = result / "stale.txt"
    stale.write_text("stale")
    set2_cache = cache_root / "mps-atlas" / "set2"
    set2_cache.mkdir()
    set2_marker = set2_cache / "keep.txt"
    set2_marker.write_text("set 2")
    utils.download_mps_atlas_grid("set1", overwrite=True, library_root=cache_root)
    assert len(retrievals) == 2
    assert not stale.exists()
    assert set2_marker.read_text() == "set 2"


def test_download_mps_atlas_grid_removes_partial_and_corrupt_files(
    monkeypatch, tmp_path
):
    metadata = {
        "set1": {"filename": "set1.zip", "filesize": 10, "md5": "0" * 32},
        "set2": utils.MPS_ATLAS_ARCHIVES["set2"],
    }
    monkeypatch.setattr(utils, "MPS_ATLAS_ARCHIVES", metadata)
    monkeypatch.setattr(
        utils, "_resolve_mps_atlas_archive_url", lambda model_set: "mock://set1"
    )

    def interrupted_retrieve(**kwargs):
        target = kwargs["path"] / kwargs["fname"]
        target.write_bytes(b"partial")
        raise OSError("connection interrupted")

    monkeypatch.setattr(utils.pooch, "retrieve", interrupted_retrieve)
    cache_root = tmp_path / "cache"
    archive_path = cache_root / "mps-atlas" / "set1" / "set1.zip"
    with pytest.raises(RuntimeError, match="connection interrupted"):
        utils.download_mps_atlas_grid("set1", library_root=cache_root)
    assert not archive_path.exists()

    archive_path.write_bytes(b"bad-length")
    with pytest.raises(ValueError, match="invalid cached archive was removed"):
        utils.download_mps_atlas_grid("set1", library_root=cache_root)
    assert not archive_path.exists()


def test_mps_atlas_published_axes_are_declared_for_both_sets():
    expected_teffs = np.arange(3500.0, 9100.0, 100.0)
    expected_loggs = np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])
    assert len(utils.MPS_ATLAS_GRID_FEHS) == 61
    np.testing.assert_array_equal(utils.MPS_ATLAS_GRID_TEFFS, expected_teffs)
    np.testing.assert_array_equal(utils.MPS_ATLAS_GRID_LOGGS, expected_loggs)
    assert utils.MPS_ATLAS_GRID_FEHS[0] == -5.0
    assert utils.MPS_ATLAS_GRID_FEHS[-1] == 1.5
    for selector in ("mps-atlas-set1", "mps-atlas-set2"):
        np.testing.assert_array_equal(
            utils.GRID_POINTS[selector]["grid_teffs"], expected_teffs
        )
        np.testing.assert_array_equal(
            utils.GRID_POINTS[selector]["grid_loggs"], expected_loggs
        )
        np.testing.assert_array_equal(
            utils.GRID_POINTS[selector]["grid_fehs"], utils.MPS_ATLAS_GRID_FEHS
        )


def test_mps_atlas_index_and_physical_unit_conversion(mps_atlas_cache):
    model_list = utils.load_mps_atlas_model_list("set1")
    set2_model_list = utils.load_mps_atlas_model_list("set2")
    assert model_list["model_set"] == "set1"
    assert model_list["combinations"].shape == (8, 3)
    np.testing.assert_array_equal(model_list["grid_teffs"], [3500.0, 3600.0])
    np.testing.assert_array_equal(model_list["grid_loggs"], [3.0, 3.5])
    np.testing.assert_array_equal(model_list["grid_fehs"], [0.0, 0.1])
    assert model_list["archive_path"] != set2_model_list["archive_path"]
    assert model_list["entries"] is not set2_model_list["entries"]
    assert not list(mps_atlas_cache.rglob("mpsa_flux_spectra.dat"))

    wavelength, flux = utils.load_mps_atlas_spectrum(3500.0, 3.0, 0.0, "set1")
    order = np.argsort(WAVELENGTH_NM, kind="stable")
    expected_wavelength = WAVELENGTH_NM[order] * u.nm
    expected_fnu = np.arange(1.0, len(WAVELENGTH_NM) + 1.0)[order] * u.erg / (
        u.s * u.cm**2 * u.Hz
    )
    expected_flux = (
        expected_fnu * ((1 * u.au) / (1 * u.R_sun)).decompose() ** 2
    ).to(
        u.erg / (u.s * u.cm**2 * u.AA),
        equivalencies=u.spectral_density(expected_wavelength),
    )

    assert wavelength.unit == u.nm
    assert flux.unit == u.erg / (u.s * u.cm**2 * u.AA)
    np.testing.assert_allclose(wavelength.value, WAVELENGTH_NM[order])
    np.testing.assert_allclose(flux.value, expected_flux.value)


def test_mps_atlas_wavelength_normalization_preserves_flux_pairing():
    raw_flux = np.arange(1.0, len(WAVELENGTH_NM) + 1.0)
    wavelength, flux, plan = utils._normalize_mps_atlas_spectral_axis(
        WAVELENGTH_NM, raw_flux, "production-pathology.dat"
    )
    order = np.argsort(WAVELENGTH_NM, kind="stable")

    np.testing.assert_array_equal(wavelength, WAVELENGTH_NM[order])
    np.testing.assert_array_equal(flux, raw_flux[order])
    assert np.all(np.diff(wavelength) > 0)

    _, scaled_flux, reused_plan = utils._normalize_mps_atlas_spectral_axis(
        WAVELENGTH_NM,
        2.0 * raw_flux,
        "same-grid-second-model.dat",
        plan=plan,
    )
    np.testing.assert_array_equal(scaled_flux, 2.0 * raw_flux[order])
    assert reused_plan is plan


def test_mps_atlas_identical_duplicate_wavelengths_are_collapsed():
    wavelength = np.array([10.0, 20.0, 20.0, 30.0])
    flux = np.array([1.0, 2.0, 2.0 * (1.0 + 5e-13), 3.0])

    normalized_wavelength, normalized_flux, _ = (
        utils._normalize_mps_atlas_spectral_axis(
            wavelength, flux, "equivalent-duplicates.dat"
        )
    )

    np.testing.assert_array_equal(normalized_wavelength, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(normalized_flux, [1.0, 2.0, 3.0])


def test_mps_atlas_conflicting_duplicate_wavelengths_raise():
    wavelength = np.array([10.0, 20.0, 20.0, 30.0])
    flux = np.array([1.0, 2.0, 2.01, 3.0])

    with pytest.raises(ValueError, match="materially different flux values"):
        utils._normalize_mps_atlas_spectral_axis(
            wavelength, flux, "conflicting-duplicates.dat"
        )


def test_mps_atlas_near_duplicate_wavelengths_are_not_merged():
    neighboring_value = np.nextafter(20.0, np.inf)
    wavelength = np.array([10.0, 20.0, neighboring_value, 30.0])
    flux = np.array([1.0, 2.0, 2.5, 3.0])

    normalized_wavelength, normalized_flux, _ = (
        utils._normalize_mps_atlas_spectral_axis(
            wavelength, flux, "near-duplicates.dat"
        )
    )

    np.testing.assert_array_equal(normalized_wavelength, wavelength)
    np.testing.assert_array_equal(normalized_flux, flux)
    assert len(normalized_wavelength) == 4


def test_mps_atlas_alias_and_explicit_sets_remain_distinct(mps_atlas_cache):
    alias = Spectrum.from_grid(
        3500.0, 3.0, 0.0, model_grid="mps-atlas", interpolate=False
    )
    set1 = Spectrum.from_grid(
        3500.0, 3.0, 0.0, model_grid="mps-atlas-set1", interpolate=False
    )
    set2 = Spectrum.from_grid(
        3500.0, 3.0, 0.0, model_grid="mps-atlas-set2", interpolate=False
    )

    assert alias.model_grid == "mps-atlas-set1"
    assert alias.model_set == "set1"
    assert set2.model_grid == "mps-atlas-set2"
    assert set2.model_set == "set2"
    assert alias.flux.ndim == 1
    assert np.all(np.diff(alias.wavelength) > 0 * u.AA)
    np.testing.assert_allclose(alias.flux.value, set1.flux.value)
    np.testing.assert_allclose(set2.flux.value, 10.0 * set1.flux.value)


def test_mps_atlas_spectrum_interpolation_and_nearest(mps_atlas_cache):
    interpolated = Spectrum.from_grid(
        3550.0, 3.25, 0.05, model_grid="mps-atlas-set1"
    )
    lower = Spectrum.from_grid(
        3500.0, 3.0, 0.0, model_grid="mps-atlas-set1"
    )
    nearest = Spectrum.from_grid(
        3501.0,
        3.01,
        0.001,
        model_grid="mps-atlas-set1",
        interpolate=False,
    )

    np.testing.assert_allclose(interpolated.flux.value, 4.5 * lower.flux.value)
    np.testing.assert_allclose(nearest.flux.value, lower.flux.value)


def test_mps_atlas_interpolation_never_uses_other_set(monkeypatch, tmp_path):
    _write_archive(
        tmp_path,
        "set1",
        {
            (3500.0, 3.0, 0.0): 1.0,
            (3500.0, 3.5, 0.0): 2.0,
            (3600.0, 3.0, 0.0): 3.0,
            (3600.0, 3.5, 0.0): 4.0,
        },
    )
    _write_archive(
        tmp_path,
        "set2",
        {
            (3500.0, 3.0, 0.0): 10.0,
            (3600.0, 3.5, 0.0): 40.0,
        },
    )

    def use_cached_archive(model_set="set1", overwrite=False, library_root=None):
        del overwrite, library_root
        return tmp_path / "mps-atlas" / utils.normalize_mps_atlas_set(model_set)

    monkeypatch.setattr(utils, "download_mps_atlas_grid", use_cached_archive)
    utils._MPS_ATLAS_INDEX_CACHE.clear()
    utils._MPS_ATLAS_WAVELENGTH_PLAN_CACHE.clear()
    utils.set_library_root(tmp_path)
    try:
        set1 = Spectrum.from_grid(
            3550.0, 3.25, 0.0, model_grid="mps-atlas-set1"
        )
        with pytest.raises(ValueError, match="selected set lacks a required corner"):
            Spectrum.from_grid(
                3550.0, 3.25, 0.0, model_grid="mps-atlas-set2"
            )
    finally:
        utils.set_library_root(None)
        utils._MPS_ATLAS_INDEX_CACHE.clear()
        utils._MPS_ATLAS_WAVELENGTH_PLAN_CACHE.clear()

    assert set1.model_set == "set1"


def test_mps_atlas_interpolation_rejects_incompatible_wavelengths(
    monkeypatch, tmp_path
):
    cache_dir = _write_archive(tmp_path, "set1", _cube_models())
    archive_path = cache_dir / "set1.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for parameters, scale in _cube_models().items():
            text = (
                _spectrum_text(scale, np.array([101.0, 201.0, 401.0]))
                if parameters == (3600.0, 3.5, 0.1)
                else _spectrum_text(scale)
            )
            archive.writestr(_member_name("set1", *parameters), text)

    def use_cached_archive(model_set="set1", overwrite=False, library_root=None):
        del model_set, overwrite, library_root
        return cache_dir

    monkeypatch.setattr(utils, "download_mps_atlas_grid", use_cached_archive)
    utils._MPS_ATLAS_INDEX_CACHE.clear()
    utils._MPS_ATLAS_WAVELENGTH_PLAN_CACHE.clear()
    utils.set_library_root(tmp_path)
    try:
        with pytest.raises(ValueError, match="do not share a wavelength grid"):
            Spectrum.from_grid(
                3550.0, 3.25, 0.05, model_grid="mps-atlas-set1"
            )
    finally:
        utils.set_library_root(None)
        utils._MPS_ATLAS_INDEX_CACHE.clear()
        utils._MPS_ATLAS_WAVELENGTH_PLAN_CACHE.clear()


def test_mps_atlas_spectral_and_binned_grids(mps_atlas_cache):
    grid = SpectralGrid(
        (3500.0, 3600.0),
        (3.0, 3.5),
        (0.0, 0.1),
        model_grid="mps-atlas-set2",
    )
    expected = Spectrum.from_grid(
        3550.0, 3.25, 0.05, model_grid="mps-atlas-set2"
    )
    assert grid.model_grid == "mps-atlas-set2"
    assert grid.model_set == "set2"
    np.testing.assert_allclose(
        grid.get_flux(3550.0, 3.25, 0.05).value,
        expected.flux.value,
    )

    binned = BinnedSpectralGrid(
        (3500.0, 3600.0),
        (3.0, 3.5),
        (0.0, 0.1),
        center=np.array([870.0, 905.0]) * u.AA,
        width=np.array([35.0, 40.0]) * u.AA,
        model_grid="mps-atlas-set1",
    )
    assert binned.model_set == "set1"
    assert binned.get_spectrum(3550.0, 3.25, 0.05).shape == (2,)


def test_mps_atlas_cache_root_override_and_invalid_requests(
    mps_atlas_cache, monkeypatch
):
    calls = []
    real_download = utils.download_mps_atlas_grid

    def record_download(model_set="set1", overwrite=False, library_root=None):
        calls.append((model_set, library_root))
        return real_download(model_set, overwrite, library_root)

    monkeypatch.setattr(utils, "download_mps_atlas_grid", record_download)
    utils.load_mps_atlas_model_list("mps-atlas-set2", library_root=mps_atlas_cache)
    assert calls[-1] == ("set2", mps_atlas_cache)

    with pytest.raises(ValueError, match="has no disk-integrated model"):
        utils.load_mps_atlas_spectrum(3700.0, 3.0, 0.0, "set1")
    with pytest.raises(ValueError, match="outside the available range"):
        Spectrum.from_grid(3400.0, 3.0, 0.0, model_grid="mps-atlas-set1")
    with pytest.raises(ValueError, match="Unknown MPS-ATLAS"):
        utils.normalize_mps_atlas_set("set3")
