import io
import tarfile

import astropy.units as u
import numpy as np
import pytest

from speclib import download_sphinx_grid as public_download_sphinx_grid
from speclib import utils
from speclib.core import BinnedSpectralGrid, Spectrum, SpectralGrid


def _spectrum_text(scale=1.0):
    return (
        "# Wavelength [um]       F [W/m2/m]\n"
        f"1.0 {1.0 * scale}\n"
        f"2.0 {2.0 * scale}\n"
        f"3.0 {3.0 * scale}\n"
    )


def _write_spectrum(cache_dir, teff, logg, metallicity, co_ratio, scale=1.0):
    model_dir = cache_dir / "NEWCORRECTED_CLOUDFREE"
    model_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"Teff_{teff:.1f}_logg_{logg}_logZ_{metallicity:+}"
        f"_CtoO_{co_ratio}.txt"
    )
    path = model_dir / name
    path.write_text(_spectrum_text(scale))
    return path


def _write_test_archive(path):
    content = _spectrum_text().encode()
    member_name = (
        "NEWCORRECTED_CLOUDFREE/"
        "Teff_3000.0_logg_4.0_logZ_+0.0_CtoO_0.5.txt"
    )
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

        ignored = b"not a spectrum"
        info = tarfile.TarInfo("NEWCORRECTED_CLOUDFREE/README.txt")
        info.size = len(ignored)
        archive.addfile(info, io.BytesIO(ignored))


def test_download_sphinx_grid_uses_v4_metadata_and_cache(monkeypatch, tmp_path):
    retrievals = []

    def fake_retrieve(**kwargs):
        retrievals.append(kwargs)
        archive_path = kwargs["path"] / kwargs["fname"]
        if not archive_path.exists():
            _write_test_archive(archive_path)
        return str(archive_path)

    monkeypatch.setattr(utils.pooch, "retrieve", fake_retrieve)

    result = utils.download_sphinx_grid(library_root=tmp_path)
    extracted = (
        result
        / "NEWCORRECTED_CLOUDFREE"
        / "Teff_3000.0_logg_4.0_logZ_+0.0_CtoO_0.5.txt"
    )
    assert result == tmp_path / "sphinx"
    assert extracted.exists()
    assert not (result / "NEWCORRECTED_CLOUDFREE" / "README.txt").exists()
    assert retrievals[0]["url"] == utils.SPHINX_ARCHIVE_URL
    assert retrievals[0]["fname"] == utils.SPHINX_ARCHIVE_FILENAME
    assert retrievals[0]["known_hash"] == utils.SPHINX_ARCHIVE_HASH

    extracted.write_text("cached")
    utils.download_sphinx_grid(library_root=tmp_path)
    assert extracted.read_text() == "cached"

    stale = result / "stale.txt"
    stale.write_text("remove me")
    utils.download_sphinx_grid(overwrite=True, library_root=tmp_path)
    assert not stale.exists()
    assert extracted.read_text().startswith("# Wavelength")


def test_download_sphinx_grid_public_alias():
    assert public_download_sphinx_grid is utils.download_sphinx_grid


def test_sphinx_extraction_failure_does_not_leave_final_file(monkeypatch, tmp_path):
    archive_path = tmp_path / "sphinx.tar.gz"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_test_archive(archive_path)
    target = (
        cache_dir
        / "NEWCORRECTED_CLOUDFREE"
        / "Teff_3000.0_logg_4.0_logZ_+0.0_CtoO_0.5.txt"
    )
    original_copyfileobj = utils.shutil.copyfileobj

    def interrupted_copy(source, destination):
        destination.write(b"partial")
        raise OSError("interrupted extraction")

    monkeypatch.setattr(utils.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="interrupted extraction"):
        utils._extract_sphinx_spectra(archive_path, cache_dir)

    assert not target.exists()
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []

    monkeypatch.setattr(utils.shutil, "copyfileobj", original_copyfileobj)
    utils._extract_sphinx_spectra(archive_path, cache_dir)
    assert target.read_text().startswith("# Wavelength")


def test_sphinx_filename_index_loading_units_and_missing_model(tmp_path):
    cache_dir = tmp_path / "sphinx"
    expected_path = _write_spectrum(cache_dir, 3000.0, 4.0, 0.0, 0.5)
    _write_spectrum(cache_dir, 3000.0, 4.2, 0.0, 0.7, scale=2.0)
    (cache_dir / "Teff_3000.0_logg_4.0_logZ_+0.0_CtoO_0.5_spectra.txt").write_text(
        _spectrum_text()
    )

    model_list = utils.load_sphinx_model_list(library_root=tmp_path)
    assert model_list["entries"][(3000.0, 4.0, 0.0, 0.5)] == expected_path
    assert model_list["combinations"].shape == (2, 4)
    np.testing.assert_array_equal(model_list["grid_co_ratios"], [0.5, 0.7])

    wavelength, flux = utils.load_sphinx_spectrum(
        3000.0, 4.0, 0.0, 0.5, library_root=tmp_path
    )
    assert wavelength.unit == u.micron
    assert flux.unit == u.W / (u.m**2 * u.m)
    np.testing.assert_allclose(wavelength.value, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(flux.value, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="has no model"):
        utils.load_sphinx_spectrum(
            3000.0, 4.5, 0.0, 0.5, library_root=tmp_path
        )


def test_sphinx_spectrum_and_grid_select_explicit_co_slice(tmp_path):
    cache_dir = tmp_path / "sphinx"
    for co_ratio, scale in ((0.5, 1.0), (0.7, 2.0)):
        _write_spectrum(cache_dir, 3000.0, 4.0, 0.0, co_ratio, scale)
        _write_spectrum(cache_dir, 3000.0, 4.5, 0.0, co_ratio, scale * 2.0)

    utils.set_library_root(tmp_path)
    try:
        with pytest.raises(ValueError, match="co_ratio must be specified"):
            Spectrum.from_grid(3000.0, 4.0, 0.0, model_grid="sphinx")

        spectrum = Spectrum.from_grid(
            3000.0,
            4.0,
            0.0,
            model_grid="sphinx",
            co_ratio=0.7,
            interpolate=False,
        )
        assert spectrum.wavelength.unit == u.AA
        assert spectrum.flux.unit.is_equivalent(u.erg / (u.s * u.cm**2 * u.AA))

        canonical = Spectrum.from_grid(
            3000.0,
            4.0,
            0.0,
            model_grid="sphinx",
            co_ratio=0.500004,
            interpolate=False,
        )
        exact = Spectrum.from_grid(
            3000.0,
            4.0,
            0.0,
            model_grid="sphinx",
            co_ratio=0.5,
            interpolate=False,
        )
        np.testing.assert_allclose(canonical.flux.value, exact.flux.value)

        grid_05 = SpectralGrid(
            (3000.0, 3000.0),
            (4.0, 4.5),
            (0.0, 0.0),
            model_grid="sphinx",
            co_ratio=0.5,
        )
        grid_07 = SpectralGrid(
            (3000.0, 3000.0),
            (4.0, 4.5),
            (0.0, 0.0),
            model_grid="sphinx",
            co_ratio=0.7,
        )
        canonical_grid = SpectralGrid(
            (3000.0, 3000.0),
            (4.0, 4.5),
            (0.0, 0.0),
            model_grid="sphinx",
            co_ratio=0.500004,
        )
    finally:
        utils.set_library_root(None)

    flux_05 = grid_05.get_flux(3000.0, 4.25, 0.0, interpolate=True)
    flux_07 = grid_07.get_flux(3000.0, 4.25, 0.0, interpolate=True)
    np.testing.assert_allclose(flux_07.value, 2.0 * flux_05.value)
    nearest = grid_05.get_flux(3000.0, 4.25, 0.0, interpolate=False)
    assert not np.allclose(nearest.value, flux_05.value)
    assert canonical_grid.co_ratio == 0.5


def test_sphinx_irregular_axis_uses_true_flanking_models(tmp_path):
    cache_dir = tmp_path / "sphinx"
    _write_spectrum(cache_dir, 3000.0, 4.0, 0.5, 0.9, scale=1.0)
    _write_spectrum(cache_dir, 3000.0, 4.2, 0.5, 0.9, scale=3.0)
    _write_spectrum(cache_dir, 3000.0, 4.25, 0.5, 0.9, scale=100.0)

    utils.set_library_root(tmp_path)
    try:
        lower = Spectrum.from_grid(
            3000.0, 4.0, 0.5, model_grid="sphinx", co_ratio=0.9
        )
        upper = Spectrum.from_grid(
            3000.0, 4.2, 0.5, model_grid="sphinx", co_ratio=0.9
        )
        interpolated = Spectrum.from_grid(
            3000.0, 4.19, 0.5, model_grid="sphinx", co_ratio=0.9
        )
        grid = SpectralGrid(
            (3000.0, 3000.0),
            (4.0, 4.25),
            (0.5, 0.5),
            model_grid="sphinx",
            co_ratio=0.9,
        )
    finally:
        utils.set_library_root(None)

    weight = (4.19 - 4.0) / (4.2 - 4.0)
    expected = lower.flux + weight * (upper.flux - lower.flux)
    np.testing.assert_allclose(interpolated.flux.value, expected.value)
    np.testing.assert_allclose(
        grid.get_flux(3000.0, 4.19, 0.5).value,
        expected.value,
    )


def test_sphinx_sparse_grid_interpolation_error(tmp_path):
    cache_dir = tmp_path / "sphinx"
    _write_spectrum(cache_dir, 3000.0, 4.0, 0.0, 0.5)
    _write_spectrum(cache_dir, 3100.0, 4.5, 0.0, 0.5)

    utils.set_library_root(tmp_path)
    try:
        grid = SpectralGrid(
            (3000.0, 3100.0),
            (4.0, 4.5),
            (0.0, 0.0),
            model_grid="sphinx",
            co_ratio=0.5,
        )
    finally:
        utils.set_library_root(None)

    with pytest.raises(ValueError, match="lacks a required corner"):
        grid.get_flux(3050.0, 4.25, 0.0, interpolate=True)
    nearest = grid.get_flux(3050.0, 4.25, 0.0, interpolate=False)
    assert nearest.shape == (3,)


def test_binned_sphinx_grid_uses_only_actual_sparse_combinations(tmp_path):
    cache_dir = tmp_path / "sphinx"
    _write_spectrum(cache_dir, 3000.0, 4.0, 0.0, 0.7)
    _write_spectrum(cache_dir, 3100.0, 4.5, 0.0, 0.7, scale=2.0)
    center = np.array([1.5, 2.5]) * u.micron
    width = np.ones(2) * u.micron

    utils.set_library_root(tmp_path)
    try:
        grid = BinnedSpectralGrid(
            (3000.0, 3100.0),
            (4.0, 4.5),
            (0.0, 0.0),
            center,
            width,
            model_grid="sphinx",
            co_ratio=0.7,
        )
    finally:
        utils.set_library_root(None)

    assert grid.points.shape == (2, 3)
    exact = grid.get_spectrum(3000.0, 4.0, 0.0)
    nearest = grid.get_spectrum(3050.0, 4.25, 0.0, interpolate=False)
    assert exact.shape == (2,)
    assert nearest.shape == (2,)
    with pytest.raises(ValueError, match="lacks a required corner"):
        grid.get_spectrum(3050.0, 4.25, 0.0, interpolate=True)
