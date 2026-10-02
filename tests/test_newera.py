"""Offline contracts derived from the released V3 inventory/header format."""

import json
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest

from speclib import Spectrum, SpectralGrid, BinnedSpectralGrid, utils


REDUCED = ("newera_gaia", "newera_jwst", "newera_lowres")


@pytest.mark.parametrize("selector", ("newera", *REDUCED))
def test_regular_backbone(selector):
    teffs = utils.GRID_POINTS[selector]["grid_teffs"]
    np.testing.assert_array_equal(teffs[teffs <= 7000], list(range(2300, 7001, 100)))
    np.testing.assert_array_equal(teffs[teffs > 7000], list(range(7200, 12001, 200)))
    np.testing.assert_array_equal(np.diff(teffs[teffs <= 7000]), 100)
    np.testing.assert_array_equal(np.diff(teffs[teffs >= 7000]), 200)
    for temperature in (7100, 7300, 7500, 7700, 7900, 8100, 8300,
                        3350, 5770, 6050, 6060, 9602):
        assert temperature not in teffs


def test_release_mapping_and_override(monkeypatch):
    monkeypatch.delenv("SPECLIB_NEWERA_RECORD_ID", raising=False)
    assert utils.get_newera_record_id() == "18108"
    assert utils.NEWERA_INDEX_FILENAME == "list_of_available_NewEraV3_models.txt"
    assert utils.NEWERA_TARBALLS == {
        "newera_gaia": "PHOENIX-NewEraV3-GAIA-DR4_v3.4-SPECTRA.tar.gz",
        "newera_jwst": "PHOENIX-NewEraV3-JWST-SPECTRA.tar.gz",
        "newera_lowres": "PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz",
    }
    for filename in utils.NEWERA_TARBALLS.values():
        assert utils._get_newera_file_url(filename) == (
            f"https://www.fdr.uni-hamburg.de/record/18108/files/{filename}?download=1"
        )
    monkeypatch.setenv("SPECLIB_NEWERA_RECORD_ID", "custom")
    assert utils.get_newera_record_id() == "custom"
    assert "/record/custom/" in utils._get_newera_file_url("model.txt")


@pytest.fixture
def hsr_inventory(monkeypatch, tmp_path):
    monkeypatch.delenv("SPECLIB_NEWERA_RECORD_ID", raising=False)
    monkeypatch.setattr(utils, "_NEWERA_INDEX_CACHE", {})
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    main_names = (
        "lte03350-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
        "lte07000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
        "lte07200-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
        "lte09602-3.95-0.5.Vega.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
    )
    main = "indx filename checksum filesize download link\n" + "\n".join(
        f"{i} {name} checksum 100 https://www.fdr.uni-hamburg.de/record/"
        f"{'16738' if i == 0 else '17670'}/files/{name}"
        for i, name in enumerate(main_names)
    )
    additional = "lte02500-3.50+0.5.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"
    catalog = {"contents": [
        {"key": name, "links": {"self": utils._get_newera_file_url(name, "17936")}}
        for name in (additional, main_names[0])
    ]}
    requests = []

    def retrieve(*, url, fname, path, **kwargs):
        requests.append(url)
        target = Path(path) / fname
        if fname == utils.NEWERA_INDEX_FILENAME:
            target.write_text(main)
        else:
            assert url == utils.NEWERA_ADDITIONAL_CATALOG_URL
            assert fname == "newera_additional_17936.json"
            target.write_text(json.dumps(catalog))
        return str(target)

    monkeypatch.setattr(utils.pooch, "retrieve", retrieve)
    return tmp_path, requests


def test_hsr_native_inventory_and_explicit_supplements(hsr_inventory):
    root, requests = hsr_inventory
    models = utils.load_newera_model_list()
    assert (3350, 5.0, 0.0, 0.0) in models["entries"]
    assert (9602, 3.95, -0.5, 0.0) in models["entries"]
    assert (2500, 3.5, 0.5, 0.0) in models["entries"]
    assert (3350, 4.5, 0.0, 0.0) not in models["entries"]
    assert models["additional_record_id"] == "17936"
    assert models["additional_path"] == root / "newera" / "newera_additional_17936.json"
    assert "/record/16738/" in models["urls"][(3350, 5.0, 0.0, 0.0)]
    assert "/record/17936/" in models["urls"][(2500, 3.5, 0.5, 0.0)]
    assert requests == [utils._get_newera_file_url(utils.NEWERA_INDEX_FILENAME),
                        utils.NEWERA_ADDITIONAL_CATALOG_URL]
    assert utils.load_newera_model_list() is models
    assert len(requests) == 2


def test_custom_hsr_record_does_not_merge_canonical_supplements(hsr_inventory, monkeypatch):
    _, requests = hsr_inventory
    monkeypatch.setenv("SPECLIB_NEWERA_RECORD_ID", "custom")
    models = utils.load_newera_model_list()
    assert models["record_id"] == "custom"
    assert "additional_path" not in models
    assert (2500, 3.5, 0.5, 0.0) not in models["entries"]
    assert len(requests) == 1 and "/record/custom/" in requests[0]


@pytest.mark.parametrize("parameters, source", [
    ((3350, 5.0, 0.0, 0.0), "16738"),
    ((2500, 3.5, 0.5, 0.0), "17936"),
    ((9602, 3.95, -0.5, 0.0), "17670"),
])
def test_hsr_download_uses_authoritative_source(hsr_inventory, monkeypatch, parameters, source):
    downloads = []
    def download(url, path, **kwargs):
        downloads.append(url)
        path.write_bytes(b"model")
    monkeypatch.setattr(utils, "download_file", download)
    path = utils.download_newera_file(*parameters)
    assert path.read_bytes() == b"model"
    assert len(downloads) == 1 and f"/record/{source}/" in downloads[0]
    with pytest.raises(FileNotFoundError, match="native inventory"):
        utils.download_newera_file(3350, 4.5, 0.0, 0.0)
    assert len(downloads) == 1


def test_hsr_subset_filters_native_tuples_including_specials(hsr_inventory, monkeypatch):
    downloads = []
    monkeypatch.setattr(utils, "download_file", lambda url, path, **kwargs: downloads.append(url))
    utils.download_newera_hsr_subset(teff_range=(3350, 3350))
    assert len(downloads) == 1 and "lte03350-5.00" in downloads[0]
    assert "/record/16738/" in downloads[0]


@pytest.fixture
def reduced_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    # Two distinct native gravities at 3350 K; sparse regular grid at 3300 K.
    for selector in REDUCED:
        directory = tmp_path / selector
        directory.mkdir()
        filename = utils.NEWERA_TARBALLS[selector].removesuffix(".tar.gz") + ".Z-0.0.txt"
        text = ""
        for teff, logg in ((3300, 5.0), (3350, 4.95), (3350, 5.0),
                           (3400, 4.5), (7000, 5.0), (7200, 5.0)):
            text += (
                "star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 "
                f"{teff} {logg} 1 0\n{teff} {teff + logg} {teff + 10}\n"
            )
        (directory / filename).write_text(text)
    return tmp_path


@pytest.mark.parametrize("selector", REDUCED)
@pytest.mark.parametrize("interpolate", (False, True))
def test_reduced_exact_special_uses_native_headers(reduced_headers, selector, interpolate):
    spectrum = Spectrum.from_grid(3350, 5.0, 0.0, model_grid=selector, interpolate=interpolate)
    np.testing.assert_allclose(spectrum.flux.to_value(u.W / (u.m**2 * u.nm)), [3350, 3355, 3360])
    assert 3350 not in spectrum.grid_teffs
    with pytest.raises(ValueError, match="No matching"):
        utils.load_newera_flux_array(3350, 4.96, 0.0, grid_name=selector)


def test_hsr_exact_vega_and_specials(hsr_inventory):
    models = utils.load_newera_model_list()
    cache = hsr_inventory[0] / "newera"
    for key in ((3350, 5.0, 0.0, 0.0), (9602, 3.95, -0.5, 0.0)):
        with h5py.File(cache / models["entries"][key], "w") as handle:
            handle["PHOENIX_SPECTRUM/wl"] = [9800, 9900, 10000]
            handle["PHOENIX_SPECTRUM/flux"] = np.log10([1, 2, 3])
        spectrum = Spectrum.from_grid(*key[:3], model_grid="newera", interpolate=False)
        assert spectrum.flux.shape == (3,)
        assert key[0] not in spectrum.grid_teffs


@pytest.mark.parametrize("selector", REDUCED)
def test_regular_grid_does_not_promote_specials_or_missing_tuples(reduced_headers, selector):
    grid = SpectralGrid((3300, 3400), (4.5, 5.0), (0.0, 0.0), model_grid=selector)
    np.testing.assert_array_equal(grid.teffs, [3300, 3400])
    np.testing.assert_array_equal(grid.points, [[3300, 5.0, 0.0], [3400, 4.5, 0.0]])
    assert 3350 not in grid.fluxes


def test_binned_regular_grid_skips_missing_newera_tuples(reduced_headers):
    grid = BinnedSpectralGrid(
        (3300, 3400), (4.5, 5.0), (0.0, 0.0),
        center=np.array([990]) * u.nm, width=np.array([30]) * u.nm,
        model_grid="newera_gaia",
    )
    np.testing.assert_array_equal(grid.teffs, [3300, 3400])
    assert 3350 not in grid.fluxes
    assert 4.5 in grid.fluxes[3300] and not grid.fluxes[3300][4.5]
