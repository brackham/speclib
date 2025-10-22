from pathlib import Path

import numpy as np
import pooch
import pytest

import speclib.utils as utils


def _write_newera_index(cache_dir: Path, lines: list[str]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "list_of_available_NewEraV3.4_models.txt"
    index_path.write_text("\n".join(lines) + "\n")
    utils._NEWERA_INDEX_CACHE.clear()
    return index_path


def test_load_newera_model_list_parses_entries(tmp_path):
    cache_dir = tmp_path / "newera"
    fname0 = "lte05000-4.50-0.0.PHOENIX-NewEraV3.4-ACES-COND-2024.HSR.h5"
    fname_alpha = "lte05000-4.50-0.0.alpha=+0.4.PHOENIX-NewEraV3.4-ACES-COND-2024.HSR.h5"
    _write_newera_index(cache_dir, [fname0, fname_alpha])

    result = utils.load_newera_model_list(cache_dir=cache_dir, record_id="17935")
    entries = result["entries"]

    assert entries[(5000, 4.5, 0.0, 0.0)] == fname0
    assert entries[(5000, 4.5, 0.0, 0.4)] == fname_alpha


def test_download_newera_file_uses_index_and_downloads(tmp_path, monkeypatch):
    cache_dir = tmp_path / "newera"
    fname = "lte05000-4.50-0.0.PHOENIX-NewEraV3.4-ACES-COND-2024.HSR.h5"
    _write_newera_index(cache_dir, [fname])

    calls: dict[str, str] = {}

    def fake_download(remote, local, verbose=False):
        calls["remote"] = remote
        Path(local).write_text("data")

    monkeypatch.setattr(utils, "download_file", fake_download)
    monkeypatch.setenv("SPECLIB_NEWERA_RECORD_ID", "99999")

    try:
        path = utils.download_newera_file(5000, 4.5, 0.0, 0.0, cache_dir=cache_dir)
    finally:
        monkeypatch.delenv("SPECLIB_NEWERA_RECORD_ID", raising=False)

    assert path.exists()
    assert calls["remote"].endswith(f"/{fname}?download=1")
    assert "/99999/" in calls["remote"]


def test_download_newera_file_respects_existing_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "newera"
    fname = "lte05000-4.50-0.0.PHOENIX-NewEraV3.4-ACES-COND-2024.HSR.h5"
    _write_newera_index(cache_dir, [fname])
    (cache_dir / fname).write_text("cached")

    monkeypatch.setattr(
        utils,
        "download_file",
        lambda *args, **kwargs: pytest.fail("download should not be called when cache exists"),
    )

    path = utils.download_newera_file(5000, 4.5, 0.0, 0.0, cache_dir=cache_dir)
    assert path.read_text() == "cached"


def test_load_newera_wavelength_array_reads_mock_file(tmp_path):
    grid_dir = tmp_path / "newera_jwst"
    grid_dir.mkdir()
    fname = grid_dir / "PHOENIX-NewEra-JWST-SPECTRA.Z-0.0.txt"

    header_tokens = ["0"] * 41
    header_tokens[7] = "1.0"
    header_tokens[8] = "4"
    header_tokens[9] = "100.0"
    header_tokens[10] = "103.0"
    header_tokens[12] = "5000"
    header_tokens[13] = "4.5"

    with open(fname, "w") as fh:
        fh.write(" ".join(header_tokens) + "\n")
        fh.write("1.0 2.0 3.0 4.0\n")

    wl = utils.load_newera_wavelength_array(5000, 4.5, 0.0, grid_name="newera_jwst", library_root=tmp_path)
    assert np.allclose(wl, np.array([100.0, 101.0, 102.0, 103.0]))


def test_load_newera_flux_array_reads_mock_file(tmp_path):
    grid_dir = tmp_path / "newera_jwst"
    grid_dir.mkdir()
    fname = grid_dir / "PHOENIX-NewEra-JWST-SPECTRA.Z-0.0.txt"

    header_tokens = ["0"] * 41
    header_tokens[7] = "1.0"
    header_tokens[8] = "4"
    header_tokens[9] = "100.0"
    header_tokens[10] = "103.0"
    header_tokens[12] = "5000"
    header_tokens[13] = "4.5"

    with open(fname, "w") as fh:
        fh.write(" ".join(header_tokens) + "\n")
        fh.write("1.0 2.0 3.0 4.0\n")

    flux = utils.load_newera_flux_array(5000, 4.5, 0.0, grid_name="newera_jwst", library_root=tmp_path)
    assert np.allclose(flux, np.array([1.0, 2.0, 3.0, 4.0]))


def test_download_newera_grid_uses_existing_tarball(tmp_path, monkeypatch):
    utils.set_library_root(tmp_path)
    try:
        grid_dir = tmp_path / "newera_jwst"
        candidate = utils.NEWERA_TARBALL_CANDIDATES["newera_jwst"][0]
        grid_dir.mkdir(parents=True, exist_ok=True)
        (grid_dir / candidate).write_text("tarball")

        def fail_retrieve(*args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("pooch.retrieve should not be invoked")

        monkeypatch.setattr(pooch, "retrieve", fail_retrieve)

        path = utils.download_newera_grid("newera_jwst", extract=False)
    finally:
        utils.set_library_root(None)

    assert path == grid_dir


def test_download_newera_grid_downloads_candidate(tmp_path, monkeypatch):
    utils.set_library_root(tmp_path)
    monkeypatch.setenv("SPECLIB_NEWERA_RECORD_ID", "12345")

    candidate = utils.NEWERA_TARBALL_CANDIDATES["newera_lowres"][0]
    called: dict[str, str] = {}

    def fake_retrieve(url, fname, path, known_hash, processor, progressbar):
        called["url"] = url
        called["fname"] = fname
        target = Path(path) / fname
        target.write_text("tarball")
        return str(target)

    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)

    try:
        utils.download_newera_grid("newera_lowres", extract=False, overwrite=True)
    finally:
        utils.set_library_root(None)
        monkeypatch.delenv("SPECLIB_NEWERA_RECORD_ID", raising=False)

    expected_url = f"https://www.fdr.uni-hamburg.de/record/12345/files/{candidate}?download=1"
    assert called["url"] == expected_url
    assert called["fname"] == candidate
