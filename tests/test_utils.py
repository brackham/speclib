from speclib import download_newera_grid as public_download_newera_grid
from speclib.utils import nearest, trilinear_interpolate
import speclib.utils as utils
import numpy as np
import pytest


@pytest.mark.parametrize("grid, value, expected", [
    ([6800, 6900, 7000, 7200, 7400], 7030, [7000, 7200]),
    ([6800, 6900, 7000, 7200, 7400], 7170, [7000, 7200]),
    ([10, 20, 30, 40], 21, [20, 30]),
    ([10, 20, 30, 40], 29, [20, 30]),
    ([0.0, 0.1, 0.2, 0.5], 0.23, [0.2, 0.5]),
    ([30, 10, 40, 20], 21, [20, 30]),
    ([10, 20, 20, 30], 21, [20, 30]),
    ([10, 20, 20, 30], 20, [20]),
    ([10, 20, 30], 10, [10]),
    ([10, 20, 30], 20, [20]),
    ([10, 20, 30], 30, [30]),
    ([10, 20, 30], 0, [10, 20]),
    ([10, 20, 30], 40, [20, 30]),
    ([10, 10, 20, 30, 30], 0, [10, 20]),
    ([10, 10, 20, 30, 30], 40, [20, 30]),
    ([10], 0, [10]),
    ([10], 10, [10]),
    ([10], 20, [10]),
    ([10, 10], 11, [10]),
    ([10, 20], 15, [10, 20]),
    ([10, 20], 10, [10]),
    ([10, 20], 0, [10, 20]),
    ([10, 20], 30, [10, 20]),
])
def test_find_bounds(grid, value, expected):
    array = np.array(grid)
    bounds = utils.find_bounds(array, value)
    np.testing.assert_array_equal(bounds, expected)
    assert bounds.dtype == array.dtype
    np.testing.assert_array_equal(array, grid)


@pytest.mark.parametrize("grid", [[], [[10, 20], [30, 40]]])
def test_find_bounds_rejects_invalid_grid_shape(grid):
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        utils.find_bounds(grid, 15)


def test_nearest_simple():
    arr = [10, 20, 30]
    val = 19
    i = nearest(arr, val)
    assert i == 20


def test_trilinear_interpolate_basic():
    fluxes = {
        1: {
            3: {5: 9, 6: 10},
            4: {5: 10, 6: 11},
        },
        2: {
            3: {5: 10, 6: 11},
            4: {5: 11, 6: 12},
        },
    }

    result = trilinear_interpolate(fluxes, ([1, 2], [3, 4], [5, 6]), (1.5, 3.5, 5.5))
    assert result == 10.5


def test_library_root_env(monkeypatch, tmp_path):
    custom = tmp_path / "cache"
    monkeypatch.setenv("SPECLIB_LIBRARY_PATH", str(custom))
    utils.set_library_root(None)
    assert utils.get_library_root() == custom


def test_set_library_root(tmp_path):
    custom = tmp_path / "other"
    utils.set_library_root(custom)
    try:
        assert utils.get_library_root() == custom
    finally:
        utils.set_library_root(None)


def test_download_newera_grid_overwrite_cleans_cache(monkeypatch, tmp_path):
    grid_name = "newera_jwst"
    utils.set_library_root(tmp_path)
    cache_dir = tmp_path / grid_name
    cache_dir.mkdir()

    leftover_file = cache_dir / "old.txt"
    leftover_file.write_text("stale")
    leftover_dir = cache_dir / "old_dir"
    leftover_dir.mkdir()
    (leftover_dir / "nested.txt").write_text("data")

    tarball_name = utils.NEWERA_TARBALLS[grid_name]
    tar_path = cache_dir / tarball_name
    tar_path.write_text("tar")

    called = {}

    def fake_resolve(name, target_cache, record_id, overwrite):
        called["overwrite"] = overwrite
        assert name == grid_name
        assert target_cache == cache_dir
        assert overwrite is True
        assert not leftover_file.exists()
        assert not leftover_dir.exists()
        tar_path.write_text("fresh")
        return tar_path

    extracted = {}

    def fake_extract(resolved_tar, destination):
        extracted["args"] = (resolved_tar, destination)

    monkeypatch.setattr(utils, "_resolve_newera_tarball", fake_resolve)
    monkeypatch.setattr(utils, "extract_missing_txt_files", fake_extract)
    monkeypatch.setattr(utils, "get_newera_record_id", lambda: "record")

    try:
        result = utils.download_newera_grid(grid_name, extract=True, overwrite=True)
    finally:
        utils.set_library_root(None)

    assert result == cache_dir
    assert called["overwrite"] is True
    assert extracted["args"] == (tar_path, cache_dir)
    assert tar_path.exists()
    assert not leftover_file.exists()
    assert not leftover_dir.exists()


def test_download_newera_grid_preserves_cache_when_not_overwriting(monkeypatch, tmp_path):
    grid_name = "newera_gaia"
    utils.set_library_root(tmp_path)
    cache_dir = tmp_path / grid_name
    cache_dir.mkdir()

    leftover_file = cache_dir / "keep.txt"
    leftover_file.write_text("present")

    tarball_name = utils.NEWERA_TARBALLS[grid_name]
    tar_path = cache_dir / tarball_name
    tar_path.write_text("cached")

    def fake_resolve(name, target_cache, record_id, overwrite):
        assert overwrite is False
        assert leftover_file.exists()
        return tar_path

    extracted = {}

    def fake_extract(resolved_tar, destination):
        extracted["args"] = (resolved_tar, destination)

    monkeypatch.setattr(utils, "_resolve_newera_tarball", fake_resolve)
    monkeypatch.setattr(utils, "extract_missing_txt_files", fake_extract)
    monkeypatch.setattr(utils, "get_newera_record_id", lambda: "record")

    try:
        result = utils.download_newera_grid(grid_name, extract=True, overwrite=False)
    finally:
        utils.set_library_root(None)

    assert result == cache_dir
    assert leftover_file.exists()
    assert extracted["args"] == (tar_path, cache_dir)


def test_download_newera_grid_skips_extraction_by_default(monkeypatch, tmp_path):
    grid_name = "newera_lowres"
    cache_dir = tmp_path / grid_name
    tar_path = cache_dir / utils.NEWERA_TARBALLS[grid_name]
    extracted = []

    def fake_resolve(name, target_cache, record_id, overwrite):
        assert name == grid_name
        assert target_cache == cache_dir
        assert overwrite is False
        return tar_path

    monkeypatch.setattr(utils, "_resolve_newera_tarball", fake_resolve)
    monkeypatch.setattr(
        utils,
        "extract_missing_txt_files",
        lambda *args: extracted.append(("missing", args)),
    )
    monkeypatch.setattr(
        utils,
        "extract_all_members",
        lambda *args: extracted.append(("all", args)),
    )
    monkeypatch.setattr(utils, "get_newera_record_id", lambda: "record")

    result = utils.download_newera_grid(grid_name, library_root=tmp_path)

    assert result == cache_dir
    assert extracted == []


def test_download_newera_grid_public_alias():
    assert public_download_newera_grid is utils.download_newera_grid
