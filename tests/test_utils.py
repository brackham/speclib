from speclib import download_newera_grid as public_download_newera_grid
from speclib.utils import nearest, trilinear_interpolate
import speclib.utils as utils


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
        result = utils.download_newera_grid(grid_name, overwrite=True)
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
        result = utils.download_newera_grid(grid_name, overwrite=False)
    finally:
        utils.set_library_root(None)

    assert result == cache_dir
    assert leftover_file.exists()
    assert extracted["args"] == (tar_path, cache_dir)


def test_download_newera_grid_public_alias():
    assert public_download_newera_grid is utils.download_newera_grid
