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
