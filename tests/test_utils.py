from speclib.utils import nearest, trilinear_interpolate


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
