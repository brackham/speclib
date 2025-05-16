from speclib.utils import nearest

def test_nearest_simple():
    arr = [10, 20, 30]
    val = 19
    i = nearest(arr, val)
    assert i == 20
