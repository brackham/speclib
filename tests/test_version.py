import importlib.metadata
import speclib


def test_version_matches_pyproject():
    """
    Ensure that speclib.__version__ matches the version in pyproject.toml.
    """
    expected = importlib.metadata.version("speclib")
    assert speclib.__version__ == expected
