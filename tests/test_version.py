import tomllib
import speclib


def test_version_matches_pyproject():
    """Ensure speclib.__version__ matches the version in pyproject.toml."""
    with open("pyproject.toml", "rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert speclib.__version__ == expected
