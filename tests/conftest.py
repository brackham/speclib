import sys
import os
from pathlib import Path
import shutil
import pytest

# Add the `src/` directory to sys.path so `speclib` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from speclib import utils


@pytest.fixture(scope="session", autouse=True)
def local_sphinx_cache(tmp_path_factory):
    """Provide a local SPHINX cache so tests avoid network access."""
    tmp_home = tmp_path_factory.mktemp("speclib_home")
    cache_dir = tmp_home / ".speclib" / "libraries" / "sphinx"
    cache_dir.mkdir(parents=True)

    data_dir = Path(__file__).parent / "data" / "sphinx"
    for fname in data_dir.iterdir():
        v4_name = fname.name.replace("_spectra.txt", ".txt")
        shutil.copy(fname, cache_dir / v4_name)

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_home)
    try:
        yield
    finally:
        if old_home is not None:
            os.environ["HOME"] = old_home
        else:
            os.environ.pop("HOME", None)
