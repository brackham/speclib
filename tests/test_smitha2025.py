import hashlib
import inspect
import io
import json

import astropy.units as u
import numpy as np
import pytest

from speclib import download_smitha2025_spectra as public_download
from speclib import utils
from speclib.core import Spectrum


STELLAR_SCALES = {"G2V": 1.0, "K0V": 2.0, "M0V": 3.0}
COMPONENT_SCALES = {"quiet": 11.0, "spot": 13.0, "penumbra": 7.0, "umbra": 2.0}


def _spectrum_bytes(stellar_type, *, wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(400.0, 602.0, 2.0)
    scale = STELLAR_SCALES[stellar_type]
    rows = [
        f" Disk integrated flux for {stellar_type} star "
        "(erg s-1 Sr-1 cm-2 nm-1)",
        "=============================================================",
        "Wavelength   quiet     spot      penumbra     umbra",
        "   [nm]      region",
        "============================================================",
    ]
    rows.extend(
        " ".join(
            [
                f"{wavelength:.3f}",
                f"{scale * COMPONENT_SCALES['quiet']:.6e}",
                f"{scale * COMPONENT_SCALES['spot']:.6e}",
                f"{scale * COMPONENT_SCALES['penumbra']:.6e}",
                f"{scale * COMPONENT_SCALES['umbra']:.6e}",
            ]
        )
        for wavelength in wavelengths
    )
    return ("\n".join(rows) + "\n").encode()


def _metadata_for_bytes(stellar_type, content):
    filename = utils.SMITHA2025_FILES[stellar_type]["filename"]
    return {
        "filename": filename,
        "filesize": len(content),
        "md5": hashlib.md5(content).hexdigest(),
    }


@pytest.fixture
def smitha2025_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "smitha2025"
    cache_dir.mkdir()
    metadata = {}
    for stellar_type in STELLAR_SCALES:
        content = _spectrum_bytes(stellar_type)
        file_metadata = _metadata_for_bytes(stellar_type, content)
        metadata[stellar_type] = file_metadata
        (cache_dir / file_metadata["filename"]).write_bytes(content)

    monkeypatch.setattr(utils, "SMITHA2025_FILES", metadata)
    utils.set_library_root(tmp_path)
    try:
        yield tmp_path
    finally:
        utils.set_library_root(None)


def test_download_smitha2025_spectra_public_alias():
    assert public_download is utils.download_smitha2025_spectra


def test_resolve_smitha2025_file_uses_pinned_edmond_metadata(monkeypatch):
    metadata = utils.SMITHA2025_FILES["K0V"]
    payload = {
        "data": {
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "files": [
                    {
                        "label": metadata["filename"],
                        "dataFile": {
                            "id": 12345,
                            "filesize": metadata["filesize"],
                            "checksum": {
                                "type": "MD5",
                                "value": metadata["md5"],
                            },
                        },
                    }
                ],
            }
        }
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda url: Response(json.dumps(payload).encode()),
    )

    assert utils._resolve_smitha2025_file_url("k0v") == (
        "https://edmond.mpg.de/api/access/datafile/12345"
    )


def test_resolve_smitha2025_file_rejects_new_dataset_version(monkeypatch):
    payload = {
        "data": {
            "latestVersion": {
                "versionNumber": 2,
                "versionMinorNumber": 0,
                "files": [],
            }
        }
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda url: Response(json.dumps(payload).encode()),
    )
    with pytest.raises(RuntimeError, match="pins version 1.0"):
        utils._resolve_smitha2025_file_url("G2V")


def test_download_smitha2025_spectra_caches_and_refreshes(monkeypatch, tmp_path):
    content = _spectrum_bytes("K0V")
    metadata = dict(utils.SMITHA2025_FILES)
    metadata["K0V"] = _metadata_for_bytes("K0V", content)
    monkeypatch.setattr(utils, "SMITHA2025_FILES", metadata)
    monkeypatch.setattr(
        utils, "_resolve_smitha2025_file_url", lambda stellar_type: "mock://K0V"
    )
    retrievals = []

    def fake_retrieve(**kwargs):
        retrievals.append(kwargs)
        target = kwargs["path"] / kwargs["fname"]
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(utils.pooch, "retrieve", fake_retrieve)
    cache_dir = utils.download_smitha2025_spectra(
        "k0v", library_root=tmp_path
    )
    expected_path = cache_dir / metadata["K0V"]["filename"]
    assert expected_path.read_bytes() == content
    assert retrievals[0]["known_hash"] == f"md5:{metadata['K0V']['md5']}"

    utils.download_smitha2025_spectra("K0V", library_root=tmp_path)
    assert len(retrievals) == 1

    utils.download_smitha2025_spectra(
        "K0V", overwrite=True, library_root=tmp_path
    )
    assert len(retrievals) == 2

    expected_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid cached file was removed"):
        utils.download_smitha2025_spectra("K0V", library_root=tmp_path)
    assert not expected_path.exists()


def test_download_smitha2025_spectra_removes_partial_file(monkeypatch, tmp_path):
    content = _spectrum_bytes("G2V")
    metadata = dict(utils.SMITHA2025_FILES)
    metadata["G2V"] = _metadata_for_bytes("G2V", content)
    monkeypatch.setattr(utils, "SMITHA2025_FILES", metadata)
    monkeypatch.setattr(
        utils, "_resolve_smitha2025_file_url", lambda stellar_type: "mock://G2V"
    )

    def interrupted_retrieve(**kwargs):
        target = kwargs["path"] / kwargs["fname"]
        target.write_bytes(b"partial")
        raise OSError("connection interrupted")

    monkeypatch.setattr(utils.pooch, "retrieve", interrupted_retrieve)
    expected_path = tmp_path / "smitha2025" / metadata["G2V"]["filename"]
    with pytest.raises(RuntimeError, match="connection interrupted"):
        utils.download_smitha2025_spectra("G2V", library_root=tmp_path)
    assert not expected_path.exists()


@pytest.mark.parametrize("stellar_type", ["G2V", "K0V", "M0V"])
@pytest.mark.parametrize("component", ["quiet", "spot", "penumbra", "umbra"])
def test_from_smitha2025_loads_every_discrete_product(
    smitha2025_cache, stellar_type, component
):
    spectrum = Spectrum.from_smitha2025(stellar_type.lower(), component=component)
    expected_source_flux = STELLAR_SCALES[stellar_type] * COMPONENT_SCALES[component]

    assert isinstance(spectrum, Spectrum)
    assert spectrum.spectral_axis.unit == u.AA
    assert spectrum.flux.unit == u.erg / (u.s * u.cm**2 * u.AA)
    assert np.all(np.diff(spectrum.spectral_axis) > 0 * u.AA)
    np.testing.assert_allclose(spectrum.flux.value, expected_source_flux / 10.0)
    assert spectrum.meta["source_library"] == "Smitha et al. (2025)"
    assert spectrum.meta["stellar_type"] == stellar_type
    assert spectrum.meta["surface_component"] == component
    assert spectrum.meta["data_doi"] == "10.17617/3.HS2EE6"
    assert spectrum.meta["native_wavelength_points"] == 101


def test_smitha2025_photosphere_alias_and_component_metadata(smitha2025_cache):
    photosphere = Spectrum.from_smitha2025(" K0V ", component="photosphere")
    quiet = Spectrum.from_smitha2025("K0V", component="quiet")
    spot = Spectrum.from_smitha2025("K0V", component="spot")

    np.testing.assert_array_equal(photosphere.flux, quiet.flux)
    assert photosphere.meta["surface_component"] == "quiet"
    assert photosphere.meta["component_teff"] == 4965 * u.K
    assert photosphere.meta["logg"] == 4.609
    assert spot.meta["component_teff"] is None
    np.testing.assert_allclose(
        spot.flux.value,
        STELLAR_SCALES["K0V"] * COMPONENT_SCALES["spot"] / 10.0,
    )


@pytest.mark.parametrize("stellar_type", ["G1V", "K1V", "M1V", "5000"])
def test_smitha2025_rejects_unavailable_stellar_types(stellar_type):
    with pytest.raises(ValueError, match="Expected one of.*G2V.*K0V.*M0V"):
        Spectrum.from_smitha2025(stellar_type)


@pytest.mark.parametrize("component", ["facula", "plage", "starspot", ""])
def test_smitha2025_rejects_unavailable_components(component):
    with pytest.raises(ValueError, match="surface component"):
        Spectrum.from_smitha2025("G2V", component=component)


def test_smitha2025_is_not_an_interpolatable_grid():
    assert "smitha2025" not in utils.VALID_MODELS
    assert "interpolate" not in inspect.signature(Spectrum.from_smitha2025).parameters


def test_smitha2025_rejects_nonmonotonic_source_axis(monkeypatch, tmp_path):
    content = _spectrum_bytes("G2V", wavelengths=[400.0, 402.0, 401.0])
    metadata = dict(utils.SMITHA2025_FILES)
    metadata["G2V"] = _metadata_for_bytes("G2V", content)
    monkeypatch.setattr(utils, "SMITHA2025_FILES", metadata)
    cache_dir = tmp_path / "smitha2025"
    cache_dir.mkdir()
    (cache_dir / metadata["G2V"]["filename"]).write_bytes(content)

    with pytest.raises(ValueError, match="strictly increasing"):
        utils.load_smitha2025_spectrum("G2V", library_root=tmp_path)


def test_smitha2025_rejects_missing_source_columns(monkeypatch, tmp_path):
    content = b"header\nheader\nheader\nheader\nheader\n400 1 2 3\n402 1 2 3\n"
    metadata = dict(utils.SMITHA2025_FILES)
    metadata["M0V"] = _metadata_for_bytes("M0V", content)
    monkeypatch.setattr(utils, "SMITHA2025_FILES", metadata)
    cache_dir = tmp_path / "smitha2025"
    cache_dir.mkdir()
    (cache_dir / metadata["M0V"]["filename"]).write_bytes(content)

    with pytest.raises(ValueError, match="must contain wavelength, quiet, spot"):
        utils.load_smitha2025_spectrum("M0V", library_root=tmp_path)


def test_smitha2025_spectrum_operations_preserve_metadata(smitha2025_cache):
    spectrum = Spectrum.from_smitha2025("M0V", component="penumbra")
    resampled = spectrum.resample(np.arange(4100.0, 5910.0, 40.0) * u.AA)
    resolved = spectrum.set_spectral_resolution(100.0 * u.AA)
    broadened = spectrum.set_spectral_resolving_power(50)

    assert isinstance(resampled, Spectrum)
    assert isinstance(resolved, Spectrum)
    assert isinstance(broadened, Spectrum)
    assert resampled.meta == spectrum.meta
    assert resolved.meta == spectrum.meta
    assert broadened.meta == spectrum.meta
    resampled.meta["stellar_type"] = "changed"
    assert spectrum.meta["stellar_type"] == "M0V"
