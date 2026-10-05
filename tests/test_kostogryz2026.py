import hashlib
import inspect
import io
import json
import shutil

import astropy.units as u
import h5py
import numpy as np
import pytest

from speclib import (
    BinnedSpecificIntensitySpectrum,
    SpecificIntensityGrid,
    SpecificIntensitySpectrum,
    download_kostogryz2026_spectra,
)
from speclib import utils


WAVELENGTH_NM = np.geomspace(200.50003, 9980.0014, 978)
MAGNETIZATIONS = ("hydro", "ssd", "B100G", "B200G", "B300G")
RELEASE_FILE_METADATA = {
    "F3_MH_00.h5": (665_449, "8fef57d9429c058d0b97a38a1e9e509e"),
    "G2_MH_00.h5": (666_727, "5f971d0a1ea5af65681974dc08a7d67e"),
    "G2_MH_m10.h5": (662_482, "b04cce190537684b36c1eb91c2ef4f26"),
    "G2_MH_p05.h5": (667_157, "6b9de30858a04332d6ac4a26b8fde979"),
    "K0_MH_00.h5": (667_539, "fcbf106f742f02652e158f30a78d2040"),
    "K4_MH_00.h5": (668_683, "91c82908c8c9bccb7496cfd57d61e4cb"),
    "M0_MH_00.h5": (667_854, "2c3d9f7913239493887f2bbdbe17ef4c"),
    "M2_MH_00.h5": (670_792, "3b95bc82d158cf8fdb2eb2bce53e4dbb"),
    "M4_MH_00.h5": (666_337, "e7d437fc7714de38c4d2d10a03ac42ea"),
    "README.md": (6_804, "f45e035f4c7e7fc71ad6f859740bd0e1"),
    "read_spectra.py": (5_291, "0b14b07351bf0bc8b9e5d97c1e542823"),
    "read_spectral_library.ipynb": (
        46_843,
        "87497487e9a7ac98ef4d4872a764627f",
    ),
}
TEFF_SPECTRUM = {
    "F3_MH_00": [
        6806.703653250776,
        6794.403700796001,
        6837.592711943168,
        6878.765045915556,
        6914.47184257339,
    ],
    "G2_MH_00": [
        5777.879635192657,
        5791.644792007823,
        5806.256555197113,
        5818.594324070224,
        5816.6874941134465,
    ],
    "G2_MH_m10": [
        5749.853803955243,
        5754.737144629751,
        5764.887927223147,
        5771.55391952327,
        5773.636906287507,
    ],
    "G2_MH_p05": [
        5780.489700133928,
        5785.445556949009,
        5817.022187869427,
        5824.368423391435,
        5848.543960631669,
    ],
    "K0_MH_00": [
        5239.6918010684,
        5252.342524704377,
        5266.354279809075,
        5276.0308798976785,
        5285.838388190418,
    ],
    "K4_MH_00": [
        4542.989275705904,
        4549.82333193872,
        4561.439461966907,
        4566.738386464709,
        4576.5922710911955,
    ],
    "M0_MH_00": [
        3707.321087232994,
        3708.9503871478337,
        3709.107794572848,
        3709.680652882536,
        3707.2847372021024,
    ],
    "M2_MH_00": [
        3518.344382128434,
        3513.477563229028,
        3511.25604708112,
        3507.0006610760506,
        3502.4476696773213,
    ],
    "M4_MH_00": [
        3196.349737991997,
        3196.421560378827,
        3194.896332936387,
        3191.031935930622,
        3187.8460388328867,
    ],
}


def _write_hdf5(cache_dir, group_name):
    metadata = utils.KOSTOGRYZ2026_MODELS[group_name]
    path = cache_dir / metadata["filename"]
    mu = utils.KOSTOGRYZ2026_NATIVE_MU
    wavelength_scale = (WAVELENGTH_NM / WAVELENGTH_NM[0])[None, :, None]
    magnetic_scale = np.arange(1.0, 6.0)[:, None, None]
    limb_scale = (0.5 + 0.5 * mu)[None, None, :]
    spectra = 1e10 * magnetic_scale * wavelength_scale * limb_scale
    if group_name == "G2_MH_00":
        spectra[1, 0, 0] = 370178759221.13745
        spectra[1, 500, 4] = 60404466671732.08
    limb_darkening = spectra / spectra[:, :, [-1]]

    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group(group_name)
        group.attrs["star_name"] = group_name
        group.attrs["MH"] = metadata["metallicity"]
        group.attrs["MH_units"] = "dex"
        group.attrs["logg"] = (
            4.609 if group_name == "K0_MH_00" else metadata["logg"]
        )
        group.attrs["logg_units"] = "log10(cm s^-2)"
        dataset = group.create_dataset("mu", data=mu)
        dataset.attrs["units"] = "cos(theta)"
        dataset = group.create_dataset("wavelengths", data=WAVELENGTH_NM)
        dataset.attrs["units"] = "nm, in vacuum"
        group.create_dataset(
            "magnetizations",
            data=np.asarray(MAGNETIZATIONS, dtype=h5py.string_dtype()),
        )
        dataset = group.create_dataset(
            "teff", data=np.asarray(TEFF_SPECTRUM[group_name])
        )
        dataset.attrs["units"] = "K"
        dataset = group.create_dataset("spectra", data=spectra)
        dataset.attrs["units"] = utils.KOSTOGRYZ2026_SOURCE_INTENSITY_UNIT
        dataset = group.create_dataset("limb_darkening", data=limb_darkening)
        dataset.attrs["units"] = "normalized to disc center"
        dataset = group.create_dataset(
            "integrated_flux", data=np.ones((5, WAVELENGTH_NM.size))
        )
        dataset.attrs["units"] = r"erg $s^{-1}$ $cm^{-2}$ $\AA^{-1}$"
    return path


@pytest.fixture
def kostogryz_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "kostogryz2026"
    cache_dir.mkdir()
    for group_name in utils.KOSTOGRYZ2026_MODELS:
        _write_hdf5(cache_dir, group_name)

    def use_cached_file(model=None, metallicity=0.0, overwrite=False, library_root=None):
        del model, metallicity, overwrite, library_root
        return cache_dir

    monkeypatch.setattr(
        utils, "download_kostogryz2026_spectra", use_cached_file
    )
    return cache_dir


def test_public_exports():
    assert download_kostogryz2026_spectra is utils.download_kostogryz2026_spectra
    assert SpecificIntensitySpectrum.__name__ == "SpecificIntensitySpectrum"
    assert SpecificIntensityGrid.__name__ == "SpecificIntensityGrid"


def test_release_inventory_and_discrete_choices():
    assert len(utils.KOSTOGRYZ2026_RELEASE_FILES) == 12
    assert len(utils.KOSTOGRYZ2026_MODELS) == 9
    assert sum(
        int(item["filesize"])
        for name, item in utils.KOSTOGRYZ2026_RELEASE_FILES.items()
        if name.endswith(".h5")
    ) == 6_003_020
    assert {
        name: (metadata["filesize"], metadata["md5"])
        for name, metadata in utils.KOSTOGRYZ2026_RELEASE_FILES.items()
    } == RELEASE_FILE_METADATA
    assert SpecificIntensityGrid.available_models() == (
        ("F3", 0.0),
        ("G2", 0.0),
        ("G2", -1.0),
        ("G2", 0.5),
        ("K0", 0.0),
        ("K4", 0.0),
        ("M0", 0.0),
        ("M2", 0.0),
        ("M4", 0.0),
    )
    assert SpecificIntensityGrid.available_magnetic_states() == MAGNETIZATIONS


def test_resolve_every_release_file_through_pinned_edmond_record(monkeypatch):
    files = []
    expected_urls = {}
    for datafile_id, (name, metadata) in enumerate(
        utils.KOSTOGRYZ2026_RELEASE_FILES.items(), start=1000
    ):
        files.append(
            {
                "label": name,
                "dataFile": {
                    "id": datafile_id,
                    "filesize": metadata["filesize"],
                    "md5": metadata["md5"],
                },
            }
        )
        expected_urls[name] = (
            f"https://edmond.mpg.de/api/access/datafile/{datafile_id}"
        )
    payload = {
        "data": {
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "files": files,
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
    for name in utils.KOSTOGRYZ2026_RELEASE_FILES:
        assert utils._resolve_kostogryz2026_file_url(name) == expected_urls[name]


def test_resolver_rejects_changed_dataset_version(monkeypatch):
    metadata = utils.KOSTOGRYZ2026_RELEASE_FILES["F3_MH_00.h5"]
    payload = {
        "data": {
            "latestVersion": {
                "versionNumber": 2,
                "versionMinorNumber": 0,
                "files": [
                    {
                        "label": metadata["filename"],
                        "dataFile": {
                            "id": 1,
                            "filesize": metadata["filesize"],
                            "md5": metadata["md5"],
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
    with pytest.raises(RuntimeError, match="version 2.0.*pins version 1.0"):
        utils._resolve_kostogryz2026_file_url("F3_MH_00.h5")


def test_download_selected_model_cache_reuse_overwrite_and_corruption(
    monkeypatch, tmp_path
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = _write_hdf5(source_dir, "G2_MH_p05")
    content = source_path.read_bytes()
    release_files = dict(utils.KOSTOGRYZ2026_RELEASE_FILES)
    release_files[source_path.name] = {
        "filename": source_path.name,
        "filesize": len(content),
        "md5": hashlib.md5(content).hexdigest(),
    }
    monkeypatch.setattr(utils, "KOSTOGRYZ2026_RELEASE_FILES", release_files)
    monkeypatch.setattr(
        utils,
        "_resolve_kostogryz2026_file_url",
        lambda filename: "mock://G2_MH_p05",
    )
    retrievals = []

    def fake_retrieve(**kwargs):
        retrievals.append(kwargs)
        target = kwargs["path"] / kwargs["fname"]
        shutil.copyfile(source_path, target)
        return str(target)

    monkeypatch.setattr(utils.pooch, "retrieve", fake_retrieve)
    cache_root = tmp_path / "cache"
    result = utils.download_kostogryz2026_spectra(
        "G2", 0.5, library_root=cache_root
    )
    cached_path = result / "G2_MH_p05.h5"
    assert result == cache_root / "kostogryz2026"
    assert retrievals[0]["known_hash"] == f"md5:{release_files[source_path.name]['md5']}"

    utils.download_kostogryz2026_spectra("G2V", 0.5, library_root=cache_root)
    assert len(retrievals) == 1
    utils.download_kostogryz2026_spectra(
        "G2", 0.5, overwrite=True, library_root=cache_root
    )
    assert len(retrievals) == 2

    cached_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid cached file was removed"):
        utils.download_kostogryz2026_spectra(
            "G2", 0.5, library_root=cache_root
        )
    assert not cached_path.exists()


def test_download_removes_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        utils,
        "_resolve_kostogryz2026_file_url",
        lambda filename: "mock://F3",
    )

    def interrupted_retrieve(**kwargs):
        target = kwargs["path"] / kwargs["fname"]
        target.write_bytes(b"partial")
        raise OSError("connection interrupted")

    monkeypatch.setattr(utils.pooch, "retrieve", interrupted_retrieve)
    expected = tmp_path / "kostogryz2026" / "F3_MH_00.h5"
    with pytest.raises(RuntimeError, match="connection interrupted"):
        utils.download_kostogryz2026_spectra("F3", library_root=tmp_path)
    assert not expected.exists()


def test_every_model_state_and_mu_loads_without_interpolation(kostogryz_cache):
    for model, metallicity in SpecificIntensityGrid.available_models():
        for state_index, state in enumerate(MAGNETIZATIONS):
            grid = SpecificIntensityGrid.from_library(
                "kostogryz2026",
                model=model,
                metallicity=metallicity,
                magnetic_state=state,
            )
            np.testing.assert_allclose(grid.mu, np.arange(0.1, 1.01, 0.1))
            assert len(grid) == 10
            assert grid.meta["teff_spectrum"].to_value(u.K) == pytest.approx(
                TEFF_SPECTRUM[grid.meta["source_model_identifier"]][state_index]
            )
            for mu in grid.mu:
                spectrum = grid.at_mu(mu)
                assert isinstance(spectrum, SpecificIntensitySpectrum)
                assert spectrum.meta["native_mu"] == mu


def test_grid_data_units_values_metadata_indexing_and_iteration(kostogryz_cache):
    grid = SpecificIntensityGrid.from_library(
        "kostogryz2026",
        model="g2v",
        metallicity=0.0,
        magnetic_state="SSD",
    )
    expected_unit = u.erg / (u.s * u.cm**2 * u.sr * u.AA)
    assert grid.spectral_axis.unit == u.AA
    assert grid.intensity.unit == expected_unit
    assert grid.intensity.shape == (10, 978)
    assert np.all(np.diff(grid.wavelength) > 0 * u.AA)
    assert grid.wavelength[0].to_value(u.AA) == pytest.approx(2005.0003)
    assert grid.wavelength[-1].to_value(u.AA) == pytest.approx(99800.014)
    assert grid[0] is grid.at_mu(0.1)
    assert grid[-1] is grid.at_mu(1.0 * u.dimensionless_unscaled)
    assert grid[2:4] == grid.spectra[2:4]
    assert tuple(iter(grid)) == grid.spectra
    assert grid.at_mu(0.1).intensity[0].value == pytest.approx(
        370178759221.13745
    )
    assert grid.at_mu(0.5).intensity[500].value == pytest.approx(
        60404466671732.08
    )
    assert grid.meta["source_library"] == "kostogryz2026"
    assert grid.meta["metallicity"] == 0.0
    assert grid.meta["metallicity_type"] == "mh"
    assert grid.at_mu(0.1).meta["metallicity_type"] == "mh"
    assert grid.meta["source_model_identifier"] == "G2_MH_00"
    assert grid.meta["magnetic_state"] == "ssd"
    assert grid.meta["magnetic_state_category"] == "small_scale_dynamo"
    assert grid.meta["source_magnetization_label"] == "ssd"
    assert grid.meta["imposed_vertical_field"] is None
    assert grid.meta["teff_spectrum"].to_value(u.K) == pytest.approx(
        5791.644792007823
    )
    assert "teff_muram" not in grid.meta
    assert grid.meta["data_doi"] == "10.17617/3.FBTIYY"
    assert grid.meta["dataset_version"] == "1.0"
    assert grid.meta["native_wavelength_points"] == 978
    assert grid.meta["wavelength_convention"] == "vacuum"
    assert grid.meta["source_filesize"] == 666_727
    assert grid.meta["native_resolving_power"].startswith("approximately 400")


def test_specific_intensity_spectrum_direct_library_constructor(kostogryz_cache):
    spectrum = SpecificIntensitySpectrum.from_library(
        "kostogryz2026",
        model="M0",
        magnetic_state="B200G",
        mu=0.7,
    )
    assert isinstance(spectrum, SpecificIntensitySpectrum)
    assert spectrum.meta["magnetic_state"] == "B200G"
    assert spectrum.meta["magnetic_state_category"] == "imposed_vertical_field"
    assert spectrum.meta["imposed_vertical_field"] == 200 * u.G
    assert spectrum.meta["source_magnetization_label"] == "B200G"


@pytest.mark.parametrize(
    ("model", "metallicity", "message"),
    [
        ("F5", 0.0, "Available models"),
        ("K0", 0.5, "Available metallicities.*No interpolation"),
        ("G2", 0.25, "Available metallicities.*No interpolation"),
    ],
)
def test_invalid_model_and_metallicity_requests(model, metallicity, message):
    with pytest.raises(ValueError, match=message):
        utils._normalize_kostogryz2026_selection(model, metallicity)


@pytest.mark.parametrize("state", ["B150G", "facula", "quiet", ""])
def test_invalid_magnetic_state_lists_native_choices(state, kostogryz_cache):
    with pytest.raises(ValueError, match="Available native states"):
        SpecificIntensityGrid.from_library(
            "kostogryz2026", model="G2", magnetic_state=state
        )


def test_invalid_mu_fails_instead_of_interpolating(kostogryz_cache):
    grid = SpecificIntensityGrid.from_library(
        "kostogryz2026", model="G2", magnetic_state="ssd"
    )
    with pytest.raises(ValueError, match="mu=0.55.*Available values.*No interpolation"):
        grid.at_mu(0.55)
    with pytest.raises(ValueError, match="mu=0.0"):
        grid.at_mu(0.0)
    with pytest.raises(TypeError, match="indices must be integers or slices"):
        grid[0.5]


def test_metadata_inconsistencies_are_resolved_explicitly(kostogryz_cache):
    metal_rich = SpecificIntensityGrid.from_library(
        "kostogryz2026", model="G2", metallicity=0.5
    )
    assert metal_rich.meta["metallicity"] == 0.5
    assert metal_rich.meta["metallicity_type"] == "mh"
    assert "Edmond file description incorrectly says 0.0" in " ".join(
        metal_rich.meta["metadata_notes"]
    )

    k0 = SpecificIntensityGrid.from_library("kostogryz2026", model="K0")
    assert k0.meta["logg"] == 4.4
    assert k0.meta["source_hdf5_logg"] == 4.609
    assert "HDF5 attribute incorrectly" in " ".join(k0.meta["metadata_notes"])

    k4 = SpecificIntensityGrid.from_library("kostogryz2026", model="K4")
    assert k4.meta["teff_spectrum"].to_value(u.K) == pytest.approx(
        4549.82333193872
    )
    assert k4.meta["teff_spectrum"] != 4293 * u.K
    assert "Teff=4293 K" in " ".join(k4.meta["metadata_notes"])


def test_limb_darkening_identity_is_validated(monkeypatch, kostogryz_cache):
    path = kostogryz_cache / "F3_MH_00.h5"
    with h5py.File(path, "r+") as h5_file:
        h5_file["F3_MH_00/limb_darkening"][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="normalized at mu=1"):
        utils.load_kostogryz2026_intensities("F3")


def test_intensity_spectrum_operations_preserve_semantics(kostogryz_cache):
    spectrum = SpecificIntensitySpectrum.from_library(
        "kostogryz2026", model="G2", magnetic_state="ssd", mu=0.5
    )
    original_meta = dict(spectrum.meta)
    selected = spectrum.select_wavelength(3000 * u.AA, 5000 * u.AA)
    resampled = spectrum.resample(np.linspace(3000, 5000, 31) * u.AA)
    regularized = selected.regularize(25 * u.AA)
    constant_width = spectrum.set_spectral_resolution(1000 * u.AA)
    constant_power = spectrum.set_spectral_resolving_power(50)
    variable_power = spectrum.set_variable_resolving_power(
        u.Quantity([spectrum.wavelength[0], spectrum.wavelength[-1]]), [40, 60]
    )
    binned = spectrum.bin(
        np.array([4000.0, 6000.0]) * u.AA,
        np.array([500.0, 500.0]) * u.AA,
    )

    for result in (
        selected,
        resampled,
        regularized,
        constant_width,
        constant_power,
        variable_power,
    ):
        assert isinstance(result, SpecificIntensitySpectrum)
        assert result.intensity.unit.is_equivalent(spectrum.intensity.unit)
        assert result.meta["source_model_identifier"] == "G2_MH_00"
        assert result.meta["native_mu"] == 0.5
    assert original_meta == spectrum.meta
    assert isinstance(binned, BinnedSpecificIntensitySpectrum)
    assert binned.intensity.unit == spectrum.intensity.unit
    assert binned.meta == spectrum.meta


def test_intensity_spectrum_slicing_preserves_semantics(kostogryz_cache):
    spectrum = SpecificIntensitySpectrum.from_library(
        "kostogryz2026", model="G2", magnetic_state="ssd", mu=0.5
    )
    boolean_mask = (spectrum.wavelength >= 3000 * u.AA) & (
        spectrum.wavelength <= 5000 * u.AA
    )
    for selected in (spectrum[10:20], spectrum[10], spectrum[boolean_mask]):
        assert isinstance(selected, SpecificIntensitySpectrum)
        assert selected.intensity.unit == spectrum.intensity.unit
        assert selected.meta["native_mu"] == 0.5


def test_constructor_rejects_flux_density_without_steradian():
    with pytest.raises(u.UnitsError, match=r"including sr\^-1"):
        SpecificIntensitySpectrum(
            spectral_axis=np.arange(3) * u.AA + 1 * u.AA,
            intensity=np.ones(3) * u.erg / (u.s * u.cm**2 * u.AA),
        )


def test_intensity_class_rejects_flux_only_library_constructors():
    with pytest.raises(TypeError, match="flux-oriented"):
        SpecificIntensitySpectrum.from_grid(5800, 4.5)
    with pytest.raises(TypeError, match="disk-integrated flux"):
        SpecificIntensitySpectrum.from_smitha2025("G2V")


def test_kostogryz_is_not_an_interpolatable_spectral_grid():
    assert "kostogryz2026" not in utils.VALID_MODELS
    assert "interpolate" not in inspect.signature(
        SpecificIntensityGrid.from_library
    ).parameters
