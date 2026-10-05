"""Offline contracts derived from the released V3 inventory/header format."""

import io
import json
import os
import tarfile
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest

from speclib import Spectrum, SpectralGrid, BinnedSpectralGrid, utils


REDUCED = ("newera_gaia", "newera_jwst", "newera_lowres")


@pytest.fixture(params=REDUCED)
def reduced_cube(request, monkeypatch, tmp_path):
    """An initially uncached archive with affine spectra and real text headers."""
    selector = request.param
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    members = {}
    prefix = utils.NEWERA_TARBALLS[selector].removesuffix(".tar.gz")
    for metallicity in (-0.5, 0.0):
        label = "Z-0.0" if metallicity == 0 else f"Z{metallicity:+.1f}"
        text = ""
        for teff in (3500, 3600):
            for logg in (4.0, 4.5):
                flux = teff / 100 + 2 * logg + 8 * metallicity + np.arange(3)
                text += (
                    "star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 "
                    f"{teff} {logg} 1 0\n"
                    + " ".join(map(str, flux)) + "\n"
                )
        members[f"{prefix}.{label}.txt"] = text.encode()
    downloads, extractions = [], []

    def retrieve(*, url, fname, path, **kwargs):
        assert fname == utils.NEWERA_TARBALLS[selector]
        downloads.append(url)
        target = Path(path) / fname
        with tarfile.open(target, "w:gz") as archive:
            for name, content in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return str(target)

    original_extract = utils.extract_member_from_tar

    def extract(tar_path, member_name, extract_dir):
        extractions.append(member_name)
        return original_extract(tar_path, member_name, extract_dir)

    monkeypatch.setattr(utils.pooch, "retrieve", retrieve)
    monkeypatch.setattr(utils, "extract_member_from_tar", extract)
    return selector, members, downloads, extractions


@pytest.mark.parametrize("coordinates", [
    (3550, 4.0, 0.0),  # Teff only
    (3500, 4.25, 0.0),  # logg only
    (3500, 4.0, -0.25),  # metallicity only: no Z-0.2 member
    (3550, 4.25, -0.25),  # all three dimensions
])
def test_reduced_uncached_interpolation_uses_native_models(
    reduced_cube, monkeypatch, coordinates
):
    selector, members, downloads, extractions = reduced_cube
    loads = []
    for helper in ("load_newera_wavelength_array", "load_newera_flux_array"):
        original = getattr(utils, helper)

        def load(teff, logg, metallicity, *args, _original=original, **kwargs):
            assert teff in (3500, 3600)
            assert logg in (4.0, 4.5)
            assert metallicity in (-0.5, 0.0)
            loads.append((teff, logg, metallicity))
            return _original(teff, logg, metallicity, *args, **kwargs)

        monkeypatch.setattr(utils, helper, load)

    spectrum = Spectrum.from_grid(*coordinates, model_grid=selector)
    teff, logg, metallicity = coordinates
    expected = teff / 100 + 2 * logg + 8 * metallicity + np.arange(3)
    np.testing.assert_allclose(
        spectrum.flux.to_value(u.W / (u.m**2 * u.nm)), expected
    )
    assert spectrum.meta["metallicity"] == metallicity
    assert spectrum.meta["metallicity_type"] == "mh"
    assert len(downloads) == 1
    assert extractions and set(extractions) <= members.keys()
    expected_corners = {
        (tt, gg, ff)
        for tt in utils.find_bounds([3500, 3600], teff)
        for gg in utils.find_bounds([4.0, 4.5], logg)
        for ff in utils.find_bounds([-0.5, 0.0], metallicity)
    }
    assert set(loads) == expected_corners

    # A repeated request must use the extracted cache and return the same flux.
    extractions.clear()
    cached = Spectrum.from_grid(teff, logg, mh=metallicity, model_grid=selector)
    np.testing.assert_array_equal(cached.flux, spectrum.flux)
    assert len(downloads) == 1 and not extractions


@pytest.mark.parametrize("interpolate", (False, True))
@pytest.mark.parametrize("metallicity", (0.0, -0.04, -0.46))
def test_reduced_exact_and_rounded_lookup_unchanged(
    reduced_cube, interpolate, metallicity
):
    selector, members, downloads, extractions = reduced_cube
    spectrum = Spectrum.from_grid(
        3500, 4.0, metallicity, model_grid=selector, interpolate=interpolate
    )
    native_metallicity = float(f"{metallicity:.1f}")
    np.testing.assert_allclose(
        spectrum.flux.to_value(u.W / (u.m**2 * u.nm)),
        43 + 8 * native_metallicity + np.arange(3),
    )
    assert spectrum.meta["metallicity"] == native_metallicity
    assert len(downloads) == len(extractions) == 1
    assert extractions[0] in members


def test_reduced_nearest_lookup_unchanged(reduced_cube):
    selector, _, _, _ = reduced_cube
    spectrum = Spectrum.from_grid(
        3540, 4.2, 0.0, model_grid=selector, interpolate=False
    )
    np.testing.assert_allclose(
        spectrum.flux.to_value(u.W / (u.m**2 * u.nm)), 43 + np.arange(3)
    )
    assert spectrum.meta["teff"] == 3500
    assert spectrum.meta["logg"] == 4.0
    # Keep the historical failure for an unavailable rounded plane when the
    # caller disables interpolation; this fix changes only interpolation.
    with pytest.raises(FileNotFoundError, match="Z-0.2.txt"):
        Spectrum.from_grid(3550, 4.25, -0.25, model_grid=selector, interpolate=False)


@pytest.mark.parametrize("coordinates", [
    (3350, 4.75, 0.0),  # Sparse special models cannot replace missing corners.
    (3300, 4.5, 0.0),  # Axis membership does not establish native existence.
])
@pytest.mark.parametrize("selector", REDUCED)
def test_reduced_sparse_cell_fails_before_spectrum_loading(
    reduced_headers, monkeypatch, selector, coordinates
):
    def unexpected(*args, **kwargs):
        pytest.fail("A missing native corner must be detected before spectrum loading")

    monkeypatch.setattr(utils, "load_newera_wavelength_array", unexpected)
    monkeypatch.setattr(utils, "load_newera_flux_array", unexpected)
    monkeypatch.setattr(utils, "_ensure_newera_txt_file", unexpected)
    with pytest.raises(ValueError, match="missing native corner Teff=3300, logg=4.5"):
        Spectrum.from_grid(*coordinates, model_grid=selector)


def test_reduced_native_points_observe_same_size_replacement(reduced_headers):
    required = {(3300, 5.0), (3300, 4.5)}
    points = utils._find_newera_reduced_native_points(
        0.0, 0.0, "newera_jwst", required
    )
    assert points == {(3300, 5.0)}
    path = utils._newera_reduced_path(0.0, 0.0, "newera_jwst")
    before = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_text(path.read_text().replace("3300 5.0 1 0", "3300 4.5 1 0"))
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    updated = utils._find_newera_reduced_native_points(
        0.0, 0.0, "newera_jwst", required
    )
    assert updated == {(3300, 4.5)}
    spectrum = Spectrum.from_grid(3300, 4.5, 0.0, model_grid="newera_jwst")
    np.testing.assert_allclose(
        spectrum.flux.to_value(u.W / (u.m**2 * u.nm)), [3300, 3305, 3310]
    )


@pytest.mark.parametrize("metallicity", (0.6, -4.1))
def test_reduced_out_of_range_metallicity_fails(reduced_cube, metallicity):
    selector, _, downloads, extractions = reduced_cube
    with pytest.raises(FileNotFoundError, match="outside the interpolation range"):
        Spectrum.from_grid(3500, 4.0, metallicity, model_grid=selector)
    assert not downloads and not extractions


@pytest.mark.parametrize("requested, native", [(0.54, 0.5), (-4.04, -4.0)])
def test_reduced_rounded_endpoint_requires_exact_native_model(
    reduced_cube, requested, native
):
    selector, members, _, _ = reduced_cube
    prefix = utils.NEWERA_TARBALLS[selector].removesuffix(".tar.gz")
    flux = 43 + 8 * native + np.arange(3)
    members[f"{prefix}.Z{native:+.1f}.txt"] = (
        "star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 3500 4.0 1 0\n"
        + " ".join(map(str, flux)) + "\n"
    ).encode()
    # Preserve main's exact one-decimal endpoint lookup, recording its real plane.
    exact = Spectrum.from_grid(3500, 4.0, requested, model_grid=selector)
    assert exact.meta["metallicity"] == native
    np.testing.assert_allclose(exact.flux.to_value(u.W / (u.m**2 * u.nm)), flux)
    # Independent-axis nearest retrieval still selects the endpoint explicitly.
    nearest = Spectrum.from_grid(
        3550, 4.0, requested, model_grid=selector, interpolate=False
    )
    assert nearest.meta["metallicity"] == native
    np.testing.assert_array_equal(nearest.flux, exact.flux)
    # An off-grid Teff cannot turn an out-of-range metallicity into interpolation.
    with pytest.raises(FileNotFoundError, match="outside the interpolation range"):
        Spectrum.from_grid(3550, 4.0, requested, model_grid=selector)


@pytest.mark.parametrize("interpolate", (False, True))
def test_reduced_failed_alpha_lookup_warns(reduced_cube, interpolate):
    selector, _, _, extractions = reduced_cube
    with pytest.warns(UserWarning, match="Alpha-enhanced models.*not yet supported") as records:
        with pytest.raises(FileNotFoundError, match="alpha=0.2"):
            Spectrum.from_grid(
                3500, 4.0, 0.0, alpha=0.2, model_grid=selector,
                interpolate=interpolate,
            )
    assert len(records) == 1
    assert extractions[0].endswith(".Z-0.0.alpha=0.2.txt")


def test_reduced_failed_metadata_io_warns(monkeypatch):
    def unreadable(*args, **kwargs):
        raise PermissionError("Native plane is unreadable")

    monkeypatch.setattr(utils, "_find_newera_reduced_native_points", unreadable)
    with pytest.warns(UserWarning, match="Alpha-enhanced models.*not yet supported"):
        with pytest.raises(PermissionError, match="Native plane is unreadable"):
            Spectrum.from_grid(3500, 4.0, 0.0, alpha=0.2, model_grid="newera_jwst")


class _CountingTextFile:
    """Count lines consumed by production readers without pre-reading the file."""

    def __init__(self, file):
        self.file = file
        self.lines = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.file.__exit__(*args)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def readline(self):
        line = self.file.readline()
        self.lines += bool(line)
        return line


@pytest.mark.parametrize("coordinates, expected_reads", [
    ((3500, 4.0, 0.0), [1]),
    ((3550, 4.25, -0.25), [7, 7]),
])
def test_reduced_metadata_stops_when_native_selection_is_known(
    reduced_cube, monkeypatch, coordinates, expected_reads
):
    selector, members, _, _ = reduced_cube
    # Required models come first, followed by a large irrelevant tail.
    tail = (
        "star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 3700 4.0 1 0\n"
        "45 46 47\n"
    ).encode() * 1000
    for name in members:
        members[name] += tail

    streams = []
    original_open = Path.open

    def open_metadata(path, *args, **kwargs):
        stream = _CountingTextFile(original_open(path, *args, **kwargs))
        streams.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_metadata)
    spectrum = Spectrum.from_grid(*coordinates, model_grid=selector)
    assert [stream.lines for stream in streams] == expected_reads
    teff, logg, metallicity = coordinates
    np.testing.assert_allclose(
        spectrum.flux.to_value(u.W / (u.m**2 * u.nm)),
        teff / 100 + 2 * logg + 8 * metallicity + np.arange(3),
    )


def test_reduced_late_special_model_takes_precedence_over_interpolation(reduced_cube):
    selector, members, _, _ = reduced_cube
    name = utils.NEWERA_TARBALLS[selector].removesuffix(".tar.gz") + ".Z-0.0.txt"
    # All regular corners precede this exact native special. Finding corners
    # alone must not prematurely end a pending exact-model search.
    members[name] += (
        b"star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 3550 4.25 1 0\n"
        b"900 901 902\n"
    )
    spectrum = Spectrum.from_grid(3550, 4.25, 0.0, model_grid=selector)
    np.testing.assert_allclose(
        spectrum.flux.to_value(u.W / (u.m**2 * u.nm)), [900, 901, 902]
    )


@pytest.mark.parametrize("coordinate", ("teff", "logg", "metallicity", "alpha"))
@pytest.mark.parametrize("value", (np.nan, np.inf))
def test_reduced_invalid_coordinates_fail_before_io(monkeypatch, coordinate, value):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid coordinates must fail before library I/O")

    monkeypatch.setattr(utils, "get_library_root", unexpected)
    coordinates = {"teff": 3500, "logg": 4.0, "metallicity": 0.0, "alpha": 0.0}
    coordinates[coordinate] = value
    with pytest.raises(ValueError, match="coordinates must be finite"):
        Spectrum.from_grid(**coordinates, model_grid="newera_jwst")


@pytest.mark.parametrize("selector", ("newera", *REDUCED))
def test_regular_backbone(selector):
    teffs = utils.GRID_POINTS[selector]["grid_teffs"]
    np.testing.assert_array_equal(teffs[teffs <= 7000], list(range(2300, 7001, 100)))
    np.testing.assert_array_equal(teffs[teffs > 7000], list(range(7200, 12001, 200)))
    np.testing.assert_array_equal(np.diff(teffs[teffs <= 7000]), 100)
    np.testing.assert_array_equal(np.diff(teffs[teffs >= 7000]), 200)
    for temperature in (7100, 7300, 7500, 7700, 7900, 8100, 8300,
                        3350, 5770, 6050, 6060, 9602):
        assert temperature not in teffs


def test_release_mapping_and_override(monkeypatch):
    monkeypatch.delenv("SPECLIB_NEWERA_RECORD_ID", raising=False)
    assert utils.get_newera_record_id() == "18108"
    assert utils.NEWERA_INDEX_FILENAME == "list_of_available_NewEraV3_models.txt"
    assert utils.NEWERA_TARBALLS == {
        "newera_gaia": "PHOENIX-NewEraV3-GAIA-DR4_v3.4-SPECTRA.tar.gz",
        "newera_jwst": "PHOENIX-NewEraV3-JWST-SPECTRA.tar.gz",
        "newera_lowres": "PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz",
    }
    for filename in utils.NEWERA_TARBALLS.values():
        assert utils._get_newera_file_url(filename) == (
            f"https://www.fdr.uni-hamburg.de/record/18108/files/{filename}?download=1"
        )
    monkeypatch.setenv("SPECLIB_NEWERA_RECORD_ID", "custom")
    assert utils.get_newera_record_id() == "custom"
    assert "/record/custom/" in utils._get_newera_file_url("model.txt")


@pytest.fixture
def hsr_inventory(monkeypatch, tmp_path):
    monkeypatch.delenv("SPECLIB_NEWERA_RECORD_ID", raising=False)
    monkeypatch.setattr(utils, "_NEWERA_INDEX_CACHE", {})
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    main_names = (
        "lte03350-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
        "lte07000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
        "lte07200-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
        "lte09602-3.95-0.5.Vega.PHOENIX-NewEra-ACES-COND-2023.HSR.h5",
    )
    main = "indx filename checksum filesize download link\n" + "\n".join(
        f"{i} {name} checksum 100 https://www.fdr.uni-hamburg.de/record/"
        f"{'16738' if i == 0 else '17670'}/files/{name}"
        for i, name in enumerate(main_names)
    )
    additional = "lte02500-3.50+0.5.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"
    catalog = {"contents": [
        {"key": name, "links": {"self": utils._get_newera_file_url(name, "17936")}}
        for name in (additional, main_names[0])
    ]}
    requests = []

    def retrieve(*, url, fname, path, **kwargs):
        requests.append(url)
        target = Path(path) / fname
        if fname == utils.NEWERA_INDEX_FILENAME:
            target.write_text(main)
        else:
            assert url == utils.NEWERA_ADDITIONAL_CATALOG_URL
            assert fname == "newera_additional_17936.json"
            target.write_text(json.dumps(catalog))
        return str(target)

    monkeypatch.setattr(utils.pooch, "retrieve", retrieve)
    return tmp_path, requests


def test_hsr_native_inventory_and_explicit_supplements(hsr_inventory):
    root, requests = hsr_inventory
    models = utils.load_newera_model_list()
    assert (3350, 5.0, 0.0, 0.0) in models["entries"]
    assert (9602, 3.95, -0.5, 0.0) in models["entries"]
    assert (2500, 3.5, 0.5, 0.0) in models["entries"]
    assert (3350, 4.5, 0.0, 0.0) not in models["entries"]
    assert models["additional_record_id"] == "17936"
    assert models["additional_path"] == root / "newera" / "newera_additional_17936.json"
    assert "/record/16738/" in models["urls"][(3350, 5.0, 0.0, 0.0)]
    assert "/record/17936/" in models["urls"][(2500, 3.5, 0.5, 0.0)]
    assert requests == [utils._get_newera_file_url(utils.NEWERA_INDEX_FILENAME),
                        utils.NEWERA_ADDITIONAL_CATALOG_URL]
    assert utils.load_newera_model_list() is models
    assert len(requests) == 2


def test_custom_hsr_record_does_not_merge_canonical_supplements(hsr_inventory, monkeypatch):
    _, requests = hsr_inventory
    monkeypatch.setenv("SPECLIB_NEWERA_RECORD_ID", "custom")
    models = utils.load_newera_model_list()
    assert models["record_id"] == "custom"
    assert "additional_path" not in models
    assert (2500, 3.5, 0.5, 0.0) not in models["entries"]
    assert len(requests) == 1 and "/record/custom/" in requests[0]


@pytest.mark.parametrize("parameters, source", [
    ((3350, 5.0, 0.0, 0.0), "16738"),
    ((2500, 3.5, 0.5, 0.0), "17936"),
    ((9602, 3.95, -0.5, 0.0), "17670"),
])
def test_hsr_download_uses_authoritative_source(hsr_inventory, monkeypatch, parameters, source):
    downloads = []
    def download(url, path, **kwargs):
        downloads.append(url)
        path.write_bytes(b"model")
    monkeypatch.setattr(utils, "download_file", download)
    path = utils.download_newera_file(*parameters)
    assert path.read_bytes() == b"model"
    assert len(downloads) == 1 and f"/record/{source}/" in downloads[0]
    with pytest.raises(FileNotFoundError, match="native inventory"):
        utils.download_newera_file(3350, 4.5, 0.0, 0.0)
    assert len(downloads) == 1


def test_hsr_subset_filters_native_tuples_including_specials(hsr_inventory, monkeypatch):
    downloads = []
    monkeypatch.setattr(utils, "download_file", lambda url, path, **kwargs: downloads.append(url))
    utils.download_newera_hsr_subset(teff_range=(3350, 3350))
    assert len(downloads) == 1 and "lte03350-5.00" in downloads[0]
    assert "/record/16738/" in downloads[0]


@pytest.fixture
def reduced_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    # Two distinct native gravities at 3350 K; sparse regular grid at 3300 K.
    for selector in REDUCED:
        directory = tmp_path / selector
        directory.mkdir()
        filename = utils.NEWERA_TARBALLS[selector].removesuffix(".tar.gz") + ".Z-0.0.txt"
        text = ""
        for teff, logg in ((3300, 5.0), (3350, 4.95), (3350, 5.0),
                           (3400, 4.5), (7000, 5.0), (7200, 5.0)):
            text += (
                "star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 "
                f"{teff} {logg} 1 0\n{teff} {teff + logg} {teff + 10}\n"
            )
        (directory / filename).write_text(text)
    return tmp_path


@pytest.mark.parametrize("selector", REDUCED)
@pytest.mark.parametrize("interpolate", (False, True))
def test_reduced_exact_special_uses_native_headers(reduced_headers, selector, interpolate):
    spectrum = Spectrum.from_grid(3350, 5.0, 0.0, model_grid=selector, interpolate=interpolate)
    np.testing.assert_allclose(spectrum.flux.to_value(u.W / (u.m**2 * u.nm)), [3350, 3355, 3360])
    assert 3350 not in spectrum.grid_teffs
    with pytest.raises(ValueError, match="No matching"):
        utils.load_newera_flux_array(3350, 4.96, 0.0, grid_name=selector)


def test_hsr_exact_vega_and_specials(hsr_inventory):
    models = utils.load_newera_model_list()
    cache = hsr_inventory[0] / "newera"
    for key in ((3350, 5.0, 0.0, 0.0), (9602, 3.95, -0.5, 0.0)):
        with h5py.File(cache / models["entries"][key], "w") as handle:
            handle["PHOENIX_SPECTRUM/wl"] = [9800, 9900, 10000]
            handle["PHOENIX_SPECTRUM/flux"] = np.log10([1, 2, 3])
        spectrum = Spectrum.from_grid(*key[:3], model_grid="newera", interpolate=False)
        assert spectrum.flux.shape == (3,)
        assert key[0] not in spectrum.grid_teffs


@pytest.mark.parametrize("selector", REDUCED)
def test_regular_grid_does_not_promote_specials_or_missing_tuples(reduced_headers, selector):
    grid = SpectralGrid((3300, 3400), (4.5, 5.0), (0.0, 0.0), model_grid=selector)
    np.testing.assert_array_equal(grid.teffs, [3300, 3400])
    np.testing.assert_array_equal(grid.points, [[3300, 5.0, 0.0], [3400, 4.5, 0.0]])
    assert 3350 not in grid.fluxes


def test_binned_regular_grid_skips_missing_newera_tuples(reduced_headers):
    grid = BinnedSpectralGrid(
        (3300, 3400), (4.5, 5.0), (0.0, 0.0),
        center=np.array([990]) * u.nm, width=np.array([30]) * u.nm,
        model_grid="newera_gaia",
    )
    np.testing.assert_array_equal(grid.teffs, [3300, 3400])
    assert 3350 not in grid.fluxes
    assert 4.5 in grid.fluxes[3300] and not grid.fluxes[3300][4.5]
