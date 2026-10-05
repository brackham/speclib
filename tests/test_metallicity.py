"""Native-coordinate migration regressions using distinct local grid planes."""

import warnings

import astropy.io.fits as fits
import astropy.units as u
import h5py
import numpy as np
import pytest

from speclib import (
    BinnedSpectralGrid, Filter, SED, SEDGrid, Spectrum, SpectralGrid, utils,
)
from speclib._metallicity import UNSET, resolve_metallicity
from test_mps_atlas import _cube_models, _write_archive, mps_atlas_cache
from test_newera import hsr_inventory, reduced_headers
from test_sphinx import _write_spectrum


SELECTORS = (
    "phoenix", "newera", "newera_gaia", "newera_jwst", "newera_lowres",
    "sphinx", "mps-atlas", "mps-atlas-set1", "mps-atlas-set2",
)


@pytest.fixture
def forbid_library_io(monkeypatch):
    def fail():
        pytest.fail("Invalid metallicity must be rejected before library I/O")

    monkeypatch.setattr(utils, "get_library_root", fail)


@pytest.fixture(params=SELECTORS)
def native_library(request, monkeypatch, tmp_path):
    """Write an affine cube whose metallicity planes differ by four units."""
    selector = request.param
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    kwargs = {"model_grid": selector}
    metallicities = (-0.5, 0.0)
    if selector.startswith("mps-atlas"):
        model_set = "set2" if selector.endswith("set2") else "set1"
        _write_archive(tmp_path, model_set, _cube_models())
        monkeypatch.setattr(
            utils, "download_mps_atlas_grid",
            lambda *args, **kw: tmp_path / "mps-atlas" / model_set,
        )
        monkeypatch.setattr(utils, "_MPS_ATLAS_INDEX_CACHE", {})
        teffs, loggs, metallicities = (3500., 3600.), (3., 3.5), (0., 0.1)
    else:
        teffs, loggs = (3500., 3600.), (4., 4.5)
        directory = tmp_path / selector
        directory.mkdir()
        entries = {}
        for ti, teff in enumerate(teffs):
            for gi, logg in enumerate(loggs):
                for mi, metallicity in enumerate(metallicities):
                    scale = 1 + ti + 2 * gi + 4 * mi
                    if selector == "phoenix":
                        metal_label = -0.0 if metallicity == 0 else metallicity
                        name = (
                            f"lte{teff:05.0f}-{logg:.2f}{metal_label:+.1f}."
                            "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
                        )
                        fits.writeto(directory / name, scale * np.array([1., 2., 3.]))
                    elif selector == "newera":
                        name = f"model_{ti}_{gi}_{mi}.h5"
                        entries[(teff, logg, metallicity, 0.)] = name
                        with h5py.File(directory / name, "w") as handle:
                            handle["PHOENIX_SPECTRUM/wl"] = [9800., 9900., 10000.]
                            handle["PHOENIX_SPECTRUM/flux"] = np.log10(
                                scale * np.array([1., 2., 3.])
                            )
                    elif selector.startswith("newera_"):
                        prefix = utils.NEWERA_TARBALLS[selector].removesuffix(".tar.gz")
                        label = "Z-0.0" if metallicity == 0 else f"Z{metallicity:+.1f}"
                        with (directory / f"{prefix}.{label}.txt").open("a") as file:
                            file.write(
                                "star BPRP PHH 20250708 02 PHOENIX1D 0 10 3 980 1000 10 "
                                f"{teff} {logg} 1 0\n{scale} {2*scale} {3*scale}\n"
                            )
                    else:
                        _write_spectrum(directory, teff, logg, metallicity, 0.5, scale)
        if selector == "phoenix":
            fits.writeto(
                directory / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
                np.array([9800., 9900., 10000.]),
            )
        elif selector == "newera":
            monkeypatch.setattr(utils, "load_newera_model_list", lambda **kw: {"entries": entries})
        elif selector == "sphinx":
            kwargs["co_ratio"] = 0.5
            monkeypatch.setattr(utils, "_SPHINX_INDEX_CACHE", {})
    return selector, kwargs, teffs, loggs, metallicities


def test_native_aliases_preserve_model_plane_and_interpolation(native_library):
    selector, kwargs, teffs, loggs, metallicities = native_library
    native = "feh" if selector == "phoenix" else "mh"
    default = Spectrum.from_grid(teffs[0], loggs[0], **kwargs)
    explicit_zero = Spectrum.from_grid(teffs[0], loggs[0], metallicity=0., **kwargs)
    np.testing.assert_array_equal(default.flux, explicit_zero.flux)
    assert default.meta["metallicity"] == 0.
    # Check both metallicity planes: aliasing must not select a different file.
    for metallicity in metallicities:
        positional = Spectrum.from_grid(teffs[0], loggs[0], metallicity, **kwargs)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            canonical = Spectrum.from_grid(teffs[0], loggs[0], metallicity=metallicity, **kwargs)
            alias = Spectrum.from_grid(teffs[0], loggs[0], **{native: metallicity}, **kwargs)
        assert not records
        np.testing.assert_array_equal(canonical.flux, positional.flux)
        np.testing.assert_array_equal(alias.flux, positional.flux)
        assert canonical.meta["metallicity"] == metallicity
        assert canonical.meta["metallicity_type"] == native
    # Reduced NewEra probes a requested plane before interpolation; preserve
    # this established behavior and interpolate Teff/logg on a cached plane.
    reduced = selector.startswith("newera_")
    midpoint = (np.mean(teffs), np.mean(loggs),
                metallicities[0] if reduced else np.mean(metallicities))
    lower = Spectrum.from_grid(teffs[0], loggs[0], metallicities[0], **kwargs).flux
    expected = lower * (2.5 if reduced else 4.5)
    interpolated = Spectrum.from_grid(*midpoint, **kwargs)
    np.testing.assert_allclose(interpolated.flux, expected)
    assert interpolated.meta["metallicity"] == midpoint[2]
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        alias = Spectrum.from_grid(
            midpoint[0], midpoint[1], **{native: midpoint[2]}, **kwargs
        )
    assert not records
    np.testing.assert_array_equal(alias.flux, interpolated.flux)
    # Off-grid nearest selection must report the selected coordinate, not input.
    nearest = Spectrum.from_grid(
        teffs[0], loggs[0], metallicity=metallicities[0] + 0.1 * np.ptp(metallicities),
        interpolate=False, **kwargs,
    )
    assert nearest.meta["metallicity"] == metallicities[0]
    np.testing.assert_array_equal(nearest.flux, lower)


@pytest.mark.parametrize("grid_class", (SpectralGrid, BinnedSpectralGrid))
def test_grid_bounds_and_native_retrieval(native_library, grid_class):
    selector, kwargs, teffs, loggs, metallicities = native_library
    if grid_class is BinnedSpectralGrid:
        source = Spectrum.from_grid(teffs[0], loggs[0], metallicities[0], **kwargs)
        kwargs = dict(kwargs, center=np.array([source.wavelength.value.mean()]) * u.AA,
                      width=np.array([1.01 * np.ptp(source.wavelength.value)]) * u.AA)
    grid = grid_class(teffs, loggs, metallicity_bds=metallicities, **kwargs)
    assert grid.metallicity_type == ("feh" if selector == "phoenix" else "mh")
    assert grid.meta["metallicity_type"] == grid.metallicity_type
    assert grid.meta["metallicities"] == tuple(metallicities)
    np.testing.assert_array_equal(grid.metallicities, metallicities)
    get = "get_flux" if grid_class is SpectralGrid else "get_spectrum"
    query = (np.mean(teffs), np.mean(loggs), np.mean(metallicities))
    result = getattr(grid, get)(query[0], query[1], metallicity=query[2])
    np.testing.assert_array_equal(result, getattr(grid, get)(*query))
    lower = getattr(grid, get)(teffs[0], loggs[0], metallicity=metallicities[0])
    np.testing.assert_allclose(result, lower * 4.5)
    native = "feh" if selector == "phoenix" else "mh"
    np.testing.assert_array_equal(
        result, getattr(grid, get)(query[0], query[1], **{native: query[2]})
    )
    for removed in ("feh_bds", "fehs", "grid_fehs"):
        with pytest.raises(AttributeError, match=removed):
            getattr(grid, removed)
    if native == "mh":
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=r"uses \[M/H\].*metallicity=.*mh="):
                getattr(grid, get)(query[0], query[1], feh=query[2])
        assert not records
    else:
        with pytest.raises(ValueError, match=r"uses \[Fe/H\]"):
            getattr(grid, get)(query[0], query[1], mh=query[2])
    if native == "mh":
        mh_grid = grid_class(teffs, loggs, mh_bds=metallicities, **kwargs)
        np.testing.assert_array_equal(result, getattr(mh_grid, get)(*query))


@pytest.mark.parametrize("keywords", (
    {"metallicity": -0.5, "feh": -0.5},
    {"metallicity": -0.5, "mh": 0.0},
    {"metallicity": 0.0, "mh": 0.0},
    {"feh": 0.0, "mh": 0.0},
))
@pytest.mark.parametrize("selector", SELECTORS)
def test_conflicting_coordinate_keywords_fail_before_io(keywords, selector):
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        Spectrum.from_grid(3500, 4., model_grid=selector, **keywords)


@pytest.mark.parametrize("grid_class", (SpectralGrid, BinnedSpectralGrid))
def test_conflicting_bounds_fail_before_io(grid_class):
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        grid_class((3500, 3500), (4., 4.), metallicity_bds=(0., 0.), mh_bds=(0., 0.))


@pytest.mark.parametrize("grid_class", (SpectralGrid, BinnedSpectralGrid))
@pytest.mark.parametrize("selector", (
    "phoenix", "newera", "newera_gaia", "newera_jwst", "newera_lowres",
))
@pytest.mark.parametrize("keywords", (
    {"metallicity": 0.}, {"feh": 0.}, {"mh": 0.}, {"feh": 0., "mh": 0.},
))
def test_grid_rejects_scalar_coordinate_overrides_before_io(
    grid_class, selector, keywords, forbid_library_io,
):
    options = {}
    if grid_class is BinnedSpectralGrid:
        options = {"center": np.array([9900.]) * u.AA,
                   "width": np.array([202.]) * u.AA}
    with pytest.raises(TypeError, match="scalar metallicity keywords.*metallicity_bds"):
        grid_class((3500, 3500), (4., 4.), metallicity_bds=(0., 0.),
                   model_grid=selector, **keywords, **options)


@pytest.mark.parametrize("grid_class", (SpectralGrid, BinnedSpectralGrid))
@pytest.mark.parametrize("missing_error", (ValueError, FileNotFoundError))
def test_grid_still_skips_unavailable_newera_models(monkeypatch, grid_class, missing_error):
    wavelength = np.array([9800., 9900., 10000.]) * u.AA
    flux_unit = u.erg / (u.s * u.cm**2 * u.AA)

    def load(cls, teff, logg, metallicity, **kwargs):
        if teff == 3600:
            raise missing_error("Native model is unavailable")
        return cls(spectral_axis=wavelength, flux=np.array([1., 2., 3.]) * flux_unit)

    monkeypatch.setattr(Spectrum, "from_grid", classmethod(load))
    options = {}
    if grid_class is BinnedSpectralGrid:
        options = {"center": np.array([9900.]) * u.AA,
                   "width": np.array([202.]) * u.AA}
    grid = grid_class((3500, 3600), (4., 4.), metallicity_bds=(0., 0.),
                      model_grid="newera_jwst", **options)
    assert not grid.fluxes[3600][4.]
    expected = [2.] if grid_class is BinnedSpectralGrid else [1., 2., 3.]
    np.testing.assert_allclose(grid.fluxes[3500][4.][0.].to_value(flux_unit), expected)
    if grid_class is SpectralGrid:
        np.testing.assert_array_equal(grid.points, [[3500., 4., 0.]])
        np.testing.assert_array_equal(grid.wavelength, wavelength)


def test_phoenix_rejects_non_native_mh():
    with pytest.raises(ValueError, match="not native"):
        Spectrum.from_grid(3500, 4., mh=0., model_grid="phoenix")


def test_newera_subset_range_alias_filters_same_inventory(hsr_inventory, monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "download_file", lambda url, path, **kw: calls.append(url))
    utils.download_newera_hsr_subset(metallicity_range=(-0.5, -0.5))
    canonical = calls.copy()
    assert len(canonical) == 1 and "Vega" in canonical[0]
    calls.clear()
    utils.download_newera_hsr_subset(mh_range=(-0.5, -0.5))
    assert calls == canonical
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        utils.download_newera_hsr_subset(
            metallicity_range=(-0.5, -0.5), mh_range=(-0.5, -0.5)
        )


@pytest.mark.parametrize("factory", (Spectrum.from_grid, SED.from_grid))
@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("coordinate", ("metallicity", "native"))
def test_factory_explicit_none_is_invalid(factory, selector, coordinate, forbid_library_io):
    name = coordinate if coordinate != "native" else (
        "feh" if selector == "phoenix" else "mh"
    )
    options = {"filters": []} if factory.__self__ is SED else {}
    message = f"{name} must be a numeric value"
    if factory.__self__ is Spectrum:
        message += "; omit the argument to use the default"
    with pytest.raises(TypeError) as error:
        factory(3500, 4., model_grid=selector, **{name: None}, **options)
    assert str(error.value) == message


@pytest.mark.parametrize("factory", (Spectrum.from_grid, SED.from_grid))
def test_factory_positional_none_is_invalid(factory, forbid_library_io):
    with pytest.raises(TypeError, match="^metallicity must be a numeric value"):
        factory(3500, 4., None)


@pytest.mark.parametrize("grid_class, name, selector", (
    (SpectralGrid, "metallicity_bds", "phoenix"),
    (SpectralGrid, "mh_bds", "sphinx"),
    (BinnedSpectralGrid, "metallicity_bds", "phoenix"),
    (BinnedSpectralGrid, "mh_bds", "sphinx"),
    (SEDGrid, "metallicity_bds", "phoenix"),
))
def test_bounds_explicit_none_is_invalid(grid_class, name, selector, forbid_library_io):
    with pytest.raises(TypeError) as error:
        grid_class((3500, 3500), (4., 4.), model_grid=selector, **{name: None})
    assert str(error.value) == f"{name} must be a pair of numeric values"


@pytest.mark.parametrize("grid_class, method, selector", (
    (SpectralGrid, "get_flux", "phoenix"),
    (SpectralGrid, "get_flux", "sphinx"),
    (BinnedSpectralGrid, "get_spectrum", "phoenix"),
    (BinnedSpectralGrid, "get_spectrum", "sphinx"),
    (SEDGrid, "get_SED", "phoenix"),
))
@pytest.mark.parametrize("coordinate", ("metallicity", "native"))
def test_grid_retrieval_explicit_none_is_invalid(grid_class, method, selector, coordinate):
    # Validation must precede any access to loaded grid data.
    grid = grid_class.__new__(grid_class)
    grid.model_grid = selector
    name = coordinate if coordinate != "native" else (
        "feh" if selector == "phoenix" else "mh"
    )
    with pytest.raises(TypeError) as error:
        getattr(grid, method)(3500, 4., **{name: None})
    assert str(error.value) == f"{name} must be a numeric value"


@pytest.mark.parametrize("helper, names", (
    (utils.load_newera_wavelength_array, ("metallicity", "mh", "z")),
    (utils.load_newera_flux_array, ("metallicity", "mh", "z")),
    (utils.download_newera_file, ("metallicity", "mh", "zscale")),
    (utils.load_mps_atlas_spectrum, ("metallicity", "mh")),
    (utils.load_sphinx_spectrum, ("metallicity", "mh")),
))
def test_utility_explicit_none_is_invalid_without_warning(helper, names, forbid_library_io):
    options = {"alpha_scale": 0.} if helper is utils.download_newera_file else {}
    for name in names:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(TypeError) as error:
                helper(3500, 4., **{name: None}, **options)
        assert str(error.value) == f"{name} must be a numeric value"
        assert not records


@pytest.mark.parametrize("name", ("metallicity_range", "mh_range"))
def test_range_explicit_none_is_invalid(name, forbid_library_io):
    with pytest.raises(TypeError) as error:
        utils.download_newera_hsr_subset(**{name: None})
    assert str(error.value) == (
        f"{name} must be a pair of numeric values; omit the argument to use the default"
    )


def test_omitted_range_preserves_full_metallicity_inventory(hsr_inventory, monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "download_file", lambda url, path, **kw: calls.append(url))
    utils.download_newera_hsr_subset(teff_range=(2500, 10000))
    assert len(calls) == 5
    assert any("Vega" in url for url in calls)
    assert any("lte02500" in url for url in calls)


def test_resolver_distinguishes_omission_from_explicit_none():
    assert resolve_metallicity(UNSET, "phoenix", default=0.) == 0.
    assert resolve_metallicity(
        UNSET, "newera", default=None, parameter="metallicity_range"
    ) is None
    with pytest.raises(TypeError, match="^Missing required argument: metallicity$"):
        resolve_metallicity(UNSET, "phoenix")
    for default in (UNSET, 0., None):
        with pytest.raises(TypeError, match="^metallicity must be a numeric value"):
            resolve_metallicity(None, "phoenix", default=default)


@pytest.mark.parametrize("keywords", ({"metallicity": None, "mh": 0.},
                                      {"metallicity": 0., "mh": None}))
def test_explicit_none_still_counts_as_supplied_for_conflicts(keywords):
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        Spectrum.from_grid(3500, 4., model_grid="sphinx", **keywords)


@pytest.mark.parametrize("helper", (
    utils.load_newera_wavelength_array, utils.load_newera_flux_array,
    utils.download_newera_file,
))
def test_newera_utility_conflicts_fail_before_io(helper):
    options = {"alpha_scale": 0.} if helper is utils.download_newera_file else {}
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        helper(3500, 4., metallicity=0., mh=0., **options)


def test_sed_and_metadata_transformations(native_library):
    selector, kwargs, teffs, loggs, metallicities = native_library
    source = Spectrum.from_grid(teffs[0], loggs[0], metallicities[0], **kwargs)
    # A unit response over the fixture wavelengths isolates coordinate routing
    # and avoids extrapolating a physical filter beyond its support.
    filt = Filter.__new__(Filter)
    filt.wl_eff = source.wavelength.mean()
    filt.bandwidth = source.wavelength[-1] - source.wavelength[0]
    filt.response = Spectrum(
        spectral_axis=source.wavelength,
        flux=np.ones(source.flux.size) * u.dimensionless_unscaled,
    )
    sed = SED.from_grid(teffs[0], loggs[0], metallicity=metallicities[0], filters=[filt], **kwargs)
    native = "feh" if selector == "phoenix" else "mh"
    alias = SED.from_grid(teffs[0], loggs[0], filters=[filt], **{native: metallicities[0]}, **kwargs)
    np.testing.assert_array_equal(alias.flux, sed.flux)
    assert sed.meta["metallicity"] == metallicities[0]
    assert sed.meta["metallicity_type"] == native
    selected = source.select_wavelength(source.wavelength[0], source.wavelength[-1])
    assert selected.meta == source.meta
    binned = source.bin(
        np.array([source.wavelength.value.mean()]) * u.AA,
        np.array([1.01 * np.ptp(source.wavelength.value)]) * u.AA,
    )
    assert binned.meta == source.meta
    binned.meta["metallicity"] = 100
    assert source.meta["metallicity"] == metallicities[0]
    if selector == "phoenix":
        grid = SEDGrid(teffs, loggs, metallicity_bds=metallicities, filters=[filt])
        query = (np.mean(teffs), np.mean(loggs), np.mean(metallicities))
        np.testing.assert_array_equal(
            grid.get_SED(query[0], query[1], metallicity=query[2]),
            grid.get_SED(query[0], query[1], feh=query[2]),
        )
        assert grid.meta["metallicity_type"] == "feh"
        for removed in ("feh_bds", "fehs", "grid_fehs"):
            with pytest.raises(AttributeError, match=removed):
                getattr(grid, removed)


@pytest.mark.parametrize("helper", (
    utils.load_newera_wavelength_array, utils.load_newera_flux_array,
))
@pytest.mark.parametrize("selector", ("newera_gaia", "newera_jwst", "newera_lowres"))
def test_reduced_utility_storage_and_native_aliases(reduced_headers, helper, selector):
    canonical = helper(3350, 5., metallicity=0., grid_name=selector)
    np.testing.assert_array_equal(canonical, helper(3350, 5., mh=0., grid_name=selector))
    with pytest.warns(DeprecationWarning, match="no .* conversion"):
        value = helper(3350, 5., z=0., grid_name=selector)
    np.testing.assert_array_equal(canonical, value)
    with pytest.raises(TypeError, match="unexpected keyword argument.*feh"):
        helper(3350, 5., feh=0., grid_name=selector)
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        helper(3350, 5., metallicity=0., z=0., grid_name=selector)


def test_hsr_utility_storage_and_native_aliases(hsr_inventory, monkeypatch):
    requests = []
    monkeypatch.setattr(utils, "download_file", lambda url, path, **kw: requests.append(url))
    canonical = utils.download_newera_file(9602, 3.95, metallicity=-0.5, alpha_scale=0.)
    assert "Vega" in canonical.name
    assert canonical == utils.download_newera_file(9602, 3.95, mh=-0.5, alpha_scale=0.)
    with pytest.warns(DeprecationWarning, match="no .* conversion"):
        assert canonical == utils.download_newera_file(9602, 3.95, zscale=-0.5, alpha_scale=0.)
    assert len(requests) == 3 and len(set(requests)) == 1
    with pytest.raises(TypeError, match="unexpected keyword argument.*feh"):
        utils.download_newera_file(9602, 3.95, feh=-0.5, alpha_scale=0.)
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        utils.download_newera_file(9602, 3.95, metallicity=-0.5, zscale=-0.5, alpha_scale=0.)


@pytest.mark.parametrize("args, keywords", (
    ((0.,), {}), ((), {"metallicity": 0.}),
    ((), {"mh": 0.}), ((), {"zscale": 0.}),
))
def test_hsr_download_requires_explicit_alpha_scale(args, keywords, forbid_library_io):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="^Missing required argument: alpha_scale$"):
            utils.download_newera_file(3500, 4., *args, **keywords)
    assert not records


@pytest.mark.parametrize("model_set", ("set1", "set2"))
def test_mps_utility_native_aliases(mps_atlas_cache, model_set):
    canonical = utils.load_mps_atlas_spectrum(3500, 3., metallicity=0.1, model_set=model_set)
    native = utils.load_mps_atlas_spectrum(3500, 3., mh=0.1, model_set=model_set)
    for actual, expected in zip(native, canonical):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(TypeError, match="unexpected keyword argument.*feh"):
        utils.load_mps_atlas_spectrum(3500, 3., feh=0.1, model_set=model_set)
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        utils.load_mps_atlas_spectrum(3500, 3., metallicity=0.1, mh=0.1)


def test_sphinx_utility_native_aliases():
    canonical = utils.load_sphinx_spectrum(3000, 4., metallicity=0., co_ratio=0.5)
    native = utils.load_sphinx_spectrum(3000, 4., mh=0., co_ratio=0.5)
    for actual, expected in zip(native, canonical):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(TypeError, match="unexpected keyword argument.*feh"):
        utils.load_sphinx_spectrum(3000, 4., feh=0., co_ratio=0.5)
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        utils.load_sphinx_spectrum(3000, 4., metallicity=0., mh=0., co_ratio=0.5)


def test_sed_coordinate_conflict():
    with pytest.raises(ValueError, match="only one metallicity keyword"):
        SED.from_grid(3500, 4., metallicity=0., feh=0., filters=[])


@pytest.mark.parametrize("selector", (
    *(selector for selector in SELECTORS if selector != "phoenix"),
))
@pytest.mark.parametrize("factory", (Spectrum.from_grid, SED.from_grid))
def test_mh_libraries_reject_feh_before_io_without_warning(selector, factory):
    options = {"filters": []} if factory.__self__ is SED else {}
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as error:
            factory(3500, 4., feh=-0.5, model_grid=selector, **options)
    assert not records
    message = str(error.value)
    assert "uses [M/H]" in message
    assert "`metallicity=` or `mh=`" in message
    assert "does not convert between [Fe/H] and [M/H]" in message


@pytest.fixture(params=("drift-phoenix", "nextgen-solar"))
def legacy_library(request, monkeypatch, tmp_path):
    selector = request.param
    monkeypatch.setattr(utils, "get_library_root", lambda: tmp_path)
    directory = tmp_path / selector
    directory.mkdir()
    if selector == "drift-phoenix":
        teffs, loggs, metallicities = (1000., 1100.), (3., 3.5), (-0.3, 0.)
        filename = "lte_{:4.0f}_{:0.1f}{:+0.1f}.7.dat.txt"
    else:
        teffs, loggs, metallicities = (1600., 1700.), (3.5, 4.), (0.,)
        filename = "lte{:05.0f}_{:+0.1f}_{:+.1f}_NextGen-solar.dat"
        np.savetxt(
            directory / "lte01600_+5.5_+0.0_NextGen-solar.dat",
            np.column_stack(([9800., 9900., 10000.], [1., 2., 3.])),
        )
    for ti, teff in enumerate(teffs):
        for gi, logg in enumerate(loggs):
            for mi, metallicity in enumerate(metallicities):
                stored = -0.0 if selector == "drift-phoenix" and metallicity == 0 else metallicity
                flux = (1 + ti + 2 * gi + 4 * mi) * np.array([1., 2., 3.])
                np.savetxt(
                    directory / filename.format(teff, logg, stored),
                    np.column_stack(([9800., 9900., 10000.], flux)),
                )
    return selector, teffs, loggs, metallicities


def test_legacy_spectrum_preserves_feh_defaults_and_interpolation(legacy_library):
    selector, teffs, loggs, metallicities = legacy_library
    for metallicity in metallicities:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            positional = Spectrum.from_grid(teffs[0], loggs[0], metallicity, model_grid=selector)
            canonical = Spectrum.from_grid(teffs[0], loggs[0], metallicity=metallicity,
                                           model_grid=selector)
            legacy = Spectrum.from_grid(teffs[0], loggs[0], feh=metallicity, model_grid=selector)
        assert not records
        np.testing.assert_array_equal(legacy.flux, positional.flux)
        np.testing.assert_array_equal(canonical.flux, positional.flux)
        assert legacy.meta == {}
        assert legacy.model_grid == selector
    default = Spectrum.from_grid(teffs[0], loggs[0], model_grid=selector)
    explicit_zero = Spectrum.from_grid(teffs[0], loggs[0], feh=0., model_grid=selector)
    np.testing.assert_array_equal(default.flux, explicit_zero.flux)
    lower = Spectrum.from_grid(teffs[0], loggs[0], metallicities[0], model_grid=selector)
    midpoint = Spectrum.from_grid(np.mean(teffs), np.mean(loggs),
                                  feh=np.mean(metallicities), model_grid=selector)
    np.testing.assert_allclose(midpoint.flux, lower.flux * (4.5 if len(metallicities) == 2 else 2.5))
    assert midpoint.meta == {}


@pytest.mark.parametrize("grid_class, method", (
    (SpectralGrid, "get_flux"), (BinnedSpectralGrid, "get_spectrum"),
))
def test_legacy_grid_preserves_feh_retrieval_without_native_metadata(
    legacy_library, grid_class, method,
):
    selector, teffs, loggs, metallicities = legacy_library
    options = {}
    if grid_class is BinnedSpectralGrid:
        options = {"center": np.array([9900.]) * u.AA,
                   "width": np.array([202.]) * u.AA}
    grid = grid_class(teffs, loggs, (min(metallicities), max(metallicities)),
                      model_grid=selector, **options)
    assert not hasattr(grid, "metallicity_type")
    assert grid.meta == {}
    query = (np.mean(teffs), np.mean(loggs), np.mean(metallicities))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        legacy = getattr(grid, method)(query[0], query[1], feh=query[2])
    assert not records
    np.testing.assert_array_equal(legacy, getattr(grid, method)(*query))
    lower = getattr(grid, method)(teffs[0], loggs[0], metallicities[0])
    np.testing.assert_allclose(legacy, lower * (4.5 if len(metallicities) == 2 else 2.5))


def test_legacy_sed_preserves_feh_without_native_metadata(legacy_library):
    selector, teffs, loggs, metallicities = legacy_library
    source = Spectrum.from_grid(teffs[0], loggs[0], metallicities[0], model_grid=selector)
    filt = Filter.__new__(Filter)
    filt.wl_eff = source.wavelength.mean()
    filt.bandwidth = source.wavelength[-1] - source.wavelength[0]
    filt.response = Spectrum(spectral_axis=source.wavelength,
                             flux=np.ones(source.flux.size) * u.dimensionless_unscaled)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        legacy = SED.from_grid(teffs[0], loggs[0], feh=metallicities[0],
                               filters=[filt], model_grid=selector)
    assert not records
    positional = SED.from_grid(teffs[0], loggs[0], metallicities[0],
                              filters=[filt], model_grid=selector)
    np.testing.assert_array_equal(legacy.flux, positional.flux)
    assert legacy.meta == {}


@pytest.mark.parametrize("selector", ("drift-phoenix", "nextgen-solar"))
@pytest.mark.parametrize("factory", (Spectrum.from_grid, SED.from_grid))
def test_legacy_mh_remains_an_unsupported_keyword(selector, factory, forbid_library_io):
    with pytest.raises(TypeError, match="unexpected keyword argument 'mh'"):
        factory(1000, 3., mh=0., model_grid=selector)


@pytest.mark.parametrize("grid_class", (SpectralGrid, BinnedSpectralGrid, SEDGrid))
@pytest.mark.parametrize("canonical_also_supplied", (False, True))
def test_removed_bounds_keyword_is_rejected_before_io(grid_class, canonical_also_supplied):
    options = {"metallicity_bds": (0., 0.)} if canonical_also_supplied else {}
    with pytest.raises(TypeError, match="feh_bds"):
        grid_class((3500, 3500), (4., 4.), feh_bds=(0., 0.), **options)


@pytest.mark.parametrize("canonical_also_supplied", (False, True))
def test_removed_range_keyword_is_rejected_before_io(canonical_also_supplied):
    options = {"metallicity_range": (-0.5, -0.5)} if canonical_also_supplied else {}
    with pytest.raises(TypeError, match="unexpected keyword argument.*feh_range"):
        utils.download_newera_hsr_subset(feh_range=(-0.5, -0.5), **options)


def test_clean_grid_and_range_signatures():
    import inspect

    for api in (SpectralGrid, BinnedSpectralGrid, SEDGrid):
        assert "feh_bds" not in inspect.signature(api).parameters
        for removed in ("feh_bds", "fehs", "grid_fehs"):
            assert not hasattr(api, removed)
    assert "feh_range" not in inspect.signature(utils.download_newera_hsr_subset).parameters
