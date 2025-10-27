import pytest
import numpy as np
import astropy.units as u

from speclib import utils
from speclib.core import Spectrum, SpectralGrid, BinnedSpectralGrid


@pytest.fixture(scope="module")
def spectral_grid():
    return SpectralGrid(
        teff_bds=(3000, 3000),
        logg_bds=(4.0, 4.5),
        feh_bds=(0.0, 0.0),
        model_grid="sphinx",
        wavelength=np.linspace(1.0, 2.0, 100) * u.micron,
    )


def test_spectral_grid_get_flux_nearest_exact_match(spectral_grid):
    spec_interp = spectral_grid.get_flux(3000, 4.0, 0.0, interpolate=True)
    spec_nearest = spectral_grid.get_flux(3000, 4.0, 0.0, interpolate=False)
    np.testing.assert_allclose(spec_interp.value, spec_nearest.value, rtol=1e-5)


def test_spectral_grid_get_flux_nearest_off_grid(spectral_grid):
    spec_nearest = spectral_grid.get_flux(3000, 4.25, 0.0, interpolate=False)
    spec_base = spectral_grid.get_flux(3000, 4.0, 0.0, interpolate=False)
    assert np.allclose(spec_nearest.value, spec_base.value)


def test_interpolated_vs_nearest_spectrum_differ(spectral_grid):
    spec_interp = spectral_grid.get_flux(3000, 4.25, 0.0, interpolate=True)
    spec_nearest = spectral_grid.get_flux(3000, 4.25, 0.0, interpolate=False)
    assert not np.allclose(spec_interp.value, spec_nearest.value)


def test_binned_grid_get_spectrum_nearest_off_grid(spectral_grid):
    center = np.linspace(1.0, 2.0, 6) * u.micron
    width = np.full_like(center, fill_value=(center[1] - center[0]))

    grid = BinnedSpectralGrid(
        teff_bds=(3000, 3000),
        logg_bds=(4.0, 4.5),
        feh_bds=(0.0, 0.0),
        center=center,
        width=width,
        model_grid="sphinx",
        wavelength=spectral_grid.wavelength,
    )

    spec_interp = grid.get_spectrum(3000, 4.25, 0.0, interpolate=True)
    spec_nearest = grid.get_spectrum(3000, 4.25, 0.0, interpolate=False)

    assert not np.allclose(spec_interp.value, spec_nearest.value)


def test_get_spectrum_deprecated_alias(spectral_grid):
    with pytest.deprecated_call():
        spectral_grid.get_spectrum(3000, 4.0, 0.0, interpolate=False)


class _DummyInterpolator:
    def __init__(self, flux):
        self._flux = flux

    def __call__(self, point):
        return self._flux


def test_newera_flux_shape_is_1d():
    grid = SpectralGrid.__new__(SpectralGrid)
    grid.model_grid = "newera_jwst"
    grid.interpolator = _DummyInterpolator(np.array([[1.0, 2.0, 3.0]]))
    grid.points = np.array([[1500.0, 4.5, 0.0]])
    grid.data = np.array([[1.0, 2.0, 3.0]])
    grid.unit = u.erg / (u.s * u.cm**2 * u.AA)
    grid.wavelength = np.linspace(1.0, 3.0, 3) * u.AA
    grid.teff_bds = (1400.0, 1600.0)
    grid.logg_bds = (4.0, 5.0)
    grid.feh_bds = (-0.5, 0.5)
    grid.teffs = np.array([1500.0])
    grid.loggs = np.array([4.5])
    grid.fehs = np.array([0.0])
    grid.fluxes = {1500.0: {4.5: {0.0: np.array([1.0, 2.0, 3.0]) * grid.unit}}}

    flux_interp = grid.get_flux(1500.0, 4.5, 0.0, interpolate=True)
    flux_nearest = grid.get_flux(1500.0, 4.5, 0.0, interpolate=False)

    assert flux_interp.shape == (3,)
    assert flux_nearest.shape == (3,)


@pytest.fixture
def mock_newera_grid(monkeypatch):
    teffs = np.array([2300.0, 2400.0])
    loggs = np.array([5.0])
    fehs = np.array([0.0])

    grid_points = {
        "grid_teffs": teffs,
        "grid_loggs": loggs,
        "grid_fehs": fehs,
        "grid_alphas": np.array([0.0]),
    }

    monkeypatch.setitem(utils.GRID_POINTS, "newera_jwst", grid_points)

    wavelength = np.array([980.0, 990.0, 1000.0])

    def fake_load_wave(teff, logg, feh, alpha=0.0, grid_name="newera_jwst"):
        assert grid_name == "newera_jwst"
        return wavelength.copy()

    def fake_load_flux(teff, logg, feh, alpha=0.0, grid_name="newera_jwst"):
        assert grid_name == "newera_jwst"
        base = float(teff)
        return np.array([base, base + 10.0, base + 20.0])

    monkeypatch.setattr(utils, "load_newera_wavelength_array", fake_load_wave)
    monkeypatch.setattr(utils, "load_newera_flux_array", fake_load_flux)

    return {
        "teffs": teffs,
        "loggs": loggs,
        "fehs": fehs,
        "wavelength": wavelength,
    }


def test_newera_spectrum_respects_interpolate_flag(mock_newera_grid):
    lower = Spectrum.from_grid(2300.0, 5.0, 0.0, model_grid="newera_jwst", interpolate=False)
    upper = Spectrum.from_grid(2400.0, 5.0, 0.0, model_grid="newera_jwst", interpolate=False)

    nearest = Spectrum.from_grid(2350.0, 5.0, 0.0, model_grid="newera_jwst", interpolate=False)
    interp = Spectrum.from_grid(2350.0, 5.0, 0.0, model_grid="newera_jwst", interpolate=True)

    np.testing.assert_allclose(nearest.flux.value, lower.flux.value)
    np.testing.assert_allclose(interp.flux.value, 0.5 * (lower.flux.value + upper.flux.value))
    assert not np.allclose(interp.flux.value, nearest.flux.value)


def test_newera_spectral_grid_respects_interpolate_flag(mock_newera_grid):
    grid = SpectralGrid(
        teff_bds=(2300.0, 2400.0),
        logg_bds=(5.0, 5.0),
        feh_bds=(0.0, 0.0),
        model_grid="newera_jwst",
    )

    lower = grid.get_flux(2300.0, 5.0, 0.0, interpolate=False)
    upper = grid.get_flux(2400.0, 5.0, 0.0, interpolate=False)
    nearest = grid.get_flux(2350.0, 5.0, 0.0, interpolate=False)
    interp = grid.get_flux(2350.0, 5.0, 0.0, interpolate=True)

    assert isinstance(nearest, u.Quantity)
    assert isinstance(interp, u.Quantity)

    np.testing.assert_allclose(nearest.value, lower.value)
    np.testing.assert_allclose(interp.value, 0.5 * (lower.value + upper.value))
    assert not np.allclose(interp.value, nearest.value)
