import pytest
import astropy.units as u
import numpy as np

from speclib import SpectralGrid, BinnedSpectralGrid
from speclib.utils import nearest


@pytest.fixture
def spectral_grid():
    teff_bds = np.array([3000, 4000])
    logg_bds = np.array([4.5, 5.0])
    feh_bds = np.array([0.0, 0.5])
    fluxes = {
        3000: {4.5: {0.0: np.ones(10) * 1.0}},
        4000: {4.5: {0.0: np.ones(10) * 2.0}},
    }
    wavelengths = np.linspace(1.0, 2.0, 10) * u.um
    return SpectralGrid(
        teff_bds=teff_bds,
        logg_bds=logg_bds,
        feh_bds=feh_bds,
        fluxes=fluxes,
        wavelengths=wavelengths,
    )


@pytest.fixture
def binned_grid(spectral_grid):
    return BinnedSpectralGrid.from_spectral_grid(
        spectral_grid, bin_edges=np.linspace(1.0, 2.0, 6) * u.um
    )


def test_spectral_grid_get_spectrum_nearest_exact_match(spectral_grid):
    teff, logg, feh = (
        spectral_grid.teffs[0],
        spectral_grid.loggs[0],
        spectral_grid.fehs[0],
    )
    spec_interp = spectral_grid.get_spectrum(teff, logg, feh, interpolate=True)
    spec_nearest = spectral_grid.get_spectrum(teff, logg, feh, interpolate=False)
    assert u.allclose(spec_interp, spec_nearest)


def test_spectral_grid_get_spectrum_nearest_off_grid(spectral_grid):
    teff = spectral_grid.teffs[0] + 10
    logg = spectral_grid.loggs[0]
    feh = spectral_grid.fehs[0]
    spec = spectral_grid.get_spectrum(teff, logg, feh, interpolate=False)

    teff_nearest = nearest(spectral_grid.teffs, teff)
    expected = spectral_grid.get_spectrum(teff_nearest, logg, feh, interpolate=False)
    assert u.allclose(spec, expected)


def test_interpolated_vs_nearest_spectrum_differ(spectral_grid):
    teff = (spectral_grid.teffs[0] + spectral_grid.teffs[1]) / 2
    logg = spectral_grid.loggs[0]
    feh = spectral_grid.fehs[0]

    interp_spec = spectral_grid.get_spectrum(teff, logg, feh, interpolate=True)
    nearest_spec = spectral_grid.get_spectrum(teff, logg, feh, interpolate=False)
    assert not u.allclose(interp_spec, nearest_spec)


def test_binned_grid_get_spectrum_nearest_off_grid(binned_grid):
    teff = binned_grid.teffs[0] + 10
    logg = binned_grid.loggs[0]
    feh = binned_grid.fehs[0]
    spec = binned_grid.get_spectrum(teff, logg, feh, interpolate=False)

    teff_nearest = nearest(binned_grid.teffs, teff)
    expected = binned_grid.get_spectrum(teff_nearest, logg, feh, interpolate=False)
    assert u.allclose(spec, expected)
