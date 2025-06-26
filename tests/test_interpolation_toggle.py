import pytest
import numpy as np
import astropy.units as u

from speclib.core import SpectralGrid, BinnedSpectralGrid


@pytest.fixture(scope="module")
def spectral_grid():
    return SpectralGrid(
        teff_bds=(3000, 3000),
        logg_bds=(4.0, 4.5),
        feh_bds=(0.0, 0.0),
        model_grid="sphinx",
        wavelength=np.linspace(1.0, 2.0, 100) * u.micron,
    )


def test_spectral_grid_get_spectrum_nearest_exact_match(spectral_grid):
    spec_interp = spectral_grid.get_spectrum(3000, 4.0, 0.0, interpolate=True)
    spec_nearest = spectral_grid.get_spectrum(3000, 4.0, 0.0, interpolate=False)
    np.testing.assert_allclose(spec_interp.value, spec_nearest.value, rtol=1e-5)


def test_spectral_grid_get_spectrum_nearest_off_grid(spectral_grid):
    spec_nearest = spectral_grid.get_spectrum(3000, 4.25, 0.0, interpolate=False)
    spec_base = spectral_grid.get_spectrum(3000, 4.0, 0.0, interpolate=False)
    assert np.allclose(spec_nearest.value, spec_base.value)


def test_interpolated_vs_nearest_spectrum_differ(spectral_grid):
    spec_interp = spectral_grid.get_spectrum(3000, 4.25, 0.0, interpolate=True)
    spec_nearest = spectral_grid.get_spectrum(3000, 4.25, 0.0, interpolate=False)
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
