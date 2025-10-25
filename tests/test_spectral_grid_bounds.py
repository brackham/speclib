import numpy as np
import astropy.units as u
import pytest

from speclib.core import SpectralGrid


def test_spectral_grid_clips_bounds_to_available_grid():
    with pytest.warns(UserWarning) as warnings:
        grid = SpectralGrid(
            teff_bds=(2800.0, 3400.0),
            logg_bds=(3.5, 5.5),
            feh_bds=(-0.5, 0.5),
            model_grid="sphinx",
            wavelength=np.linspace(1.0, 2.0, 10) * u.micron,
        )

    messages = {str(record.message) for record in warnings}

    assert (
        "teff_bds (2800.0, 3400.0) truncated to valid range (3000.0, 3000.0)"
        in messages
    )
    assert (
        "logg_bds (3.5, 5.5) truncated to valid range (4.0, 4.5)" in messages
    )
    assert (
        "feh_bds (-0.5, 0.5) truncated to valid range (0.0, 0.0)" in messages
    )

    assert grid.teff_bds == (3000.0, 3000.0)
    assert grid.logg_bds == (4.0, 4.5)
    assert grid.feh_bds == (0.0, 0.0)

    # Ensure the grid still loads spectra covering the truncated range.
    assert np.array_equal(grid.teffs, np.array([3000.0]))
    assert np.array_equal(grid.loggs, np.array([4.0, 4.5]))
    assert np.array_equal(grid.fehs, np.array([0.0]))
