import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from scipy.interpolate import NearestNDInterpolator

from speclib import Spectrum
from speclib import utils
from speclib.core import SpectralGrid


FLUX_UNIT = u.erg / u.s / u.cm**2 / u.AA
LINE_CENTERS = np.array([4500.0, 5500.0, 6500.0])


def _synthetic_spectrum(scale=1.0):
    wave = np.concatenate(
        (
            np.arange(4300.0, 5200.0, 0.04),
            np.arange(5200.0, 6000.0, 0.08),
            np.arange(6000.0, 6700.0, 0.12),
        )
    )
    intrinsic_sigma = 0.24 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    flux = np.ones(wave.size)
    for center in LINE_CENTERS:
        flux += scale * np.exp(-0.5 * ((wave - center) / intrinsic_sigma) ** 2)
    return Spectrum(spectral_axis=wave * u.AA, flux=flux * FLUX_UNIT)


def _measure_fwhm(wavelength, flux, center):
    mask = np.abs(wavelength - center) < 15.0
    wave_window = wavelength[mask]
    flux_window = flux[mask]
    baseline = np.median(np.concatenate((flux_window[:30], flux_window[-30:])))
    profile = flux_window - baseline
    peak_index = np.argmax(profile)
    half_maximum = profile[peak_index] / 2.0

    left = np.interp(
        half_maximum,
        profile[: peak_index + 1],
        wave_window[: peak_index + 1],
    )
    right = np.interp(
        half_maximum,
        profile[peak_index:][::-1],
        wave_window[peak_index:][::-1],
    )
    return right - left


def _line_area_and_centroid(wavelength, flux):
    area = np.trapezoid(flux, wavelength)
    centroid = np.trapezoid(wavelength * flux, wavelength) / area
    return area, centroid


def _make_grid():
    grid = SpectralGrid.__new__(SpectralGrid)
    first = _synthetic_spectrum(scale=1.0)
    second = _synthetic_spectrum(scale=1.5)
    grid.model_grid = "phoenix"
    grid.wavelength = first.wavelength.copy()
    grid.unit = FLUX_UNIT
    grid.points = np.array([[3000.0, 4.0, 0.0], [3100.0, 4.0, 0.0]])
    grid.teffs = np.array([3000.0, 3100.0])
    grid.loggs = np.array([4.0])
    grid.fehs = np.array([0.0])
    grid.teff_bds = (3000.0, 3100.0)
    grid.logg_bds = (4.0, 4.0)
    grid.feh_bds = (0.0, 0.0)
    grid.fluxes = {
        3000.0: {4.0: {0.0: first.flux.copy()}},
        3100.0: {4.0: {0.0: second.flux.copy()}},
    }
    grid.data = np.vstack((first.flux.value, second.flux.value))
    grid.interpolator = NearestNDInterpolator(grid.points, grid.data)
    return grid


def _assert_grid_unchanged(grid, original_data, original_fluxes):
    np.testing.assert_array_equal(grid.data, original_data)
    for point, expected in zip(grid.points, original_fluxes):
        teff, logg, feh = point
        np.testing.assert_array_equal(
            grid.fluxes[teff][logg][feh].value,
            expected,
        )


def test_spectrum_set_spectral_resolution_has_constant_delta_lambda():
    spectrum = _synthetic_spectrum()
    original_flux = spectrum.flux.copy()

    result = spectrum.set_spectral_resolution(4.0 * u.AA)

    widths = np.array(
        [
            _measure_fwhm(result.wavelength.value, result.flux.value, center)
            for center in LINE_CENTERS
        ]
    )
    np.testing.assert_allclose(widths, 4.0, rtol=0.03)
    np.testing.assert_array_equal(result.spectral_axis, spectrum.spectral_axis)
    np.testing.assert_array_equal(spectrum.flux, original_flux)
    assert result is not spectrum


def test_spectrum_set_spectral_resolving_power_has_constant_r():
    spectrum = _synthetic_spectrum()
    original_flux = spectrum.flux.copy()

    result = spectrum.set_spectral_resolving_power(1500.0)

    widths = np.array(
        [
            _measure_fwhm(result.wavelength.value, result.flux.value, center)
            for center in LINE_CENTERS
        ]
    )
    np.testing.assert_allclose(LINE_CENTERS / widths, 1500.0, rtol=0.03)
    np.testing.assert_array_equal(result.spectral_axis, spectrum.spectral_axis)
    np.testing.assert_array_equal(spectrum.flux, original_flux)
    assert result is not spectrum


def test_phase_shifted_lines_preserve_centroids():
    wavelength = np.arange(4900.0, 5100.001, 0.01)
    input_sigma = 0.02
    centroid_shifts = []
    for center in (4999.83, 4999.91, 5000.07, 5000.16):
        flux = np.exp(-0.5 * ((wavelength - center) / input_sigma) ** 2)
        spectrum = Spectrum(
            spectral_axis=wavelength * u.AA,
            flux=flux * FLUX_UNIT,
        )

        result = spectrum.set_spectral_resolution(4.0 * u.AA)

        _, input_centroid = _line_area_and_centroid(wavelength, flux)
        _, output_centroid = _line_area_and_centroid(
            wavelength,
            result.flux.value,
        )
        centroid_shifts.append(output_centroid - input_centroid)

    np.testing.assert_allclose(centroid_shifts, 0.0, atol=1e-6)


def test_spectrum_resolving_power_preserves_isolated_line_flux():
    wavelength = np.arange(4000.0, 8000.01, 0.05)
    center = 6000.13
    flux = np.exp(-0.5 * ((wavelength - center) / 0.08) ** 2)
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=flux * FLUX_UNIT,
    )

    result = spectrum.set_spectral_resolving_power(1500.0)

    input_integral, _ = _line_area_and_centroid(wavelength, flux)
    output_integral, _ = _line_area_and_centroid(
        wavelength,
        result.flux.value,
    )
    assert output_integral == pytest.approx(input_integral, rel=5e-4)


def test_edge_line_has_no_interior_flux_guarantee():
    wavelength = np.arange(5000.0, 5100.01, 0.02)
    flux = np.exp(-0.5 * ((wavelength - 5000.5) / 0.05) ** 2)
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=flux * FLUX_UNIT,
    )

    result = spectrum.set_spectral_resolution(4.0 * u.AA)

    input_integral, _ = _line_area_and_centroid(wavelength, flux)
    output_integral, _ = _line_area_and_centroid(
        wavelength,
        result.flux.value,
    )
    assert output_integral < 0.8 * input_integral


def test_spectrum_resolution_methods_support_descending_wavelengths():
    spectrum = _synthetic_spectrum()
    descending = Spectrum(
        spectral_axis=spectrum.spectral_axis[::-1].copy(),
        flux=spectrum.flux[::-1].copy(),
    )

    ascending_result = spectrum.set_spectral_resolving_power(1500.0)
    descending_result = descending.set_spectral_resolving_power(1500.0)

    np.testing.assert_array_equal(
        descending_result.spectral_axis,
        descending.spectral_axis,
    )
    np.testing.assert_allclose(
        descending_result.flux[::-1],
        ascending_result.flux,
        rtol=1e-12,
    )


def test_spectrum_resolution_methods_return_independent_objects():
    spectrum = _synthetic_spectrum()
    spectrum.meta["nested"] = {"value": 1}

    result = spectrum.set_spectral_resolution(4.0 * u.AA)
    result.spectral_axis[0] = 4200.0 * u.AA
    result.meta["nested"]["value"] = 2

    assert spectrum.spectral_axis[0] != result.spectral_axis[0]
    assert spectrum.meta["nested"]["value"] == 1


def test_spectrum_resolution_methods_reject_masks_and_uncertainties():
    spectrum = _synthetic_spectrum()
    masked = Spectrum(
        spectral_axis=spectrum.spectral_axis,
        flux=spectrum.flux,
        mask=np.zeros(spectrum.flux.size, dtype=bool),
    )
    uncertain = Spectrum(
        spectral_axis=spectrum.spectral_axis,
        flux=spectrum.flux,
        uncertainty=StdDevUncertainty(np.ones(spectrum.flux.size)),
    )

    with pytest.raises(NotImplementedError, match="masked"):
        masked.set_spectral_resolution(4.0 * u.AA)
    with pytest.raises(NotImplementedError, match="uncertainties"):
        uncertain.set_spectral_resolving_power(1500.0)


def test_spectrum_rejects_under_sampled_wavelength_resolution():
    wavelength = np.arange(4900.0, 5101.0, 1.0)
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=np.ones(wavelength.size) * FLUX_UNIT,
    )

    with pytest.raises(ValueError, match="under-sampled"):
        spectrum.set_spectral_resolution(1.99 * u.AA)

    result = spectrum.set_spectral_resolution(2.01 * u.AA)
    np.testing.assert_array_equal(result.spectral_axis, spectrum.spectral_axis)


def test_spectrum_rejects_under_sampled_resolving_power():
    wavelength = np.arange(5000.0, 5101.0, 1.0)
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=np.ones(wavelength.size) * FLUX_UNIT,
    )

    with pytest.raises(ValueError, match="under-sampled"):
        spectrum.set_spectral_resolving_power(3000.0)


def test_spectrum_rejects_impractical_temporary_grid():
    wavelength = 5000.0 + np.concatenate(
        ([0.0, 1e-8], np.arange(1.0, 101.0))
    )
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=np.ones(wavelength.size) * FLUX_UNIT,
    )

    with pytest.raises(ValueError, match="safety limit"):
        spectrum.set_spectral_resolution(3.0 * u.AA)


@pytest.mark.parametrize("value", [0.0 * u.AA, -1.0 * u.AA, np.nan * u.AA])
def test_spectrum_set_spectral_resolution_rejects_nonpositive_or_nonfinite(value):
    with pytest.raises(ValueError):
        _synthetic_spectrum().set_spectral_resolution(value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_spectrum_set_spectral_resolving_power_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _synthetic_spectrum().set_spectral_resolving_power(value)


def test_spectrum_resolution_methods_reject_invalid_units_and_shapes():
    spectrum = _synthetic_spectrum()
    with pytest.raises(u.UnitsError):
        spectrum.set_spectral_resolution(1.0 * u.s)
    with pytest.raises(ValueError):
        spectrum.set_spectral_resolution(np.array([1.0, 2.0]) * u.AA)
    with pytest.raises(u.UnitsError):
        spectrum.set_spectral_resolving_power(1500.0 * u.AA)
    with pytest.raises(ValueError):
        spectrum.set_spectral_resolving_power(np.array([1000.0, 1500.0]))


@pytest.mark.parametrize(
    ("method_name", "value"),
    [
        ("set_spectral_resolution", 4.0 * u.AA),
        ("set_spectral_resolving_power", 1500.0),
    ],
)
def test_spectral_grid_resolution_methods_update_all_spectra_without_mutation(
    method_name,
    value,
):
    grid = _make_grid()
    original_data = grid.data.copy()
    original_fluxes = [
        grid.fluxes[teff][logg][feh].value.copy()
        for teff, logg, feh in grid.points
    ]

    result = getattr(grid, method_name)(value)

    assert result is not grid
    np.testing.assert_array_equal(result.wavelength, grid.wavelength)
    assert not np.array_equal(result.data, grid.data)
    _assert_grid_unchanged(grid, original_data, original_fluxes)

    for row, (teff, logg, feh) in zip(result.data, result.points):
        np.testing.assert_array_equal(
            row,
            result.fluxes[teff][logg][feh].value,
        )
        assert not np.array_equal(row, grid.fluxes[teff][logg][feh].value)

    np.testing.assert_allclose(
        result.interpolator(result.points[0]).reshape(-1),
        result.data[0],
    )

    source_flux = grid.fluxes[3000.0][4.0][0.0]
    expected_spectrum = getattr(
        Spectrum(spectral_axis=grid.wavelength, flux=source_flux),
        method_name,
    )(value)
    np.testing.assert_allclose(result.data[0], expected_spectrum.flux.value)

    widths = np.array(
        [
            _measure_fwhm(result.wavelength.value, result.data[0], center)
            for center in LINE_CENTERS
        ]
    )
    if method_name == "set_spectral_resolution":
        np.testing.assert_allclose(widths, value.to_value(u.AA), rtol=0.03)
    else:
        np.testing.assert_allclose(LINE_CENTERS / widths, value, rtol=0.03)


def test_spectral_grid_result_has_independent_mutable_state():
    grid = _make_grid()
    original_wavelength = grid.wavelength.copy()
    original_points = grid.points.copy()
    original_teffs = grid.teffs.copy()
    original_flux = grid.fluxes[3000.0][4.0][0.0].copy()

    result = grid.set_spectral_resolution(4.0 * u.AA)
    result.wavelength[0] = 4200.0 * u.AA
    result.points[0, 0] = 9999.0
    result.teffs[0] = 9999.0
    result.fluxes[3000.0][4.0][0.0][0] = 99.0 * FLUX_UNIT

    np.testing.assert_array_equal(grid.wavelength, original_wavelength)
    np.testing.assert_array_equal(grid.points, original_points)
    np.testing.assert_array_equal(grid.teffs, original_teffs)
    np.testing.assert_array_equal(grid.fluxes[3000.0][4.0][0.0], original_flux)


@pytest.fixture
def synthetic_grid_loader(monkeypatch):
    monkeypatch.setitem(
        utils.GRID_POINTS,
        "phoenix",
        {
            "grid_teffs": np.array([3000.0, 3100.0]),
            "grid_loggs": np.array([4.0]),
            "grid_fehs": np.array([0.0]),
        },
    )

    def fake_from_grid(cls, teff, logg, feh=0.0, **kwargs):
        return _synthetic_spectrum(scale=1.0 + (teff - 3000.0) / 1000.0)

    monkeypatch.setattr(Spectrum, "from_grid", classmethod(fake_from_grid))


def _construct_synthetic_grid(**kwargs):
    return SpectralGrid(
        teff_bds=(3000.0, 3100.0),
        logg_bds=(4.0, 4.0),
        feh_bds=(0.0, 0.0),
        model_grid="phoenix",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("constructor_keyword", "method_name", "value"),
    [
        ("spectral_resolution", "set_spectral_resolution", 4.0 * u.AA),
        (
            "spectral_resolving_power",
            "set_spectral_resolving_power",
            1500.0,
        ),
    ],
)
def test_spectral_grid_constructor_options_match_methods(
    synthetic_grid_loader,
    constructor_keyword,
    method_name,
    value,
):
    base_grid = _construct_synthetic_grid()
    expected = getattr(base_grid, method_name)(value)
    constructed = _construct_synthetic_grid(**{constructor_keyword: value})

    np.testing.assert_array_equal(constructed.wavelength, base_grid.wavelength)
    np.testing.assert_allclose(constructed.data, expected.data, rtol=1e-12)


def test_spectral_grid_constructor_rejects_both_resolution_options(
    synthetic_grid_loader,
):
    with pytest.raises(ValueError, match="Specify only one"):
        _construct_synthetic_grid(
            spectral_resolution=4.0 * u.AA,
            spectral_resolving_power=1500.0,
        )


def test_spectral_grid_constructor_does_not_regularize(
    synthetic_grid_loader,
    monkeypatch,
):
    def fail_if_called(self, delta_lambda=None):
        raise AssertionError("regularize() must not be called implicitly")

    monkeypatch.setattr(Spectrum, "regularize", fail_if_called)
    grid = _construct_synthetic_grid(spectral_resolution=4.0 * u.AA)

    assert grid.wavelength is not None
