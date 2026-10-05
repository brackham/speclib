import copy
import warnings

import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import WCS
import numpy as np
import pytest
from specutils import Spectrum as SpecutilsSpectrum

from speclib import BinnedSpectrum, Spectrum, utils


FLUX_UNIT = u.erg / (u.s * u.cm**2 * u.AA)


def test_spectrum_uses_modern_specutils_base():
    assert Spectrum.__bases__ == (SpecutilsSpectrum,)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AstropyDeprecationWarning)
        Spectrum(flux=np.ones(3) * FLUX_UNIT, spectral_axis=np.arange(3) * u.AA)


@pytest.mark.parametrize("wave_unit", [u.AA, u.nm, u.micron])
def test_spectrum_basic(wave_unit):
    wave = (np.linspace(5000, 6000, 100) * u.AA).to(wave_unit)
    flux = np.ones_like(wave.value) * FLUX_UNIT
    spec = Spectrum(spectral_axis=wave, flux=flux)

    assert isinstance(spec, SpecutilsSpectrum)
    assert spec.spectral_axis.unit == wave_unit
    assert spec.flux.unit == flux.unit
    np.testing.assert_array_equal(spec.spectral_axis, wave)
    np.testing.assert_array_equal(spec.flux, flux)


def test_spectrum_resample():
    wave = np.linspace(5000, 6000, 100) * u.AA
    flux = np.ones_like(wave.value) * u.erg / u.s / u.cm**2 / u.AA
    spec = Spectrum(spectral_axis=wave, flux=flux)

    spec.meta["source"] = {"name": "synthetic"}
    new_wave = np.linspace(510, 590, 50) * u.nm
    new_spec = spec.resample(new_wave)

    assert type(new_spec) is Spectrum
    assert new_spec.spectral_axis.unit == u.nm
    assert new_spec.flux.unit == spec.flux.unit
    np.testing.assert_array_equal(new_spec.spectral_axis, new_wave)
    np.testing.assert_allclose(new_spec.flux.value, 1.0)
    assert new_spec.meta == spec.meta
    new_spec.meta["source"]["name"] = "changed"
    assert spec.meta["source"]["name"] == "synthetic"


@pytest.mark.parametrize("operation", ["slice", "world_slice", "deepcopy"])
def test_spectrum_slicing_and_copying_preserve_state(operation):
    wave = np.arange(500., 510.) * u.nm
    flux = np.arange(10.) * FLUX_UNIT
    mask = np.arange(10) % 3 == 0
    uncertainty = np.full(10, 0.1) * FLUX_UNIT
    spec = Spectrum(
        flux=flux,
        spectral_axis=wave,
        mask=mask,
        uncertainty=StdDevUncertainty(uncertainty),
        meta={"source": {"name": "synthetic"}},
    )
    selected = slice(2, 7) if operation != "deepcopy" else slice(None)
    if operation == "slice":
        result = spec[selected]
    elif operation == "world_slice":
        result = spec[502 * u.nm:507 * u.nm]
    else:
        result = copy.deepcopy(spec)

    assert type(result) is Spectrum
    assert result.spectral_axis.unit == wave.unit
    assert result.flux.unit == flux.unit
    assert result.uncertainty.unit == uncertainty.unit
    np.testing.assert_array_equal(result.spectral_axis, wave[selected])
    np.testing.assert_array_equal(result.flux, flux[selected])
    np.testing.assert_array_equal(result.mask, mask[selected])
    np.testing.assert_array_equal(
        result.uncertainty.array, uncertainty.value[selected]
    )
    assert result.meta["source"] == spec.meta["source"]
    result.meta["source"]["name"] = "changed"
    assert spec.meta["source"]["name"] == "synthetic"
    if operation == "deepcopy":
        result.flux[0] = 100 * FLUX_UNIT
        result.mask[0] = False
        result.uncertainty.array[0] = 10
        assert spec.flux[0] == flux[0]
        assert spec.mask[0]
        assert spec.uncertainty.array[0] == 0.1


def test_multidimensional_spectrum_defaults_to_last_spectral_axis():
    # Equal axis lengths are ambiguous to specutils 2 without an explicit index.
    wave = np.arange(3) * u.nm
    flux = np.arange(9).reshape(3, 3) * FLUX_UNIT
    spec = Spectrum(flux=flux, spectral_axis=wave)

    assert spec.spectral_axis_index == 1
    result = spec[0]
    assert type(result) is Spectrum
    np.testing.assert_array_equal(result.flux, flux[0])
    np.testing.assert_array_equal(result.spectral_axis, wave)

    first_axis = Spectrum(flux=flux, spectral_axis=wave, spectral_axis_index=0)
    assert first_axis.spectral_axis_index == 0


def test_fits_wcs_moves_spectral_axis_and_ancillary_arrays_last():
    wcs = WCS({
        "WCSAXES": 2,
        "CTYPE1": "LINEAR",
        "CTYPE2": "WAVE",
        "CUNIT2": "Angstrom",
        "CRVAL2": 5000.,
        "CDELT2": 1.,
        "CRPIX2": 1.,
    })
    flux = np.arange(12.).reshape(4, 3) * FLUX_UNIT
    mask = flux.value % 3 == 0
    uncertainty = (flux.value + 1) * 0.1 * FLUX_UNIT
    spec = Spectrum(
        flux=flux,
        wcs=wcs,
        mask=mask,
        uncertainty=StdDevUncertainty(uncertainty),
    )

    assert spec.spectral_axis_index == 1
    np.testing.assert_array_equal(spec.flux, flux.T)
    np.testing.assert_array_equal(spec.mask, mask.T)
    np.testing.assert_array_equal(spec.uncertainty.array, uncertainty.value.T)
    np.testing.assert_allclose(
        spec.wavelength.to_value(u.AA), np.arange(5000., 5004.)
    )

    unmoved = Spectrum(flux=flux, wcs=wcs, move_spectral_axis=None)
    assert unmoved.spectral_axis_index == 0
    np.testing.assert_array_equal(unmoved.flux, flux)


@pytest.mark.parametrize("resample", [False, True])
def test_from_grid_returns_speclib_spectrum(monkeypatch, resample):
    # Only model I/O is substituted; construction and resampling use specutils 2.
    wave = np.linspace(500., 510., 101)
    monkeypatch.setattr(
        utils, "load_newera_wavelength_array", lambda *args: wave.copy()
    )
    monkeypatch.setattr(
        utils, "load_newera_flux_array", lambda *args: np.ones(wave.size)
    )
    monkeypatch.setattr(
        utils, "_find_newera_reduced_native_points",
        lambda *args, **kwargs: {(4700., 4.6)},
    )
    new_wave = np.linspace(501., 509., 41) * u.nm if resample else None
    spec = Spectrum.from_grid(
        teff=4700,
        logg=4.6,
        metallicity=0.0,
        model_grid="newera_jwst",
        wavelength=new_wave,
    )

    assert type(spec) is Spectrum
    assert spec.model_grid == "newera_jwst"
    assert spec.flux.unit == FLUX_UNIT
    expected_wave = new_wave if resample else wave * u.nm
    np.testing.assert_allclose(
        spec.wavelength.to_value(u.nm), expected_wave.to_value(u.nm)
    )
    np.testing.assert_allclose(
        spec.flux.value, (1 * u.W / (u.m**2 * u.nm)).to_value(FLUX_UNIT)
    )


def test_spectrum_regularize_preserves_type_units_and_metadata():
    wave = np.array([5000., 5001., 5003., 5004., 5006.]) * u.AA
    spec = Spectrum(
        flux=np.ones(5) * FLUX_UNIT,
        spectral_axis=wave,
        meta={"source": {"name": "synthetic"}},
    )
    result = spec.regularize()

    assert type(result) is Spectrum
    assert result.spectral_axis.unit == wave.unit
    assert result.flux.unit == FLUX_UNIT
    # Preserve the existing linspace convention, including its point count.
    np.testing.assert_array_equal(
        result.spectral_axis, np.linspace(5000., 5006., 6) * u.AA
    )
    np.testing.assert_allclose(result.flux.value, 1.0)
    assert result.meta == spec.meta
    result.meta["source"]["name"] = "changed"
    assert spec.meta["source"]["name"] == "synthetic"


@pytest.mark.parametrize(
    "method, args",
    [
        ("set_spectral_resolution", (1 * u.AA,)),
        ("set_spectral_resolving_power", (2000.,)),
        (
            "set_variable_resolving_power",
            (np.array([5000., 5100.]) * u.AA, [1800., 2200.]),
        ),
    ],
)
def test_spectrum_convolution_preserves_type_units_and_metadata(method, args):
    wave = np.linspace(5000., 5100., 1001) * u.AA
    flux = (1 - 0.5 * np.exp(-0.5 * ((wave.value - 5050) / 0.2)**2)) * FLUX_UNIT
    spec = Spectrum(
        flux=flux,
        spectral_axis=wave,
        meta={"source": {"name": "synthetic"}},
    )
    result = getattr(spec, method)(*args)

    assert type(result) is Spectrum
    assert result.spectral_axis.unit == wave.unit
    assert result.flux.unit == FLUX_UNIT
    np.testing.assert_array_equal(result.spectral_axis, wave)
    np.testing.assert_array_equal(spec.flux, flux)
    assert result.flux.min() > spec.flux.min()
    assert result.meta == spec.meta
    result.meta["source"]["name"] = "changed"
    assert spec.meta["source"]["name"] == "synthetic"


def test_spectrum_bin_preserves_container_and_flux_density():
    wave = np.linspace(5000., 5100., 101) * u.AA
    spec = Spectrum(flux=np.full(101, 2.) * FLUX_UNIT, spectral_axis=wave)
    center = np.array([5020., 5050., 5080.]) * u.AA
    width = np.full(3, 10.) * u.AA
    result = spec.bin(center, width)

    assert type(result) is BinnedSpectrum
    assert result.flux.unit == FLUX_UNIT
    np.testing.assert_array_equal(result.center, center)
    np.testing.assert_array_equal(result.width, width)
    np.testing.assert_allclose(result.flux.value, 2.)
