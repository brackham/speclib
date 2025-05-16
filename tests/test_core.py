from speclib import Spectrum
import astropy.units as u
import numpy as np

def test_spectrum_basic():
    wave = np.linspace(5000, 6000, 100) * u.AA
    flux = np.ones_like(wave.value) * u.erg / u.s / u.cm**2 / u.AA
    spec = Spectrum(spectral_axis=wave, flux=flux)

    assert spec.wavelength.unit == u.AA
    assert spec.flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.AA)

def test_spectrum_resample():
    wave = np.linspace(5000, 6000, 100) * u.AA
    flux = np.ones_like(wave.value) * u.erg / u.s / u.cm**2 / u.AA
    spec = Spectrum(spectral_axis=wave, flux=flux)

    new_wave = np.linspace(5100, 5900, 50) * u.AA
    new_spec = spec.resample(new_wave)

    assert len(new_spec.wavelength) == 50
