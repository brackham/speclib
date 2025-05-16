from speclib import Spectrum, Filter, apply_filter
import numpy as np
import astropy.units as u

def test_filter_initialization():
    f = Filter("2MASS J")
    assert f.name == "2MASS J"
    assert hasattr(f, "response")

def test_apply_filter_succeeds():
    wave = np.linspace(11000, 13500, 100) * u.AA
    flux = np.ones_like(wave.value) * u.erg / u.s / u.cm**2 / u.AA
    spec = Spectrum(spectral_axis=wave, flux=flux)
    filt = Filter("2MASS J")

    result = apply_filter(spec, filt)
    assert result.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.AA)  # might change later
