from speclib import Spectrum, Filter, SED


def test_sed_from_spectrum():
    spec = Spectrum.from_grid(3000, 4.0, 0.0, model_grid="sphinx")
    filt = Filter("2MASS J")
    sed = SED(spec, [filt])
    assert sed.flux.unit.is_equivalent(filt.zeropoint_flux.unit)
