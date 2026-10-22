"""Scientific regression contracts for NIRCam Module A Grism R time series."""
from dataclasses import replace
import hashlib
from importlib.resources import files
import importlib.util
from pathlib import Path
import socket

import astropy.units as u
import numpy as np
import pytest

from speclib import Spectrum, SpectralGrid, get_profile, list_instrument_profiles
from test_instruments import FLUX_UNIT, make_grid

PREFIX = 'jwst/nircam/tsgrism/modulea/grismr/'
FILTERS = {'f277w': (2.410, 3.179), 'f322w2': (2.413, 4.083),
           'f356w': (3.100, 4.041), 'f444w': (3.835, 4.985)}
ROOT = files('speclib.data').joinpath('instruments')


def polynomial(x):
    # Independent transcription of Greene et al. (2017), Section 3.2.
    return 3.35*x**4 - 41.9*x**3 + 95.5*x**2 + 536*x - 240


def test_nircam_configuration_and_offline_provenance(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('NIRCam profile attempted network access')
    monkeypatch.setattr(socket, 'socket', no_network)
    assert tuple(m for m in list_instrument_profiles() if m.startswith('jwst/nircam/')) == tuple(
        sorted(PREFIX+f for f in FILTERS))
    official = {'f277w': [2.395, 3.179], 'f322w2': [2.413, 4.083],
                'f356w': [3.100, 4.041], 'f444w': [3.835, 5.084]}
    records = []
    for name, bounds in FILTERS.items():
        p = get_profile((PREFIX+name).upper())
        assert p.mode == PREFIX+name
        assert get_profile('jwst/nircam/grismr/'+name).mode == p.mode
        assert p.quality == 'nominal'
        assert all(s in p.configuration for s in ('Module A', 'Grism R', name.upper(), 'first order'))
        record = p.provenance
        records.append(record)
        assert record['filter_5_percent_limits_um'][name] == official[name]
        assert record['tso_f444w_detector_red_limit_um'] == 5.03
        np.testing.assert_array_equal(p.wavelength[[0, -1]].to_value(u.um), [2.4, 5.0])
        for field in ('institution', 'source_kind', 'source_definition', 'equation',
                      'processing', 'reproduction_script', 'redistribution_basis',
                      'calibration_identifier', 'boundary_assumption'):
            assert record[field]
        assert p.coverage_source_url in (record['coverage_source_url'], record['tso_source_url'])
        assert p.source_url == record['publication_url']
        assert record['output_sha256'] == hashlib.sha256(
            ROOT.joinpath(record['output_file']).read_bytes()).hexdigest()
        np.testing.assert_array_equal(p.valid_intervals.to_value(u.um), [bounds])
        assert p.wavelength[0] < p.valid_intervals.min() < p.valid_intervals.max() < p.wavelength[-1]
    assert all(r == records[0] for r in records)  # One shared prescription.
    records[0]['coefficients_ascending'].clear()
    assert get_profile(PREFIX+'f277w').provenance['coefficients_ascending']
    for mode in ('jwst/nircam/grismc/f444w', 'jwst/nircam/tsgrism/moduleb/grismr/f444w',
                 'jwst/nircam/wfss/modulea/grismr/f444w', PREFIX+'f250m'):
        with pytest.raises(ValueError, match='Unknown instrument profile'):
            get_profile(mode)


def test_nircam_polynomial_and_interpolation():
    p = get_profile(PREFIX+'f322w2')
    np.testing.assert_allclose(p.resolving_power, polynomial(p.wavelength.value), rtol=4e-15)
    # Dense off-knot comparison; reproducing the polynomial is distinct from
    # its few-percent fit to the optical model or the numerical convolution.
    x = np.linspace(*p.provenance['calibration_domain_um'], 100003)
    residual = p.resolving_power_at(x*u.um)/polynomial(x)-1
    assert np.max(np.abs(residual)) < 4e-8
    assert p.resolving_power_at(4*u.um) == pytest.approx(1608., rel=1e-12)
    for x in (p.wavelength.value[0]-1e-4, p.wavelength.value[-1]+1e-4):
        with pytest.raises(ValueError, match='extrapolation'):
            p.resolving_power_at(x*u.um)


def test_nircam_curve_reproduces_offline():
    script = Path(__file__).parents[1]/'scripts/instruments/convert_nircam.py'
    spec = importlib.util.spec_from_file_location('convert_nircam', script)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    record = get_profile(PREFIX+'f444w').provenance
    assert converter.csv_bytes(converter.convert(record)) == ROOT.joinpath(record['output_file']).read_bytes()


@pytest.fixture(scope='module')
def source():
    wave = np.linspace(2.37, 5.12, 55001)
    flux = np.ones(wave.size)
    for center in (2.7, 3.3, 3.6, 4.5):
        flux += 100*np.exp(-.5*((wave-center)/.00003)**2)
    return Spectrum(spectral_axis=wave*u.um, flux=flux*FLUX_UNIT,
                    meta={'nested': {'values': [1]}})


def test_nircam_broadening_all_filters(source):
    for name, center in zip(FILTERS, (2.7, 3.3, 3.6, 4.5)):
        result = source.observed_by(PREFIX+name)
        x, y = result.wavelength.to_value(u.um), result.flux.value-1
        local = np.abs(x-center) < .01
        x, y = x[local], y[local]
        peak = y.argmax()
        lower = np.interp(y[peak]/2, y[:peak+1], x[:peak+1])
        upper = np.interp(y[peak]/2, y[peak:][::-1], x[peak:][::-1])
        # Intrinsic line width and half-height sampling add small broadening.
        assert center/(upper-lower) == pytest.approx(polynomial(center), rel=.007)


def test_nircam_padding_and_outside_wings():
    for name, (lo, hi) in FILTERS.items():
        p = get_profile(PREFIX+name)
        wave = np.linspace(lo-.02, hi+.02, 20001)
        flux = np.zeros(wave.size)
        for edge, sign in ((lo, -1), (hi, 1)):
            center = edge+sign*.2*edge/polynomial(edge)
            flux += np.exp(-.5*((wave-center)/.000025)**2)
        source = Spectrum(spectral_axis=wave*u.um, flux=flux*FLUX_UNIT)
        observed = source.observed_by(p)
        # The input lines lie outside output coverage; wings remain at BOTH edges.
        assert min(observed.flux.value[0], observed.flux.value[-1]) > .005
        for lower, upper in ((lo, hi+.02), (lo-.02, hi)):
            with pytest.raises(ValueError, match='padding'):
                source.select_wavelength(lower*u.um, upper*u.um).observed_by(p)
        # Calibration covers the output but cannot support convolution.
        short = replace(p, wavelength=np.array([lo, hi])*u.um,
                        resolving_power=polynomial(np.array([lo, hi])))
        with pytest.raises(ValueError, match='Calibration.*padding'):
            source.observed_by(short)
        if name in ('f277w', 'f444w'):
            # Merely clipping output to the published model edges is unsafe.
            edge_bounds = [2.4, hi] if name == 'f277w' else [lo, 5.0]
            with pytest.raises(ValueError, match='Calibration.*padding'):
                source.observed_by(replace(p, valid_intervals=[edge_bounds]*u.um))


def test_nircam_spectrum_grid_sampling_and_ownership(source):
    original = source.flux.copy()
    grid = make_grid(source)
    original_grid = grid.data.copy()
    # Each sampling route uses a different filter; the fourth is covered above.
    for name, options in (
        ('f277w', {}), ('f322w2', {'wavelength': np.linspace(2.5, 4., 121)*u.um}),
        ('f444w', {'samples_per_resolution_element': 5}),
    ):
        p = get_profile(PREFIX+name)
        spectrum = source.observed_by(p, **options)
        observed = grid.observed_by(p, **options)
        assert type(spectrum) is Spectrum and type(observed) is SpectralGrid
        np.testing.assert_allclose(observed.data[0], spectrum.flux.value, rtol=1e-12)
        np.testing.assert_array_equal(spectrum.valid_intervals, p.valid_intervals)
        np.testing.assert_array_equal(observed.valid_intervals, p.valid_intervals)
        assert spectrum.meta['instrument_profile']['mode'] == p.mode
        if not options:
            lo, hi = p.valid_intervals[0].to_value(u.um)
            # Include exact decimal edges despite um/Angstrom conversion roundoff.
            wave_um = np.round(source.wavelength.to_value(u.um), 12)
            np.testing.assert_array_equal(spectrum.wavelength, source.wavelength[
                (wave_um >= lo) & (wave_um <= hi)])
        elif 'wavelength' in options:
            np.testing.assert_array_equal(spectrum.wavelength, options['wavelength'])
        else:
            x = spectrum.wavelength.to_value(u.um)
            np.testing.assert_allclose(np.diff(x), x[:-1]/(5*p.resolving_power_at(x[:-1]*u.um)), rtol=1e-11)
        for interpolate in (False, True):
            extracted = observed.get_spectrum(3000., 4., 0., interpolate=interpolate)
            flux = observed.get_flux(3000., 4., 0., interpolate=interpolate)
            assert type(extracted) is Spectrum and isinstance(flux, u.Quantity)
            np.testing.assert_array_equal(extracted.flux, flux)
            np.testing.assert_array_equal(extracted.valid_intervals, p.valid_intervals)
            saved = observed.data.copy()
            extracted.flux[0] = 999*FLUX_UNIT
            extracted.meta['nested']['values'].append(3)
            np.testing.assert_array_equal(observed.data, saved)
            assert observed.meta['nested']['values'] == [1, 2]
        spectrum.flux[0] = 999*FLUX_UNIT
        spectrum.meta['nested']['values'].append(3)
    np.testing.assert_array_equal(source.flux, original)
    np.testing.assert_array_equal(grid.data, original_grid)
    assert source.meta == {'nested': {'values': [1]}}
    assert grid.meta == {'nested': {'values': [1, 2]}}
