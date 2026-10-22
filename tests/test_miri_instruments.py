"""Scientific contracts for nominal MIRI P750L/SLITLESSPRISM matching."""
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

MODE = 'jwst/miri/lrs/slitless/slitlessprism'
ROOT = files('speclib.data').joinpath('instruments')


def test_configuration_provenance_and_integrity_offline(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('MIRI lookup attempted network access')
    monkeypatch.setattr(socket, 'socket', no_network)
    p = get_profile('JWST/MIRI/LRS/SLITLESS')
    assert p.mode == MODE
    assert tuple(m for m in list_instrument_profiles() if m.startswith('jwst/miri/')) == (MODE,)
    assert p.quality == 'nominal'
    assert 'P750L' in p.configuration and 'SLITLESSPRISM' in p.configuration
    np.testing.assert_array_equal(p.valid_intervals.to_value(u.um), [[5.5, 12.]])
    np.testing.assert_array_equal(p.wavelength[[0, -1]].to_value(u.um), [5., 14.])
    record = p.provenance
    for field in ('institution', 'source_kind', 'source_definition', 'equation',
                  'processing', 'reproduction_script', 'redistribution_basis',
                  'calibration_identifier', 'boundary_assumption', 'domain_basis',
                  'calibration_file_review', 'subarray_assumption'):
        assert record[field]
    assert record['retrieved'] == '2026-10-22'
    assert p.source_url == record['documentation_url']
    assert p.coverage_source_url == record['coverage_source_url']
    assert record['ground_measurement']['resolving_power'] == 95.3
    assert record['output_sha256'] == hashlib.sha256(
        ROOT.joinpath(record['output_file']).read_bytes()).hexdigest()
    record['coefficients'].clear()
    assert p.provenance['coefficients']
    for mode in ('jwst/miri/lrs/slit', 'jwst/miri/lrs/fixedslit',
                 'jwst/miri/mrs', 'jwst/miri/wfss',
                 'jwst/miri/lrs/slitless/slitlessprism_ip',
                 'jwst/miri/lrs/slitless/slitlessprism_ips'):
        with pytest.raises(ValueError, match='Unknown instrument profile'):
            get_profile(mode)


def test_nominal_design_values_interpolation_and_reproduction():
    p = get_profile(MODE)
    # Independent STScI values, NOT an in-flight Gaussian calibration claim.
    np.testing.assert_allclose(p.resolving_power_at([5., 7.5, 10.]*u.um), [40., 100., 160.])
    wave = np.linspace(5., 14., 10003)
    np.testing.assert_allclose(p.resolving_power_at(wave*u.um), 24*wave-80, rtol=8e-16)
    assert np.all(np.isfinite(p.resolving_power)) and np.all(p.resolving_power > 0)
    # The design approximation is explicitly continued to 14 um (R=256).
    assert p.resolving_power_at(14*u.um) == 256
    for x in (4.999, 14.001):
        with pytest.raises(ValueError, match='extrapolation'):
            p.resolving_power_at(x*u.um)
    script = Path(__file__).parents[1]/'scripts/instruments/convert_miri.py'
    spec = importlib.util.spec_from_file_location('convert_miri', script)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    record = p.provenance
    assert converter.csv_bytes(converter.convert(record)) == ROOT.joinpath(record['output_file']).read_bytes()


@pytest.fixture(scope='module')
def source():
    wave = np.linspace(5., 14., 45001)
    flux = np.ones(wave.size)
    for center in (5.7, 7.5, 11.8):
        flux += 100*np.exp(-.5*((wave-center)/.0002)**2)
    return Spectrum(spectral_axis=wave*u.um, flux=flux*FLUX_UNIT,
                    meta={'nested': {'values': [1]}})


def test_representative_broad_lines(source):
    result = source.observed_by(MODE)
    x, y = result.wavelength.to_value(u.um), result.flux.value-1
    for center in (5.7, 7.5, 11.8):
        local = np.abs(x-center) < .15
        xx, yy = x[local], y[local]
        peak = yy.argmax()
        lower = np.interp(yy[peak]/2, yy[:peak+1], xx[:peak+1])
        upper = np.interp(yy[peak]/2, yy[peak:][::-1], xx[peak:][::-1])
        # Intrinsic width, log-Gaussian convention and numerical remapping are
        # separate from the accuracy of the nominal instrumental prescription.
        assert center/(upper-lower) == pytest.approx(24*center-80, rel=.007)


def test_boundary_wings_padding_and_no_calibration_extension():
    p = get_profile(MODE)
    wave = np.linspace(5., 14., 45001)
    centers = (5.5-.2*5.5/52, 12+.2*12/208)
    flux = sum(np.exp(-.5*((wave-c)/.0003)**2) for c in centers)
    source = Spectrum(spectral_axis=wave*u.um, flux=flux*FLUX_UNIT)
    result = source.observed_by(p)
    # Both input line centers are OUTSIDE the final output interval.
    assert min(result.flux.value[0], result.flux.value[-1]) > .003
    for lo, hi in ((5.5, 14.), (5., 12.)):
        with pytest.raises(ValueError, match='padding'):
            source.select_wavelength(lo*u.um, hi*u.um).observed_by(p)
    short = replace(p, wavelength=[5.5, 12.]*u.um, resolving_power=[52., 208.])
    with pytest.raises(ValueError, match='Calibration.*padding'):
        source.observed_by(short)
    # Optical coverage endpoints cannot be output endpoints without extra R support.
    with pytest.raises(ValueError, match='Calibration.*padding'):
        source.observed_by(replace(p, valid_intervals=[[5., 14.]]*u.um))
    with pytest.raises(ValueError, match='coverage'):
        source.observed_by(p, wavelength=[5.3, 5.4]*u.um)
    with pytest.raises(ValueError, match='Implicit.*edges'):
        source.observed_by(p, wavelength=[5.5, 5.6]*u.um)


def test_sampling_grid_agreement_and_ownership(source):
    saved = source.flux.copy()
    grid = make_grid(source)
    grid_saved = grid.data.copy()
    p = get_profile(MODE)
    for options in ({}, {'wavelength': np.linspace(5.6, 11.9, 121)*u.um},
                    {'samples_per_resolution_element': 5}):
        result = source.observed_by(p, **options)
        observed = grid.observed_by(p, **options)
        assert type(result) is Spectrum and type(observed) is SpectralGrid
        np.testing.assert_allclose(observed.data[0], result.flux.value, rtol=1e-12)
        np.testing.assert_array_equal(result.valid_intervals, p.valid_intervals)
        np.testing.assert_array_equal(observed.valid_intervals, p.valid_intervals)
        assert result.meta['instrument_profile']['mode'] == MODE
        if not options:
            wave = np.round(source.wavelength.to_value(u.um), 12)
            np.testing.assert_array_equal(result.wavelength, source.wavelength[(wave >= 5.5) & (wave <= 12.)])
        elif 'wavelength' in options:
            np.testing.assert_array_equal(result.wavelength, options['wavelength'])
        else:
            x = result.wavelength.to_value(u.um)
            np.testing.assert_allclose(np.diff(x), x[:-1]/(5*(24*x[:-1]-80)), rtol=1e-11)
        for interpolate in (False, True):
            observed_saved = observed.data.copy()
            extracted = observed.get_spectrum(3000., 4., 0., interpolate=interpolate)
            flux = observed.get_flux(3000., 4., 0., interpolate=interpolate)
            assert type(extracted) is Spectrum and isinstance(flux, u.Quantity)
            np.testing.assert_array_equal(extracted.flux, flux)
            np.testing.assert_array_equal(extracted.valid_intervals, p.valid_intervals)
            extracted.flux[0] = 999*FLUX_UNIT
            extracted.meta['nested']['values'].append(3)
            np.testing.assert_array_equal(observed.data, observed_saved)
            assert observed.meta['nested']['values'] == [1, 2]
        result.meta['nested']['values'].append(3)
        result.flux[0] = 999*FLUX_UNIT
    np.testing.assert_array_equal(source.flux, saved)
    np.testing.assert_array_equal(grid.data, grid_saved)
    assert source.meta == {'nested': {'values': [1]}}
