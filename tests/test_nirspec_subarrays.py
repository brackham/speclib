"""Regular BOTS subarrays: coverage contracts, without repeating R validation."""
from dataclasses import replace
import hashlib
from importlib.resources import files
import socket

import astropy.units as u
import numpy as np
import pytest

from speclib import Spectrum, get_profile, list_instrument_profiles
from test_instruments import FLUX_UNIT, make_grid

PREFIX = 'jwst/nirspec/bots/s1600a1/'
# Independent transcription of STScI BOTS wavelength ranges and gaps, Table 1.
# Columns: SUB1024A, SUB1024B, SUB512 (also SUB512S); micrometers.
MATRIX = {
    'g140m/f070lp': ('0.70-1.22', '1.22-1.27', '1.22-1.27'),
    'g140h/f070lp': ('0.82-1.07', '1.07-1.27', '1.07-1.19'),
    'g140m/f100lp': ('0.97-1.22', '1.22-1.87', '1.22-1.55'),
    'g140h/f100lp': ('0.97-1.07,1.59-1.83', '1.07-1.31,1.35-1.59', '1.07-1.19,1.47-1.59'),
    'g235m/f170lp': ('1.66-2.03', '2.03-3.12', '2.03-2.58'),
    'g235h/f170lp': ('1.66-1.79,2.67-3.07', '1.79-2.20,2.27-2.67', '1.79-2.00,2.47-2.67'),
    'g395m/f290lp': ('2.87-3.35', '3.35-5.18', '3.35-4.27'),
    'g395h/f290lp': ('2.87-3.03,4.51-5.18', '3.03-3.72,3.82-4.51', '3.03-3.38,4.17-4.51'),
    'prism/clear': (None, '0.60-5.30', '0.60-5.30'),
}


def test_complete_coverage_matrix_and_shared_calibration_offline(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('Subarray profile attempted network access')
    monkeypatch.setattr(socket, 'socket', no_network)
    expected_modes = set()
    for suffix, columns in MATRIX.items():
        original = get_profile('jwst/nirspec/' + suffix)
        for subarray, cell in zip(('sub1024a', 'sub1024b', 'sub512', 'sub512s'), (*columns, columns[2])):
            if cell is None:
                continue
            mode = PREFIX + subarray + '/' + suffix
            expected_modes.add(mode)
            p = get_profile(mode.upper())
            assert p.mode == mode and p.configuration == 'BOTS/S1600A1/' + subarray.upper()
            intervals = [[float(x) for x in pair.split('-')] for pair in cell.split(',')]
            np.testing.assert_array_equal(p.valid_intervals.to_value(u.um), intervals, err_msg=mode)
            np.testing.assert_array_equal(p.wavelength, original.wavelength)
            np.testing.assert_array_equal(p.resolving_power, original.resolving_power)
            assert p.provenance_ref == original.provenance_ref
            assert p.coverage_source_url == original.coverage_source_url
            assert 'selected-subarray' in ' '.join(p.assumptions)
            record = p.provenance
            assert record['coverage_source_url'] == p.coverage_source_url
            assert record['coverage_retrieved'] == '2026-10-22'
            assert record['coverage_table'].startswith('Table 1')
            assert record['subarray_source_url'] and record['subarray_resolution_assumption']
            geometry = record['subarrays'][subarray.upper()]
            assert geometry['height_pixels'] == (16 if subarray == 'sub512s' else 32)
            assert record['nominal_detector_gap_um'] == original.provenance['nominal_detector_gap_um']
            asset = files('speclib.data').joinpath('instruments', record['output_file'])
            assert hashlib.sha256(asset.read_bytes()).hexdigest() == record['output_sha256']
    assert len(expected_modes) == 35
    assert {m for m in list_instrument_profiles() if m.startswith(PREFIX)} == expected_modes


def test_unavailable_combinations_and_sub2048_compatibility():
    for suffix in MATRIX:
        short = get_profile('jwst/nirspec/' + suffix)
        explicit = get_profile(PREFIX + 'sub2048/' + suffix)
        assert explicit.mode == short.mode == 'jwst/nirspec/' + suffix
        assert explicit.configuration == 'BOTS/S1600A1/SUB2048'
        np.testing.assert_array_equal(explicit.valid_intervals, short.valid_intervals)
    for mode in ('sub1024a/prism/clear', 'sub1024/g395h/f290lp',
                 'sub512/g395h/f170lp', 'sub256m2_prism/prism/clear'):
        with pytest.raises(ValueError, match='Unknown instrument profile'):
            get_profile(PREFIX + mode)


@pytest.fixture(scope='module')
def source():
    wave = np.linspace(2.8, 5.25, 24501)
    return Spectrum(spectral_axis=wave*u.um, flux=np.ones(wave.size)*FLUX_UNIT,
                    meta={'nested': {'values': [1]}})


def test_subarray_omissions_and_physical_detector_gap(source):
    for subarray, gap in (('sub1024a', (3.03, 4.51)), ('sub1024b', (3.72, 3.82)),
                          ('sub512', (3.38, 4.17))):
        p = get_profile(PREFIX + subarray + '/g395h/f290lp')
        result = source.observed_by(p)
        x = result.wavelength.to_value(u.um)
        assert not np.any((x > gap[0]) & (x < gap[1]))
        np.testing.assert_array_equal(result.valid_intervals, p.valid_intervals)
        # The physical gap is narrower than the unread region for A and 512.
        assert p.provenance['nominal_detector_gap_um'] == [[3.72, 3.82]]
        with pytest.raises(ValueError, match='coverage|gap'):
            source.observed_by(p, wavelength=np.linspace(*gap, 5)*u.um)
        with pytest.raises(ValueError, match='gap|interval'):
            result.resample(np.linspace(3., 5., 50)*u.um)


def test_boundary_wings_real_padding_and_calibration_support(source):
    p = get_profile(PREFIX + 'sub1024a/g395h/f290lp')
    wave = source.wavelength.to_value(u.um)
    # Line centers lie in the unread middle region, just outside both intervals.
    centers = [3.03 + .2*3.03/p.resolving_power_at(3.03*u.um),
               4.51 - .2*4.51/p.resolving_power_at(4.51*u.um)]
    flux = sum(np.exp(-.5*((wave-c)/.00005)**2) for c in centers)
    lines = Spectrum(spectral_axis=source.wavelength, flux=flux*FLUX_UNIT)
    result = lines.observed_by(p)
    for edge in (3.03, 4.51):
        index = np.argmin(abs(result.wavelength.to_value(u.um)-edge))
        assert result.flux.value[index] > .01
    for lo, hi in ((2.87, 5.25), (2.8, 5.18)):
        with pytest.raises(ValueError, match='padding'):
            source.select_wavelength(lo*u.um, hi*u.um).observed_by(p)
    # Registry boundaries stay inside existing calibration padding; no extension.
    cropped = replace(p, wavelength=[2.87, 5.18]*u.um,
                      resolving_power=p.resolving_power_at([2.87, 5.18]*u.um))
    with pytest.raises(ValueError, match='Calibration.*padding'):
        source.observed_by(cropped)


def test_sampling_grid_metadata_and_ownership(source):
    mode = PREFIX + 'sub512s/g395h/f290lp'
    p = get_profile(mode)
    grid = make_grid(source)
    saved, saved_grid = source.flux.copy(), grid.data.copy()
    explicit = np.r_[np.linspace(3.05, 3.36, 101), np.linspace(4.19, 4.49, 101)]*u.um
    for options in ({}, {'wavelength': explicit}, {'samples_per_resolution_element': 5}):
        observed = source.observed_by(mode, **options)
        result = grid.observed_by(mode, **options)
        np.testing.assert_allclose(result.data[0], observed.flux.value, rtol=1e-12)
        np.testing.assert_allclose(result.wavelength.to_value(u.um),
                                   observed.wavelength.to_value(u.um), rtol=1e-14)
        np.testing.assert_array_equal(observed.valid_intervals, p.valid_intervals)
        assert observed.meta['instrument_profile']['mode'] == mode
        assert observed.meta['instrument_profile']['configuration'] == p.configuration
        assert observed.meta['instrument_profile']['valid_intervals_um'] == p.valid_intervals.value.tolist()
        if not options:
            x = source.wavelength.to_value(u.um)
            mask = ((x >= 3.03) & (x <= 3.38)) | ((x >= 4.17) & (x <= 4.51))
            np.testing.assert_allclose(observed.wavelength, source.wavelength[mask], rtol=1e-14)
        elif 'wavelength' in options:
            np.testing.assert_allclose(observed.wavelength.to_value(u.um), explicit.value, rtol=1e-14)
        else:
            for lo, hi in p.valid_intervals.to_value(u.um):
                x = observed.wavelength.to_value(u.um)
                x = x[(x >= lo) & (x <= hi)]
                np.testing.assert_allclose(np.diff(x), x[:-1]/(5*p.resolving_power_at(x[:-1]*u.um)), rtol=1e-10)
        extracted = result.get_spectrum(3000., 4., 0., interpolate=False)
        assert type(extracted) is Spectrum
        assert isinstance(result.get_flux(3000., 4., 0., interpolate=False), u.Quantity)
        extracted.flux[0] = 999*FLUX_UNIT
        extracted.meta['nested']['values'].append(3)
        np.testing.assert_array_equal(result.data[0], observed.flux.value)
        assert result.meta['nested']['values'] == [1, 2]
        observed.flux[0] = 999*FLUX_UNIT
        observed.meta['nested']['values'].append(3)
    np.testing.assert_array_equal(source.flux, saved)
    np.testing.assert_array_equal(grid.data, saved_grid)
    assert source.meta['nested']['values'] == [1]
