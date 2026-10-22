"""Independent physical anchors and reproducibility for Phase 2 calibrations."""
import hashlib
from importlib.resources import files
import importlib.util
import json
import os
from pathlib import Path
import socket

import astropy.units as u
import numpy as np
import pytest

from speclib import get_profile, list_instrument_profiles

ROOT = files('speclib.data').joinpath('instruments')
MODES = {
    'hst/stis/g430l': ([[.29, .57]], 'derived', '52X2'),
    'hst/stis/g750l': ([[.524, 1.027]], 'derived', '52X2'),
    'hst/wfc3/g102': ([[.8, 1.15]], 'nominal', 'first order'),
    'hst/wfc3/g141': ([[1.075, 1.7]], 'nominal', 'first order'),
    'jwst/niriss/soss/order1': ([[.9, 2.8]], 'nominal', 'SUBSTRIP256'),
    'jwst/niriss/soss/order2': ([[.6, 1.]], 'nominal', 'SUBSTRIP256'),
}


@pytest.fixture(scope='module')
def converter():
    script = Path(__file__).parents[1] / 'scripts/instruments/convert_phase2.py'
    spec = importlib.util.spec_from_file_location('convert_phase2', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_profiles_offline_and_provenance(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('Bundled profile attempted network access')
    monkeypatch.setattr(socket, 'socket', no_network)
    assert tuple(m for m in list_instrument_profiles()
                 if m.startswith(('hst/', 'jwst/niriss/'))) == tuple(sorted(MODES))
    for mode, (coverage, quality, configuration) in MODES.items():
        p = get_profile(mode.upper())
        assert p.mode == mode and p.quality == quality
        assert configuration in p.configuration
        assert 'NIRSpec' not in p.source and 'BOTS' not in p.configuration
        np.testing.assert_array_equal(p.valid_intervals.to_value(u.um), coverage)
        assert p.wavelength[0] < p.valid_intervals.min()
        assert p.wavelength[-1] > p.valid_intervals.max()
        record = p.provenance
        for key in ('institution', 'source_kind', 'source_definition', 'processing',
                    'calibration_identifier', 'equation', 'reproduction_script',
                    'redistribution_basis', 'redistribution_reviewed'):
            assert record[key]
        assert record['retrieved'] == '2026-10-20'
        assert record['output_sha256'] == hashlib.sha256(
            ROOT.joinpath(record['output_file']).read_bytes()).hexdigest()
        if 'lsf_sources' in record:
            assert all(s['url'].endswith(s['filename']) for s in record['lsf_sources'])
            record['lsf_sources'][0]['sha256'] = 'changed'
            assert p.provenance['lsf_sources'][0]['sha256'] != 'changed'
        else:
            record['coefficients'].clear()
            assert p.provenance['coefficients']
        assert p.coverage_source_url.startswith('https://')
        assert np.ptp(p.resolving_power) > 0


@pytest.mark.parametrize('alias,mode', [
    ('hst/stis/ccd/52x2/g430l', 'hst/stis/g430l'),
    ('hst/stis/ccd/52x2/g750l', 'hst/stis/g750l'),
    ('jwst/niriss/soss/substrip256/order1', 'jwst/niriss/soss/order1'),
    ('jwst/niriss/soss/substrip256/order2', 'jwst/niriss/soss/order2'),
])
def test_phase2_explicit_aliases(alias, mode):
    assert get_profile(alias.upper()).mode == mode


@pytest.mark.parametrize('mode', [
    'hst/stis/ccd/52x0.2/g430l', 'hst/stis/g750l/8975',
    'hst/wfc3/g280', 'jwst/niriss/soss/substrip96/order1',
    'jwst/niriss/soss/order3',
])
def test_phase2_unsupported_configurations(mode):
    with pytest.raises(ValueError, match='Unknown instrument profile'):
        get_profile(mode)


@pytest.mark.parametrize('mode,wavelength,expected_r', [
    # Direct half-height crossings inspected in each pinned 52x2.0 text column,
    # converted with the separate grating handbook's average dispersion.
    ('hst/stis/g430l', .32, .32/(1e-4*2.73*1.3864622145641883)),
    ('hst/stis/g430l', .55, .55/(1e-4*2.73*1.5319230200945628)),
    ('hst/stis/g750l', .7, .7/(1e-4*4.92*1.659052874902875)),
    ('hst/stis/g750l', 1., 1./(1e-4*4.92*2.)),
    # WFC3 Table 8.1 reference R, and SOSS Table 2 measured dispersions.
    ('hst/wfc3/g102', 1., 210.), ('hst/wfc3/g141', 1.4, 130.),
    ('jwst/niriss/soss/order1', 1.25, 651.0416666666667),
    ('jwst/niriss/soss/order2', .8, 856.5310492505353),
])
def test_published_and_derived_reference_values(mode, wavelength, expected_r):
    assert get_profile(mode).resolving_power_at(wavelength*u.um) == pytest.approx(expected_r, rel=1e-12)


@pytest.mark.parametrize('mode', MODES)
def test_phase2_interpolation_matches_prescription(mode):
    p = get_profile(mode)
    record = p.provenance
    wave = np.linspace(*record['calibration_domain_um'], 1001)
    if 'lsf_sources' in record:
        anchors = sorted(record['derived_lsf_anchors_um_pixels']
                         + record['handbook_anchors_um_pixels'])
        aw, fw = np.array(anchors).T
        fwhm = np.interp(wave, aw, fw)
        expected = wave/(1e-4*record['dispersion_angstrom_per_pixel']*fwhm)
        # Sampling R rather than FWHM introduces a small interpolation error.
        tolerance = 2e-6
    elif mode.startswith('hst/wfc3'):
        c = record['coefficients']
        expected = wave*c['resolving_power']/c['reference_wavelength_um']
        tolerance = 1e-14
    else:
        c = record['coefficients']
        expected = 1000*wave/(c['resolution_pixels']*c['dispersion_nm_per_pixel'])
        tolerance = 1e-14
    np.testing.assert_allclose(p.resolving_power_at(wave*u.um), expected, rtol=tolerance)


def test_soss_orders_have_distinct_resolution_and_practical_coverage():
    first = get_profile('jwst/niriss/soss/order1')
    second = get_profile('jwst/niriss/soss/order2')
    assert second.resolving_power_at(.95*u.um) > 2*first.resolving_power_at(.95*u.um)
    assert second.valid_intervals.max() == 1*u.um
    assert second.provenance['geometric_coverage_um'][1] > 1
    # Albert et al. Section 5.3: R=650 +/-10 at 1.25 um (two-pixel convention).
    assert abs(first.resolving_power_at(1.25*u.um)-650) < 10


def test_half_maximum_width_definition(converter):
    # Asymmetric triangle: half-height crossings at -1 and +0.5.
    assert converter.half_maximum_width([-2, 0, 1], [0, 2, 0]) == 1.5
    assert converter.half_maximum_width([8, 10, 11], [0, 200, 0]) == 1.5


@pytest.mark.parametrize('x,y', [
    ([0, 0, 1], [0, 1, 0]), ([0, 1, 2], [1, 1, 1]),
    ([0, 1, 2], [0, 0, 0]), ([0, 1, 2], [0, -1, 0]),
    ([0, 1, 2], [0, np.nan, 0]), ([0, 1], [0, 1]),
])
def test_invalid_half_maximum_input(converter, x, y):
    with pytest.raises(ValueError):
        converter.half_maximum_width(x, y)


def test_lsf_aperture_selection_and_pins(converter):
    source = (b'STIS Model Line Spread Function vs Aperture at 320 nm\n'
              b'Rel pixel 52x0.1 52x0.2 52x0.5 52x2.0\n'
              b'-2 4 4 4 0\n0 4 4 4 2\n1 4 4 4 0\n')
    record = {'sha256': hashlib.sha256(source).hexdigest(), 'rows': 3, 'wavelength_um': .32}
    assert converter.lsf_width(source, record) == 1.5
    with pytest.raises(ValueError, match='checksum'):
        converter.lsf_width(source+b'changed', record)
    with pytest.raises(ValueError, match='header'):
        converter.lsf_width(source, record | {'wavelength_um': .55})
    with pytest.raises(ValueError, match='dimensions'):
        converter.lsf_width(source, record | {'rows': 4})


def test_analytic_outputs_reproduce_offline(converter, tmp_path):
    manifest = json.loads(ROOT.joinpath('phase2_sources.json').read_text())
    for record in manifest['sources'].values():
        if 'lsf_sources' not in record:
            content = converter.csv_bytes(converter.convert(record, tmp_path))
            assert content == ROOT.joinpath(record['output_file']).read_bytes()


@pytest.mark.skipif(not os.environ.get('SPECLIB_STIS_SOURCE_DIR'), reason='Optional offline official STIS LSF comparison')
def test_stis_outputs_reproduce_from_pinned_sources(converter):
    source_dir = Path(os.environ['SPECLIB_STIS_SOURCE_DIR'])
    manifest = json.loads(ROOT.joinpath('phase2_sources.json').read_text())
    for record in manifest['sources'].values():
        if 'lsf_sources' not in record:
            continue
        for source in record['lsf_sources']:
            content = (source_dir/source['filename']).read_bytes()
            width = converter.lsf_width(content, source)
            reference = dict(record['handbook_comparison_um_pixels'])[source['wavelength_um']]
            assert abs(width-reference) < .1  # Rounded handbook/model consistency, not physical accuracy.
            with pytest.raises(ValueError, match='checksum'):
                converter.lsf_width(content+b'changed', source)
        output = converter.csv_bytes(converter.convert(record, source_dir))
        assert output == ROOT.joinpath(record['output_file']).read_bytes()
