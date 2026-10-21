"""Offline integrity and provenance of the bundled NIRSpec calibration assets."""
from dataclasses import FrozenInstanceError
from importlib.resources import files
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest

from speclib import get_profile, list_instrument_profiles

ROOT = files('speclib.data').joinpath('instruments')
COVERAGE = {
    'prism/clear': [[.6, 5.3]],
    'g140m/f070lp': [[.7, 1.27]],
    'g140m/f100lp': [[.97, 1.87]],
    'g140h/f070lp': [[.82, 1.27]],
    'g140h/f100lp': [[.97, 1.31], [1.35, 1.83]],
    'g235m/f170lp': [[1.66, 3.12]],
    'g235h/f170lp': [[1.66, 2.20], [2.27, 3.07]],
    'g395m/f290lp': [[2.87, 5.18]],
    'g395h/f290lp': [[2.87, 3.72], [3.82, 5.18]],
}
# Mid-table values independently inspected in the official source FITS tables.
REFERENCES = {
    'prism': (3.25, 118.4612045288086),
    'g140m': (1.25, 891.669677734375),
    'g140h': (1.2000000476837158, 2293.0419921875),
    'g235m': (2.5, 1064.858154296875),
    'g235h': (2.304800033569336, 2646.392578125),
    'g395m': (4.25, 1076.785400390625),
    'g395h': (4.249800205230713, 2925.3623046875),
}


def test_bundled_nirspec_registry_and_calibration_offline(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('Registry attempted network access')
    monkeypatch.setattr(socket, 'socket', no_network)
    expected = tuple(sorted('jwst/nirspec/'+name for name in COVERAGE))
    assert tuple(mode for mode in list_instrument_profiles()
                 if mode.startswith('jwst/nirspec/')) == expected
    for mode in expected:
        p = get_profile(mode.upper())
        assert get_profile(p) is p
        assert p.mode == mode
        assert p.source and p.source_url and p.calibration_version and p.assumptions
        assert p.quality == 'tabulated'
        assert p.configuration == 'BOTS/S1600A1/SUB2048'
        provenance = p.provenance
        assert provenance['output_file'] == mode.split('/')[-2]+'.csv'
        assert provenance['url'].endswith(provenance['filename'])
        assert provenance['source_kind'] and provenance['source_definition']
        assert provenance['processing'] and provenance['publication_url'].startswith('https://doi.org/')
        assert provenance['redistribution_url'].startswith('https://')
        assert provenance['output_sha256'] == hashlib.sha256(
            ROOT.joinpath(provenance['output_file']).read_bytes()
        ).hexdigest()
        provenance['url'] = 'changed'
        assert p.provenance['url'] != 'changed'
        suffix = mode.removeprefix('jwst/nirspec/')
        np.testing.assert_array_equal(p.valid_intervals.value, COVERAGE[suffix], err_msg=mode)
        explicit = get_profile('JWST/NIRSPEC/BOTS/S1600A1/SUB2048/'+suffix)
        np.testing.assert_array_equal(explicit.valid_intervals, p.valid_intervals, err_msg=mode)
        assert np.ptp(p.resolving_power) > 0
    # Dataclass/buffer immutability is common to all profiles, not mode-specific.
    with pytest.raises(FrozenInstanceError):
        p.mode = 'changed'
    with pytest.raises(ValueError):
        p.resolving_power[0] = 1
    with pytest.raises(ValueError):
        p.wavelength.value[0] = 1
    for name, reference in REFERENCES.items():
        with ROOT.joinpath(name+'.csv').open() as stream:
            data = np.loadtxt(stream, delimiter=',')
        np.testing.assert_array_equal(data[len(data)//2], reference, err_msg=name)
    with pytest.raises(ValueError, match='Unknown instrument profile'):
        get_profile('jwst/nirspec/bots/s1600a1/sub512/prism/clear')
    with pytest.raises(TypeError):
        get_profile(123)


@pytest.mark.skipif(not os.environ.get('SPECLIB_NIRSPEC_SOURCE_DIR'), reason='Optional offline official FITS comparison')
def test_conversion_matches_all_official_samples():
    script = Path(__file__).parents[1] / 'scripts/instruments/convert_nirspec.py'
    spec = importlib.util.spec_from_file_location('convert_nirspec', script)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    manifest = json.loads(ROOT.joinpath('sources.json').read_text())
    for name, expected in manifest['sources'].items():
        path = Path(os.environ['SPECLIB_NIRSPEC_SOURCE_DIR']) / expected['url'].rsplit('/', 1)[1]
        official = converter.convert(path.read_bytes(), expected)
        with ROOT.joinpath(name + '.csv').open() as stream:
            bundled = np.loadtxt(stream, delimiter=',')
        np.testing.assert_array_equal(bundled, official)
        with pytest.raises(ValueError, match='checksum'):
            converter.convert(path.read_bytes() + b'changed', expected)
