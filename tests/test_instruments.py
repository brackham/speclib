"""Scientific and offline-data contracts for nominal instrument matching."""

from dataclasses import replace
import copy

import astropy.units as u
from astropy.nddata import StdDevUncertainty
import numpy as np
import pytest
from scipy.interpolate import NearestNDInterpolator

from speclib import (InstrumentProfile, Spectrum, SpectralGrid,
                     get_profile)
from speclib import core


FLUX_UNIT = u.erg / u.s / u.cm**2 / u.AA


def coverage_pieces(spectrum):
    return tuple(spectrum.select_wavelength(a, b) for a, b in spectrum.valid_intervals)


@pytest.fixture
def profile():
    return InstrumentProfile(
        mode='test/variable', wavelength=np.array([.8, 1., 1.3]) * u.um,
        resolving_power=np.array([100., 120., 150.]),
        valid_intervals=np.array([[.9, 1.2]]) * u.um,
        source='Synthetic test calibration', source_url='https://example.org/test',
        calibration_version='test-1', assumptions=('Negligible source width',),
        quality='nominal',
    )


@pytest.fixture
def spectrum():
    wave = np.linspace(.8, 1.3, 6001)
    flux = np.ones(len(wave))
    for center in [.905, 1., 1.19]:
        flux += 100 * np.exp(-.5 * ((wave - center) / .00012)**2)
    return Spectrum(spectral_axis=wave * u.um, flux=flux * FLUX_UNIT,
                    meta={'nested': {'values': [1, 2]}})


def make_grid(spectrum):
    grid = SpectralGrid.__new__(SpectralGrid)
    grid.model_grid = 'phoenix'
    grid.wavelength = spectrum.wavelength.copy()
    grid.unit = spectrum.flux.unit
    grid.points = np.array([[3000., 4., 0.], [3100., 4., 0.]])
    grid.teffs, grid.loggs, grid.metallicities = np.array([3000., 3100.]), np.array([4.]), np.array([0.])
    grid.teff_bds, grid.logg_bds, grid.metallicity_bds = (3000., 3100.), (4., 4.), (0., 0.)
    grid.fluxes = {3000.: {4.: {0.: spectrum.flux.copy()}}, 3100.: {4.: {0.: 2 * spectrum.flux}}}
    grid.data = np.vstack([spectrum.flux.value, 2 * spectrum.flux.value])
    grid.meta = {'nested': {'values': [1, 2]}}
    grid.interpolator = NearestNDInterpolator(grid.points, grid.data)
    return grid


@pytest.mark.parametrize('changes', [
    {'resolving_power': [1., 0., 3.]},
    {'resolving_power': [1., np.nan, 3.]},
    {'resolving_power': [1., 2.]},
    {'wavelength': [1., .9, 1.3] * u.um},
    {'valid_intervals': [[.9, 1.], [.95, 1.2]] * u.um},
    {'valid_intervals': [[.7, 1.2]] * u.um},
    {'source': ''},
])
def test_profile_validation(profile, changes):
    with pytest.raises(ValueError):
        replace(profile, **changes)


def test_profile_interpolation_never_extrapolates(profile):
    assert profile.provenance == {}  # Custom profiles need no bundled record.
    np.testing.assert_allclose(profile.resolving_power_at([900., 1100.] * u.nm), [110., 130.])
    with pytest.raises(ValueError, match='extrapolation'):
        profile.resolving_power_at(.7 * u.um)
    with pytest.raises(TypeError):
        profile.resolving_power_at([1., 1.1])
    with pytest.raises(u.UnitsError):
        profile.resolving_power_at(1 * u.s)


# Exact values independently inspected in the downloaded official FITS tables.


def test_preserves_sampling_and_matches_generic(profile, spectrum):
    original = spectrum.flux.copy()
    observed = spectrum.observed_by(profile)
    assert type(observed) is Spectrum
    intervals = observed.valid_intervals
    intervals[0, 0] = .1*u.um
    np.testing.assert_array_equal(observed.valid_intervals, profile.valid_intervals)
    selected = (spectrum.wavelength >= .9 * u.um) & (spectrum.wavelength <= 1.2 * u.um)
    expected = spectrum.set_variable_resolving_power(profile.wavelength, profile.resolving_power)
    np.testing.assert_array_equal(observed.wavelength, spectrum.wavelength[selected])
    assert observed.spectral_axis.unit == spectrum.spectral_axis.unit
    # Different adaptive convolution domains introduce small remapping differences.
    np.testing.assert_allclose(observed.flux.value, expected.flux.value[selected], rtol=2e-3)
    assert observed.flux.unit == spectrum.flux.unit
    assert observed.meta['instrument_profile']['mode'] == profile.mode
    observed.flux[0] = 999 * FLUX_UNIT
    observed.meta['nested']['values'].append(3)
    np.testing.assert_array_equal(spectrum.flux, original)
    assert spectrum.meta == {'nested': {'values': [1, 2]}}


def test_model_and_calibration_padding_are_required(profile, spectrum):
    with pytest.raises(ValueError, match='padding'):
        spectrum.select_wavelength(.9*u.um, 1.2*u.um).observed_by(profile)
    with pytest.raises(ValueError, match='Calibration.*padding'):
        spectrum.observed_by(replace(profile, wavelength=[.9, 1., 1.2]*u.um))


@pytest.mark.parametrize('descending', [False, True])
def test_explicit_sampling(profile, spectrum, descending):
    target = np.linspace(.92, 1.18, 80) * u.um
    if descending:
        target = target[::-1]
    source = spectrum if not descending else Spectrum(
        spectral_axis=spectrum.spectral_axis[::-1], flux=spectrum.flux[::-1])
    result = source.observed_by(profile, wavelength=target)
    np.testing.assert_array_equal(result.wavelength, target)
    # Compare to the generic method while keeping real source padding.
    full = spectrum.set_variable_resolving_power(profile.wavelength, profile.resolving_power).resample(np.sort(target))
    expected = full.flux.value[::-1] if descending else full.flux.value
    np.testing.assert_allclose(result.flux.value, expected, rtol=.002)


def test_generated_sampling(profile, spectrum):
    result = spectrum.observed_by(profile, samples_per_resolution_element=3)
    wave = result.wavelength.to_value(u.um)
    expected = wave[:-1] / (3 * profile.resolving_power_at(wave[:-1] * u.um))
    np.testing.assert_allclose(np.diff(wave), expected, rtol=1e-12)
    assert wave.min() > .9 and wave.max() < 1.2


@pytest.mark.parametrize('options,error', [
    ({'wavelength': [1., 1.1] * u.um, 'samples_per_resolution_element': 3}, 'mutually exclusive'),
    ({'wavelength': [1., 1.] * u.um}, 'monotonic'),
    ({'wavelength': [1., np.nan] * u.um}, 'finite'),
    ({'wavelength': [.9, 1., 1.2] * u.um}, 'bin edges'),
    ({'samples_per_resolution_element': 0}, 'positive'),
    ({'samples_per_resolution_element': 1e15}, 'memory'),
])
def test_sampling_validation(profile, spectrum, options, error):
    with pytest.raises(ValueError, match=error):
        spectrum.observed_by(profile, **options)


def test_input_sampling_and_state_validation(profile, spectrum):
    coarse = Spectrum(spectral_axis=np.linspace(.8, 1.3, 10) * u.um, flux=np.ones(10) * FLUX_UNIT)
    with pytest.raises(ValueError, match='under-sampled'):
        coarse.observed_by(profile)
    spectrum.mask = np.zeros(len(spectrum.flux), dtype=bool)
    with pytest.raises(NotImplementedError, match='masked'):
        spectrum.observed_by(profile)
    spectrum.mask = None
    spectrum.uncertainty = StdDevUncertainty(np.ones(len(spectrum.flux)))
    with pytest.raises(NotImplementedError, match='uncertainties'):
        spectrum.observed_by(profile)


def test_native_resolution_metadata(profile, spectrum):
    spectrum.meta['native_resolving_power'] = 100
    with pytest.raises(ValueError, match='native resolving power is insufficient'):
        spectrum.observed_by(profile)
    spectrum.meta['native_resolving_power'] = 'approximately 500'
    with pytest.warns(UserWarning, match='native resolving power'):
        spectrum.observed_by(profile)


@pytest.mark.parametrize('sampling', [{}, {'samples_per_resolution_element': 3}])
def test_grid_matches_spectrum_without_mutation(profile, spectrum, sampling):
    grid = make_grid(spectrum)
    grid.wavelength = grid.wavelength.to(u.nm)
    original = grid.data.copy()
    expected = spectrum.observed_by(profile, **sampling)
    result = grid.observed_by(profile, **sampling)
    assert type(result) is SpectralGrid
    np.testing.assert_array_equal(result.valid_intervals, profile.valid_intervals)
    np.testing.assert_allclose(
        result.wavelength.to_value(u.um), expected.wavelength.to_value(u.um),
        rtol=1e-14,
    )
    if not sampling:
        assert result.wavelength.unit == grid.wavelength.unit
    np.testing.assert_allclose(result.data[0], expected.flux.value)
    np.testing.assert_allclose(result.get_flux(3050, 4., 0.).value, 1.5 * expected.flux.value)
    result.fluxes[3000.][4.][0.][0] = 100 * FLUX_UNIT
    result.data[0, 0] = 123
    result.points[0, 0] = 1
    result.meta['nested']['values'].append(3)
    np.testing.assert_array_equal(grid.data, original)
    np.testing.assert_array_equal(grid.fluxes[3000.][4.][0.].value, original[0])
    assert grid.points[0, 0] == 3000
    assert grid.meta['nested']['values'] == [1, 2]


@pytest.fixture(scope='module')
def nirspec_spectrum():
    wave = np.geomspace(.5, 6., 30000) * u.um
    return Spectrum(spectral_axis=wave, flux=np.ones(len(wave)) * FLUX_UNIT)


@pytest.mark.parametrize('mode', ['jwst/nirspec/prism/clear', 'jwst/nirspec/g395h/f290lp'])
def test_continuous_and_gapped_coverage_and_auto_sampling(mode, nirspec_spectrum):
    p = get_profile(mode)
    for options in ({}, {'samples_per_resolution_element': 3}):
        result = nirspec_spectrum.observed_by(mode, **options)
        assert type(result) is Spectrum
        segments = coverage_pieces(result)
        assert len(segments) == len(p.valid_intervals)
        for segment, (lower, upper) in zip(segments, p.valid_intervals):
            assert segment.wavelength.min() >= lower
            assert segment.wavelength.max() <= upper
            assert np.all(np.isfinite(segment.flux))
            if not options:
                selected = (nirspec_spectrum.wavelength >= lower) & (nirspec_spectrum.wavelength <= upper)
                np.testing.assert_array_equal(segment.wavelength, nirspec_spectrum.wavelength[selected])


def test_g395h_gaps_and_grid(nirspec_spectrum, monkeypatch):
    mode = 'jwst/nirspec/g395h/f290lp'
    p = get_profile(mode)
    with pytest.raises(ValueError, match='detector gap'):
        nirspec_spectrum.observed_by(mode, wavelength=[3.6, 3.77, 3.9] * u.um)
    target = np.r_[np.linspace(3., 3.6, 20), np.linspace(4., 5., 20)] * u.um
    expected = nirspec_spectrum.observed_by(mode, wavelength=target)
    grid = make_grid(nirspec_spectrum)
    calls = []
    build = core._build_variable_resolving_power_plan
    def counting(*args):
        calls.append(1)
        return build(*args)
    monkeypatch.setattr(core, '_build_variable_resolving_power_plan', counting)
    results = grid.observed_by(p, wavelength=target)
    assert type(results) is SpectralGrid and len(calls) == 2
    np.testing.assert_array_equal(results.wavelength, expected.spectral_axis)
    np.testing.assert_allclose(results.data[0], expected.flux.value)
    partial = nirspec_spectrum.observed_by(mode, wavelength=np.linspace(4., 5., 10) * u.um)
    assert type(partial) is Spectrum
    np.testing.assert_array_equal(partial.valid_intervals, p.valid_intervals[1:])
    extracted = results.get_spectrum(3050., 4., feh=0.)
    assert type(extracted) is Spectrum
    np.testing.assert_array_equal(extracted.valid_intervals, results.valid_intervals)
    np.testing.assert_array_equal(extracted.spectral_axis, results.wavelength)
    np.testing.assert_array_equal(extracted.flux, results.get_flux(3050., 4., feh=0.))
    assert extracted.meta['instrument_profile'] == results.meta['instrument_profile']
    assert not np.any((extracted.wavelength > 3.72*u.um) & (extracted.wavelength < 3.82*u.um))
    centers = np.r_[np.linspace(3.1, 3.5, 5), np.linspace(4.2, 4.8, 5)]*u.um
    resampled = extracted.resample(centers)
    assert len(resampled.valid_intervals) == 2
    binned = extracted.bin(center=[3.3, 4.5]*u.um, width=[.1, .1]*u.um)
    assert len(binned.valid_intervals) == 2
    extracted.meta['nested']['values'].append(3)
    extracted.meta['instrument_profile']['mode'] = 'changed'
    extracted.meta['valid_intervals'] = [[3., 3.6]]*u.um
    assert results.meta['nested']['values'] == [1, 2]
    assert results.meta['instrument_profile']['mode'] == mode
    np.testing.assert_array_equal(results.valid_intervals, p.valid_intervals)
    extracted = results.get_spectrum(3000., 4., 0., interpolate=False)
    with pytest.raises(ValueError, match='detector gap'):
        extracted.resample([3.76, 3.77]*u.um)
    with pytest.raises(ValueError, match='detector gap'):
        extracted.bin(center=[3.7]*u.um, width=[.1]*u.um)
    extracted.flux[0] = 123*FLUX_UNIT
    extracted.spectral_axis[0] *= 1.001
    np.testing.assert_array_equal(results.data[0], expected.flux.value)
    np.testing.assert_array_equal(results.fluxes[3000.][4.][0.], expected.flux)
    np.testing.assert_array_equal(results.wavelength, expected.spectral_axis)


def test_descending_input(profile, spectrum):
    descending = Spectrum(spectral_axis=spectrum.wavelength[::-1], flux=spectrum.flux[::-1])
    result = descending.observed_by(profile)
    expected = spectrum.observed_by(profile)
    np.testing.assert_array_equal(result.wavelength, expected.wavelength[::-1])
    np.testing.assert_allclose(result.flux.value, expected.flux.value[::-1])


def narrow_line(mode, center, red_limit=None):
    """Resolve an intrinsic 1%-FWHM line on a compact, fine local input axis."""
    profile = get_profile('jwst/nirspec/' + mode)
    curve = profile.wavelength.to_value(u.um)
    r = profile.resolving_power
    coordinate = np.r_[0., np.cumsum(.5 * (r[1:] + r[:-1]) * np.diff(np.log(curve)))]
    coarse = np.exp(np.interp(np.linspace(0, coordinate[-1], int(8*coordinate[-1])+2),
                             coordinate, np.log(curve)))
    expected_r = profile.resolving_power_at(center*u.um)
    offsets = (np.arange(-780, 781) + .371) / (600 * expected_r)
    wave = np.unique(np.r_[coarse, center*np.exp(offsets)])
    wave = wave[(wave >= curve[0]) & (wave <= curve[-1])]
    if red_limit is not None:
        wave = np.r_[wave[wave < red_limit], red_limit]
    sigma = .01 / (expected_r * np.sqrt(8*np.log(2)))
    flux = center/wave * np.exp(-.5*(np.log(wave/center)/sigma)**2)
    return profile, Spectrum(spectral_axis=wave*u.um, flux=flux*FLUX_UNIT)


def measured_r(segment, center, reference_r):
    """Independent PCHIP half-height crossings of the actual returned flux."""
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    wave = segment.spectral_axis.to_value(u.um)
    selected = abs(np.log(wave/center)) < 1.2/reference_r
    x, y = wave[selected], segment.flux.value[selected]
    peak = int(y.argmax())
    half = y[peak]/2
    left = np.flatnonzero(y[:peak] < half)[-1]
    right = peak + 1 + np.flatnonzero(y[peak+1:] < half)[0]
    interpolator = PchipInterpolator(x, y)
    root = lambda z: float(interpolator(z)) - half
    low = brentq(root, x[left], x[left+1], xtol=center*1e-13)
    high = brentq(root, x[right-1], x[right], xtol=center*1e-13)
    return center/(high-low)


@pytest.mark.parametrize('mode,center', [
    ('prism/clear', .75),
    ('g395m/f290lp', 4.5), ('g395h/f290lp', 4.5),
])
def test_nirspec_isolated_line_resolution(mode, center):
    profile, source = narrow_line(mode, center)
    original = source.flux.copy()
    result = source.observed_by(profile)
    segment = next(s for s in coverage_pieces(result) if s.wavelength.min() < center*u.um < s.wavelength.max())
    reference = profile.resolving_power_at(center*u.um)
    # The measured ~0.1–0.25% remapping bias is far below the 5% goal.
    assert measured_r(segment, center, reference) == pytest.approx(reference, rel=.005)
    np.testing.assert_array_equal(source.flux, original)


@pytest.mark.parametrize('mode,edge,sign,placement', [
    ('prism/clear', .6, 1, 1.), ('prism/clear', 5.3, -1, -.2),
    ('g395h/f290lp', 3.72, -1, 1.), ('g395h/f290lp', 3.82, 1, -.2),
])
def test_nirspec_boundary_profiles(mode, edge, sign, placement):
    p = get_profile('jwst/nirspec/' + mode)
    center = edge + sign*placement*edge/p.resolving_power_at(edge*u.um)
    p, source = narrow_line(mode, center)
    result = source.observed_by(p)
    segment = min(coverage_pieces(result), key=lambda s: min(abs(s.wavelength.to_value(u.um)-edge)))
    full = source.set_variable_resolving_power(p.wavelength, p.resolving_power)
    x = segment.wavelength.to_value(u.um)
    local = abs(np.log(x/center)) < 1.2/p.resolving_power_at(center*u.um)
    indices = np.searchsorted(source.wavelength.to_value(u.um), x[local])
    # Gaps are output geometry: even a line centered in the gap retains its
    # physical wing at the edge. No cropping-induced renormalization is allowed.
    assert segment.flux.value[local].max() > .005
    np.testing.assert_allclose(segment.flux.value[local], full.flux.value[indices],
                               atol=.005*full.flux.value.max())
    if placement == 1:
        reference = p.resolving_power_at(center*u.um)
        assert measured_r(segment, center, reference) == pytest.approx(reference, rel=.005)


def test_prism_5_5_um_source_is_adequate():
    p, short = narrow_line('prism/clear', 5.27, red_limit=5.5)
    _, complete = narrow_line('prism/clear', 5.27)
    result = short.observed_by(p)
    reference = p.resolving_power_at(5.27*u.um)
    assert measured_r(result, 5.27, reference) == pytest.approx(reference, rel=.005)
    full = complete.set_variable_resolving_power(p.wavelength, p.resolving_power)
    indices = np.searchsorted(complete.wavelength.to_value(u.um), result.wavelength.to_value(u.um))
    np.testing.assert_allclose(result.flux.value, full.flux.value[indices],
                               atol=.005*full.flux.value.max())
    with pytest.raises(ValueError, match='padding'):
        short.select_wavelength(wl_max=5.31*u.um).observed_by(p)


def test_padding_includes_broad_kernels_originating_outside_coverage(profile):
    # A narrow boundary kernel alone would miss a much broader source kernel
    # just outside coverage. The support envelope must inspect the calibration
    # outside the requested interval, including its interior low-R extremum.
    p = replace(profile, wavelength=np.array([.5, .85, .86, .89, .9, 1.1, 1.5])*u.um,
                resolving_power=np.array([1000., 1000., 15., 1000., 1000., 1000., 1000.]),
                valid_intervals=np.array([[.9, 1.1]])*u.um)
    wave = np.unique(np.r_[np.geomspace(.5, 1.5, 30000), np.linspace(.8599, .8601, 201)])
    source = Spectrum(spectral_axis=wave*u.um,
                      flux=np.exp(-.5*((wave-.86)/.00002)**2)*FLUX_UNIT)
    result = source.observed_by(p)
    complete = source.set_variable_resolving_power(p.wavelength, p.resolving_power)
    selected = (wave >= .9) & (wave <= 1.1)
    assert result.flux.value.max() > .0001
    np.testing.assert_allclose(result.flux.value, complete.flux.value[selected],
                               atol=.005*complete.flux.value[selected].max())


@pytest.mark.parametrize('samples', [3, 20])
def test_output_binning_explains_apparent_line_width(samples):
    from scipy.special import ndtr
    from synphot.binning import calculate_bin_edges
    center = 4.5
    p, source = narrow_line('prism/clear', center)
    result = source.observed_by(p, samples_per_resolution_element=samples)
    r = p.resolving_power_at(center*u.um)
    x = result.spectral_axis.to_value(u.um)
    edges = calculate_bin_edges(x*u.um).to_value(u.um)
    k = np.sqrt(8*np.log(2))
    sigma = np.hypot(.01/(r*k), 2*np.arcsinh(1/(2*r))/k)
    ideal = np.diff(ndtr(np.log(edges/center)/sigma))/np.diff(edges)
    ideal = Spectrum(spectral_axis=result.spectral_axis, flux=ideal*FLUX_UNIT)
    actual_r = measured_r(result, center, r)
    # Compare to the independently integrated/sampled profile, not an
    # unbinned width. Three samples/FWHM can give >5% apparent R error.
    assert actual_r == pytest.approx(measured_r(ideal, center, r), rel=.005)
    assert abs(actual_r/r-1) < (.08 if samples == 3 else .005)


def test_true_source_gap_is_not_a_detector_gap(nirspec_spectrum):
    # Missing source samples in the padding must not be bridged, even when
    # those wavelengths are outside final detector coverage.
    wave = nirspec_spectrum.wavelength
    selected = (wave < 3.719 * u.um) | (wave > 3.73 * u.um)
    source = Spectrum(spectral_axis=wave[selected], flux=nirspec_spectrum.flux[selected])
    with pytest.raises(ValueError, match='under-sampled'):
        source.observed_by('jwst/nirspec/g395h/f290lp')


def test_finite_flux_and_empty_grid(profile, spectrum):
    spectrum.flux[2000] = np.nan * FLUX_UNIT
    with pytest.raises(ValueError, match='finite source flux'):
        spectrum.observed_by(profile)
    grid = make_grid(spectrum)
    grid.points = np.empty((0, 3))
    with pytest.raises(ValueError, match='no spectra'):
        grid.observed_by(profile)


def test_sphinx_resolution_warning_and_interior_peak(profile, spectrum):
    spectrum.model_grid = 'sphinx'
    with pytest.warns(UserWarning, match='low-resolution source'):
        spectrum.observed_by(profile)
    del spectrum.model_grid
    peaked = replace(profile, resolving_power=[100., 1000., 150.])
    spectrum.meta['native_resolving_power'] = 900
    with pytest.raises(ValueError, match='insufficient'):
        spectrum.observed_by(peaked)


@pytest.fixture
def covered_spectrum():
    wave = np.r_[np.linspace(.9, 1.1, 2001), np.linspace(1.3, 1.5, 2001)]*u.um
    values = np.where(wave.value < 1.2, 1+2*wave.value, 100+3*wave.value)
    return Spectrum(spectral_axis=wave, flux=values*FLUX_UNIT,
                    meta={'valid_intervals': [[.9, 1.1], [1.3, 1.5]]*u.um,
                          'nested': {'values': [1, 2]}})


def test_coverage_validation_and_defensive_metadata(covered_spectrum):
    source = covered_spectrum
    assert source.valid_intervals.unit == u.um
    original = source.valid_intervals.copy()
    limits = source.valid_intervals
    limits[0, 0] = .1*u.um
    np.testing.assert_array_equal(source.valid_intervals, original)
    clone = copy.deepcopy(source)
    clone.meta['nested']['values'].append(3)
    assert source.meta['nested']['values'] == [1, 2]
    np.testing.assert_array_equal(clone.valid_intervals, original)
    with pytest.raises(ValueError, match='outside explicit'):
        Spectrum(spectral_axis=[1., 1.2, 1.4]*u.um, flux=np.ones(3)*FLUX_UNIT,
                 meta={'valid_intervals': original})
    for invalid in ([[1.1, .9]], [[.9, 1.1], [1., 1.2]], [[.9, np.nan]]):
        with pytest.raises(ValueError, match='valid_intervals'):
            Spectrum(spectral_axis=[1., 1.01]*u.um, flux=np.ones(2)*FLUX_UNIT,
                     meta={'valid_intervals': invalid*u.um})


@pytest.mark.parametrize('descending,both_intervals', [(False, False), (True, True)])
def test_gap_aware_resampling(covered_spectrum, descending, both_intervals):
    source = covered_spectrum
    target = np.linspace(.94, 1.06, 21)*u.um
    if both_intervals:
        target = np.r_[target.value, np.linspace(1.34, 1.46, 21)]*u.um
    if descending:
        target = target[::-1]
        source = Spectrum(spectral_axis=source.spectral_axis[::-1], flux=source.flux[::-1], meta=source.meta)
    result = source.resample(target)
    np.testing.assert_array_equal(result.spectral_axis, target)
    assert len(result.valid_intervals) == (2 if both_intervals else 1)
    for a, b in result.valid_intervals:
        mask = (target >= a) & (target <= b)
        interval = next((x, y) for x, y in source.valid_intervals if a >= x and b <= y)
        selected = (source.spectral_axis >= interval[0]) & (source.spectral_axis <= interval[1])
        # Independent single-interval synphot call; red flux is deliberately
        # very different so bridging the gap would not pass unnoticed.
        plain = Spectrum(spectral_axis=source.spectral_axis[selected], flux=source.flux[selected])
        expected = plain.resample(np.sort(target[mask]))
        np.testing.assert_allclose(result.flux.value[mask],
                                   expected.flux.value[::-1] if descending else expected.flux.value,
                                   rtol=1e-12)
    result.meta['nested']['values'].append(3)
    assert covered_spectrum.meta['nested']['values'] == [1, 2]


@pytest.mark.parametrize('target,error', [
    ([1., 1.2, 1.4], 'detector gap'),
    ([.8, 1.], 'coverage'),
    ([1.08, 1.095, 1.34, 1.4], 'bin edges'),
    ([1., 1.34, 1.4], 'two centers'),
])
def test_gap_aware_resampling_rejections(covered_spectrum, target, error):
    for taper in (False, True):
        with pytest.raises(ValueError, match=error):
            covered_spectrum.resample(target*u.um, taper=taper)


def test_no_extrapolation_from_coverage_metadata(covered_spectrum):
    # Physical coverage metadata does not authorize extrapolating a truncated
    # sample array, including after upstream specutils slicing.
    subset = covered_spectrum[:1000]
    with pytest.raises(ValueError, match='sampled wavelength coverage'):
        subset.resample([1.05, 1.06]*u.um)
    with pytest.raises(ValueError, match='sampled wavelength coverage'):
        subset.bin(center=[1.05]*u.um, width=[.01]*u.um)


def test_gap_aware_binning_and_bin_coverage(covered_spectrum):
    center, width = [1., 1.4]*u.um, [.02, .02]*u.um
    result = covered_spectrum.bin(center, width)
    np.testing.assert_allclose(result.flux.value, [3., 104.2], rtol=1e-12)
    np.testing.assert_allclose(result.valid_intervals.to_value(u.um), [[.99, 1.01], [1.39, 1.41]])
    result.meta['nested']['values'].append(3)
    assert covered_spectrum.meta['nested']['values'] == [1, 2]


@pytest.mark.parametrize('center,width', [(1.09, .05), (1.2, .4), (.9, .02)])
def test_gap_crossing_bins_are_rejected(covered_spectrum, center, width):
    with pytest.raises(ValueError, match='Bin edges'):
        covered_spectrum.bin(center=[center]*u.um, width=[width]*u.um)


@pytest.mark.parametrize('limits,expected', [
    ((.95, 1.05), [[.95, 1.05]]),
    ((1., 1.4), [[1., 1.1], [1.3, 1.4]]),
    ((1.2, 1.4), [[1.3, 1.4]]),
])
def test_wavelength_selection_updates_coverage(covered_spectrum, limits, expected):
    result = covered_spectrum.select_wavelength(limits[0]*u.um, limits[1]*u.um)
    np.testing.assert_allclose(result.valid_intervals.to_value(u.um), expected)
    assert not np.any((result.wavelength > 1.1*u.um) & (result.wavelength < 1.3*u.um))
    assert len(covered_spectrum.valid_intervals) == 2
    with pytest.raises(ValueError, match='no wavelength samples'):
        covered_spectrum.select_wavelength(1.15*u.um, 1.25*u.um)


@pytest.mark.parametrize('operation', [
    lambda s: s.regularize(.001*u.um),
    lambda s: s.set_spectral_resolution(.005*u.um),
    lambda s: s.set_spectral_resolving_power(100),
    lambda s: s.set_variable_resolving_power([.8, 1.6]*u.um, [100, 120]),
    lambda s: s.observed_by('jwst/nirspec/prism/clear'),
])
def test_unsafe_processing_rejects_known_gaps(covered_spectrum, operation):
    with pytest.raises(ValueError, match='discontinuous valid_intervals'):
        operation(covered_spectrum)


def test_covered_grid_rejects_broadening(covered_spectrum):
    grid = make_grid(covered_spectrum)
    grid.meta['valid_intervals'] = covered_spectrum.valid_intervals
    for method, args in ((grid.set_spectral_resolution, [.005*u.um]),
                         (grid.set_spectral_resolving_power, [100]),
                         (grid.set_variable_resolving_power, [[.8, 1.6]*u.um, [100, 120]]),
                         (grid.observed_by, ['jwst/nirspec/prism/clear'])):
        with pytest.raises(ValueError, match='discontinuous valid_intervals'):
            method(*args)


def test_no_gap_inference_from_nonuniform_sampling(covered_spectrum):
    plain = Spectrum(spectral_axis=covered_spectrum.spectral_axis, flux=covered_spectrum.flux)
    assert plain.valid_intervals is None
    # Historical behavior remains available for arrays with no explicit gaps.
    assert np.all(np.isfinite(plain.resample([1.18, 1.2, 1.22]*u.um).flux))


def test_continuous_coverage_propagates_through_broadening(profile, spectrum):
    observed = spectrum.observed_by(profile)
    result = observed.set_spectral_resolving_power(80)
    np.testing.assert_array_equal(result.valid_intervals, observed.valid_intervals)
    assert result.meta is not observed.meta


def test_covered_intensity_operations(covered_spectrum):
    from speclib import SpecificIntensitySpectrum
    spec = SpecificIntensitySpectrum(spectral_axis=covered_spectrum.spectral_axis,
                                     intensity=covered_spectrum.flux/u.sr,
                                     meta=covered_spectrum.meta)
    selected = spec.select_wavelength(.94*u.um, 1.06*u.um)
    assert type(selected) is SpecificIntensitySpectrum
    result = selected.resample(np.linspace(.95, 1.05, 20)*u.um)
    assert type(result) is SpecificIntensitySpectrum
    binned = spec.bin(center=[1., 1.4]*u.um, width=[.02, .02]*u.um)
    assert len(binned.valid_intervals) == 2
    with pytest.raises(ValueError, match='Bin edges'):
        spec.bin(center=[1.09]*u.um, width=[.05]*u.um)


def test_regularization_of_one_valid_interval(covered_spectrum):
    selected = covered_spectrum.select_wavelength(.95*u.um, 1.05*u.um)
    result = selected.regularize(.001*u.um)
    assert len(result.valid_intervals) == 1
    assert np.all(np.isfinite(result.flux))
    np.testing.assert_allclose(np.diff(result.spectral_axis.to_value(u.um)), .001, rtol=1e-12)
    assert result.wavelength.min() > selected.wavelength.min()
    assert result.wavelength.max() < selected.wavelength.max()


@pytest.mark.parametrize('state', ['mask', 'uncertainty'])
def test_covered_operations_do_not_silently_discard_ancillary_state(covered_spectrum, state):
    if state == 'mask':
        covered_spectrum.mask = np.zeros(covered_spectrum.flux.size, dtype=bool)
    else:
        covered_spectrum.uncertainty = StdDevUncertainty(np.ones(covered_spectrum.flux.size))
    with pytest.raises(NotImplementedError, match='masks or uncertainties'):
        covered_spectrum.resample([.99, 1., 1.01]*u.um)
    with pytest.raises(NotImplementedError, match='masks or uncertainties'):
        covered_spectrum.bin(center=[1.]*u.um, width=[.02]*u.um)


def test_coverage_uses_wavelength_for_frequency_spectral_axis(covered_spectrum):
    frequency = covered_spectrum.spectral_axis.to(u.THz, equivalencies=u.spectral())
    source = Spectrum(spectral_axis=frequency, flux=covered_spectrum.flux, meta=covered_spectrum.meta)
    target = np.r_[np.linspace(.94, 1.06, 20), np.linspace(1.34, 1.46, 20)]*u.um
    result = source.resample(target)
    reference = covered_spectrum.resample(target)
    np.testing.assert_allclose(result.flux.value, reference.flux.value, rtol=1e-12)
    np.testing.assert_array_equal(result.valid_intervals, reference.valid_intervals)


def test_covered_multidimensional_binning_is_rejected(covered_spectrum):
    source = Spectrum(spectral_axis=covered_spectrum.spectral_axis,
                      flux=np.stack([covered_spectrum.flux.value]*2)*FLUX_UNIT,
                      meta=covered_spectrum.meta)
    with pytest.raises(ValueError, match='one-dimensional'):
        source.bin(center=[1.]*u.um, width=[.02]*u.um)


PHASE2_LINES = [
    ('hst/stis/g430l', .43), ('hst/stis/g750l', .75),
    ('hst/wfc3/g102', .97), ('hst/wfc3/g141', 1.4),
    ('jwst/niriss/soss/order1', 1.25), ('jwst/niriss/soss/order2', .8),
]


@pytest.fixture(scope='module')
def phase2_source():
    wave = np.geomspace(.25, 3., 40000)*u.um
    return Spectrum(spectral_axis=wave, flux=np.ones(wave.size)*FLUX_UNIT,
                    meta={'nested': {'values': [1, 2]}})


@pytest.mark.parametrize('mode,center', PHASE2_LINES)
def test_phase2_coverage_sampling_grid_and_ownership(mode, center, phase2_source):
    profile = get_profile(mode)
    lower, upper = profile.valid_intervals[0].to_value(u.um)
    targets = np.linspace(lower+.1*(upper-lower), upper-.1*(upper-lower), 40)*u.um
    for options in ({}, {'samples_per_resolution_element': 5}, {'wavelength': targets}):
        observed = phase2_source.observed_by(mode, **options)
        assert type(observed) is Spectrum
        np.testing.assert_array_equal(observed.valid_intervals, profile.valid_intervals)
        assert observed.wavelength.min() >= lower*u.um
        assert observed.wavelength.max() <= upper*u.um
        # Full interval boundaries retain sufficient local model/calibration padding.
        np.testing.assert_allclose(observed.flux.value, 1., rtol=2e-3)
        assert observed.meta['instrument_profile']['configuration'] == profile.configuration
        if not options:
            selected = ((phase2_source.wavelength >= lower*u.um)
                        & (phase2_source.wavelength <= upper*u.um))
            np.testing.assert_array_equal(observed.wavelength, phase2_source.wavelength[selected])
    grid = make_grid(phase2_source).observed_by(profile, wavelength=targets)
    assert type(grid) is SpectralGrid
    np.testing.assert_allclose(grid.wavelength.to_value(u.um),
                               observed.wavelength.to_value(u.um), rtol=1e-14)
    np.testing.assert_allclose(grid.data[0], observed.flux.value, rtol=1e-12)
    np.testing.assert_allclose(grid.data[1], 2*observed.flux.value, rtol=1e-12)
    extracted = grid.get_spectrum(3000., 4., 0., interpolate=False)
    extracted.meta['instrument_profile']['mode'] = 'changed'
    extracted.meta['nested']['values'].append(3)
    assert grid.meta['instrument_profile']['mode'] == mode
    assert phase2_source.meta == {'nested': {'values': [1, 2]}}
    assert np.all(phase2_source.flux.value == 1)
    # No source padding must still fail for the new nominal prescriptions.
    with pytest.raises(ValueError, match='padding'):
        phase2_source.select_wavelength(lower*u.um, upper*u.um).observed_by(mode)


@pytest.mark.parametrize('mode,center', PHASE2_LINES)
def test_phase2_isolated_line_widths(mode, center):
    profile = get_profile(mode)
    reference_r = profile.resolving_power_at(center*u.um)
    fwhm = center/reference_r
    # Resolve the intrinsically narrow line and its wings on a local fine mesh.
    wave = np.unique(np.r_[np.geomspace(.25, 3., 40000),
                           np.linspace(center-5*fwhm, center+5*fwhm, 6001)])
    flux = np.exp(-.5*((wave-center)/(fwhm/100))**2)
    source = Spectrum(spectral_axis=wave*u.um, flux=flux*FLUX_UNIT)
    result = source.observed_by(profile)
    assert measured_r(result, center, reference_r) == pytest.approx(reference_r, rel=.005)


def test_soss_orders_stay_independent(phase2_source):
    first = phase2_source.observed_by('jwst/niriss/soss/order1')
    second = phase2_source.observed_by('jwst/niriss/soss/order2')
    assert type(first) is Spectrum and type(second) is Spectrum
    assert first.valid_intervals[0, 0] < second.valid_intervals[0, 1]
    assert first.meta['instrument_profile']['mode'].endswith('order1')
    assert second.meta['instrument_profile']['mode'].endswith('order2')
    with pytest.raises(ValueError, match='outside|coverage'):
        phase2_source.observed_by('jwst/niriss/soss/order2', wavelength=[1.1, 1.2]*u.um)
