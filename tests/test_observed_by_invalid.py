"""``observed_by(..., on_invalid="nan")``: unsupported instrument bins -> NaN."""

import copy
import warnings

import astropy.units as u
import numpy as np
import pytest
from scipy.interpolate import NearestNDInterpolator
from synphot.binning import calculate_bin_edges

from speclib import InstrumentProfile, Spectrum, SpectralGrid, get_profile

FLUX_UNIT = u.erg / u.s / u.cm**2 / u.AA


def make_grid(spectrum):
    grid = SpectralGrid.__new__(SpectralGrid)
    grid.model_grid = 'phoenix'
    grid.wavelength = spectrum.wavelength.copy()
    grid.unit = spectrum.flux.unit
    grid.points = np.array([[3000., 4., 0.], [3100., 4., 0.]])
    grid.teffs, grid.loggs, grid.metallicities = (
        np.array([3000., 3100.]), np.array([4.]), np.array([0.]))
    grid.teff_bds, grid.logg_bds, grid.metallicity_bds = (
        (3000., 3100.), (4., 4.), (0., 0.))
    grid.fluxes = {3000.: {4.: {0.: spectrum.flux.copy()}},
                   3100.: {4.: {0.: 3 * spectrum.flux}}}
    grid.data = np.vstack([spectrum.flux.value, 3 * spectrum.flux.value])
    grid.meta = {}
    grid.interpolator = NearestNDInterpolator(grid.points, grid.data)
    return grid


def expected_valid(profile, centers):
    """Independent classification: both original implicit edges in one interval."""
    values = np.asarray(centers.to_value(u.um))
    order = np.argsort(values)
    edges = calculate_bin_edges(values[order] * u.um).to_value(u.um)
    ok = np.zeros(values.size, dtype=bool)
    for lower, upper in profile.explicit_intervals.to_value(u.um):
        ok[order] |= (edges[:-1] >= lower) & (edges[1:] <= upper)
    return ok


def structured_flux(wave_um):
    return 1 + 0.5 * np.sin(1500 * wave_um)**2


# --- synthetic profile: exact comparison with strict calculations -----------

CAL = dict(source='s', source_url='https://example.org', calibration_version='1',
           assumptions=('a',), quality='nominal')


def synthetic(interval):
    return InstrumentProfile(
        mode='test/mode', wavelength=np.array([.5, .8, 1.1]) * u.um,
        resolving_power=np.array([300., 350., 400.]),
        valid_intervals=np.array([interval]) * u.um, **CAL)


@pytest.fixture(scope='module')
def source():
    wave = np.geomspace(.3, 1.6, 80000) * u.um
    return Spectrum(spectral_axis=wave, flux=structured_flux(wave.value) * FLUX_UNIT)


@pytest.fixture(scope='module')
def irregular():
    return np.sort(np.random.default_rng(0).uniform(.52, 1.08, 150)) * u.um


@pytest.mark.parametrize('descending', [False, True])
def test_valid_bins_use_original_edges_and_nan_elsewhere(source, irregular, descending):
    narrow, wide = synthetic([.6, 1.]), synthetic([.5, 1.1])
    centers = irregular[::-1] if descending else irregular
    reference = source.observed_by(wide, wavelength=centers)
    with pytest.warns(UserWarning, match='of 150 requested wavelength bins') as record:
        result = source.observed_by(narrow, wavelength=centers, on_invalid='nan')
    assert len(record) == 1
    assert 'test/mode' in str(record[0].message)
    np.testing.assert_array_equal(result.wavelength, centers)
    valid = expected_valid(narrow, centers)
    assert 0 < valid.sum() < len(centers)
    np.testing.assert_array_equal(result.valid_mask, valid)
    assert np.all(np.isnan(result.flux.value[~valid]))
    # Same bins as an all-supported strict call on the whole array, including
    # the outermost supported bins, whose edges depend on excluded neighbors.
    np.testing.assert_allclose(result.flux.value[valid], reference.flux.value[valid], rtol=1e-5)
    assert result.meta['instrument_profile']['requested_bins'] == len(centers)
    assert result.meta['instrument_profile']['valid_bins'] == valid.sum()
    covered = np.sort(result.wavelength.to_value(u.um)[valid])
    low, high = result.valid_intervals.to_value(u.um)[0]
    assert low < covered[0] and covered[-1] < high  # returned coverage = outer bin edges


def test_recomputing_edges_would_change_boundary_bins(source, irregular):
    """Guard: trimming first (the old workaround) changes the edge bins."""
    narrow = synthetic([.6, 1.])
    with pytest.warns(UserWarning):
        result = source.observed_by(narrow, wavelength=irregular, on_invalid='nan')
    valid = result.valid_mask
    trimmed = source.observed_by(narrow, wavelength=irregular[valid])
    np.testing.assert_allclose(result.flux.value[valid][1:-1], trimmed.flux.value[1:-1], rtol=1e-4)
    assert abs(result.flux.value[valid][0] / trimmed.flux.value[0] - 1) > 1e-4


def test_center_supported_but_edge_not(source):
    profile = synthetic([.6, 1.])
    centers = np.array([.7, .8, .9, .98, .9999]) * u.um  # last center inside, edge past 1.0
    with pytest.warns(UserWarning, match='implicit bin edges'):
        result = source.observed_by(profile, wavelength=centers, on_invalid='nan')
    np.testing.assert_array_equal(result.valid_mask, [True, True, True, True, False])
    assert np.isnan(result.flux.value[-1])
    with pytest.raises(ValueError):
        source.observed_by(profile, wavelength=centers)


def test_all_valid_matches_strict_and_does_not_warn(source):
    profile = synthetic([.5, 1.1])
    centers = np.linspace(.55, 1.05, 40) * u.um
    strict = source.observed_by(profile, wavelength=centers)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = source.observed_by(profile, wavelength=centers, on_invalid='nan')
    np.testing.assert_array_equal(result.flux.value, strict.flux.value)
    assert result.valid_mask.all()


def test_strict_default_unsupported_and_option_validation(source, irregular):
    narrow = synthetic([.6, 1.])
    with pytest.raises(ValueError, match='outside valid coverage'):
        source.observed_by(narrow, wavelength=irregular)
    with pytest.raises(ValueError, match='outside valid coverage'):
        source.observed_by(narrow, wavelength=irregular, on_invalid='raise')
    with pytest.raises(ValueError, match='on_invalid'):
        source.observed_by(narrow, wavelength=irregular, on_invalid='ignore')
    with pytest.raises(ValueError, match='mutually exclusive'):
        source.observed_by(narrow, wavelength=irregular,
                           samples_per_resolution_element=3, on_invalid='nan')


def test_completely_unsupported_raises_clearly(source):
    narrow = synthetic([.6, 1.])
    centers = np.linspace(1.2, 1.4, 10) * u.um
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # an error, not a warning plus all-NaN output
        with pytest.raises(ValueError, match='None of the 10 requested'):
            source.observed_by(narrow, wavelength=centers, on_invalid='nan')
        with pytest.raises(ValueError, match='None of the 10 requested'):
            make_grid(source).observed_by(narrow, wavelength=centers, on_invalid='nan')


def test_calibration_padding_excludes_bins_without_extrapolating(source):
    # Valid interval reaches the calibration edge, so edge bins lack kernel padding.
    profile = synthetic([.5, 1.1])
    centers = np.linspace(.502, 1.098, 60) * u.um
    with pytest.warns(UserWarning, match='convolution padding'):
        result = source.observed_by(profile, wavelength=centers, on_invalid='nan')
    assert 0 < result.valid_mask.sum() < len(centers)
    assert not result.valid_mask[0] and not result.valid_mask[-1]
    assert np.all(np.isnan(result.flux.value[~result.valid_mask]))
    assert np.all(np.isfinite(result.flux.value[result.valid_mask]))


def test_source_model_failures_remain_errors(irregular):
    narrow = synthetic([.6, 1.])
    short = Spectrum(spectral_axis=np.linspace(.62, .98, 20000) * u.um,
                     flux=np.ones(20000) * FLUX_UNIT)
    with pytest.raises(ValueError, match='must cover'):
        short.observed_by(narrow, wavelength=irregular, on_invalid='nan')
    sparse = Spectrum(spectral_axis=np.linspace(.3, 1.6, 300) * u.um,
                      flux=np.ones(300) * FLUX_UNIT)
    with pytest.raises(ValueError, match='under-sampled|insufficient|sampl'):
        sparse.observed_by(narrow, wavelength=irregular, on_invalid='nan')


# --- real profiles, exoTEDRF-style untrimmed arrays ---------------------------

EXTRACTED = [
    ('jwst/niriss/soss/order1', (0.846, 2.834)),
    ('jwst/niriss/soss/order2', (0.500, 1.409)),
    ('jwst/nirspec/g395m/f290lp', (2.589, 5.183)),
    ('jwst/nirspec/g395h/f290lp', (2.674, 3.721)),   # NRS1
    ('jwst/nirspec/g395h/f290lp', (3.819, 5.177)),   # NRS2
    ('jwst/nirspec/g395h/f290lp', (2.674, 5.177)),   # both detectors, gap in between
]


@pytest.fixture(scope='module')
def broad_source():
    wave = np.geomspace(.25, 7., 90000) * u.um
    return Spectrum(spectral_axis=wave, flux=np.ones(wave.size) * FLUX_UNIT)


@pytest.mark.parametrize('mode,limits', EXTRACTED)
@pytest.mark.parametrize('descending', [False, True])
def test_real_modes_untrimmed_arrays(broad_source, mode, limits, descending):
    profile = get_profile(mode)
    # Irregular sampling, as for a wavelength solution rather than a model grid.
    t = np.sort(np.random.default_rng(1).uniform(0, 1, 120))
    centers = (limits[0] + (limits[1] - limits[0]) * t) * u.um
    if descending:
        centers = centers[::-1]
    valid = expected_valid(profile, centers)
    n_invalid = np.count_nonzero(~valid)
    assert valid.any()
    for target in (broad_source, make_grid(broad_source)):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            result = target.observed_by(profile, wavelength=centers, on_invalid='nan')
        mine = [w for w in record if issubclass(w.category, UserWarning)
                and 'requested wavelength bins' in str(w.message)]
        assert len(mine) == (1 if n_invalid else 0)
        if n_invalid:
            assert mode in str(mine[0].message) and f'{n_invalid} of 120' in str(mine[0].message)
        np.testing.assert_array_equal(result.wavelength, centers)
        np.testing.assert_array_equal(result.valid_mask, valid)
        flux = result.flux if isinstance(result, Spectrum) else None
        if flux is None:
            assert result.data.shape == (2, 120)
            rows = result.data
        else:
            rows = flux.value[None]
        assert np.all(np.isnan(rows[:, ~valid]))
        np.testing.assert_allclose(rows[:, valid], rows[:, valid] * 0 + np.array([[1.], [3.]])[:rows.shape[0]]
                                   if rows.shape[0] == 2 else 1., rtol=5e-3)
        # Modeled coverage never spans a detector gap or unsupported region.
        for low, high in result.valid_intervals.to_value(u.um):
            assert any(a <= low and high <= b
                       for a, b in profile.explicit_intervals.to_value(u.um))


def test_g395h_gap_bins_are_nan(broad_source):
    profile = get_profile('jwst/nirspec/g395h/f290lp')
    centers = np.linspace(3.5, 4., 51) * u.um   # straddles the 3.72-3.82 um gap
    with pytest.warns(UserWarning):
        result = broad_source.observed_by(profile, wavelength=centers, on_invalid='nan')
    um = centers.to_value(u.um)
    gap = (um > 3.725) & (um < 3.815)          # bins wholly inside the gap
    assert gap.sum() > 5
    assert np.all(np.isnan(result.flux.value[gap]))
    assert np.all(np.isfinite(result.flux.value[um < 3.71]))
    assert np.all(np.isfinite(result.flux.value[um > 3.83]))
    np.testing.assert_array_equal(result.valid_mask, expected_valid(profile, centers))
    assert len(result.valid_intervals) == 2


def test_real_profile_valid_bins_match_strict(broad_source):
    mode = 'jwst/niriss/soss/order2'
    profile = get_profile(mode)
    centers = np.linspace(.5, 1.2, 400) * u.um
    structured = Spectrum(spectral_axis=broad_source.wavelength,
                          flux=structured_flux(broad_source.wavelength.value) * FLUX_UNIT)
    with pytest.warns(UserWarning):
        result = structured.observed_by(mode, wavelength=centers, on_invalid='nan')
    valid = result.valid_mask
    inside = np.flatnonzero(valid)[1:-1]
    # Interior bins share edges with the strict subset call.
    strict = structured.observed_by(mode, wavelength=centers[valid])
    np.testing.assert_allclose(result.flux.value[inside], strict.flux.value[1:-1], rtol=1e-4)


# --- grids and downstream operations -----------------------------------------

def test_grid_shares_mask_and_downstream_interpolation(source, irregular):
    narrow = synthetic([.6, 1.])
    with pytest.warns(UserWarning) as record:
        spec = source.observed_by(narrow, wavelength=irregular, on_invalid='nan')
    with pytest.warns(UserWarning) as grid_record:
        grid = make_grid(source).observed_by(narrow, wavelength=irregular, on_invalid='nan')
    assert len(grid_record) == 1 and len(record) == 1
    valid = spec.valid_mask
    np.testing.assert_array_equal(grid.valid_mask, valid)
    np.testing.assert_array_equal(grid.wavelength, irregular)
    assert np.all(np.isnan(grid.data[:, ~valid]))
    assert np.all(np.isfinite(grid.data[:, valid]))
    np.testing.assert_allclose(grid.data[0][valid], spec.flux.value[valid], rtol=1e-12)
    np.testing.assert_allclose(grid.data[1][valid], 3 * spec.flux.value[valid], rtol=1e-12)

    for interpolate in (True, False):
        mid = grid.get_spectrum(3050., 4., 0., interpolate=interpolate)
        assert isinstance(mid, Spectrum)
        np.testing.assert_array_equal(mid.valid_mask, valid)
        assert np.all(np.isnan(mid.flux.value[~valid]))
        assert np.all(np.isfinite(mid.flux.value[valid]))
    mid = grid.get_spectrum(3050., 4., 0.)
    np.testing.assert_allclose(mid.flux.value[valid], 2 * spec.flux.value[valid], rtol=1e-12)
    np.testing.assert_array_equal(mid.valid_intervals, spec.valid_intervals)
    assert mid.meta['instrument_profile']['valid_bins'] == valid.sum()


def test_nan_filled_spectrum_downstream_operations(source, irregular):
    narrow = synthetic([.6, 1.])
    with pytest.warns(UserWarning):
        spec = source.observed_by(narrow, wavelength=irregular, on_invalid='nan')
    valid = spec.valid_mask

    # Round trips keep the NaN-filled axis and mask (deepcopy and re-construction).
    clone = copy.deepcopy(spec)
    np.testing.assert_array_equal(clone.valid_mask, valid)
    rebuilt = Spectrum(spectral_axis=spec.spectral_axis, flux=spec.flux, meta=spec.meta)
    np.testing.assert_array_equal(rebuilt.flux.value, spec.flux.value)

    # Flux outside coverage is only allowed as NaN, never as a physical value.
    bad = spec.flux.value.copy()
    bad[~valid] = 1.
    with pytest.raises(ValueError, match='outside explicit'):
        Spectrum(spectral_axis=spec.spectral_axis, flux=bad * FLUX_UNIT, meta=spec.meta)

    # Resampling uses only supported samples; NaNs cannot leak into valid bins.
    targets = spec.wavelength[valid][2:-2]
    resampled = spec.resample(targets)
    assert np.all(np.isfinite(resampled.flux.value))
    good = spec.flux.value[valid]
    assert resampled.flux.value.min() >= good.min() - 1e-12
    assert resampled.flux.value.max() <= good.max() + 1e-12
    with pytest.raises(ValueError, match='outside valid coverage'):
        spec.resample(spec.wavelength)       # includes unsupported centers

    # Selecting the supported range gives a clean NaN-free spectrum.
    selected = spec.select_wavelength(spec.wavelength[valid].min(), spec.wavelength[valid].max())
    assert np.all(np.isfinite(selected.flux.value))
    assert selected.valid_mask.all()

    # Operations that would smear or invent flux across NaNs are rejected.
    for call in (lambda: spec.set_spectral_resolving_power(500),
                 lambda: spec.regularize(),
                 lambda: spec.observed_by(narrow)):
        with pytest.raises(ValueError, match='NaN-filled|outside'):
            call()
    with pytest.warns(UserWarning):
        grid = make_grid(source).observed_by(narrow, wavelength=irregular, on_invalid='nan')
    with pytest.raises(ValueError, match='NaN-filled|outside'):
        grid.set_spectral_resolving_power(500)

    # Binning ignores NaN bins when its bins lie in supported coverage.
    center = u.Quantity([float(np.median(spec.wavelength.to_value(u.um)[valid]))], u.um)
    binned = spec.bin(center, [0.02] * u.um)
    assert np.all(np.isfinite(binned.flux.value))


def test_valid_mask_without_coverage_is_all_true(source):
    assert source.valid_mask.all() and source.valid_mask.shape == source.wavelength.shape
    assert make_grid(source).valid_mask.all()
