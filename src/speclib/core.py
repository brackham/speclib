import copy
import os
from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import speclib.utils as utils
from scipy.interpolate import NearestNDInterpolator
from specutils import Spectrum1D

import warnings

import synphot as sp

__all__ = ["Spectrum", "BinnedSpectrum", "SpectralGrid", "BinnedSpectralGrid"]


_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_SAMPLES_PER_FWHM = 10
_VARIABLE_SAMPLES_PER_FWHM = 20
_MIN_SAMPLES_PER_FWHM = 2
# Limit the estimated peak working set for temporary coordinate/flux arrays.
# Eight float64-sized arrays per temporary point is deliberately conservative.
_MAX_TEMPORARY_WORKING_BYTES = 512 * 1024**2
_ESTIMATED_BYTES_PER_TEMPORARY_POINT = 8 * np.dtype(float).itemsize
_VARIABLE_KERNEL_RADIUS_SIGMA = 4.0
_ESTIMATED_BYTES_PER_SPARSE_NONZERO = 24
_MAX_VARIABLE_KERNEL_NONZEROS = 20_000_000


def _sphinx_grid_slice(co_ratio):
    """Return grid axes and exact combinations for one SPHINX C/O slice."""

    if co_ratio is None:
        raise ValueError(
            "co_ratio must be specified for the SPHINX I V4 model grid. "
            "Available values are 0.3, 0.5, 0.7, and 0.9."
        )

    model_list = utils.load_sphinx_model_list()
    available_co_ratios = model_list["grid_co_ratios"]
    matches = np.flatnonzero(np.isclose(available_co_ratios, co_ratio))
    if not matches.size:
        raise ValueError(
            f"C/O={co_ratio} is not available in SPHINX I V4. "
            f"Available values are {available_co_ratios.tolist()}."
        )
    canonical_co_ratio = float(available_co_ratios[matches[0]])
    combinations = model_list["combinations"]
    combinations = combinations[combinations[:, 3] == canonical_co_ratio]

    grid_points = {
        "grid_teffs": np.unique(combinations[:, 0]),
        "grid_loggs": np.unique(combinations[:, 1]),
        "grid_fehs": np.unique(combinations[:, 2]),
        "grid_co_ratios": available_co_ratios,
    }
    return grid_points, combinations, canonical_co_ratio


def _flanking_values(grid, value):
    """Return the true lower and upper grid values around a query."""

    grid = np.asarray(grid)
    lower = grid[grid <= value].max() if np.any(grid <= value) else grid.min()
    upper = grid[grid >= value].min() if np.any(grid >= value) else grid.max()
    return np.array([lower, upper])


def _nearest_available_index(points, requested):
    """Return the nearest actual grid combination in grid-step units."""

    scales = []
    for axis in range(3):
        values = np.unique(points[:, axis])
        scales.append(np.median(np.diff(values)) if len(values) > 1 else 1.0)
    distances = np.sum(
        ((points - np.asarray(requested, dtype=float)) / np.asarray(scales)) ** 2,
        axis=1,
    )
    return int(np.argmin(distances))


def _mps_atlas_grid_slice(model_set):
    """Return grid axes and exact combinations for one MPS-ATLAS set."""

    model_list = utils.load_mps_atlas_model_list(model_set)
    grid_points = {
        "grid_teffs": model_list["grid_teffs"],
        "grid_loggs": model_list["grid_loggs"],
        "grid_fehs": model_list["grid_fehs"],
    }
    return grid_points, model_list["combinations"]


def _validate_mps_atlas_ranges(teff, logg, metallicity, grid_points, model_set):
    """Reject coordinates outside one MPS-ATLAS set's available axes."""

    coordinates = (teff, logg, metallicity)
    axis_names = ("Teff", "logg", "[M/H]")
    axes = (
        grid_points["grid_teffs"],
        grid_points["grid_loggs"],
        grid_points["grid_fehs"],
    )
    for name, value, axis in zip(axis_names, coordinates, axes):
        if value < np.min(axis) or value > np.max(axis):
            raise ValueError(
                f"MPS-ATLAS {model_set} {name}={value} is outside the "
                f"available range [{np.min(axis)}, {np.max(axis)}]."
            )


def _validate_delta_lambda(delta_lambda):
    """Return a positive scalar wavelength FWHM in Angstrom."""
    if not isinstance(delta_lambda, u.Quantity):
        raise TypeError(
            "delta_lambda must be an astropy Quantity with wavelength units"
        )
    if not delta_lambda.unit.is_equivalent(u.AA):
        raise u.UnitsError("delta_lambda must have wavelength units")

    value = np.asarray(delta_lambda.to_value(u.AA))
    if value.ndim != 0:
        raise ValueError("delta_lambda must be a scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("delta_lambda must be finite and positive")
    return value


def _validate_resolving_power(resolving_power):
    """Return a positive scalar dimensionless resolving power."""
    if isinstance(resolving_power, u.Quantity):
        if not resolving_power.unit.is_equivalent(u.dimensionless_unscaled):
            raise u.UnitsError("resolving_power must be dimensionless")
        value = np.asarray(resolving_power.to_value(u.dimensionless_unscaled))
    else:
        if isinstance(resolving_power, (bool, np.bool_)):
            raise TypeError("resolving_power must be a real scalar")
        value = np.asarray(resolving_power)

    if value.ndim != 0:
        raise ValueError("resolving_power must be a scalar")
    if np.iscomplexobj(value):
        raise TypeError("resolving_power must be a real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("resolving_power must be a real scalar") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("resolving_power must be finite and positive")
    return value


def _validate_resolving_power_curve(
    wavelength,
    resolving_power,
    spectrum_wavelength,
):
    """Return ascending wavelength and resolving-power profile arrays."""
    if not isinstance(wavelength, u.Quantity):
        raise TypeError(
            "wavelength must be an astropy Quantity with wavelength units"
        )
    if not wavelength.unit.is_equivalent(u.AA):
        raise u.UnitsError("wavelength must have wavelength units")

    curve_wavelength = np.asarray(wavelength.to_value(u.AA))
    if curve_wavelength.ndim != 1 or curve_wavelength.size < 2:
        raise ValueError("wavelength must be a one-dimensional array")
    if np.issubdtype(curve_wavelength.dtype, np.bool_) or np.iscomplexobj(
        curve_wavelength
    ):
        raise TypeError("wavelength must contain real values")
    try:
        curve_wavelength = curve_wavelength.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("wavelength must contain real values") from exc
    if not np.all(np.isfinite(curve_wavelength)):
        raise ValueError("wavelength must contain only finite values")
    if np.any(curve_wavelength <= 0):
        raise ValueError("wavelength must contain only positive values")

    differences = np.diff(curve_wavelength)
    if np.all(differences > 0):
        pass
    elif np.all(differences < 0):
        curve_wavelength = curve_wavelength[::-1]
    else:
        raise ValueError("wavelength must be strictly monotonic")

    if isinstance(resolving_power, u.Quantity):
        if not resolving_power.unit.is_equivalent(u.dimensionless_unscaled):
            raise u.UnitsError("resolving_power must be dimensionless")
        curve_resolving_power = np.asarray(
            resolving_power.to_value(u.dimensionless_unscaled)
        )
    else:
        curve_resolving_power = np.asarray(resolving_power)

    if curve_resolving_power.ndim != 1:
        raise ValueError("resolving_power must be a one-dimensional array")
    if curve_resolving_power.shape != curve_wavelength.shape:
        raise ValueError("wavelength and resolving_power must have matching shapes")
    if np.issubdtype(curve_resolving_power.dtype, np.bool_):
        raise TypeError("resolving_power must contain real values")
    if np.iscomplexobj(curve_resolving_power):
        raise TypeError("resolving_power must contain real values")
    try:
        curve_resolving_power = curve_resolving_power.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("resolving_power must contain real values") from exc
    if not np.all(np.isfinite(curve_resolving_power)):
        raise ValueError("resolving_power must contain only finite values")
    if np.any(curve_resolving_power <= 0):
        raise ValueError("resolving_power must contain only positive values")

    if differences[0] < 0:
        curve_resolving_power = curve_resolving_power[::-1]

    spectrum_values = np.asarray(spectrum_wavelength.to_value(u.AA), dtype=float)
    spectrum_minimum = np.min(spectrum_values)
    spectrum_maximum = np.max(spectrum_values)
    spectrum_ascending = np.sort(spectrum_values)
    blue_tolerance = 1e-6 * (
        spectrum_ascending[1] - spectrum_ascending[0]
    )
    red_tolerance = 1e-6 * (
        spectrum_ascending[-1] - spectrum_ascending[-2]
    )
    blue_gap = curve_wavelength[0] - spectrum_minimum
    red_gap = spectrum_maximum - curve_wavelength[-1]
    if 0.0 < blue_gap <= blue_tolerance:
        curve_wavelength[0] = spectrum_minimum
        blue_gap = 0.0
    if 0.0 < red_gap <= red_tolerance:
        curve_wavelength[-1] = spectrum_maximum
        red_gap = 0.0
    misses_blue_edge = blue_gap > 0.0
    misses_red_edge = red_gap > 0.0
    if misses_blue_edge or misses_red_edge:
        raise ValueError(
            "The resolving-power curve must cover the full wavelength range "
            "of the input spectrum; extrapolation is not supported"
        )

    return curve_wavelength, curve_resolving_power


def _build_gaussian_convolution_plan(
    wavelength,
    *,
    delta_lambda=None,
    resolving_power=None,
    validate_sampling=True,
):
    """Build reusable coordinate and kernel information for Gaussian smoothing."""
    if (delta_lambda is None) == (resolving_power is None):
        raise ValueError("Specify exactly one resolution mode")

    wave = np.asarray(wavelength.to_value(u.AA), dtype=float)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("The wavelength axis must contain at least two points")
    if not np.all(np.isfinite(wave)):
        raise ValueError("The wavelength axis must contain only finite values")

    differences = np.diff(wave)
    if np.all(differences > 0):
        order = np.arange(wave.size)
    elif np.all(differences < 0):
        order = np.arange(wave.size - 1, -1, -1)
    else:
        raise ValueError("The wavelength axis must be strictly monotonic")

    wave_ascending = wave[order]
    if resolving_power is None:
        coordinate = wave_ascending
        coordinate_fwhm = delta_lambda
        logarithmic = False
    else:
        if np.any(wave_ascending <= 0):
            raise ValueError(
                "Wavelengths must be positive for resolving-power convolution"
            )
        coordinate = np.log(wave_ascending)
        coordinate_fwhm = 2.0 * np.arcsinh(1.0 / (2.0 * resolving_power))
        logarithmic = True

    coordinate_differences = np.diff(coordinate)
    maximum_spacing = np.max(coordinate_differences)
    required_fwhm = _MIN_SAMPLES_PER_FWHM * maximum_spacing
    if validate_sampling and coordinate_fwhm < required_fwhm:
        resolution_name = (
            "resolving power" if logarithmic else "wavelength resolution"
        )
        raise ValueError(
            f"Requested {resolution_name} is under-sampled by the input grid: "
            f"the target FWHM in the convolution coordinate is "
            f"{coordinate_fwhm:.6g}, but at least {required_fwhm:.6g} is "
            f"required for {_MIN_SAMPLES_PER_FWHM} samples per FWHM."
        )

    span = coordinate[-1] - coordinate[0]
    target_spacing = min(
        coordinate_fwhm / _SAMPLES_PER_FWHM,
        np.min(coordinate_differences),
    )
    interval_count_float = span / target_spacing
    maximum_temporary_points = (
        _MAX_TEMPORARY_WORKING_BYTES
        // _ESTIMATED_BYTES_PER_TEMPORARY_POINT
    )
    if (
        not np.isfinite(interval_count_float)
        or interval_count_float + 1 > maximum_temporary_points
    ):
        estimated_mib = (
            (interval_count_float + 1)
            * _ESTIMATED_BYTES_PER_TEMPORARY_POINT
            / 1024**2
        )
        raise ValueError(
            "Temporary convolution grid is impractical: the estimated "
            f"working set is {estimated_mib:.1f} MiB, exceeding the "
            f"{_MAX_TEMPORARY_WORKING_BYTES / 1024**2:.0f} MiB safety limit."
        )

    interval_count = max(1, int(np.ceil(interval_count_float)))
    uniform_coordinate = np.linspace(
        coordinate[0], coordinate[-1], interval_count + 1
    )
    uniform_spacing = uniform_coordinate[1] - uniform_coordinate[0]
    sigma_pixels = coordinate_fwhm * _FWHM_TO_SIGMA / uniform_spacing
    kernel_points = int(np.ceil(8.0 * sigma_pixels)) + 1
    estimated_bytes = (
        uniform_coordinate.size * _ESTIMATED_BYTES_PER_TEMPORARY_POINT
        + kernel_points * np.dtype(float).itemsize
    )
    if estimated_bytes > _MAX_TEMPORARY_WORKING_BYTES:
        raise ValueError(
            "Temporary convolution grid is impractical: the estimated "
            f"working set is {estimated_bytes / 1024**2:.1f} MiB, exceeding "
            f"the {_MAX_TEMPORARY_WORKING_BYTES / 1024**2:.0f} MiB safety limit."
        )

    return {
        "coordinate": coordinate,
        "logarithmic": logarithmic,
        "order": order,
        "uniform_coordinate": uniform_coordinate,
        "sigma_pixels": sigma_pixels,
        "wave_ascending": wave_ascending,
    }


def _build_variable_resolving_power_plan(
    wavelength,
    curve_wavelength,
    curve_resolving_power,
):
    """Build a Gaussian convolution plan for sampled resolving power."""
    if np.all(curve_resolving_power == curve_resolving_power[0]):
        return _build_gaussian_convolution_plan(
            wavelength,
            resolving_power=curve_resolving_power[0],
        )

    wave = np.asarray(wavelength.to_value(u.AA), dtype=float)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("The wavelength axis must contain at least two points")
    if not np.all(np.isfinite(wave)):
        raise ValueError("The wavelength axis must contain only finite values")

    differences = np.diff(wave)
    if np.all(differences > 0):
        order = np.arange(wave.size)
    elif np.all(differences < 0):
        order = np.arange(wave.size - 1, -1, -1)
    else:
        raise ValueError("The wavelength axis must be strictly monotonic")

    wave_ascending = wave[order]
    if np.any(wave_ascending <= 0):
        raise ValueError(
            "Wavelengths must be positive for resolving-power convolution"
        )
    coordinate = np.log(wave_ascending)
    coordinate_differences = np.diff(coordinate)

    resolving_power_at_samples = np.interp(
        wave_ascending,
        curve_wavelength,
        curve_resolving_power,
    )
    maximum_interval_resolving_power = np.maximum(
        resolving_power_at_samples[:-1],
        resolving_power_at_samples[1:],
    )
    interior_curve = (
        (curve_wavelength > wave_ascending[0])
        & (curve_wavelength < wave_ascending[-1])
    )
    interior_indices = np.searchsorted(
        wave_ascending,
        curve_wavelength[interior_curve],
        side="right",
    ) - 1
    valid_indices = interior_indices < maximum_interval_resolving_power.size
    np.maximum.at(
        maximum_interval_resolving_power,
        interior_indices[valid_indices],
        curve_resolving_power[interior_curve][valid_indices],
    )

    interval_fwhm = 2.0 * np.arcsinh(
        1.0 / (2.0 * maximum_interval_resolving_power)
    )
    under_sampled = interval_fwhm < (
        _MIN_SAMPLES_PER_FWHM * coordinate_differences
    )
    if np.any(under_sampled):
        index = np.flatnonzero(under_sampled)[0]
        raise ValueError(
            "Requested resolving power is under-sampled by the input grid "
            f"near {wave_ascending[index]:.6g} Angstrom: the target FWHM "
            f"in log wavelength is {interval_fwhm[index]:.6g}, but at least "
            f"{_MIN_SAMPLES_PER_FWHM * coordinate_differences[index]:.6g} "
            f"is required for {_MIN_SAMPLES_PER_FWHM} samples per FWHM."
        )

    curve_in_range = (
        (curve_wavelength > wave_ascending[0])
        & (curve_wavelength < wave_ascending[-1])
    )
    metric_wavelength = np.unique(
        np.concatenate(
            (wave_ascending, curve_wavelength[curve_in_range])
        )
    )
    metric_coordinate = np.log(metric_wavelength)
    metric_resolving_power = np.interp(
        metric_wavelength,
        curve_wavelength,
        curve_resolving_power,
    )
    metric_fwhm = 2.0 * np.arcsinh(
        1.0 / (2.0 * metric_resolving_power)
    )
    resolution_coordinate = np.zeros(metric_coordinate.size)
    resolution_coordinate[1:] = np.cumsum(
        0.5
        * (1.0 / metric_fwhm[:-1] + 1.0 / metric_fwhm[1:])
        * np.diff(metric_coordinate)
    )
    interval_count_float = (
        _VARIABLE_SAMPLES_PER_FWHM * resolution_coordinate[-1]
    )
    maximum_temporary_points = (
        _MAX_TEMPORARY_WORKING_BYTES
        // _ESTIMATED_BYTES_PER_TEMPORARY_POINT
    )
    if (
        not np.isfinite(interval_count_float)
        or interval_count_float + 1 > maximum_temporary_points
    ):
        estimated_mib = (
            (interval_count_float + 1)
            * _ESTIMATED_BYTES_PER_TEMPORARY_POINT
            / 1024**2
        )
        raise ValueError(
            "Variable resolving-power convolution is impractical: the "
            f"adaptive grid alone requires an estimated {estimated_mib:.1f} "
            "MiB working set, exceeding the "
            f"{_MAX_TEMPORARY_WORKING_BYTES / 1024**2:.0f} MiB safety limit."
        )

    interval_count = max(1, int(np.ceil(interval_count_float)))
    adaptive_resolution_coordinate = np.linspace(
        0.0,
        resolution_coordinate[-1],
        interval_count + 1,
    )
    adaptive_edges = np.interp(
        adaptive_resolution_coordinate,
        resolution_coordinate,
        metric_coordinate,
    )
    adaptive_centers = 0.5 * (adaptive_edges[:-1] + adaptive_edges[1:])

    coordinate_fwhm = 2.0 * np.arcsinh(
        1.0 / (2.0 * resolving_power_at_samples)
    )
    sigma = coordinate_fwhm * _FWHM_TO_SIGMA
    support = _VARIABLE_KERNEL_RADIUS_SIGMA * sigma
    starts = np.searchsorted(
        adaptive_edges,
        coordinate - support,
        side="right",
    ) - 1
    stops = np.searchsorted(
        adaptive_edges,
        coordinate + support,
        side="left",
    )
    starts = np.clip(starts, 0, adaptive_centers.size - 1)
    stops = np.clip(stops, 1, adaptive_centers.size)
    counts = stops - starts
    kernel_nonzeros = int(np.sum(counts, dtype=np.int64))
    remap_nonzeros = 2 * wave_ascending.size
    estimated_bytes = (
        kernel_nonzeros * _ESTIMATED_BYTES_PER_SPARSE_NONZERO
        + remap_nonzeros * _ESTIMATED_BYTES_PER_SPARSE_NONZERO
        + (
            wave_ascending.size
            + adaptive_edges.size
            + metric_coordinate.size
        )
        * _ESTIMATED_BYTES_PER_TEMPORARY_POINT
    )
    if (
        kernel_nonzeros > _MAX_VARIABLE_KERNEL_NONZEROS
        or estimated_bytes > _MAX_TEMPORARY_WORKING_BYTES
    ):
        raise ValueError(
            "Variable resolving-power convolution is impractical: the "
            f"estimated plan has {kernel_nonzeros:,} Gaussian weights and "
            f"requires {estimated_bytes / 1024**2:.1f} MiB, exceeding the "
            "work or memory safety limit. Supply a more compact spectrum or "
            "a more smoothly varying resolving-power curve."
        )

    from scipy.sparse import csc_matrix
    from scipy.special import ndtr

    indptr = np.empty(wave_ascending.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(kernel_nonzeros, dtype=np.int32)
    weights = np.empty(kernel_nonzeros, dtype=float)
    for source_index, (start, stop) in enumerate(zip(starts, stops)):
        destination_indices = np.arange(start, stop, dtype=np.int32)
        data_start = indptr[source_index]
        data_stop = indptr[source_index + 1]
        indices[data_start:data_stop] = destination_indices
        edge_offsets = (
            adaptive_edges[start : stop + 1] - coordinate[source_index]
        ) / sigma[source_index]
        source_weights = np.diff(ndtr(edge_offsets))
        source_weights /= np.sum(source_weights)
        weights[data_start:data_stop] = source_weights

    kernel_matrix = csc_matrix(
        (weights, indices, indptr),
        shape=(adaptive_centers.size, wave_ascending.size),
    )

    input_edges = np.empty(wave_ascending.size + 1)
    input_edges[0] = coordinate[0]
    input_edges[-1] = coordinate[-1]
    input_edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    input_widths = np.diff(input_edges)
    adaptive_widths = np.diff(adaptive_edges)
    output_matrix = _build_density_interpolation_matrix(
        adaptive_centers,
        adaptive_widths,
        coordinate,
        input_widths,
    )
    adaptive_coverage = np.asarray(output_matrix.sum(axis=0)).reshape(-1)
    source_coverage = np.asarray(adaptive_coverage @ kernel_matrix).reshape(-1)
    source_normalization = 1.0 / source_coverage

    return {
        "coordinate": coordinate,
        "input_widths": input_widths,
        "kernel_matrix": kernel_matrix,
        "logarithmic": True,
        "order": order,
        "output_matrix": output_matrix,
        "source_normalization": source_normalization,
        "variable": True,
        "wave_ascending": wave_ascending,
    }


def _build_density_interpolation_matrix(
    source_coordinate,
    source_widths,
    destination_coordinate,
    destination_widths,
):
    """Return a sparse linear interpolation from source to destination mass."""
    from scipy.sparse import csr_matrix

    upper_indices = np.searchsorted(
        source_coordinate,
        destination_coordinate,
        side="right",
    )
    interior = (upper_indices > 0) & (
        upper_indices < source_coordinate.size
    )
    row_counts = np.where(interior, 2, 1)
    indptr = np.empty(destination_coordinate.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    indices = np.empty(indptr[-1], dtype=np.int32)
    weights = np.empty(indptr[-1], dtype=float)

    row_starts = indptr[:-1]
    blue = upper_indices == 0
    red = upper_indices == source_coordinate.size
    indices[row_starts[blue]] = 0
    weights[row_starts[blue]] = (
        destination_widths[blue] / source_widths[0]
    )
    indices[row_starts[red]] = source_coordinate.size - 1
    weights[row_starts[red]] = (
        destination_widths[red] / source_widths[-1]
    )

    upper = upper_indices[interior]
    lower = upper - 1
    fractions = (
        destination_coordinate[interior] - source_coordinate[lower]
    ) / (source_coordinate[upper] - source_coordinate[lower])
    starts = row_starts[interior]
    indices[starts] = lower
    indices[starts + 1] = upper
    weights[starts] = (
        destination_widths[interior]
        * (1.0 - fractions)
        / source_widths[lower]
    )
    weights[starts + 1] = (
        destination_widths[interior]
        * fractions
        / source_widths[upper]
    )

    return csr_matrix(
        (weights, indices, indptr),
        shape=(destination_coordinate.size, source_coordinate.size),
    )


def _apply_gaussian_convolution_plan(flux, plan):
    """Apply a prepared Gaussian convolution plan to one flux vector."""
    from astropy.convolution import Gaussian1DKernel, convolve

    values = np.asarray(flux, dtype=float)
    if values.ndim != 1 or values.size != plan["order"].size:
        raise ValueError(
            "Flux must be one-dimensional and match the wavelength axis"
        )

    values_ascending = values[plan["order"]]
    if plan.get("variable", False):
        density = plan["wave_ascending"] * values_ascending
        source_mass = (
            density
            * plan["input_widths"]
            * plan["source_normalization"]
        )
        adaptive_mass = plan["kernel_matrix"] @ source_mass
        output_mass = plan["output_matrix"] @ adaptive_mass
        output_density = output_mass / plan["input_widths"]
        output_ascending = output_density / plan["wave_ascending"]
        output = np.empty_like(output_ascending)
        output[plan["order"]] = output_ascending
        return output

    if plan["logarithmic"]:
        # d(lambda) = lambda d(ln(lambda)), so lambda * F_lambda is the
        # appropriate density to convolve when preserving wavelength-integrated
        # flux in the continuous, unbounded-domain limit.
        density = plan["wave_ascending"] * values_ascending
    else:
        density = values_ascending

    uniform_density = np.interp(
        plan["uniform_coordinate"],
        plan["coordinate"],
        density,
    )

    kernel = Gaussian1DKernel(plan["sigma_pixels"])
    convolved_density = convolve(uniform_density, kernel, boundary="extend")
    output_density = np.interp(
        plan["coordinate"],
        plan["uniform_coordinate"],
        convolved_density,
        left=convolved_density[0],
        right=convolved_density[-1],
    )

    if plan["logarithmic"]:
        output_ascending = output_density / plan["wave_ascending"]
    else:
        output_ascending = output_density

    output = np.empty_like(output_ascending)
    output[plan["order"]] = output_ascending
    return output


def _validate_spectrum_convolution_state(spectrum):
    """Reject ancillary state that cannot be propagated scientifically."""
    if spectrum.mask is not None:
        raise NotImplementedError(
            "Spectral convolution does not yet support masked spectra"
        )
    if spectrum.uncertainty is not None:
        raise NotImplementedError(
            "Spectral convolution does not yet propagate uncertainties"
        )


class Spectrum(Spectrum1D):
    """
    A wrapper class for `~specutils.Spectrum1D` with extended functionality for
    working with stellar model spectra.

    This class adds capabilities to:
    - Load and interpolate spectra from various model grids
    - Resample spectra using `synphot` to conserve flux
    - Convolve spectra to lower resolution
    - Bin spectra into custom wavelength intervals

    Parameters
    ----------
    **kwargs : dict
        Arguments passed to the base `Spectrum1D` initializer.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_grid(
        self,
        teff,
        logg,
        feh=0,
        alpha=0.0,
        wavelength=None,
        wl_min=None,
        wl_max=None,
        model_grid="phoenix",
        interpolate=True,
        verbose=False,
        *,
        co_ratio=None,
    ):
        """
        Load a model spectrum from a library.

        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model. For SPHINX this selects the filename's
            ``logZ`` metallicity parameter; for MPS-ATLAS it is [M/H].

        alpha : float, optional
            Alpha enhancement for NewEra models. This selects a fixed model
            slice and is not an interpolation dimension.

        co_ratio : float, optional
            Carbon-to-oxygen ratio. Required when ``model_grid="sphinx"`` and
            ignored by other model grids.

        wavelength : `~astropy.units.Quantity`, optional
            Wavelengths of the interpolated spectrum.

        wl_min : `~astropy.units.Quantity`, optional
            Minimum wavelength of the model spectrum.

        wl_max : `~astropy.units.Quantity`, optional
            Maximum wavelength of the model spectrum.

        model_grid : str, optional
            Name of the model grid.

        verbose : bool, optional
            Print details for debugging.

        interpolate : bool, optional
            Whether to interpolate between grid points. If `True` (default), the spectrum
            will be trilinearly interpolated in (Teff, logg, [Fe/H]) space. If `False`,
            the nearest available grid point will be used without interpolation.

        Returns
        -------
        spec : `~speclib.Spectrum`
            A spectrum for the specified parameters.
        """
        # First check that the model_grid is valid.
        requested_model_grid = model_grid.lower()
        if requested_model_grid not in utils.VALID_MODELS:
            raise NotImplementedError(
                f'"{requested_model_grid}" model grid not found. '
                + "Currently supported models are: "
                + str(utils.VALID_MODELS)
            )
        mps_atlas_set = None
        if requested_model_grid in utils.MPS_ATLAS_SELECTORS:
            mps_atlas_set = utils.normalize_mps_atlas_set(requested_model_grid)
            self.model_grid = utils.canonical_mps_atlas_selector(mps_atlas_set)
        else:
            self.model_grid = requested_model_grid

        # Define grid points. SPHINX is sparse, so derive the selected C/O slice
        # from the exact filenames in the extracted V4 archive.
        sphinx_combinations = None
        mps_atlas_combinations = None
        if self.model_grid == "sphinx":
            self.grid_points, sphinx_combinations, co_ratio = _sphinx_grid_slice(
                co_ratio
            )
        elif mps_atlas_set is not None:
            _validate_mps_atlas_ranges(
                teff,
                logg,
                feh,
                utils.GRID_POINTS[self.model_grid],
                mps_atlas_set,
            )
            self.grid_points, mps_atlas_combinations = _mps_atlas_grid_slice(
                mps_atlas_set
            )
            _validate_mps_atlas_ranges(
                teff, logg, feh, self.grid_points, mps_atlas_set
            )
        else:
            self.grid_points = utils.GRID_POINTS[self.model_grid]
        self.grid_teffs = self.grid_points["grid_teffs"]
        self.grid_loggs = self.grid_points["grid_loggs"]
        self.grid_fehs = self.grid_points["grid_fehs"]

        if self.model_grid == "phoenix":
            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg/(s * cm^3)")
            cache_dir = utils.get_library_root() / "phoenix"
            cache_dir.mkdir(parents=True, exist_ok=True)

            ftp_url = "ftp://phoenix.astro.physik.uni-goettingen.de"
            fname_str = (
                "lte{:05.0f}-{:0.2f}{:+0.1f}."
                + "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
            )

            # The convention of the PHOENIX model grids is that
            # [Fe/H] = 0.0 is written as a negative number.
            if feh == 0:
                feh = -0.0

            # Load the wavelength array
            wave_local_path = cache_dir / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
            try:
                wave_lib = fits.getdata(wave_local_path)
            except FileNotFoundError:
                wave_remote_path = os.path.join(
                    ftp_url, "HiResFITS", "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
                )
                utils.download_file(wave_remote_path, wave_local_path, verbose)
                wave_lib = fits.getdata(wave_local_path)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not interpolate and not model_in_grid:
                teff = utils.nearest(self.grid_teffs, teff)
                logg = utils.nearest(self.grid_loggs, logg)
                feh = utils.nearest(self.grid_fehs, feh)
                model_in_grid = True  # force nearest model retrieval
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = utils.load_flux_array(
                                fname, cache_dir, ftp_url
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the flux array
                fname = fname_str.format(teff, logg, feh)
                flux = utils.load_flux_array(fname, cache_dir, ftp_url)

        elif self.model_grid == "newera":
            import h5py

            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg / (s * cm^3)")
            cache_dir = utils.get_library_root() / "newera"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Ensure feh is float-compatible with naming convention
            if feh == 0:
                feh = -0.0

            # Define bounds
            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            alpha_in_grid = alpha in self.grid_points.get("grid_alphas", [0.0])
            use_alpha = feh >= -2.0 and feh <= 0.0 and alpha_in_grid

            def load_flux_from_h5(teff_, logg_, feh_, alpha_=0.0):
                local_path = utils.download_newera_file(
                    teff_, logg_, feh_, alpha_, cache_dir=cache_dir
                )
                with h5py.File(local_path, "r") as h5:
                    # Wavelengths in vacuum Ångströms
                    wl = h5["PHOENIX_SPECTRUM/wl"][:]

                    # Flux is log10(F_lambda) in erg / (s cm² cm)
                    log_flux = h5["PHOENIX_SPECTRUM/flux"][:]
                    flux_per_cm = 10**log_flux

                return wl, flux_per_cm

            if not model_in_grid and not interpolate:
                teff = utils.nearest(self.grid_teffs, teff)
                logg = utils.nearest(self.grid_loggs, logg)
                feh = utils.nearest(self.grid_fehs, feh)
                model_in_grid = True

            if not model_in_grid:
                teff_bds = utils.find_bounds(self.grid_teffs, teff)
                logg_bds = utils.find_bounds(self.grid_loggs, logg)
                feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                wave_lib = None
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            wl, flx = load_flux_from_h5(tt, gg, ff, alpha)
                            flux_dict[tt][gg][ff] = flx
                            if wave_lib is None:
                                wave_lib = wl

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            else:
                wave_lib, flux = load_flux_from_h5(teff, logg, feh, alpha)

        elif self.model_grid in ["newera_gaia", "newera_jwst", "newera_lowres"]:
            grid_name = self.model_grid
            lib_wave_unit = u.nm
            lib_flux_unit = u.W / (u.m**2 * u.nm)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            alpha_in_grid = alpha in self.grid_points.get("grid_alphas", [0.0])

            def load_flux(teff_, logg_, feh_, alpha_=0.0):
                return utils.load_newera_flux_array(
                    teff_, logg_, feh_, alpha_, grid_name
                )

            def load_wave(teff_, logg_, feh_, alpha_=0.0):
                return utils.load_newera_wavelength_array(
                    teff_, logg_, feh_, alpha_, grid_name
                )

            if not model_in_grid and not interpolate:
                teff = utils.nearest(self.grid_teffs, teff)
                logg = utils.nearest(self.grid_loggs, logg)
                feh = utils.nearest(self.grid_fehs, feh)
                model_in_grid = True

            if not model_in_grid:
                teff_bds = utils.find_bounds(self.grid_teffs, teff)
                logg_bds = utils.find_bounds(self.grid_loggs, logg)
                feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                wave_lib = None
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            if wave_lib is None:
                                wave_lib = load_wave(tt, gg, ff, alpha)
                            flux_dict[tt][gg][ff] = load_flux(tt, gg, ff, alpha)

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            else:
                wave_lib = load_wave(teff, logg, feh, alpha)
                flux = load_flux(teff, logg, feh, alpha)

        elif self.model_grid == "drift-phoenix":
            # Only works if the user has already cached the DRIFT-PHOENIX model grid
            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg/(s * cm^2 * angstrom)")
            cache_dir = utils.get_library_root() / "drift-phoenix"
            cache_dir.mkdir(parents=True, exist_ok=True)

            fname_str = "lte_{:4.0f}_{:0.1f}{:+0.1f}.7.dat.txt"

            # The convention of the DRIFT-PHOENIX model grids is that
            # [Fe/H] = 0.0 is written as a negative number.
            if feh == 0:
                feh = -0.0

            # Load the wavelength array
            wave_local_path = cache_dir / "lte_1000_3.0-0.0.7.dat.txt"
            wave_lib = np.loadtxt(wave_local_path, unpack=True, usecols=0)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = np.loadtxt(
                                cache_dir / fname, unpack=True, usecols=1
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the wavelength and flux arrays
                fname = fname_str.format(teff, logg, feh)
                wave_lib, flux = np.loadtxt(cache_dir / fname, unpack=True)

        elif self.model_grid == "nextgen-solar":
            # Only works if the user has already cached the NextGen model grid
            lib_wave_unit = u.AA
            lib_flux_unit = u.Unit("erg/(s * cm^2 * angstrom)")
            cache_dir = utils.get_library_root() / "nextgen-solar"
            cache_dir.mkdir(parents=True, exist_ok=True)

            fname_str = "lte{:05.0f}_{:+0.1f}_{:+.1f}_NextGen-solar.dat"

            # Load the wavelength array
            wave_local_path = cache_dir / "lte01600_+5.5_+0.0_NextGen-solar.dat"
            wave_lib = np.loadtxt(wave_local_path, unpack=True, usecols=0)

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = utils.find_bounds(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = utils.find_bounds(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = utils.find_bounds(self.grid_fehs, feh)

                flux_dict = {}
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            fname = fname_str.format(tt, gg, ff)
                            flux_dict[tt][gg][ff] = np.loadtxt(
                                cache_dir / fname, unpack=True, usecols=1
                            )

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                # Load the wavelength and flux arrays
                fname = fname_str.format(teff, logg, feh)
                wave_lib, flux = np.loadtxt(cache_dir / fname, unpack=True)

        elif self.model_grid == "sphinx":
            lib_wave_unit = u.micron
            lib_flux_unit = u.Unit("W/(m^2 * m)")

            def load_wave_flux(teff_, logg_, metallicity_):
                wavelength_, flux_ = utils.load_sphinx_spectrum(
                    teff_, logg_, metallicity_, co_ratio
                )
                return (
                    wavelength_.to_value(lib_wave_unit),
                    flux_.to_value(lib_flux_unit),
                )

            teff_in_grid = teff in self.grid_teffs
            logg_in_grid = logg in self.grid_loggs
            feh_in_grid = feh in self.grid_fehs
            model_in_grid = all([teff_in_grid, logg_in_grid, feh_in_grid])
            requested = np.array([teff, logg, feh], dtype=float)
            exact_combinations = sphinx_combinations[:, :3]
            model_in_grid = model_in_grid and np.any(
                np.all(np.isclose(exact_combinations, requested), axis=1)
            )
            if not model_in_grid and not interpolate:
                nearest_index = _nearest_available_index(
                    exact_combinations, requested
                )
                teff, logg, feh = exact_combinations[nearest_index]
                model_in_grid = True
            if not model_in_grid:
                if teff_in_grid:
                    teff_bds = [teff, teff]
                else:
                    teff_bds = _flanking_values(self.grid_teffs, teff)
                if logg_in_grid:
                    logg_bds = [logg, logg]
                else:
                    logg_bds = _flanking_values(self.grid_loggs, logg)
                if feh_in_grid:
                    feh_bds = [feh, feh]
                else:
                    feh_bds = _flanking_values(self.grid_fehs, feh)

                flux_dict = {}
                wave_lib = None
                for tt in teff_bds:
                    flux_dict[tt] = {}
                    for gg in logg_bds:
                        flux_dict[tt][gg] = {}
                        for ff in feh_bds:
                            wavelength_, flux_ = load_wave_flux(tt, gg, ff)
                            if wave_lib is None:
                                wave_lib = wavelength_
                            elif not np.array_equal(wave_lib, wavelength_):
                                raise ValueError(
                                    "SPHINX spectra required for interpolation "
                                    "do not share a wavelength grid."
                                )
                            flux_dict[tt][gg][ff] = flux_

                flux = utils.trilinear_interpolate(
                    flux_dict, (teff_bds, logg_bds, feh_bds), (teff, logg, feh)
                )

            elif model_in_grid:
                wave_lib, flux = load_wave_flux(teff, logg, feh)

        elif mps_atlas_set is not None:
            lib_wave_unit = u.AA
            lib_flux_unit = u.erg / (u.s * u.cm**2 * u.AA)

            def load_wave_flux(teff_, logg_, metallicity_):
                wavelength_, flux_ = utils.load_mps_atlas_spectrum(
                    teff_, logg_, metallicity_, mps_atlas_set
                )
                return (
                    wavelength_.to_value(lib_wave_unit),
                    flux_.to_value(lib_flux_unit),
                )

            requested = np.array([teff, logg, feh], dtype=float)
            exact_combinations = mps_atlas_combinations[:, :3]
            model_in_grid = np.any(
                np.all(
                    np.isclose(
                        exact_combinations, requested, rtol=0.0, atol=1e-8
                    ),
                    axis=1,
                )
            )
            if not model_in_grid and not interpolate:
                nearest_index = _nearest_available_index(
                    exact_combinations, requested
                )
                teff, logg, feh = exact_combinations[nearest_index]
                model_in_grid = True

            if model_in_grid:
                wave_lib, flux = load_wave_flux(teff, logg, feh)
            else:
                teff_bds = _flanking_values(self.grid_teffs, teff)
                logg_bds = _flanking_values(self.grid_loggs, logg)
                feh_bds = _flanking_values(self.grid_fehs, feh)

                flux_dict = {}
                wave_lib = None
                try:
                    for tt in teff_bds:
                        flux_dict[tt] = {}
                        for gg in logg_bds:
                            flux_dict[tt][gg] = {}
                            for ff in feh_bds:
                                wavelength_, flux_ = load_wave_flux(tt, gg, ff)
                                if wave_lib is None:
                                    wave_lib = wavelength_
                                elif not np.array_equal(wave_lib, wavelength_):
                                    raise ValueError(
                                        "MPS-ATLAS spectra required for "
                                        "interpolation do not share a wavelength "
                                        "grid."
                                    )
                                flux_dict[tt][gg][ff] = flux_
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot interpolate this MPS-ATLAS {mps_atlas_set} "
                        "point because the selected set lacks a required "
                        f"corner model: {exc}"
                    ) from exc

                flux = utils.trilinear_interpolate(
                    flux_dict,
                    (teff_bds, logg_bds, feh_bds),
                    (teff, logg, feh),
                )

        # Load `~speclib.Spectrum` object
        spec = Spectrum(
            spectral_axis=wave_lib * lib_wave_unit,
            flux=flux * lib_flux_unit,
        )
        # Ensure spectra are ordered correctly.
        idx_order = np.argsort(spec.wavelength)
        spec = Spectrum(
            spectral_axis=spec.wavelength[idx_order], flux=spec.flux[idx_order]
        )
        # Change to default units
        default_wave_unit = u.AA
        default_flux_unit = u.Unit("erg/(s * cm^2 * angstrom)")
        spec = Spectrum(
            spectral_axis=spec.spectral_axis.to(default_wave_unit),
            flux=spec.flux.to(default_flux_unit),
        )

        # Crop to wavelength min and max, if given
        if not all(v is None for v in [wl_min, wl_max]):
            if wl_min is None:
                wl_min = spec.wavelength.min()
            if wl_max is None:
                wl_max = spec.wavelength.max()
            mask = np.logical_or(spec.wavelength <= wl_min, spec.wavelength >= wl_max)
            spec = Spectrum(spectral_axis=spec.wavelength[~mask], flux=spec.flux[~mask])

        # Resample the spectrum to the desired wavelength array
        if wavelength is not None:
            spec = spec.resample(wavelength)

        spec.model_grid = self.model_grid
        if mps_atlas_set is not None:
            spec.model_set = mps_atlas_set

        return spec

    @u.quantity_input(wavelength=u.AA)
    def resample(self, wavelength, taper=False):
        """
        Resample a spectrum while conserving flux.

        Parameters
        ----------
        wavelength : `~astropy.units.Quantity`
            A new wavelength axis. Unit must be specified.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
             A resampled spectrum.
        """
        if taper:
            force = "taper"
        else:
            force = None
        # Convert wavelengths arrays to same unit
        wave_old = self.wavelength.to(u.AA)
        wave_new = wavelength.to(u.AA)
        # waveunits = "angstrom"

        # The input value without a unit
        flux_old = self.flux.value

        # Make an observation object with synphot
        spectrum = sp.spectrum.SourceSpectrum(
            sp.models.Empirical1D, points=wave_old, lookup_table=flux_old
        )
        throughput = np.ones(len(wave_old)) * u.dimensionless_unscaled
        filt = sp.spectrum.SpectralElement(
            sp.models.Empirical1D,
            points=wave_old,
            lookup_table=throughput,
        )
        obs = sp.observation.Observation(spectrum, filt, binset=wave_new, force=force)

        # Save the new binned flux array in a `~speclib.Spectrum` object
        spec_new = Spectrum(
            spectral_axis=wavelength, flux=obs.binflux.value * self.flux.unit
        )

        return spec_new

    @u.quantity_input(delta_lambda=u.AA)
    def regularize(self, delta_lambda=None):
        """
        Resample a spectrum to a regularly spaced wavelength grid.

        Parameters
        ----------
        delta_lambda : `~astropy.units.Quantity`, optional
            The spacing of the new wavelength grid. Defaults to the smallest
            spacing in the orignal grid.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
            A resampled spectrum.
        """
        wl_min = self.wavelength.min()
        wl_max = self.wavelength.max()
        if delta_lambda is None:
            delta_lambda = np.diff(self.wavelength).min()
        n_points = int((wl_max.value - wl_min.value) / delta_lambda.value)
        regular_grid = (
            np.linspace(wl_min.value, wl_max.value, n_points) * delta_lambda.unit
        )
        spec_new = self.resample(regular_grid)

        return spec_new

    def set_spectral_resolution(self, delta_lambda):
        """
        Return a spectrum with constant Gaussian wavelength resolution.

        Parameters
        ----------
        delta_lambda : `~astropy.units.Quantity`
            Positive scalar Gaussian FWHM with wavelength units.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
            A new spectrum on the original spectral axis. The input spectrum is
            not modified.

        Notes
        -----
        The intrinsic input line width is assumed to be negligible relative to
        ``delta_lambda``. No deconvolution or correction for finite input
        resolution is performed. The input must provide at least two samples
        per requested FWHM across every wavelength interval. Internally, the
        temporary grid has at least ten samples per FWHM and is no coarser than
        the finest input interval.

        Features within four Gaussian standard deviations of a boundary can
        lose flux or have a truncated profile because no spectrum is invented
        outside the observed range. Masked spectra and spectra with uncertainty
        arrays are rejected because propagating that state is outside the scope
        of this operation.
        """
        _validate_spectrum_convolution_state(self)
        delta_lambda_value = _validate_delta_lambda(delta_lambda)
        plan = _build_gaussian_convolution_plan(
            self.wavelength,
            delta_lambda=delta_lambda_value,
        )
        convolved_flux = _apply_gaussian_convolution_plan(self.flux.value, plan)
        return Spectrum(
            spectral_axis=self.spectral_axis.copy(),
            flux=convolved_flux * self.flux.unit,
            meta=copy.deepcopy(self.meta),
        )

    def set_spectral_resolving_power(self, resolving_power):
        """Return a spectrum broadened to constant resolving power.

        Parameters
        ----------
        resolving_power : float or `~astropy.units.Quantity`
            Positive scalar resolving power, :math:`R = \\lambda/\\Delta\\lambda`.
            A quantity must be dimensionless.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
            A new spectrum on the original spectral axis. The input spectrum is
            not modified.

        Notes
        -----
        Convolution is evaluated on a uniform log-wavelength grid. To use the
        physical wavelength-flux measure, ``lambda * F_lambda`` is convolved
        and the result is divided by wavelength afterward. This preserves
        wavelength-integrated flux in the continuous, unbounded-domain limit;
        finite boundaries and numerical remapping limit exact conservation.

        The stationary Gaussian therefore applies to flux per logarithmic
        wavelength interval. Expressed as ``F_lambda``, an isolated feature has
        a log-normal-like profile that is not exactly symmetric in linear
        wavelength. This distinction is small at ordinary astronomical
        resolving powers. The input must provide at least two samples per
        requested log-wavelength FWHM across every interval.

        The intrinsic input line width is assumed to be negligible relative to
        the requested broadening. Wavelength-dependent resolving power is not
        supported. Features within four Gaussian standard deviations of a
        boundary do not have the same flux or resolution guarantees as interior
        features. Masked spectra and spectra with uncertainty arrays are
        rejected.
        """
        _validate_spectrum_convolution_state(self)
        resolving_power_value = _validate_resolving_power(resolving_power)
        plan = _build_gaussian_convolution_plan(
            self.wavelength,
            resolving_power=resolving_power_value,
        )
        convolved_flux = _apply_gaussian_convolution_plan(self.flux.value, plan)
        return Spectrum(
            spectral_axis=self.spectral_axis.copy(),
            flux=convolved_flux * self.flux.unit,
            meta=copy.deepcopy(self.meta),
        )

    def set_variable_resolving_power(self, wavelength, resolving_power):
        """Return a spectrum broadened to wavelength-dependent resolving power.

        Parameters
        ----------
        wavelength : `~astropy.units.Quantity`
            Strictly monotonic wavelengths sampling the resolving-power curve.
            The curve must cover the full wavelength range of the spectrum.
        resolving_power : array-like or `~astropy.units.Quantity`
            Positive dimensionless resolving power values corresponding to
            ``wavelength``.

        Returns
        -------
        spec_new : `~speclib.Spectrum`
            A new spectrum on the original spectral axis. The input spectrum is
            not modified.

        Notes
        -----
        Resolving power is linearly interpolated between the supplied samples;
        extrapolation is not supported. The curve is intended to vary smoothly
        relative to the local resolution element. Each input wavelength is
        broadened by a source-centered Gaussian using the wavelength-coordinate
        and flux conventions of :meth:`set_spectral_resolving_power`. The
        intrinsic input line width is assumed negligible, and the output
        wavelength sampling is unchanged. Operations whose estimated sparse
        plan exceeds the work or memory safety limit are rejected.
        """
        _validate_spectrum_convolution_state(self)
        curve_wavelength, curve_resolving_power = (
            _validate_resolving_power_curve(
                wavelength,
                resolving_power,
                self.wavelength,
            )
        )
        plan = _build_variable_resolving_power_plan(
            self.wavelength,
            curve_wavelength,
            curve_resolving_power,
        )
        convolved_flux = _apply_gaussian_convolution_plan(self.flux.value, plan)
        return Spectrum(
            spectral_axis=self.spectral_axis.copy(),
            flux=convolved_flux * self.flux.unit,
            meta=copy.deepcopy(self.meta),
        )

    @u.quantity_input(center=u.AA, width=u.AA)
    def bin(self, center, width):
        """
        Bin a model spectrum within specified wavelength bins.

        Parameters
        ----------
        center : `~astropy.units.Quantity`
            The centers of the wavelength bins.

        width : `~astropy.units.Quantity`
            The widths of the wavelength bins.

        Returns
        -------
        `~speclib.BinnedSpectrum`

        """
        wavelength = self.wavelength
        flux = self.flux
        binned_fluxes = []
        for cen, wid in zip(center, width):
            lower = cen - wid / 2.0
            upper = cen + wid / 2.0
            idx = np.where((wavelength >= lower) & (wavelength <= upper))

            # Adjust for bins that are slightly wider than the wavelength range
            # due to discretization of the wavelength grid
            scale_factor = (upper - lower) / (wavelength[idx][-1] - wavelength[idx][0])

            binned_flux = (
                scale_factor
                * np.trapezoid(flux[idx], wavelength[idx])
                / (upper - lower)
            )
            binned_fluxes.append(binned_flux)
        binned_fluxes = u.Quantity(binned_fluxes)

        return BinnedSpectrum(center, width, binned_fluxes)


class BinnedSpectrum(object):
    """
    Represents a spectrum that has been binned into specified wavelength intervals.

    Useful for simulating photometric measurements or spectral channels.

    Attributes
    ----------
    center : `~astropy.units.Quantity`
        The centers of the wavelength bins.

    width : `~astropy.units.Quantity`
        The widths of the wavelength bins.

    lower : `~astropy.units.Quantity`
        The lower bounds of the wavelength bins.

    upper : `~astropy.units.Quantity`
        The upper bounds of the wavelength bins.

    flux : `~astropy.units.Quantity`
        The binned flux array.
    """

    @u.quantity_input(center=u.AA, width=u.AA)
    def __init__(self, center, width, flux):
        """
        Parameters
        ----------
        center : `~astropy.units.Quantity`
            The centers of the wavelength bins.

        width : `~astropy.units.Quantity`
            The widths of the wavelength bins.

        flux : iterable
            The binned flux array.
        """
        self.center = center
        self.width = width
        self.lower = center - width / 2.0
        self.upper = center + width / 2.0
        self.flux = flux


class SpectralGrid(object):
    """
    Represents a multi-dimensional grid of synthetic spectra from a model library.

    Provides fast access to preloaded spectra and supports trilinear interpolation
    in (Teff, logg, [Fe/H]) space.

    Attributes
    ----------
    teff_bds : iterable
        The lower and upper bounds of the model temperatures to load.

    logg_bds : iterable
        The lower and upper bounds of the model logg values to load.

    feh_bds : iterable
        The lower and upper bounds of the model [Fe/H] to load.

    wavelength : `~astropy.units.Quantity`
        Wavelengths of the interpolated spectrum.

    fluxes : dict
        The fluxes of the model grid. Sorted by fluxes[teff][logg][feh].

    model_grid : str
        Accepted model selector. PHOENIX, SPHINX, and the reduced NewEra
        products are the primary documented families; compatibility selectors
        have additional cache requirements.

    """

    def _clip_bounds_to_grid(
        self,
        bounds,
        grid_values: np.ndarray,
        param_name: str,
    ) -> tuple[float, float]:
        """Clip the requested bounds to the available grid range.

        Parameters
        ----------
        bounds : iterable
            Two-element sequence specifying the requested lower and upper bound.
        grid_values : `numpy.ndarray`
            The discrete grid values available for the parameter.
        param_name : str
            Name of the parameter, used in warning messages.

        Returns
        -------
        tuple of float
            The bounds snapped to valid grid values.
        """

        values = np.asarray(bounds, dtype=float)
        if values.shape != (2,):
            raise ValueError(f"{param_name} must contain exactly two bounds.")

        lower = float(values[0])
        upper = float(values[1])
        if lower > upper:
            raise ValueError(
                f"{param_name} lower bound {lower} exceeds upper bound {upper}."
            )

        grid_min = float(np.min(grid_values))
        grid_max = float(np.max(grid_values))

        clipped_lower = float(np.clip(lower, grid_min, grid_max))
        clipped_upper = float(np.clip(upper, grid_min, grid_max))

        lower_candidates = grid_values[grid_values <= clipped_lower]
        if lower_candidates.size:
            aligned_lower = float(lower_candidates.max())
        else:
            aligned_lower = grid_min

        upper_candidates = grid_values[grid_values >= clipped_upper]
        if upper_candidates.size:
            aligned_upper = float(upper_candidates.min())
        else:
            aligned_upper = grid_max

        clipped_bounds = (aligned_lower, aligned_upper)

        if clipped_lower != lower or clipped_upper != upper:
            warnings.warn(
                f"{param_name} {(lower, upper)} truncated to valid range {clipped_bounds}",
                UserWarning,
            )

        return clipped_bounds

    def __init__(
        self,
        teff_bds,
        logg_bds,
        feh_bds,
        wavelength=None,
        spectral_resolution=None,
        model_grid="phoenix",
        spectral_resolving_power=None,
        co_ratio=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        teff_bds : iterable
            The lower and upper bounds of the model temperatures to load.

        logg_bds : iterable
            The lower and upper bounds of the model logg values to load.

        feh_bds : iterable
            The lower and upper bounds of the model [Fe/H] to load.

        wavelength : `~astropy.units.Quantity`, optional
            Wavelengths of the interpolated spectrum.

        spectral_resolution : `~astropy.units.Quantity`
            Constant Gaussian FWHM in wavelength to apply to every spectrum.

        spectral_resolving_power : float or `~astropy.units.Quantity`, optional
            Constant resolving power to apply to every spectrum.

        model_grid : str, optional
            Name of the model grid.

        co_ratio : float, optional
            Fixed carbon-to-oxygen ratio for a SPHINX grid instance. Required
            for ``model_grid="sphinx"`` and ignored for other grids.
        """
        if (
            spectral_resolution is not None
            and spectral_resolving_power is not None
        ):
            raise ValueError(
                "Specify only one of spectral_resolution or "
                "spectral_resolving_power."
            )
        if spectral_resolution is not None:
            _validate_delta_lambda(spectral_resolution)
        if spectral_resolving_power is not None:
            _validate_resolving_power(spectral_resolving_power)

        # First check that the model_grid is valid.
        requested_model_grid = model_grid.lower()
        if requested_model_grid not in utils.VALID_MODELS:
            raise NotImplementedError(
                f'"{requested_model_grid}" model grid not found. '
                + "Currently supported models are: "
                + str(utils.VALID_MODELS)
            )

        self.co_ratio = co_ratio
        self.model_set = None
        mps_atlas_combinations = None
        if requested_model_grid in utils.MPS_ATLAS_SELECTORS:
            self.model_set = utils.normalize_mps_atlas_set(requested_model_grid)
            self.model_grid = utils.canonical_mps_atlas_selector(self.model_set)
        else:
            self.model_grid = requested_model_grid
        if self.model_grid == "sphinx":
            self.grid_points, _, self.co_ratio = _sphinx_grid_slice(co_ratio)
        elif self.model_set is not None:
            self.grid_points, mps_atlas_combinations = _mps_atlas_grid_slice(
                self.model_set
            )
        else:
            self.grid_points = utils.GRID_POINTS[self.model_grid]
        self.grid_teffs = self.grid_points["grid_teffs"]
        self.grid_loggs = self.grid_points["grid_loggs"]
        self.grid_fehs = self.grid_points["grid_fehs"]

        # Then ensure that the bounds given are valid.
        self.teff_bds = self._clip_bounds_to_grid(teff_bds, self.grid_teffs, "teff_bds")
        self.logg_bds = self._clip_bounds_to_grid(logg_bds, self.grid_loggs, "logg_bds")
        self.feh_bds = self._clip_bounds_to_grid(feh_bds, self.grid_fehs, "feh_bds")

        # Define the values covered in the grid
        subset = np.logical_and(
            self.grid_teffs >= self.teff_bds[0], self.grid_teffs <= self.teff_bds[1]
        )
        self.teffs = self.grid_teffs[subset]

        subset = np.logical_and(
            self.grid_loggs >= self.logg_bds[0], self.grid_loggs <= self.logg_bds[1]
        )
        self.loggs = self.grid_loggs[subset]

        subset = np.logical_and(
            self.grid_fehs >= self.feh_bds[0], self.grid_fehs <= self.feh_bds[1]
        )
        self.fehs = self.grid_fehs[subset]

        # Load the fluxes
        fluxes = {}
        points = []
        data = []
        spec = None
        spectrum_kwargs = dict(kwargs)
        if self.model_grid == "sphinx":
            spectrum_kwargs["co_ratio"] = self.co_ratio
        if mps_atlas_combinations is not None:
            within_bounds = (
                (mps_atlas_combinations[:, 0] >= self.teff_bds[0])
                & (mps_atlas_combinations[:, 0] <= self.teff_bds[1])
                & (mps_atlas_combinations[:, 1] >= self.logg_bds[0])
                & (mps_atlas_combinations[:, 1] <= self.logg_bds[1])
                & (mps_atlas_combinations[:, 2] >= self.feh_bds[0])
                & (mps_atlas_combinations[:, 2] <= self.feh_bds[1])
            )
            parameter_combinations = mps_atlas_combinations[within_bounds, :3]
        else:
            parameter_combinations = np.array(
                [
                    (teff, logg, feh)
                    for teff in self.teffs
                    for logg in self.loggs
                    for feh in self.fehs
                ]
            )

        for teff, logg, feh in parameter_combinations:
            fluxes.setdefault(teff, {}).setdefault(logg, {})
            try:
                spec = Spectrum.from_grid(
                    teff,
                    logg,
                    feh,
                    model_grid=self.model_grid,
                    **spectrum_kwargs,
                )
            except ValueError:
                if self.model_set is not None:
                    raise
                # Skip combinations that do not exist in sparse grids.
                continue

            # Resample the spectrum to the desired wavelength array
            if wavelength is not None:
                spec = spec.resample(wavelength)

            fluxes[teff][logg][feh] = spec.flux
            points.append([teff, logg, feh])
            data.append(spec.flux.value)

        self.fluxes = fluxes

        if spec is not None:
            self.wavelength = spec.wavelength
            self.unit = spec.flux.unit
        else:
            self.wavelength = None
            self.unit = u.dimensionless_unscaled

        if points:
            self.points = np.array(points)
            self.data = np.vstack(data)
            self.interpolator = NearestNDInterpolator(self.points, self.data)
        else:
            self.points = np.empty((0, 3))
            self.data = np.empty((0,))
            self.interpolator = None

        if spectral_resolution is not None:
            resolved_grid = self.set_spectral_resolution(spectral_resolution)
            self.fluxes = resolved_grid.fluxes
            self.data = resolved_grid.data
            self.interpolator = resolved_grid.interpolator
        elif spectral_resolving_power is not None:
            resolved_grid = self.set_spectral_resolving_power(
                spectral_resolving_power
            )
            self.fluxes = resolved_grid.fluxes
            self.data = resolved_grid.data
            self.interpolator = resolved_grid.interpolator

    def _with_gaussian_convolution(self, plan):
        """Return an in-memory copy with a convolution plan applied."""
        if self.wavelength is None or not self.points.size:
            raise ValueError("SpectralGrid contains no spectra")

        new_fluxes = {
            teff: {
                logg: dict(fluxes_by_feh)
                for logg, fluxes_by_feh in fluxes_by_logg.items()
            }
            for teff, fluxes_by_logg in self.fluxes.items()
        }
        new_rows = []
        for teff, logg, feh in self.points:
            flux = self.fluxes[teff][logg][feh].to(self.unit)
            convolved_values = _apply_gaussian_convolution_plan(flux.value, plan)
            convolved_flux = convolved_values * self.unit
            new_fluxes[teff][logg][feh] = convolved_flux
            new_rows.append(convolved_values)

        new_grid = copy.copy(self)
        for attribute in (
            "wavelength",
            "points",
            "teffs",
            "loggs",
            "fehs",
            "grid_teffs",
            "grid_loggs",
            "grid_fehs",
        ):
            if hasattr(self, attribute):
                value = getattr(self, attribute)
                setattr(
                    new_grid,
                    attribute,
                    value.copy() if hasattr(value, "copy") else copy.copy(value),
                )
        if hasattr(self, "grid_points"):
            new_grid.grid_points = copy.deepcopy(self.grid_points)
        new_grid.fluxes = new_fluxes
        new_grid.data = np.vstack(new_rows)
        new_grid.interpolator = NearestNDInterpolator(
            new_grid.points, new_grid.data
        )
        return new_grid

    def set_spectral_resolution(self, delta_lambda):
        """Return a new grid with constant Gaussian wavelength resolution.

        Parameters
        ----------
        delta_lambda : `~astropy.units.Quantity`
            Positive scalar Gaussian FWHM with wavelength units.

        Returns
        -------
        grid_new : `~speclib.SpectralGrid`
            A new grid containing the convolved spectra.

        Notes
        -----
        Every stored spectrum is convolved to the requested wavelength FWHM.
        The original grid and its wavelength sampling are unchanged.
        Intrinsic input line widths are assumed to be negligible. Sampling and
        boundary behavior match :meth:`Spectrum.set_spectral_resolution`.
        """
        if self.wavelength is None or not self.points.size:
            raise ValueError("SpectralGrid contains no spectra")
        delta_lambda_value = _validate_delta_lambda(delta_lambda)
        plan = _build_gaussian_convolution_plan(
            self.wavelength,
            delta_lambda=delta_lambda_value,
        )
        return self._with_gaussian_convolution(plan)

    def set_spectral_resolving_power(self, resolving_power):
        """Return a new grid with constant resolving power.

        Parameters
        ----------
        resolving_power : float or `~astropy.units.Quantity`
            Positive scalar dimensionless resolving power.

        Returns
        -------
        grid_new : `~speclib.SpectralGrid`
            A new grid containing the convolved spectra.

        Notes
        -----
        Every stored spectrum is convolved on the same internal log-wavelength
        grid. The original grid and its wavelength sampling are unchanged.
        Intrinsic input line widths are assumed to be negligible. Flux,
        sampling, and boundary conventions match
        :meth:`Spectrum.set_spectral_resolving_power`.
        """
        if self.wavelength is None or not self.points.size:
            raise ValueError("SpectralGrid contains no spectra")
        resolving_power_value = _validate_resolving_power(resolving_power)
        plan = _build_gaussian_convolution_plan(
            self.wavelength,
            resolving_power=resolving_power_value,
        )
        return self._with_gaussian_convolution(plan)

    def set_variable_resolving_power(self, wavelength, resolving_power):
        """Return a new grid with wavelength-dependent resolving power.

        Parameters
        ----------
        wavelength : `~astropy.units.Quantity`
            Strictly monotonic wavelengths sampling the resolving-power curve.
            The curve must cover the full grid wavelength range.
        resolving_power : array-like or `~astropy.units.Quantity`
            Positive dimensionless resolving power values corresponding to
            ``wavelength``.

        Returns
        -------
        grid_new : `~speclib.SpectralGrid`
            A new grid containing the convolved spectra.

        Notes
        -----
        The same variable-width convolution plan is reused for every stored
        spectrum. The original grid and its wavelength sampling are unchanged.
        Other conventions and limitations match
        :meth:`Spectrum.set_variable_resolving_power`.
        """
        if self.wavelength is None or not self.points.size:
            raise ValueError("SpectralGrid contains no spectra")
        curve_wavelength, curve_resolving_power = (
            _validate_resolving_power_curve(
                wavelength,
                resolving_power,
                self.wavelength,
            )
        )
        plan = _build_variable_resolving_power_plan(
            self.wavelength,
            curve_wavelength,
            curve_resolving_power,
        )
        return self._with_gaussian_convolution(plan)

    def get_flux(self, teff, logg, feh, interpolate=True):
        """
        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model.

        interpolate : bool, optional
            Whether to interpolate between grid points. If `True` (default), the spectrum
            will be trilinearly interpolated in (Teff, logg, [Fe/H]) space. If `False`,
            the nearest available grid point will be used without interpolation.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            The interpolated flux array as a 1-D vector aligned to ``self.wavelength``.
        """

        # First check that the values are within the grid
        teff_in_grid = self.teff_bds[0] <= teff <= self.teff_bds[1]
        logg_in_grid = self.logg_bds[0] <= logg <= self.logg_bds[1]
        feh_in_grid = self.feh_bds[0] <= feh <= self.feh_bds[1]

        booleans = [teff_in_grid, logg_in_grid, feh_in_grid]
        params = ["teff", "logg", "feh"]
        inputs = [teff, logg, feh]
        ranges = [self.teff_bds, self.logg_bds, self.feh_bds]

        if not all(booleans):
            message = "Input values are out of grid range.\n\n"
            for b, p, i, r in zip(booleans, params, inputs, ranges):
                if not b:
                    message += f"\tInput {p}: {i}. Valid range: {r}\n"
            raise ValueError(message)

        if self.model_grid in [
            "newera_gaia",
            "newera_jwst",
            "newera_lowres",
            "sphinx",
            "mps-atlas-set1",
            "mps-atlas-set2",
        ]:
            if self.interpolator is None or not self.points.size:
                raise ValueError("SpectralGrid contains no spectra")

            if not interpolate:
                if self.model_grid in {
                    "sphinx",
                    "mps-atlas-set1",
                    "mps-atlas-set2",
                }:
                    nearest_index = _nearest_available_index(
                        self.points, (teff, logg, feh)
                    )
                    flux = self.data[nearest_index]
                else:
                    flux = self.interpolator((teff, logg, feh))
            else:
                try:
                    flux = utils.trilinear_interpolate(
                        self.fluxes,
                        (self.teffs, self.loggs, self.fehs),
                        (teff, logg, feh),
                    )
                except KeyError:
                    if self.model_grid == "sphinx":
                        raise ValueError(
                            "Cannot interpolate this SPHINX point because the "
                            "selected C/O slice lacks a required corner model."
                        ) from None
                    if self.model_set is not None:
                        raise ValueError(
                            f"Cannot interpolate this MPS-ATLAS {self.model_set} "
                            "point because the selected set lacks a required "
                            "corner model."
                        ) from None
                    # Fall back to nearest-neighbour evaluation for sparse grids
                    flux = self.interpolator((teff, logg, feh))

            if not isinstance(flux, u.Quantity):
                flux = u.Quantity(flux, unit=self.unit, copy=False)
            else:
                flux = flux.to(self.unit)

            if flux.ndim > 1:
                # Some interpolators (e.g., nearest-neighbour fallbacks) can return
                # fluxes with a leading singleton dimension; flatten to a 1-D vector.
                flux = flux.reshape(-1)

            if flux.ndim != 1:
                raise ValueError(
                    "Interpolated flux has unexpected shape; expected a 1-D array"
                )

            return flux

        # If not interpolating, then just return the closest point in the grid.
        if not interpolate:
            teff = utils.nearest(self.teffs, teff)
            logg = utils.nearest(self.loggs, logg)
            feh = utils.nearest(self.fehs, feh)

            return self.fluxes[teff][logg][feh]

        # Otherwise, interpolate using the helper
        return utils.trilinear_interpolate(
            self.fluxes,
            (self.teffs, self.loggs, self.fehs),
            (teff, logg, feh),
        )

    def get_spectrum(self, teff, logg, feh, interpolate=True):
        """Deprecated alias for :meth:`get_flux`.

        .. deprecated:: 0.1.0
            Use :meth:`get_flux` instead.
        """

        warnings.warn(
            "SpectralGrid.get_spectrum is deprecated and will be removed in a "
            "future release. Use SpectralGrid.get_flux instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.get_flux(teff, logg, feh, interpolate=interpolate)


class BinnedSpectralGrid(object):
    """
    Represents a multi-dimensional grid of binned spectra from a model library.

    Supports trilinear interpolation over the parameter space.

    Attributes
    ----------
    teff_bds : iterable
        The lower and upper bounds of the model temperatures to load.

    logg_bds : iterable
        The lower and upper bounds of the model logg values to load.

    feh_bds : iterable
        The lower and upper bounds of the model [Fe/H] to load.

    center : `~astropy.units.Quantity`
        The centers of the wavelength bins.

    width : `~astropy.units.Quantity`
        The widths of the wavelength bins.

    lower : `~astropy.units.Quantity`
        The lower bounds of the wavelength bins.

    upper : `~astropy.units.Quantity`
        The upper bounds of the wavelength bins.

    fluxes : dict
        The fluxes of the model grid. Sorted by fluxes[teff][logg][feh].

    model_grid : str
        Accepted model selector. Support and cache requirements vary by model
        family.

    """

    def __init__(
        self, teff_bds, logg_bds, feh_bds, center, width, model_grid="phoenix", **kwargs
    ):
        """
        Parameters
        ----------
        teff_bds : iterable
            The lower and upper bounds of the model temperatures to load.

        logg_bds : iterable
            The lower and upper bounds of the model logg values to load.

        feh_bds : iterable
            The lower and upper bounds of the model [Fe/H] to load.

        center : `~astropy.units.Quantity`
            The centers of the wavelength bins.

        width : `~astropy.units.Quantity`
            The widths of the wavelength bins.

        model_grid : str, optional
            Accepted model selector. Support and cache requirements vary by
            model family.
        """
        # First check that the model_grid is valid.
        requested_model_grid = model_grid.lower()
        if requested_model_grid not in utils.VALID_MODELS:
            raise NotImplementedError(
                f'"{requested_model_grid}" model grid not found. '
                + "Currently supported models are: "
                + str(utils.VALID_MODELS)
            )
        self.model_set = None
        if requested_model_grid in utils.MPS_ATLAS_SELECTORS:
            self.model_set = utils.normalize_mps_atlas_set(requested_model_grid)
            self.model_grid = utils.canonical_mps_atlas_selector(self.model_set)
        else:
            self.model_grid = requested_model_grid

        # Define grid points
        sphinx_combinations = None
        mps_atlas_combinations = None
        if self.model_grid == "sphinx":
            co_ratio = kwargs.get("co_ratio")
            (
                self.grid_points,
                sphinx_combinations,
                self.co_ratio,
            ) = _sphinx_grid_slice(co_ratio)
            kwargs["co_ratio"] = self.co_ratio
        elif self.model_set is not None:
            self.grid_points, mps_atlas_combinations = _mps_atlas_grid_slice(
                self.model_set
            )
        else:
            self.grid_points = utils.GRID_POINTS[self.model_grid]
        self.grid_teffs = self.grid_points["grid_teffs"]
        self.grid_loggs = self.grid_points["grid_loggs"]
        self.grid_fehs = self.grid_points["grid_fehs"]

        # Then ensure that the bounds given are valid.
        teff_bds = np.array(teff_bds)
        teff_bds = (
            self.grid_teffs[self.grid_teffs <= teff_bds.min()].max(),
            self.grid_teffs[self.grid_teffs >= teff_bds.max()].min(),
        )
        self.teff_bds = teff_bds

        logg_bds = np.array(logg_bds)
        logg_bds = (
            self.grid_loggs[self.grid_loggs <= logg_bds.min()].max(),
            self.grid_loggs[self.grid_loggs >= logg_bds.max()].min(),
        )
        self.logg_bds = logg_bds

        feh_bds = np.array(feh_bds)
        feh_bds = (
            self.grid_fehs[self.grid_fehs <= feh_bds.min()].max(),
            self.grid_fehs[self.grid_fehs >= feh_bds.max()].min(),
        )
        self.feh_bds = feh_bds

        # Define the values covered in the grid
        subset = np.logical_and(
            self.grid_teffs >= self.teff_bds[0], self.grid_teffs <= self.teff_bds[1]
        )
        self.teffs = self.grid_teffs[subset]

        subset = np.logical_and(
            self.grid_loggs >= self.logg_bds[0], self.grid_loggs <= self.logg_bds[1]
        )
        self.loggs = self.grid_loggs[subset]

        subset = np.logical_and(
            self.grid_fehs >= self.feh_bds[0], self.grid_fehs <= self.feh_bds[1]
        )
        self.fehs = self.grid_fehs[subset]

        # Load the fluxes
        self.center = center
        self.width = width
        self.lower = center - width / 2.0
        self.upper = center + width / 2.0

        fluxes = {teff: {logg: {} for logg in self.loggs} for teff in self.teffs}
        sparse_combinations = (
            sphinx_combinations
            if self.model_grid == "sphinx"
            else mps_atlas_combinations
        )
        if sparse_combinations is not None:
            within_bounds = (
                (sparse_combinations[:, 0] >= self.teff_bds[0])
                & (sparse_combinations[:, 0] <= self.teff_bds[1])
                & (sparse_combinations[:, 1] >= self.logg_bds[0])
                & (sparse_combinations[:, 1] <= self.logg_bds[1])
                & (sparse_combinations[:, 2] >= self.feh_bds[0])
                & (sparse_combinations[:, 2] <= self.feh_bds[1])
            )
            self.points = sparse_combinations[within_bounds, :3]
            for teff, logg, feh in self.points:
                bs = Spectrum.from_grid(
                    teff, logg, feh, model_grid=self.model_grid, **kwargs
                ).bin(center, width)
                fluxes[teff][logg][feh] = bs.flux
        else:
            for teff in self.teffs:
                for logg in self.loggs:
                    for feh in self.fehs:
                        bs = Spectrum.from_grid(
                            teff, logg, feh, model_grid=self.model_grid, **kwargs
                        ).bin(center, width)
                        fluxes[teff][logg][feh] = bs.flux
        self.fluxes = fluxes

    def get_spectrum(self, teff, logg, feh, interpolate=True):
        """
        Parameters
        ----------
        teff : float
            Effective temperature of the model in Kelvin.

        logg : float
            Surface gravity of the model in cgs units.

        feh : float
            [Fe/H] of the model.

        interpolate : bool, optional
            Whether to interpolate between grid points. If `True` (default), the spectrum
            will be trilinearly interpolated in (Teff, logg, [Fe/H]) space. If `False`,
            the nearest available grid point will be used without interpolation.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            The interpolated flux array.
        """

        # First check that the values are within the grid
        teff_in_grid = self.teff_bds[0] <= teff <= self.teff_bds[1]
        logg_in_grid = self.logg_bds[0] <= logg <= self.logg_bds[1]
        feh_in_grid = self.feh_bds[0] <= feh <= self.feh_bds[1]

        booleans = [teff_in_grid, logg_in_grid, feh_in_grid]
        params = ["teff", "logg", "feh"]
        inputs = [teff, logg, feh]
        ranges = [self.teff_bds, self.logg_bds, self.feh_bds]

        if not all(booleans):
            message = "Input values are out of grid range.\n\n"
            for b, p, i, r in zip(booleans, params, inputs, ranges):
                if not b:
                    message += f"\tInput {p}: {i}. Valid range: {r}\n"
            raise ValueError(message)

        if self.model_grid == "sphinx" or self.model_set is not None:
            if not self.points.size:
                raise ValueError("BinnedSpectralGrid contains no spectra")
            if not interpolate:
                nearest_index = _nearest_available_index(
                    self.points, (teff, logg, feh)
                )
                nearest_teff, nearest_logg, nearest_feh = self.points[
                    nearest_index
                ]
                return self.fluxes[nearest_teff][nearest_logg][nearest_feh]
            try:
                return utils.trilinear_interpolate(
                    self.fluxes,
                    (self.teffs, self.loggs, self.fehs),
                    (teff, logg, feh),
                )
            except KeyError:
                if self.model_set is not None:
                    raise ValueError(
                        f"Cannot interpolate this MPS-ATLAS {self.model_set} "
                        "point because the selected set lacks a required corner "
                        "model."
                    ) from None
                raise ValueError(
                    "Cannot interpolate this SPHINX point because the selected "
                    "C/O slice lacks a required corner model."
                ) from None

        # If not interpolating, then just return the closest point in the grid.
        if not interpolate:
            teff = utils.nearest(self.teffs, teff)
            logg = utils.nearest(self.loggs, logg)
            feh = utils.nearest(self.fehs, feh)

            return self.fluxes[teff][logg][feh]

        # Otherwise, interpolate using the helper
        return utils.trilinear_interpolate(
            self.fluxes,
            (self.teffs, self.loggs, self.fehs),
            (teff, logg, feh),
        )
