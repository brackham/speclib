"""Explicit wavelength coverage; spacing alone never declares a detector gap."""

import astropy.units as u
import numpy as np


def intervals_from_meta(meta):
    """Validate metadata and return detached, canonical wavelength intervals."""
    intervals = meta.get("valid_intervals")
    if intervals is None:
        return None
    if not isinstance(intervals, u.Quantity):
        raise TypeError("meta['valid_intervals'] must be a wavelength Quantity")
    raw = intervals.to_value(u.um)
    if np.iscomplexobj(raw):
        raise TypeError("valid_intervals must contain real wavelengths")
    values = np.asarray(raw, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or not len(values)
            or not np.all(np.isfinite(values)) or np.any(values <= 0)
            or np.any(values[:, 1] <= values[:, 0])
            or np.any(values[1:, 0] <= values[:-1, 1])):
        raise ValueError("valid_intervals must be positive, finite, ordered disjoint intervals of shape (n, 2)")
    # Immutable buffer, independent of the caller's quantity and metadata.
    return u.Quantity(np.frombuffer(values.tobytes(), dtype=float).reshape(-1, 2), u.um, copy=False)


def interval_membership(intervals, lower, upper=None):
    """Return interval indices for points/ranges; -1 means invalid coverage."""
    low = np.asarray(lower.to_value(u.um))
    high = low if upper is None else np.asarray(upper.to_value(u.um))
    owners = np.full(low.shape, -1, dtype=int)
    for index, (left, right) in enumerate(intervals.to_value(u.um)):
        tolerance = 8 * np.finfo(float).eps * right
        owners[(low >= left - tolerance) & (high <= right + tolerance)] = index
    return owners


def clipped_intervals(intervals, lower, upper, wavelength):
    """Intersect explicit limits, omitting intervals with no retained samples."""
    values = intervals.to_value(u.um).copy()
    values[:, 0] = np.maximum(values[:, 0], lower.to_value(u.um))
    values[:, 1] = np.minimum(values[:, 1], upper.to_value(u.um))
    keep = values[:, 0] < values[:, 1]
    wave = wavelength.to_value(u.um)
    keep &= np.array([np.any((wave >= a) & (wave <= b)) for a, b in values])
    if not np.any(keep):
        raise ValueError("Selection contains no valid wavelength coverage")
    return values[keep] * u.um


def require_continuous_coverage(obj):
    """Reject processing that cannot account for absent source light."""
    intervals = intervals_from_meta(getattr(obj, "meta", {}))
    if intervals is not None and len(intervals) > 1:
        raise ValueError(
            "This operation does not support discontinuous valid_intervals / detector gaps; "
            "use the continuous underlying model, or select one valid interval first"
        )
