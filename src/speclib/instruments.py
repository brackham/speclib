"""Offline nominal Gaussian instrument-resolution profiles.

Calibration coordinates are treated as vacuum wavelengths. Coverage describes
detector geometry separately from the continuous source used for convolution.
"""

from dataclasses import dataclass
from importlib.resources import files
import json
import tomllib

import astropy.units as u
import numpy as np


__all__ = ["InstrumentProfile", "get_profile", "list_instrument_profiles"]


def _wavelength_values(wavelength):
    if not isinstance(wavelength, u.Quantity):
        raise TypeError("wavelength must be an astropy Quantity with wavelength units")
    values = np.asarray(wavelength.to_value(u.um))
    if np.iscomplexobj(values) or values.dtype.kind == "b":
        raise TypeError("wavelength must contain real values")
    values = values.astype(float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("wavelength must contain finite positive values")
    return values


@dataclass(frozen=True, eq=False)
class InstrumentProfile:
    """A sampled resolving-power curve and one or more valid intervals.

    Parameters
    ----------
    mode : str
        Canonical hierarchical identifier (normalized to lowercase).
    wavelength : astropy.units.Quantity
        Ascending vacuum wavelengths of the calibration, including padding.
    resolving_power : array-like
        Positive dimensionless R values; wavelength/R is adopted as Gaussian FWHM.
    valid_intervals : astropy.units.Quantity
        Shape (n, 2), ordered, nonoverlapping wavelength intervals.
    source, source_url, calibration_version, assumptions : str or tuple of str
        Calibration provenance and scientific assumptions.
    quality : {"tabulated", "derived", "nominal"}
        Provenance classification, not a precision guarantee.
    configuration, coverage_source_url, resolution_definition : str
        Aperture/subarray, coverage provenance, and definition of R.
    provenance_ref : str, optional
        Bundled manifest reference, ``filename.json#source_key``. Empty for
        profiles without a bundled record. See :attr:`provenance` for original
        data links, identifiers, processing and reproducibility information.

    Notes
    -----
    Arrays are copied into immutable buffers. Lookup performs no network I/O.
    User-defined profiles use the same validation as bundled profiles.
    """

    mode: str
    wavelength: u.Quantity
    resolving_power: np.ndarray
    valid_intervals: u.Quantity
    source: str
    source_url: str
    calibration_version: str
    assumptions: tuple[str, ...]
    quality: str = "tabulated"
    configuration: str = ""
    coverage_source_url: str = ""
    resolution_definition: str = "R=lambda/delta_lambda; Gaussian FWHM"
    provenance_ref: str = ""

    def __post_init__(self):
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("mode must be a nonempty hierarchical identifier")
        wave = _wavelength_values(self.wavelength)
        intervals = _wavelength_values(self.valid_intervals)
        r = self.resolving_power
        if isinstance(r, u.Quantity):
            r = r.to_value(u.dimensionless_unscaled)
        r = np.asarray(r)
        if np.iscomplexobj(r) or r.dtype.kind == "b":
            raise TypeError("resolving_power must contain real values")
        r = r.astype(float)
        if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0):
            raise ValueError("Calibration wavelength must be ascending with at least two samples")
        if r.shape != wave.shape or not np.all(np.isfinite(r)) or np.any(r <= 0):
            raise ValueError("resolving_power must be positive, finite, and match wavelength")
        if intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals):
            raise ValueError("valid_intervals must have shape (n, 2)")
        if np.any(intervals[:, 0] >= intervals[:, 1]) or np.any(
            intervals[1:, 0] <= intervals[:-1, 1]
        ):
            raise ValueError("valid_intervals must be ordered, disjoint, and have positive width")
        if intervals.min() < wave[0] or intervals.max() > wave[-1]:
            raise ValueError(
                "Calibration must cover valid_intervals; extrapolation is not supported"
            )
        if self.quality not in {"tabulated", "derived", "nominal"}:
            raise ValueError("quality must be tabulated, derived, or nominal")
        for name in ("source", "source_url", "calibration_version", "resolution_definition"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if isinstance(self.assumptions, str) or not self.assumptions or not all(
            isinstance(item, str) and item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a nonempty sequence of strings")
        object.__setattr__(self, "mode", self.mode.strip().lower())
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        # bytes-backed arrays cannot be made writable, even via .value.
        for name, values, unit in (
            ("wavelength", wave, u.um),
            ("valid_intervals", intervals, u.um),
            ("resolving_power", r, None),
        ):
            array = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
            value = array if unit is None else u.Quantity(array, unit, copy=False)
            object.__setattr__(self, name, value)

    def resolving_power_at(self, wavelength):
        """Linearly interpolate R inside the calibration domain; never extrapolate."""
        wave = _wavelength_values(wavelength)
        curve = self.wavelength.to_value(u.um)
        if np.any(wave < curve[0]) or np.any(wave > curve[-1]):
            raise ValueError("Wavelength outside calibration; extrapolation is not supported")
        return np.interp(wave, curve, self.resolving_power)

    @property
    def provenance(self):
        """Detached source-provenance dictionary, read offline from the bundle.

        Manifest defaults are combined with the selected source record. Each
        access returns a fresh dictionary, so editing it cannot alter profiles
        or packaged records. An unreferenced custom profile returns ``{}``.
        Coverage and Speclib's adopted LSF definition remain separately available
        as ``coverage_source_url`` and ``resolution_definition``.
        """
        if not self.provenance_ref:
            return {}
        filename, key = self.provenance_ref.split("#", 1)
        manifest = json.loads(
            files("speclib.data").joinpath("instruments", filename).read_text()
        )
        return (manifest.get("defaults", {})
                | {"retrieved": manifest["retrieved"]}
                | manifest["sources"][key])


def _registry():
    root = files("speclib.data").joinpath("instruments")
    with root.joinpath("profiles.toml").open("rb") as stream:
        registry = tomllib.load(stream)
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported instrument registry schema")
    return root, registry


def list_instrument_profiles():
    """Return a sorted tuple of canonical mode identifiers (aliases omitted)."""
    _, registry = _registry()
    return tuple(sorted(item["mode"] for item in registry["profiles"]))


def get_profile(mode):
    """Resolve a case-insensitive identifier, or return an InstrumentProfile.

    NIRSpec shorthand identifiers default to BOTS/S1600A1/SUB2048. Explicit
    identifiers use ``jwst/nirspec/bots/s1600a1/sub2048/<disperser>/<filter>``.
    STIS defaults to the prime CCD setting with a centered point source through
    52X2; SOSS defaults to CLEAR/SUBSTRIP256, with each order a separate profile.
    Unsupported configurations fail instead of falling back to this default.
    """
    if isinstance(mode, InstrumentProfile):
        return mode
    if not isinstance(mode, str):
        raise TypeError("mode must be a string or InstrumentProfile")
    root, registry = _registry()
    key = mode.strip().lower()
    for item in registry["profiles"]:
        if key in (item["mode"], *item.get("aliases", [])):
            with root.joinpath(item["curve"]).open("r") as stream:
                curve = np.loadtxt(stream, delimiter=",")
            metadata = registry["defaults"] | {
                key: value for key, value in item.items()
                if key not in {"mode", "curve", "aliases", "valid_intervals_um"}
            }
            return InstrumentProfile(
                mode=item["mode"], wavelength=curve[:, 0] * u.um,
                resolving_power=curve[:, 1],
                valid_intervals=np.array(item["valid_intervals_um"]) * u.um,
                **metadata,
            )
    raise ValueError(f"Unknown instrument profile {mode!r}; use list_instrument_profiles()")
