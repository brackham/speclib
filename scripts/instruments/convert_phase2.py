"""Reproduce STIS, WFC3 and NIRISS profiles from pinned LSFs/published equations.

Run with Poetry. Only --download accesses the network. Source metadata and all
scientific coefficients live in phase2_sources.json; checksum pins are never
updated by this script.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np


DATA = Path(__file__).resolve().parents[2] / "src/speclib/data/instruments"


def half_maximum_width(offset, amplitude):
    """Width between the nearest half-height crossings around the global peak.

    Keep the source normalization: dividing by the peak would give the same
    crossings. A Gaussian fit/second moment would weight the wings differently.
    """
    x, y = np.asarray(offset, dtype=float), np.asarray(amplitude, dtype=float)
    if (x.ndim != 1 or x.size < 3 or y.shape != x.shape
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0) or np.any(y < 0) or y.max() <= 0):
        raise ValueError("Invalid LSF coordinates or amplitudes")
    peak = np.argmax(y)
    half = y[peak] / 2
    left = np.flatnonzero(y[:peak] < half)
    right = np.flatnonzero(y[peak:] < half)
    if not left.size or not right.size:
        raise ValueError("LSF does not bracket both half-maximum crossings")
    i, j = left[-1], peak + right[0]
    lower = x[i] + (half-y[i]) * (x[i+1]-x[i]) / (y[i+1]-y[i])
    upper = x[j-1] + (half-y[j-1]) * (x[j]-x[j-1]) / (y[j]-y[j-1])
    return upper - lower


def lsf_width(source, expected):
    """Validate the exact official file and select only the 52x2.0 column."""
    if hashlib.sha256(source).hexdigest() != expected["sha256"]:
        raise ValueError("Source checksum changed; review calibration before updating")
    lines = source.decode("ascii").splitlines()
    wavelength_nm = expected["wavelength_um"] * 1000
    if (lines[0] != f"STIS Model Line Spread Function vs Aperture at {wavelength_nm:g} nm"
            or lines[1].split() != ["Rel", "pixel", "52x0.1", "52x0.2", "52x0.5", "52x2.0"]):
        raise ValueError("Unexpected LSF wavelength or aperture header")
    table = np.loadtxt(io.StringIO("\n".join(lines[2:])))
    if table.shape != (expected["rows"], 5):
        raise ValueError("Unexpected LSF dimensions")
    return half_maximum_width(table[:, 0], table[:, 4])


def convert(record, source_dir):
    """Return vacuum micrometers and dimensionless R for one manifest record."""
    wave = np.linspace(*record["calibration_domain_um"], record["curve_samples"])
    if "lsf_sources" in record:
        anchors = list(record["handbook_anchors_um_pixels"])
        for source in record["lsf_sources"]:
            width = lsf_width((source_dir / source["filename"]).read_bytes(), source)
            anchors.append([source["wavelength_um"], width])
        anchors = np.array(sorted(anchors))
        # Include anchors exactly so the sampled prescription retains every knot.
        wave = np.unique(np.r_[wave, anchors[:, 0]])
        widths = np.interp(wave, anchors[:, 0], anchors[:, 1])
        resolving_power = wave / (1e-4 * record["dispersion_angstrom_per_pixel"] * widths)
    else:
        coefficients = record["coefficients"]
        if "reference_wavelength_um" in coefficients:
            resolving_power = (wave * coefficients["resolving_power"]
                               / coefficients["reference_wavelength_um"])
        else:
            resolving_power = (1000 * wave / (coefficients["resolution_pixels"]
                                             * coefficients["dispersion_nm_per_pixel"]))
    return np.column_stack((wave, resolving_power))


def csv_bytes(data):
    """Serialize with sufficient precision for lossless float64 round trips."""
    stream = io.StringIO()
    np.savetxt(stream, data, delimiter=",", fmt="%.17g",
               header="wavelength_um,resolving_power")
    return stream.getvalue().encode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((DATA / "phase2_sources.json").read_text())
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, record in manifest["sources"].items():
        if args.download:
            for source in record.get("lsf_sources", []):
                with urlopen(source["url"], timeout=60) as response:
                    content = response.read()
                lsf_width(content, source)  # Validate before replacing cached input.
                (args.source_dir / source["filename"]).write_bytes(content)
        content = csv_bytes(convert(record, args.source_dir))
        if hashlib.sha256(content).hexdigest() != record["output_sha256"]:
            raise ValueError(f"Output checksum changed for {name}; review before replacing")
        (args.output_dir / record["output_file"]).write_bytes(content)
        print(f"{name}: reproduced {record['output_file']} with pinned checksum")


if __name__ == "__main__":
    main()
