"""Reproduce bundled NIRSpec curves from checksum-pinned official FITS files.

Run with Poetry from the repository root. Downloads occur only with --download;
--source-dir permits completely offline conversion. No resampling is performed.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

from astropy.io import fits
import numpy as np


DATA = Path(__file__).resolve().parents[2] / "src/speclib/data/instruments"


def convert(source, expected):
    """Validate a pinned source and return every wavelength/R pair as float64."""
    if hashlib.sha256(source).hexdigest() != expected["sha256"]:
        raise ValueError("Source checksum changed; review calibration before updating")
    with fits.open(io.BytesIO(source)) as hdus:
        table = hdus[1]
        if table.columns["WAVELENGTH"].unit != "MICRONS":
            raise ValueError("Unexpected wavelength unit")
        data = np.column_stack((table.data["WAVELENGTH"], table.data["R"])).astype(float)
        dispersion = table.data["DLDS"].astype(float)
        if len(data) != expected["rows"] or hdus[0].header["DATE"] != expected["fits_date"]:
            raise ValueError("Unexpected source version or row count")
        if not np.all(np.isfinite(data)) or np.any(data <= 0):
            raise ValueError("Nonphysical calibration values")
        if np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError("Nonmonotonic calibration wavelengths")
        # The updated high-R tables interpolate R and dispersion separately;
        # they satisfy the documented 2.2-pixel definition to within 1 percent.
        if not np.allclose(data[:, 1], data[:, 0] / (2.2 * dispersion), rtol=0.01):
            raise ValueError("R inconsistent with the published resolution definition")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    manifest = json.loads((DATA / "sources.json").read_text())
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["sources"].items():
        source_path = args.source_dir / expected["url"].rsplit("/", 1)[1]
        if args.download:
            with urlopen(expected["url"], timeout=60) as response:
                source = response.read()
            convert(source, expected)  # Never cache an unvalidated replacement.
            source_path.write_bytes(source)
        data = convert(source_path.read_bytes(), expected)
        np.savetxt(
            args.output_dir / f"{name}.csv", data, delimiter=",", fmt="%.17g",
            header="wavelength_um,resolving_power",
        )
        print(f"{name}: validated {len(data)} unchanged source samples")


if __name__ == "__main__":
    main()
