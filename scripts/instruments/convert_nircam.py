"""Reproduce the nominal NIRCam grism curve from Greene et al. (2017).

Run with Poetry. No input data download is needed: the published coefficients,
units and adopted domain are recorded in nircam_sources.json. The script checks
the output checksum before writing and never changes scientific choices or pins.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parents[2] / "src/speclib/data/instruments"


def convert(record):
    """Sample dimensionless R at the manifest's vacuum-micrometer coordinates."""
    wave = np.linspace(*record["calibration_domain_um"], record["curve_samples"])
    resolving_power = np.polynomial.polynomial.polyval(
        wave, record["coefficients_ascending"]
    )
    return np.column_stack((wave, resolving_power))


def csv_bytes(data):
    """Serialize float64 coordinates and R with round-trip precision."""
    stream = io.StringIO()
    np.savetxt(stream, data, delimiter=",", fmt="%.17g",
               header="wavelength_um,resolving_power")
    return stream.getvalue().encode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    manifest = json.loads((DATA / "nircam_sources.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, record in manifest["sources"].items():
        content = csv_bytes(convert(record))
        if hashlib.sha256(content).hexdigest() != record["output_sha256"]:
            raise ValueError(f"Output checksum changed for {name}; review before replacing")
        (args.output_dir / record["output_file"]).write_bytes(content)
        print(f"{name}: reproduced {record['output_file']} with pinned checksum")


if __name__ == "__main__":
    main()
