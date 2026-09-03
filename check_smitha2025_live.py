#!/usr/bin/env python
"""Live smoke test for Smitha et al. (2025) support in speclib.

This script accesses the real Edmond data release. It checks:

- public download/prefetch API
- all 3 stellar types x 4 canonical surface components
- photosphere -> quiet alias
- wavelength sampling, coverage, and ordering
- flux units and positivity
- physical flux normalization against sigma * Teff**4
- consistent wavelength grids
- ordinary Spectrum operations
- correct rejection of under-sampled resolution degradation
- basic identifying metadata

Run from the repository root with:

    poetry run python check_smitha2025_live.py
"""

from __future__ import annotations

import sys

import astropy.units as u
import numpy as np
from astropy.constants import sigma_sb

from speclib import Spectrum, download_smitha2025_spectra


STELLAR_TYPES = ("G2V", "K0V", "M0V")
COMPONENTS = ("quiet", "spot", "penumbra", "umbra")

# Published component effective temperatures from Smitha et al. (2025).
# No combined-spot Teff is given in Table 1.
TEFF = {
    "G2V": {
        "quiet": 5810.0,
        "penumbra": 5102.0,
        "umbra": 3819.0,
    },
    "K0V": {
        "quiet": 4965.0,
        "penumbra": 4410.0,
        "umbra": 3401.0,
    },
    "M0V": {
        "quiet": 3696.0,
        "penumbra": 3609.0,
        "umbra": 3399.0,
    },
}

EXPECTED_N_WAVELENGTH = 869
EXPECTED_WL_MIN = 200.500 * u.nm
EXPECTED_WL_MAX = 6010.001 * u.nm
EXPECTED_FLUX_UNIT = u.erg / (u.s * u.cm**2 * u.AA)


def assert_close(value, expected, *, rtol=1e-10, atol=0.0):
    """Convenience wrapper around numpy allclose."""
    np.testing.assert_allclose(
        value,
        expected,
        rtol=rtol,
        atol=atol,
    )


def integrated_surface_flux(spec: Spectrum) -> u.Quantity:
    """Integrate F_lambda over wavelength."""
    wavelength = spec.wavelength.to(u.AA)
    flux = spec.flux.to(EXPECTED_FLUX_UNIT)

    integral = np.trapezoid(
        flux.value,
        wavelength.value,
    )

    return integral * flux.unit * wavelength.unit


def check_spectrum(stellar_type: str, component: str) -> Spectrum:
    """Load and validate one Smitha spectrum."""
    print(
        f"  Loading {stellar_type:3s} / {component:9s} ... ",
        end="",
        flush=True,
    )

    spec = Spectrum.from_smitha2025(
        stellar_type,
        component=component,
    )

    assert isinstance(spec, Spectrum)

    wavelength = spec.wavelength.to(u.AA)
    flux = spec.flux.to(EXPECTED_FLUX_UNIT)

    # Basic file properties.
    assert len(wavelength) == EXPECTED_N_WAVELENGTH
    assert len(flux) == EXPECTED_N_WAVELENGTH

    assert_close(
        wavelength[0].to_value(u.nm),
        EXPECTED_WL_MIN.to_value(u.nm),
        atol=1e-6,
    )
    assert_close(
        wavelength[-1].to_value(u.nm),
        EXPECTED_WL_MAX.to_value(u.nm),
        atol=1e-6,
    )

    assert np.all(np.diff(wavelength.value) > 0)
    assert np.all(np.isfinite(wavelength.value))
    assert np.all(np.isfinite(flux.value))
    assert np.all(flux.value > 0)

    assert flux.unit.is_equivalent(EXPECTED_FLUX_UNIT)

    # Check identifying metadata where exposed as direct attributes.
    if hasattr(spec, "stellar_type"):
        assert spec.stellar_type == stellar_type

    if hasattr(spec, "component"):
        assert spec.component == component

    print("OK")
    return spec


def check_flux_normalization(
    stellar_type: str,
    component: str,
    spec: Spectrum,
) -> None:
    """Compare integrated F_lambda to sigma * Teff^4."""
    teff = TEFF[stellar_type][component]

    measured = integrated_surface_flux(spec).to(
        u.erg / (u.s * u.cm**2)
    )

    expected = (sigma_sb * (teff * u.K) ** 4).to(
        u.erg / (u.s * u.cm**2)
    )

    ratio = (measured / expected).to_value(u.dimensionless_unscaled)
    difference_percent = 100.0 * (ratio - 1.0)

    print(
        f"  {stellar_type:3s} / {component:9s}: "
        f"integral / sigmaT^4 = {ratio:.6f} "
        f"({difference_percent:+.3f}%)"
    )

    # The real release agrees at about the 0.25% level. Allow a little
    # headroom while remaining strict enough to catch a unit/normalization
    # mistake.
    assert abs(ratio - 1.0) < 0.005


def print_metadata(spec: Spectrum) -> None:
    """Show Smitha-specific metadata without assuming every attribute name."""
    print("\nMetadata found on representative spectrum:")

    candidates = (
        "model_grid",
        "model",
        "library",
        "stellar_type",
        "component",
        "teff",
        "component_teff",
        "logg",
        "paper_doi",
        "data_doi",
        "dataset_doi",
        "dataset_version",
        "source_filename",
        "source_md5",
        "native_resolution",
        "native_resolving_power",
        "flux_definition",
        "wavelength_convention",
    )

    found = False

    for name in candidates:
        if hasattr(spec, name):
            print(f"  {name}: {getattr(spec, name)!r}")
            found = True

    if not found:
        print("  No candidate Smitha-specific attributes found.")
        print("  Public-ish object attributes:")

        for name in sorted(spec.__dict__):
            if not name.startswith("_"):
                print(f"    {name}: {spec.__dict__[name]!r}")


def check_resolution_operations(spec: Spectrum) -> None:
    """Exercise supported and under-sampled resolving-power behavior."""
    print("\n6. Exercising ordinary Spectrum operations")

    # Resampling.
    new_wavelength = np.linspace(3000.0, 10000.0, 200) * u.AA
    resampled = spec.resample(new_wavelength)

    assert isinstance(resampled, Spectrum)
    assert len(resampled.wavelength) == len(new_wavelength)
    assert np.all(np.isfinite(resampled.flux.value))

    print("   resample(): OK")

    # The released Smitha wavelength grid is sparse and nonuniform.
    # R=50 is safely broad enough for the coarsest region of the grid.
    degraded = spec.set_spectral_resolving_power(50)

    assert isinstance(degraded, Spectrum)
    assert len(degraded.wavelength) == len(spec.wavelength)
    assert np.all(np.isfinite(degraded.flux.value))

    print("   set_spectral_resolving_power(50): OK")

    # R=100 is under-sampled somewhere on the released wavelength grid.
    # Confirm that speclib rejects this explicitly instead of silently
    # performing an inadequately sampled convolution.
    try:
        spec.set_spectral_resolving_power(100)

    except ValueError as exc:
        message = str(exc)

        assert "under-sampled" in message

        print("   under-sampled R=100 correctly rejected: OK")

    else:
        raise AssertionError(
            "Expected set_spectral_resolving_power(100) "
            "to fail because the released Smitha wavelength grid "
            "does not provide two samples per FWHM everywhere."
        )


def main() -> int:
    print("Smitha et al. (2025) live-data smoke test")
    print("=" * 52)

    # This should download missing files and validate/reuse cached files.
    print("\n1. Prefetching/validating the Edmond release")

    download_smitha2025_spectra()

    print("   Prefetch completed.")

    print("\n2. Loading all 12 canonical spectra")

    spectra = {}

    for stellar_type in STELLAR_TYPES:
        for component in COMPONENTS:
            spectra[(stellar_type, component)] = check_spectrum(
                stellar_type,
                component,
            )

    print("\n3. Checking wavelength-grid consistency")

    reference_wave = spectra[("G2V", "quiet")].wavelength.to_value(u.AA)

    for key, spec in spectra.items():
        np.testing.assert_allclose(
            spec.wavelength.to_value(u.AA),
            reference_wave,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"Wavelength mismatch for {key}",
        )

    print("   All 12 products use the same wavelength grid.")

    print("\n4. Checking photosphere -> quiet alias")

    for stellar_type in STELLAR_TYPES:
        quiet = spectra[(stellar_type, "quiet")]

        photosphere = Spectrum.from_smitha2025(
            stellar_type,
            component="photosphere",
        )

        np.testing.assert_allclose(
            photosphere.wavelength.to_value(u.AA),
            quiet.wavelength.to_value(u.AA),
        )

        np.testing.assert_allclose(
            photosphere.flux.to_value(EXPECTED_FLUX_UNIT),
            quiet.flux.to_value(EXPECTED_FLUX_UNIT),
        )

        if hasattr(photosphere, "component"):
            assert photosphere.component == "quiet"

        print(f"   {stellar_type}: alias OK")

    print("\n5. Checking physical flux normalization")

    for stellar_type in STELLAR_TYPES:
        for component in ("quiet", "penumbra", "umbra"):
            check_flux_normalization(
                stellar_type,
                component,
                spectra[(stellar_type, component)],
            )

    # Use a representative spectrum for normal Spectrum operations.
    representative = spectra[("K0V", "penumbra")]

    check_resolution_operations(representative)

    print_metadata(representative)

    print("\n" + "=" * 52)
    print("ALL SMITHA ET AL. (2025) LIVE CHECKS PASSED")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except Exception as exc:
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
