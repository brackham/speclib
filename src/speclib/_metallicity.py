"""Native coordinate definitions and validation at public API boundaries.

Numeric values are passed through unchanged. No abundance conversion is made.
"""

import warnings


class _Unspecified:
    def __repr__(self):
        return "<unspecified>"


UNSET = _Unspecified()

# These compatibility selectors retain their historical feh keyword and do
# not participate in the native-coordinate API migration.
LEGACY_METALLICITY_SELECTORS = {"drift-phoenix", "nextgen-solar"}

METALLICITY_TYPES = {
    "phoenix": "feh",
    "newera": "mh",
    "newera_gaia": "mh",
    "newera_jwst": "mh",
    "newera_lowres": "mh",
    "sphinx": "mh",
    "mps-atlas": "mh",
    "mps-atlas-set1": "mh",
    "mps-atlas-set2": "mh",
    "kostogryz2026": "mh",
}


def reject_grid_metallicity_keywords(kwargs):
    """Reject removed bounds names and scalar overrides before grid loading."""
    if "feh_bds" in kwargs:
        raise TypeError("`feh_bds` was removed; use `metallicity_bds` instead.")
    scalar_names = [name for name in ("metallicity", "feh", "mh") if name in kwargs]
    if scalar_names:
        raise TypeError(
            "Grid constructors do not accept scalar metallicity keywords "
            f"({', '.join(scalar_names)}); use `metallicity_bds` to select "
            "the grid coordinates."
        )


def resolve_metallicity(
    metallicity,
    model_grid,
    *,
    feh=UNSET,
    mh=UNSET,
    default=UNSET,
    parameter="metallicity",
    storage_aliases=None,
):
    """Resolve exactly one input; only UNSET represents an omitted argument."""
    suffix = parameter.removeprefix("metallicity")
    inputs = {parameter: metallicity, f"mh{suffix}": mh}
    if not suffix:
        inputs["feh"] = feh
    inputs.update(storage_aliases or {})
    supplied = {name: value for name, value in inputs.items() if value is not UNSET}
    if len(supplied) > 1:
        raise ValueError(
            f"Specify only one metallicity keyword ({', '.join(supplied)}); "
            "multiple forms are ambiguous even when their values agree."
        )
    if not supplied:
        if default is UNSET:
            raise TypeError(f"Missing required argument: {parameter}")
        return default
    name, value = next(iter(supplied.items()))
    if value is None:
        expected = "a pair of numeric values" if suffix else "a numeric value"
        message = f"{name} must be {expected}"
        if default is not UNSET:
            message += "; omit the argument to use the default"
        raise TypeError(message)
    selector = model_grid.lower()
    if selector in LEGACY_METALLICITY_SELECTORS:
        if name == f"mh{suffix}":
            raise TypeError(
                f"unexpected keyword argument '{name}' for legacy selector {model_grid}"
            )
        return value
    native_type = METALLICITY_TYPES.get(selector)
    if name == f"mh{suffix}" and native_type != "mh":
        definition = ", which uses [Fe/H]" if native_type == "feh" else ""
        raise ValueError(
            f"`{name}` is not native to {model_grid}{definition}. "
            f"Use `{parameter}=`"
            + (" or `feh=`." if not suffix and native_type == "feh" else ".")
        )
    if name == "feh" and native_type != "feh":
        definition = f"{model_grid} uses [M/H]; " if native_type == "mh" else ""
        alternative = " or `mh=`" if native_type == "mh" else ""
        raise ValueError(
            f"{definition}`feh` is a native alias only for "
            f"PHOENIX-ACES ([Fe/H]). Use `metallicity=`{alternative} instead. "
            "SpecLib does not convert between [Fe/H] and [M/H]."
        )
    elif name in (storage_aliases or {}):
        warnings.warn(
            f"`{name}` is deprecated; use `{parameter}=` for native [M/H]. "
            "The numeric value is unchanged; no abundance conversion is performed.",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


class MetallicityGridMetadata:
    """Record numeric native metallicity axes and their scientific definition."""

    def _set_metallicity_metadata(self):
        if self.model_grid in LEGACY_METALLICITY_SELECTORS:
            self.meta = {}
            return
        self.metallicity_type = METALLICITY_TYPES[self.model_grid]
        self.meta = {
            "source_library": self.model_grid,
            "metallicities": tuple(float(value) for value in self.metallicities),
            "metallicity_type": self.metallicity_type,
        }
