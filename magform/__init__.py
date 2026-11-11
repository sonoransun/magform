"""magform: engineering models for ultra-high magnetic-pressure processing.

The package models pulsed magnetic pressures in the 1.6-4.0 GPa range (63-100 T
fields) and every process the original concept note proposes around them:
electromagnetic (magnetic-pulse) forging, room-temperature deformation of
refractory metals, impact welding of dissimilar alloys, high-pressure synthesis
of superhard materials, pore closure in aerospace alloys, pulse-power network
sizing, and the inverted magnet architecture as a containment structure.  Each
result carries a feasibility status (state of the art, extrapolation,
aspirational, infeasible) traced to cited benchmarks, and the claims ledger in
:mod:`magform.claims` maps every sentence of the concept note to a verdict.
:mod:`magform.press` assembles the models into a press design and compares the
ways of building it; :mod:`magform.magnetar` is a labelled thought experiment
(fields of magnetar strength, granted by premise) whose status is ``assumed``.

All quantities are SI unless a name says otherwise.  The package uses only the
Python standard library, as does ``tools/build_docs.py``, which generates the
data behind the documentation site from these models.

Sub-modules are imported lazily, so ``import magform`` is cheap and
``magform.press`` (for example) loads on first use.
"""

from __future__ import annotations

import importlib

__version__ = "0.2.0"

# In dependency order: each module imports only from those to its left.
_SUBMODULES = (
    "constants", "errors", "status", "benchmarks", "materials",
    "magnetics", "fieldmap", "conductor", "plasticity", "shock", "eddy", "microstructure",
    "containment", "synthesis", "welding", "porosity", "pulse", "magnetar",
    "press", "claims", "cli",
)

# Convenience re-exports: name -> (module, attribute)
_EXPORTS = {
    "Status": ("status", "Status"),
    "Verdict": ("status", "Verdict"),
    "Material": ("materials", "Material"),
    "get_material": ("materials", "get"),
    "magnetic_pressure": ("magnetics", "magnetic_pressure"),
    "field_for_pressure": ("magnetics", "field_for_pressure"),
    "PressSpec": ("press", "PressSpec"),
    "design_press": ("press", "design_press"),
    "compare_architectures": ("press", "compare_architectures"),
    "reference_spec": ("press", "reference_spec"),
    "evaluate_claims": ("claims", "evaluate_all"),
}

__all__ = ["__version__", *_SUBMODULES, *_EXPORTS]


def __getattr__(name: str):
    """Lazily import sub-modules and re-exported names on first access."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _EXPORTS:
        mod, attr = _EXPORTS[name]
        return getattr(importlib.import_module(f".{mod}", __name__), attr)
    raise AttributeError(f"module 'magform' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - interactive convenience
    return sorted(set(globals()) | set(__all__))
