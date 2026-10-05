"""Simplify and tear ODE and DAE systems represented as engine IR.

This module re-exports the package API and is cubie's own; each
submodule's docstring names the source of its parts.
"""

from cubie.odesystems.symbolic.structural.simplify import (
    structural_simplify,
    SimplifiedSystem,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.structural.errors import (
    ExtraEquationsSystemError,
    ExtraVariablesSystemError,
    InvalidSystemError,
)

__all__ = [
    "structural_simplify",
    "SimplifiedSystem",
    "StructuralState",
    "ExtraEquationsSystemError",
    "ExtraVariablesSystemError",
    "InvalidSystemError",
]
