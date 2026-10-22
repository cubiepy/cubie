"""CUDA-ready ordinary differential equation (ODE) system factories.

The :func:`create_ODE_system` helper is the primary entry point. It
builds symbolic system definitions into :class:`SymbolicODE` instances
that inherit from :class:`BaseODE` and compile CUDA device functions
through :class:`cubie.CUDAFactory`. Data containers such as
:class:`ODEData` and :class:`SystemValues` describe metadata consumed by
integrator factories. See :mod:`cubie.odesystems.variables` for the
states, parameters and observables a system declares, and
:class:`SystemLayout` for the position of each variable in each device
buffer.

Subpackages
-----------

``cubie.odesystems.symbolic``
    Generates CUDA-ready kernels from :mod:`sympy` expressions and emits the
    solver helpers that integrate with :mod:`cubie.integrators`.
"""

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.ODEData import ODEData, SystemSizes
from cubie.odesystems.variables import (
    CompiledParameter,
    Parameter,
    ParameterSet,
    State,
    Variable,
    ValuedSet,
    VariableSet,
)
from cubie.odesystems.baseODE import BaseODE, ODECache
from cubie.odesystems.symbolic import (SymbolicODE, create_ODE_system,
                                       load_cellml_model)

__all__ = [
    "BaseODE",
    "CompiledParameter",
    "ODECache",
    "ODEData",
    "Parameter",
    "ParameterSet",
    "State",
    "SystemLayout",
    "SystemSizes",
    "Variable",
    "ValuedSet",
    "VariableSet",
    "SymbolicODE",
    "create_ODE_system",
    "load_cellml_model",
]
