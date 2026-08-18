"""Matrix-free solver factories used by integrator kernels.

The package exposes CUDA device function factories for linear and nonlinear
solvers that are consumed by modules in :mod:`cubie.integrators`.
"""

from cubie.result_codes import CUBIE_RESULT_CODES
from .base_solver import MatrixFreeSolverConfig
from .linear_solver_base import (
    LinearSolverBase,
    LinearSolverBaseConfig,
    LinearSolverCache,
)
from .linear_solver import (
    MRLinearSolver,
    MRLinearSolverConfig,
)
from .bicgstab_solver import (
    BiCGSTABSolver,
    BiCGSTABSolverConfig,
)
from .lu_solver import (
    LUSolver,
    LUSolverConfig,
)
from .newton_krylov import (
    NewtonKrylov,
    NewtonKrylovConfig,
    NewtonKrylovCache,
)


__all__ = [
    "LinearSolverBase",
    "LinearSolverBaseConfig",
    "LinearSolverCache",
    "MRLinearSolver",
    "MRLinearSolverConfig",
    "BiCGSTABSolver",
    "BiCGSTABSolverConfig",
    "LUSolver",
    "LUSolverConfig",
    "MatrixFreeSolverConfig",
    "NewtonKrylov",
    "NewtonKrylovConfig",
    "NewtonKrylovCache",
    "CUBIE_RESULT_CODES",
]
