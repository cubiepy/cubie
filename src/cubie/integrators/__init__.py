"""Numerical integration algorithms and settings for ODE solving.

``SingleIntegratorRun`` is the primary entry point. It composes a device
loop callable from controller, algorithm, and loop factories based on the
provided
:class:`cubie.integrators.IntegratorRunSettings.IntegratorRunSettings`.

This package collects CUDA-oriented integration components, including
algorithm factories, solver helpers, loop builders, and controller
utilities that orchestrate initial value problem (IVP) integrations.

Subpackages
-----------
algorithms
    Explicit and implicit step factories that share configuration helpers
    and provide Euler, backward Euler, predictor-corrector, and
    Crank--Nicolson implementations.
loops
    CUDA loop factories that assemble device functions and manage shared
    and local memory layouts for IVP execution.
matrix_free_solvers
    Matrix-free Newton--Krylov and linear solver factories consumed by the
    implicit algorithms.
step_control
    Adaptive and fixed-step controller factories used to update step sizes
    on device.

Notes
-----
``CUBIE_RESULT_CODES`` (re-exported here and from :mod:`cubie`) is the single
status-code vocabulary; device functions OR its values into the per-run status
word. Iteration counts are returned separately through the ``counters`` array,
not packed into the status word.
"""

from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    CrankNicolsonStep,
    ExplicitEulerStep,
    ExplicitStepConfig,
    ImplicitStepConfig,
    get_algorithm_step,
)
from cubie.integrators.loops import IVPLoop
from cubie.integrators.matrix_free_solvers import (
    MRLinearSolver,
    MRLinearSolverConfig,
    LinearSolverCache,
    BiCGSTABSolver,
    BiCGSTABSolverConfig,
    NewtonKrylov,
    NewtonKrylovConfig,
    NewtonKrylovCache,
)
from cubie.integrators.step_control import (
    AdaptiveIController,
    AdaptivePIController,
    AdaptivePIDController,
    FixedStepController,
    GustafssonController,
    SciMLPIController,
    get_controller,
)


__all__ = [
    "SingleIntegratorRun",
    "CUBIE_RESULT_CODES",
    "get_algorithm_step",
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "ExplicitEulerStep",
    "BackwardsEulerStep",
    "BackwardsEulerPCStep",
    "CrankNicolsonStep",
    "IVPLoop",
    "MRLinearSolver",
    "MRLinearSolverConfig",
    "LinearSolverCache",
    "BiCGSTABSolver",
    "BiCGSTABSolverConfig",
    "NewtonKrylov",
    "NewtonKrylovConfig",
    "NewtonKrylovCache",
    "AdaptiveIController",
    "AdaptivePIController",
    "AdaptivePIDController",
    "FixedStepController",
    "GustafssonController",
    "SciMLPIController",
    "get_controller",
]
