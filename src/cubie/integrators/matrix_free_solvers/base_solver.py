"""Base configuration for matrix-free solver factories.

This module provides shared configuration infrastructure for the
Newton and Krylov solvers in
:mod:`cubie.integrators.matrix_free_solvers`.

Published Classes
-----------------
:class:`MatrixFreeSolverConfig`
    Attrs configuration base for solver factories, extending
    :class:`~cubie.CUDAFactory.MultipleInstanceCUDAFactoryConfig`
    with vector size and norm device function fields.

:class:`MatrixFreeSolver`
    Factory base class managing a :class:`~cubie.integrators.norms.ScaledNorm`
    instance and prefixed tolerance parameter extraction.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver.MRLinearSolver`
    Concrete MR/SD linear solver subclass.
:class:`~cubie.integrators.matrix_free_solvers.bicgstab_solver.BiCGSTABSolver`
    Concrete BiCGSTAB linear solver subclass.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Concrete Newton--Krylov solver subclass.
:class:`~cubie.CUDAFactory.MultipleInstanceCUDAFactory`
    Parent factory providing prefixed parameter support.
"""

from typing import Any, Callable, Dict, Optional, Set

from attrs import field, frozen
from numpy import ndarray

from cubie._utils import (
    device_function_field,
    getype_validator,
    PrecisionDType,
)
from cubie.CUDAFactory import (
    MultipleInstanceCUDAFactory,
    MultipleInstanceCUDAFactoryConfig,
)
from cubie.integrators.norms import ScaledNorm


@frozen
class MatrixFreeSolverConfig(MultipleInstanceCUDAFactoryConfig):
    """Base configuration for matrix-free solver factories.

    Provides common attributes shared by LinearSolverBaseConfig and
    NewtonKrylovConfig including precision, vector size, and
    Numba/CUDA type accessors.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Solver vector length (must be >= 1).
    norm_fn : Optional[Callable]
        Compiled norm function for convergence checks. Updated when
        norm factory rebuilds; changes invalidate solver cache.
    """

    solver_width: int = field(
        default=0, validator=getype_validator(int, 1)
    )
    norm_fn: Optional[Callable] = device_function_field(prefixed=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


class MatrixFreeSolver(MultipleInstanceCUDAFactory):
    """Base factory for matrix-free solver device functions.

    Provides shared infrastructure for tolerance parameter mapping
    and norm factory management. Subclasses set `solver_type`
    to enable automatic mapping of prefixed parameters (e.g.,
    "krylov_atol" -> "atol" for norm updates).

    Attributes
    ----------
    solver_type : str
        Prefix for tolerance parameters (e.g., "krylov_" or "newton_").
        Set by subclasses.
    norm : ScaledNorm
        Factory for scaled norm device function used in convergence checks.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_type: str,
        solver_width: int,
        norm: Optional[ScaledNorm] = None,
        **kwargs,
    ) -> None:
        """Initialize base solver with norm factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        solver_type : str
            Prefix for tolerance parameters (e.g., "krylov" or "newton").
        solver_width : int
            Solver vector length.
        norm : ScaledNorm, optional
            Norm owned by the solver.
        **kwargs
            Default-norm settings; ``n_states`` defaults to the width.
        """
        self.solver_type = solver_type
        super().__init__(instance_label=solver_type)
        n_states = kwargs.pop("n_states", solver_width)
        if norm is None:
            norm = ScaledNorm(
                precision=precision,
                solver_width=solver_width,
                n_states=n_states,
                instance_label=solver_type,
                **kwargs,
            )
        self.norm = norm

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Update the owned norm, then the solver with its ``norm_fn``."""
        recognized = self.norm.update(updates, silent=True)
        updates[self.prefixed("norm_fn")] = self.norm.device_function
        return recognized | self.update_compile_settings(
            updates, silent=True
        )

    @property
    def atol(self) -> ndarray:
        """Absolute tolerance for the solver."""
        return self.norm.atol

    @property
    def rtol(self) -> ndarray:
        """Relative tolerance for the solver."""
        return self.norm.rtol

    @property
    def max_iters(self) -> int:
        """Maximum iterations allowed for the solver."""
        return self.compile_settings.max_iters

    @property
    def solver_width(self) -> int:
        """Return the solver vector length."""
        return self.compile_settings.solver_width
