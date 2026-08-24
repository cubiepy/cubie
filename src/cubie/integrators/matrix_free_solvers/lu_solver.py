"""Direct sparse LU linear solver.

Wraps the generated ``lu_solve`` device function (see
:mod:`cubie.odesystems.symbolic.codegen.lu_solver`) in the linear
solver calling contract shared with the iterative solvers. The solve
factorises ``beta*M - gamma*a_ij*h*J`` at the current evaluation
point and is exact per call, reporting one iteration.

Published Classes
-----------------
:class:`LUSolverConfig`
    Attrs configuration for the direct LU solver factory.

:class:`LUSolver`
    CUDAFactory subclass that compiles the direct solve into the
    linear solver device contract.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    Abstract parent providing shared infrastructure.
:mod:`cubie.odesystems.symbolic.codegen.lu_solver`
    Generates the device function this factory wraps.
"""

from typing import Any, Callable, Dict, Optional

from attrs import field, frozen, validators

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    is_device_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import cuda, int32
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBase,
    LinearSolverBaseConfig,
    LinearSolverCache,
)


@frozen
class LUSolverConfig(LinearSolverBaseConfig):
    """Configuration for LUSolver compilation.

    Attributes
    ----------
    lu_solve_function : Optional[Callable]
        Injected generated direct-solve device function.
    lu_nnz : int
        Factor buffer length; zero for substitution-only variants.
    lu_factor_location : str
        Memory location for the per-call factor buffer.
    """

    lu_solve_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    lu_nnz: int = field(
        default=0, validator=getype_validator(int, 0)
    )
    lu_factor_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    # The direct solve ignores the incoming guess in ``x``.
    zero_initial_guess: bool = field(default=True, init=False)

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return direct solver configuration as dictionary."""
        return {
            "linear_correction_type": "lu",
            "zero_initial_guess": self.zero_initial_guess,
            "lu_factor_location": self.lu_factor_location,
        }


class LUSolver(LinearSolverBase):
    """Factory for direct sparse LU solver device functions.

    Wraps the generated ``lu_solve`` helper in the shared linear
    solver contract.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Length of the right-hand side and solution vectors.
    **kwargs
        Forwarded to :class:`LUSolverConfig` and the norm factory.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        **kwargs,
    ) -> None:
        super().__init__(
            config_class=LUSolverConfig,
            precision=precision,
            solver_width=solver_width,
            **kwargs,
        )

    def register_buffers(self) -> None:
        """Register the factor buffer with buffer_registry."""
        config = self.compile_settings
        buffer_registry.register(
            "lu_factor",
            self,
            config.lu_nnz,
            config.lu_factor_location,
        )

    @property
    def linear_correction_type(self) -> str:
        """Return ``"lu"`` as the correction strategy."""
        return "lu"

    @property
    def max_iters(self) -> int:
        """Return one: the direct solve finishes in a single pass."""
        return 1

    def build(self) -> LinearSolverCache:
        """Compile the direct solve into the linear solver contract.

        Returns
        -------
        LinearSolverCache
            Container with the compiled linear_solver device
            function.
        """
        config = self.compile_settings
        lu_solve = config.lu_solve_function
        jit_kwargs = self.jit_kwargs

        alloc_factor = buffer_registry.get_allocator(
            "lu_factor", self
        )

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **jit_kwargs,
        )
        def linear_solver(
            state,
            parameters,
            drivers,
            base_state,
            cached_aux,
            t,
            h,
            a_ij,
            rhs,
            x,
            shared,
            persistent_local,
            krylov_iters_out,
        ):
            """Run one direct LU solve; writes x, reports one iteration."""
            factor = alloc_factor(shared, persistent_local)
            krylov_iters_out[0] = int32(1)
            return lu_solve(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                a_ij,
                rhs,
                x,
                factor,
            )

        # no cover: end
        return LinearSolverCache(linear_solver=linear_solver)
