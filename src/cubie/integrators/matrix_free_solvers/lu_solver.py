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
from cubie.cuda_simsafe import cuda, int32, selp
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBase,
    LinearSolverBaseConfig,
    LinearSolverCache,
)
from cubie.result_codes import CUBIE_RESULT_CODES


@frozen
class LUSolverConfig(LinearSolverBaseConfig):
    """Configuration for LUSolver compilation.

    The iterative-solver fields inherited from
    :class:`LinearSolverBaseConfig` are inert for the direct solve
    but keep ``settings_dict`` round-tripping through hot-swaps.

    Attributes
    ----------
    lu_solve_function : Optional[Callable]
        Generated direct-solve device function injected by the
        owning step's helper refresh.
    lu_nnz : int
        Factor buffer length; zero for a scalar-emitted factor.
    """

    lu_solve_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    lu_nnz: int = field(
        default=0, validator=getype_validator(int, 0)
    )

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return direct solver configuration as dictionary."""
        return {
            "krylov_max_iters": self.max_iters,
            "linear_correction_type": "lu",
            "zero_initial_guess": self.zero_initial_guess,
        }


class LUSolver(LinearSolverBase):
    """Factory for direct sparse LU solver device functions.

    Wraps the generated ``lu_solve`` helper in the shared linear
    solver contract. The solve writes the exact solution
    unconditionally and ignores the incoming guess in ``x``, so the
    config always declares ``zero_initial_guess=True``.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Length of the right-hand side and solution vectors.
    **kwargs
        Forwarded to :class:`LUSolverConfig` and the norm factory.
        A ``zero_initial_guess`` value is overridden to ``True``.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        **kwargs,
    ) -> None:
        kwargs.pop("zero_initial_guess", None)
        super().__init__(
            config_class=LUSolverConfig,
            precision=precision,
            solver_width=solver_width,
            zero_initial_guess=True,
            **kwargs,
        )

    def register_buffers(self) -> None:
        """Register the factor buffer with buffer_registry."""
        config = self.compile_settings
        buffer_registry.register(
            "lu_factor",
            self,
            config.lu_nnz,
            "local",
        )

    @property
    def linear_correction_type(self) -> str:
        """Return ``"lu"`` as the correction strategy."""
        return "lu"

    def build(self) -> LinearSolverCache:
        """Compile the direct solve into the linear solver contract.

        Returns
        -------
        LinearSolverCache
            Container with the compiled linear_solver device
            function.

        Notes
        -----
        ``lu_solve_function`` may still be ``None`` at build time;
        the owning step injects it before the wrapper first compiles
        into a kernel.
        """
        config = self.compile_settings
        lu_solve = config.lu_solve_function
        use_cached_auxiliaries = config.use_cached_auxiliaries
        jit_kwargs = self.jit_kwargs

        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        singular_pivot = int32(CUBIE_RESULT_CODES.SINGULAR_PIVOT)
        alloc_factor = buffer_registry.get_allocator(
            "lu_factor", self
        )

        if use_cached_auxiliaries:
            # no cover: start
            @cuda.jit(
                device=True,
                inline=True,
                **jit_kwargs,
            )
            def linear_solver_cached(
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
                """Run one cached direct LU solve.

                Parameters
                ----------
                state : array of numba_precision
                    Evaluation state for the Jacobian entries.
                parameters : array of numba_precision
                    Model parameters.
                drivers : array of numba_precision
                    External drivers.
                base_state : array of numba_precision
                    Unused; the cached solve evaluates at ``state``.
                cached_aux : array of numba_precision
                    Cached auxiliary values filled by prepare_jac.
                t : numba_precision
                    Stage time.
                h : numba_precision
                    Step size.
                a_ij : numba_precision
                    Stage coefficient scaling the Jacobian term.
                rhs : array of numba_precision
                    Right-hand side; read only.
                x : array of numba_precision
                    Solution vector, written unconditionally; the
                    incoming guess is ignored.
                shared : array
                    Shared memory pool.
                persistent_local : array
                    Persistent local memory pool.
                krylov_iters_out : array of int32
                    Single-element array receiving the iteration
                    count (always one).

                Returns
                -------
                int32
                    ``0`` on a clean factorisation,
                    ``SINGULAR_PIVOT`` when any pivot was floored.
                """
                factor = alloc_factor(shared, persistent_local)
                floored = lu_solve(
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
                krylov_iters_out[0] = int32(1)
                return selp(
                    floored != int32(0), singular_pivot, success
                )

            # no cover: end
            return LinearSolverCache(
                linear_solver=linear_solver_cached
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
            t,
            h,
            a_ij,
            rhs,
            x,
            shared,
            persistent_local,
            krylov_iters_out,
        ):
            """Run one direct LU solve.

            Parameters
            ----------
            state : array of numba_precision
                Stage increment (Newton form) or evaluation state
                (at-state form), forwarded to the generated solve.
            parameters : array of numba_precision
                Model parameters.
            drivers : array of numba_precision
                External drivers.
            base_state : array of numba_precision
                Stage base state for the Newton-form evaluation
                point.
            t : numba_precision
                Stage time.
            h : numba_precision
                Step size.
            a_ij : numba_precision
                Stage coefficient.
            rhs : array of numba_precision
                Right-hand side; read only.
            x : array of numba_precision
                Solution vector, written unconditionally; the
                incoming guess is ignored.
            shared : array
                Shared memory pool.
            persistent_local : array
                Persistent local memory pool.
            krylov_iters_out : array of int32
                Single-element array receiving the iteration count
                (always one).

            Returns
            -------
            int32
                ``0`` on a clean factorisation, ``SINGULAR_PIVOT``
                when any pivot was floored.
            """
            factor = alloc_factor(shared, persistent_local)
            floored = lu_solve(
                state,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                rhs,
                x,
                factor,
            )
            krylov_iters_out[0] = int32(1)
            return selp(
                floored != int32(0), singular_pivot, success
            )

        # no cover: end
        return LinearSolverCache(linear_solver=linear_solver)
