"""Matrix-free preconditioned MR/SD linear solver.

This module builds CUDA device functions that implement
steepest-descent or minimal-residual iterations without forming
Jacobian matrices explicitly. The solver expects caller-supplied
operator and preconditioner callbacks.

Published Classes
-----------------
:class:`MRLinearSolverConfig`
    Attrs configuration for the MR linear solver factory.

:class:`MRLinearSolver`
    CUDAFactory subclass that compiles a preconditioned iterative
    linear solver for use inside Newton--Krylov iterations or
    Rosenbrock-W methods.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    Abstract parent providing shared infrastructure.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Newton--Krylov solver that wraps a linear solver.
:mod:`cubie.integrators.algorithms.ode_implicitstep`
    Implicit step base class that creates linear solver instances.
"""

from math import sqrt as math_sqrt
from typing import Dict, Any

from attrs import field, validators, frozen
from cubie.cuda_simsafe import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBaseConfig,
    LinearSolverBase,
    LinearSolverCache,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, selp
from cubie.result_codes import CUBIE_RESULT_CODES


@frozen
class MRLinearSolverConfig(LinearSolverBaseConfig):
    """Configuration for MRLinearSolver compilation.

    Attributes
    ----------
    linear_correction_type : str
        Line-search strategy ('steepest_descent' or 'minimal_residual').
    preconditioned_vec_location : str
        Memory location for preconditioned_vec buffer.
    temp_location : str
        Memory location for temp buffer.
    """

    linear_correction_type: str = field(
        default="minimal_residual",
        validator=validators.in_(["steepest_descent", "minimal_residual"]),
    )
    preconditioned_vec_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    temp_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return linear solver configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            "krylov_max_iters": self.max_iters,
            "linear_correction_type": self.linear_correction_type,
            "preconditioned_vec_location": self.preconditioned_vec_location,
            "temp_location": self.temp_location,
        }


class MRLinearSolver(LinearSolverBase):
    """Factory for MR/SD linear solver device functions.

    Implements steepest-descent or minimal-residual iterations
    for solving linear systems without forming Jacobian matrices.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Length of residual and search-direction vectors.
    **kwargs
        Forwarded to :class:`MRLinearSolverConfig` and the norm
        factory.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        **kwargs,
    ) -> None:
        super().__init__(
            config_class=MRLinearSolverConfig,
            precision=precision,
            solver_width=solver_width,
            **kwargs,
        )

    def register_buffers(self) -> None:
        """Register device buffers with buffer_registry."""
        config = self.compile_settings
        buffer_registry.register(
            "preconditioned_vec",
            self,
            config.solver_width,
            config.preconditioned_vec_location,
            precision=config.precision,
        )
        buffer_registry.register(
            "temp",
            self,
            config.solver_width,
            config.temp_location,
            precision=config.precision,
        )
        buffer_registry.register(
            "mr_precond_scratch",
            self,
            config.solver_width,
            "local",
            precision=config.precision,
        )
        buffer_registry.register(
            "mr_chain_scratch",
            self,
            config.chain_scratch_elements,
            "local",
            precision=config.precision,
        )

    @property
    def linear_correction_type(self) -> str:
        """Return correction strategy."""
        return self.compile_settings.linear_correction_type

    def build(self) -> LinearSolverCache:
        """Compile linear solver device function.

        Returns
        -------
        LinearSolverCache
            Container with compiled linear_solver device function.
        """
        config = self.compile_settings

        # Device Functions
        operator_apply = config.operator_apply
        preconditioner = config.preconditioner
        scaled_norm_fn = config.norm_device_function

        # Config parameters
        n = config.solver_width
        linear_correction_type = config.linear_correction_type
        max_iters = config.max_iters
        use_cached_auxiliaries = config.use_cached_auxiliaries
        jit_kwargs = self.jit_kwargs

        # Compute flags for correction type
        sd_flag = linear_correction_type == "steepest_descent"
        mr_flag = linear_correction_type == "minimal_residual"
        preconditioned = preconditioner is not None
        chained_precond = config.preconditioner_is_chained
        reference_is_state = config.norm_reference == "state"

        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_iters)
        precision_numba = config.numba_precision
        typed_zero = precision_numba(0.0)
        typed_reduction = config.residual_reduction
        typed_floor = config.residual_floor
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_linear_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED
        )

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_precond = get_alloc("preconditioned_vec", self)
        alloc_temp = get_alloc("temp", self)
        alloc_precond_scratch = get_alloc("mr_precond_scratch", self)
        alloc_chain_scratch = get_alloc("mr_chain_scratch", self)

        # no cover: start
        # Bind the norm's scaling reference at compile time.
        if reference_is_state:
            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def weighted_norm(values, state, base_state):
                return scaled_norm_fn(values, state)
        else:
            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def weighted_norm(values, state, base_state):
                return scaled_norm_fn(values, base_state)
        # no cover: end

        # Build device function based on cached auxiliaries flag
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
                """Run one cached preconditioned steepest-descent or MR solve.

                Parameters
                ----------
                state : array of numba_precision
                    State vector forwarded to operator and preconditioner.
                parameters : array of numba_precision
                    Model parameters forwarded to operator and preconditioner.
                drivers : array of numba_precision
                    External drivers forwarded to operator and preconditioner.
                base_state : array of numba_precision
                    Base state for n-stage operators (unused for single-stage).
                cached_aux : array of numba_precision
                    Cached auxiliary values for the operator and preconditioner.
                t : numba_precision
                    Stage time forwarded to operator and preconditioner.
                h : numba_precision
                    Step size used by the operator evaluation.
                a_ij : numba_precision
                    Stage coefficient forwarded to operator and preconditioner.
                rhs : array of numba_precision
                    Right-hand side; overwritten with the running residual.
                x : array of numba_precision
                    Initial guess; overwritten with the final solution.
                shared : array
                    Shared memory pool.
                persistent_local : array
                    Persistent local memory pool.
                krylov_iters_out : array of int32
                    Single-element array receiving the iteration count.

                Returns
                -------
                int32
                    ``0`` on convergence, ``4`` when the iteration limit
                    is reached.
                """

                # Allocate buffers from registry
                preconditioned_vec = alloc_precond(shared, persistent_local)
                temp = alloc_temp(shared, persistent_local)
                precond_scratch = alloc_precond_scratch(
                    shared, persistent_local
                )
                if chained_precond:
                    chain_scratch = alloc_chain_scratch(
                        shared, persistent_local
                    )
                else:
                    chain_scratch = precond_scratch

                # The stopping target is fixed against the untouched
                # right-hand side before it becomes the residual:
                # ||r|| <= floor + reduction * ||b||.
                rhs_norm2 = weighted_norm(rhs, state, base_state)
                tol = typed_floor + typed_reduction * precision_numba(
                    math_sqrt(rhs_norm2)
                )
                tol2 = tol * tol

                operator_apply(
                    state,
                    parameters,
                    drivers,
                    cached_aux,
                    base_state,
                    t,
                    h,
                    a_ij,
                    x,
                    temp,
                )
                # Compute initial residual rhs = rhs - temp
                for i in range(n_val):
                    rhs[i] = rhs[i] - temp[i]
                acc = weighted_norm(rhs, state, base_state)
                mask = activemask()
                converged = acc <= tol2

                iter_count = int32(0)
                for _ in range(max_iters_val):
                    if all_sync(mask, converged):
                        break

                    iter_count += int32(1)
                    if preconditioned:
                        preconditioner(
                            state,
                            parameters,
                            drivers,
                            cached_aux,
                            base_state,
                            t,
                            h,
                            a_ij,
                            rhs,
                            preconditioned_vec,
                            temp,
                            precond_scratch,
                            chain_scratch,
                        )
                    else:
                        for i in range(n_val):
                            preconditioned_vec[i] = rhs[i]

                    operator_apply(
                        state,
                        parameters,
                        drivers,
                        cached_aux,
                        base_state,
                        t,
                        h,
                        a_ij,
                        preconditioned_vec,
                        temp,
                    )
                    numerator = typed_zero
                    denominator = typed_zero
                    if sd_flag:
                        for i in range(n_val):
                            zi = preconditioned_vec[i]
                            numerator += rhs[i] * zi
                            denominator += temp[i] * zi
                    elif mr_flag:
                        for i in range(n_val):
                            ti = temp[i]
                            numerator += ti * rhs[i]
                            denominator += ti * ti

                    if denominator != typed_zero:
                        alpha = numerator / denominator
                    else:
                        alpha = typed_zero

                    if not converged:
                        for i in range(n_val):
                            x[i] += alpha * preconditioned_vec[i]
                            rhs[i] -= alpha * temp[i]
                    acc = weighted_norm(rhs, state, base_state)

                    converged = converged or (acc <= tol2)

                # Log "exceeded linear iters" status if still not converged
                final_status = selp(
                    converged, success, max_linear_iters_exceeded
                )
                krylov_iters_out[0] = iter_count
                return final_status

            # no cover: end
            return LinearSolverCache(linear_solver=linear_solver_cached)

        else:
            # Device function for non-cached variant
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
                """Run one preconditioned steepest-descent or minimal-residual solve.

                Parameters
                ----------
                state
                    State vector forwarded to the operator and preconditioner.
                parameters
                    Model parameters forwarded to the operator and preconditioner.
                drivers
                    External drivers forwarded to the operator and preconditioner.
                base_state
                    Base state for n-stage operators (unused for single-stage).
                t
                    Stage time forwarded to the operator and preconditioner.
                h
                    Step size used by the operator evaluation.
                a_ij
                    Stage coefficient forwarded to the operator and preconditioner.
                rhs
                    Right-hand side of the linear system. Overwritten with the current
                    residual.
                x
                    Iterand provided as the initial guess and overwritten with the
                    final solution.
                shared
                    Shared memory array for selective buffer allocation.
                persistent_local
                    Persistent local memory array for selective buffer allocation.
                krylov_iters_out
                    Single-element int32 array to receive the iteration count.

                Returns
                -------
                int
                    ``0`` on convergence or ``4`` when the iteration limit is reached.

                Notes
                -----
                ``rhs`` is updated in place to hold the running residual, and ``temp``
                is reused as the scratch vector passed to the preconditioner. The
                iteration therefore keeps just two auxiliary vectors of length ``n``.
                The operator, preconditioner behaviour, and correction strategy are
                fixed by the factory closure, while ``state``, ``parameters``, and
                ``drivers`` are treated as read-only context values.
                """

                # Allocate buffers from registry
                preconditioned_vec = alloc_precond(shared, persistent_local)
                temp = alloc_temp(shared, persistent_local)
                precond_scratch = alloc_precond_scratch(
                    shared, persistent_local
                )
                if chained_precond:
                    chain_scratch = alloc_chain_scratch(
                        shared, persistent_local
                    )
                else:
                    chain_scratch = precond_scratch

                # The stopping target is fixed against the untouched
                # right-hand side before it becomes the residual:
                # ||r|| <= floor + reduction * ||b||.
                rhs_norm2 = weighted_norm(rhs, state, base_state)
                tol = typed_floor + typed_reduction * precision_numba(
                    math_sqrt(rhs_norm2)
                )
                tol2 = tol * tol

                operator_apply(
                    state, parameters, drivers, base_state, t, h, a_ij, x, temp
                )
                # Compute initial residual rhs = rhs - temp
                for i in range(n_val):
                    rhs[i] = rhs[i] - temp[i]
                acc = weighted_norm(rhs, state, base_state)
                mask = activemask()
                converged = acc <= tol2

                iter_count = int32(0)
                for _ in range(max_iters_val):
                    if all_sync(mask, converged):
                        break

                    iter_count += int32(1)
                    if preconditioned:
                        preconditioner(
                            state,
                            parameters,
                            drivers,
                            base_state,
                            t,
                            h,
                            a_ij,
                            rhs,
                            preconditioned_vec,
                            temp,
                            precond_scratch,
                            chain_scratch,
                        )
                    else:
                        for i in range(n_val):
                            preconditioned_vec[i] = rhs[i]

                    operator_apply(
                        state,
                        parameters,
                        drivers,
                        base_state,
                        t,
                        h,
                        a_ij,
                        preconditioned_vec,
                        temp,
                    )
                    numerator = typed_zero
                    denominator = typed_zero
                    if sd_flag:
                        for i in range(n_val):
                            zi = preconditioned_vec[i]
                            numerator += rhs[i] * zi
                            denominator += temp[i] * zi
                    elif mr_flag:
                        for i in range(n_val):
                            ti = temp[i]
                            numerator += ti * rhs[i]
                            denominator += ti * ti

                    if denominator != typed_zero:
                        alpha = numerator / denominator
                    else:
                        alpha = typed_zero

                    if not converged:
                        for i in range(n_val):
                            x[i] += alpha * preconditioned_vec[i]
                            rhs[i] -= alpha * temp[i]
                    acc = weighted_norm(rhs, state, base_state)

                    converged = converged or (acc <= tol2)

                # Log "exceeded linear iters" status if still not converged
                final_status = selp(
                    converged, success, max_linear_iters_exceeded
                )
                krylov_iters_out[0] = iter_count
                return final_status

            # no cover: end
            return LinearSolverCache(linear_solver=linear_solver)
