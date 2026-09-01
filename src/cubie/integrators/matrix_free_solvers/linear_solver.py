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
from cubie.cuda_simsafe import unroll_if

from cubie._utils import PrecisionDType
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    IterativeLinearSolverConfig,
    IterativeLinearSolverBase,
    LinearSolverCache,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, fmin, selp
from cubie.result_codes import CUBIE_RESULT_CODES


@frozen
class MRLinearSolverConfig(IterativeLinearSolverConfig):
    """Configuration for MRLinearSolver compilation.

    Attributes
    ----------
    linear_correction_type : str
        Line-search strategy ('steepest_descent' or 'minimal_residual').
    """

    linear_correction_type: str = field(
        default="minimal_residual",
        validator=validators.in_(["steepest_descent", "minimal_residual"]),
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
        settings = super().settings_dict
        settings["linear_correction_type"] = self.linear_correction_type
        return settings


class MRLinearSolver(IterativeLinearSolverBase):
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
        )
        buffer_registry.register(
            "temp",
            self,
            config.solver_width,
            config.temp_location,
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
        jit_kwargs = self.jit_kwargs

        # Compute flags for correction type
        sd_flag = linear_correction_type == "steepest_descent"
        mr_flag = linear_correction_type == "minimal_residual"
        preconditioned = preconditioner is not None
        zero_initial_guess = config.zero_initial_guess
        reference_is_state = config.norm_reference == "state"

        # Convert types for device function
        n_val = int32(n)
        unroll = config.unroll
        max_iters_val = int32(max_iters)
        precision_numba = config.numba_precision
        typed_zero = precision_numba(0.0)
        typed_reduction = config.residual_reduction
        typed_floor = config.residual_floor
        typed_largest = config.largest_finite
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_linear_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED
        )

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_precond = get_alloc("preconditioned_vec", self)
        alloc_temp = get_alloc("temp", self)

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
            """Run one preconditioned SD/MR solve; overwrites rhs and x."""

            # Allocate buffers from registry
            preconditioned_vec = alloc_precond(shared, persistent_local)
            temp = alloc_temp(shared, persistent_local)

            # Target: ||r|| <= floor + reduction * ||b||.
            rhs_norm2 = weighted_norm(rhs, state, base_state)
            tol = typed_floor + typed_reduction * precision_numba(
                math_sqrt(rhs_norm2)
            )
            tol2 = fmin(tol * tol, typed_largest)

            if zero_initial_guess:
                acc = rhs_norm2
            else:
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
                for i in unroll_if(range(n_val), unroll.solver_element):
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
                    )
                else:
                    for i in unroll_if(range(n_val), unroll.solver_element):
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
                    for i in unroll_if(range(n_val), unroll.solver_element):
                        zi = preconditioned_vec[i]
                        numerator += rhs[i] * zi
                        denominator += temp[i] * zi
                elif mr_flag:
                    for i in unroll_if(range(n_val), unroll.solver_element):
                        ti = temp[i]
                        numerator += ti * rhs[i]
                        denominator += ti * ti

                if denominator != typed_zero:
                    alpha = numerator / denominator
                else:
                    alpha = typed_zero

                if not converged:
                    for i in unroll_if(range(n_val), unroll.solver_element):
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
