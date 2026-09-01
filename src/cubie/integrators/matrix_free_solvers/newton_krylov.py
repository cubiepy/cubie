"""Newton--Krylov solver factories for matrix-free integrators.

This module wraps a linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver_base` to build
Newton iterations suitable for CUDA device execution. The solve
accepts when the warm-started contraction estimate bounds the update
error below the scaled tolerance, or when an update stops contracting
and is under tolerance. A non-finite update exits early.

Published Classes
-----------------
:class:`NewtonKrylovConfig`
    Attrs configuration for the Newton--Krylov solver factory.

:class:`NewtonKrylovCache`
    Cache container holding the compiled Newton--Krylov device
    function.

:class:`NewtonKrylov`
    CUDAFactory subclass that compiles a damped Newton--Krylov
    solver wrapping a
    :class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.base_solver.MatrixFreeSolver`
    Parent factory providing norm and tolerance management.
:class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    Inner linear solver used for the Newton correction equation.
:mod:`cubie.integrators.algorithms.ode_implicitstep`
    Implicit step base class that creates :class:`NewtonKrylov`
    instances.
"""

from math import sqrt as math_sqrt
from typing import Callable, Optional, Set, Dict, Any

from attrs import define, field, validators, frozen
from cubie.cuda_simsafe import cuda, int32
from numpy import finfo as np_finfo
from numpy import int32 as np_int32
from numpy import ndarray

from cubie._utils import (
    PrecisionDType,
    build_config,
    inrangetype_validator,
    is_device_validator,
)
from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
    MatrixFreeSolver,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDADispatcherCache
from cubie.cuda_simsafe import (
    activemask,
    all_sync,
    unroll_if,
    selp,
)
from cubie.result_codes import CUBIE_RESULT_CODES

from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBase,
)
from cubie.integrators.norms import CorrectionNorm, DIRKCorrectionNorm


@frozen
class NewtonKrylovConfig(MatrixFreeSolverConfig):
    """Configuration for NewtonKrylov solver compilation.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Solver vector length.
    max_iters : int
        Maximum Newton iterations permitted, defaulting to eight.
    norm_device_function : Optional[Callable]
        Compiled correction norm for convergence checks.
    residual_function : Optional[Callable]
        Device function evaluating residuals.
    linear_solver_function : Optional[Callable]
        Device function for solving linear systems.
    delta_location : str
        Memory location for delta buffer.
    residual_location : str
        Memory location for residual buffer.
    krylov_iters_local_location : str
        Memory location for the single-element Krylov iteration
        counter buffer.
    prev_theta_location : str
        Memory location for the persistent contraction-estimate
        buffer.

    Notes
    -----
    Tolerance arrays (newton_atol, newton_rtol) are managed by the solver's
    norm factory and accessed via NewtonKrylov.newton_atol/newton_rtol
    properties.
    """

    max_iters: int = field(
        default=8,
        validator=inrangetype_validator(int, 1, 32767),
        metadata={"prefixed": True},
    )
    use_cached_auxiliaries: bool = field(default=False)
    residual_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    linear_solver_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    delta_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    residual_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    krylov_iters_local_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    prev_theta_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return Newton-Krylov configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary. Note: newton_atol and newton_rtol
            are not included here; access them via solver.newton_atol
            and solver.newton_rtol properties which delegate to the
            norm factory.
        """
        return {
            "newton_max_iters": self.max_iters,
            "delta_location": self.delta_location,
            "residual_location": self.residual_location,
            "krylov_iters_local_location": self.krylov_iters_local_location,
            "prev_theta_location": self.prev_theta_location,
        }


@define
class NewtonKrylovCache(CUDADispatcherCache):
    """Cache container for NewtonKrylov outputs.

    Attributes
    ----------
    newton_krylov_solver : Callable
        Compiled CUDA device function for Newton-Krylov solving.
    """

    newton_krylov_solver: Callable = field(validator=is_device_validator)


class NewtonKrylov(MatrixFreeSolver):
    """Factory for Newton--Krylov solver device functions.

    Uses a matrix-free linear solver for each Newton correction.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Solver vector length.
    linear_solver : LinearSolverBase
        :class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
        instance for solving linear systems.
    **kwargs
        Forwarded to :class:`NewtonKrylovConfig` and the norm
        factory. Includes prefixed tolerance parameters
        (``newton_atol``, ``newton_rtol``).

    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        linear_solver: LinearSolverBase,
        norm: Optional[CorrectionNorm] = None,
        **kwargs,
    ) -> None:
        """Initialize NewtonKrylov with parameters.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        solver_width : int
            Solver vector length.
        linear_solver : LinearSolverBase
            Inner linear solver, constructed with
            ``zero_initial_guess=True``.
        norm : CorrectionNorm, optional
            Correction norm. Defaults to DIRK scaling.
        **kwargs
            Newton solver settings.
        """

        if norm is None:
            norm = DIRKCorrectionNorm(
                precision=precision,
                solver_width=solver_width,
                n=solver_width,
                instance_label="newton",
                **kwargs,
            )
        super().__init__(
            precision=precision,
            solver_type="newton",
            solver_width=solver_width,
            norm=norm,
            **kwargs,
        )

        config = build_config(
            NewtonKrylovConfig,
            required={
                "precision": precision,
                "solver_width": solver_width,
                "norm_device_function": self.norm.device_function,
            },
            instance_label="newton",
            **kwargs,
        )

        self.linear_solver = linear_solver
        self._require_child_zero_guess()
        self.setup_compile_settings(config)

        self.register_buffers()

    def _require_child_zero_guess(self) -> None:
        """Reject a linear solver not built for a zero guess."""
        if not self.linear_solver.compile_settings.zero_initial_guess:
            raise ValueError(
                "NewtonKrylov requires a linear solver constructed "
                "with zero_initial_guess=True."
            )

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        # Register buffers with buffer_registry
        config = self.compile_settings

        buffer_registry.register(
            "delta", self, config.solver_width, config.delta_location
        )
        buffer_registry.register(
            "residual",
            self,
            config.solver_width,
            config.residual_location,
        )
        buffer_registry.register(
            "krylov_iters_local",
            self,
            1,
            config.krylov_iters_local_location,
            dtype=np_int32,
        )
        # Warm-started contraction estimate carried between solves.
        buffer_registry.register(
            "prev_theta",
            self,
            1,
            config.prev_theta_location,
            persistent=True,
        )
        # Record the linear solver as a child at registration time so
        # clear_parent cascades reach it before this solver has built;
        # build refreshes the same named registration with real sizes.
        buffer_registry.register_child(
            self, self.linear_solver, name="linear_solver"
        )

    def build(self) -> NewtonKrylovCache:
        """Compile the Newton solver."""
        config = self.compile_settings

        # Extract parameters from config
        residual_function = config.residual_function
        linear_solver_fn = config.linear_solver_function
        correction_norm_fn = config.norm_device_function

        n = config.solver_width
        max_iters = int32(config.max_iters)

        numba_precision = config.numba_precision
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_newton_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
        )
        newton_divergence = int32(CUBIE_RESULT_CODES.NEWTON_DIVERGENCE)
        # Largest finite value; comparison against it is False for
        # inf and NaN.
        typed_huge = numba_precision(float(np_finfo(config.precision).max))
        # Acceptance bound on eta * ||dz||.
        kappa = numba_precision(0.01)
        # First-iteration acceptance bound on ||dz||.
        first_iteration_bound = numba_precision(1.0e-5)
        # Decay floor on the carried contraction estimate.
        theta_decay = numba_precision(0.3)
        # Growth ratio that flags divergence.
        theta_divergence_bound = numba_precision(2.0)
        n_val = int32(n)
        unroll_solver_element = config.unroll.unroll_solver_element

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_delta = get_alloc("delta", self)
        alloc_residual = get_alloc("residual", self)
        alloc_krylov_iters_local = get_alloc("krylov_iters_local", self)
        alloc_prev_theta = get_alloc("prev_theta", self)

        # Get child allocators for linear solver. Named registration
        # keeps re-registration idempotent if the linear solver
        # instance is ever replaced.
        alloc_lin_shared, alloc_lin_persistent = (
            buffer_registry.get_child_allocators(
                self, self.linear_solver, name="linear_solver"
            )
        )

        use_cached_auxiliaries = config.use_cached_auxiliaries

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def newton_krylov_solver(
            stage_increment,
            parameters,
            drivers,
            cached_aux,
            t,
            h,
            a_ij,
            base_state,
            step_start,
            shared_scratch,
            persistent_scratch,
            counters,
        ):
            """Solve the nonlinear system."""

            # Allocate buffers from registry
            delta = alloc_delta(shared_scratch, persistent_scratch)
            residual = alloc_residual(shared_scratch, persistent_scratch)
            prev_theta_store = alloc_prev_theta(
                shared_scratch, persistent_scratch
            )
            lin_shared = alloc_lin_shared(shared_scratch, persistent_scratch)
            lin_persistent = alloc_lin_persistent(
                shared_scratch, persistent_scratch
            )
            krylov_iters_local = alloc_krylov_iters_local(
                shared_scratch, persistent_scratch
            )

            # Warm-started contraction estimate; zero marks a fresh
            # scratch buffer or a failed previous solve. Callers zero
            # the persistent scratch before the first solve.
            stored_theta = prev_theta_store[0]
            prev_theta = selp(
                stored_theta > typed_zero, stored_theta, typed_one
            )

            # RMS norm of the previous accepted full-step correction;
            # zero marks unavailable contraction history.
            ndz_prev = typed_zero

            converged = False
            failed = False
            diverged = False

            # Track the latest active iteration's linear status.
            last_lin_status = success

            iters_count = int32(0)
            total_krylov_iters = int32(0)
            iteration = int32(0)
            mask = activemask()
            for _ in range(max_iters):
                if all_sync(mask, converged | failed):
                    break
                iteration += int32(1)
                active = (not converged) & (not failed)

                residual_function(
                    stage_increment,
                    parameters,
                    drivers,
                    t,
                    h,
                    a_ij,
                    base_state,
                    residual,
                )
                for i in unroll_if(range(n_val), unroll_solver_element):
                    residual[i] = -residual[i]
                    delta[i] = typed_zero

                krylov_iters_local[0] = int32(0)
                # The frozen solve evaluates at the step start.
                if use_cached_auxiliaries:
                    solve_state = step_start
                else:
                    solve_state = stage_increment
                lin_status = linear_solver_fn(
                    solve_state,
                    parameters,
                    drivers,
                    base_state,
                    cached_aux,
                    t,
                    h,
                    a_ij,
                    residual,
                    delta,
                    lin_shared,
                    lin_persistent,
                    krylov_iters_local,
                )

                total_krylov_iters += selp(
                    active, krylov_iters_local[0], int32(0)
                )
                last_lin_status = selp(active, lin_status, last_lin_status)
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )

                norm2_dz = correction_norm_fn(
                    delta,
                    stage_increment,
                    base_state,
                    step_start,
                    a_ij,
                )
                ndz = numba_precision(math_sqrt(norm2_dz))

                # A failed linear solve yields no usable correction:
                # nothing commits and no contraction evidence accrues.
                judged = active & (lin_status == success)
                history = ndz_prev > typed_zero
                ndz_prev_safe = selp(history, ndz_prev, typed_one)
                theta = selp(
                    history,
                    max(theta_decay * prev_theta, ndz / ndz_prev_safe),
                    prev_theta,
                )
                small_first_step = (iteration == int32(1)) & (
                    ndz < first_iteration_bound
                )
                eta_accept = (theta < typed_one) & (
                    theta * ndz < kappa * (typed_one - theta)
                )

                # Flag if theta is growing, exit if non-finite.
                nonfinite = judged & (not (norm2_dz <= typed_huge))
                diverged = diverged | (
                    judged
                    & history
                    & (theta > theta_divergence_bound)
                    & (ndz > typed_one)
                )
                # Accept a non-contracting update under tolerance.
                converged_floor = (
                    judged
                    & history
                    & (ndz >= ndz_prev)
                    & (ndz <= typed_one)
                )
                failed = failed | nonfinite

                commit = (
                    judged
                    & (not nonfinite)
                    & (not converged_floor)
                )
                for i in unroll_if(range(n_val), unroll_solver_element):
                    stage_increment[i] = selp(
                        commit,
                        stage_increment[i] + delta[i],
                        stage_increment[i],
                    )
                converged = (
                    converged
                    | converged_floor
                    | (commit & (eta_accept | small_first_step))
                )
                ndz_prev = selp(commit, ndz, typed_zero)
                prev_theta = selp(judged & history, theta, prev_theta)

            # Store contraction history; failed solves reset it to 1.
            prev_theta_store[0] = selp(
                converged, min(prev_theta, typed_one), typed_one
            )

            fail_bits = selp(failed, int32(0), max_newton_iters_exceeded)
            fail_bits = selp(
                failed | diverged,
                int32(fail_bits | newton_divergence),
                fail_bits,
            )
            fail_bits = selp(
                last_lin_status != success,
                int32(fail_bits | last_lin_status),
                fail_bits,
            )
            final_status = selp(converged, success, fail_bits)

            counters[0] = iters_count
            counters[1] = total_krylov_iters

            return final_status

        # no cover: end
        return NewtonKrylovCache(newton_krylov_solver=newton_krylov_solver)

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Set[str]:
        """Update compile settings and invalidate cache if changed.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of settings to update.
        silent : bool, default False
            If True, suppress warnings about unrecognized keys.
        **kwargs
            Additional settings as keyword arguments.

        Returns
        -------
        set
            Set of recognized parameter names that were updated.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        recognized = set()

        # Guard any swapped-in linear solver before it compiles.
        self._require_child_zero_guess()

        # Forward krylov-prefixed params to linear solver
        recognized |= self.linear_solver.update(all_updates, silent=True)
        # Add linear_solver_function to updates for compile settings
        all_updates["linear_solver_function"] = (
            self.linear_solver.device_function
        )
        recognized |= super().update(all_updates, silent=True)

        # Buffer locations handled by registry
        recognized |= buffer_registry.update(
            self, updates_dict=all_updates, silent=True
        )
        self.register_buffers()

        return recognized

    @property
    def device_function(self) -> Callable:
        """Return cached Newton-Krylov solver device function."""
        return self.get_cached_output("newton_krylov_solver")

    @property
    def newton_atol(self) -> ndarray:
        """Return absolute tolerance array."""
        return self.norm.atol

    @property
    def newton_rtol(self) -> ndarray:
        """Return relative tolerance array."""
        return self.norm.rtol

    @property
    def newton_max_iters(self) -> int:
        """Return maximum Newton iterations."""
        return self.max_iters

    @property
    def krylov_atol(self) -> ndarray:
        """Return the Krylov absolute tolerance array from nested linear
        solver.
        """
        return self.linear_solver.atol

    @property
    def krylov_rtol(self) -> ndarray:
        """Return the Krylov relative tolerance array from nested linear
        solver.
        """
        return self.linear_solver.rtol

    @property
    def krylov_max_iters(self) -> int:
        """Return max linear iterations from nested linear solver."""
        return self.linear_solver.max_iters

    @property
    def krylov_residual_reduction(self) -> float:
        """Return the linear solver's relative stopping factor."""
        return self.linear_solver.krylov_residual_reduction

    @property
    def krylov_residual_floor(self) -> float:
        """Return the linear solver's weighted-residual floor."""
        return self.linear_solver.krylov_residual_floor

    @property
    def linear_correction_type(self) -> str:
        """Return correction type from nested linear solver."""
        return self.linear_solver.linear_correction_type

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return merged Newton and linear solver configuration.

        Combines Newton-level settings from compile_settings with
        linear solver settings from nested linear_solver instance,
        plus tolerance arrays from the norm factory.

        Returns
        -------
        dict
            Merged configuration dictionary containing both Newton
            parameters, linear solver parameters, and tolerance arrays.
        """
        combined = dict(self.linear_solver.settings_dict)
        combined.update(self.compile_settings.settings_dict)
        combined["newton_atol"] = self.newton_atol
        combined["newton_rtol"] = self.newton_rtol
        return combined
