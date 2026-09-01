"""Consistent-initialisation solver factory for DAE systems.

Compiles a one-shot device function that solves for a consistent
state at ``t0``, called by the loop before the first step and the
t0 save. Mode ``"brown"`` corrects only the zero-mass components
through the ``init_residual``/``init_lu_solve`` helpers;
``"shampine"`` commits one backward-Euler solve of the initial step
size through the standard residual; ``"none"`` (and any system
without algebraic rows) compiles a no-op. Uses a damped Newton
solver with a direct LU inner. Ported from OrdinaryDiffEq.jl's
``BrownFullBasicInit`` and ``ShampineCollocationInit``: Copyright
(c) 2016-2020 ChrisRackauckas, Yingbo Ma, Julia Computing Inc, and
other contributors. MIT; see THIRD_PARTY_LICENSES.

Published Classes
-----------------
:class:`DAEInitialiserConfig`
    Attrs configuration for the initialiser factory.

:class:`DAEInitialiserCache`
    Cache container holding the compiled device function.

:class:`DAEInitialiser`
    CUDAFactory owning a direct LU linear solver and a correction
    norm, compiling the initialisation device function.
"""

from typing import Any, Callable, Dict, Optional, Set, Tuple

from attrs import define, field, frozen, validators
from numpy import finfo as np_finfo
from numpy import int32 as np_int32

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
)
from cubie._utils import (
    PrecisionDType,
    build_config,
    getype_validator,
    is_device_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import (
    activemask,
    all_sync,
    any_sync,
    unroll_if,
    cuda,
    int32,
    selp,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ODEImplicitStep,
)
from cubie.integrators.norms import DIRKCorrectionNorm
from cubie.result_codes import CUBIE_RESULT_CODES

DAE_INITIALISATION_MODES = ("brown", "shampine", "none")
"""Accepted values of the ``dae_initialisation`` setting."""

INIT_NEWTON_MAX_ITERS = 50
"""Fixed Newton iteration cap for the initialisation solve."""

INIT_MAX_BACKTRACKS = 12
"""Halvings tried before a correction is judged non-improving."""


@frozen
class DAEInitialiserConfig(CUDAFactoryConfig):
    """Compile settings for the DAE consistent-initialisation solve.

    Parameters
    ----------
    n
        Number of state variables.
    mass_flags
        Per-state mass-diagonal flags as delivered by
        :attr:`~cubie.odesystems.baseODE.BaseODE.mass_diagonal_flags`:
        ``True`` for a differential row, ``False`` for a torn
        algebraic row.
    dae_initialisation
        Initialisation mode: ``"brown"`` corrects only the
        algebraic components; ``"shampine"`` commits one
        backward-Euler solve of the initial step size; ``"none"``
        disables the pass.
    increment_location
        Memory location for the Newton increment buffer.
    get_solver_helper_fn
        Callable with the ``get_solver_helper`` contract serving
        helper device functions.
    residual_function
        Mode's residual device function, injected at build time.
    linear_solver_function
        Direct-LU solve device function, injected at build time.
    norm_function
        Correction-norm device function, injected at build time.
    """

    n: int = field(default=1, validator=getype_validator(int, 1))
    mass_flags: Tuple[bool, ...] = field(
        default=(),
        validator=validators.deep_iterable(
            validators.instance_of(bool),
            validators.instance_of(tuple),
        ),
    )
    dae_initialisation: str = field(
        default="brown",
        validator=validators.in_(DAE_INITIALISATION_MODES),
    )
    increment_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    get_solver_helper_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(validators.is_callable()),
        eq=False,
    )
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
    norm_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if len(self.mass_flags) != self.n:
            raise ValueError(
                "mass_flags must carry one flag per state: got "
                f"{len(self.mass_flags)} flags for n={self.n}."
            )

    @property
    def is_noop(self) -> bool:
        """Return whether the solve compiles to a no-op."""
        return self.dae_initialisation == "none" or all(
            self.mass_flags
        )


@define
class DAEInitialiserCache(CUDADispatcherCache):
    """Cache container for the compiled initialisation function.

    Attributes
    ----------
    initialise_state
        Compiled CUDA device function correcting the state in place.
    """

    initialise_state: Callable = field(validator=is_device_validator)


class DAEInitialiser(CUDAFactory):
    """Factory for the consistent-initialisation device function.

    Owns a direct LU linear solver and a correction norm, and
    compiles a one-shot damped-Newton device function correcting the
    state at ``t0`` in place.

    Parameters
    ----------
    precision
        Numerical precision for computations.
    n
        Number of state variables.
    mass_flags
        Per-state mass-diagonal flags, ``True`` for a differential
        row.
    **kwargs
        Config fields and Newton solver settings, typically the
        algorithm step's ``settings_dict`` passed wholesale. The
        Newton cap is a fixed 50; ``newton_max_iters``, ``None``
        values, and unrecognised keys are ignored.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        mass_flags,
        **kwargs,
    ) -> None:
        super().__init__()

        kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        tolerance_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ("newton_atol", "newton_rtol")
        }
        lu_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key == "lu_factor_location"
        }
        self.linear_solver = ODEImplicitStep._construct_linear_solver(
            precision=precision,
            solver_width=n,
            norm=None,
            norm_reference="base_state",
            linear_correction_type="lu",
            **lu_kwargs,
        )
        self.norm = DIRKCorrectionNorm(
            precision=precision,
            solver_width=n,
            n=n,
            instance_label="newton",
            **tolerance_kwargs,
        )

        config = build_config(
            DAEInitialiserConfig,
            required={
                "precision": precision,
                "n": int(n),
                "mass_flags": tuple(mass_flags),
            },
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register solve buffers and the linear-solver footprint."""
        config = self.compile_settings
        size = 0 if config.is_noop else config.n
        counter_size = 0 if config.is_noop else 1
        buffer_registry.clear_own(self)
        buffer_registry.register(
            "init_increment",
            self,
            size,
            config.increment_location,
        )
        buffer_registry.register("init_delta", self, size, "local")
        buffer_registry.register("init_residual", self, size, "local")
        buffer_registry.register("init_base", self, size, "local")
        buffer_registry.register(
            "init_lin_iters",
            self,
            counter_size,
            "local",
            dtype=np_int32,
        )
        if not config.is_noop:
            buffer_registry.register_child(
                self, self.linear_solver, name="linear_solver"
            )

    def build_solver_helpers(self) -> None:
        """Wire the mode's residual and LU solve into the solve."""
        config = self.compile_settings

        get_fn = config.get_solver_helper_fn
        if config.dae_initialisation == "brown":
            residual = get_fn("init_residual").device_function
            lu_result = get_fn("init_lu_solve")
        else:
            request_kwargs = {"beta": 1.0, "gamma": 1.0}
            residual = get_fn(
                "residual", **request_kwargs
            ).device_function
            lu_result = get_fn("lu_solve", **request_kwargs)

        self.linear_solver.update(
            lu_solve_function=lu_result.device_function,
            lu_nnz=lu_result.lu_nnz,
        )
        self.update_compile_settings(
            {
                "residual_function": residual,
                "linear_solver_function": (
                    self.linear_solver.device_function
                ),
                "norm_function": self.norm.device_function,
            }
        )

    def build(self) -> DAEInitialiserCache:
        """Compile the initialisation device function.

        Returns
        -------
        DAEInitialiserCache
            Cache holding the compiled device function.
        """
        if self.compile_settings.is_noop:
            # no cover: start
            @cuda.jit(device=True, inline=True, **self.jit_kwargs)
            def initialise_state(
                state,
                parameters,
                drivers,
                t,
                h,
                shared_scratch,
                persistent_scratch,
                counters,
            ):
                """No-op initialisation for disabled or non-DAE runs."""
                return int32(0)

            # no cover: end
            return DAEInitialiserCache(
                initialise_state=initialise_state
            )

        # The helper refresh replaces the settings snapshot; read after.
        self.build_solver_helpers()
        config = self.compile_settings

        residual_function = config.residual_function
        linear_solver_fn = config.linear_solver_function
        norm_function = config.norm_function
        numba_precision = config.numba_precision
        n = int32(config.n)
        unroll_solver_element = config.unroll.solver_element
        max_iters = int32(INIT_NEWTON_MAX_ITERS)
        max_backtracks = int32(INIT_MAX_BACKTRACKS)
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        typed_half = numba_precision(0.5)
        typed_huge = numba_precision(float(np_finfo(config.precision).max))
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
        )
        newton_divergence = int32(CUBIE_RESULT_CODES.NEWTON_DIVERGENCE)
        dae_init_failed = int32(
            CUBIE_RESULT_CODES.DAE_INITIALISATION_FAILED
        )

        get_alloc = buffer_registry.get_allocator
        alloc_increment = get_alloc("init_increment", self, zero=True)
        alloc_delta = get_alloc("init_delta", self)
        alloc_residual = get_alloc("init_residual", self)
        alloc_base = get_alloc("init_base", self)
        alloc_lin_iters = get_alloc("init_lin_iters", self)
        alloc_lin_shared, alloc_lin_persistent = (
            buffer_registry.get_child_allocators(
                self, self.linear_solver, name="linear_solver"
            )
        )

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def initialise_state(
            state,
            parameters,
            drivers,
            t,
            h,
            shared_scratch,
            persistent_scratch,
            counters,
        ):
            """Solve for a consistent t0 state; commit on convergence."""
            increment = alloc_increment(
                shared_scratch, persistent_scratch
            )
            delta = alloc_delta(shared_scratch, persistent_scratch)
            residual = alloc_residual(
                shared_scratch, persistent_scratch
            )
            base = alloc_base(shared_scratch, persistent_scratch)
            lin_iters = alloc_lin_iters(
                shared_scratch, persistent_scratch
            )
            lin_shared = alloc_lin_shared(
                shared_scratch, persistent_scratch
            )
            lin_persistent = alloc_lin_persistent(
                shared_scratch, persistent_scratch
            )

            residual_function(
                increment,
                parameters,
                drivers,
                t,
                h,
                typed_one,
                state,
                residual,
            )
            residual_norm2 = typed_zero
            for i in unroll_if(range(n), unroll_solver_element):
                residual_norm2 += residual[i] * residual[i]
                residual[i] = -residual[i]

            converged = False
            failed = False
            last_lin_status = success
            iters_count = int32(0)
            total_lin_iters = int32(0)
            mask = activemask()
            for _ in range(max_iters):
                if all_sync(mask, converged | failed):
                    break
                active = (not converged) & (not failed)

                for i in unroll_if(range(n), unroll_solver_element):
                    delta[i] = typed_zero
                lin_iters[0] = int32(0)
                lin_status = linear_solver_fn(
                    increment,
                    parameters,
                    drivers,
                    state,
                    increment,
                    t,
                    h,
                    typed_one,
                    residual,
                    delta,
                    lin_shared,
                    lin_persistent,
                    lin_iters,
                )
                judged = active & (lin_status == success)
                last_lin_status = selp(
                    active, lin_status, last_lin_status
                )
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )
                total_lin_iters += selp(
                    active, lin_iters[0], int32(0)
                )

                norm2_dz = norm_function(
                    delta, increment, state, state, typed_one
                )
                nonfinite = not (norm2_dz <= typed_huge)
                # A full step below tolerance commits directly.
                small_step = (
                    judged & (not nonfinite) & (norm2_dz < typed_one)
                )
                if small_step:
                    for i in unroll_if(range(n), unroll_solver_element):
                        increment[i] = increment[i] + delta[i]

                # Halve the step until the residual norm improves.
                for i in unroll_if(range(n), unroll_solver_element):
                    base[i] = increment[i]
                found_step = False
                step_scale = typed_one
                alpha = typed_one
                for _backtrack in range(max_backtracks):
                    active_bt = (
                        judged
                        & (not nonfinite)
                        & (not small_step)
                        & (not found_step)
                    )
                    if not any_sync(mask, active_bt):
                        break
                    if active_bt:
                        for i in unroll_if(range(n), unroll_solver_element):
                            increment[i] = (
                                base[i] + alpha * delta[i]
                            )
                        residual_function(
                            increment,
                            parameters,
                            drivers,
                            t,
                            h,
                            typed_one,
                            state,
                            residual,
                        )
                        trial_norm2 = typed_zero
                        for i in unroll_if(range(n), unroll_solver_element):
                            trial_norm2 += (
                                residual[i] * residual[i]
                            )
                        if trial_norm2 < residual_norm2:
                            for i in unroll_if(range(n), unroll_solver_element):
                                residual[i] = -residual[i]
                            residual_norm2 = trial_norm2
                            step_scale = alpha
                            found_step = True
                    alpha *= typed_half

                backtrack_failed = (
                    judged
                    & (not nonfinite)
                    & (not small_step)
                    & (not found_step)
                )
                if backtrack_failed:
                    for i in unroll_if(range(n), unroll_solver_element):
                        increment[i] = base[i]

                converged = converged | small_step | (
                    found_step
                    & (
                        step_scale * step_scale * norm2_dz
                        < typed_one
                    )
                )
                failed = failed | (
                    active
                    & (
                        nonfinite
                        | (lin_status != success)
                        | backtrack_failed
                    )
                )

            fail_bits = selp(
                failed, newton_divergence, max_iters_exceeded
            )
            fail_bits = selp(
                last_lin_status != success,
                int32(fail_bits | last_lin_status),
                fail_bits,
            )
            counters[0] = iters_count
            counters[1] = total_lin_iters

            # Differential increments are exactly zero; commit all.
            for i in unroll_if(range(n), unroll_solver_element):
                state[i] = state[i] + selp(
                    converged, increment[i], typed_zero
                )
            return selp(
                converged,
                int32(0),
                int32(fail_bits | dae_init_failed),
            )

        # no cover: end
        return DAEInitialiserCache(initialise_state=initialise_state)

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Set[str]:
        """Update initialiser and owned-component parameters.

        Parameters
        ----------
        updates_dict
            Mapping of parameter names to new values.
        silent
            Unrecognised keys never raise here; parents filter.
        **kwargs
            Additional parameters to update.

        Returns
        -------
        set of str
            Names of parameters that were recognised.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)
        if not all_updates:
            return set()

        # The cap and correction type never follow the stage solver.
        child_updates = {
            key: value
            for key, value in all_updates.items()
            if key not in ("newton_max_iters", "linear_correction_type")
        }
        if "n" in all_updates:
            child_updates["solver_width"] = all_updates["n"]

        recognized = self.linear_solver.update(
            child_updates, silent=True
        )
        recognized |= self.norm.update(child_updates, silent=True)
        recognized |= self.update_compile_settings(
            all_updates, silent=True
        )
        recognized |= buffer_registry.update(
            self, updates_dict=all_updates, silent=True
        )
        self.register_buffers()

        return recognized

    @property
    def device_function(self) -> Callable:
        """Return the compiled initialisation device function."""
        return self.get_cached_output("initialise_state")

    @property
    def dae_initialisation(self) -> str:
        """Return the configured initialisation mode."""
        return self.compile_settings.dae_initialisation

    @property
    def is_noop(self) -> bool:
        """Return whether the compiled solve is a no-op."""
        return self.compile_settings.is_noop

    @property
    def newton_max_iters(self) -> int:
        """Return the fixed Newton iteration cap."""
        return INIT_NEWTON_MAX_ITERS

    @property
    def get_solver_helper_fn(self) -> Optional[Callable]:
        """Return the helper getter wired into the initialiser."""
        return self.compile_settings.get_solver_helper_fn
