"""Consistent-initialisation solver factory for DAE systems.

Compiles a one-shot device function that solves for a consistent
state at ``t0``, called by the loop before the first step and the
t0 save. Mode ``"brown"`` corrects only the zero-mass components
through the ``init_residual``/``init_lu_solve`` helpers;
``"shampine"`` commits one backward-Euler solve of the initial step
size through the standard residual; ``"none"`` (and any system
without algebraic rows) compiles a no-op. Ported from
DifferentialEquations.jl's ``BrownFullBasicInit`` and
``ShampineCollocationInit``.

Published Classes
-----------------
:class:`DAEInitialiserConfig`
    Attrs configuration for the initialiser factory.

:class:`DAEInitialiserCache`
    Cache container holding the compiled device function.

:class:`DAEInitialiser`
    CUDAFactory owning a
    :class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    and compiling the initialisation device function.
"""

from typing import Any, Callable, Dict, Optional, Set, Tuple

from attrs import define, field, frozen, validators

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
from cubie.cuda_simsafe import cuda, int32, selp
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ODEImplicitStep,
)
from cubie.result_codes import CUBIE_RESULT_CODES

DAE_INITIALISATION_MODES = ("brown", "shampine", "none")
"""Accepted values of the ``dae_initialisation`` setting."""


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
    solver_function
        Compiled Newton solver device function.
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
    solver_function: Optional[Callable] = field(
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

    Owns a :class:`NewtonKrylov` over a direct LU linear solve and
    compiles a one-shot device function correcting the state at
    ``t0`` in place.

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
        # Fixed 50 cap; the stage solver's budget never applies.
        newton_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ODEImplicitStep._NEWTON_KRYLOV_PARAMS
            and key != "newton_max_iters"
        }
        lu_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key == "lu_factor_location"
        }
        linear_solver = ODEImplicitStep._construct_linear_solver(
            precision=precision,
            solver_width=n,
            norm=None,
            norm_reference="base_state",
            linear_correction_type="lu",
            **lu_kwargs,
        )
        self.solver = NewtonKrylov(
            precision=precision,
            solver_width=n,
            linear_solver=linear_solver,
            newton_max_iters=50,
            **newton_kwargs,
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
        """Register the increment buffer and the solver footprint."""
        config = self.compile_settings
        increment_size = 0 if config.is_noop else config.n
        buffer_registry.clear_own(self)
        buffer_registry.register(
            "init_increment",
            self,
            increment_size,
            config.increment_location,
        )
        if not config.is_noop:
            buffer_registry.register_child(
                self, self.solver, name="solver"
            )

    def build_solver_helpers(self) -> None:
        """Wire the mode's residual and LU solve into the solver."""
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

        self.solver.update(
            lu_solve_function=lu_result.device_function,
            lu_nnz=lu_result.lu_nnz,
            residual_function=residual,
            use_cached_auxiliaries=False,
            solver_width=config.n,
        )
        self.update_compile_settings(
            {"solver_function": self.solver.device_function}
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

        solver_fn = config.solver_function
        numba_precision = config.numba_precision
        n = int32(config.n)
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        dae_init_failed = int32(
            CUBIE_RESULT_CODES.DAE_INITIALISATION_FAILED
        )

        get_alloc = buffer_registry.get_allocator
        alloc_increment = get_alloc("init_increment", self, zero=True)
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
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
            solver_shared = alloc_solver_shared(
                shared_scratch, persistent_scratch
            )
            solver_persistent = alloc_solver_persistent(
                shared_scratch, persistent_scratch
            )

            # increment stands in for the unused cached_aux argument.
            status = solver_fn(
                increment,
                parameters,
                drivers,
                increment,
                t,
                h,
                typed_one,
                state,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )

            # Differential increments are exactly zero; commit all.
            converged = status == int32(0)
            for i in range(n):
                state[i] = state[i] + selp(
                    converged, increment[i], typed_zero
                )
            return selp(
                converged,
                int32(0),
                int32(status | dae_init_failed),
            )

        # no cover: end
        return DAEInitialiserCache(initialise_state=initialise_state)

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Set[str]:
        """Update initialiser and owned-solver parameters.

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
        solver_updates = {
            key: value
            for key, value in all_updates.items()
            if key not in ("newton_max_iters", "linear_correction_type")
        }
        if "n" in all_updates:
            solver_updates["solver_width"] = all_updates["n"]

        recognized = self.solver.update(solver_updates, silent=True)
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
    def get_solver_helper_fn(self) -> Optional[Callable]:
        """Return the helper getter wired into the initialiser."""
        return self.compile_settings.get_solver_helper_fn
