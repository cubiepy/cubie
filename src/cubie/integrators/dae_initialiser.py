"""Consistent-initialisation solver factory for DAE systems.

Compiles a one-shot device function that solves for a consistent
state at ``t0``, called by the loop before the first step and the
t0 save. Mode ``"brown"`` corrects only the zero-mass components
through the ``init_residual``/``init_operator``/``init_lu_solve``
helpers; ``"shampine"`` commits one backward-Euler solve of the
initial step size through the standard residual and operator.

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
from numpy import asarray as np_asarray
from numpy import int8 as np_int8

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
)
from cubie._utils import (
    PrecisionDType,
    build_config,
    getype_validator,
    inrangetype_validator,
    is_device_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import cuda, int32, selp
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    _CORRECTION_TYPE_CLASSES,
    _validated_correction_type,
    ODEImplicitStep,
)
from cubie.odesystems.solver_helpers import PRECONDITIONER_ROLES

DAE_INITIALISATION_MODES = ("brown", "shampine", "none")
"""Accepted values of the ``dae_initialisation`` setting."""


def _flags_converter(value) -> Tuple[bool, ...]:
    """Normalise the algebraic-row flags to a tuple of bools."""
    return tuple(bool(flag) for flag in value)


@frozen
class DAEInitialiserConfig(CUDAFactoryConfig):
    """Compile settings for the DAE consistent-initialisation solve.

    Parameters
    ----------
    n
        Number of state variables.
    algebraic_flags
        Per-state flags, ``True`` for a torn algebraic row (zero
        mass diagonal), ``False`` for a differential row.
    dae_initialisation
        Initialisation mode: ``"brown"`` corrects only the
        algebraic components; ``"shampine"`` commits one
        backward-Euler solve of the initial step size.
    preconditioner_type
        Preconditioner role name used by the ``"shampine"`` mode's
        Krylov solve; the ``"brown"`` mode always uses the identity.
    preconditioner_order
        Series terms for the preconditioner; ``None`` resolves to
        the type's declared default.
    increment_location
        Memory location for the Newton increment buffer.
    get_solver_helper_fn
        Callable with the ``get_solver_helper`` contract serving
        helper device functions.
    solver_function
        Compiled Newton solver device function, refreshed on build.
    """

    n: int = field(default=1, validator=getype_validator(int, 1))
    algebraic_flags: Tuple[bool, ...] = field(
        default=(), converter=_flags_converter
    )
    dae_initialisation: str = field(
        default="brown",
        validator=validators.in_(("brown", "shampine")),
    )
    preconditioner_type: str = field(
        default="jacobi",
        validator=[
            validators.instance_of(str),
            validators.in_(PRECONDITIONER_ROLES),
        ],
    )
    _preconditioner_order: Optional[int] = field(
        default=None,
        validator=validators.optional(
            inrangetype_validator(int, 0, 2)
        ),
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
        if len(self.algebraic_flags) != self.n:
            raise ValueError(
                "algebraic_flags must carry one flag per state: got "
                f"{len(self.algebraic_flags)} flags for n={self.n}."
            )

    @property
    def preconditioner_order(self) -> int:
        """Return the series-term count, resolving unset by type."""
        if self._preconditioner_order is not None:
            return int(self._preconditioner_order)
        return PRECONDITIONER_ROLES[
            self.preconditioner_type
        ].default_preconditioner_order


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

    Owns a :class:`NewtonKrylov` and compiles a one-shot device
    function correcting the state at ``t0`` in place.

    Parameters
    ----------
    precision
        Numerical precision for computations.
    n
        Number of state variables.
    algebraic_flags
        Per-state flags, ``True`` for a torn algebraic row.
    dae_initialisation
        ``"brown"`` or ``"shampine"``.
    **kwargs
        Solver settings forwarded to the owned Newton and linear
        solvers (``newton_atol``, ``krylov_max_iters``,
        ``linear_correction_type``, buffer locations, ...) plus the
        config fields above. Unrecognised keys are ignored.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        algebraic_flags,
        dae_initialisation: str = "brown",
        **kwargs,
    ) -> None:
        super().__init__()

        newton_norm = kwargs.pop("newton_norm", None)
        krylov_norm = kwargs.pop("krylov_norm", None)
        linear_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ODEImplicitStep._LINEAR_SOLVER_PARAMS
            and value is not None
        }
        newton_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ODEImplicitStep._NEWTON_KRYLOV_PARAMS
            and value is not None
        }
        # Cold-start default budget; stage solves warm-start with 8.
        newton_kwargs.setdefault("newton_max_iters", 50)

        linear_solver = ODEImplicitStep._construct_linear_solver(
            precision=precision,
            solver_width=n,
            norm=krylov_norm,
            norm_reference="base_state",
            zero_initial_guess=True,
            **linear_kwargs,
        )
        self.solver = NewtonKrylov(
            precision=precision,
            solver_width=n,
            linear_solver=linear_solver,
            norm=newton_norm,
            **newton_kwargs,
        )

        config = build_config(
            DAEInitialiserConfig,
            required={
                "precision": precision,
                "n": int(n),
                "algebraic_flags": algebraic_flags,
                "dae_initialisation": dae_initialisation,
            },
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register the increment buffer and the solver child."""
        config = self.compile_settings
        buffer_registry.register(
            "init_increment",
            self,
            config.n,
            config.increment_location,
        )
        # Recorded here so clear_parent cascades reach the solver.
        buffer_registry.register_child(self, self.solver, name="solver")

    @property
    def uses_direct_solver(self) -> bool:
        """Return whether the linear correction is a direct LU solve."""
        return (
            self.solver.linear_solver.linear_correction_type == "lu"
        )

    def build_solver_helpers(self) -> None:
        """Wire the mode's helper chain into the owned Newton solver."""
        config = self.compile_settings
        get_fn = config.get_solver_helper_fn

        if config.dae_initialisation == "brown":
            residual = get_fn("init_residual").device_function
            if self.uses_direct_solver:
                lu_result = get_fn("init_lu_solve")
                self.solver.update(
                    lu_solve_function=lu_result.device_function,
                    lu_nnz=lu_result.lu_nnz,
                    residual_function=residual,
                    use_cached_auxiliaries=False,
                    solver_width=config.n,
                )
            else:
                # The brown solve runs unpreconditioned.
                preconditioner = get_fn(
                    "no_preconditioner"
                ).device_function
                operator = get_fn("init_operator").device_function
                self.solver.update(
                    operator_apply=operator,
                    preconditioner=preconditioner,
                    residual_function=residual,
                    use_cached_auxiliaries=False,
                    solver_width=config.n,
                )
        else:
            request_kwargs = {"beta": 1.0, "gamma": 1.0}
            residual = get_fn(
                "residual", **request_kwargs
            ).device_function
            if self.uses_direct_solver:
                lu_result = get_fn("lu_solve", **request_kwargs)
                self.solver.update(
                    lu_solve_function=lu_result.device_function,
                    lu_nnz=lu_result.lu_nnz,
                    residual_function=residual,
                    use_cached_auxiliaries=False,
                    solver_width=config.n,
                )
            else:
                preconditioner = get_fn(
                    config.preconditioner_type,
                    preconditioner_order=config.preconditioner_order,
                    **request_kwargs,
                ).device_function
                operator = get_fn(
                    "linear_operator", **request_kwargs
                ).device_function
                self.solver.update(
                    operator_apply=operator,
                    preconditioner=preconditioner,
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
        # The helper refresh replaces the settings snapshot; read after.
        self.build_solver_helpers()
        config = self.compile_settings

        solver_fn = config.solver_function
        numba_precision = config.numba_precision
        n = int32(config.n)
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        zero_flag = np_int8(0)

        if config.dae_initialisation == "brown":
            commit_row = [
                1 if flag else 0 for flag in config.algebraic_flags
            ]
        else:
            commit_row = [1] * config.n
        commit_flags = np_asarray(commit_row, dtype=np_int8)

        get_alloc = buffer_registry.get_allocator
        alloc_increment = get_alloc("init_increment", self)
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

            for i in range(n):
                increment[i] = typed_zero

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

            converged = status == int32(0)
            for i in range(n):
                commit = converged & (commit_flags[i] != zero_flag)
                state[i] = state[i] + selp(
                    commit, increment[i], typed_zero
                )
            return status

        # no cover: end
        return DAEInitialiserCache(initialise_state=initialise_state)

    def _swap_linear_solver(self, new_type: str) -> None:
        """Swap the linear-solver class when the type demands it."""
        new_type = _validated_correction_type(new_type)
        current = self.solver.linear_solver
        if new_type == current.linear_correction_type:
            return
        if _CORRECTION_TYPE_CLASSES[new_type] is type(current):
            return
        carried = current.settings_dict
        carried["linear_correction_type"] = new_type
        carried["zero_initial_guess"] = True
        replacement = ODEImplicitStep._construct_linear_solver(
            precision=current.precision,
            solver_width=current.solver_width,
            norm=current.norm,
            norm_reference="base_state",
            **carried,
        )
        buffer_registry.clear_parent(current)
        # NewtonKrylov re-registers the child in its update.
        self.solver.linear_solver = replacement

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

        recognized = set()

        if "linear_correction_type" in all_updates:
            self._swap_linear_solver(
                all_updates["linear_correction_type"]
            )
            recognized.add("linear_correction_type")

        if "n" in all_updates:
            all_updates["solver_width"] = all_updates["n"]

        recognized |= self.solver.update(all_updates, silent=True)
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
