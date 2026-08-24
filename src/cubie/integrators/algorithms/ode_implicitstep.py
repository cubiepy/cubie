"""Infrastructure for implicit integration step implementations.

Published Classes
-----------------
:class:`ImplicitStepConfig`
    Configuration container extending :class:`BaseStepConfig` with
    implicit-specific fields (beta, gamma, preconditioner order).

:class:`ODEImplicitStep`
    Abstract base for implicit algorithms. Owns a
    :class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    or
    :class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    instance and delegates solver parameter updates.

See Also
--------
:class:`~cubie.integrators.algorithms.base_algorithm_step.BaseAlgorithmStep`
    Parent factory class.
:class:`~cubie.integrators.algorithms.ode_explicitstep.ODEExplicitStep`
    Explicit counterpart.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Nonlinear solver consumed by implicit steps.
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Set, Tuple
from warnings import warn

from attrs import field, frozen, validators
from numpy import ndarray

from cubie._utils import (
    inrangetype_validator,
    is_device_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
    AlgorithmDefaults,
)
from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBase,
)
from cubie.integrators.matrix_free_solvers.lu_solver import (
    LUSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.stage_predictors import (
    tableau_supports_dense_prediction,
)
from cubie.odesystems.solver_helpers import PRECONDITIONER_ROLES

_VALID_CORRECTION_TYPES = (
    "steepest_descent",
    "minimal_residual",
    "bicgstab",
    "lu",
)

#: Correction type used when no algorithm or user setting selects one.
DEFAULT_LINEAR_CORRECTION_TYPE = "minimal_residual"

# Correction identifiers mapped to the solver class each selects.
_CORRECTION_TYPE_CLASSES = {
    "steepest_descent": MRLinearSolver,
    "minimal_residual": MRLinearSolver,
    "bicgstab": BiCGSTABSolver,
    "lu": LUSolver,
}


def _validated_correction_type(value: str) -> str:
    """Return ``value`` if it is a recognised correction identifier.

    Raises
    ------
    ValueError
        If ``value`` is not a recognised identifier.
    """
    if value not in _VALID_CORRECTION_TYPES:
        valid = ", ".join(repr(v) for v in _VALID_CORRECTION_TYPES)
        raise ValueError(
            f"linear_correction_type must be one of {valid}; got "
            f"'{value}'."
        )
    return value


@frozen
class ImplicitStepConfig(BaseStepConfig):
    """Configuration settings for implicit integration steps.

    Parameters
    ----------
    beta
        Implicit integration coefficient applied to the stage derivative.
    gamma
        Implicit integration coefficient applied to the mass matrix product.
    preconditioner_order
        Number of series terms the preconditioner carries; order zero
        on ``'jacobi'`` is the plain diagonal solve. Unset, it takes
        the type's default: two for ``'neumann'``, none for
        ``'jacobi'``.
    use_smoothed_error
        Provide a smoothed error to the step-size controller.
    inexact_newton
        Freeze the Newton iteration matrix at the step-start state
        (simplified Newton).
    prefactored
        With ``inexact_newton`` and a direct solver on a diagonal
        tableau, store finished step-start LU factors per distinct
        tableau diagonal instead of frozen Jacobian entries.
    cached_auxiliaries_location
        Buffer location for the step-start Jacobian cache.

    Notes
    -----
    The mass matrix is not an algorithm parameter: it belongs to the
    ODE system, and mass-consuming solver helpers read it from the
    system when generated through ``get_solver_helper_fn``.
    """

    _beta: float = field(
        default=1.0, validator=inrangetype_validator(float, 0, 1)
    )
    _gamma: float = field(
        default=1.0, validator=inrangetype_validator(float, 0, 1)
    )
    _preconditioner_order: Optional[int] = field(
        default=None,
        validator=validators.optional(
            inrangetype_validator(int, 0, 2)
        ),
    )
    preconditioner_type: str = field(
        default="neumann",
        validator=[
            validators.instance_of(str),
            validators.in_(PRECONDITIONER_ROLES),
        ],
    )
    use_smoothed_error: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    inexact_newton: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    prefactored: bool = field(
        default=True, validator=validators.instance_of(bool)
    )
    cached_auxiliaries_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    solver_function = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    prepare_jacobian_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    error_solver_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )

    def __attrs_post_init__(self) -> None:
        """Warn when a smoothing request has no tableau support."""
        super().__attrs_post_init__()
        if self.use_smoothed_error and not self.smoothed_error_capable:
            warn(
                "use_smoothed_error has no effect: the tableau does "
                "not support a smoothed error estimate."
            )

    @property
    def smoothed_error_capable(self) -> bool:
        """Return whether the tableau supports the smoothed estimate."""
        return (
            self.tableau is not None
            and self.tableau.supports_smoothed_error
        )

    @property
    def smoothed_error_enabled(self) -> bool:
        """Return whether smoothing is both requested and capable."""
        return self.use_smoothed_error and self.smoothed_error_capable

    @property
    def smoothing_gamma(self) -> float:
        """Return the tableau smoothing coefficient cast to precision."""
        return self.precision(self.tableau.smoothing_gamma)

    @property
    def solver_width(self) -> int:
        """Return the solver vector length."""
        return self.n

    @property
    def preconditioner_order(self) -> int:
        """Return the series-term count, resolving unset by type."""
        if self._preconditioner_order is not None:
            return int(self._preconditioner_order)
        return PRECONDITIONER_ROLES[
            self.preconditioner_type
        ].default_preconditioner_order

    @property
    def beta(self) -> float:
        """Return the implicit integration beta coefficient."""
        return self.precision(self._beta)

    @property
    def gamma(self) -> float:
        """Return the implicit integration gamma coefficient."""
        return self.precision(self._gamma)

    @property
    def settings_dict(self) -> dict:
        """Return configuration fields as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update(
            {
                "beta": self.beta,
                "gamma": self.gamma,
                "preconditioner_order": self.preconditioner_order,
                "preconditioner_type": self.preconditioner_type,
                "use_smoothed_error": self.use_smoothed_error,
                "inexact_newton": self.inexact_newton,
                "prefactored": self.prefactored,
                "get_solver_helper_fn": self.get_solver_helper_fn,
            }
        )
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    """Base helper for implicit integration algorithms."""

    # Union of parameters accepted by every linear solver class.
    _LINEAR_SOLVER_PARAMS = frozenset(
        {
            "linear_correction_type",
            "krylov_atol",
            "krylov_rtol",
            "krylov_max_iters",
            "krylov_residual_reduction",
            "krylov_residual_floor",
            # MR buffer locations
            "preconditioned_vec_location",
            "temp_location",
            # BiCGSTAB buffer locations
            "r0_hat_location",
            "p_location",
            "v_location",
            "tmp_location",
            "s_hat_location",
            # LU buffer locations
            "lu_factor_location",
        }
    )

    # Parameters accepted by NewtonKrylov
    _NEWTON_KRYLOV_PARAMS = frozenset(
        {
            "newton_atol",
            "newton_rtol",
            "newton_max_iters",
            "delta_location",
            "residual_location",
            "krylov_iters_local_location",
            "prev_theta_location",
        }
    )

    def __init__(
        self,
        config: ImplicitStepConfig,
        _defaults: AlgorithmDefaults,
        **kwargs,
    ) -> None:
        """Initialise the implicit step with its configuration.

        Parameters
        ----------
        config
            Configuration describing the implicit step.
        _defaults
            Algorithm family (e.g. FIRK or DIRK) default settings for
            other solver components.
        **kwargs
            Optional solver parameters (krylov_atol, krylov_max_iters,
            newton_rtol, etc.). None values are ignored and defaults
            from solver config classes are used. ``newton_norm``
            supplies a :class:`CorrectionNorm` for Newton solves;
            ``krylov_norm`` supplies a :class:`ScaledNorm` for the
            linear solver's convergence weighting; when absent each
            solver builds its default.

        Notes
        -----
        The class attribute ``is_linear`` selects the solver
        arrangement: linearly-implicit steps own their linear solver
        directly, all others wrap it in a :class:`NewtonKrylov`.
        """
        super().__init__(config, _defaults)

        # Subclasses that support dense stage prediction construct a
        # DenseStagePredictor here after solver construction.
        self.dense_predictor = None

        # Set by subclasses needing a separate solver for smoothing.
        self.error_solver = None

        newton_norm = kwargs.pop("newton_norm", None)
        krylov_norm = kwargs.pop("krylov_norm", None)

        # Extract kwargs for each solver, filtering None values
        linear_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._LINEAR_SOLVER_PARAMS and v is not None
        }
        newton_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._NEWTON_KRYLOV_PARAMS and v is not None
        }

        solver_width = config.solver_width

        # Newton solves weight the norm by the stage base state,
        # linearly-implicit solves by the model state.
        norm_reference = "state" if self.is_linear else "base_state"

        linear_solver = self._construct_linear_solver(
            precision=config.precision,
            solver_width=solver_width,
            norm=krylov_norm,
            norm_reference=norm_reference,
            zero_initial_guess=not self.is_linear,
            **linear_kwargs,
        )

        if self.is_linear:
            self.solver = linear_solver
        else:
            self.solver = NewtonKrylov(
                precision=config.precision,
                solver_width=solver_width,
                linear_solver=linear_solver,
                norm=newton_norm,
                **newton_kwargs,
            )

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        pass

    @staticmethod
    def _construct_linear_solver(
        precision,
        solver_width,
        norm,
        norm_reference,
        **linear_kwargs,
    ):
        """Construct the linear solver ``linear_correction_type`` selects."""
        correction_type = _validated_correction_type(
            linear_kwargs.pop(
                "linear_correction_type",
                DEFAULT_LINEAR_CORRECTION_TYPE,
            )
        )
        solver_class = _CORRECTION_TYPE_CLASSES[correction_type]
        if solver_class is MRLinearSolver:
            linear_kwargs["linear_correction_type"] = correction_type
        return solver_class(
            precision=precision,
            solver_width=solver_width,
            norm=norm,
            norm_reference=norm_reference,
            **linear_kwargs,
        )

    def _swap_linear_solver(self, new_type: str) -> None:
        """Swap the linear-solver class when the correction type demands.

        A value that crosses the MR/BiCGSTAB class boundary rebuilds
        the linear solver from the outgoing instance's
        ``settings_dict`` and shared norm; the operator and
        preconditioner device functions are re-injected by the next
        ``build_implicit_helpers`` run. Same-type values and
        within-class MR/SD switches change no class and are left to
        the owned solver's own update.

        Parameters
        ----------
        new_type
            Correction strategy identifier from the pending update.
        """
        new_type = _validated_correction_type(new_type)
        norm_reference = "state" if self.is_linear else "base_state"

        replacement = self._replacement_linear_solver(
            self.linear_solver, new_type, norm_reference
        )
        if replacement is not None:
            if self.is_linear:
                self.solver = replacement
            else:
                # NewtonKrylov re-registers the child in its update.
                self.solver.linear_solver = replacement

        if self.error_solver is not None:
            replacement = self._replacement_linear_solver(
                self.error_solver, new_type, "base_state"
            )
            if replacement is not None:
                self.error_solver = replacement

    def _replacement_linear_solver(
        self,
        current: LinearSolverBase,
        new_type: str,
        norm_reference: str,
    ) -> Optional[LinearSolverBase]:
        """Return a rebuilt ``current`` when ``new_type`` changes class.

        Returns ``None`` for a within-class change, which the owned
        solver's own update handles.
        """
        if new_type == current.linear_correction_type:
            return None
        if _CORRECTION_TYPE_CLASSES[new_type] is type(current):
            return None

        carried = current.settings_dict
        carried["linear_correction_type"] = new_type
        replacement = self._construct_linear_solver(
            precision=current.precision,
            solver_width=current.solver_width,
            norm=current.norm,
            norm_reference=norm_reference,
            **carried,
        )
        buffer_registry.clear_parent(current)
        return replacement

    def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
        """Update algorithm and owned solver parameters.

        Parameters
        ----------
        updates_dict : dict, optional
            Mapping of parameter names to new values.
        silent : bool, default=False
            Suppress warnings for unrecognized parameters.
        **kwargs
            Additional parameters to update.

        Returns
        -------
        set[str]
            Names of parameters that were successfully recognized.

        Notes
        -----
        Delegates solver parameters to the owned solver instance and,
        when the algorithm owns a dense stage predictor, predictor
        parameters to the predictor. A ``linear_correction_type``
        value that implies a different linear-solver class replaces
        the linear solver with an instance rebuilt from the old
        solver's ``settings_dict``.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        recognized = set()

        # Step settings first; the solver reads refreshed solver_width.
        recognized |= super().update(all_updates, silent=True)

        # Swap the solver class first so pending parameters apply to
        # the replacement.
        if "linear_correction_type" in all_updates:
            self._swap_linear_solver(all_updates["linear_correction_type"])
            recognized.add("linear_correction_type")

        if "n" in all_updates:
            all_updates["solver_width"] = (
                self.compile_settings.solver_width
            )

        recognized |= self.solver.update(all_updates, silent=True)

        # Push the children's rebuilt device functions into the step
        # settings.
        compiled_functions = {
            "solver_function": self.solver.device_function
        }

        if self.dense_predictor is not None:
            recognized |= self.dense_predictor.update(
                all_updates, silent=True
            )
            compiled_functions["predictor_function"] = (
                self.dense_predictor.device_function
                if self.dense_prediction
                else None
            )

        if self.error_solver is not None:
            # The error solve is single-stage: width n, not s*n.
            recognized |= self.error_solver.update(
                all_updates,
                solver_width=self.compile_settings.n,
                silent=True,
            )

        recognized |= super().update(compiled_functions, silent=True)

        return recognized

    @property
    def smooth_error(self) -> bool:
        """Return whether error smoothing compiles into the step."""
        return bool(
            self.compile_settings.smoothed_error_enabled
            and self.is_adaptive
        )

    @property
    def dense_prediction(self) -> bool:
        """Return whether dense stage prediction compiles into the step.

        True only when the algorithm owns a predictor, prediction is
        requested, the tableau meets the transform preconditions, and
        the tableau carries a positive calibrated ratio ceiling for
        the configured precision.
        """
        if self.dense_predictor is None:
            return False
        config = self.compile_settings
        ratio_limit = float(
            config.tableau.dense_prediction_ratio_limit(
                config.precision
            )
        )
        return bool(
            config.attempt_dense_prediction
            and ratio_limit > 0.0
            and tableau_supports_dense_prediction(config.tableau)
        )

    def build(self) -> StepCache:
        """Create and cache the device helpers for the implicit algorithm.

        Returns
        -------
        StepCache
            Container with the compiled step and nonlinear solver.
        """
        # The helper refresh replaces the settings snapshot; read after.
        self.build_implicit_helpers()
        config = self.compile_settings

        evaluate_f = config.evaluate_f
        numba_precision = config.numba_precision
        n = config.n
        evaluate_observables = config.evaluate_observables
        evaluate_driver_at_t = config.evaluate_driver_at_t
        n_drivers = config.n_drivers
        solver_function = config.solver_function

        return self.build_step(
            evaluate_f,
            evaluate_observables,
            evaluate_driver_at_t,
            solver_function,
            numba_precision,
            n,
            n_drivers,
        )

    @abstractmethod
    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:
        """Build and return the implicit step device function.

        Parameters
        ----------
        evaluate_f
            Device function for evaluating the ODE right-hand side f(t, y).
        evaluate_observables
            Device function for evaluating observables.
        evaluate_driver_at_t
            Optional device function evaluating drivers at arbitrary times.
        solver_function
            Device function for running internal solver.
        numba_precision
            Numba precision for compiled device buffers.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the device step implementation.
        """
        raise NotImplementedError

    def _helper_request_kwargs(self) -> dict:
        """Return the shared request fields from the step settings."""
        config = self.compile_settings
        return {
            "beta": float(config.beta),
            "gamma": float(config.gamma),
            "preconditioner_order": config.preconditioner_order,
            "a_ij": self.baked_stage_diagonal,
        }

    # Stage data for prefactored-LU requests on tableau-less steps.
    _PREFACTOR_STAGE_DATA = None
    # Diagonal baked into direct solves on tableau-less steps.
    _BAKED_STAGE_DIAGONAL = None

    @property
    def _prefactor_stage_data(self) -> Tuple[tuple, tuple]:
        """Return (coefficients, nodes) for prefactored-LU requests."""
        if self._PREFACTOR_STAGE_DATA is not None:
            return self._PREFACTOR_STAGE_DATA
        tableau = self.compile_settings.tableau
        return tableau.stage_coefficients, tableau.stage_nodes

    @property
    def baked_stage_diagonal(self) -> Optional[float]:
        """Return the diagonal baked into direct solves, else ``None``."""
        if self._PREFACTOR_STAGE_DATA is not None:
            return self._BAKED_STAGE_DIAGONAL
        tableau = self.compile_settings.tableau
        if tableau is None:
            return None
        return tableau.equal_diagonals

    def _build_inexact_helpers(
        self, residual: Callable
    ) -> tuple:
        """Wire the frozen-Jacobian solver chain.

        Returns
        -------
        tuple
            The prepare device function and the ``cached_auxiliaries``
            element count.
        """
        config = self.compile_settings
        request_kwargs = self._helper_request_kwargs()
        get_fn = config.get_solver_helper_fn

        if self.uses_direct_solver:
            if config.prefactored:
                coefficients, nodes = self._prefactor_stage_data
                lu_result = get_fn(
                    "lu_solve",
                    jacobian_at="step",
                    prefactored=True,
                    stage_coefficients=coefficients,
                    stage_nodes=nodes,
                    **request_kwargs,
                )
            else:
                lu_result = get_fn(
                    "lu_solve",
                    jacobian_at="step",
                    **request_kwargs,
                )
            prepare_function = lu_result.prepare_jac
            cached_count = lu_result.cached_auxiliary_count
            self.solver.update(
                lu_solve_function=lu_result.device_function,
                lu_nnz=lu_result.lu_nnz,
                residual_function=residual,
                use_cached_auxiliaries=True,
                solver_width=config.solver_width,
            )
        else:
            preconditioner = get_fn(
                config.preconditioner_type,
                jacobian_at="step",
                **request_kwargs,
            ).device_function
            operator_result = get_fn(
                "linear_operator", jacobian_at="step", **request_kwargs
            )
            prepare_function = operator_result.prepare_jac
            cached_count = operator_result.cached_auxiliary_count
            self.solver.update(
                operator_apply=operator_result.device_function,
                preconditioner=preconditioner,
                residual_function=residual,
                use_cached_auxiliaries=True,
                solver_width=config.solver_width,
            )
        return prepare_function, cached_count

    def build_implicit_helpers(self) -> None:
        """Construct the nonlinear solver chain used by implicit methods."""

        config = self.compile_settings
        request_kwargs = self._helper_request_kwargs()

        get_fn = config.get_solver_helper_fn

        # Get device functions from ODE system
        residual = get_fn("residual", **request_kwargs).device_function

        prepare_function = None
        cached_count = 0
        if self.uses_cached_solve:
            prepare_function, cached_count = (
                self._build_inexact_helpers(residual)
            )
        elif self.uses_direct_solver:
            lu_result = get_fn(
                "lu_solve",
                **request_kwargs,
            )
            self.solver.update(
                lu_solve_function=lu_result.device_function,
                lu_nnz=lu_result.lu_nnz,
                residual_function=residual,
                use_cached_auxiliaries=False,
                solver_width=config.solver_width,
            )
        else:
            preconditioner = get_fn(
                config.preconditioner_type, **request_kwargs
            ).device_function
            operator = get_fn(
                "linear_operator", **request_kwargs
            ).device_function

            self.solver.update(
                operator_apply=operator,
                preconditioner=preconditioner,
                residual_function=residual,
                use_cached_auxiliaries=False,
                solver_width=config.solver_width,
            )

        buffer_registry.update_buffer(
            "cached_auxiliaries", self, size=cached_count
        )
        self.update_compile_settings(
            {
                "solver_function": self.solver.device_function,
                "prepare_jacobian_function": prepare_function,
            }
        )

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` to indicate the algorithm is implicit."""
        return True

    @property
    def beta(self) -> float:
        """Return the implicit integration beta coefficient."""

        return self.compile_settings.beta

    @property
    def gamma(self) -> float:
        """Return the implicit integration gamma coefficient."""

        return self.compile_settings.gamma

    @property
    def preconditioner_order(self) -> int:
        """Return the number of preconditioner series terms."""

        return int(self.compile_settings.preconditioner_order)

    @property
    def preconditioner_type(self) -> str:
        """Return the type of preconditioner used by the linear solver."""
        return self.compile_settings.preconditioner_type

    @property
    def krylov_atol(self) -> ndarray:
        """Return the absolute tolerance array for linear solve."""
        return self.solver.krylov_atol

    @property
    def krylov_rtol(self) -> ndarray:
        """Return the relative tolerance array for linear solve."""
        return self.solver.krylov_rtol

    @property
    def krylov_max_iters(self) -> int:
        """Return the maximum number of linear iterations allowed."""
        return int(self.solver.krylov_max_iters)

    @property
    def krylov_residual_reduction(self) -> float:
        """Return the linear solver's relative stopping factor."""
        return self.solver.krylov_residual_reduction

    @property
    def krylov_residual_floor(self) -> float:
        """Return the linear solver's weighted-residual floor."""
        return self.solver.krylov_residual_floor

    @property
    def linear_correction_type(self) -> str:
        """Return the linear correction strategy identifier."""
        return self.solver.linear_correction_type

    @property
    def solver_diagnostics(self) -> Dict[str, Any]:
        """Return the solver settings reported when runs fail."""
        settings = self.compile_settings
        diagnostics = {
            "linear_correction_type": self.linear_correction_type,
            "preconditioner_type": self.preconditioner_type,
            "use_smoothed_error": settings.use_smoothed_error,
            "smoothed_error_capable": settings.smoothed_error_capable,
        }
        if not self.is_linear:
            diagnostics["inexact_newton"] = settings.inexact_newton
        return diagnostics

    @property
    def linear_solver(self) -> LinearSolverBase:
        """Return the linear solver, unwrapping Newton when present."""
        if self.is_linear:
            return self.solver
        return self.solver.linear_solver

    @property
    def uses_direct_solver(self) -> bool:
        """Return whether the linear correction is a direct LU solve."""
        return self.linear_correction_type == "lu"

    @property
    def uses_cached_solve(self) -> bool:
        """Return whether the step runs a frozen-Jacobian solve."""
        return bool(
            self.compile_settings.inexact_newton and not self.is_linear
        )

    @property
    def newton_atol(self) -> Optional[ndarray]:
        """Return the Newton absolute tolerance array."""
        return getattr(self.solver, "newton_atol", None)

    @property
    def newton_rtol(self) -> Optional[ndarray]:
        """Return the Newton relative tolerance array."""
        return getattr(self.solver, "newton_rtol", None)

    @property
    def newton_max_iters(self) -> Optional[int]:
        """Return the maximum allowed Newton iterations."""
        val = getattr(self.solver, "newton_max_iters", None)
        return int(val) if val is not None else None

    @property
    def settings_dict(self) -> dict:
        """Return merged algorithm and solver settings.

        Combines implicit step configuration (beta, gamma, etc.)
        with solver settings (Newton and linear solver parameters).

        Returns
        -------
        dict
            Merged configuration dictionary containing:
            - Base step settings (n, n_drivers, precision) from BaseStepConfig
            - Implicit step settings (beta, gamma, preconditioner_order,
              get_solver_helper_fn) from ImplicitStepConfig
            - Solver settings (newton_atol, krylov_rtol, etc.)
              from NewtonKrylov or LinearSolverBase
            - All buffer location parameters from solver hierarchy
        """
        settings = super().settings_dict
        settings.update(self.solver.settings_dict)
        return settings
