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
from typing import Callable, Optional, Union, Set

from attrs import field, validators, frozen
from numpy import ndarray

from cubie._utils import (
    inrangetype_validator,
    is_device_validator,
    sequence_to_tuple,
)
from cubie.buffer_registry import buffer_registry
from cubie.odesystems.solver_helpers import (
    SolverHelperKind,
    SolverHelperRequest,
    resolve_chained_kind,
    resolve_preconditioner_kind,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBase,
)
from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.stage_predictors import (
    tableau_supports_dense_prediction,
)

_VALID_CORRECTION_TYPES = (
    "steepest_descent",
    "minimal_residual",
    "bicgstab",
)


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
        Order of the truncated Neumann preconditioner.

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
    preconditioner_order: int = field(
        default=2, validator=inrangetype_validator(int, 1, 32)
    )
    preconditioner_type: Union[str, tuple] = field(
        default="neumann",
        converter=sequence_to_tuple,
    )
    solver_function = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )

    @property
    def solver_width(self) -> int:
        """Return the solver vector length."""
        return self.n

    @property
    def preconditioner_is_chained(self) -> bool:
        """Return whether the preconditioner composes multiple types."""
        return (
            isinstance(self.preconditioner_type, tuple)
            and len(self.preconditioner_type) >= 2
        )

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
                "get_solver_helper_fn": self.get_solver_helper_fn,
            }
        )
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    """Base helper for implicit integration algorithms."""

    # Union of parameters accepted by all linear solver types.
    # Params not applicable to the chosen solver are silently
    # ignored during construction.
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
        }
    )

    # Parameters accepted by NewtonKrylov
    _NEWTON_KRYLOV_PARAMS = frozenset(
        {
            "newton_atol",
            "newton_rtol",
            "newton_max_iters",
            "newton_stop_criterion",
            "newton_residual_atol",
            "delta_location",
            "residual_location",
            "krylov_iters_local_location",
            "prev_theta_location",
        }
    )

    def __init__(
        self,
        config: ImplicitStepConfig,
        _controller_defaults: StepControlDefaults,
        **kwargs,
    ) -> None:
        """Initialise the implicit step with its configuration.

        Parameters
        ----------
        config
            Configuration describing the implicit step.
        _controller_defaults
           Per-algorithm default runtime collaborators.
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
        super().__init__(config, _controller_defaults)

        # Subclasses that support dense stage prediction construct a
        # DenseStagePredictor here after solver construction.
        self.dense_predictor = None

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
        """Construct the linear solver ``linear_correction_type`` selects.

        ``"bicgstab"`` selects :class:`BiCGSTABSolver`; the MR/SD
        identifiers (and absence) select :class:`MRLinearSolver`.
        Keys in ``linear_kwargs`` the selected configuration does not
        define are ignored.
        """
        correction_type = _validated_correction_type(
            linear_kwargs.pop("linear_correction_type", "minimal_residual")
        )
        if correction_type == "bicgstab":
            return BiCGSTABSolver(
                precision=precision,
                solver_width=solver_width,
                norm=norm,
                norm_reference=norm_reference,
                **linear_kwargs,
            )
        return MRLinearSolver(
            precision=precision,
            solver_width=solver_width,
            linear_correction_type=correction_type,
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
        current = self.linear_solver
        if new_type == current.linear_correction_type:
            return
        if "bicgstab" not in (new_type, current.linear_correction_type):
            return

        carried = current.settings_dict
        carried["linear_correction_type"] = new_type
        replacement = self._construct_linear_solver(
            precision=current.precision,
            solver_width=current.solver_width,
            norm=current.norm,
            norm_reference="state" if self.is_linear else "base_state",
            **carried,
        )

        buffer_registry.clear_parent(current)
        if self.is_linear:
            self.solver = replacement
        else:
            # NewtonKrylov re-registers the named child registration
            # when its update runs later in the same update pass.
            self.solver.linear_solver = replacement

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

        # Step settings first, so the solver update below reads the
        # refreshed solver_width.
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

        recognized |= super().update(compiled_functions, silent=True)

        return recognized

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
        }

    def _resolve_preconditioner(
        self,
        cached: bool = False,
        n_stage: bool = False,
        **request_kwargs,
    ) -> Callable:
        """Resolve ``preconditioner_type`` into a device function.

        Parameters
        ----------
        cached
            Request the cached-auxiliaries variant (Rosenbrock-W).
        n_stage
            Request the flattened all-stages variant (FIRK).
        **request_kwargs
            Request fields forwarded to the helper request.

        Returns
        -------
        Callable
            One generated preconditioner: a concrete kind for a
            single configured type, or one composed (chained)
            generated helper for a sequence of types.
        """
        config = self.compile_settings
        preconditioner_type = config.preconditioner_type
        if isinstance(preconditioner_type, str):
            types = (preconditioner_type,)
        else:
            types = tuple(preconditioner_type)

        kinds = tuple(
            resolve_preconditioner_kind(
                type_name, cached=cached, n_stage=n_stage
            )
            for type_name in types
        )

        if len(kinds) == 1:
            request = SolverHelperRequest(
                kind=kinds[0], **request_kwargs
            )
        else:
            request = SolverHelperRequest(
                kind=resolve_chained_kind(
                    cached=cached, n_stage=n_stage
                ),
                chained_kinds=kinds,
                **request_kwargs,
            )
        get_fn = config.get_solver_helper_fn
        return get_fn(request).device_function

    def build_implicit_helpers(self) -> None:
        """Construct the nonlinear solver chain used by implicit methods.

        Populates the owned solver with operator, preconditioner, and
        residual device functions, then stores the compiled solver
        function in compile settings.
        """

        config = self.compile_settings
        request_kwargs = self._helper_request_kwargs()

        get_fn = config.get_solver_helper_fn

        # Get device functions from ODE system
        preconditioner = self._resolve_preconditioner(**request_kwargs)
        residual = get_fn(
            SolverHelperRequest(
                kind=SolverHelperKind.STAGE_RESIDUAL, **request_kwargs
            )
        ).device_function
        operator = get_fn(
            SolverHelperRequest(
                kind=SolverHelperKind.LINEAR_OPERATOR, **request_kwargs
            )
        ).device_function

        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            preconditioner_is_chained=(
                config.preconditioner_is_chained
            ),
            residual_function=residual,
            solver_width=config.solver_width,
        )

        self.update_compile_settings(
            solver_function=self.solver.device_function
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
        """Return the order of the Neumann preconditioner."""

        return int(self.compile_settings.preconditioner_order)

    @property
    def preconditioner_type(self) -> Union[str, list]:
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
    def linear_solver(self) -> LinearSolverBase:
        """Return the linear solver, unwrapping Newton when present."""
        if self.is_linear:
            return self.solver
        return self.solver.linear_solver

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
