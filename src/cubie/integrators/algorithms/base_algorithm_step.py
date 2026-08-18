"""Base classes and shared configuration for integration step factories.

Published Classes
-----------------
:class:`ButcherTableau`
    Attrs container for Butcher tableau coefficients with typed
    accessors and FSAL detection.

:class:`StepControlDefaults`
    Per-algorithm default settings for step controllers.

:class:`BaseStepConfig`
    Abstract attrs configuration shared by explicit and implicit steps.

:class:`StepCache`
    Cache container for compiled step and optional nonlinear solver
    device functions.

:class:`BaseAlgorithmStep`
    Abstract CUDAFactory base for all integration step implementations.

Constants
---------
:data:`ALL_ALGORITHM_STEP_PARAMETERS`
    Set of all keyword arguments accepted across all algorithm types.

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactory`
    Parent factory class.
:class:`~cubie.integrators.algorithms.ode_explicitstep.ODEExplicitStep`
    Explicit step intermediate base.
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Implicit step intermediate base.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Set, Any, Tuple, Sequence
import warnings

from attrs import define, field, validators, frozen
from numpy import (
    float16 as np_float16,
    float32 as np_float32,
    sum as np_sum,
)

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    is_device_validator,
    opt_getype_validator,
    precision_converter,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
    _CubieConfigBase,
)

ALL_ALGORITHM_STEP_PARAMETERS = {
    "algorithm",
    "precision",
    "n",
    "attempt_dense_prediction",
    "evaluate_f",
    "evaluate_observables",
    "evaluate_driver_at_t",
    "get_solver_helper_fn",
    "driver_del_t",
    "beta",
    "gamma",
    "preconditioner_order",
    "preconditioner_type",
    "krylov_atol",
    "krylov_rtol",
    "krylov_max_iters",
    "krylov_residual_reduction",
    "krylov_residual_floor",
    "linear_correction_type",
    "newton_atol",
    "newton_rtol",
    "newton_max_iters",
    "use_smoothed_error",
    "n_drivers",
    # DIRK buffer location parameters
    "stage_increment_location",
    "stage_increment_history_location",
    "stage_base_location",
    "accumulator_location",
    # Dense stage predictor buffer location parameters
    "previous_step_size_location",
    # ERK buffer location parameters
    "stage_rhs_location",
    "stage_accumulator_location",
    # FIRK buffer location parameters
    "stage_driver_stack_location",
    "stage_state_location",
    # Rosenbrock buffer location parameters
    "stage_store_location",
    "cached_auxiliaries_location",
    # BackwardsEuler buffer location parameters
    "increment_cache_location",
    # CrankNicolson buffer location parameters
    "dxdt_location",
    # MR/SD solver buffer location parameters
    "preconditioned_vec_location",
    "temp_location",
    # BiCGSTAB solver buffer location parameters
    "r0_hat_location",
    "p_location",
    "v_location",
    "tmp_location",
    "s_hat_location",
    "delta_location",
    "residual_location",
    # Newton-Krylov buffer location parameters
    "krylov_iters_local_location",
    "prev_theta_location",
    # Rosenbrock int32 buffer location parameters
    "base_state_placeholder_location",
    "krylov_iters_out_location",
}
"""All keyword arguments accepted by integration step constructors.

These parameters can be passed as keyword arguments to any
:class:`BaseAlgorithmStep` subclass or via
:func:`~cubie.integrators.algorithms.get_algorithm_step`. Parent
components use this set to filter kwargs before forwarding.

.. list-table:: Parameter Summary
   :header-rows: 1

   * - Parameter
     - Accepted By
     - Description
   * - ``algorithm``
     - :func:`get_algorithm_step`
     - Algorithm name or :class:`ButcherTableau` instance.
   * - ``precision``
     - :class:`BaseStepConfig`
     - Floating-point dtype for CUDA computations.
   * - ``n``
     - :class:`BaseStepConfig`
     - Number of state variables.
   * - ``n_drivers``
     - :class:`BaseStepConfig`
     - Number of external driver signals.
   * - ``evaluate_f``
     - :class:`BaseStepConfig`
     - Device function evaluating the ODE RHS.
   * - ``evaluate_observables``
     - :class:`BaseStepConfig`
     - Device function evaluating observables.
   * - ``evaluate_driver_at_t``
     - :class:`BaseStepConfig`
     - Device function evaluating drivers at a given time.
   * - ``get_solver_helper_fn``
     - :class:`BaseStepConfig`
     - Callable returning device helpers for solver construction.
   * - ``driver_del_t``
     - Rosenbrock algorithms
     - Device function for driver time derivative.
   * - ``beta``
     - :class:`ImplicitStepConfig`
     - Implicit integration coefficient on stage derivative.
   * - ``gamma``
     - :class:`ImplicitStepConfig`
     - Implicit integration coefficient on mass matrix product.
   * - ``M``
     - :class:`ImplicitStepConfig`
     - Mass matrix for residual and Jacobian actions.
   * - ``preconditioner_order``
     - :class:`ImplicitStepConfig`
     - Series terms the preconditioner carries; zero on ``jacobi``
       is the plain diagonal solve.
   * - ``preconditioner_type``
     - :class:`ImplicitStepConfig`
     - Preconditioner selection: ``"neumann"`` or ``"jacobi"``.
   * - ``krylov_atol``
     - :class:`LinearSolverBaseConfig`
     - Absolute tolerance for the linear solver.
   * - ``krylov_rtol``
     - :class:`LinearSolverBaseConfig`
     - Relative tolerance for the linear solver.
   * - ``krylov_max_iters``
     - :class:`LinearSolverBaseConfig`
     - Maximum linear solver iterations.
   * - ``krylov_residual_reduction``
     - :class:`LinearSolverBaseConfig`
     - Relative term of the linear stopping rule
       ``||r|| <= floor + reduction * ||b||``.
   * - ``krylov_residual_floor``
     - :class:`LinearSolverBaseConfig`
     - Absolute term of the linear stopping rule, in weighted-norm
       units.
   * - ``linear_correction_type``
     - :class:`LinearSolverBaseConfig`
     - Correction strategy identifier.
   * - ``newton_atol``
     - :class:`NewtonKrylovConfig`
     - Absolute tolerance for Newton iteration.
   * - ``newton_rtol``
     - :class:`NewtonKrylovConfig`
     - Relative tolerance for Newton iteration.
   * - ``newton_max_iters``
     - :class:`NewtonKrylovConfig`
     - Maximum Newton iterations.
   * - ``use_smoothed_error``
     - :class:`ImplicitStepConfig`
     - Use an extra solve to smooth the error estimate.
   * - Buffer location parameters
     - Various algorithm configs
     - Memory location (``'local'`` or ``'shared'``) for
       working buffers. Names follow the pattern
       ``<buffer>_location``.
"""


@frozen
class ButcherTableau(_CubieConfigBase):
    """Generic Butcher tableau object.

    Attributes
    ----------
    a
        `a` matrix of the weights of other substages to the current stage
        gradient
    b
        'b' matrix of weights of the stage gradients to the final estimate (
        row 0) and the next-order-up for error calculation (row 1).
    b_hat
        Embedded weights for the higher-order estimate used when calculating
        an error signal.
    embedded_order
        Classical order of the embedded companion; declared with b_hat.
    c
        'c' vector of the substage times (in proportion of step size)
    order
        Classical order of the accuracy of the method - error grows like O(
        n^order)

    Methods
    -------
    stage_count
        Return the number of stages described by the tableau.
    has_error_estimate
        Returns ``True`` when embedded error weights are supplied.
    typed_rows(rows, numba_precision)
        Returns a given matrix (rows) as precision-typed tuples for each stage.
    """

    a: Tuple[Tuple[float, ...], ...] = field()
    b: Tuple[float, ...] = field()
    c: Tuple[float, ...] = field()
    order: int = field()
    b_hat: Optional[Tuple[float, ...]] = field(default=None)
    # Classical order of the embedded companion described by b_hat.
    embedded_order: Optional[int] = field(
        default=None, validator=opt_getype_validator(int, 1)
    )
    # Calibrated dense-prediction step-ratio ceilings, one per
    # precision; zero disables dense prediction at that precision.
    dense_prediction_ratio_float16: float = field(default=0.0)
    dense_prediction_ratio_float32: float = field(default=0.0)
    dense_prediction_ratio_float64: float = field(default=0.0)

    def __attrs_post_init__(self) -> None:
        """Validate tableau structure after initialisation."""
        super().__attrs_post_init__()
        if self.b_hat is not None and len(self.b_hat) != self.stage_count:
            raise ValueError("b_hat must match the number of stages in b")
        if (self.b_hat is None) != (self.embedded_order is None):
            raise ValueError(
                "b_hat and embedded_order must be declared together"
            )

    def _validate_weight_sums(self) -> None:
        """Validate that solution and embedded weights sum to one.

        Runge--Kutta quadrature weights must sum to one; the RK-family
        tableaus call this from their own ``__attrs_post_init__``.
        Rosenbrock-W weights obey a different consistency condition and
        do not use this check.
        """
        if self.b_hat is not None and abs(1.0 - np_sum(self.b_hat)) > 1e-8:
            raise ValueError("b_hat must sum to one")
        if abs(1.0 - np_sum(self.b)) > 1e-8:
            raise ValueError("b must sum to one")

    def _validate_stage_node_consistency(self) -> None:
        """Validate that every stage node matches its ``A`` row sum.

        Runge--Kutta stage equations evaluate stage ``i`` at the state
        implied by row ``i`` of ``A``, so ``c[i]`` must equal
        ``sum(a[i])`` for the stage times and stage states to describe
        the same point. Predicates that infer stage meaning from ``c``
        (such as first-same-as-last detection) rely on this relation.
        Rosenbrock-W tableaus obey a different consistency condition
        and do not use this check.
        """
        for stage_index in range(self.stage_count):
            row = self.a[stage_index]
            node = self.c[stage_index]
            if abs(node - np_sum(row)) > 1e-8:
                raise ValueError(
                    f"Stage {stage_index} node c={node} does not equal "
                    f"its A row sum {np_sum(row)}; the stage time and "
                    "stage state disagree."
                )

    @property
    def d(self) -> Optional[Tuple[float, ...]]:
        """Return coefficients for embedded error estimation."""

        if self.b_hat is None:
            return None
        return tuple(
            b_value - b_hat_value
            for b_value, b_hat_value in zip(self.b, self.b_hat)
        )

    @property
    def smoothing_gamma(self) -> float:
        """Return the bottom-right ``a`` element for the error smoother."""
        return float(self.a[-1][-1])

    @property
    def supports_smoothed_error(self) -> bool:
        """Return whether the tableau defines a smoothed error estimate."""
        return False

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return len(self.b)

    @property
    def stage_coefficients(self) -> Tuple[Tuple[float, ...], ...]:
        """Return the stage coupling matrix as canonical row tuples."""
        return tuple(tuple(row) for row in self.a)

    @property
    def stage_nodes(self) -> Tuple[float, ...]:
        """Return the stage nodes as a canonical tuple."""
        return tuple(self.c)

    @property
    def has_error_estimate(self) -> bool:
        """Return ``True`` when embedded error weights are supplied."""
        error_coeffs = self.d
        if error_coeffs is None:
            return False
        return any(weight != 0.0 for weight in error_coeffs)

    def typed_rows(
        self,
        rows: Sequence[Sequence[float]],
        numba_precision: type,
    ) -> Tuple[Tuple[float, ...], ...]:
        """Pad and convert tableau rows to the requested precision."""

        typed_rows = []
        for row in rows:
            padded = list(row)
            if len(padded) < self.stage_count:
                padded.extend([0.0] * (self.stage_count - len(padded)))
            typed_rows.append(
                tuple(numba_precision(value) for value in padded)
            )
        return tuple(typed_rows)

    def typed_columns(
        self,
        rows: Sequence[Sequence[float]],
        numba_precision: type,
    ) -> Tuple[Tuple[float, ...], ...]:
        """Transpose and convert tableau rows to the requested precision.

        Pad rows to the configured stage count, convert each entry using
        ``numba_precision``, and return the data in column-major order.
        """
        typed_rows = self.typed_rows(rows, numba_precision)
        stage_count = self.stage_count
        return tuple(
            tuple(row[col_idx] for row in typed_rows)
            for col_idx in range(stage_count)
        )

    def a_flat(self, precision):
        """Return a flattened (1d) row-major version of the `a` matrix."""
        typed_rows = self.typed_rows(self.a, precision)
        flat_list: list = []
        for row in typed_rows:
            flat_list.extend(row)
        return tuple(precision(value) for value in flat_list)

    def explicit_terms(self, precision):
        """
        Return the a matrix in typed column tuples with diagonal and higher
        elements set to zero.

        Parameters
        ----------
        precision

        Returns
        -------
        tuple of tuples of float
        """
        typed_rows = self.typed_rows(self.a, precision)
        stage_count = self.stage_count
        return tuple(
            tuple(
                (row[col_idx] if row_idx > col_idx else precision(0.0))
                for row_idx, row in enumerate(typed_rows)
            )
            for col_idx in range(stage_count)
        )

    def typed_vector(
        self,
        vector: Sequence[float],
        numba_precision: type,
    ) -> Tuple[float, ...]:
        """Return ``vector`` typed with ``numba_precision``."""

        return tuple(numba_precision(value) for value in vector)

    def error_weights(
        self,
        numba_precision: type,
    ) -> Optional[Tuple[float, ...]]:
        """Return precision-typed weights for the embedded error estimate."""

        if not self.has_error_estimate:
            return None
        error_coeffs = self.d
        return self.typed_vector(error_coeffs, numba_precision)

    def dense_prediction_ratio_limit(
        self,
        precision: PrecisionDType,
    ) -> float:
        """Return the calibrated dense-prediction ratio ceiling.

        Dense prediction is applied only while the step-size ratio
        ``next dt / previous dt`` stays at or below this value. Zero
        means dense prediction is unavailable at that precision.
        """

        typed_precision = precision_converter(precision)
        if typed_precision is np_float16:
            value = self.dense_prediction_ratio_float16
        elif typed_precision is np_float32:
            value = self.dense_prediction_ratio_float32
        else:
            value = self.dense_prediction_ratio_float64
        return typed_precision(value)

    @property
    def first_stage_is_explicit(self) -> bool:
        """Return whether the first stage needs no implicit solve."""

        first_row = self.a[0] if self.a else ()
        return len(first_row) == 0 or first_row[0] == 0.0

    @property
    def prediction_sample_stages(self) -> Tuple[int, ...]:
        """Return the stage sampled at each distinct stage time.

        Stages sharing an entry of ``c`` sample the derivative at
        the same time, so only the last such stage contributes a
        sample to dense prediction.
        """

        last_stage_at_node = {}
        for stage, node in enumerate(self.c):
            last_stage_at_node[node] = stage
        return tuple(last_stage_at_node.values())

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the first and last stages align.

        Stage-0 RHS reuse requires the first stage to evaluate at the
        step-start state itself: ``c[0] == 0`` with an all-zero first
        ``A`` row (an explicit first stage). A tableau whose first
        stage is implicit evaluates a different state even when its
        node sits at the step start, so ``c[0]`` alone is not enough.
        """

        return bool(
            self.c
            and self.c[0] == 0.0
            and self.c[-1] == 1.0
            and self.a[-1] == self.b
            and not any(self.a[0])
        )

    @property
    def can_reuse_accepted_start(self) -> bool:
        """Return ``True`` when stage 0 evaluates at the step-start time.

        Licenses reuse of quantities that depend only on the stage
        time, such as driver values sampled at the step start;
        state-dependent reuse additionally needs
        :attr:`first_same_as_last`.
        """

        return bool(self.c and (self.c[0] == 0.0))

    @property
    def accumulates_output(self) -> bool:
        """Returns `False` if one stage's state equals the output."""
        return self.b_matches_a_row is None

    @property
    def accumulates_error(self) -> bool:
        """Returns `False` if one stage's error equals the output."""
        return self.b_hat_matches_a_row is None

    def _find_matching_row(
        self, target_weights: Optional[Tuple[float, ...]]
    ) -> Optional[int]:
        """Find row in coupling matrix that matches target weights.

        Parameters
        ----------
        target_weights : Optional[Tuple[float, ...]]
            Weight vector to match against rows of coupling matrix `a`.
            If None, returns None immediately.

        Returns
        -------
        Optional[int]
            Zero-based row index where a[row] matches target_weights
            within tolerance of 1e-15. If multiple rows match, returns
            the last matching row. Returns None if no match found.
        """
        if target_weights is None:
            return None

        tolerance = 1e-15
        stage_count = self.stage_count
        matching_row = None

        # Iterate through all rows to find matches, preferring the last
        for row_idx in range(len(self.a)):
            row = self.a[row_idx]
            # Compare only up to stage_count elements
            row_slice = row[:stage_count]
            target_slice = target_weights[:stage_count]

            # Check element-wise equality within tolerance
            matches = True
            for i in range(stage_count):
                if abs(row_slice[i] - target_slice[i]) > tolerance:
                    matches = False
                    break

            if matches:
                matching_row = row_idx

        return matching_row

    @property
    def b_matches_a_row(self) -> Optional[int]:
        """Return row index where a[row] equals b, or None if no match.

        This property identifies tableaus where the last stage increment
        already contains the exact combination needed for the proposed
        state, enabling compile-time optimization to avoid redundant
        accumulation.

        Returns
        -------
        Optional[int]
            Zero-based row index where a[row] matches b within tolerance
            of 1e-15, preferring the last matching row if multiple exist.
            Returns None if no match is found.
        """
        return self._find_matching_row(self.b)

    @property
    def b_hat_matches_a_row(self) -> Optional[int]:
        """Return row index where a[row] equals b_hat, or None if no match.

        This property identifies tableaus where a stage increment already
        contains the exact combination needed for the embedded error
        estimate, enabling compile-time optimization to avoid redundant
        accumulation.

        Returns
        -------
        Optional[int]
            Zero-based row index where a[row] matches b_hat within
            tolerance of 1e-15, preferring the last matching row if
            multiple exist. Returns None if b_hat is None or no match
            is found.
        """
        return self._find_matching_row(self.b_hat)


@define
class StepControlDefaults:
    """Per-algorithm defaults for step controller settings."""

    step_controller: Dict[str, Any] = field(factory=dict)

    def copy(self) -> "StepControlDefaults":
        """Return a deep-copy of the defaults container."""
        return StepControlDefaults(
            step_controller=dict(self.step_controller),
        )


@frozen
class BaseStepConfig(CUDAFactoryConfig, ABC):
    """Configuration shared by explicit and implicit integration steps.

    Parameters
    ----------
    precision
        Numerical precision to apply to device buffers. Supported values are
        ``float16``, ``float32``, and ``float64``.
    n
        Number of state entries advanced by each step call.
    n_drivers
        Number of external driver signals consumed by the step (>= 0).
    evaluate_f
        Device function that evaluates the system right-hand side f(t, y).
    evaluate_observables
        Device function that evaluates the system observables.
    evaluate_driver_at_t
        Device function that evaluates driver arrays for a given time t.
    get_solver_helper_fn
        Optional callable that returns device helpers required by the
        nonlinear solver construction.
    tableau
        Butcher tableau of the method; None on tableau-less steps.
    """

    n: int = field(default=1, validator=getype_validator(int, 1))
    n_drivers: int = field(default=0, validator=getype_validator(int, 0))
    evaluate_f: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    evaluate_observables: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    evaluate_driver_at_t: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    get_solver_helper_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(validators.is_callable()),
        eq=False,
    )
    tableau: Optional[ButcherTableau] = field(
        default=None,
        validator=validators.optional(
            validators.instance_of(ButcherTableau)
        ),
    )

    @property
    def settings_dict(self) -> Dict[str, object]:
        """Return a mutable view of the configuration state."""

        return {
            "n": self.n,
            "n_drivers": self.n_drivers,
            "precision": self.precision,
        }

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the first and last stages align.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        if self.tableau is None:
            return False
        return self.tableau.first_same_as_last

    @property
    def can_reuse_accepted_start(self) -> bool:
        """Return ``True`` when the accepted state seeds the next proposal.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        if self.tableau is None:
            return False
        return self.tableau.can_reuse_accepted_start

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        if self.tableau is None:
            return 1
        return self.tableau.stage_count


@define
class StepCache(CUDADispatcherCache):
    """Container for compiled device helpers used by an algorithm step.

    Parameters
    ----------
    step
        Device function that advances the integration state.
    nonlinear_solver
        Optional device function used by implicit methods to perform
        nonlinear solves.
    """

    step: Callable = field(validator=is_device_validator)
    nonlinear_solver: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
    )


class BaseAlgorithmStep(CUDAFactory):
    """Base class implementing cache and configuration handling for steps.

    The class exposes properties and an ``update`` helper shared by concrete
    explicit and implicit algorithms. Concrete subclasses implement
    ``build`` to compile device helpers and provide metadata about resource
    usage.
    """

    #: Linearly-implicit steps own their linear solver directly (no
    #: Newton iteration) and override this to ``True``.
    is_linear = False

    def __init__(
        self,
        config: BaseStepConfig,
        _controller_defaults: StepControlDefaults,
    ) -> None:
        """Initialise the algorithm step with its configuration object and its
        default runtime settings for collaborators.

        Parameters
        ----------
        config
            Configuration describing the algorithm step.
        _controller_defaults
            Per-algorithm default step controller settings.
        """

        super().__init__()
        self._controller_defaults = _controller_defaults.copy()
        self.setup_compile_settings(config)

    def register_buffers(self) -> None:
        """Register buffers required by the algorithm step."""
        pass

    def update(
        self,
        updates_dict: Optional[Dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> Set[str]:
        """Apply configuration updates and invalidate caches when needed.

        Parameters
        ----------
        updates_dict
            Mapping of configuration keys to their new values.
        silent
            When ``True``, suppress warnings about inapplicable keys.
        **kwargs
            Additional configuration updates supplied inline.

        Returns
        -------
        set
            Set of configuration keys that were recognized and updated.

        Raises
        ------
        KeyError
            Raised when an unknown key is provided while ``silent`` is False.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)

        recognised |= buffer_registry.update(self, updates_dict, silent=True)
        self.register_buffers()

        unrecognised = set(updates_dict.keys()) - recognised

        # Check if unrecognized parameters are valid algorithm step parameters
        # but not applicable to this specific algorithm
        valid_but_inapplicable = unrecognised & ALL_ALGORITHM_STEP_PARAMETERS
        truly_invalid = unrecognised - ALL_ALGORITHM_STEP_PARAMETERS

        # Mark valid algorithm parameters as recognized to prevent error propagation
        recognised |= valid_but_inapplicable

        if valid_but_inapplicable and not silent:
            algorithm_type = self.__class__.__name__
            params_str = ", ".join(sorted(valid_but_inapplicable))
            warnings.warn(
                f"Parameters {{{params_str}}} are not recognized by {algorithm_type}; "
                "updates have been ignored.",
                UserWarning,
                stacklevel=2,
            )

        if not silent and truly_invalid:
            raise KeyError(
                f"Unrecognized parameters in update: {truly_invalid}. "
                "These parameters were not updated.",
            )

        return recognised

    @property
    def n_drivers(self) -> int:
        """Return the configured number of external drivers."""

        return int(self.compile_settings.n_drivers)

    @property
    def n(self) -> int:
        """Return the number of state variables advanced per step."""

        return self.compile_settings.n

    @property
    def controller_defaults(self) -> StepControlDefaults:
        """Return per-algorithm default settings for controllers, solvers."""
        return self._controller_defaults.copy()

    @property
    @abstractmethod
    def threads_per_step(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_multistage(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        raise NotImplementedError

    @property
    def tableau(self) -> Optional[ButcherTableau]:
        """Return the configured tableau; None on tableau-less steps."""

        return self.compile_settings.tableau

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the first and last stages align.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        return self.compile_settings.first_same_as_last

    @property
    def can_reuse_accepted_start(self) -> bool:
        """Return ``True`` when the accepted state seeds the next proposal.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        return self.compile_settings.can_reuse_accepted_start

    @property
    @abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def order(self) -> int:
        """Return the classical order of accuracy of the algorithm."""
        raise NotImplementedError

    @property
    def controller_order(self) -> int:
        """Return the order of accuracy used for step-size control."""
        tableau = self.compile_settings.tableau
        if tableau is None or tableau.embedded_order is None:
            return self.order
        return min(self.order, tableau.embedded_order)

    @property
    def step_function(self) -> Callable:
        """Return the cached device function that advances the solution."""
        return self.get_cached_output("step")

    @property
    def settings_dict(self) -> Dict[str, object]:
        """Return the configuration dictionary for the algorithm step."""
        return self.compile_settings.settings_dict

    @property
    def evaluate_f(self) -> Optional[Callable]:
        """Return the compiled device derivative function."""
        return self.compile_settings.evaluate_f

    @property
    def evaluate_observables(self) -> Optional[Callable]:
        """Return the compiled device observables function."""
        return self.compile_settings.evaluate_observables

    @property
    def get_solver_helper_fn(self) -> Optional[Callable]:
        """Return the helper factory used to build solver device functions.

        Returns
        -------
        Callable or None
            Callable that yields device helpers for solver construction when
            available.
        """
        return self.compile_settings.get_solver_helper_fn

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return self.compile_settings.stage_count
