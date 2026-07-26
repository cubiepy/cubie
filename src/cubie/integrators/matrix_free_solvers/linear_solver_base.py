"""Base class for matrix-free linear solvers.

This module provides the abstract base class and shared configuration
for iterative linear solvers (MR/SD and BiCGSTAB) that operate without
forming Jacobian matrices explicitly.

Published Classes
-----------------
:class:`LinearSolverBaseConfig`
    Attrs configuration base for linear solver factories.

:class:`LinearSolverCache`
    Cache container holding the compiled linear solver device function.

:class:`LinearSolverBase`
    Abstract factory base providing shared infrastructure for all
    linear solver variants.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver.MRLinearSolver`
    Minimal-residual / steepest-descent concrete subclass.
:class:`~cubie.integrators.matrix_free_solvers.bicgstab_solver.BiCGSTABSolver`
    BiCGSTAB concrete subclass.
:class:`~cubie.integrators.matrix_free_solvers.base_solver.MatrixFreeSolver`
    Parent factory providing norm and tolerance management.
"""

from abc import abstractmethod
from typing import Callable, Dict, Any, Optional, Set

from attrs import Converter, define, field, validators, frozen
from numpy import finfo as np_finfo
from numpy import ndarray

from cubie._utils import (
    PrecisionDType,
    build_config,
    inrangetype_validator,
    is_device_validator,
    opt_getype_validator,
)
from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
    MatrixFreeSolver,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDADispatcherCache
from cubie.integrators.norms import ScaledNorm


def _default_residual_reduction(value, self_):
    """Resolve ``None`` to machine epsilon at the config precision."""
    if value is None:
        return float(np_finfo(self_.precision).eps)
    return value


def _default_residual_floor(value, self_):
    """Resolve ``None`` to ``sqrt(eps)`` at the config precision."""
    if value is None:
        return float(np_finfo(self_.precision).eps) ** 0.5
    return value


@frozen
class LinearSolverBaseConfig(MatrixFreeSolverConfig):
    """Base configuration for linear solver compilation.

    Attributes
    ----------
    operator_apply : Optional[Callable]
        Device function applying operator F @ v.
    preconditioner : Optional[Callable]
        Device function for approximate inverse preconditioner.
    use_cached_auxiliaries : bool
        Whether to use cached auxiliary arrays (determines signature).
    preconditioner_is_chained : bool
        Whether ``preconditioner`` is a chained composite, which takes
        a trailing ``chain_scratch`` buffer (determines signature).
    norm_reference : str
        Which device-function argument the weighted norm scales
        against: ``"state"`` (direct solves, where the first argument
        holds the model state) or ``"base_state"`` (Newton-owned
        solves, where the first argument holds the stage increment).
    _residual_reduction : Optional[float]
        Factor the weighted residual must fall below, relative to the
        weighted right-hand side, for the solve to stop. ``None``
        resolves to machine epsilon at construction so the floor
        criterion governs;
        :class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
        derives the adaptive controller's ``rtol``, divided by one
        hundred for linearly-implicit (``is_linear``) steps.
    _residual_floor : Optional[float]
        Absolute term of the stopping rule, in weighted-norm units
        (one sits at the ``krylov_atol``/``krylov_rtol`` envelope).
        ``None`` resolves to ``sqrt(eps)`` of the configured
        precision at construction.
    """

    operator_apply: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    preconditioner: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    use_cached_auxiliaries: bool = field(default=False)
    preconditioner_is_chained: bool = field(default=False)
    norm_reference: str = field(
        default="state",
        validator=validators.in_(["state", "base_state"]),
    )
    _residual_reduction: Optional[float] = field(
        default=None,
        converter=Converter(_default_residual_reduction, takes_self=True),
        validator=validators.optional(
            inrangetype_validator(float, 0.0, 1.0)
        ),
        metadata={"prefixed": True},
    )
    _residual_floor: Optional[float] = field(
        default=None,
        converter=Converter(_default_residual_floor, takes_self=True),
        validator=opt_getype_validator(float, 0.0),
        metadata={"prefixed": True},
    )

    @property
    def residual_reduction(self) -> float:
        """Return the relative stopping factor in configured precision."""
        return self.precision(self._residual_reduction)

    @property
    def residual_floor(self) -> float:
        """Return the absolute stopping term in configured precision."""
        return self.precision(self._residual_floor)

    @property
    def chain_scratch_elements(self) -> int:
        """Return the chained-preconditioner scratch buffer length."""
        return (
            self.solver_width
            if self.preconditioner_is_chained
            else 0
        )


@define
class LinearSolverCache(CUDADispatcherCache):
    """Cache container for linear solver outputs.

    Attributes
    ----------
    linear_solver : Callable
        Compiled CUDA device function for linear solving.
    """

    linear_solver: Callable = field(validator=is_device_validator)


class LinearSolverBase(MatrixFreeSolver):
    """Abstract factory base for linear solver device functions.

    Provides shared infrastructure for iterative linear solvers
    including buffer registration, update delegation, and tolerance
    property forwarding.

    Parameters
    ----------
    config_class : type
        Attrs config class for the concrete solver variant.
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Length of residual and search-direction vectors.
    instance_label : str
        Prefix for tolerance parameters.
    norm : ScaledNorm, optional
        Weighted norm used for convergence checks.
    **kwargs
        Forwarded to config class and the norm factory.

    See Also
    --------
    :class:`LinearSolverBaseConfig`
        Base configuration for linear solvers.
    :class:`~cubie.integrators.matrix_free_solvers.linear_solver.MRLinearSolver`
        MR/SD concrete subclass.
    :class:`~cubie.integrators.matrix_free_solvers.bicgstab_solver.BiCGSTABSolver`
        BiCGSTAB concrete subclass.
    """

    def __init__(
        self,
        config_class: type,
        precision: PrecisionDType,
        solver_width: int,
        instance_label: str = "krylov",
        norm: Optional[ScaledNorm] = None,
        **kwargs,
    ) -> None:
        config = build_config(
            config_class,
            required={
                "precision": precision,
                "solver_width": solver_width,
            },
            instance_label=instance_label,
            **kwargs,
        )

        super().__init__(
            precision=precision,
            solver_type="krylov",
            solver_width=solver_width,
            norm=norm,
            **kwargs,
        )

        self.setup_compile_settings(config)
        self.register_buffers()

    @abstractmethod
    def register_buffers(self) -> None:
        """Register device buffers with buffer_registry."""
        ...

    @abstractmethod
    def build(self) -> LinearSolverCache:
        """Compile linear solver device function.

        Returns
        -------
        LinearSolverCache
            Container with compiled linear_solver device function.
        """
        ...

    @property
    @abstractmethod
    def linear_correction_type(self) -> str:
        """Return the correction strategy identifier."""
        ...

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

        recognized = super().update(all_updates, silent=True)

        recognized |= buffer_registry.update(
            self, updates_dict=all_updates, silent=True
        )
        self.register_buffers()

        return recognized

    @property
    def device_function(self) -> Callable:
        """Return cached linear solver device function."""
        return self.get_cached_output("linear_solver")

    @property
    def krylov_atol(self) -> ndarray:
        """Return absolute tolerance array."""
        return self.atol

    @property
    def krylov_rtol(self) -> ndarray:
        """Return relative tolerance array."""
        return self.rtol

    @property
    def krylov_max_iters(self) -> int:
        """Return maximum iterations."""
        return self.max_iters

    @property
    def krylov_residual_reduction(self) -> float:
        """Return the relative residual stopping factor."""
        return self.compile_settings.residual_reduction

    @property
    def krylov_residual_floor(self) -> float:
        """Return the weighted-residual floor."""
        return self.compile_settings.residual_floor

    @property
    def use_cached_auxiliaries(self) -> bool:
        """Return whether cached auxiliaries are used."""
        return self.compile_settings.use_cached_auxiliaries

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return linear solver configuration as dictionary.

        Combines config settings with tolerance arrays from norm factory.

        Returns
        -------
        dict
            Configuration dictionary including krylov_atol and krylov_rtol
            from the norm factory.
        """
        result = dict(self.compile_settings.settings_dict)
        result["krylov_atol"] = self.krylov_atol
        result["krylov_rtol"] = self.krylov_rtol
        result["krylov_residual_reduction"] = self.krylov_residual_reduction
        result["krylov_residual_floor"] = self.krylov_residual_floor
        return result
