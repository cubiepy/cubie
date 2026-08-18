"""Solver-helper roles, requests, and cache containers.

A :class:`SolverHelperRequest` names a *role* (which mathematical
helper) and the axes ``jacobian_at`` / ``prefactored`` / ``stacked``
(how state and auxiliaries reach the generated code). Requests never
mutate the ODE system's configuration. Roles are declarative
:class:`SolverHelperRole` subclasses collected into
:data:`ROLE_REGISTRY` (and :data:`PRECONDITIONER_ROLES` when they
name a ``preconditioner_type``). Legality derives from the declared
capabilities; ``jacobian_at="step"`` on a role without a Jacobian
normalises to ``"stage"``. Concrete roles and source generation live
in :mod:`cubie.odesystems.symbolic.helper_registry`; this module
never imports the symbolic pipeline.

Published Classes
-----------------
:class:`HelperVariant`
    Enumeration of helper variants.
:class:`SolverHelperRole`
    Declarative base class for helper roles.
:class:`SolverHelperRequest`
    Frozen description of one helper lookup.
:class:`HelperResult`
    A bound helper member: device callable plus typed metadata.
:class:`SolverHelperCache`
    Memoized generated factories and bound members for one live ODE
    build.
"""

from enum import Enum
from typing import Any, Callable, FrozenSet, Optional, Tuple, Type

from attrs import Factory, define, field, frozen, validators

from cubie._utils import inrangetype_validator

__all__ = [
    "HelperVariant",
    "SolverHelperRole",
    "ROLE_REGISTRY",
    "PRECONDITIONER_ROLES",
    "SCALAR_FACTORY_ARGS",
    "SCALED_FACTORY_ARGS",
    "ORDERED_FACTORY_ARGS",
    "SolverHelperRequest",
    "HelperResult",
    "SolverHelperCache",
]


SCALAR_FACTORY_ARGS = ("constants", "precision", "lineinfo")
"""Binding contract of helpers taking no implicit scaling."""

SCALED_FACTORY_ARGS = (
    "constants",
    "precision",
    "beta",
    "gamma",
    "lineinfo",
)
"""Binding contract of helpers scaled by beta and gamma."""

ORDERED_FACTORY_ARGS = (
    "constants",
    "precision",
    "beta",
    "gamma",
    "order",
    "lineinfo",
)
"""Binding contract of preconditioners carrying a series order."""


class HelperVariant(Enum):
    """One member per legal jacobian_at/prefactored/stacked combination."""

    PLAIN = "plain"
    CACHED = "cached"
    AT_STATE = "at_state"
    STACKED_STAGES = "stacked_stages"
    CACHED_STACKED = "cached_stacked"
    PREFACTORED = "prefactored"
    PREFACTORED_STACKED = "prefactored_stacked"

    @property
    def cached(self) -> bool:
        """Return whether this is the cached-auxiliaries variant."""
        return self is HelperVariant.CACHED

    @property
    def stacked_stages(self) -> bool:
        """Return whether this is the flattened all-stages variant."""
        return self is HelperVariant.STACKED_STAGES

    @property
    def uses_cached_aux(self) -> bool:
        """Return whether the generated helper reads cached_aux."""
        return self in (
            HelperVariant.CACHED,
            HelperVariant.CACHED_STACKED,
            HelperVariant.PREFACTORED,
            HelperVariant.PREFACTORED_STACKED,
        )

    @property
    def takes_stage_data(self) -> bool:
        """Return whether the request carries stage coefficient data."""
        return self in (
            HelperVariant.STACKED_STAGES,
            HelperVariant.CACHED_STACKED,
            HelperVariant.PREFACTORED,
            HelperVariant.PREFACTORED_STACKED,
        )


ROLE_REGISTRY = {}
"""All declared roles, keyed by :attr:`SolverHelperRole.name`."""

PRECONDITIONER_ROLES = {}
"""Preconditioner roles keyed by their user-facing type name."""


class SolverHelperRole:
    """Declarative base class for solver-helper roles.

    Subclasses declare capabilities as class attributes and register
    themselves at class creation.

    Attributes
    ----------
    name
        Role identifier used in hashing and factory names.
    jacobian_carrying
        Whether the emitted source evaluates the system Jacobian.
    stacked_capable
        Whether a flattened all-stages variant exists.
    prefactor_capable
        Whether a step-start prefactored substitution variant exists.
    returns_aux_count
        Whether generation returns ``(source, aux_count)`` and the
        imported factory carries an ``aux_count`` attribute.
    returns_lu_nnz
        Whether generation returns ``(source, lu_nnz)`` and the
        imported factory carries an ``lu_nnz`` attribute.
    factory_args
        Names of the factory-binding arguments. Declared, never
        introspected.
    preconditioner_type_name
        ``preconditioner_type`` value this role serves, or ``None``.
    default_preconditioner_order
        Series terms an unset ``preconditioner_order`` resolves to
        for this role.
    """

    name = None
    jacobian_carrying = False
    stacked_capable = False
    prefactor_capable = False
    returns_aux_count = False
    returns_lu_nnz = False
    factory_args = SCALED_FACTORY_ARGS
    preconditioner_type_name = None
    default_preconditioner_order = 0

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ROLE_REGISTRY[cls.name] = cls
        if cls.preconditioner_type_name is not None:
            PRECONDITIONER_ROLES[cls.preconditioner_type_name] = cls

    @classmethod
    def legal_variants(cls) -> FrozenSet[HelperVariant]:
        """Return the variants this role accepts, from its capabilities."""
        variants = {HelperVariant.PLAIN, HelperVariant.CACHED}
        if cls.jacobian_carrying:
            variants.add(HelperVariant.AT_STATE)
        if cls.stacked_capable:
            variants.add(HelperVariant.STACKED_STAGES)
            if cls.jacobian_carrying and not cls.prefactor_capable:
                variants.add(HelperVariant.CACHED_STACKED)
        if cls.prefactor_capable:
            variants.add(HelperVariant.PREFACTORED)
            if cls.stacked_capable:
                variants.add(HelperVariant.PREFACTORED_STACKED)
        return frozenset(variants)

    @classmethod
    def uses_cache_selection(cls, variant: HelperVariant) -> bool:
        """Return whether emitted source keys on the JVP cache plan."""
        return variant in (
            HelperVariant.CACHED,
            HelperVariant.CACHED_STACKED,
        )

    @classmethod
    def prepare_request_kwargs(cls, request) -> dict:
        """Return request kwargs for the companion prepare member."""
        return {"role": "prepare_jac", "jacobian_at": "step"}

    @classmethod
    def generate(cls, system, request, func_name: str) -> Any:
        """Emit generated factory source for one request.

        Parameters
        ----------
        system
            The symbolic ODE system serving the request.
        request
            The immutable helper request.
        func_name
            Name for the generated factory function.

        Returns
        -------
        str or tuple of (str, int)
            Generated source, with the auxiliary count appended when
            :attr:`returns_aux_count` is set.
        """
        raise NotImplementedError(
            f"Role '{cls.name}' declares no source generator; concrete "
            "roles are defined in "
            "cubie.odesystems.symbolic.helper_registry."
        )

    @classmethod
    def validate(cls, system, request, cache_policy) -> None:
        """Run per-request diagnostics; default is a no-op."""
        return None


def _role_converter(value: str) -> Type[SolverHelperRole]:
    """Resolve a role name or preconditioner type name to its class."""
    if value in ROLE_REGISTRY:
        return ROLE_REGISTRY[value]
    if value in PRECONDITIONER_ROLES:
        return PRECONDITIONER_ROLES[value]
    raise ValueError(
        f"Unknown solver-helper role '{value}'. Registered "
        f"roles: {sorted(ROLE_REGISTRY)}; preconditioner types: "
        f"{sorted(PRECONDITIONER_ROLES)}."
    )


_JACOBIAN_AT_VALUES = ("stage", "state", "step")

_AXES_TO_VARIANT = {
    ("stage", False, False): HelperVariant.PLAIN,
    ("stage", False, True): HelperVariant.STACKED_STAGES,
    ("state", False, False): HelperVariant.AT_STATE,
    ("step", False, False): HelperVariant.CACHED,
    ("step", False, True): HelperVariant.CACHED_STACKED,
    ("step", True, False): HelperVariant.PREFACTORED,
    ("step", True, True): HelperVariant.PREFACTORED_STACKED,
}


@frozen
class SolverHelperRequest:
    """Immutable description of one solver-helper lookup.

    Parameters
    ----------
    role
        Helper role, as a :class:`SolverHelperRole` subclass or its
        registered name.
    jacobian_at
        Where the helper's Jacobian lives: ``"stage"`` follows the
        iterate at ``base_state + a_ij * state``, ``"state"`` is the
        state argument itself, ``"step"`` is frozen at the step
        start through ``cached_aux``. ``"step"`` on a role without a
        Jacobian normalises to ``"stage"``.
    prefactored
        Substitute against step-start LU factors instead of
        factorising per call; requires ``jacobian_at="step"``.
    stacked
        Serve one flattened ``s * n`` helper over all stages.
    beta
        Shift scaling applied to the mass-matrix term, where the
        helper consumes it.
    gamma
        Weight applied to the Jacobian term, where the helper
        consumes it.
    preconditioner_order
        Polynomial order of series preconditioners, where the helper
        consumes it; ``None`` resolves to the role's declared
        default.
    stage_coefficients
        Stage coupling matrix for stage-data-consuming requests
        (tableau row tuples).
    stage_nodes
        Stage nodes for stage-data-consuming requests (tableau
        tuple).

    Raises
    ------
    ValueError
        If the axes are inconsistent, the role does not serve the
        combination, or a stage-data-consuming request omits its
        stage data.

    Notes
    -----
    Stage-data-consuming combinations (``stacked=True`` or
    ``prefactored=True``) require stage data at construction; others
    drop it.
    """

    role: Type[SolverHelperRole] = field(converter=_role_converter)
    jacobian_at: str = field(default="stage")
    prefactored: bool = field(default=False, converter=bool)
    stacked: bool = field(default=False, converter=bool)
    beta: float = field(default=1.0, converter=float)
    gamma: float = field(default=1.0, converter=float)
    preconditioner_order: Optional[int] = field(
        default=None,
        validator=validators.optional(inrangetype_validator(int, 0, 2)),
    )
    stage_coefficients: Optional[Tuple[tuple, ...]] = field(default=None)
    stage_nodes: Optional[tuple] = field(default=None)
    variant: HelperVariant = field(
        default=HelperVariant.PLAIN, init=False, repr=False
    )

    def __attrs_post_init__(self):
        if self.preconditioner_order is None:
            object.__setattr__(
                self,
                "preconditioner_order",
                self.role.default_preconditioner_order,
            )
        if self.jacobian_at not in _JACOBIAN_AT_VALUES:
            raise ValueError(
                f"jacobian_at must be one of {_JACOBIAN_AT_VALUES}; "
                f"got '{self.jacobian_at}'."
            )
        if self.prefactored and self.jacobian_at != "step":
            raise ValueError(
                "prefactored=True requires jacobian_at='step'."
            )
        point = self.jacobian_at
        if point == "step" and not self.role.jacobian_carrying:
            point = "stage"
        key = (point, self.prefactored, self.stacked)
        variant = _AXES_TO_VARIANT.get(key)
        if variant is None or variant not in self.role.legal_variants():
            raise ValueError(
                f"Role '{self.role.name}' does not serve "
                f"jacobian_at='{self.jacobian_at}', "
                f"prefactored={self.prefactored}, "
                f"stacked={self.stacked}."
            )
        object.__setattr__(self, "variant", variant)
        if self.variant.takes_stage_data:
            if self.stage_coefficients is None or self.stage_nodes is None:
                raise ValueError(
                    f"Variant '{self.variant.value}' requires stage "
                    "coefficients and stage nodes."
                )
        else:
            object.__setattr__(self, "stage_coefficients", None)
            object.__setattr__(self, "stage_nodes", None)

    def _cubie_canonical_(self) -> tuple:
        """Return the canonical identity of this request."""
        return (
            "SolverHelperRequest",
            self.role.name,
            self.variant.value,
            self.beta,
            self.gamma,
            self.preconditioner_order,
            self.stage_coefficients,
            self.stage_nodes,
        )


@define
class HelperResult:
    """One bound helper member.

    Attributes
    ----------
    device_function
        The compiled device callable.
    cached_auxiliary_count
        Auxiliary slot count for cached Jacobian-carrying members and
        ``prepare_jac`` itself; ``None`` otherwise.
    prepare_jac
        Device callable filling the member's auxiliary cache. Set on
        cached Jacobian-carrying members; ``None`` otherwise.
    lu_nnz
        Factor buffer length for ``lu_solve`` members (zero for a
        scalar-emitted factor); ``None`` otherwise.
    """

    device_function: Callable
    cached_auxiliary_count: Optional[int] = None
    prepare_jac: Optional[Callable] = None
    lu_nnz: Optional[int] = None


@define
class SolverHelperCache:
    """Memoized helper products for one live ODE build.

    The maps are intentionally mutable: they memoize products derived
    from immutable requests and the immutable ODE snapshot. A true ODE
    compile-setting change rebuilds the ODE build product and starts a
    fresh member map.

    Attributes
    ----------
    factories
        Imported generated factories keyed by ``source_hash``.
    members
        Bound helper members keyed by ``member_hash``.
    """

    factories: dict = Factory(dict)
    members: dict = Factory(dict)
