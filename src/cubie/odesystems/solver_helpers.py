"""Solver-helper roles, requests, and cache containers.

A :class:`SolverHelperRequest` names a *role* (which mathematical
helper) and a *variant* (how state and auxiliaries reach the
generated code). Requests never mutate the ODE system's
configuration. Roles are declarative :class:`SolverHelperRole`
subclasses collected into :data:`ROLE_REGISTRY` (and
:data:`PRECONDITIONER_ROLES` when they name a
``preconditioner_type``). Variant legality derives from the declared
capabilities; ``CACHED`` on a role without a Jacobian normalises to
``PLAIN``. Concrete roles and source generation live in
:mod:`cubie.odesystems.symbolic.helper_registry`; this module never
imports the symbolic pipeline.

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
    """How state and auxiliaries reach a generated helper.

    ``PLAIN`` evaluates J at ``base_state + a_ij * state``. ``CACHED``
    reads auxiliaries from the buffer ``prepare_jac`` fills.
    ``AT_STATE`` evaluates J at ``state``; ``a_ij`` scales only.
    ``STACKED_STAGES`` flattens all stages into one ``s * n`` helper.
    """

    PLAIN = "plain"
    CACHED = "cached"
    AT_STATE = "at_state"
    STACKED_STAGES = "stacked_stages"

    @property
    def cached(self) -> bool:
        """Return whether this is the cached-auxiliaries variant."""
        return self is HelperVariant.CACHED

    @property
    def stacked_stages(self) -> bool:
        """Return whether this is the flattened all-stages variant."""
        return self is HelperVariant.STACKED_STAGES


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
        return frozenset(variants)

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


def _variant_converter(value: Any) -> HelperVariant:
    """Accept a variant enum member or its string value."""
    if isinstance(value, HelperVariant):
        return value
    return HelperVariant(value)


@frozen
class SolverHelperRequest:
    """Immutable description of one solver-helper lookup.

    Parameters
    ----------
    role
        Helper role, as a :class:`SolverHelperRole` subclass or its
        registered name.
    variant
        Helper variant, as an enum member or its string value.
        ``CACHED`` on a role without a Jacobian normalises to
        ``PLAIN``.
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
        Stage coupling matrix for ``STACKED_STAGES`` requests
        (tableau row tuples).
    stage_nodes
        Stage nodes for ``STACKED_STAGES`` requests (tableau tuple).

    Raises
    ------
    ValueError
        If the role does not accept the variant, or a stacked request
        omits its stage data.

    Notes
    -----
    ``STACKED_STAGES`` requires stage data at construction; other
    variants drop it.
    """

    role: Type[SolverHelperRole] = field(converter=_role_converter)
    variant: HelperVariant = field(
        default=HelperVariant.PLAIN, converter=_variant_converter
    )
    beta: float = field(default=1.0, converter=float)
    gamma: float = field(default=1.0, converter=float)
    preconditioner_order: Optional[int] = field(
        default=None,
        validator=validators.optional(inrangetype_validator(int, 0, 2)),
    )
    stage_coefficients: Optional[Tuple[tuple, ...]] = field(default=None)
    stage_nodes: Optional[tuple] = field(default=None)

    def __attrs_post_init__(self):
        if self.preconditioner_order is None:
            object.__setattr__(
                self,
                "preconditioner_order",
                self.role.default_preconditioner_order,
            )
        if (
            self.variant is HelperVariant.CACHED
            and not self.role.jacobian_carrying
        ):
            object.__setattr__(self, "variant", HelperVariant.PLAIN)
        if self.variant not in self.role.legal_variants():
            raise ValueError(
                f"Role '{self.role.name}' does not accept variant "
                f"'{self.variant.value}'."
            )
        if self.variant.stacked_stages:
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
