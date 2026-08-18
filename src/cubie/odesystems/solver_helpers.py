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

import sympy as sp
from attrs import Factory, define, field, frozen

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
    factory_args
        Names of the factory-binding arguments. Declared, never
        introspected.
    preconditioner_type_name
        ``preconditioner_type`` value this role serves, or ``None``.
    """

    name = None
    jacobian_carrying = False
    stacked_capable = False
    returns_aux_count = False
    factory_args = SCALED_FACTORY_ARGS
    preconditioner_type_name = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise TypeError(
                f"{cls.__name__} must declare a 'name' class attribute."
            )
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


def _role_converter(value: Any) -> Type[SolverHelperRole]:
    """Accept a role class or its registered name."""
    if isinstance(value, str):
        try:
            return ROLE_REGISTRY[value]
        except KeyError:
            raise ValueError(
                f"Unknown solver-helper role '{value}'. Registered "
                f"roles: {sorted(ROLE_REGISTRY)}."
            ) from None
    if isinstance(value, type) and issubclass(value, SolverHelperRole):
        return value
    raise TypeError(
        "role must be a SolverHelperRole subclass or a registered "
        f"role name; got {value!r}."
    )


def _variant_converter(value: Any) -> HelperVariant:
    """Accept a variant enum member or its string value."""
    if isinstance(value, HelperVariant):
        return value
    return HelperVariant(value)


def _stage_value_repr(value: Any) -> str:
    """Return the canonical text form of one stage entry."""
    return sp.srepr(sp.sympify(value))


def _stage_matrix_converter(value: Any) -> Optional[Tuple[tuple, ...]]:
    """Normalise stage coefficients to a tuple of row tuples."""
    if value is None:
        return None
    return tuple(tuple(row) for row in value)


def _stage_vector_converter(value: Any) -> Optional[tuple]:
    """Normalise stage nodes to a tuple."""
    if value is None:
        return None
    return tuple(value)


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
        Polynomial order of Neumann preconditioners, where the helper
        consumes it.
    stage_coefficients
        Stage coupling matrix for ``STACKED_STAGES`` requests,
        row-major. Entries may be floats or exact SymPy numbers.
    stage_nodes
        Stage nodes expressed as timestep fractions for
        ``STACKED_STAGES`` requests.

    Raises
    ------
    ValueError
        If the role does not accept the variant, or the stage data do
        not match the variant.

    Notes
    -----
    Stage identity uses the canonical SymPy text form: exact and
    floating tableau entries hash separately. ``STACKED_STAGES``
    requires stage data at construction; other variants reject it.
    """

    role: Type[SolverHelperRole] = field(converter=_role_converter)
    variant: HelperVariant = field(
        default=HelperVariant.PLAIN, converter=_variant_converter
    )
    beta: float = field(default=1.0, converter=float)
    gamma: float = field(default=1.0, converter=float)
    preconditioner_order: int = field(default=2, converter=int)
    stage_coefficients: Optional[Tuple[tuple, ...]] = field(
        default=None, converter=_stage_matrix_converter, eq=False
    )
    stage_nodes: Optional[tuple] = field(
        default=None, converter=_stage_vector_converter, eq=False
    )
    _stage_identity: Optional[tuple] = field(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
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
            rows = tuple(
                tuple(_stage_value_repr(value) for value in row)
                for row in self.stage_coefficients
            )
            nodes = tuple(
                _stage_value_repr(value) for value in self.stage_nodes
            )
            object.__setattr__(self, "_stage_identity", (rows, nodes))
        elif (
            self.stage_coefficients is not None
            or self.stage_nodes is not None
        ):
            raise ValueError(
                f"Variant '{self.variant.value}' does not consume "
                "stage coefficients or stage nodes."
            )

    @property
    def stage_identity(self) -> Optional[tuple]:
        """Canonical identity of the stage specification, if any."""
        return self._stage_identity

    def _cubie_canonical_(self) -> tuple:
        """Return the canonical identity of this request."""
        return (
            "SolverHelperRequest",
            self.role.name,
            self.variant.value,
            self.beta,
            self.gamma,
            self.preconditioner_order,
            self._stage_identity,
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
    """

    device_function: Callable
    cached_auxiliary_count: Optional[int] = None
    prepare_jac: Optional[Callable] = None


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
