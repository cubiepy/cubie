"""Solver-helper request and cache containers.

Implicit algorithms consume generated device helpers (operators,
residuals, preconditioners, Jacobian caches) from an ODE system. Each
lookup is described by an immutable :class:`SolverHelperRequest`
derived from the requesting algorithm's compile settings — helper
request state never mutates the ODE system's configuration, so an ODE
may serve multiple algorithms or beta/gamma/tableau bindings without
its identity depending on request order.

The generation side (source emitters and their binding contracts)
lives in :mod:`cubie.odesystems.symbolic.helper_registry`; this module
holds the request/product containers and the request *identity* they
carry, so the abstract ODE base can reference them without importing
the symbolic pipeline. Because canonical stage identity is part of a
request from construction, the SymPy-based canonical text form of
stage entries is owned here (SymPy is a core dependency; the boundary
this module keeps is the symbolic codegen pipeline, not SymPy).

:data:`HELPER_KIND_TRAITS` is the single authority for kind-level
traits (stage awareness, chained-composition membership). Request
validation, source-identity hashing, and the symbolic registry all
derive from it; :data:`STAGE_AWARE_KINDS` and :data:`CHAINED_KINDS`
are derived views.

Published Classes
-----------------
:class:`SolverHelperKind`
    Enumeration of concrete generated-helper kinds.
:class:`HelperKindTraits`
    Kind-level trait record; one entry per kind in
    :data:`HELPER_KIND_TRAITS`.
:class:`SolverHelperRequest`
    Frozen description of one helper lookup.
:class:`HelperResult`
    A bound helper member: device callable plus typed metadata.
:class:`SolverHelperCache`
    Memoized generated factories and bound members for one live ODE
    build.
"""

from enum import Enum
from typing import Any, Callable, Optional, Tuple

import sympy as sp
from attrs import Factory, define, field, frozen

__all__ = [
    "SolverHelperKind",
    "HelperKindTraits",
    "HELPER_KIND_TRAITS",
    "STAGE_AWARE_KINDS",
    "CHAINED_KINDS",
    "SolverHelperRequest",
    "HelperResult",
    "SolverHelperCache",
    "resolve_preconditioner_kind",
    "resolve_chained_kind",
]


class SolverHelperKind(Enum):
    """Concrete generated-helper kinds.

    The chained kinds compose concrete preconditioners in one
    generated source; the composed stages travel on the request's
    ``chained_kinds`` field, so a composed preconditioner is one
    ordinary generated helper with a source identity of its own.
    Member values follow the naming rule
    ``[n_stage_]<type>_preconditioner[_cached|_at_state]`` that
    :func:`resolve_preconditioner_kind` and
    :func:`resolve_chained_kind` rely on. The ``_AT_STATE`` family
    evaluates the Jacobian at the ``state`` argument, with ``a_ij``
    scaling the matrix only.
    """

    LINEAR_OPERATOR = "linear_operator"
    LINEAR_OPERATOR_CACHED = "linear_operator_cached"
    LINEAR_OPERATOR_AT_STATE = "linear_operator_at_state"
    NEUMANN_PRECONDITIONER = "neumann_preconditioner"
    NEUMANN_PRECONDITIONER_CACHED = "neumann_preconditioner_cached"
    NEUMANN_PRECONDITIONER_AT_STATE = "neumann_preconditioner_at_state"
    JACOBI_PRECONDITIONER = "jacobi_preconditioner"
    JACOBI_PRECONDITIONER_CACHED = "jacobi_preconditioner_cached"
    JACOBI_PRECONDITIONER_AT_STATE = "jacobi_preconditioner_at_state"
    CHAINED_PRECONDITIONER = "chained_preconditioner"
    CHAINED_PRECONDITIONER_CACHED = "chained_preconditioner_cached"
    CHAINED_PRECONDITIONER_AT_STATE = "chained_preconditioner_at_state"
    MASS_APPLY = "mass_apply"
    MASS_SOLVE = "mass_solve"
    STAGE_RESIDUAL = "stage_residual"
    N_STAGE_RESIDUAL = "n_stage_residual"
    N_STAGE_LINEAR_OPERATOR = "n_stage_linear_operator"
    N_STAGE_NEUMANN_PRECONDITIONER = "n_stage_neumann_preconditioner"
    N_STAGE_JACOBI_PRECONDITIONER = "n_stage_jacobi_preconditioner"
    N_STAGE_CHAINED_PRECONDITIONER = "n_stage_chained_preconditioner"
    PREPARE_JAC = "prepare_jac"
    CALCULATE_CACHED_JVP = "calculate_cached_jvp"
    TIME_DERIVATIVE_RHS = "time_derivative_rhs"


@frozen
class HelperKindTraits:
    """Kind-level traits of one generated-helper kind.

    Attributes
    ----------
    stage_aware
        Whether the kind's emitted source depends on the stage
        specification.
    chained_members
        Concrete stage kinds a chained kind may compose, or ``None``
        for non-chained kinds.
    """

    stage_aware: bool = False
    chained_members: Optional[frozenset] = None

    @property
    def chained(self) -> bool:
        """Whether this kind composes concrete preconditioners."""
        return self.chained_members is not None


HELPER_KIND_TRAITS = {
    SolverHelperKind.LINEAR_OPERATOR: HelperKindTraits(),
    SolverHelperKind.LINEAR_OPERATOR_CACHED: HelperKindTraits(),
    SolverHelperKind.LINEAR_OPERATOR_AT_STATE: HelperKindTraits(),
    SolverHelperKind.NEUMANN_PRECONDITIONER: HelperKindTraits(),
    SolverHelperKind.NEUMANN_PRECONDITIONER_CACHED: HelperKindTraits(),
    SolverHelperKind.NEUMANN_PRECONDITIONER_AT_STATE: HelperKindTraits(),
    SolverHelperKind.JACOBI_PRECONDITIONER: HelperKindTraits(),
    SolverHelperKind.JACOBI_PRECONDITIONER_CACHED: HelperKindTraits(),
    SolverHelperKind.JACOBI_PRECONDITIONER_AT_STATE: HelperKindTraits(),
    SolverHelperKind.MASS_APPLY: HelperKindTraits(),
    SolverHelperKind.MASS_SOLVE: HelperKindTraits(),
    SolverHelperKind.CHAINED_PRECONDITIONER: HelperKindTraits(
        chained_members=frozenset(
            (
                SolverHelperKind.NEUMANN_PRECONDITIONER,
                SolverHelperKind.JACOBI_PRECONDITIONER,
            )
        ),
    ),
    SolverHelperKind.CHAINED_PRECONDITIONER_CACHED: HelperKindTraits(
        chained_members=frozenset(
            (
                SolverHelperKind.NEUMANN_PRECONDITIONER_CACHED,
                SolverHelperKind.JACOBI_PRECONDITIONER_CACHED,
            )
        ),
    ),
    SolverHelperKind.CHAINED_PRECONDITIONER_AT_STATE: HelperKindTraits(
        chained_members=frozenset(
            (
                SolverHelperKind.NEUMANN_PRECONDITIONER_AT_STATE,
                SolverHelperKind.JACOBI_PRECONDITIONER_AT_STATE,
            )
        ),
    ),
    SolverHelperKind.STAGE_RESIDUAL: HelperKindTraits(),
    SolverHelperKind.N_STAGE_RESIDUAL: HelperKindTraits(
        stage_aware=True,
    ),
    SolverHelperKind.N_STAGE_LINEAR_OPERATOR: HelperKindTraits(
        stage_aware=True,
    ),
    SolverHelperKind.N_STAGE_NEUMANN_PRECONDITIONER: HelperKindTraits(
        stage_aware=True,
    ),
    SolverHelperKind.N_STAGE_JACOBI_PRECONDITIONER: HelperKindTraits(
        stage_aware=True,
    ),
    SolverHelperKind.N_STAGE_CHAINED_PRECONDITIONER: HelperKindTraits(
        stage_aware=True,
        chained_members=frozenset(
            (
                SolverHelperKind.N_STAGE_NEUMANN_PRECONDITIONER,
                SolverHelperKind.N_STAGE_JACOBI_PRECONDITIONER,
            )
        ),
    ),
    SolverHelperKind.PREPARE_JAC: HelperKindTraits(),
    SolverHelperKind.CALCULATE_CACHED_JVP: HelperKindTraits(),
    SolverHelperKind.TIME_DERIVATIVE_RHS: HelperKindTraits(),
}
"""Single authority for kind-level traits, one entry per kind."""

_untraited = [
    kind for kind in SolverHelperKind if kind not in HELPER_KIND_TRAITS
]
if _untraited:
    raise RuntimeError(
        f"SolverHelperKind members missing traits: {_untraited}"
    )


STAGE_AWARE_KINDS = frozenset(
    kind
    for kind, traits in HELPER_KIND_TRAITS.items()
    if traits.stage_aware
)
"""Kinds whose emitted source depends on the stage specification."""


CHAINED_KINDS = frozenset(
    kind
    for kind, traits in HELPER_KIND_TRAITS.items()
    if traits.chained
)
"""Kinds whose emitted source composes concrete preconditioners."""


def resolve_preconditioner_kind(
    type_name: str,
    cached: bool = False,
    n_stage: bool = False,
    at_state: bool = False,
) -> SolverHelperKind:
    """Return the concrete kind for one preconditioner type name.

    Parameters
    ----------
    type_name
        User-facing preconditioner type (``"neumann"``, ``"jacobi"``).
    cached
        Select the cached-auxiliaries variant (Rosenbrock-W).
    n_stage
        Select the flattened all-stages variant (FIRK).
    at_state
        Select the variant evaluating J at the ``state`` argument.

    Raises
    ------
    ValueError
        If no concrete kind exists for the combination.
    """
    prefix = "n_stage_" if n_stage else ""
    suffix = "_cached" if cached else "_at_state" if at_state else ""
    try:
        return SolverHelperKind(
            f"{prefix}{type_name}_preconditioner{suffix}"
        )
    except ValueError:
        raise ValueError(
            f"Unknown preconditioner type '{type_name}' "
            f"(cached={cached}, n_stage={n_stage}, "
            f"at_state={at_state})."
        ) from None


def resolve_chained_kind(
    cached: bool = False,
    n_stage: bool = False,
    at_state: bool = False,
) -> SolverHelperKind:
    """Return the chained kind for a preconditioner variant family.

    Parameters
    ----------
    cached
        Select the cached-auxiliaries variant (Rosenbrock-W).
    n_stage
        Select the flattened all-stages variant (FIRK).
    at_state
        Select the variant evaluating J at the ``state`` argument.

    Raises
    ------
    ValueError
        If no chained kind exists for the combination.
    """
    prefix = "n_stage_" if n_stage else ""
    suffix = "_cached" if cached else "_at_state" if at_state else ""
    try:
        return SolverHelperKind(
            f"{prefix}chained_preconditioner{suffix}"
        )
    except ValueError:
        raise ValueError(
            "No chained preconditioner exists for "
            f"cached={cached}, n_stage={n_stage}, "
            f"at_state={at_state}."
        ) from None


def _kind_converter(value: Any) -> SolverHelperKind:
    """Accept a kind enum member or its string value."""
    if isinstance(value, SolverHelperKind):
        return value
    return SolverHelperKind(value)


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


def _chained_kinds_converter(value: Any) -> Optional[tuple]:
    """Normalise composed stage kinds to a tuple of enum members."""
    if value is None:
        return None
    return tuple(_kind_converter(member) for member in value)


@frozen
class SolverHelperRequest:
    """Immutable description of one solver-helper lookup.

    Parameters
    ----------
    kind
        Concrete helper kind, as an enum member or its string value.
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
        Stage coupling matrix for stage-aware helpers, row-major.
        Entries may be floats or exact SymPy numbers.
    stage_nodes
        Stage nodes expressed as timestep fractions for stage-aware
        helpers.
    chained_kinds
        Ordered concrete stage kinds composed by a chained kind, in
        application order.

    Notes
    -----
    Stage entries participate in identity through their canonical
    SymPy text form, so exact and floating forms of the same tableau
    are distinguished deliberately — they emit different source.
    Unsupported combinations fail at construction: stage-aware kinds
    require stage data and other kinds reject it; chained kinds
    require at least two concrete stage kinds from their own variant
    family and other kinds reject them.
    """

    kind: SolverHelperKind = field(converter=_kind_converter)
    beta: float = field(default=1.0, converter=float)
    gamma: float = field(default=1.0, converter=float)
    preconditioner_order: int = field(default=2, converter=int)
    stage_coefficients: Optional[Tuple[tuple, ...]] = field(
        default=None, converter=_stage_matrix_converter, eq=False
    )
    stage_nodes: Optional[tuple] = field(
        default=None, converter=_stage_vector_converter, eq=False
    )
    chained_kinds: Optional[Tuple[SolverHelperKind, ...]] = field(
        default=None, converter=_chained_kinds_converter
    )
    _stage_identity: Optional[tuple] = field(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
        traits = HELPER_KIND_TRAITS[self.kind]
        if traits.chained:
            allowed = traits.chained_members
            if (
                self.chained_kinds is None
                or len(self.chained_kinds) < 2
                or any(
                    member not in allowed
                    for member in self.chained_kinds
                )
            ):
                raise ValueError(
                    f"Helper kind '{self.kind.value}' requires "
                    "chained_kinds naming at least two concrete "
                    "preconditioner kinds from its variant family."
                )
        elif self.chained_kinds is not None:
            raise ValueError(
                f"Helper kind '{self.kind.value}' does not compose "
                "chained preconditioner stages."
            )
        if traits.stage_aware:
            if self.stage_coefficients is None or self.stage_nodes is None:
                raise ValueError(
                    f"Helper kind '{self.kind.value}' requires stage "
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
                f"Helper kind '{self.kind.value}' does not consume "
                "stage coefficients or stage nodes."
            )

    @property
    def stage_identity(self) -> Optional[tuple]:
        """Canonical identity of the stage specification, if any."""
        return self._stage_identity

    @property
    def chain_identity(self) -> Optional[tuple]:
        """Canonical identity of the composed stage kinds, if any."""
        if self.chained_kinds is None:
            return None
        return tuple(member.value for member in self.chained_kinds)

    def _cubie_canonical_(self) -> tuple:
        """Return the canonical identity of this request."""
        return (
            "SolverHelperRequest",
            self.kind.value,
            self.beta,
            self.gamma,
            self.preconditioner_order,
            self._stage_identity,
            self.chain_identity,
        )


@define
class HelperResult:
    """One bound helper member.

    Attributes
    ----------
    device_function
        The compiled device callable.
    cached_auxiliary_count
        Number of precomputed auxiliary slots the helper populates or
        consumes. Set for ``prepare_jac``; ``None`` otherwise.
    mass_is_identity
        Whether the serving system's mass matrix is the identity
        (``mass is None``).
    """

    device_function: Callable
    cached_auxiliary_count: Optional[int] = None
    mass_is_identity: Optional[bool] = None


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
