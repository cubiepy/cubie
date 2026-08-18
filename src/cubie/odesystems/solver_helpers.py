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
traits. Request validation, source-identity hashing, and the symbolic
registry all derive from it; :data:`STAGE_AWARE_KINDS` is a derived
view.

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
    "SolverHelperRequest",
    "HelperResult",
    "SolverHelperCache",
    "resolve_preconditioner_kind",
]


class SolverHelperKind(Enum):
    """Concrete generated-helper kinds.

    Member values follow the naming rule
    ``[n_stage_]<type>_preconditioner[_cached|_at_state]`` that
    :func:`resolve_preconditioner_kind` relies on. The ``_AT_STATE``
    family evaluates the Jacobian at the ``state`` argument, with
    ``a_ij`` scaling the matrix only.
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
    APPLY_MASS = "apply_mass"
    EVALUATE_INV_MASS_F = "evaluate_inv_mass_f"
    STAGE_RESIDUAL = "stage_residual"
    N_STAGE_RESIDUAL = "n_stage_residual"
    N_STAGE_LINEAR_OPERATOR = "n_stage_linear_operator"
    N_STAGE_NEUMANN_PRECONDITIONER = "n_stage_neumann_preconditioner"
    N_STAGE_JACOBI_PRECONDITIONER = "n_stage_jacobi_preconditioner"
    PREPARE_JAC = "prepare_jac"
    CALCULATE_CACHED_JVP = "calculate_cached_jvp"
    TIME_DERIVATIVE_RHS = "time_derivative_rhs"


@frozen
class HelperKindTraits:
    """Kind-level traits of one generated-helper kind.

    Attributes
    ----------
    stage_aware
        Whether emitted source depends on the stage specification.
    selection_aware
        Whether emitted source depends on the cache selection.
    """

    stage_aware: bool = False
    selection_aware: bool = False


HELPER_KIND_TRAITS = {
    SolverHelperKind.LINEAR_OPERATOR: HelperKindTraits(),
    SolverHelperKind.LINEAR_OPERATOR_CACHED: HelperKindTraits(
        selection_aware=True,
    ),
    SolverHelperKind.LINEAR_OPERATOR_AT_STATE: HelperKindTraits(),
    SolverHelperKind.NEUMANN_PRECONDITIONER: HelperKindTraits(),
    SolverHelperKind.NEUMANN_PRECONDITIONER_CACHED: HelperKindTraits(
        selection_aware=True,
    ),
    SolverHelperKind.NEUMANN_PRECONDITIONER_AT_STATE: HelperKindTraits(),
    SolverHelperKind.JACOBI_PRECONDITIONER: HelperKindTraits(),
    SolverHelperKind.JACOBI_PRECONDITIONER_CACHED: HelperKindTraits(
        selection_aware=True,
    ),
    SolverHelperKind.JACOBI_PRECONDITIONER_AT_STATE: HelperKindTraits(),
    SolverHelperKind.APPLY_MASS: HelperKindTraits(),
    SolverHelperKind.EVALUATE_INV_MASS_F: HelperKindTraits(),
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
    SolverHelperKind.PREPARE_JAC: HelperKindTraits(
        selection_aware=True,
    ),
    SolverHelperKind.CALCULATE_CACHED_JVP: HelperKindTraits(
        selection_aware=True,
    ),
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

    Notes
    -----
    Stage entries participate in identity through their canonical
    SymPy text form, so exact and floating forms of the same tableau
    are distinguished deliberately — they emit different source.
    Stage-aware kinds require stage data at construction and other
    kinds reject it.
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
    _stage_identity: Optional[tuple] = field(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
        traits = HELPER_KIND_TRAITS[self.kind]
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

    def _cubie_canonical_(self) -> tuple:
        """Return the canonical identity of this request."""
        return (
            "SolverHelperRequest",
            self.kind.value,
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
        Number of precomputed auxiliary slots the helper populates or
        consumes. Set for ``prepare_jac``; ``None`` otherwise.
    """

    device_function: Callable
    cached_auxiliary_count: Optional[int] = None


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
