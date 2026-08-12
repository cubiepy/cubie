"""Canonical configuration for symbolic operation ordering."""

from collections.abc import Mapping
from typing import (
    Any,
    Mapping as MappingType,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from cubie.odesystems.solver_helpers import (
    CACHED_AUX_HELPER_KINDS,
    OPERATION_ORDERING_HELPER_KINDS,
    SolverHelperKind,
)


OPERATION_ORDERINGS = ("kahn", "greedy", "dfs", "liveness_auto")
"""Supported assignment-ordering methods."""


OPERATION_ORDERING_FAMILIES = (
    "dxdt",
    "observables",
    *(
        kind.value
        for kind in SolverHelperKind
        if kind in OPERATION_ORDERING_HELPER_KINDS
    ),
)
"""Generated function families that independently order operations."""


CACHED_AUX_FAMILIES = tuple(
    kind.value
    for kind in SolverHelperKind
    if kind in CACHED_AUX_HELPER_KINDS
)
"""Families sharing the ``cached_aux`` slot layout; one method only."""


class OperationOrderingMap(NamedTuple):
    """Immutable canonical sparse family overrides."""

    overrides: Tuple[Tuple[str, str], ...]


OperationOrdering = Union[str, OperationOrderingMap]
OperationOrderingInput = Union[str, MappingType[str, str]]


def _invalid_method(
    method: Any, family: Optional[str] = None
) -> ValueError:
    """Return the exact validation error for an unsupported method."""

    location = ""
    if family is not None:
        location = f" for family {family!r}"
    return ValueError(
        f"Invalid operation_ordering method{location}: {method!r}. "
        f"Expected one of {OPERATION_ORDERINGS}."
    )


def normalize_operation_ordering(value: Any) -> OperationOrdering:
    """Return immutable canonical operation-ordering configuration.

    A method string is retained as the all-family shorthand. A mapping
    becomes an immutable canonical value containing non-Kahn overrides;
    a mapping with none collapses to ``"kahn"``. Every family in
    :data:`CACHED_AUX_FAMILIES` must resolve to one shared method.
    """

    if isinstance(value, OperationOrderingMap):
        value = dict(value.overrides)
    if isinstance(value, str):
        if value not in OPERATION_ORDERINGS:
            raise _invalid_method(value)
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "operation_ordering must be a supported method string or "
            "a mapping from generated-function family to method."
        )

    unknown = tuple(
        key for key in value if key not in OPERATION_ORDERING_FAMILIES
    )
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(
            f"Unknown operation_ordering families: {names}. Expected "
            f"keys from {OPERATION_ORDERING_FAMILIES}."
        )

    for family, method in value.items():
        if method not in OPERATION_ORDERINGS:
            raise _invalid_method(method, family)

    cluster_methods = {
        family: value.get(family, "kahn")
        for family in CACHED_AUX_FAMILIES
    }
    if len(set(cluster_methods.values())) > 1:
        assignments = ", ".join(
            f"{family}={method!r}"
            for family, method in cluster_methods.items()
        )
        raise ValueError(
            "Families sharing cached Jacobian auxiliaries must use "
            f"one ordering method; got {assignments}. Set every "
            f"family in {CACHED_AUX_FAMILIES} to the same method "
            "(omitted families default to 'kahn')."
        )

    overrides = tuple(
        (family, value[family])
        for family in OPERATION_ORDERING_FAMILIES
        if family in value and value[family] != "kahn"
    )
    if not overrides:
        return "kahn"
    return OperationOrderingMap(overrides)


def resolve_operation_ordering(
    configuration: OperationOrdering,
    family: str,
) -> str:
    """Resolve one generated family, defaulting sparse maps to Kahn."""

    if family not in OPERATION_ORDERING_FAMILIES:
        raise ValueError(
            f"Unknown operation_ordering family: {family!r}. Expected "
            f"one of {OPERATION_ORDERING_FAMILIES}."
        )
    if isinstance(configuration, str):
        return configuration
    for configured_family, method in configuration.overrides:
        if configured_family == family:
            return method
    return "kahn"


__all__ = [
    "CACHED_AUX_FAMILIES",
    "OPERATION_ORDERINGS",
    "OPERATION_ORDERING_FAMILIES",
    "OperationOrdering",
    "OperationOrderingInput",
    "OperationOrderingMap",
    "normalize_operation_ordering",
    "resolve_operation_ordering",
]
