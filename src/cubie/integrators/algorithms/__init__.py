"""Factories for explicit and implicit algorithm step implementations."""

from typing import Any, Mapping, Optional, Tuple, Type, Union

from attrs import frozen

from .base_algorithm_step import (
    AlgorithmDefaults,
    BaseAlgorithmStep,
    ButcherTableau,
)
from .ode_explicitstep import ExplicitStepConfig
from .ode_implicitstep import ImplicitStepConfig, ODEImplicitStep
from .backwards_euler import BackwardsEulerStep
from .backwards_euler_predict_correct import BackwardsEulerPCStep
from .crank_nicolson import CrankNicolsonStep
from .explicit_euler import ExplicitEulerStep
from .generic_dirk import (
    DIRKStep,
)
from .generic_dirk_tableaus import DIRK_TABLEAU_REGISTRY, DIRKTableau
from .generic_firk import (
    FIRKStep,
)
from .generic_firk_tableaus import FIRK_TABLEAU_REGISTRY, FIRKTableau
from .generic_erk import (
    ERKStep,
    ERKTableau,
)
from .generic_erk_tableaus import ERK_TABLEAU_REGISTRY
from .generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from .generic_rosenbrockw_tableaus import (
    ROSENBROCK_TABLEAUS,
    RosenbrockTableau,
)


__all__ = [
    "AlgorithmFacts",
    "algorithm_facts",
    "resolve_algorithm",
    "algorithm_is_adaptive",
    "get_algorithm_step",
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "ExplicitEulerStep",
    "BackwardsEulerStep",
    "BackwardsEulerPCStep",
    "CrankNicolsonStep",
    "DIRKStep",
    "FIRKStep",
    "ERKStep",
    "GenericRosenbrockWStep",
    "_ALGORITHM_REGISTRY",
    "DIRKTableau",
    "DIRK_TABLEAU_REGISTRY",
    "FIRKTableau",
    "FIRK_TABLEAU_REGISTRY",
    "ERKTableau",
    "ERK_TABLEAU_REGISTRY",
    "RosenbrockTableau",
    "ROSENBROCK_TABLEAUS",
]

_ALGORITHM_REGISTRY = {
    "euler": ExplicitEulerStep,
    "backwards_euler": BackwardsEulerStep,
    "backwards_euler_pc": BackwardsEulerPCStep,
    "crank_nicolson": CrankNicolsonStep,
    "dirk": DIRKStep,
    "firk": FIRKStep,
    "erk": ERKStep,
    "rosenbrock": GenericRosenbrockWStep,
}

_TABLEAU_REGISTRY_BY_ALGORITHM = {
    key: (constructor, None)
    for key, constructor in _ALGORITHM_REGISTRY.items()
}

for alias, tableau in ERK_TABLEAU_REGISTRY.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (ERKStep, tableau)

for alias, tableau in DIRK_TABLEAU_REGISTRY.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (DIRKStep, tableau)

for alias, tableau in FIRK_TABLEAU_REGISTRY.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (FIRKStep, tableau)

for alias, tableau in ROSENBROCK_TABLEAUS.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (
        GenericRosenbrockWStep,
        tableau,
    )


def resolve_alias(
    alias: str,
) -> Tuple[Type[BaseAlgorithmStep], Optional[ButcherTableau]]:
    """Return the step constructor and tableau associated with ``alias``."""

    key = alias.lower()
    if key not in _TABLEAU_REGISTRY_BY_ALGORITHM:
        raise KeyError(alias)
    return _TABLEAU_REGISTRY_BY_ALGORITHM[key]


def algorithm_is_adaptive(alias: str) -> bool:
    """Return whether ``alias`` carries an embedded error estimate.

    Parameters
    ----------
    alias
        Algorithm alias registered in the tableau registry.

    Returns
    -------
    bool
        ``True`` when the algorithm produces an error estimate that an
        adaptive controller can act on.

    Raises
    ------
    KeyError
        If ``alias`` is not a registered algorithm alias.
    """

    algorithm_type, tableau = resolve_alias(alias)
    if tableau is None:
        tableau = algorithm_type.default_tableau
    if tableau is not None:
        return tableau.has_error_estimate
    return algorithm_type.has_error_estimate


@frozen
class AlgorithmFacts:
    """What an algorithm choice fixes before a step is built.

    Attributes
    ----------
    step_class
        The step class the choice selects.
    tableau
        The tableau in effect, ``None`` for fixed schemes.
    defaults
        Family defaults overlaid with the tableau's.
    has_error_estimate, is_implicit, is_linear
        Flags of the step class and tableau.
    """

    step_class: Type[BaseAlgorithmStep]
    tableau: Optional[ButcherTableau]
    defaults: AlgorithmDefaults
    has_error_estimate: bool
    is_implicit: bool
    is_linear: bool


def resolve_algorithm(
    algorithm: Union[str, ButcherTableau],
    tableau: Optional[ButcherTableau] = None,
) -> Tuple[Type[BaseAlgorithmStep], Optional[ButcherTableau]]:
    """Return the step class and tableau for ``algorithm``.

    A named tableau's alias keeps its tableau; a bare family alias
    takes ``tableau``, or the family's ``default_tableau`` without
    one.

    Raises
    ------
    ValueError
        Unknown ``algorithm``, or ``tableau`` of another family.
    TypeError
        ``algorithm`` is neither a name nor a tableau.
    """
    if isinstance(algorithm, ButcherTableau):
        return resolve_supplied_tableau(algorithm)
    if not isinstance(algorithm, str):
        raise TypeError(
            "Expected algorithm name or ButcherTableau instance, "
            f"received {type(algorithm).__name__}."
        )
    try:
        step_class, registry_tableau = resolve_alias(algorithm)
    except KeyError as exc:
        raise ValueError(f"Unknown algorithm '{algorithm}'.") from exc
    if registry_tableau is not None:
        return step_class, registry_tableau
    if tableau is None:
        return step_class, step_class.default_tableau
    tableau_class, _ = resolve_supplied_tableau(tableau)
    if tableau_class is not step_class:
        raise ValueError(
            f"Tableau of type {type(tableau).__name__} does not belong "
            f"to algorithm '{algorithm}'."
        )
    return step_class, tableau


def algorithm_facts(
    algorithm: Union[str, ButcherTableau],
    tableau: Optional[ButcherTableau] = None,
) -> AlgorithmFacts:
    """Return the facts of ``algorithm`` per :func:`resolve_algorithm`.

    Raises
    ------
    ValueError
        Unknown ``algorithm``, or ``tableau`` of another family.
    """
    step_class, tableau = resolve_algorithm(algorithm, tableau)
    defaults = step_class.family_defaults(tableau)
    if tableau is not None:
        defaults.settings.update(tableau.defaults)
        has_error_estimate = tableau.has_error_estimate
    else:
        has_error_estimate = step_class.has_error_estimate
    return AlgorithmFacts(
        step_class=step_class,
        tableau=tableau,
        defaults=defaults,
        has_error_estimate=has_error_estimate,
        is_implicit=issubclass(step_class, ODEImplicitStep),
        is_linear=step_class.is_linear,
    )


def resolve_supplied_tableau(
    tableau: ButcherTableau,
) -> Tuple[Type[BaseAlgorithmStep], ButcherTableau]:
    """Return the step constructor matching ``tableau``."""

    if isinstance(tableau, ERKTableau):
        return ERKStep, tableau
    if isinstance(tableau, DIRKTableau):
        return DIRKStep, tableau
    if isinstance(tableau, FIRKTableau):
        return FIRKStep, tableau
    if isinstance(tableau, RosenbrockTableau):
        return GenericRosenbrockWStep, tableau
    raise TypeError(
        "Received tableau of type "
        f"{type(tableau).__name__} which does not match known algorithms."
    )


def get_algorithm_step(
    precision: type,
    settings: Optional[Mapping[str, Any]] = None,
    warn_on_unused: bool = False,
    **kwargs: Any,
) -> BaseAlgorithmStep:
    """Thin factory which filters arguments and instantiates an algorithm.

    Parameters
    ----------
    precision
        Floating-point dtype used when compiling the step implementation.
    settings
        Mapping of settings applied to the algorithm. Must include
        ``"algorithm"`` and can contain any keywords from
        ``ALL_ALGORITHM_STEP_PARAMETERS``.
    warn_on_unused
        If ``True``, issue a warning for settings that the selected algorithm
        does not accept.
    **kwargs
        Additional keywords from ``ALL_ALGORITHM_STEP_PARAMETERS``. These
        override entries provided in ``settings``.

    Returns
    -------
    BaseAlgorithmStep
        The requested step instance.

    Raises
    ------
    ValueError
        Raised when settings['algorithm'] does not match a known algorithm
        type or when required configuration keys are missing.
    """

    algorithm_settings = {}
    if settings is not None:
        algorithm_settings.update(settings)
    algorithm_settings.update(kwargs)

    algorithm_value = algorithm_settings.pop("algorithm", None)
    if algorithm_value is None:
        raise ValueError("Algorithm settings must include 'algorithm'.")

    algorithm_type, resolved_tableau = resolve_algorithm(
        algorithm_value, algorithm_settings.get("tableau")
    )

    algorithm_settings["precision"] = precision

    if resolved_tableau is not None:
        algorithm_settings["tableau"] = resolved_tableau

    # Pass all settings to algorithm __init__ which uses build_config
    # internally build_config filters to valid config fields and handles
    # defaults
    return algorithm_type(**algorithm_settings)
