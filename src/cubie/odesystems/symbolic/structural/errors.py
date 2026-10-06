"""Exceptions raised by structural simplification consistency checks.

The exception types follow StateSelection.jl (commit 74df007e,
``src/utils.jl``, ``InvalidSystemException``,
``ExtraVariablesSystemException`` and
``ExtraEquationsSystemException``).
"""

from typing import Sequence


class InvalidSystemError(ValueError):
    """The system is structurally singular or otherwise invalid."""


class ExtraVariablesSystemError(InvalidSystemError):
    """The system has more unknowns than equations.

    The reported variable list is a best-effort heuristic; the true
    extra variables depend on the model.
    """


class ExtraEquationsSystemError(InvalidSystemError):
    """The system has more equations than unknowns.

    The reported equation list is a best-effort heuristic; the true
    extra equations depend on the model.
    """


def raise_unmatched(
    summary: str,
    equations: Sequence[str],
    variables: Sequence[str],
) -> None:
    """Raise for the equations and variables a matching leaves unmatched.

    The message is ``summary`` followed by the unmatched equations and
    variables, one per line. Nothing is raised when both are empty.

    Raises
    ------
    InvalidSystemError
        When both equations and variables are unmatched.
    ExtraEquationsSystemError
        When only equations are unmatched.
    ExtraVariablesSystemError
        When only variables are unmatched.
    """

    message = summary
    if equations:
        message += "\nUnmatched equation(s):\n" + "\n".join(equations)
    if variables:
        message += "\nUnmatched variable(s):\n" + "\n".join(variables)
    if equations and variables:
        raise InvalidSystemError(message)
    if equations:
        raise ExtraEquationsSystemError(message)
    if variables:
        raise ExtraVariablesSystemError(message)
