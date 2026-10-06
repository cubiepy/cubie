"""Exceptions raised by structural simplification consistency checks.

The exception types follow StateSelection.jl (commit 74df007e,
``src/utils.jl``, ``InvalidSystemException``,
``ExtraVariablesSystemException`` and
``ExtraEquationsSystemException``).
"""


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
