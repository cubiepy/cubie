"""Mass-matrix conversion shared by the codegen builders.

Published Functions
-------------------
:func:`mass_matrix_ir`
    Normalise a mass matrix (``None``, SymPy matrix, NumPy array, or
    nested sequences) into row-major IR entries, defaulting to the
    identity.
:func:`mass_matrix_inverse_ir`
    Return the inverse mass matrix as row-major IR entries.
"""

from typing import List

import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import from_sympy

__all__ = ["mass_matrix_ir", "mass_matrix_inverse_ir"]


def _entry_to_ir(entry) -> ir.Expr:
    """Convert one matrix entry to an IR expression.

    Integer entries become floats so emitted mass terms carry an
    explicit float literal.
    """
    if isinstance(entry, ir.Expr):
        if isinstance(entry, ir.Num) and isinstance(
            entry.value, int
        ):
            return ir.num(float(entry.value))
        return entry
    if isinstance(entry, int):
        return ir.num(float(entry))
    if isinstance(entry, float):
        return ir.num(entry)
    # NumPy scalars expose item(); SymPy scalars convert directly.
    item = getattr(entry, "item", None)
    if item is not None:
        return ir.num(float(item()))
    converted = from_sympy(entry)
    if isinstance(converted, ir.Num) and isinstance(
        converted.value, int
    ):
        return ir.num(float(converted.value))
    return converted


def mass_matrix_ir(M, n: int) -> List[List[ir.Expr]]:
    """Return the mass matrix as row-major IR entries.

    Parameters
    ----------
    M
        Mass matrix as ``None`` (identity), a SymPy matrix, a NumPy
        array, or nested sequences.
    n
        State dimension used for the identity default.

    Returns
    -------
    list of list
        Row-major IR entries.
    """
    if M is None:
        return [
            [ir.ONE if i == j else ir.ZERO for j in range(n)]
            for i in range(n)
        ]
    tolist = getattr(M, "tolist", None)
    rows = tolist() if tolist is not None else [list(row) for row in M]
    return [[_entry_to_ir(entry) for entry in row] for row in rows]


def mass_matrix_inverse_ir(M, n: int) -> List[List[ir.Expr]]:
    """Return ``M**-1`` as row-major IR entries.

    Raises
    ------
    ValueError
        If the mass matrix is singular.
    """
    if M is None:
        return [
            [ir.ONE if i == j else ir.ZERO for j in range(n)]
            for i in range(n)
        ]
    tolist = getattr(M, "tolist", None)
    rows = tolist() if tolist is not None else [list(row) for row in M]
    exact_rows = []
    for row in rows:
        exact_row = []
        for entry in row:
            value = sp.sympify(entry)
            if value.is_Float:
                value = sp.Rational(value)
            exact_row.append(value)
        exact_rows.append(exact_row)
    try:
        inverse = sp.Matrix(exact_rows).inv()
    except ValueError as error:
        raise ValueError(
            "The mass matrix is singular and has no inverse."
        ) from error
    converted = []
    for i in range(n):
        converted_row = []
        for j in range(n):
            entry = inverse[i, j]
            if entry.is_number:
                converted_row.append(_entry_to_ir(float(entry)))
            else:
                converted_row.append(_entry_to_ir(entry))
        converted.append(converted_row)
    return converted
