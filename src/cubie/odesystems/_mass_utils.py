"""Mass-matrix row-flag conversion.

Published Functions
-------------------
:func:`mass_diagonal_flags`
    Normalise a mass matrix (``None`` or a 0/1 diagonal) into a
    per-row tuple of booleans (``True`` for an identity row).
"""

from typing import Tuple

import numpy as np

__all__ = ["mass_diagonal_flags"]


def mass_diagonal_flags(M, n: int) -> Tuple[bool, ...]:
    """Return per-row mass flags for a 0/1 diagonal mass matrix.

    Parameters
    ----------
    M
        Mass matrix as ``None`` (identity) or an ``n`` x ``n``
        0/1 diagonal (NumPy array or nested sequences).
    n
        State dimension.

    Returns
    -------
    tuple of bool
        ``True`` for an identity (differential) row, ``False`` for a
        zero (algebraic residual) row.

    Raises
    ------
    ValueError
        If ``M`` is not ``None``, an ``n`` x ``n`` matrix, or carries
        any entry other than a 0/1 diagonal.
    """
    if M is None:
        return (True,) * n
    matrix = np.asarray(M, dtype=np.float64)
    if matrix.shape != (n, n):
        raise ValueError(
            f"Mass matrix shape {matrix.shape} does not match the "
            f"state dimension {n}."
        )
    diagonal = np.diag(matrix)
    off_diagonal = matrix - np.diag(diagonal)
    if np.any(off_diagonal != 0.0) or not np.all(
        (diagonal == 0.0) | (diagonal == 1.0)
    ):
        raise ValueError(
            "Mass matrices are derived by structural simplification "
            "and are always 0/1 diagonals (identity rows for "
            "differential states, zero rows for torn algebraic "
            "residuals); got a matrix with other entries."
        )
    return tuple(bool(entry) for entry in diagonal)
