"""Mass-matrix row-flag conversion shared by the codegen builders.

Structural simplification is the only source of mass matrices: a
system's mass is either ``None`` (identity) or a 0/1 diagonal with
identity rows for differential states and zero rows for torn
algebraic residuals. The builders consume that structure as per-row
flags; no matrix values enter generated source.

Published Functions
-------------------
:func:`mass_diagonal_flags`
    Re-export of
    :func:`cubie.odesystems._mass_utils.mass_diagonal_flags`.
:func:`mass_matrix_is_identity`
    Return whether the mass matrix is ``None`` or a literal identity.
:func:`block_eigenstructure`
    Real block eigenstructure of ``inv(A)`` for the FIRK transform.
"""

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from cubie.odesystems._mass_utils import mass_diagonal_flags

__all__ = [
    "mass_diagonal_flags",
    "mass_matrix_is_identity",
    "block_eigenstructure",
]


def mass_matrix_is_identity(M) -> bool:
    """Return whether the mass matrix is ``None`` or a literal identity."""
    if M is None:
        return True
    matrix = np.asarray(M, dtype=np.float64)
    return matrix.ndim == 2 and bool(
        np.array_equal(matrix, np.eye(matrix.shape[0]))
    )


@lru_cache(maxsize=None)
def block_eigenstructure(
    a_rows: Tuple[Tuple[float, ...], ...],
) -> Tuple[
    Tuple[float, ...],
    Tuple[Tuple[float, float], ...],
    Tuple[Tuple[float, ...], ...],
    Tuple[Tuple[float, ...], ...],
]:
    """Return the real block eigenstructure of ``inv(A)``.

    Decomposes ``inv(A) = T @ L @ inv(T)`` with ``L`` real block
    diagonal: one 1x1 block per real eigenvalue, one 2x2 block
    ``[[alpha, beta], [-beta, alpha]]`` per conjugate pair.
    Eigenvalues sort by (real part, |imag|); eigenvectors normalise
    on their largest-magnitude entry.

    Parameters
    ----------
    a_rows
        Butcher tableau ``A`` matrix as row-major float tuples.

    Returns
    -------
    tuple
        ``(real_eigenvalues, complex_pairs, transform,
        inverse_transform)``; pairs are ``(alpha, beta)`` with
        ``beta > 0``, matrices row-major tuples.

    Raises
    ------
    ValueError
        If ``A`` is singular or ``inv(A)`` does not reassemble.
    """
    a = np.asarray(a_rows, dtype=np.float64)
    inverse_a = np.linalg.inv(a)
    eigenvalues, eigenvectors = np.linalg.eig(inverse_a)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-12 * scale

    order = sorted(
        range(len(eigenvalues)),
        key=lambda k: (
            float(eigenvalues[k].real),
            abs(float(eigenvalues[k].imag)),
        ),
    )
    used: set = set()
    real_values: List[float] = []
    real_columns: List[np.ndarray] = []
    pair_values: List[Tuple[float, float]] = []
    pair_columns: List[np.ndarray] = []
    for idx in order:
        if idx in used:
            continue
        value = eigenvalues[idx]
        vector = eigenvectors[:, idx]
        if abs(value.imag) <= tolerance:
            used.add(idx)
            column = vector.real.copy()
            pivot = int(np.argmax(np.abs(column)))
            column = column / column[pivot]
            real_values.append(float(value.real))
            real_columns.append(column)
            continue
        partner = None
        for other in order:
            if other in used or other == idx:
                continue
            if abs(eigenvalues[other] - np.conj(value)) <= tolerance:
                partner = other
                break
        if partner is None:
            raise ValueError(
                "Eigenvalues of inv(A) do not pair into conjugates; "
                "the block transform is unavailable for this tableau."
            )
        used.add(idx)
        used.add(partner)
        if value.imag < 0.0:
            value = np.conj(value)
            vector = np.conj(vector)
        pivot = int(np.argmax(np.abs(vector)))
        vector = vector / vector[pivot]
        pair_values.append((float(value.real), float(value.imag)))
        pair_columns.append(vector.real.copy())
        pair_columns.append(vector.imag.copy())

    transform = np.column_stack(real_columns + pair_columns)
    stage_count = a.shape[0]
    lam = np.zeros((stage_count, stage_count), dtype=np.float64)
    for slot, value in enumerate(real_values):
        lam[slot, slot] = value
    offset = len(real_values)
    for slot, (alpha, beta) in enumerate(pair_values):
        row = offset + 2 * slot
        lam[row, row] = alpha
        lam[row, row + 1] = beta
        lam[row + 1, row] = -beta
        lam[row + 1, row + 1] = alpha
    inverse_transform = np.linalg.inv(transform)
    if not np.allclose(
        inverse_a,
        transform @ lam @ inverse_transform,
        rtol=1e-9,
        atol=1e-9 * scale,
    ):
        raise ValueError(
            "Block eigenstructure failed to reassemble inv(A); the "
            "block transform is unavailable for this tableau."
        )
    return (
        tuple(real_values),
        tuple(pair_values),
        tuple(tuple(float(v) for v in row) for row in transform),
        tuple(
            tuple(float(v) for v in row) for row in inverse_transform
        ),
    )
