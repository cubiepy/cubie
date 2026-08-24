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

import mpmath
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
    stage_count = len(a_rows)
    # mpmath arithmetic keeps the decomposition machine-independent.
    with mpmath.workdps(60):
        a = mpmath.matrix(
            [[mpmath.mpf(v) for v in row] for row in a_rows]
        )
        try:
            inverse_a = a**-1
        except ZeroDivisionError as exc:
            raise ValueError(
                "A is singular; the block transform is unavailable "
                "for this tableau."
            ) from exc
        eigenvalues, eigenvectors = mpmath.eig(inverse_a)
        scale = max(
            mpmath.mpf(1),
            max(abs(value) for value in eigenvalues),
        )
        tolerance = mpmath.mpf("1e-12") * scale

        order = sorted(
            range(len(eigenvalues)),
            key=lambda k: (
                float(mpmath.re(eigenvalues[k])),
                abs(float(mpmath.im(eigenvalues[k]))),
            ),
        )
        used: set = set()
        real_values: List[float] = []
        real_columns: List[list] = []
        pair_values: List[Tuple[float, float]] = []
        pair_columns: List[list] = []
        for idx in order:
            if idx in used:
                continue
            value = eigenvalues[idx]
            vector = [eigenvectors[row, idx] for row in range(stage_count)]
            if abs(mpmath.im(value)) <= tolerance:
                used.add(idx)
                column = [mpmath.re(entry) for entry in vector]
                pivot = max(
                    range(stage_count), key=lambda k: abs(column[k])
                )
                column = [entry / column[pivot] for entry in column]
                real_values.append(float(mpmath.re(value)))
                real_columns.append([float(entry) for entry in column])
                continue
            partner = None
            for other in order:
                if other in used or other == idx:
                    continue
                gap = abs(eigenvalues[other] - mpmath.conj(value))
                if gap <= tolerance:
                    partner = other
                    break
            if partner is None:
                raise ValueError(
                    "Eigenvalues of inv(A) do not pair into conjugates; "
                    "the block transform is unavailable for this "
                    "tableau."
                )
            used.add(idx)
            used.add(partner)
            if mpmath.im(value) < 0:
                value = mpmath.conj(value)
                vector = [mpmath.conj(entry) for entry in vector]
            pivot = max(
                range(stage_count), key=lambda k: abs(vector[k])
            )
            vector = [entry / vector[pivot] for entry in vector]
            pair_values.append(
                (float(mpmath.re(value)), float(mpmath.im(value)))
            )
            pair_columns.append(
                [float(mpmath.re(entry)) for entry in vector]
            )
            pair_columns.append(
                [float(mpmath.im(entry)) for entry in vector]
            )

        columns = real_columns + pair_columns
        transform_mp = mpmath.matrix(
            [
                [mpmath.mpf(columns[col][row]) for col in
                 range(stage_count)]
                for row in range(stage_count)
            ]
        )
        try:
            inverse_transform_mp = transform_mp**-1
        except ZeroDivisionError as exc:
            raise ValueError(
                "Block eigenstructure failed to reassemble inv(A); "
                "the block transform is unavailable for this tableau."
            ) from exc
        transform = tuple(
            tuple(float(transform_mp[row, col]) for col in
                  range(stage_count))
            for row in range(stage_count)
        )
        inverse_transform = tuple(
            tuple(float(inverse_transform_mp[row, col]) for col in
                  range(stage_count))
            for row in range(stage_count)
        )
        inverse_a_float = np.array(
            [
                [float(inverse_a[row, col]) for col in
                 range(stage_count)]
                for row in range(stage_count)
            ]
        )

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
    if not np.allclose(
        inverse_a_float,
        np.array(transform) @ lam @ np.array(inverse_transform),
        rtol=1e-9,
        atol=1e-9 * float(scale),
    ):
        raise ValueError(
            "Block eigenstructure failed to reassemble inv(A); the "
            "block transform is unavailable for this tableau."
        )
    return (
        tuple(real_values),
        tuple(pair_values),
        transform,
        inverse_transform,
    )
