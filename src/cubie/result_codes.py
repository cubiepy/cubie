"""Central result-code vocabulary shared across the whole package.

Published Classes
-----------------
:class:`CUBIE_RESULT_CODES`
    Bit-flag status codes OR-combined into the per-run status word that
    propagates from the device solver up to
    :class:`~cubie.batchsolving.Solver`.

Notes
-----
The codes are a bit-field: device functions OR their contributions into a
single ``int32`` status word (``SUCCESS`` is the empty ``0``), and the host
side decodes the word back into named flags via
:class:`CUBIE_RESULT_CODES`. This is the single source of truth — device code
captures the integer values as compile-time closure constants rather than
repeating magic literals.
"""

from enum import IntFlag


class CUBIE_RESULT_CODES(IntFlag):
    """Status/result codes for a single integration run.

    Members
    -------
    SUCCESS
        No error; step accepted / current step kept.
    MAX_NEWTON_ITERATIONS_EXCEEDED
        Newton iteration did not converge within its iteration budget.
    MAX_LINEAR_ITERATIONS_EXCEEDED
        Inner linear (Krylov) solve did not converge within its budget.
    STEP_TOO_SMALL
        Adaptive controller rejected a step whose proposed ``dt`` would fall
        at or below ``dt_min``.
    DT_EFF_EFFECTIVELY_ZERO
        Effective step size collapsed to zero (reserved).
    MAX_LOOP_ITERS_EXCEEDED
        Integration loop hit its maximum iteration count (reserved).
    STAGNATION
        Integration made no progress in ``t`` over consecutive steps.
    BICGSTAB_BREAKDOWN
        BiCGSTAB linear solve broke down (a recurrence scalar collapsed to
        zero) before converging.
    NEWTON_DIVERGENCE
        Newton iteration diverged: an update above tolerance grew more
        than twofold or the update norm was not finite.
    DAE_INITIALISATION_FAILED
        The consistent-initialisation solve at ``t0`` did not converge;
        the run ended at the ``t0`` save with the uncorrected values.
    """

    SUCCESS = 0
    MAX_NEWTON_ITERATIONS_EXCEEDED = 2
    MAX_LINEAR_ITERATIONS_EXCEEDED = 4
    STEP_TOO_SMALL = 8
    DT_EFF_EFFECTIVELY_ZERO = 16
    MAX_LOOP_ITERS_EXCEEDED = 32
    STAGNATION = 64
    BICGSTAB_BREAKDOWN = 128
    NEWTON_DIVERGENCE = 256
    DAE_INITIALISATION_FAILED = 1024


def decode_status_codes(status_codes):
    """Decode per-run status words into named result flags.

    Parameters
    ----------
    status_codes
        Iterable of integer status words (one per run), or ``None``.

    Returns
    -------
    dict[int, list[str]]
        Mapping from run index to the list of :class:`CUBIE_RESULT_CODES`
        member names set in that run's status word. Successful runs
        (status ``0``) are omitted, so an empty mapping means every run
        succeeded.
    """
    messages = {}
    if status_codes is None:
        return messages
    flags = [flag for flag in CUBIE_RESULT_CODES if flag.value != 0]
    for run_index, code in enumerate(status_codes):
        code = int(code)
        if code == 0:
            continue
        messages[run_index] = [
            flag.name for flag in flags if code & flag.value == flag.value
        ]
    return messages


__all__ = ["CUBIE_RESULT_CODES", "decode_status_codes"]
