"""Per-system calibration of algorithm and linear-solver settings.

:func:`run_calibration` (exposed as :meth:`cubie.Solver.calibrate`)
measures a staged panel of candidate configurations against a
representative input grid and reports the fastest viable one, plus
every candidate within an equivalence margin of it. Stages: a
structural prune enumerates only legal candidates; a rising ladder of
short screening solves (a probe at ``screen_fraction**2`` of the
duration, then a screen at ``screen_fraction``) gates each candidate
on failure counts and on a time budget relative to the stage's
fastest, with each candidate's host compile overlapping earlier
candidates' kernels; survivors are ranked on a few full-duration
solves scored by the lowest time. A candidate over budget at any rung
is scheduled no further solves; a launched kernel itself cannot be
aborted in-process. Results persist as a markdown file next to the
system's generated sources and are reloaded on a repeat call under
the same conditions. Requires a real GPU; raises under the CUDA
simulator.

Published Objects
-----------------
:class:`CandidateSpec`
    One candidate configuration: an algorithm alias plus settings.
:class:`CandidateResult`
    Measured outcome for one candidate in one stage.
:class:`CalibrationResult`
    Winner, equivalence set, candidate measurements, and the system
    feature record.
:func:`run_calibration`
    Execute the staged tournament for a configured solver.

See Also
--------
:meth:`cubie.batchsolving.solver.Solver.calibrate`
    User-facing entry point delegating here.
"""

import json
import logging
from hashlib import sha256
from math import ceil
from pathlib import Path
from warnings import warn
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from attrs import define, frozen
from numpy import asarray, count_nonzero, generic, isfinite, ndarray
from numpy.linalg import eigvals

from cubie.cache_root import get_cache_root
from cubie.cuda_backend import CUDA_BACKEND
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda
from cubie.integrators.algorithms import resolve_alias
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_erk import ERKStep
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from cubie.integrators.stage_predictors import (
    tableau_supports_dense_prediction,
)

logger = logging.getLogger(__name__)


CALIBRATION_FAMILIES = ("erk", "dirk", "firk", "rosenbrock")
"""Algorithm families the calibration panel covers, in run order."""

FAMILY_REPRESENTATIVES = {
    "dirk": "kvaerno3",
    "firk": "radau_iia_5",
    "rosenbrock": "ros3p",
}
"""Mid-order tableau used for each implicit family's solver stages."""

# Panels list only tableaus with an embedded error estimate.
FAMILY_ORDER_PANELS = {
    "erk": ("bogacki-shampine-32", "tsit5", "vern7"),
    "dirk": ("kvaerno3", "l_stable_sdirk_4", "kvaerno5"),
    "firk": ("radau_iia_3", "radau_iia_5", "radau_iia_9"),
    "rosenbrock": ("rosenbrock23", "ros3p", "rodas3p"),
}
"""Tableau aliases spanning a few orders of each family."""

PRECONDITIONER_PANEL = (
    ("jacobi", 0),
    ("jacobi", 1),
    ("jacobi", 2),
    ("neumann", 2),
    ("none", 0),
)
"""(type, order) pairs swept in each implicit family's first stage."""

_FAMILY_STEP_CLASSES = {
    "erk": ERKStep,
    "dirk": DIRKStep,
    "firk": FIRKStep,
    "rosenbrock": GenericRosenbrockWStep,
}

_AXIS_FIELDS = (
    "linear_correction_type",
    "preconditioner_type",
    "preconditioner_order",
    "inexact_newton",
    "prefactored",
    "use_smoothed_error",
    "attempt_dense_prediction",
)


def _alias_tableau(alias: str):
    """Return the tableau registered under ``alias``."""
    _, tableau = resolve_alias(alias)
    return tableau


def _alias_family(alias: str) -> str:
    """Return the family key whose step class serves ``alias``."""
    step_class, _ = resolve_alias(alias)
    for family, cls in _FAMILY_STEP_CLASSES.items():
        if step_class is cls:
            return family
    raise ValueError(
        f"Algorithm '{alias}' is outside the calibration families "
        f"{CALIBRATION_FAMILIES}."
    )


def _alias_is_adaptive(alias: str) -> bool:
    """Return whether ``alias`` carries an embedded error estimate."""
    tableau = _alias_tableau(alias)
    return tableau is not None and tableau.has_error_estimate


def _supports_smoothing(alias: str) -> bool:
    """Return whether ``alias``'s tableau defines a smoothed estimate."""
    tableau = _alias_tableau(alias)
    return bool(
        tableau is not None
        and getattr(tableau, "supports_smoothed_error", False)
    )


def _supports_prediction(alias: str, family: str) -> bool:
    """Return whether dense stage prediction applies to ``alias``."""
    if family not in ("dirk", "firk"):
        return False
    tableau = _alias_tableau(alias)
    return tableau is not None and tableau_supports_dense_prediction(
        tableau
    )


@frozen
class CandidateSpec:
    """One candidate configuration in the calibration panel.

    Parameters
    ----------
    label
        Human-readable identifier shown in reports.
    family
        Algorithm family key (``"erk"``, ``"dirk"``, ``"firk"``, or
        ``"rosenbrock"``).
    algorithm
        Tableau alias passed to the solver as ``algorithm``.
    settings
        Canonical (name, value) pairs of solver keyword overrides.
    """

    label: str
    family: str
    algorithm: str
    settings: Tuple[Tuple[str, Any], ...] = ()

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Settings pairs as a keyword-argument dictionary."""
        return dict(self.settings)

    @property
    def key(self) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        """Identity of the configuration, independent of its label."""
        return (self.algorithm, self.settings)


@define
class CandidateResult:
    """Measured outcome for one candidate in one stage.

    Parameters
    ----------
    spec
        The candidate configuration measured.
    stage
        Name of the tournament stage the measurement belongs to.
    times_ms
        Per-solve times in milliseconds for the timed solves.
    screen_ms
        Screening-solve time in milliseconds; ``None`` when the
        candidate carried over from an earlier stage.
    failures
        Failed-run count from the last solve inspected.
    runs
        Trajectory count each solve integrated.
    dropped
        Whether the candidate was removed before timed ranking.
    reason
        Why the candidate was dropped, empty otherwise.
    """

    spec: CandidateSpec
    stage: str
    times_ms: Tuple[float, ...] = ()
    screen_ms: Optional[float] = None
    failures: int = 0
    runs: int = 0
    dropped: bool = False
    reason: str = ""

    @property
    def best_ms(self) -> float:
        """Lowest measured solve time, ``inf`` when never timed."""
        return min(self.times_ms) if self.times_ms else float("inf")

    @property
    def failure_fraction(self) -> float:
        """Failed runs as a fraction of the trajectory count."""
        if self.runs == 0:
            return 0.0
        return self.failures / self.runs


@define
class CalibrationResult:
    """Complete calibration report.

    Parameters
    ----------
    candidates
        Every candidate measurement from every stage, in run order.
    winner
        Fastest viable candidate from the final ranking, or ``None``
        when no candidate integrated the grid acceptably.
    equivalent
        Final-stage candidates within the equivalence margin of the
        winner, winner included.
    features
        System feature record (size, spectral radius, precision, ...)
        for building selection heuristics offline.
    applied_settings
        Settings applied to the calling solver, empty when nothing
        was applied.
    """

    candidates: List[CandidateResult]
    winner: Optional[CandidateResult]
    equivalent: List[CandidateResult]
    features: Dict[str, Any]
    applied_settings: Dict[str, Any]

    def summary(self) -> str:
        """Return a formatted table of the final ranking."""
        lines = []
        header = (
            f"{'stage':<20}{'candidate':<40}{'best ms':>10}"
            f"{'failed':>8}  note"
        )
        lines.append(header)
        lines.append("-" * len(header))
        equivalent_keys = {
            result.spec.key for result in self.equivalent
        }
        for result in self.candidates:
            if result.dropped:
                note = f"dropped: {result.reason}"
                best = ""
            else:
                best = f"{result.best_ms:.3f}"
                note = ""
                if self.winner is not None:
                    if result.spec.key == self.winner.spec.key:
                        note = "winner"
                    elif (
                        result.stage == self.winner.stage
                        and result.spec.key in equivalent_keys
                    ):
                        note = "equivalent"
            lines.append(
                f"{result.stage:<20}{result.spec.label:<40}"
                f"{best:>10}{result.failures:>8}  {note}"
            )
        return "\n".join(lines)

    def to_records(self) -> List[Dict[str, Any]]:
        """Return flat per-candidate dictionaries for tabular export."""
        records = []
        for result in self.candidates:
            record = dict(self.features)
            record.update(
                stage=result.stage,
                label=result.spec.label,
                family=result.spec.family,
                algorithm=result.spec.algorithm,
                best_ms=result.best_ms,
                screen_ms=result.screen_ms,
                times_ms=";".join(
                    f"{value:.4f}" for value in result.times_ms
                ),
                failures=result.failures,
                dropped=result.dropped,
                reason=result.reason,
            )
            for name in _AXIS_FIELDS:
                record[name] = result.spec.settings_dict.get(name)
            records.append(record)
        return records


def preconditioner_stage_specs(
    family: str, representative: str, has_mass: bool = False
) -> List[CandidateSpec]:
    """Return the BiCGSTAB preconditioner sweep for a family.

    Parameters
    ----------
    family
        Implicit family key.
    representative
        Tableau alias the family's solver stages run on.
    has_mass
        Whether the system carries a mass matrix.

    Returns
    -------
    list of CandidateSpec
        One candidate per legal (type, order) pair: ``"firk"`` omits
        Jacobi orders above zero, and mass-matrix systems omit
        Neumann.
    """
    panel = PRECONDITIONER_PANEL
    if family == "firk":
        panel = tuple(
            pair
            for pair in panel
            if not (pair[0] == "jacobi" and pair[1] > 0)
        )
    if has_mass:
        panel = tuple(
            pair for pair in panel if pair[0] != "neumann"
        )
    specs = []
    for p_type, p_order in panel:
        tag = f"{p_type}-{p_order}" if p_type != "none" else "none"
        specs.append(
            CandidateSpec(
                label=f"{representative} bicgstab {tag}",
                family=family,
                algorithm=representative,
                settings=(
                    ("linear_correction_type", "bicgstab"),
                    ("preconditioner_type", p_type),
                    ("preconditioner_order", p_order),
                ),
            )
        )
    return specs


def solver_stage_specs(
    family: str,
    representative: str,
    preconditioner: Tuple[str, int],
) -> List[CandidateSpec]:
    """Return the linear-solver-by-Newton-variant panel for a family.

    Parameters
    ----------
    family
        Implicit family key.
    representative
        Tableau alias the candidates run on.
    preconditioner
        Winning (type, order) pair from the preconditioner stage,
        applied to every iterative candidate.

    Returns
    -------
    list of CandidateSpec
        Rosenbrock-W gets one candidate per correction type. Newton
        families cross the iterative solvers with exact and inexact
        Newton; the direct solver adds the prefactored variant, and
        DIRK additionally the frozen-entry refactoring variant.
    """
    p_type, p_order = preconditioner
    tag = f"{p_type}-{p_order}" if p_type != "none" else "none"
    iterative = (
        ("preconditioner_type", p_type),
        ("preconditioner_order", p_order),
    )
    prefix = representative
    if family == "rosenbrock":
        return [
            CandidateSpec(
                label=f"{prefix} bicgstab {tag}",
                family=family,
                algorithm=representative,
                settings=(
                    ("linear_correction_type", "bicgstab"),
                )
                + iterative,
            ),
            CandidateSpec(
                label=f"{prefix} mr {tag}",
                family=family,
                algorithm=representative,
                settings=(
                    ("linear_correction_type", "minimal_residual"),
                )
                + iterative,
            ),
            CandidateSpec(
                label=f"{prefix} lu",
                family=family,
                algorithm=representative,
                settings=(("linear_correction_type", "lu"),),
            ),
        ]
    specs = []
    for correction, name in (
        ("bicgstab", "bicgstab"),
        ("minimal_residual", "mr"),
    ):
        specs.append(
            CandidateSpec(
                label=f"{prefix} {name} {tag} exact",
                family=family,
                algorithm=representative,
                settings=(
                    ("linear_correction_type", correction),
                )
                + iterative,
            )
        )
        specs.append(
            CandidateSpec(
                label=f"{prefix} {name} {tag} inexact",
                family=family,
                algorithm=representative,
                settings=(
                    ("linear_correction_type", correction),
                    ("inexact_newton", True),
                )
                + iterative,
            )
        )
    specs.append(
        CandidateSpec(
            label=f"{prefix} lu exact",
            family=family,
            algorithm=representative,
            settings=(("linear_correction_type", "lu"),),
        )
    )
    if family == "dirk":
        # Only DIRK separates frozen-entry refactoring from stored
        # step-start factors.
        specs.append(
            CandidateSpec(
                label=f"{prefix} lu inexact",
                family=family,
                algorithm=representative,
                settings=(
                    ("linear_correction_type", "lu"),
                    ("inexact_newton", True),
                    ("prefactored", False),
                ),
            )
        )
    specs.append(
        CandidateSpec(
            label=f"{prefix} lu prefactored",
            family=family,
            algorithm=representative,
            settings=(
                ("linear_correction_type", "lu"),
                ("inexact_newton", True),
                ("prefactored", True),
            ),
        )
    )
    return specs


def toggle_stage_specs(
    family: str,
    representative: str,
    base_settings: Tuple[Tuple[str, Any], ...],
) -> List[CandidateSpec]:
    """Return the smoothed-error and predictor toggle cross.

    Parameters
    ----------
    family
        Implicit family key.
    representative
        Tableau alias the candidates run on.
    base_settings
        Winning settings from the solver stage, carried into every
        toggle candidate.

    Returns
    -------
    list of CandidateSpec
        The cross of every toggle the representative tableau supports;
        empty when it supports none.
    """
    axes = []
    if _supports_smoothing(representative):
        axes.append(("use_smoothed_error", (True, False)))
    if _supports_prediction(representative, family):
        axes.append(("attempt_dense_prediction", (True, False)))
    if not axes:
        return []
    combos = [()]
    for name, values in axes:
        combos = [
            existing + ((name, value),)
            for existing in combos
            for value in values
        ]
    specs = []
    for combo in combos:
        flags = " ".join(
            "{}={}".format(
                "smooth"
                if name == "use_smoothed_error"
                else "predict",
                "on" if value else "off",
            )
            for name, value in combo
        )
        specs.append(
            CandidateSpec(
                label=f"{representative} {flags}",
                family=family,
                algorithm=representative,
                settings=base_settings + combo,
            )
        )
    return specs


def order_stage_specs(
    family: str,
    winner_settings: Tuple[Tuple[str, Any], ...],
    incumbent: str,
) -> List[CandidateSpec]:
    """Return the family order panel under the winning settings.

    Parameters
    ----------
    family
        Family key selecting the order panel.
    winner_settings
        Settings of the family's winning configuration so far.
    incumbent
        Tableau alias of the winning configuration; listed first so
        every stage re-times its incumbent in the same window.

    Returns
    -------
    list of CandidateSpec
        One candidate per adaptive alias in the panel. Settings that a
        panel tableau does not support (smoothing on a tableau without
        a smoothed estimate, prediction on an ineligible tableau) are
        removed from that candidate.
    """
    aliases = [incumbent] + [
        alias
        for alias in FAMILY_ORDER_PANELS[family]
        if alias != incumbent
    ]
    specs = []
    for alias in aliases:
        if not _alias_is_adaptive(alias):
            continue
        settings = []
        for name, value in winner_settings:
            if name == "use_smoothed_error" and not _supports_smoothing(
                alias
            ):
                continue
            if (
                name == "attempt_dense_prediction"
                and not _supports_prediction(alias, family)
            ):
                continue
            settings.append((name, value))
        specs.append(
            CandidateSpec(
                label=alias,
                family=family,
                algorithm=alias,
                settings=tuple(settings),
            )
        )
    return specs


def erk_stage_specs() -> List[CandidateSpec]:
    """Return the explicit family's order panel."""
    specs = []
    for alias in FAMILY_ORDER_PANELS["erk"]:
        if not _alias_is_adaptive(alias):
            continue
        specs.append(
            CandidateSpec(
                label=alias, family="erk", algorithm=alias
            )
        )
    return specs


def complete_apply_settings(spec: CandidateSpec) -> Dict[str, Any]:
    """Return the full settings update the winner implies.

    Every axis the calibration owns is materialised explicitly for
    implicit winners, so applying the winner to a solver overwrites
    any previous non-default value on those axes rather than leaving
    it behind.

    Parameters
    ----------
    spec
        The winning candidate.

    Returns
    -------
    dict
        Keyword updates including ``algorithm``.
    """
    updates = {"algorithm": spec.algorithm}
    if spec.family == "erk":
        return updates
    settings = spec.settings_dict
    updates["linear_correction_type"] = settings.get(
        "linear_correction_type", "minimal_residual"
    )
    updates["inexact_newton"] = settings.get("inexact_newton", False)
    updates["prefactored"] = settings.get("prefactored", True)
    if updates["linear_correction_type"] != "lu":
        p_type = settings.get("preconditioner_type", "jacobi")
        default_order = 2 if p_type == "neumann" else 0
        updates["preconditioner_type"] = p_type
        updates["preconditioner_order"] = settings.get(
            "preconditioner_order", default_order
        )
    if _supports_smoothing(spec.algorithm):
        default_on = spec.family == "firk"
        updates["use_smoothed_error"] = settings.get(
            "use_smoothed_error", default_on
        )
    if _supports_prediction(spec.algorithm, spec.family):
        updates["attempt_dense_prediction"] = settings.get(
            "attempt_dense_prediction", True
        )
    return updates


def _device_to_host(array: Any) -> ndarray:
    """Copy a device array to the host, whatever its container."""
    if hasattr(array, "copy_to_host"):
        return array.copy_to_host()
    if hasattr(array, "get"):
        return array.get()
    return asarray(array)


def _candidate_base_kwargs(
    parent: Any,
    duration: float,
) -> Dict[str, Any]:
    """Return candidate solver kwargs copied from the parent; unset
    summary cadences are pinned to ``duration``."""
    kwargs = {}
    for name in (
        "atol",
        "rtol",
        "dt_min",
        "dt_max",
        "save_every",
        "summarise_every",
        "sample_summaries_every",
    ):
        value = getattr(parent, name)
        if value is not None:
            kwargs[name] = value
    kwargs["output_types"] = list(parent.output_types)
    for name in (
        "saved_state_indices",
        "saved_observable_indices",
        "summarised_state_indices",
        "summarised_observable_indices",
    ):
        value = getattr(parent, name)
        if value is not None:
            kwargs[name] = asarray(value)
    integrator = parent.kernel.single_integrator
    if integrator.has_summary_outputs:
        if "summarise_every" not in kwargs:
            kwargs["summarise_every"] = float(duration)
        if "sample_summaries_every" not in kwargs:
            kwargs["sample_summaries_every"] = (
                float(duration) / 100.0
            )
    policy = parent.kernel.cache_policy
    if not policy.cache_enabled:
        kwargs["cache"] = False
    elif policy.cache_dir is not None:
        kwargs["cache"] = policy.cache_dir
    # Candidates join the parent's stream group: one shared stream.
    kwargs["memory_settings"] = {
        "memory_manager": parent.kernel.memory_manager,
        "stream_group": parent.stream_group,
    }
    return kwargs


def _system_features(
    parent: Any, t0: float, n_runs: int, duration: float
) -> Dict[str, Any]:
    """Return the feature record for heuristic building.

    Parameters
    ----------
    parent
        The solver being calibrated.
    t0
        Initial integration time for the Jacobian evaluation.
    n_runs
        Trajectory count of the calibration grid.
    duration
        Full-length solve duration.

    Returns
    -------
    dict
        System size counts, precision, mass-matrix flag, tolerances,
        and the spectral radius of the state Jacobian at the initial
        state (``None`` when it cannot be evaluated).
    """
    system = parent.system
    sizes = system.sizes
    features = {
        "system": getattr(system, "name", type(system).__name__),
        "n_states": int(sizes.states),
        "n_observables": int(sizes.observables),
        "n_parameters": int(sizes.parameters),
        "n_drivers": int(sizes.drivers),
        "n_runs": int(n_runs),
        "precision": parent.precision.__name__,
        "has_mass_matrix": system.mass is not None,
        "atol": _scalar_or_none(parent.atol),
        "rtol": _scalar_or_none(parent.rtol),
        "duration": float(duration),
        "spectral_radius": None,
    }
    try:
        evaluator = system._get_neumann_evaluator(
            parent.kernel.cache_policy
        )
        jacobian = evaluator.jacobian(system.indices, t0=t0)
        if isfinite(jacobian).all():
            features["spectral_radius"] = float(
                abs(eigvals(jacobian)).max()
            )
    except Exception:
        logger.debug(
            "Spectral-radius feature unavailable", exc_info=True
        )
    return features


def _scalar_or_none(value: Any) -> Optional[float]:
    """Return the tightest tolerance entry, ``None`` when unset."""
    if value is None:
        return None
    array = asarray(value)
    if array.size == 0:
        return None
    return float(array.min())


class _CalibrationRunner:
    """Build, screen, and time candidate solvers for one calibration.

    Candidate launches share the parent's group stream. Screen solves
    enqueue asynchronously (``on_device=True``) while later candidates
    compile on the host; timed solves run serially with CUDA events
    bracketing each launch.
    """

    def __init__(
        self,
        parent: Any,
        inits: ndarray,
        params: ndarray,
        drivers: Optional[Dict[str, Any]],
        base_kwargs: Dict[str, Any],
        duration: float,
        settling_time: float,
        t0: float,
        screen_rungs: Sequence[Tuple[float, float]],
        n_repeats: int,
        failure_tolerance: float,
        screen_budget_factor: float,
        blocksize: int,
        verbose: bool,
    ) -> None:
        self._parent = parent
        self._solver_class = type(parent)
        self._system = parent.system
        self._drivers = drivers
        self._base_kwargs = base_kwargs
        self._duration = float(duration)
        self._settling = float(settling_time)
        self._t0 = float(t0)
        self._screen_rungs = tuple(screen_rungs)
        self._n_repeats = int(n_repeats)
        self._failure_tolerance = float(failure_tolerance)
        self._screen_budget_factor = float(screen_budget_factor)
        self._blocksize = int(blocksize)
        self._verbose = bool(verbose)
        self._n_runs = int(inits.shape[1])
        self._inits = cuda.to_device(inits)
        self._params = cuda.to_device(params)
        # spec.key -> live candidate solver, reused across stages.
        self._live: Dict[Any, Any] = {}
        # Keys of family winners kept alive for the final ranking.
        self._protected: set = set()
        self.achieved_waves = None

    @property
    def n_runs(self) -> int:
        """Trajectory count of the calibration grid."""
        return self._n_runs

    def _emit(self, message: str) -> None:
        """Print progress when verbose; always log at debug level."""
        logger.debug(message)
        if self._verbose:
            print(message, flush=True)

    def _build_solver(self, spec: CandidateSpec) -> Any:
        """Construct the candidate solver for ``spec``."""
        kwargs = dict(self._base_kwargs)
        kwargs.update(spec.settings_dict)
        return self._solver_class(
            self._system, algorithm=spec.algorithm, **kwargs
        )

    def _launch(
        self,
        solver: Any,
        duration: float,
        settling_time: float,
        first: bool,
    ) -> Tuple[Any, Any]:
        """Enqueue one solve, no sync; return its CUDA event pair."""
        kwargs = {}
        if first and self._drivers is not None:
            kwargs["drivers"] = self._drivers
        stream = solver.stream
        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record(stream)
        solver.solve(
            self._inits,
            self._params,
            duration=duration,
            settling_time=settling_time,
            t0=self._t0,
            blocksize=self._blocksize,
            on_device=True,
            **kwargs,
        )
        end_event.record(stream)
        return (start_event, end_event)

    def _elapsed_ms(self, token: Tuple[Any, Any]) -> float:
        """Return the elapsed milliseconds of a synced event pair."""
        return float(cuda.event_elapsed_time(*token))

    def _read_failures(self, solver: Any) -> int:
        """Return the failed-run count of the candidate's last solve."""
        codes = _device_to_host(solver.kernel.device_status_codes)
        return int(count_nonzero(asarray(codes).ravel()))

    def _timed_solve(self, solver: Any) -> Tuple[float, int]:
        """Run one full-duration solve; return (ms, failed runs)."""
        token = self._launch(
            solver, self._duration, self._settling, first=False
        )
        solver.kernel.synchronize()
        return self._elapsed_ms(token), self._read_failures(solver)

    def _probe_waves(self, solver: Any) -> None:
        """Record achieved occupancy waves; warn once when under two."""
        if self.achieved_waves is not None:
            return
        try:
            waves = _achieved_waves(solver, self._blocksize)
        except Exception:
            logger.debug("Occupancy probe failed", exc_info=True)
            return
        self.achieved_waves = waves
        if waves < 2.0:
            warn(
                f"Calibration grid fills {waves:.2f} occupancy "
                "waves at the first candidate's kernel; below two "
                "waves timings may not rank configurations reliably."
            )

    def run_stage(
        self, specs: Sequence[CandidateSpec], stage: str
    ) -> List[CandidateResult]:
        """Screen and time one stage of the tournament.

        Parameters
        ----------
        specs
            Candidate configurations to measure, incumbent first.
        stage
            Stage name recorded on every result.

        Returns
        -------
        list of CandidateResult
            One result per unique candidate: dropped entries carry a
            reason; survivors carry their timed solves.
        """
        results = []
        cached = []
        fresh = []
        seen = set()
        first_duration, first_settling = self._screen_rungs[0]
        for spec in specs:
            if spec.key in seen:
                continue
            seen.add(spec.key)
            result = CandidateResult(
                spec=spec, stage=stage, runs=self._n_runs
            )
            results.append(result)
            if spec.key in self._live:
                cached.append((result, self._live[spec.key]))
                continue
            try:
                solver = self._build_solver(spec)
                token = self._launch(
                    solver, first_duration, first_settling, first=True
                )
            except Exception as exc:
                result.dropped = True
                result.reason = f"{type(exc).__name__}: {exc}"
                self._emit(
                    f"  {spec.label}: build/screen failed "
                    f"({result.reason})"
                )
                continue
            self._live[spec.key] = solver
            fresh.append([result, solver, token])

        # Gate fresh candidates at each rung of the screen ladder.
        for index, (
            rung_duration,
            rung_settling,
        ) in enumerate(self._screen_rungs):
            if index > 0:
                for entry in fresh:
                    entry[2] = self._launch(
                        entry[1],
                        rung_duration,
                        rung_settling,
                        first=False,
                    )
            for result, solver, token in fresh:
                solver.kernel.synchronize()
                result.screen_ms = self._elapsed_ms(token)
                result.failures = self._read_failures(solver)
            if fresh:
                self._probe_waves(fresh[0][1])
            fractions = [
                result.failure_fraction
                for result, _, _ in fresh
            ] + [result.failure_fraction for result, _ in cached]
            floor = min(fractions) if fractions else 0.0
            rung_times = [
                result.screen_ms for result, _, _ in fresh
            ]
            budget = (
                min(rung_times) * self._screen_budget_factor
                if rung_times
                else None
            )
            remaining = []
            for result, solver, token in fresh:
                if (
                    result.failure_fraction
                    > floor + self._failure_tolerance
                ):
                    result.dropped = True
                    result.reason = (
                        f"screen failures {result.failures}/"
                        f"{result.runs}"
                    )
                elif (
                    budget is not None
                    and result.screen_ms > budget
                ):
                    result.dropped = True
                    result.reason = (
                        f"screen {result.screen_ms:.1f} ms over "
                        f"budget {budget:.1f} ms"
                    )
                else:
                    remaining.append([result, solver, token])
                    continue
                self._emit(f"  {result.spec.label}: {result.reason}")
                self.close_candidate(result.spec)
            fresh = remaining

        survivors = cached + [
            (result, solver) for result, solver, _ in fresh
        ]

        # Round-robin the timed solves across candidates.
        times = {id(result): [] for result, _ in survivors}
        for _ in range(self._n_repeats):
            for result, solver in survivors:
                elapsed_ms, failures = self._timed_solve(solver)
                times[id(result)].append(elapsed_ms)
                result.failures = failures
        for result, _ in survivors:
            result.times_ms = tuple(times[id(result)])
            self._emit(
                f"  {result.spec.label}: {result.best_ms:.3f} ms "
                f"({result.failures} failed)"
            )

        # Re-gate on the full-length failure counts.
        timed = [result for result, _ in survivors]
        fractions = [result.failure_fraction for result in timed]
        floor = min(fractions) if fractions else 0.0
        for result in timed:
            if (
                result.failure_fraction
                > floor + self._failure_tolerance
            ):
                result.dropped = True
                result.reason = (
                    f"failures {result.failures}/{result.runs}"
                )
                self.close_candidate(result.spec)
        return results

    def stage_winner(
        self, results: Sequence[CandidateResult]
    ) -> Optional[CandidateResult]:
        """Return the fastest non-dropped candidate, if any."""
        viable = [
            result
            for result in results
            if not result.dropped and result.times_ms
        ]
        if not viable:
            return None
        return min(viable, key=lambda result: result.best_ms)

    def protect(self, spec: CandidateSpec) -> None:
        """Keep the candidate alive through later pruning passes."""
        self._protected.add(spec.key)

    def close_candidate(self, spec: CandidateSpec) -> None:
        """Release the candidate's solver if it is live."""
        solver = self._live.pop(spec.key, None)
        if solver is not None:
            solver.close()

    def prune(self, keep: Sequence[CandidateSpec]) -> None:
        """Release live candidates outside ``keep`` and the protected
        set."""
        keep_keys = {spec.key for spec in keep} | self._protected
        for key in list(self._live):
            if key not in keep_keys:
                self._live.pop(key).close()

    def close_all(self) -> None:
        """Release every live candidate solver, protected included."""
        for solver in self._live.values():
            solver.close()
        self._live.clear()
        self._protected.clear()


def _achieved_waves(solver: Any, blocksize: int) -> float:
    """Return occupancy waves the grid fills at the actual geometry."""
    kernel_factory = solver.kernel
    (kern,) = kernel_factory.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    runs = int(kernel_factory.run_params[0].runs)
    pad = 4 if kernel_factory.shared_memory_needs_padding else 0
    padded_bytes = kernel_factory.shared_memory_bytes + pad
    dynshared = padded_bytes * min(runs, blocksize)
    actual_blocksize, dynshared = kernel_factory.limit_blocksize(
        blocksize, dynshared, padded_bytes, runs
    )
    dynshared = max(4, dynshared)
    context = cuda.current_context()
    blocks_per_sm = context.get_active_blocks_per_multiprocessor(
        cufunc, actual_blocksize, dynshared
    )
    device = cuda.get_current_device()
    threads_per_loop = (
        kernel_factory.single_integrator.threads_per_step
    )
    runs_per_block = actual_blocksize // threads_per_loop
    total_blocks = ceil(runs / runs_per_block)
    resident = blocks_per_sm * device.MULTIPROCESSOR_COUNT
    return total_blocks / resident


def _screen_rungs(
    base_kwargs: Dict[str, Any],
    duration: float,
    settling_time: float,
    screen_fraction: float,
) -> Tuple[Tuple[float, float], ...]:
    """Return the (duration, settling) screening ladder, ascending.

    A probe rung at ``screen_fraction**2`` of the duration precedes a
    screen rung at ``screen_fraction``; each is raised to every
    configured output interval and capped at the full duration, and
    coinciding rungs collapse into one.
    """
    fraction = float(screen_fraction)
    rungs = []
    for rung_fraction in (fraction * fraction, fraction):
        rung_duration = float(duration) * rung_fraction
        for name in ("save_every", "summarise_every"):
            value = base_kwargs.get(name)
            if value is not None:
                rung_duration = max(rung_duration, float(value))
        rung_duration = min(rung_duration, float(duration))
        if duration > 0.0:
            scale = rung_duration / float(duration)
        else:
            scale = 1.0
        rung = (rung_duration, float(settling_time) * scale)
        if not rungs or rungs[-1] != rung:
            rungs.append(rung)
    return tuple(rungs)


def _json_safe(value: Any) -> Any:
    """Return ``value`` converted to JSON-serialisable primitives."""
    if isinstance(value, generic):
        return value.item()
    if isinstance(value, ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in
                value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _calibration_conditions(
    parent: Any,
    base_kwargs: Dict[str, Any],
    duration: float,
    settling_time: float,
    t0: float,
    n_runs: int,
    families: Sequence[str],
    blocksize: int,
) -> Dict[str, Any]:
    """Return the condition record keying a calibration result."""
    settings = {
        name: value
        for name, value in base_kwargs.items()
        if name not in ("memory_settings", "cache")
    }
    device = cuda.get_current_device()
    device_name = device.name
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    return _json_safe(
        {
            "fn_hash": parent.system.fn_hash,
            "backend": CUDA_BACKEND,
            "device": device_name,
            "precision": parent.precision.__name__,
            "duration": float(duration),
            "settling_time": float(settling_time),
            "t0": float(t0),
            "n_runs": int(n_runs),
            "families": list(families),
            "blocksize": int(blocksize),
            "settings": settings,
        }
    )


def _calibration_cache_path(
    parent: Any, conditions: Dict[str, Any]
) -> Path:
    """Return the markdown cache path for a condition record."""
    key = sha256(
        json.dumps(conditions, sort_keys=True).encode()
    ).hexdigest()[:12]
    system_dir = get_cache_root() / parent.kernel._system_name
    return system_dir / f"calibration_{key}.md"


def _result_payload(report: "CalibrationResult") -> Dict[str, Any]:
    """Return the JSON payload for a calibration report."""
    candidates = []
    winner_index = None
    equivalent_indices = []
    for index, result in enumerate(report.candidates):
        spec = result.spec
        candidates.append(
            {
                "label": spec.label,
                "family": spec.family,
                "algorithm": spec.algorithm,
                "settings": [
                    [name, _json_safe(value)]
                    for name, value in spec.settings
                ],
                "stage": result.stage,
                "times_ms": list(result.times_ms),
                "screen_ms": result.screen_ms,
                "failures": result.failures,
                "runs": result.runs,
                "dropped": result.dropped,
                "reason": result.reason,
            }
        )
        if (
            report.winner is not None
            and result is report.winner
        ):
            winner_index = index
        if any(result is entry for entry in report.equivalent):
            equivalent_indices.append(index)
    return {
        "candidates": candidates,
        "winner_index": winner_index,
        "equivalent_indices": equivalent_indices,
        "features": _json_safe(report.features),
    }


def _write_calibration_file(
    path: Path,
    report: "CalibrationResult",
    conditions: Dict[str, Any],
) -> None:
    """Write the calibration report as a markdown file."""
    winner_line = "no viable candidate"
    if report.winner is not None:
        winner_line = (
            f"{report.winner.spec.label} "
            f"({report.winner.best_ms:.3f} ms)"
        )
    content = (
        f"# Calibration: {conditions['fn_hash'][:12]}\n\n"
        f"Winner: {winner_line}\n\n"
        "```text\n"
        f"{report.summary()}\n"
        "```\n\n"
        "## Conditions\n\n"
        "```json\n"
        f"{json.dumps(conditions, sort_keys=True, indent=2)}\n"
        "```\n\n"
        "## Result\n\n"
        "```json\n"
        f"{json.dumps(_result_payload(report), indent=2)}\n"
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def _load_calibration_file(
    path: Path,
) -> Optional["CalibrationResult"]:
    """Load a calibration report from a markdown file."""
    try:
        text = path.read_text(encoding="utf8")
        marker = "## Result"
        block = text[text.index(marker):]
        block = block[block.index("```json") + len("```json"):]
        payload = json.loads(block[: block.index("```")])
        candidates = []
        for entry in payload["candidates"]:
            spec = CandidateSpec(
                label=entry["label"],
                family=entry["family"],
                algorithm=entry["algorithm"],
                settings=tuple(
                    (name, value)
                    for name, value in entry["settings"]
                ),
            )
            candidates.append(
                CandidateResult(
                    spec=spec,
                    stage=entry["stage"],
                    times_ms=tuple(entry["times_ms"]),
                    screen_ms=entry["screen_ms"],
                    failures=entry["failures"],
                    runs=entry["runs"],
                    dropped=entry["dropped"],
                    reason=entry["reason"],
                )
            )
        winner_index = payload["winner_index"]
        winner = (
            candidates[winner_index]
            if winner_index is not None
            else None
        )
        equivalent = [
            candidates[index]
            for index in payload["equivalent_indices"]
        ]
        return CalibrationResult(
            candidates=candidates,
            winner=winner,
            equivalent=equivalent,
            features=payload["features"],
            applied_settings={},
        )
    except Exception:
        logger.debug(
            "Calibration cache unreadable: %s", path, exc_info=True
        )
        return None


def run_calibration(
    parent: Any,
    initial_values: Any,
    parameters: Any,
    drivers: Optional[Dict[str, Any]] = None,
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    grid_type: str = "verbatim",
    families: Optional[Sequence[str]] = None,
    equivalence_margin: float = 0.10,
    failure_tolerance: float = 0.01,
    screen_fraction: float = 0.0625,
    screen_budget_factor: float = 5.0,
    n_repeats: int = 3,
    apply: bool = True,
    verbose: bool = True,
    blocksize: int = 256,
) -> CalibrationResult:
    """Run the staged calibration tournament for a solver.

    Parameters
    ----------
    parent
        The configured :class:`~cubie.batchsolving.solver.Solver`
        whose system, tolerances, and output configuration every
        candidate replicates.
    initial_values
        Initial-state input for the representative grid, as accepted
        by :meth:`Solver.build_grid`.
    parameters
        Parameter input for the representative grid.
    drivers
        Driver samples plus interpolation settings; required when the
        system declares drivers.
    duration
        Full-length solve duration candidates are ranked on.
    settling_time
        Warm-up period preceding output collection.
    t0
        Initial integration time.
    grid_type
        Grid construction strategy for dict inputs.
    families
        Families to calibrate; defaults to every legal family
        (explicit families are excluded on mass-matrix systems).
    equivalence_margin
        Relative margin under which final candidates are reported as
        equivalent to the winner.
    failure_tolerance
        Allowed failed-run fraction above the stage minimum before a
        candidate is dropped.
    screen_fraction
        Fraction of ``duration`` the screening solve integrates; a
        probe at ``screen_fraction**2`` runs first. Both rungs are
        raised to any configured output interval.
    screen_budget_factor
        Multiple of the rung's fastest screening time above which a
        candidate is dropped without further solves.
    n_repeats
        Timed full-duration solves per surviving candidate; the
        lowest time is the candidate's score.
    apply
        Apply the winner's configuration to ``parent`` when ``True``.
    verbose
        Print per-candidate progress lines.
    blocksize
        CUDA block size for candidate launches.

    Returns
    -------
    CalibrationResult
        Winner, equivalence set, all candidate measurements, and the
        system feature record. Written as a markdown file beside the
        system's generated sources when caching is enabled, and
        reloaded from there on a repeat call under identical
        conditions.

    Raises
    ------
    RuntimeError
        Under the CUDA simulator.
    ValueError
        If the system declares drivers but none are supplied, or if
        an explicit family is requested on a mass-matrix system.
    """
    if CUDA_SIMULATION:
        raise RuntimeError(
            "Solver.calibrate measures kernel times on a real GPU "
            "and does not run under the CUDA simulator."
        )
    system = parent.system
    has_mass = system.mass is not None
    if families is None:
        families = tuple(
            family
            for family in CALIBRATION_FAMILIES
            if not (has_mass and family == "erk")
        )
    else:
        families = tuple(families)
        unknown = set(families) - set(CALIBRATION_FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown calibration families {sorted(unknown)}; "
                f"choose from {CALIBRATION_FAMILIES}."
            )
        if has_mass and "erk" in families:
            raise ValueError(
                "Explicit algorithms cannot integrate a mass-matrix "
                "system; remove 'erk' from families."
            )
    if system.sizes.drivers > 0 and drivers is None:
        raise ValueError(
            "The system declares drivers; calibrate requires the "
            "driver samples that solves will use."
        )

    inits, params = parent.build_grid(
        initial_values, parameters, grid_type=grid_type
    )
    base_kwargs = _candidate_base_kwargs(parent, duration)
    conditions = _calibration_conditions(
        parent,
        base_kwargs,
        duration,
        settling_time,
        t0,
        inits.shape[1],
        families,
        blocksize,
    )
    cache_enabled = parent.kernel.cache_policy.cache_enabled
    cache_path = _calibration_cache_path(parent, conditions)
    if cache_enabled and cache_path.exists():
        report = _load_calibration_file(cache_path)
        if report is not None:
            applied_settings = {}
            if report.winner is not None and apply:
                applied_settings = complete_apply_settings(
                    report.winner.spec
                )
                parent.update(dict(applied_settings))
            report.applied_settings = applied_settings
            if verbose:
                print(
                    f"calibration loaded from {cache_path}",
                    flush=True,
                )
            return report

    rungs = _screen_rungs(
        base_kwargs, duration, settling_time, screen_fraction
    )
    features = _system_features(
        parent, t0, inits.shape[1], duration
    )

    runner = _CalibrationRunner(
        parent=parent,
        inits=inits,
        params=params,
        drivers=drivers,
        base_kwargs=base_kwargs,
        duration=duration,
        settling_time=settling_time,
        t0=t0,
        screen_rungs=rungs,
        n_repeats=n_repeats,
        failure_tolerance=failure_tolerance,
        screen_budget_factor=screen_budget_factor,
        blocksize=blocksize,
        verbose=verbose,
    )

    all_results = []
    family_winners = []
    try:
        for family in families:
            winner = _run_family(
                runner, family, all_results, has_mass
            )
            if winner is not None:
                family_winners.append(winner)
                runner.protect(winner.spec)
            runner.prune(())

        winner = None
        equivalent = []
        if len(family_winners) > 1:
            runner._emit("final: family winners head-to-head")
            final_results = runner.run_stage(
                [result.spec for result in family_winners], "final"
            )
            all_results.extend(final_results)
            winner = runner.stage_winner(final_results)
            pool = final_results
        elif family_winners:
            winner = family_winners[0]
            pool = [winner]
        else:
            pool = []
        if winner is not None:
            ceiling = winner.best_ms * (1.0 + equivalence_margin)
            equivalent = [
                result
                for result in pool
                if not result.dropped
                and result.times_ms
                and result.best_ms <= ceiling
            ]

        applied_settings = {}
        if winner is not None and apply:
            applied_settings = complete_apply_settings(winner.spec)
            parent.update(dict(applied_settings))
            runner._emit(
                f"applied: {winner.spec.label} -> parent solver"
            )
    finally:
        runner.close_all()

    features["achieved_waves"] = runner.achieved_waves
    report = CalibrationResult(
        candidates=all_results,
        winner=winner,
        equivalent=equivalent,
        features=features,
        applied_settings=applied_settings,
    )
    if cache_enabled:
        _write_calibration_file(cache_path, report, conditions)
        runner._emit(f"calibration written to {cache_path}")
    return report


def _run_family(
    runner: _CalibrationRunner,
    family: str,
    all_results: List[CandidateResult],
    has_mass: bool,
) -> Optional[CandidateResult]:
    """Run one family's stages; return its winning result, if any."""
    if family == "erk":
        runner._emit("erk: order panel")
        results = runner.run_stage(erk_stage_specs(), "erk:orders")
        all_results.extend(results)
        return runner.stage_winner(results)

    representative = FAMILY_REPRESENTATIVES[family]
    runner._emit(f"{family}: preconditioner sweep")
    precond_results = runner.run_stage(
        preconditioner_stage_specs(
            family, representative, has_mass
        ),
        f"{family}:precond",
    )
    all_results.extend(precond_results)
    precond_winner = runner.stage_winner(precond_results)
    if precond_winner is not None:
        runner.prune([precond_winner.spec])
    if precond_winner is None:
        # No viable iterative candidate; run stage 2 with jacobi-0.
        preconditioner = ("jacobi", 0)
    else:
        settings = precond_winner.spec.settings_dict
        preconditioner = (
            settings["preconditioner_type"],
            settings["preconditioner_order"],
        )

    runner._emit(f"{family}: linear solver x Newton variant")
    solver_results = runner.run_stage(
        solver_stage_specs(family, representative, preconditioner),
        f"{family}:solver",
    )
    all_results.extend(solver_results)
    solver_winner = runner.stage_winner(solver_results)
    if solver_winner is None:
        runner._emit(f"{family}: no viable solver configuration")
        return None
    runner.prune([solver_winner.spec])

    toggle_specs = toggle_stage_specs(
        family, representative, solver_winner.spec.settings
    )
    if toggle_specs:
        runner._emit(f"{family}: smoothed-error / predictor toggles")
        toggle_results = runner.run_stage(
            toggle_specs, f"{family}:toggles"
        )
        all_results.extend(toggle_results)
        toggle_winner = runner.stage_winner(toggle_results)
        if toggle_winner is not None:
            solver_winner = toggle_winner
        runner.prune([solver_winner.spec])

    runner._emit(f"{family}: order panel")
    order_results = runner.run_stage(
        order_stage_specs(
            family, solver_winner.spec.settings, representative
        ),
        f"{family}:orders",
    )
    all_results.extend(order_results)
    order_winner = runner.stage_winner(order_results)
    return (
        order_winner if order_winner is not None else solver_winner
    )
