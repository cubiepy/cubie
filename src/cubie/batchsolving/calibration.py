"""Per-system calibration of algorithm and linear-solver settings.

:func:`run_calibration` (exposed as :meth:`cubie.Solver.calibrate`)
measures a staged panel of candidate configurations against a
representative input grid and reports the fastest viable one, plus
every candidate within an equivalence margin of it. Stages: a
structural prune enumerates only legal candidates; one short screening
solve per candidate gates on failure counts, with each candidate's
host compile overlapping earlier candidates' kernels; survivors are
ranked on a few full-duration solves scored by the lowest time. Every
candidate's ``dt_min`` is floored at ``duration / max_steps``, so a
candidate pinned at its minimum step finishes with ``STEP_TOO_SMALL``
failures and is gated out.

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

import logging
from time import perf_counter
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from attrs import define, frozen
from numpy import asarray, count_nonzero, isfinite, ndarray
from numpy.linalg import eigvals

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
            f"{'stage':<16}{'candidate':<44}{'best ms':>10}"
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
                f"{result.stage:<16}{result.spec.label:<44}"
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
    family: str, representative: str
) -> List[CandidateSpec]:
    """Return the BiCGSTAB preconditioner sweep for a family.

    Parameters
    ----------
    family
        Implicit family key.
    representative
        Tableau alias the family's solver stages run on.

    Returns
    -------
    list of CandidateSpec
        One candidate per legal (type, order) pair; ``"firk"`` omits
        Jacobi orders above zero.
    """
    panel = PRECONDITIONER_PANEL
    if family == "firk":
        panel = tuple(
            pair
            for pair in panel
            if not (pair[0] == "jacobi" and pair[1] > 0)
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
        updates["preconditioner_type"] = settings.get(
            "preconditioner_type", "jacobi"
        )
        updates["preconditioner_order"] = settings.get(
            "preconditioner_order", 0
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
    settling_time: float,
    max_steps: int,
) -> Dict[str, Any]:
    """Assemble solver kwargs replicating the parent's configuration.

    Parameters
    ----------
    parent
        The solver being calibrated.
    duration
        Full-length solve duration.
    settling_time
        Warm-up period preceding output collection.
    max_steps
        Step budget flooring every candidate's ``dt_min`` at
        ``(duration + settling_time) / max_steps``.

    Returns
    -------
    dict
        Keyword arguments for candidate solver construction. Unset
        summary cadences are pinned to the full duration.
    """
    kwargs = {}
    for name in (
        "atol",
        "rtol",
        "dt_max",
        "save_every",
        "summarise_every",
        "sample_summaries_every",
    ):
        value = getattr(parent, name)
        if value is not None:
            kwargs[name] = value
    dt_floor = float(duration + settling_time) / float(max_steps)
    parent_dt_min = parent.dt_min
    if parent_dt_min is not None:
        kwargs["dt_min"] = max(float(parent_dt_min), dt_floor)
    else:
        kwargs["dt_min"] = dt_floor
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
    """Return a float for scalar tolerances, ``None`` otherwise."""
    if value is None:
        return None
    array = asarray(value)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return None


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
        screen_duration: float,
        screen_settling: float,
        n_repeats: int,
        failure_tolerance: float,
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
        self._screen_duration = float(screen_duration)
        self._screen_settling = float(screen_settling)
        self._n_repeats = int(n_repeats)
        self._failure_tolerance = float(failure_tolerance)
        self._blocksize = int(blocksize)
        self._verbose = bool(verbose)
        self._n_runs = int(inits.shape[1])
        if CUDA_SIMULATION:
            self._inits = inits
            self._params = params
        else:
            self._inits = cuda.to_device(inits)
            self._params = cuda.to_device(params)
        # spec.key -> live candidate solver, reused across stages.
        self._live: Dict[Any, Any] = {}
        # Keys of family winners kept alive for the final ranking.
        self._protected: set = set()
        # Latest simulator result per solver, read for status codes.
        self._last_result: Dict[int, Any] = {}

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

    def _solve(
        self,
        solver: Any,
        duration: float,
        settling_time: float,
        first: bool,
    ) -> None:
        """Run one solve on the candidate's stream.

        On hardware the launch is enqueued without a sync; under the
        simulator the solve runs synchronously.
        """
        kwargs = {}
        if first and self._drivers is not None:
            kwargs["drivers"] = self._drivers
        if CUDA_SIMULATION:
            kwargs["nan_error_trajectories"] = False
        else:
            kwargs["on_device"] = True
        result = solver.solve(
            self._inits,
            self._params,
            duration=duration,
            settling_time=settling_time,
            t0=self._t0,
            blocksize=self._blocksize,
            **kwargs,
        )
        if CUDA_SIMULATION:
            self._last_result[id(solver)] = result

    def _read_failures(self, solver: Any) -> int:
        """Return the failed-run count of the candidate's last solve."""
        if CUDA_SIMULATION:
            codes = self._last_result[id(solver)].status_codes
        else:
            codes = _device_to_host(
                solver.kernel.device_status_codes
            )
        return int(count_nonzero(asarray(codes).ravel()))

    def _timed_solve(self, solver: Any) -> Tuple[float, int]:
        """Run one full-duration solve; return (ms, failed runs)."""
        if CUDA_SIMULATION:
            start = perf_counter()
            self._solve(
                solver, self._duration, self._settling, first=False
            )
            elapsed_ms = 1000.0 * (perf_counter() - start)
            return elapsed_ms, self._read_failures(solver)
        stream = solver.stream
        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record(stream)
        self._solve(
            solver, self._duration, self._settling, first=False
        )
        end_event.record(stream)
        solver.kernel.synchronize()
        elapsed_ms = cuda.event_elapsed_time(start_event, end_event)
        return float(elapsed_ms), self._read_failures(solver)

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
        pending = []
        seen = set()
        for spec in specs:
            if spec.key in seen:
                continue
            seen.add(spec.key)
            result = CandidateResult(
                spec=spec, stage=stage, runs=self._n_runs
            )
            results.append(result)
            if spec.key in self._live:
                pending.append((result, self._live[spec.key], False))
                continue
            try:
                solver = self._build_solver(spec)
                self._solve(
                    solver,
                    self._screen_duration,
                    self._screen_settling,
                    first=True,
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
            pending.append((result, solver, True))

        # Drop candidates whose failure fraction exceeds the floor.
        screened = []
        for result, solver, fresh in pending:
            if fresh:
                if not CUDA_SIMULATION:
                    solver.kernel.synchronize()
                result.failures = self._read_failures(solver)
            screened.append((result, solver))
        fractions = [
            result.failure_fraction
            for result, _ in screened
        ]
        floor = min(fractions) if fractions else 0.0
        survivors = []
        for result, solver in screened:
            if (
                result.failure_fraction
                > floor + self._failure_tolerance
            ):
                result.dropped = True
                result.reason = (
                    f"screen failures {result.failures}/"
                    f"{result.runs}"
                )
                self._emit(f"  {result.spec.label}: {result.reason}")
                self.close_candidate(result.spec)
                continue
            survivors.append((result, solver))

        for result, solver in survivors:
            times = []
            for _ in range(self._n_repeats):
                elapsed_ms, failures = self._timed_solve(solver)
                times.append(elapsed_ms)
                result.failures = failures
            result.times_ms = tuple(times)
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
            self._last_result.pop(id(solver), None)
            solver.close()

    def prune(self, keep: Sequence[CandidateSpec]) -> None:
        """Release live candidates outside ``keep`` and the protected
        set."""
        keep_keys = {spec.key for spec in keep} | self._protected
        for key in list(self._live):
            if key not in keep_keys:
                solver = self._live.pop(key)
                self._last_result.pop(id(solver), None)
                solver.close()

    def close_all(self) -> None:
        """Release every live candidate solver, protected included."""
        for solver in self._live.values():
            solver.close()
        self._live.clear()
        self._protected.clear()
        self._last_result.clear()


def _clamped_screen_timing(
    base_kwargs: Dict[str, Any],
    duration: float,
    settling_time: float,
    screen_fraction: float,
) -> Tuple[float, float]:
    """Return (screen duration, screen settling) for the tournament.

    The screen duration is ``duration * screen_fraction``, raised to
    every configured output interval and capped at the full duration.
    """
    screen_duration = float(duration) * float(screen_fraction)
    for name in ("save_every", "summarise_every"):
        value = base_kwargs.get(name)
        if value is not None:
            screen_duration = max(screen_duration, float(value))
    screen_duration = min(screen_duration, float(duration))
    if duration > 0.0:
        scale = screen_duration / float(duration)
    else:
        scale = 1.0
    return screen_duration, float(settling_time) * scale


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
    n_repeats: int = 3,
    max_steps: int = 1_000_000,
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
        Fraction of ``duration`` the screening solve integrates,
        raised to any configured output interval.
    n_repeats
        Timed full-duration solves per surviving candidate; the
        lowest time is the candidate's score.
    max_steps
        Step budget flooring every candidate's ``dt_min`` at
        ``(duration + settling_time) / max_steps``.
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
        system feature record.

    Raises
    ------
    ValueError
        If the system declares drivers but none are supplied, or if
        an explicit family is requested on a mass-matrix system.
    """
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
    base_kwargs = _candidate_base_kwargs(
        parent, duration, settling_time, max_steps
    )
    screen_duration, screen_settling = _clamped_screen_timing(
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
        screen_duration=screen_duration,
        screen_settling=screen_settling,
        n_repeats=n_repeats,
        failure_tolerance=failure_tolerance,
        blocksize=blocksize,
        verbose=verbose,
    )

    all_results = []
    family_winners = []
    try:
        for family in families:
            winner = _run_family(runner, family, all_results)
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

    return CalibrationResult(
        candidates=all_results,
        winner=winner,
        equivalent=equivalent,
        features=features,
        applied_settings=applied_settings,
    )


def _run_family(
    runner: _CalibrationRunner,
    family: str,
    all_results: List[CandidateResult],
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
        preconditioner_stage_specs(family, representative),
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
