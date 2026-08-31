"""Per-system selection of algorithm and linear-solver settings.

:func:`run_calibration` (exposed as :meth:`cubie.Solver.calibrate`)
races candidate solver configurations on one representative batch
and reports the fastest one that integrates it acceptably. Each
family runs in stages: preconditioners, linear solvers, error
options, tableau order. Short trial solves drop failing or slow
candidates before full-length timing; full-length measurements are
recorded per configuration and reused. The winner is applied to the
calling solver by default.

Published Objects
-----------------
:class:`CandidateSpec`
    One candidate configuration: an algorithm alias plus settings.
:class:`CandidateResult`
    Measured outcome for one candidate.
:class:`CalibrationResult`
    Winner, ranking, and every candidate measurement.
:func:`run_calibration`
    Race the candidate configurations for a configured solver.
"""

import logging
from math import ceil
from warnings import warn
from typing import Any, Dict, List, Optional, Sequence, Tuple

from attrs import define, frozen
from numpy import asarray, count_nonzero, isfinite, ndarray
from numpy.linalg import eigvals

from cubie.cuda_simsafe import cuda
from cubie.integrators.algorithms import resolve_alias
from cubie.integrators.stage_predictors import (
    tableau_supports_dense_prediction,
)

logger = logging.getLogger(__name__)


CALIBRATION_FAMILIES = ("erk", "dirk", "firk", "rosenbrock")
"""Algorithm families raced, in run order."""

# Bare family keywords default to errorless DIRK/FIRK tableaus.
FAMILY_REPRESENTATIVES = {
    "dirk": "kvaerno3",
    "firk": "radau_iia_5",
    "rosenbrock": "ros3p",
}
"""Adaptive mid-order tableau racing each implicit family's settings."""

FAMILY_ORDERS = {
    "erk": ("bogacki-shampine-32", "tsit5", "vern7"),
    "dirk": ("kvaerno3", "l_stable_sdirk_4", "kvaerno5"),
    "firk": ("radau_iia_3", "radau_iia_5", "radau_iia_9"),
    "rosenbrock": ("rosenbrock23", "ros3p", "rodas3p"),
}
"""Adaptive tableau aliases spanning low to high order per family."""

PRECONDITIONERS = (
    ("jacobi", 0),
    ("jacobi", 1),
    ("jacobi", 2),
    ("neumann", 1),
    ("neumann", 2),
    ("none", 0),
)
"""(type, order) pairs raced in each implicit family's first stage."""

TRIAL_FRACTION = 0.0625
"""Trial-solve length as a fraction of the calibration duration."""

TRIAL_BUDGET_FACTOR = 3.0
"""Trial budget as a multiple of the fastest candidate's trial."""

FAILURE_TOLERANCE = 0.01
"""Allowed failed-run fraction above the best candidate's."""

N_REPEATS = 3
"""Timed full-duration solves per surviving candidate."""

BLOCKSIZE = 256
"""CUDA block size the races launch with."""

BLOCKSIZES = (64, 128, 256)
"""Block sizes the winner is re-timed across."""

_SETTING_FIELDS = (
    "linear_correction_type",
    "preconditioner_type",
    "preconditioner_order",
    "inexact_newton",
    "prefactored",
    "use_smoothed_error",
    "attempt_dense_prediction",
)


@frozen
class CandidateSpec:
    """One candidate configuration in the calibration race.

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
        (name, value) pairs of solver keyword overrides.
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
        return (self.algorithm, tuple(sorted(self.settings)))


@define
class CandidateResult:
    """Measured outcome for one candidate.

    Parameters
    ----------
    spec
        The candidate configuration measured.
    stage
        Name of the race stage the measurement belongs to.
    times_ms
        Per-solve times in milliseconds for the full-length solves.
    trial_ms
        Time of the last short trial solve in milliseconds; ``None``
        when the candidate never ran one.
    failures
        Failed-run count from the last solve inspected.
    runs
        Trajectory count each solve integrated.
    dropped
        Whether the candidate was removed before full-length timing.
    reason
        Why the candidate was dropped, empty otherwise.
    """

    spec: CandidateSpec
    stage: str
    times_ms: Tuple[float, ...] = ()
    trial_ms: Optional[float] = None
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
        Fastest candidate that integrated the batch acceptably, or
        ``None`` when no candidate did.
    ranking
        Every configuration that survived full-length timing,
        fastest first.
    features
        System description (sizes, precision, tolerances, ...)
        accompanying the measurements.
    applied_settings
        Settings applied to the calling solver, empty when nothing
        was applied.
    """

    candidates: List[CandidateResult]
    winner: Optional[CandidateResult]
    ranking: List[CandidateResult]
    features: Dict[str, Any]
    applied_settings: Dict[str, Any]

    def summary(self) -> str:
        """Return a formatted table of every candidate measurement."""
        lines = []
        header = (
            f"{'stage':<28}{'candidate':<40}{'best ms':>10}"
            f"{'failed':>8}  note"
        )
        lines.append(header)
        lines.append("-" * len(header))
        ranks = {
            id(result): position + 1
            for position, result in enumerate(self.ranking)
        }
        for result in self.candidates:
            if result.dropped:
                note = f"dropped: {result.reason}"
                best = ""
            else:
                best = f"{result.best_ms:.3f}"
                note = ""
                rank = ranks.get(id(result))
                if rank == 1:
                    note = "winner"
                elif rank is not None:
                    note = f"rank {rank}"
            lines.append(
                f"{result.stage:<28}{result.spec.label:<40}"
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
                trial_ms=result.trial_ms,
                times_ms=";".join(
                    f"{value:.4f}" for value in result.times_ms
                ),
                failures=result.failures,
                dropped=result.dropped,
                reason=result.reason,
            )
            for name in _SETTING_FIELDS:
                record[name] = result.spec.settings_dict.get(name)
            records.append(record)
        return records


def preconditioner_specs(
    family: str, representative: str
) -> List[CandidateSpec]:
    """Return one BiCGSTAB candidate per preconditioner pair.

    Pairs the package rejects for the system fail at build and are
    reported as dropped candidates.
    """
    newton = ()
    if family != "rosenbrock":
        newton = (("inexact_newton", False),)
    specs = []
    for p_type, p_order in PRECONDITIONERS:
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
                )
                + newton,
            )
        )
    return specs


def linear_solver_specs(
    family: str,
    representative: str,
    preconditioner: Tuple[str, int],
) -> List[CandidateSpec]:
    """Return the linear-solver and Newton-variant candidates.

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
        One candidate per linear solver for Rosenbrock-W; the
        Newton families cross the linear solvers with the Newton
        variants each supports.
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
        for inexact in (False, True):
            variant = "inexact" if inexact else "exact"
            specs.append(
                CandidateSpec(
                    label=f"{prefix} {name} {tag} {variant}",
                    family=family,
                    algorithm=representative,
                    settings=(
                        ("linear_correction_type", correction),
                        ("inexact_newton", inexact),
                    )
                    + iterative,
                )
            )
    specs.append(
        CandidateSpec(
            label=f"{prefix} lu exact",
            family=family,
            algorithm=representative,
            settings=(
                ("linear_correction_type", "lu"),
                ("inexact_newton", False),
            ),
        )
    )
    if family == "dirk":
        # DIRK alone separates refactoring from stored factors.
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


def error_option_specs(
    family: str,
    representative: str,
    base_settings: Tuple[Tuple[str, Any], ...],
    precision: type,
) -> List[CandidateSpec]:
    """Return the smoothed-error and stage-predictor candidates.

    Parameters
    ----------
    family
        Implicit family key.
    representative
        Tableau alias the candidates run on.
    base_settings
        Winning settings from the linear-solver stage, carried into
        every candidate.
    precision
        Solve precision, deciding stage-predictor availability.

    Returns
    -------
    list of CandidateSpec
        The on/off cross of each error option the representative
        tableau can compile; empty when it supports none.
    """
    _, tableau = resolve_alias(representative)
    options = []
    if tableau.supports_smoothed_error:
        options.append(("use_smoothed_error", (True, False)))
    if (
        family in ("dirk", "firk")
        and tableau_supports_dense_prediction(tableau)
        and tableau.dense_prediction_ratio_limit(precision) > 0.0
    ):
        options.append(("attempt_dense_prediction", (True, False)))
    if not options:
        return []
    combos = [()]
    for name, values in options:
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


def order_specs(
    family: str,
    settings: Tuple[Tuple[str, Any], ...],
    incumbent: str,
) -> List[CandidateSpec]:
    """Return the family's order race under the winning settings.

    The incumbent tableau leads; the rest of the family's order
    list follows under the same settings.
    """
    aliases = [incumbent] + [
        alias
        for alias in FAMILY_ORDERS[family]
        if alias != incumbent
    ]
    return [
        CandidateSpec(
            label=alias,
            family=family,
            algorithm=alias,
            settings=settings,
        )
        for alias in aliases
    ]


def erk_specs() -> List[CandidateSpec]:
    """Return the explicit family's order race."""
    return [
        CandidateSpec(label=alias, family="erk", algorithm=alias)
        for alias in FAMILY_ORDERS["erk"]
    ]


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
    """Return the system description accompanying the measurements.

    Covers size counts, precision, mass-matrix flag, tolerances, and
    the spectral radius of the state Jacobian at the initial state
    (``None`` when it cannot be evaluated).
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
    """Build, trial, and time candidate solvers for one calibration.

    Candidates share the parent's stream group; a candidate's host
    build and kernel compile overlap the solves queued before it.
    Full-length measurements are recorded per configuration and
    reused when a later stage names the same configuration.
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
        trials: Sequence[Tuple[float, float]],
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
        self._trials = tuple(trials)
        self._verbose = bool(verbose)
        self._n_runs = int(inits.shape[1])
        self._inits = cuda.to_device(inits)
        self._params = cuda.to_device(params)
        # Configuration key -> fully timed CandidateResult.
        self._recorded: Dict[Any, CandidateResult] = {}
        self.achieved_waves = None

    @property
    def n_runs(self) -> int:
        """Trajectory count of the calibration batch."""
        return self._n_runs

    @property
    def precision(self) -> type:
        """Solve precision of the parent solver."""
        return self._parent.precision

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

    def _compile(self, solver: Any) -> None:
        """Compile the candidate's kernel without launching it."""
        solver.compile(
            self._inits,
            self._params,
            drivers=self._drivers,
            duration=self._trials[0][0],
            settling_time=self._trials[0][1],
            t0=self._t0,
        )

    def _launch(
        self,
        solver: Any,
        duration: float,
        settling_time: float,
        blocksize: int = BLOCKSIZE,
    ) -> Tuple[Any, Any]:
        """Enqueue one solve, no sync; return its CUDA event pair."""
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
            blocksize=blocksize,
            on_device=True,
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

    def _sync(self, solver: Any) -> None:
        """Wait for every launch queued on the shared stream."""
        solver.kernel.synchronize()

    def _probe_waves(self, solver: Any) -> None:
        """Record achieved occupancy waves; warn once when under two."""
        if self.achieved_waves is not None:
            return
        try:
            waves = _achieved_waves(solver)
        except Exception:
            logger.debug("Occupancy probe failed", exc_info=True)
            return
        self.achieved_waves = waves
        if waves < 2.0:
            warn(
                f"Calibration batch fills {waves:.2f} occupancy "
                "waves at the first candidate's kernel; below two "
                "waves timings may not rank configurations reliably."
            )

    def run_stage(
        self, specs: Sequence[CandidateSpec], stage: str
    ) -> Tuple[List[CandidateResult], List[CandidateResult]]:
        """Measure one stage of candidates.

        Fresh configurations pass the ascending trial gates, then
        run ``N_REPEATS`` full-length solves queued round-robin and
        read after one synchronize. Recorded configurations are not
        solved again.

        Returns
        -------
        tuple of (list of CandidateResult, list of CandidateResult)
            New measurements, and the comparison pool including
            recalled results.
        """
        results = []
        recalled = []
        live = []
        seen = set()
        timed = []
        try:
            for spec in specs:
                if spec.key in seen:
                    continue
                seen.add(spec.key)
                previous = self._recorded.get(spec.key)
                if previous is not None:
                    recalled.append(previous)
                    continue
                result = CandidateResult(
                    spec=spec, stage=stage, runs=self._n_runs
                )
                results.append(result)
                solver = None
                try:
                    solver = self._build_solver(spec)
                    # Compile overlaps solves queued on the stream.
                    self._compile(solver)
                    token = self._launch(solver, *self._trials[0])
                except Exception as exc:
                    result.dropped = True
                    result.reason = f"{type(exc).__name__}: {exc}"
                    self._emit(
                        f"  {spec.label}: failed ({result.reason})"
                    )
                    if solver is not None:
                        solver.close()
                    continue
                live.append([result, solver, token])

            for index, trial in enumerate(self._trials):
                if not live:
                    break
                if index > 0:
                    for entry in live:
                        entry[2] = self._launch(entry[1], *trial)
                self._sync(live[0][1])
                for result, solver, token in live:
                    result.trial_ms = self._elapsed_ms(token)
                    result.failures = self._read_failures(solver)
                self._probe_waves(live[0][1])
                live = self._gate_trials(live, recalled)

            # Queue all full-length solves; read after one sync.
            tokens = {id(entry[0]): [] for entry in live}
            for _ in range(N_REPEATS):
                for result, solver, _ in live:
                    tokens[id(result)].append(
                        self._launch(
                            solver, self._duration, self._settling
                        )
                    )
            if live:
                self._sync(live[0][1])
            for result, solver, _ in live:
                result.times_ms = tuple(
                    self._elapsed_ms(token)
                    for token in tokens[id(result)]
                )
                result.failures = self._read_failures(solver)
                self._emit(
                    f"  {result.spec.label}: "
                    f"{result.best_ms:.3f} ms "
                    f"({result.failures} failed)"
                )
            timed = [entry[0] for entry in live]
            self._gate_failures(timed, recalled)
            for result in timed:
                if not result.dropped:
                    self._recorded[result.spec.key] = result
        finally:
            for _, solver, _ in live:
                solver.close()
        pool = recalled + [
            result for result in timed if not result.dropped
        ]
        return results, pool

    def _gate_trials(
        self,
        live: List[List[Any]],
        recalled: Sequence[CandidateResult],
    ) -> List[List[Any]]:
        """Drop candidates that fail or far exceed the trial budget."""
        fractions = [
            entry[0].failure_fraction for entry in live
        ] + [result.failure_fraction for result in recalled]
        floor = min(fractions) if fractions else 0.0
        times = [entry[0].trial_ms for entry in live]
        budget = (
            min(times) * TRIAL_BUDGET_FACTOR if times else None
        )
        survivors = []
        for entry in live:
            result, solver, _ = entry
            if result.failure_fraction > floor + FAILURE_TOLERANCE:
                result.dropped = True
                result.reason = (
                    f"trial failures {result.failures}/"
                    f"{result.runs}"
                )
            elif budget is not None and result.trial_ms > budget:
                result.dropped = True
                result.reason = (
                    f"trial {result.trial_ms:.1f} ms over budget "
                    f"{budget:.1f} ms"
                )
            else:
                survivors.append(entry)
                continue
            self._emit(f"  {result.spec.label}: {result.reason}")
            solver.close()
        return survivors

    def _gate_failures(
        self,
        timed: Sequence[CandidateResult],
        recalled: Sequence[CandidateResult],
    ) -> None:
        """Drop timed candidates on their full-length failure counts."""
        fractions = [
            result.failure_fraction for result in timed
        ] + [result.failure_fraction for result in recalled]
        floor = min(fractions) if fractions else 0.0
        for result in timed:
            if result.failure_fraction > floor + FAILURE_TOLERANCE:
                result.dropped = True
                result.reason = (
                    f"failures {result.failures}/{result.runs}"
                )
                self._emit(
                    f"  {result.spec.label}: {result.reason}"
                )

    def stage_winner(
        self, pool: Sequence[CandidateResult]
    ) -> Optional[CandidateResult]:
        """Return the fastest viable candidate in ``pool``, if any."""
        viable = [
            result
            for result in pool
            if not result.dropped and result.times_ms
        ]
        if not viable:
            return None
        return min(viable, key=lambda result: result.best_ms)

    def ranking(self) -> List[CandidateResult]:
        """Return every recorded configuration, fastest first."""
        return sorted(
            self._recorded.values(),
            key=lambda result: result.best_ms,
        )

    def sweep_blocksize(
        self, winner: CandidateResult
    ) -> List[CandidateResult]:
        """Re-time the winner across block sizes; the race block
        size reuses the winner's recorded solves."""
        rows = []
        fresh = []
        for size in BLOCKSIZES:
            spec = CandidateSpec(
                label=f"{winner.spec.label} blocksize {size}",
                family=winner.spec.family,
                algorithm=winner.spec.algorithm,
                settings=winner.spec.settings,
            )
            result = CandidateResult(
                spec=spec, stage="blocksize", runs=self._n_runs
            )
            rows.append(result)
            if size == BLOCKSIZE:
                result.times_ms = winner.times_ms
                result.failures = winner.failures
            else:
                fresh.append((size, result))
        if fresh:
            solver = self._build_solver(winner.spec)
            try:
                tokens = {id(result): [] for _, result in fresh}
                # Block size is a compile setting: group launches by
                # size and compile before the first timed launch so
                # host-side rebuilds stay out of the event windows.
                for size, result in fresh:
                    solver.update({"blocksize": size}, silent=True)
                    self._compile(solver)
                    for _ in range(N_REPEATS):
                        tokens[id(result)].append(
                            self._launch(
                                solver,
                                self._duration,
                                self._settling,
                                blocksize=size,
                            )
                        )
                self._sync(solver)
                failures = self._read_failures(solver)
                for size, result in fresh:
                    result.times_ms = tuple(
                        self._elapsed_ms(token)
                        for token in tokens[id(result)]
                    )
                    result.failures = failures
            finally:
                solver.close()
        for result in rows:
            self._emit(
                f"  {result.spec.label}: {result.best_ms:.3f} ms "
                f"({result.failures} failed)"
            )
        return rows


def _achieved_waves(solver: Any) -> float:
    """Return occupancy waves the batch fills at the actual geometry."""
    kernel_factory = solver.kernel
    (kern,) = kernel_factory.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    runs = int(kernel_factory.run_params[0].runs)
    # Shared memory is static, so the occupancy query carries no
    # dynamic allocation; the compiled geometry is the launch geometry.
    actual_blocksize = kernel_factory.launch_blocksize
    context = cuda.current_context()
    blocks_per_sm = context.get_active_blocks_per_multiprocessor(
        cufunc, actual_blocksize, 0
    )
    device = cuda.get_current_device()
    runs_per_block = kernel_factory.runs_per_block
    total_blocks = ceil(runs / runs_per_block)
    resident = blocks_per_sm * device.MULTIPROCESSOR_COUNT
    return total_blocks / resident


def _trial_durations(
    base_kwargs: Dict[str, Any],
    duration: float,
    settling_time: float,
) -> Tuple[Tuple[float, float], ...]:
    """Return the ascending (duration, settling) trial lengths.

    A short trial at ``TRIAL_FRACTION**2`` of the duration precedes
    one at ``TRIAL_FRACTION``; each is raised to every configured
    output interval and capped at the full duration, and coinciding
    lengths collapse into one.
    """
    fraction = TRIAL_FRACTION
    trials = []
    for trial_fraction in (fraction * fraction, fraction):
        trial_duration = float(duration) * trial_fraction
        for name in ("save_every", "summarise_every"):
            value = base_kwargs.get(name)
            if value is not None:
                trial_duration = max(trial_duration, float(value))
        trial_duration = min(trial_duration, float(duration))
        if duration > 0.0:
            scale = trial_duration / float(duration)
        else:
            scale = 1.0
        trial = (trial_duration, float(settling_time) * scale)
        if not trials or trials[-1] != trial:
            trials.append(trial)
    return tuple(trials)


def run_calibration(
    parent: Any,
    initial_values: Any,
    parameters: Any,
    drivers: Optional[Dict[str, Any]] = None,
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    grid_type: str = "verbatim",
    apply: bool = True,
    verbose: bool = True,
) -> CalibrationResult:
    """Race solver configurations for a solver and pick the fastest.

    Parameters
    ----------
    parent
        The configured :class:`~cubie.batchsolving.solver.Solver`
        whose system, tolerances, and output configuration every
        candidate replicates.
    initial_values
        Initial state values for each integration run, as accepted
        by :meth:`Solver.solve`.
    parameters
        Parameter values for each run, as accepted by
        :meth:`Solver.solve`.
    drivers
        Driver samples or configuration matching
        :class:`cubie.array_interpolator.ArrayInterpolator`.
    duration
        Total integration time candidates are ranked on.
    settling_time
        Warm-up period before recording outputs.
    t0
        Initial integration time.
    grid_type
        Strategy for constructing the integration grid from inputs.
        Only used when dict inputs trigger grid construction.
    apply
        Apply the winner's configuration to ``parent`` when ``True``.
    verbose
        Print per-candidate progress lines.

    Returns
    -------
    CalibrationResult
        Winner, ranking, and every candidate measurement. A
        candidate that fails to build or integrate is reported as
        dropped with its error message.

    Raises
    ------
    ValueError
        If the system declares drivers but none are supplied.
    """
    system = parent.system
    if system.sizes.drivers > 0 and drivers is None:
        raise ValueError(
            "The system declares drivers; calibrate requires the "
            "driver samples that solves will use."
        )

    inits, params = parent.build_grid(
        initial_values, parameters, grid_type=grid_type
    )
    base_kwargs = _candidate_base_kwargs(parent, duration)
    trials = _trial_durations(base_kwargs, duration, settling_time)
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
        trials=trials,
        verbose=verbose,
    )

    all_results = []
    for family in CALIBRATION_FAMILIES:
        _run_family(runner, family, all_results)

    ranking = runner.ranking()
    winner = ranking[0] if ranking else None
    if winner is not None:
        runner._emit("blocksize: winner re-timed")
        all_results.extend(runner.sweep_blocksize(winner))

    applied_settings = {}
    if winner is not None and apply:
        applied_settings = {
            "algorithm": winner.spec.algorithm,
            **winner.spec.settings_dict,
        }
        parent.update(dict(applied_settings))
        runner._emit(
            f"applied: {winner.spec.label} -> parent solver"
        )

    features["achieved_waves"] = runner.achieved_waves
    return CalibrationResult(
        candidates=all_results,
        winner=winner,
        ranking=ranking,
        features=features,
        applied_settings=applied_settings,
    )


def _run_family(
    runner: _CalibrationRunner,
    family: str,
    all_results: List[CandidateResult],
) -> None:
    """Race one family's stages; recorded times carry winners
    forward. Candidates the system cannot build drop individually
    with the error message."""
    if family == "erk":
        runner._emit("erk: orders")
        results, _ = runner.run_stage(erk_specs(), "erk:orders")
        all_results.extend(results)
        return

    representative = FAMILY_REPRESENTATIVES[family]
    runner._emit(f"{family}: preconditioners")
    results, pool = runner.run_stage(
        preconditioner_specs(family, representative),
        f"{family}:preconditioners",
    )
    all_results.extend(results)
    best = runner.stage_winner(pool)
    if best is None:
        # No viable iterative candidate; race stage 2 with jacobi-0.
        preconditioner = ("jacobi", 0)
    else:
        settings = best.spec.settings_dict
        preconditioner = (
            settings["preconditioner_type"],
            settings["preconditioner_order"],
        )

    runner._emit(f"{family}: linear solvers")
    results, pool = runner.run_stage(
        linear_solver_specs(family, representative, preconditioner),
        f"{family}:linear-solvers",
    )
    all_results.extend(results)
    best = runner.stage_winner(pool)
    if best is None:
        runner._emit(f"{family}: no viable configuration")
        return

    option_specs = error_option_specs(
        family, representative, best.spec.settings, runner.precision
    )
    if option_specs:
        runner._emit(f"{family}: error options")
        results, pool = runner.run_stage(
            option_specs, f"{family}:error-options"
        )
        all_results.extend(results)
        option_best = runner.stage_winner(pool)
        if option_best is not None:
            best = option_best

    runner._emit(f"{family}: orders")
    results, _ = runner.run_stage(
        order_specs(family, best.spec.settings, representative),
        f"{family}:orders",
    )
    all_results.extend(results)
