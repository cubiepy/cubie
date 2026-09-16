"""Per-system selection of algorithm and linear-solver settings.

:func:`run_calibration` (exposed as :meth:`cubie.Solver.calibrate`)
races candidate solver configurations on one representative batch
and reports the fastest one in the top success tier. Each family runs
in stages: preconditioners, linear solvers, error options, tableau
order. Every candidate is timed on the solver itself; measurements
are recorded per configuration and reused. The winner is applied to
the calling solver by default.

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

from typing import Any, Dict, List, Optional, Sequence, Tuple

from attrs import define, frozen
from numpy import asarray

from cubie.batchsolving.comparison import (
    Candidate,
    ComparisonRunner,
    rank_timings,
)
from cubie.integrators.algorithms import resolve_alias
from cubie.integrators.stage_predictors import (
    tableau_supports_dense_prediction,
)


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

    @property
    def solver_settings(self) -> Dict[str, Any]:
        """The algorithm and settings as one ``Solver.update`` dict."""
        return {"algorithm": self.algorithm, **self.settings_dict}


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
        Kernel milliseconds of every timed solve.
    failures
        Runs with a nonzero status code.
    runs
        Trajectory count each solve integrated.
    dropped
        Whether the candidate could not be timed.
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
    def timed(self) -> bool:
        """Whether the candidate has a time."""
        return bool(self.times_ms) and not self.dropped

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

    @property
    def success_rate(self) -> float:
        """Share of the batch that integrated without a status flag."""
        return 1.0 - self.failure_fraction


@define
class CalibrationResult:
    """Complete calibration report.

    Parameters
    ----------
    candidates
        Every candidate measurement from every stage, in run order.
    winner
        Fastest candidate of the top success tier, or ``None`` when no
        candidate was timed.
    ranking
        Every timed configuration: the top success tier by time, then
        the rest by time.
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


def _system_features(
    parent: Any, t0: float, n_runs: int, duration: float
) -> Dict[str, Any]:
    """Return the system description accompanying the measurements."""
    system = parent.system
    sizes = system.sizes
    return {
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
    }


def _scalar_or_none(value: Any) -> Optional[float]:
    """Return the tightest tolerance entry, ``None`` when unset."""
    if value is None:
        return None
    array = asarray(value)
    if array.size == 0:
        return None
    return float(array.min())


class _CalibrationRace:
    """Race stages of candidate specs on one comparison runner.

    Timed configurations are recorded by key and recalled when a later
    stage names the same one.
    """

    def __init__(self, runner: ComparisonRunner) -> None:
        self._runner = runner
        self._recorded: Dict[Any, CandidateResult] = {}
        self.waves: Optional[float] = None

    @property
    def precision(self) -> type:
        """Solve precision of the solver."""
        return self._runner._solver.precision

    def emit(self, message: str) -> None:
        """Log ``message``; print it when verbose."""
        self._runner.emit(message)

    def run_stage(
        self, specs: Sequence[CandidateSpec], stage: str
    ) -> Tuple[List[CandidateResult], List[CandidateResult]]:
        """Time one stage's fresh configurations.

        Returns
        -------
        tuple of (list of CandidateResult, list of CandidateResult)
            New measurements, and the comparison pool including
            recalled results.
        """
        runner = self._runner
        recalled = []
        fresh = []
        seen = set()
        for spec in specs:
            if spec.key in seen:
                continue
            seen.add(spec.key)
            previous = self._recorded.get(spec.key)
            if previous is not None:
                recalled.append(previous)
            else:
                fresh.append(spec)
        results = []
        if fresh:
            candidates = [
                Candidate(spec.label, spec.solver_settings)
                for spec in fresh
            ]
            runner.compile(candidates)
            timings = runner.time(candidates)
            for spec, timing in zip(fresh, timings):
                result = CandidateResult(
                    spec=spec,
                    stage=stage,
                    times_ms=timing.times_ms,
                    failures=timing.failures,
                    runs=timing.runs,
                    dropped=bool(timing.error),
                    reason=timing.error,
                )
                results.append(result)
                if result.timed:
                    self._recorded[spec.key] = result
                    if self.waves is None or timing.waves < self.waves:
                        self.waves = timing.waves
        pool = recalled + [result for result in results if result.timed]
        return results, pool

    @staticmethod
    def stage_winner(
        pool: Sequence[CandidateResult],
    ) -> Optional[CandidateResult]:
        """Return the top-ranked candidate in ``pool``, if any."""
        ranking = rank_timings(pool)
        return ranking[0] if ranking else None

    def ranking(self) -> List[CandidateResult]:
        """Return every recorded configuration, ranked."""
        return rank_timings(list(self._recorded.values()))


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
        candidate shares.
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
    if drivers is not None:
        parent._configure_drivers(drivers)
    features = _system_features(
        parent, t0, inits.shape[1], duration
    )
    runner = ComparisonRunner(
        parent, inits, params, duration, settling_time, t0, verbose
    )
    race = _CalibrationRace(runner)
    all_results = []
    with runner:
        runner.set_batch()
        for family in CALIBRATION_FAMILIES:
            _run_family(race, family, all_results)

    ranking = race.ranking()
    winner = ranking[0] if ranking else None

    applied_settings = {}
    if winner is not None and apply:
        applied_settings = winner.spec.solver_settings
        parent.update(dict(applied_settings))
        runner.emit(f"applied: {winner.spec.label} -> parent solver")

    features["achieved_waves"] = race.waves
    return CalibrationResult(
        candidates=all_results,
        winner=winner,
        ranking=ranking,
        features=features,
        applied_settings=applied_settings,
    )


def _run_family(
    race: _CalibrationRace,
    family: str,
    all_results: List[CandidateResult],
) -> None:
    """Race one family's stages; recorded times carry winners
    forward. Candidates the system cannot build drop individually
    with the error message."""
    if family == "erk":
        race.emit("erk: orders")
        results, _ = race.run_stage(erk_specs(), "erk:orders")
        all_results.extend(results)
        return

    representative = FAMILY_REPRESENTATIVES[family]
    race.emit(f"{family}: preconditioners")
    results, pool = race.run_stage(
        preconditioner_specs(family, representative),
        f"{family}:preconditioners",
    )
    all_results.extend(results)
    best = race.stage_winner(pool)
    if best is None:
        # No viable iterative candidate; race stage 2 with jacobi-0.
        preconditioner = ("jacobi", 0)
    else:
        settings = best.spec.settings_dict
        preconditioner = (
            settings["preconditioner_type"],
            settings["preconditioner_order"],
        )

    race.emit(f"{family}: linear solvers")
    results, pool = race.run_stage(
        linear_solver_specs(family, representative, preconditioner),
        f"{family}:linear-solvers",
    )
    all_results.extend(results)
    best = race.stage_winner(pool)
    if best is None:
        race.emit(f"{family}: no viable configuration")
        return

    option_specs = error_option_specs(
        family, representative, best.spec.settings, race.precision
    )
    if option_specs:
        race.emit(f"{family}: error options")
        results, pool = race.run_stage(
            option_specs, f"{family}:error-options"
        )
        all_results.extend(results)
        option_best = race.stage_winner(pool)
        if option_best is not None:
            best = option_best

    race.emit(f"{family}: orders")
    results, _ = race.run_stage(
        order_specs(family, best.spec.settings, representative),
        f"{family}:orders",
    )
    all_results.extend(results)
