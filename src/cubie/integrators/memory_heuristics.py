"""Measured heuristics for default CUDA buffer memory locations.

Gates operate on registry-declared byte sizes and are calibrated per
GPU architecture by ``benchmarks/memory_location_sweep.py --fit``
(explicit rules) and ``benchmarks/placement_study.py`` (implicit
rules). Unknown architectures fall back to the ``DEFAULT_ARCH``
entry.
"""


from attrs import define
from numpy import dtype as np_dtype

from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import compute_capability_code
from cubie.integrators.algorithms import (
    BackwardsEulerStep,
    DIRKStep,
    ERKStep,
    FIRKStep,
)

MEASURED_STEP_TYPES = (
    ERKStep,
    DIRKStep,
    FIRKStep,
    BackwardsEulerStep,
)
"""Algorithm families with benchmarked placement rules."""

STATE_PAIR_KEYS = ("state_location", "proposed_state_location")


@define(frozen=True)
class MemoryThresholds:
    """Placement gates calibrated for one GPU architecture.

    Attributes
    ----------
    heavy_spill_bytes : int
        Explicit-branch footprint marking a spilled kernel.
    spill_floor_bytes : int
        Minimum footprint for the explicit stage-buffer rule.
    state_pair_max_bytes : int
        Largest state pair moved to shared on the explicit branch.
    explicit_work_max_bytes : int
        Largest explicit stage-buffer group moved to shared.
    explicit_work_min_stages : int
        Minimum stage count for the explicit stage-buffer rule.
    implicit_floor_bytes : int
        Implicit-branch footprint below which nothing relocates.
    implicit_deep_bytes : int
        Implicit-branch footprint marking a deeply spilled kernel.
    implicit_cold_max_bytes : int
        Largest state pair moved to shared on deep implicit spills.
    implicit_stage_max_bytes : int
        Largest implicit stage-buffer group moved to shared.
    """

    heavy_spill_bytes: int
    spill_floor_bytes: int
    state_pair_max_bytes: int
    explicit_work_max_bytes: int
    explicit_work_min_stages: int
    implicit_floor_bytes: int
    implicit_deep_bytes: int
    implicit_cold_max_bytes: int
    implicit_stage_max_bytes: int


THRESHOLDS_BY_ARCH: dict[str, MemoryThresholds] = {
    # RTX 4070 SUPER, cubie 0.9.0 mlir; explicit gates from the
    # sweep fit, implicit gates from the placement study.
    "8.9": MemoryThresholds(
        heavy_spill_bytes=2048,
        spill_floor_bytes=512,
        state_pair_max_bytes=1024,
        explicit_work_max_bytes=512,
        explicit_work_min_stages=7,
        implicit_floor_bytes=768,
        implicit_deep_bytes=1536,
        implicit_cold_max_bytes=1024,
        implicit_stage_max_bytes=768,
    ),
}

DEFAULT_ARCH = "8.9"
"""Fallback architecture for cards without a calibrated entry."""


def resolve_thresholds(
    arch: str | None = None,
) -> MemoryThresholds:
    """Return the thresholds calibrated for a GPU architecture."""
    if arch is None:
        arch = compute_capability_code()
    if arch is None or arch not in THRESHOLDS_BY_ARCH:
        arch = DEFAULT_ARCH
    return THRESHOLDS_BY_ARCH[arch]


@define(frozen=True)
class DeclaredSizes:
    """Registry-declared quantities the placement gates operate on.

    Attributes
    ----------
    itemsize : int
        Bytes per element of the run's precision.
    is_implicit : bool
        Whether the algorithm step embeds a nonlinear solve.
    is_linear : bool
        Whether the step is linearly implicit (Rosenbrock family).
    stacked_width : bool
        Whether the nonlinear solve couples all stages (width > n).
    stage_count : int
        Stage count of the algorithm's tableau (1 when untableaued).
    footprint_bytes : int
        Declared per-thread local plus persistent footprint.
    state_pair_bytes : int
        Size of the state and proposed-state pair.
    work_group_bytes : int
        Non-aliased size of the algorithm step's own buffers.
    work_location_keys : Tuple[str, ...]
        ``{name}_location`` settings for the step's own buffers.
    """

    itemsize: int
    is_implicit: bool
    is_linear: bool
    stacked_width: bool
    stage_count: int
    footprint_bytes: int
    state_pair_bytes: int
    work_group_bytes: int
    work_location_keys: tuple[str, ...]


def declared_sizes(single_integrator_run) -> DeclaredSizes:
    """Measure the declared buffer sizes of a constructed run."""
    loop = single_integrator_run._loop
    algo_step = single_integrator_run._algo_step
    itemsize = np_dtype(single_integrator_run.precision).itemsize

    work_names = buffer_registry.relocatable_buffer_names(algo_step)
    work_elements = buffer_registry.nonaliased_elements(
        algo_step, work_names
    )
    footprint_elements = buffer_registry.declared_local_elements(loop)

    n_states = loop.compile_settings.n_states
    config = algo_step.compile_settings
    tableau = getattr(config, "tableau", None)
    stage_count = tableau.stage_count if tableau is not None else 1
    solver_width = getattr(config, "solver_width", n_states)

    return DeclaredSizes(
        itemsize=itemsize,
        is_implicit=algo_step.is_implicit,
        is_linear=bool(getattr(algo_step, "is_linear", False)),
        stacked_width=solver_width > n_states,
        stage_count=stage_count,
        footprint_bytes=footprint_elements * itemsize,
        state_pair_bytes=2 * n_states * itemsize,
        work_group_bytes=work_elements * itemsize,
        work_location_keys=tuple(
            f"{name}_location" for name in work_names
        ),
    )


def placement_candidates(
    sizes: DeclaredSizes,
    thresholds: MemoryThresholds,
) -> list[tuple[str, ...]]:
    """Return the placement groups whose gates fire, best first."""
    itemsize = sizes.itemsize
    footprint = sizes.footprint_bytes
    candidates: list[tuple[bool, tuple[str, ...]]] = []

    if sizes.is_implicit:
        if footprint < thresholds.implicit_floor_bytes:
            return []
        # Rosenbrock stage relocation corrupts solves; kept local.
        stage_allowed = (
            not sizes.is_linear
            and sizes.stage_count >= 2
            and 0
            < sizes.work_group_bytes
            <= thresholds.implicit_stage_max_bytes
        )
        deep = footprint >= thresholds.implicit_deep_bytes
        candidates = [
            (
                deep
                and not sizes.stacked_width
                and sizes.state_pair_bytes
                <= thresholds.implicit_cold_max_bytes,
                STATE_PAIR_KEYS,
            ),
            (stage_allowed, sizes.work_location_keys),
        ]
    else:
        if footprint >= thresholds.heavy_spill_bytes:
            candidates = [
                (
                    sizes.state_pair_bytes
                    <= thresholds.state_pair_max_bytes
                    * (itemsize // 4),
                    STATE_PAIR_KEYS,
                ),
            ]
        elif footprint >= thresholds.spill_floor_bytes:
            candidates = [
                (
                    0
                    < sizes.work_group_bytes
                    <= thresholds.explicit_work_max_bytes
                    and sizes.stage_count
                    >= thresholds.explicit_work_min_stages,
                    sizes.work_location_keys,
                ),
            ]

    return [keys for fires, keys in candidates if fires]


def auto_memory_locations(
    single_integrator_run,
    user_location_keys: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Return shared placements measured faster than all-local.

    A buffer group containing any user-supplied ``*_location`` key
    is skipped whole; at most one group is selected.
    """
    algo_step = single_integrator_run._algo_step
    if not isinstance(algo_step, MEASURED_STEP_TYPES):
        return {}

    sizes = declared_sizes(single_integrator_run)
    thresholds = resolve_thresholds()

    for keys in placement_candidates(sizes, thresholds):
        if not (set(keys) & set(user_location_keys)):
            return {key: "shared" for key in keys}
    return {}
