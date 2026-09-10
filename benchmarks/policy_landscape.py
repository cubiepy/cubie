#!/usr/bin/env python
"""Engine for the performance-policy landscape benchmarks.

One ``Solver`` per candidate kernel (arm), the launch cells of each
arm, timing with the ``Solver.optimize`` protocol, one JSON row per
configuration, and scoring helpers. Compile workers fill the shared
kernel cache first; arms with an identical cubin are aliases and are
timed once; timing runs in blocks of ``--block-arms`` solvers with the
first arm re-timed in every block. Lock the SM and memory clocks
(``nvidia-smi -lgc``, ``-lmc``) on a quiet GPU.
"""

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import time
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import cubie
from cubie.backend.utils import (
    INSTRUCTION_CACHE_BYTES,
    active_blocks_per_multiprocessor,
    device_hardware,
    kernel_resources,
)
from cubie.batchsolving.optimize import (
    LOCAL_LAUNCH_BLOCKSIZES,
    SHARED_LAUNCH_BLOCKSIZES,
)
from cubie.cache_root import get_cache_root, set_cache_root
from cubie.cuda_backend import IS_MLIR
from cubie.cuda_simsafe import (
    ALL_UNROLL_PARAMETERS,
    CUDA_SIMULATION,
    UnrollChoice,
    cuda,
)
from cubie.buffer_registry import buffer_registry
from cubie.time_logger import default_timelogger

FULL = UnrollChoice.FULL
ROLLED = UnrollChoice.ROLLED
PRECISION = np.float32
NATURAL = 99
"""``resident_blocks`` value above any occupancy: the natural launch."""

ROUNDS = 2
TIMED_SOLVES = 3
GPU_WARM_SOLVES = 3
CAP = 2.0
REFERENCE_BLOCKSIZE = 64
BLOCK_ARMS = 8
"""Solvers alive at once during timing; each holds its batch buffers."""

COMPILE_TASKS_PER_CHILD = 8
"""Compiles per worker process; a replaced worker returns its memory."""

UNROLL_GROUPS = (
    "unroll_stage",
    "unroll_step_element",
    "unroll_accumulator",
    "unroll_solver_element",
    "unroll_norms",
    "unroll_other_small",
    "unroll_newton_exits",
    "unroll_krylov_exits",
)
"""Every unroll group, in ``UnrollFlags`` order."""

# --- systems -----------------------------------------------------------


def build_lorenz():
    """The three-state Lorenz system of the publication benchmark."""
    return cubie.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": 21.0},
        constants={"sigma": 10.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=PRECISION,
    )


def build_lorenz96(n):
    lines = []
    for i in range(1, n + 1):
        ip1 = i % n + 1
        im1 = (i - 2) % n + 1
        im2 = (i - 3) % n + 1
        lines.append(
            "dx{0} = (x{1} - x{2}) * x{3} - x{0} + F".format(
                i, ip1, im2, im1
            )
        )
    states = {
        "x{0}".format(i): 9.0 if i == 1 else 8.0 for i in range(1, n + 1)
    }
    return cubie.create_ODE_system(
        "\n".join(lines),
        states=states,
        parameters={"F": 8.0},
        name=f"Lorenz96_{n}",
        precision=PRECISION,
    )


FABBRI_CELLML = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)
FABBRI_PARAMETERS = (
    "Rate_modulation_experiments_ACh",
    "Rate_modulation_experiments_Iso_cas",
)


def build_fabbri():
    """The Fabbri-Linder sinoatrial model with autonomic modulation on."""
    system = cubie.load_cellml_model(
        str(FABBRI_CELLML),
        precision=PRECISION,
        parameters=list(FABBRI_PARAMETERS),
        voltage_variable="Membrane$V_ode",
    )
    system.set_constants({"Rate_modulation_experiments_ANS": 1.0})
    return system


def build_chain(n, consts_per_eq, n_params=2):
    """Nonlinear nearest-neighbour ring chain of the placement bank."""
    rng = np.random.default_rng(1234)
    eqs = []
    constants = {}
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        terms = [f"0.2*x{im1} + 0.3*x{ip1}"]
        for c in range(consts_per_eq):
            cname = f"k{i}_{c}"
            constants[cname] = float(rng.uniform(0.5, 5.0))
            if c == 0:
                terms.append(f"-{cname}*x{i}")
            else:
                terms.append(f"+ {cname}*x{ip1}/(1.0 + x{i}*x{i})")
        pj = i % n_params
        terms.append(f"+ 0.05*p{pj}*x{i}*x{ip1}")
        eqs.append(f"dx{i} = " + " ".join(terms))
    return cubie.create_ODE_system(
        dxdt=eqs,
        states={f"x{i}": 0.5 for i in range(n)},
        parameters={f"p{j}": 1.0 for j in range(n_params)},
        constants=constants,
        precision=PRECISION,
        name=f"chain_{n}s_{consts_per_eq}c",
    )


def grid_param(name, low, high):
    def grid(solver, n_runs):
        return solver.build_grid(
            parameters={name: np.linspace(low, high, n_runs)}
        )

    return grid


def grid_fabbri(solver, n_runs):
    """ACh by Iso mesh, truncated to ``n_runs`` trajectories."""
    side = int(np.ceil(np.sqrt(n_runs)))
    ach, iso = np.meshgrid(
        np.linspace(0.0, 2e-8, side), np.linspace(0.0, 1.0, side)
    )
    return solver.build_grid(
        parameters={
            FABBRI_PARAMETERS[0]: ach.ravel()[:n_runs],
            FABBRI_PARAMETERS[1]: iso.ravel()[:n_runs],
        }
    )


def grid_chain(solver, n_runs):
    return solver.build_grid(
        parameters={
            "p0": np.linspace(0.9, 1.1, n_runs),
            "p1": np.linspace(1.1, 0.9, n_runs),
        }
    )


TIGHT = {"atol": 1e-6, "rtol": 1e-6, "dt_min": 1e-12, "dt_max": 1e3}
FABBRI_TOLS = {"atol": 1e-6, "rtol": 1e-4, "dt_min": 1e-12, "dt_max": 1e-2}

SYSTEMS = {
    "lorenz": dict(
        build=build_lorenz, grid=grid_param("rho", 0.0, 21.0),
        n_states=3, kwargs=TIGHT, erk_duration=512.0,
    ),
    "lorenz96_10": dict(
        build=lambda: build_lorenz96(10), grid=grid_param("F", 0.0, 16.0),
        n_states=10, kwargs=TIGHT, erk_duration=32.0,
    ),
    "lorenz96_20": dict(
        build=lambda: build_lorenz96(20), grid=grid_param("F", 0.0, 16.0),
        n_states=20, kwargs=TIGHT, erk_duration=32.0,
    ),
    "lorenz96_40": dict(
        build=lambda: build_lorenz96(40), grid=grid_param("F", 0.0, 16.0),
        n_states=40, kwargs=TIGHT, erk_duration=32.0,
    ),
    "chain20": dict(
        build=lambda: build_chain(20, 3), grid=grid_chain,
        n_states=20, kwargs=TIGHT, erk_duration=51.2,
    ),
    "chain32": dict(
        build=lambda: build_chain(32, 3), grid=grid_chain,
        n_states=32, kwargs=TIGHT, erk_duration=25.6,
    ),
    "chain32_c8": dict(
        build=lambda: build_chain(32, 8), grid=grid_chain,
        n_states=32, kwargs=TIGHT, erk_duration=51.2,
    ),
    "chain64": dict(
        build=lambda: build_chain(64, 3), grid=grid_chain,
        n_states=64, kwargs=TIGHT, erk_duration=51.2,
    ),
    "fabbri": dict(
        build=build_fabbri, grid=grid_fabbri,
        n_states=35, kwargs=FABBRI_TOLS, erk_duration=1.0,
    ),
}

# --- algorithms --------------------------------------------------------

LU = dict(linear_correction_type="lu", inexact_newton=True, prefactored=True)
BICG = dict(
    linear_correction_type="bicgstab",
    preconditioner_type="jacobi",
    inexact_newton=True,
    prefactored=True,
)

ALGOS = {
    "tsit5": ("ERK", "none", dict(algorithm="tsit5")),
    "bogacki-shampine-32": (
        "ERK", "none", dict(algorithm="bogacki-shampine-32")
    ),
    "vern7": ("ERK", "none", dict(algorithm="vern7")),
    "kvaerno3": ("DIRK", "lu", dict(algorithm="kvaerno3", **LU)),
    "kvaerno5": ("DIRK", "lu", dict(algorithm="kvaerno5", **LU)),
    "kvaerno3_bicgstab": (
        "DIRK", "bicgstab", dict(algorithm="kvaerno3", **BICG)
    ),
    "radau_iia_3": ("FIRK", "lu", dict(algorithm="radau_iia_3", **LU)),
    "radau_iia_5": ("FIRK", "lu", dict(algorithm="radau_iia_5", **LU)),
    "radau_iia_5_bicgstab": (
        "FIRK", "bicgstab", dict(algorithm="radau_iia_5", **BICG)
    ),
    "rosenbrock23": (
        "ROS", "lu", dict(algorithm="rosenbrock23", linear_correction_type="lu")
    ),
    "rosenbrock23_bicgstab": (
        "ROS", "bicgstab",
        dict(
            algorithm="rosenbrock23",
            linear_correction_type="bicgstab",
            preconditioner_type="jacobi",
        ),
    ),
}

# Settled durations of the unroll holdout and placement banks (seconds
# of integration time per solve); ERK configs use the system's
# ``erk_duration``. Unlisted implicit configs fall back to 1.0.
DURATIONS = {
    ("chain20", "kvaerno3"): 25.6,
    ("chain20", "kvaerno3_bicgstab"): 0.4,
    ("chain20", "kvaerno5"): 6.4,
    ("chain20", "radau_iia_3"): 0.8,
    ("chain20", "radau_iia_5"): 51.2,
    ("chain20", "radau_iia_5_bicgstab"): 0.2,
    ("chain32_c8", "kvaerno3"): 0.8,
    ("chain32_c8", "kvaerno3_bicgstab"): 0.05,
    ("chain32_c8", "kvaerno5"): 0.05,
    ("chain32_c8", "radau_iia_3"): 0.05,
    ("chain32_c8", "radau_iia_5"): 0.1,
    ("chain32_c8", "radau_iia_5_bicgstab"): 0.05,
    ("chain64", "kvaerno3"): 0.05,
    ("chain64", "kvaerno3_bicgstab"): 0.05,
    ("chain64", "kvaerno5"): 0.05,
    ("chain64", "radau_iia_3"): 0.05,
    ("chain64", "radau_iia_5"): 0.05,
    ("chain64", "radau_iia_5_bicgstab"): 0.05,
    ("lorenz96_10", "kvaerno3"): 2.0,
    ("lorenz96_10", "kvaerno3_bicgstab"): 1.0,
    ("lorenz96_10", "kvaerno5"): 4.0,
    ("lorenz96_10", "radau_iia_3"): 1.0,
    ("lorenz96_10", "radau_iia_5"): 4.0,
    ("lorenz96_10", "radau_iia_5_bicgstab"): 1.0,
    ("lorenz96_40", "kvaerno3"): 1.0,
    ("lorenz96_40", "kvaerno3_bicgstab"): 1.0,
    ("lorenz96_40", "kvaerno5"): 1.0,
    ("lorenz96_40", "radau_iia_3"): 1.0,
    ("lorenz96_40", "radau_iia_5"): 1.0,
    ("lorenz96_40", "radau_iia_5_bicgstab"): 1.0,
    ("lorenz", "kvaerno3"): 16.0,
    ("lorenz", "kvaerno3_bicgstab"): 2.0,
    ("lorenz", "radau_iia_3"): 2.0,
    ("lorenz", "radau_iia_5"): 512.0,
    ("lorenz", "radau_iia_5_bicgstab"): 8.0,
    ("lorenz", "rosenbrock23"): 512.0,
    ("lorenz", "rosenbrock23_bicgstab"): 8.0,
    ("lorenz96_20", "kvaerno3"): 1.0,
    ("lorenz96_20", "kvaerno5"): 2.0,
    ("chain32", "kvaerno3"): 1.6,
    ("chain32", "kvaerno5"): 3.2,
    ("chain32", "radau_iia_3"): 0.2,
    ("chain32", "radau_iia_5"): 0.4,
    ("chain32", "rosenbrock23"): 25.6,
    ("fabbri", "radau_iia_5"): 1.0,
}


def duration_for(system_name, algo_name):
    family = ALGOS[algo_name][0]
    if family == "ERK":
        return SYSTEMS[system_name]["erk_duration"]
    return DURATIONS.get((system_name, algo_name), 1.0)


def base_kwargs(system_name, algo_name, duration):
    """Solver keyword arguments shared by every arm of a configuration."""
    kwargs = dict(SYSTEMS[system_name]["kwargs"])
    kwargs.update(ALGOS[algo_name][2])
    kwargs.update(
        output_types=["state"],
        save_every=duration,
        time_logging_level="silent",
    )
    return kwargs


# --- arms ----------------------------------------------------------------


@dataclass
class ArmSpec:
    """One candidate kernel: explicit settings on top of the base."""

    label: str
    settings: Dict[str, Any] = field(default_factory=dict)
    auto_performance: bool = True
    in_optimize: bool = False


@dataclass
class Cell:
    """One launch geometry of an arm."""

    name: str
    blocksize: int
    resident: Optional[int]
    blocks_per_sm: int
    dynamic_shared: int
    warm_ms: List[float] = field(default_factory=list)
    times_ms: List[float] = field(default_factory=list)
    capped: bool = False

    @property
    def key(self):
        return (self.blocksize, self.blocks_per_sm, self.dynamic_shared)

    @property
    def best_ms(self):
        if self.times_ms:
            return min(self.times_ms)
        if self.warm_ms:
            return min(self.warm_ms)
        return float("inf")


@dataclass
class Arm:
    """A built arm: its solver, compile facts and cells."""

    spec: ArmSpec
    solver: Any = None
    config_hash: str = ""
    cubin_sha: str = ""
    regs: int = 0
    frame: int = 0
    sass_bytes: int = 0
    shared_per_run: int = 0
    resolved: Dict[str, Any] = field(default_factory=dict)
    cells: Dict[str, Cell] = field(default_factory=dict)
    cell_keys: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    alias_of: Optional[str] = None
    output_check: Dict[str, Any] = field(default_factory=dict)
    compile_s: float = 0.0
    error: Optional[str] = None
    blocks: List[int] = field(default_factory=list)


def encode_settings(settings):
    """JSON-safe copy of an arm's settings."""
    out = {}
    for key, value in settings.items():
        if isinstance(value, UnrollChoice):
            out[key] = value.name
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def decode_settings(settings):
    out = {}
    for key, value in settings.items():
        if key in ALL_UNROLL_PARAMETERS:
            if isinstance(value, str):
                out[key] = UnrollChoice[value]
            elif isinstance(value, list):
                out[key] = tuple(value)
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def settings_label(settings):
    """Short label of a settings dictionary."""
    parts = []
    for key, value in settings.items():
        if isinstance(value, UnrollChoice):
            value = "1" if value is FULL else "0"
        parts.append(f"{key.removeprefix('unroll_').removesuffix('_location')}={value}")
    return " ".join(parts) or "current"


# --- building ------------------------------------------------------------


def build_system(system_name):
    return SYSTEMS[system_name]["build"]()


def build_solver(system, system_name, algo_name, spec, duration):
    kwargs = base_kwargs(system_name, algo_name, duration)
    kwargs.update(spec.settings)
    return cubie.Solver(
        system, auto_performance=spec.auto_performance, **kwargs
    )


def compiled_cubin(dispatcher) -> bytes:
    (definition,) = dispatcher.overloads.values()
    library = definition._codelibrary
    if hasattr(library, "get_cubin"):
        return bytes(library.get_cubin().code)
    return bytes(library._cubin)


def buffer_tree(root):
    """Every buffer group reachable from ``root`` through child links."""
    stack = [root]
    seen = []
    while stack:
        parent = stack.pop()
        group = buffer_registry._groups.get(parent)
        if group is None or any(parent is p for p, _ in seen):
            continue
        seen.append((parent, group))
        stack.extend(group.children.values())
    return seen


def resolved_settings(solver):
    """Unroll flags and buffer locations the built kernel uses."""
    run = solver.kernel.single_integrator
    # The step carries the auto_performance Newton-exit choice.
    flags = run._algo_step.compile_settings.unroll
    out = {
        name: "1" if getattr(flags, name) == FULL.value else "0"
        for name in UNROLL_GROUPS
    }
    for parent, group in buffer_tree(run._loop):
        for name in group.relocatable_names():
            entry = group.entries[name]
            if entry.size > 0:
                out[f"{name}_location"] = entry.location
    return out


def _compile_worker(payload):
    (
        system_name,
        algo_name,
        label,
        settings,
        auto_performance,
        n_runs,
        duration,
        cache_root,
        icache_bytes,
    ) = payload
    set_cache_root(cache_root)
    apply_icache_override(icache_bytes)
    started = time.perf_counter()
    solver = None
    try:
        system = build_system(system_name)
        spec = ArmSpec(label, decode_settings(settings), auto_performance)
        solver = build_solver(
            system, system_name, algo_name, spec, duration
        )
        sizes = system.sizes
        solver.compile(
            np.zeros((sizes.states, n_runs), dtype=PRECISION),
            np.zeros((sizes.parameters, n_runs), dtype=PRECISION),
            duration=duration,
        )
        return (
            system_name, algo_name, label, solver.kernel.config_hash,
            time.perf_counter() - started, None,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            system_name, algo_name, label, "",
            time.perf_counter() - started, repr(exc)[:300],
        )
    finally:
        if solver is not None:
            solver.close()


def icache_bytes_from(args):
    """Return the ``--icache-kib`` argument in bytes, or ``None``."""
    if args.icache_kib is None:
        return None
    return int(args.icache_kib) * 1024


def apply_icache_override(icache_bytes):
    """Enter a measured instruction-cache size for this device."""
    if icache_bytes is None or CUDA_SIMULATION:
        return
    major, minor = cuda.get_current_device().compute_capability
    INSTRUCTION_CACHE_BYTES[(int(major), int(minor))] = int(icache_bytes)


def compile_in_workers(jobs, workers, icache_bytes, log):
    """Compile ``(system, algo, ArmSpec, n_runs, duration)`` jobs ahead."""
    if workers <= 1 or not jobs:
        return
    cache_root = str(get_cache_root())
    payloads = [
        (
            system_name, algo_name, spec.label,
            encode_settings(spec.settings), spec.auto_performance,
            n_runs, duration, cache_root, icache_bytes,
        )
        for system_name, algo_name, spec, n_runs, duration in jobs
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        min(workers, len(payloads)),
        maxtasksperchild=COMPILE_TASKS_PER_CHILD,
    ) as pool:
        for result in pool.imap_unordered(_compile_worker, payloads):
            system_name, algo_name, label, digest, seconds, error = result
            if error:
                log(f"  compile FAILED {system_name}/{algo_name} {label}: "
                    f"{error}")
            else:
                log(f"  compiled {system_name}/{algo_name} {label:32s} "
                    f"{seconds:6.1f} s {digest[:12]}")


# --- cells ---------------------------------------------------------------


def cells_for(arm, blocksizes):
    """Enumerate the launch cells of a built arm."""
    kernel = arm.solver.kernel
    if blocksizes is None:
        blocksizes = (
            SHARED_LAUNCH_BLOCKSIZES
            if arm.shared_per_run > 0
            else LOCAL_LAUNCH_BLOCKSIZES
        )
    cells = {}
    keys = {}
    # The launch the kernel chooses for itself.
    kernel.resident_blocks = None
    actual, dynamic = kernel.launch_geometry(None)
    blocks = active_blocks_per_multiprocessor(kernel.kernel, actual, dynamic)
    keys["auto"] = (actual, blocks, dynamic)
    cells[f"bs{actual}x{blocks}"] = Cell(
        f"bs{actual}x{blocks}", actual, None, blocks, dynamic
    )
    for blocksize in blocksizes:
        kernel.resident_blocks = NATURAL
        actual, dynamic = kernel.launch_geometry(blocksize)
        if actual != blocksize:
            continue
        natural = active_blocks_per_multiprocessor(
            kernel.kernel, actual, dynamic
        )
        targets = [("natural", NATURAL), ("rule", None)]
        if arm.frame > 0:
            for cut in (1, 2):
                if natural - cut >= 1:
                    targets.append((f"cut{cut}", natural - cut))
            if natural - 2 > 1:
                targets.append(("one", 1))
        for role, resident in targets:
            kernel.resident_blocks = resident
            actual, dynamic = kernel.launch_geometry(blocksize)
            blocks = active_blocks_per_multiprocessor(
                kernel.kernel, actual, dynamic
            )
            key = (actual, blocks, dynamic)
            keys[f"{role}@bs{blocksize}"] = key
            if key in {c.key for c in cells.values()}:
                continue
            name = f"bs{blocksize}x{blocks}"
            cells[name] = Cell(name, actual, resident, blocks, dynamic)
    kernel.resident_blocks = None
    arm.cells = cells
    arm.cell_keys = keys


def cell_by_key(arm, key):
    for cell in arm.cells.values():
        if cell.key == key:
            return cell
    return None


# --- timing --------------------------------------------------------------


def kernel_ms(solver):
    return float(
        sum(
            event.elapsed_time_ms()
            for event in solver.kernel._cuda_events
            if event.name.startswith("kernel_chunk")
        )
    )


def solve_ms(arm, cell, d_inits, d_params, duration):
    kernel = arm.solver.kernel
    kernel.resident_blocks = cell.resident
    arm.solver.solve(
        d_inits, d_params, duration=duration, blocksize=cell.blocksize,
        on_device=True,
    )
    kernel.synchronize()
    return kernel_ms(arm.solver)


def output_check(arm, reference, inits, params, duration):
    """Host solve at the reference launch; compare with ``reference``."""
    kernel = arm.solver.kernel
    kernel.resident_blocks = NATURAL
    with contextlib.redirect_stdout(io.StringIO()):
        result = arm.solver.solve(
            inits, params, duration=duration,
            blocksize=REFERENCE_BLOCKSIZE, nan_error_trajectories=False,
        )
    codes = np.asarray(result.status_codes).ravel()
    state_last = np.array(result.state[-1])
    check = dict(
        failed=int(np.count_nonzero(codes & 0xFFFF)),
        runs=int(codes.size),
    )
    if reference is not None:
        nan_a = np.isnan(reference)
        nan_b = np.isnan(state_last)
        per_run = np.abs(
            np.nan_to_num(reference) - np.nan_to_num(state_last)
        ).max(axis=0)
        check.update(
            max_abs_diff=float(per_run.max()),
            runs_differing=int(np.count_nonzero(per_run)),
            nan_match=bool(np.array_equal(nan_a, nan_b)),
        )
    del result
    return check, state_last


def time_arms(arms, d_inits, d_params, duration, log, cap=CAP):
    """Fill the cell timings of every timed (non-alias) arm."""
    timed = [arm for arm in arms if arm.alias_of is None and arm.error is None]
    units = [(arm, cell) for arm in timed for cell in arm.cells.values()]
    if not units:
        return
    first_arm, first_cell = units[0]
    for _ in range(GPU_WARM_SOLVES):
        solve_ms(first_arm, first_cell, d_inits, d_params, duration)
    floor = float("inf")
    current = None
    for round_index in range(ROUNDS):
        ordered = units if round_index == 0 else units[::-1]
        for arm, cell in ordered:
            if cell.capped:
                continue
            if arm is not current:
                # The first solve after a kernel switch is slow.
                solve_ms(arm, cell, d_inits, d_params, duration)
                current = arm
            warm = solve_ms(arm, cell, d_inits, d_params, duration)
            cell.warm_ms.append(warm)
            if not cell.times_ms and warm > cap * floor:
                # Cap only after a second slow warm-up.
                warm = solve_ms(arm, cell, d_inits, d_params, duration)
                cell.warm_ms.append(warm)
            if not cell.times_ms and warm > cap * floor:
                cell.capped = True
                log(f"  capped {arm.spec.label} {cell.name} warm "
                    f"{min(cell.warm_ms):.2f} ms > {cap}x floor "
                    f"{floor:.2f}")
                continue
            for _ in range(TIMED_SOLVES):
                cell.times_ms.append(
                    solve_ms(arm, cell, d_inits, d_params, duration)
                )
            floor = min(floor, cell.best_ms)
            log(f"  round {round_index + 1} {arm.spec.label:32s} "
                f"{cell.name:9s} warm {warm:9.2f} timed "
                + " ".join(f"{t:9.2f}" for t in cell.times_ms[-TIMED_SOLVES:])
                + " ms")


# --- per-configuration driver -------------------------------------------


def _build_arm(arm, system, system_name, algo_name, duration, inits, params):
    """Build and compile an arm's solver on its own copy of the system."""
    # Each arm builds on its own system copy.
    solver = build_solver(
        deepcopy(system), system_name, algo_name, arm.spec, duration
    )
    arm.solver = solver
    solver.compile(inits, params, duration=duration)
    solver.kernel.launch_geometry(REFERENCE_BLOCKSIZE)
    return solver


def _close_arm(arm):
    if arm.solver is not None:
        arm.solver.close()
        arm.solver = None


def _arm_facts(arm, blocksizes, started):
    kernel = arm.solver.kernel
    arm.config_hash = kernel.config_hash
    arm.cubin_sha = hashlib.sha256(
        compiled_cubin(kernel.kernel)
    ).hexdigest()
    resources = kernel_resources(kernel.kernel)
    arm.regs = resources.registers_per_thread
    arm.frame = resources.local_bytes_per_thread
    arm.sass_bytes = resources.sass_bytes
    pad = 4 if kernel.shared_memory_needs_padding else 0
    arm.shared_per_run = int(kernel.shared_memory_bytes + pad)
    arm.resolved = resolved_settings(arm.solver)
    arm.compile_s = time.perf_counter() - started
    # Aliases keep their own cell keys.
    cells_for(arm, blocksizes)


def run_config(
    system_name,
    algo_name,
    specs: Sequence[ArmSpec],
    n_runs,
    duration,
    log,
    blocksizes=None,
    cap=CAP,
    block_arms=BLOCK_ARMS,
):
    """Build, check and time every arm; return the record row.

    Compile facts first, one arm at a time; then timing in blocks of
    ``block_arms`` solvers, the first timed arm rebuilt into each.
    """
    system = build_system(system_name)
    family, solver_kind, _ = ALGOS[algo_name]
    arms = []
    seen = {}
    inits = params = None
    for spec in specs:
        arm = Arm(spec)
        arms.append(arm)
        started = time.perf_counter()
        try:
            if inits is None:
                probe = build_solver(
                    deepcopy(system), system_name, algo_name, spec,
                    duration,
                )
                inits, params = SYSTEMS[system_name]["grid"](probe, n_runs)
                probe.close()
            _build_arm(
                arm, system, system_name, algo_name, duration, inits,
                params,
            )
            _arm_facts(arm, blocksizes, started)
            if arm.cubin_sha in seen:
                arm.alias_of = seen[arm.cubin_sha]
                arm.cells = {}
                log(f"  {spec.label:32s} alias of {arm.alias_of}")
            else:
                seen[arm.cubin_sha] = spec.label
                log(f"  {spec.label:32s} regs {arm.regs:3d} frame "
                    f"{arm.frame:5d} sass {arm.sass_bytes // 1024:5d} KiB "
                    f"shared/run {arm.shared_per_run:4d} "
                    f"cells {' '.join(arm.cells)} ({arm.compile_s:.1f} s)")
        except Exception as exc:  # noqa: BLE001
            arm.error = repr(exc)[:300]
            log(f"  {spec.label:32s} FAILED {arm.error}")
        _close_arm(arm)
    timed = [arm for arm in arms if arm.alias_of is None and arm.error is None]
    blocks = []
    if timed:
        reference_arm, others = timed[0], timed[1:]
        width = block_arms - 1
        blocks = [
            [reference_arm] + others[i:i + width]
            for i in range(0, len(others), width)
        ] or [[reference_arm]]
    d_inits = cuda.to_device(inits) if inits is not None else None
    d_params = cuda.to_device(params) if params is not None else None
    reference_state = None
    reference_by_block = []
    default_timelogger.set_verbosity("silent")
    for index, block in enumerate(blocks):
        if len(blocks) > 1:
            log(f"  block {index + 1}/{len(blocks)}: "
                f"{' | '.join(arm.spec.label for arm in block)}")
        alive = []
        for arm in block:
            try:
                _build_arm(
                    arm, system, system_name, algo_name, duration, inits,
                    params,
                )
                arm.blocks.append(index)
                alive.append(arm)
            except Exception as exc:  # noqa: BLE001
                arm.error = repr(exc)[:300]
                log(f"  {arm.spec.label:32s} FAILED {arm.error}")
                _close_arm(arm)
        for arm in alive:
            if arm.output_check:
                continue
            try:
                arm.output_check, state_last = output_check(
                    arm, reference_state, inits, params, duration
                )
                if reference_state is None:
                    reference_state = state_last
            except Exception as exc:  # noqa: BLE001
                arm.error = repr(exc)[:300]
                log(f"  {arm.spec.label:32s} FAILED in the output "
                    f"check {arm.error}")
                _close_arm(arm)
        alive = [arm for arm in alive if arm.solver is not None]
        reference = blocks[0][0]
        before = {name: len(cell.times_ms)
                  for name, cell in reference.cells.items()}
        time_arms(alive, d_inits, d_params, duration, log, cap=cap)
        reference_by_block.append({
            name: min(cell.times_ms[before[name]:])
            for name, cell in reference.cells.items()
            if len(cell.times_ms) > before[name]
        })
        for arm in alive:
            _close_arm(arm)
    hardware = device_hardware()
    row = dict(
        system=system_name,
        algo=algo_name,
        family=family,
        solver=solver_kind,
        n_states=SYSTEMS[system_name]["n_states"],
        n_runs=int(n_runs),
        duration=float(duration),
        backend="mlir" if IS_MLIR else "numba-cuda",
        cubie=cubie.__version__,
        hardware=dict(
            compute_capability=list(hardware.compute_capability),
            multiprocessor_count=hardware.multiprocessor_count,
            l2_cache_bytes=hardware.l2_cache_bytes,
            shared_memory_per_multiprocessor=(
                hardware.shared_memory_per_multiprocessor
            ),
            reserved_shared_memory_per_block=(
                hardware.reserved_shared_memory_per_block
            ),
            max_dynamic_shared_memory_per_block=(
                hardware.max_dynamic_shared_memory_per_block
            ),
            instruction_cache_bytes=hardware.instruction_cache_bytes,
            name=str(cuda.get_current_device().name),
        ),
        protocol=dict(
            rounds=ROUNDS, timed_solves=TIMED_SOLVES, cap=cap,
            block_arms=block_arms,
        ),
        reference_by_block=reference_by_block,
        arms=[],
    )
    for arm in arms:
        entry = dict(
            label=arm.spec.label,
            settings=encode_settings(arm.spec.settings),
            auto_performance=arm.spec.auto_performance,
            in_optimize=arm.spec.in_optimize,
            alias_of=arm.alias_of,
            error=arm.error,
            blocks=arm.blocks,
            config_hash=arm.config_hash,
            cubin_sha=arm.cubin_sha,
            regs=arm.regs,
            frame=arm.frame,
            sass_bytes=arm.sass_bytes,
            shared_per_run=arm.shared_per_run,
            compile_s=round(arm.compile_s, 2),
            resolved=arm.resolved,
            output_check=arm.output_check,
            cell_keys={k: list(v) for k, v in arm.cell_keys.items()},
            cells={
                cell.name: dict(
                    blocksize=cell.blocksize,
                    resident=cell.resident,
                    blocks_per_sm=cell.blocks_per_sm,
                    dynamic_shared=cell.dynamic_shared,
                    warm_ms=cell.warm_ms,
                    times_ms=cell.times_ms,
                    capped=cell.capped,
                )
                for cell in arm.cells.values()
            },
        )
        row["arms"].append(entry)
        if arm.alias_of is None and arm.error is None:
            for cell in arm.cells.values():
                log(f"{system_name}/{algo_name} {arm.spec.label:32s} "
                    f"{cell.name:10s} blk/SM {cell.blocks_per_sm:2d} "
                    f"best {cell.best_ms:9.3f} ms"
                    f"{' capped' if cell.capped else ''}")
    return row


# --- records ---------------------------------------------------------------


def run_signature(system_name, algo_name, n_runs, duration, specs,
                  blocksizes, icache_bytes, cap, block_arms):
    """Identity of one configuration run for resuming a records file."""
    return dict(
        system=system_name,
        algo=algo_name,
        n_runs=int(n_runs),
        duration=float(duration),
        arms=[spec.label for spec in specs],
        blocksizes=None if blocksizes is None else list(blocksizes),
        icache_bytes=icache_bytes,
        cap=float(cap),
        block_arms=int(block_arms),
    )


def done_signatures(path: Path):
    """Run signatures of the rows already in ``path``."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "run_signature" in row:
                out.append(row["run_signature"])
    return out


def load_rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_row(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


# --- scoring helpers -----------------------------------------------------


def row_arms(row):
    """Arms of a row keyed by label, aliases resolved to their kernel."""
    by_label = {arm["label"]: arm for arm in row["arms"]}
    resolved = {}
    for label, arm in by_label.items():
        target = arm
        while target.get("alias_of"):
            target = by_label[target["alias_of"]]
        resolved[label] = (arm, target)
    return resolved


def cell_time(target, key):
    """Best milliseconds of the cell with geometry ``key`` on ``target``."""
    for cell in target["cells"].values():
        if [cell["blocksize"], cell["blocks_per_sm"],
                cell["dynamic_shared"]] == list(key):
            if cell["times_ms"]:
                return min(cell["times_ms"])
            if cell["warm_ms"]:
                return min(cell["warm_ms"])
    return None


def policy_time(row, label, role_blocksizes):
    """Time of arm ``label`` at the first available ``role@bsN`` cell."""
    arms = row_arms(row)
    if label not in arms:
        return None, None
    arm, target = arms[label]
    if target.get("error"):
        return None, None
    for role, blocksize in role_blocksizes:
        # ``blocksize`` None names the launch the kernel chose itself.
        name = role if blocksize is None else f"{role}@bs{blocksize}"
        key = arm["cell_keys"].get(name)
        if key is None:
            continue
        ms = cell_time(target, key)
        if ms is not None:
            return ms, f"{label}@{name}"
    return None, None


def best_overall(row, labels=None):
    """Fastest (ms, label, cell) over every timed cell of the row."""
    best = (float("inf"), None, None)
    for arm in row["arms"]:
        if arm.get("alias_of") or arm.get("error"):
            continue
        if labels is not None and arm["label"] not in labels:
            continue
        for name, cell in arm["cells"].items():
            if not cell["times_ms"]:
                continue
            ms = min(cell["times_ms"])
            if ms < best[0]:
                best = (ms, arm["label"], name)
    return best


def optimize_choice(row):
    """Fastest cell ``Solver.optimize`` would have timed on this row."""
    best = (float("inf"), None)
    arms = row_arms(row)
    for label, (arm, target) in arms.items():
        if not arm["in_optimize"] or target.get("error"):
            continue
        blocksizes = (
            SHARED_LAUNCH_BLOCKSIZES
            if target["shared_per_run"] > 0
            else LOCAL_LAUNCH_BLOCKSIZES
        )
        for blocksize in blocksizes:
            for role in ("rule", "cut1", "cut2"):
                key = arm["cell_keys"].get(f"{role}@bs{blocksize}")
                if key is None:
                    continue
                ms = cell_time(target, key)
                if ms is not None and ms < best[0]:
                    best = (ms, f"{label}@{role}@bs{blocksize}")
    return best


def format_loss(ms, best):
    if ms is None or best is None or best == float("inf"):
        return "   n/a "
    return f"{100 * (ms / best - 1):6.1f}%"


# --- factorials ------------------------------------------------------------

ERK_UNROLL_GROUPS = (
    "unroll_stage",
    "unroll_step_element",
    "unroll_accumulator",
    "unroll_norms",
    "unroll_other_small",
)
"""Unroll groups with sites in an explicit Runge-Kutta kernel."""

ERK_BUFFERS = (
    "stage_rhs",
    "stage_accumulator",
    "state",
    "proposed_state",
    "error",
)
"""Relocatable buffers of an adaptive explicit Runge-Kutta kernel."""

PUBLICATION_LORENZ = dict(atol=1e-5, rtol=1e-5, dt=2.0 ** -10)
"""Tolerances and first step of the GPUODEBenchmarks Lorenz problem."""


def factorial_arms(groups, buffers, extra_settings=None):
    """Every combination of rolled groups and shared buffers.

    Labels: ``u<bits>`` (1 = full) and ``p<bits>`` (1 = shared); the
    all-full, all-local arm comes first.
    """
    extra = dict(extra_settings or {})
    arms = []
    unroll_levels = list(product((FULL, ROLLED), repeat=len(groups)))
    place_levels = list(product(("local", "shared"), repeat=len(buffers)))
    for unroll in unroll_levels:
        for placement in place_levels:
            settings = dict(extra)
            label = []
            if groups:
                settings.update(dict(zip(groups, unroll)))
                label.append(
                    "u" + "".join("1" if u is FULL else "0" for u in unroll)
                )
            if buffers:
                settings.update(
                    {f"{name}_location": loc
                     for name, loc in zip(buffers, placement)}
                )
                label.append(
                    "p" + "".join("1" if p == "shared" else "0"
                                  for p in placement)
                )
            arms.append(ArmSpec(
                " ".join(label), settings, auto_performance=False
            ))
    return arms


def score_factorial(row, groups, buffers, within=0.05):
    """Rank every timed cell and count each factor's paired effect."""
    arms = row_arms(row)
    reference_label = row["arms"][0]["label"]
    ref_arm, ref_target = arms[reference_label]
    ref_key = ref_arm["cell_keys"].get(f"natural@bs{REFERENCE_BLOCKSIZE}")
    ref_ms = cell_time(ref_target, ref_key) if ref_key else None
    rows = []
    for label, (arm, target) in arms.items():
        if target.get("error") or arm.get("alias_of"):
            continue
        for name, cell in target["cells"].items():
            if not cell["times_ms"] and not cell["warm_ms"]:
                continue
            ms = min(cell["times_ms"] or cell["warm_ms"])
            rows.append((ms, label, name, cell["blocks_per_sm"],
                         cell["capped"], arm.get("alias_of")))
    rows.sort()
    best_ms = rows[0][0] if rows else float("inf")
    print(f"{row['system']}/{row['algo']} n_runs {row['n_runs']} duration "
          f"{row['duration']} reference {reference_label}@natural@bs"
          f"{REFERENCE_BLOCKSIZE} = {ref_ms} ms; {len(rows)} timed cells")
    header = (f"{'arm':22s} {'cell':10s} {'blk/SM':>6s} {'ms':>9s} "
              f"{'/best':>7s} {'/ref':>7s}  note")
    print(header)
    print("-" * len(header))
    for ms, label, cell, blocks, capped, alias in rows:
        note = "capped" if capped else ""
        ratio_ref = "" if ref_ms is None else f"{ms / ref_ms:7.3f}"
        print(f"{label:22s} {cell:10s} {blocks:6d} {ms:9.3f} "
              f"{ms / best_ms:7.3f} {ratio_ref:>7s}  {note}")
    aliases = [(a["label"], a["alias_of"]) for a in row["arms"]
               if a.get("alias_of")]
    if aliases:
        print(f"aliases (identical cubin, not timed): {len(aliases)}")
        for label, target in aliases:
            print(f"  {label:22s} = {target}")
    within_best = sum(1 for r in rows if r[0] <= (1 + within) * best_ms)
    print(f"cells within {int(100 * within)}% of the best: {within_best} "
          f"of {len(rows)}")
    # Paired effect of each factor at every setting of the others.
    factors = [(g, "u", i) for i, g in enumerate(groups)]
    factors += [(b, "p", i) for i, b in enumerate(buffers)]
    # Aliases read their kernel's cells.
    by_label_cell = {}
    for label, (arm, target) in arms.items():
        if target.get("error"):
            continue
        for name, cell in target["cells"].items():
            if cell["times_ms"] or cell["warm_ms"]:
                by_label_cell[(label, name)] = min(
                    cell["times_ms"] or cell["warm_ms"]
                )
    print(f"{'factor':32s} {'on wins':>8s} {'off wins':>9s} "
          f"{'within':>7s}  (paired cells, > {int(100 * within)}%; "
          "on = full / shared)")
    for name, axis, index in factors:
        wins = losses = ties = 0
        for (label, cell), on_ms in by_label_cell.items():
            bits = {part[0]: part[1:] for part in label.split()}
            if bits.get(axis, "")[index:index + 1] != "1":
                continue
            flipped = list(bits[axis])
            flipped[index] = "0"
            partner = label.replace(
                axis + bits[axis], axis + "".join(flipped)
            )
            off_ms = by_label_cell.get((partner, cell))
            if off_ms is None:
                continue
            if on_ms < off_ms / (1 + within):
                wins += 1
            elif off_ms < on_ms / (1 + within):
                losses += 1
            else:
                ties += 1
        shown = f"{name} ({'full' if axis == 'u' else 'shared'})"
        print(f"{shown:32s} {wins:8d} {losses:9d} {ties:7d}")


# --- command line ----------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--out", type=Path, required=True,
        help="records file (JSON lines); a configuration recorded with "
        "the same run count, duration, arms, block sizes, icache, cap "
        "and block-arms settings is skipped on a re-run",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="parallel compile processes ahead of the timing pass",
    )
    parser.add_argument(
        "--icache-kib", type=int, default=None,
        help="measured instruction-cache capacity of this device in "
        "KiB (benchmarks/icache_probe.py); enters the auto_performance "
        "table for an unmeasured compute capability",
    )
    parser.add_argument(
        "--cap", type=float, default=CAP,
        help="warm-up multiple of the running floor above which a "
        "cell is capped",
    )
    parser.add_argument(
        "--block-arms", type=_at_least_two, default=BLOCK_ARMS,
        help="solvers alive at once during timing, at least 2 (each "
        "holds its batch buffers; the reference arm is re-timed in "
        "every block)",
    )
    parser.add_argument(
        "--blocksizes", default=None,
        help="comma-separated block sizes for every arm (default: the "
        "launch_candidates sets)",
    )
    parser.add_argument(
        "--duration-scale", type=float, default=1.0,
        help="multiply every configuration's duration (shorter solves "
        "on a faster card; keep each solve above 20 ms)",
    )
    parser.add_argument(
        "--log", type=Path, default=None,
        help="also append progress lines to this file",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="score the records file and exit",
    )


def _at_least_two(text):
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError("--block-arms must be at least 2")
    return value


def parse_blocksizes(text):
    if text is None:
        return None
    return tuple(int(value) for value in text.split(","))


def check_device(log):
    if CUDA_SIMULATION:
        raise SystemExit("run on a real GPU, not under NUMBA_ENABLE_CUDASIM")
    hardware = device_hardware()
    capability = hardware.compute_capability
    measured = capability in INSTRUCTION_CACHE_BYTES
    icache = f"{hardware.instruction_cache_bytes // 1024} KiB"
    if not measured:
        icache += " (UNMEASURED fallback: pass --icache-kib)"
    log(f"device {cuda.get_current_device().name} cc {capability} "
        f"SMs {hardware.multiprocessor_count} L2 "
        f"{hardware.l2_cache_bytes // 2**20} MiB shared/SM "
        f"{hardware.shared_memory_per_multiprocessor // 1024} KiB "
        f"opt-in {hardware.max_dynamic_shared_memory_per_block // 1024} "
        f"KiB icache {icache}")
    log(f"backend {'mlir' if IS_MLIR else 'numba-cuda'} cubie "
        f"{cubie.__version__} cache {get_cache_root()}")


def make_logger(path: Optional[Path]):
    handle = path.open("a", encoding="utf-8") if path else None

    def log(message):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return log


def run_jobs(configs, arms_for, args, log, duration_override=None):
    """Compile ahead, then time each ``(system, algo, n_runs)`` config.

    ``duration_override`` replaces the bank durations;
    ``args.duration_scale`` multiplies either.
    """
    finished = done_signatures(args.out)
    blocksizes = parse_blocksizes(args.blocksizes)
    icache_bytes = icache_bytes_from(args)
    jobs = []
    plans = []
    for system_name, algo_name, n_runs in configs:
        settled = (
            duration_for(system_name, algo_name)
            if duration_override is None
            else duration_override
        )
        duration = settled * args.duration_scale
        specs = arms_for(system_name, algo_name)
        signature = run_signature(
            system_name, algo_name, n_runs, duration, specs, blocksizes,
            icache_bytes, args.cap, args.block_arms,
        )
        if signature in finished:
            continue
        plans.append(
            (system_name, algo_name, n_runs, duration, specs, signature)
        )
        jobs.extend(
            (system_name, algo_name, spec, n_runs, duration)
            for spec in specs
        )
    if not plans:
        log("nothing to do; every configuration is in the records file")
        return
    log(f"{len(plans)} configurations, {len(jobs)} kernels to compile")
    started = time.perf_counter()
    compile_in_workers(jobs, args.workers, icache_bytes, log)
    log(f"compiles done in {time.perf_counter() - started:.0f} s")
    for system_name, algo_name, n_runs, duration, specs, signature in plans:
        log(f"=== {system_name}/{algo_name} n_runs {n_runs} duration "
            f"{duration} arms {len(specs)}")
        started = time.perf_counter()
        row = run_config(
            system_name, algo_name, specs, n_runs, duration, log,
            blocksizes=blocksizes, cap=args.cap,
            block_arms=args.block_arms,
        )
        row["run_signature"] = signature
        row["wall_s"] = round(time.perf_counter() - started, 1)
        append_row(args.out, row)
        log(f"=== {system_name}/{algo_name} done in {row['wall_s']} s")


__all__ = [
    "ALGOS", "SYSTEMS", "Arm", "ArmSpec", "Cell", "FULL", "ROLLED",
    "UNROLL_GROUPS", "add_common_arguments", "apply_icache_override",
    "best_overall", "cell_time", "check_device", "duration_for",
    "format_loss", "icache_bytes_from", "load_rows", "make_logger",
    "optimize_choice", "policy_time", "row_arms", "run_jobs",
    "settings_label",
]
