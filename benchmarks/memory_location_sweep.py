#!/usr/bin/env python
"""Overnight sweep: buffer memory locations across sizes and algorithms.

Measures kernel runtime for placement variants of every registered
buffer group (loop arrays, algorithm work buffers, nonlinear/linear
solver vectors) over a grid of system sizes (states, parameters,
drivers, observables, baked constants) and algorithm families. The
results feed the size-based location heuristics tracked by issue
#329.

Each configuration runs in a fresh subprocess so the kernel is
compiled for exactly that placement. Results append to a JSONL file;
re-running skips configurations already recorded, so the sweep is
resumable.

Usage::

    python benchmarks/memory_location_sweep.py            # full sweep
    python benchmarks/memory_location_sweep.py --phase core
    python benchmarks/memory_location_sweep.py --single '<config json>'
    python benchmarks/memory_location_sweep.py --fit      # derive rules

The single-config mode is internal: the driver invokes it in a
subprocess per configuration.

``--fit`` derives the :class:`MemoryThresholds` constants in
``cubie.integrators.memory_heuristics`` from the recorded results
and prints a paste-ready ``THRESHOLDS_BY_ARCH`` entry plus a
validation report, so calibrating a new card is: run the sweep
there, run ``--fit``, and commit the emitted entry keyed by the
card's architecture code. Records written by older sweeps lack the
declared-size fields, so ``--fit`` reconstructs them host-side (no
kernel compilation) and caches them beside the results file.

Timing follows benchmarks/lorenz_mean_runtime.py: kernel-only CUDA
event times (per-chunk ``kernel_chunk_i`` events), one warm-up solve
to absorb JIT compilation, then ``repeats`` timed solves.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory_location_sweep_results.jsonl",
)
DECLARED_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory_location_sweep_declared_cache.jsonl",
)
REPEATS = 5
NRUNS = 32768
BLOCKSIZE = 256
DURATION = 0.05
DT = 1e-3

# Buffer-location kwargs per placement group. Loop groups apply to
# every algorithm; algorithm and solver groups are keyed by family.
LOOP_GROUPS = {
    "state": ["state_location", "proposed_state_location"],
    "params": ["parameters_location"],
    "drivers": ["drivers_location", "proposed_drivers_location"],
    "obs": ["observables_location", "proposed_observables_location"],
}
ALGO_GROUPS = {
    "euler": [],
    "tsit5": ["stage_rhs_location", "stage_accumulator_location"],
    "dirk": [
        "stage_increment_location",
        "stage_base_location",
        "accumulator_location",
        "stage_rhs_location",
    ],
    "firk": [
        "stage_increment_location",
        "stage_driver_stack_location",
        "stage_state_location",
    ],
    "rosenbrock": [
        "stage_rhs_location",
        "stage_store_location",
        "cached_auxiliaries_location",
    ],
    "backwards_euler": ["increment_cache_location"],
    "crank_nicolson": ["dxdt_location"],
}
SOLVER_GROUPS = {
    "euler": [],
    "tsit5": [],
    "dirk": [
        "delta_location",
        "residual_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
    "firk": [
        "delta_location",
        "residual_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
    "rosenbrock": ["preconditioned_vec_location", "temp_location"],
    "backwards_euler": [
        "delta_location",
        "residual_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
    "crank_nicolson": [
        "delta_location",
        "residual_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
}
IMPLICIT = {"dirk", "firk", "rosenbrock", "backwards_euler",
            "crank_nicolson"}



def _create_folded_system(*args, constants, parameters=None, **kwargs):
    """Create a system declaring every named value, then fold some."""
    from cubie import create_ODE_system

    merged = {**(parameters or {}), **constants}
    system = create_ODE_system(*args, parameters=merged, **kwargs)
    live = system.compile_settings.parameter_values
    system.set_categories(
        parameters={
            name: value
            for name, value in live.items()
            if name not in constants
        },
        constants=dict(constants),
    )
    return system

def variant_kwargs(algorithm, variant, n_drivers, n_observables):
    """Return the ``*_location`` kwargs for a named placement variant.

    Returns None when the variant does not apply to this
    configuration (e.g. solver placement on an explicit algorithm).
    """
    groups = []
    if variant == "local":
        return {}
    elif variant == "state_shared":
        groups = [LOOP_GROUPS["state"]]
    elif variant == "params_shared":
        groups = [LOOP_GROUPS["params"]]
    elif variant == "drivers_shared":
        if n_drivers == 0:
            return None
        groups = [LOOP_GROUPS["drivers"]]
    elif variant == "obs_shared":
        if n_observables == 0:
            return None
        groups = [LOOP_GROUPS["obs"]]
    elif variant == "state_params_shared":
        groups = [LOOP_GROUPS["state"], LOOP_GROUPS["params"]]
    elif variant == "algo_shared":
        if not ALGO_GROUPS[algorithm]:
            return None
        groups = [ALGO_GROUPS[algorithm]]
    elif variant == "solver_shared":
        if algorithm not in IMPLICIT:
            return None
        groups = [SOLVER_GROUPS[algorithm]]
    elif variant == "all_shared":
        groups = [
            LOOP_GROUPS["state"],
            LOOP_GROUPS["params"],
            ALGO_GROUPS[algorithm],
            SOLVER_GROUPS[algorithm],
        ]
        if n_drivers:
            groups.append(LOOP_GROUPS["drivers"])
        if n_observables:
            groups.append(LOOP_GROUPS["obs"])
    else:
        raise ValueError(f"unknown variant {variant}")
    return {key: "shared" for group in groups for key in group}


def make_system(cfg):
    """Build the synthetic benchmark system for a configuration.

    Nonlinear nearest-neighbour chain: every state couples to its
    ring neighbours with one nonlinear term, parameters cycle over
    the equations, drivers force the first equations additively, and
    observables are pairwise products. Baked constants are unique
    literals so codegen cannot fold them together.
    """
    n = cfg["n_states"]
    n_params = cfg["n_params"]
    n_drivers = cfg["n_drivers"]
    n_obs = cfg["n_observables"]
    consts_per_eq = cfg["consts_per_eq"]
    precision = np.float32 if cfg["precision"] == "f32" else np.float64

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
        if n_params:
            pj = i % n_params
            terms.append(f"+ 0.05*p{pj}*x{i}*x{ip1}")
        if n_drivers:
            dj = i % n_drivers
            terms.append(f"+ 0.1*d{dj}")
        eqs.append(f"dx{i} = " + " ".join(terms))
    for j in range(n_obs):
        a = j % n
        b = (j * 7 + 1) % n
        eqs.append(f"o{j} = x{a}*x{b}")

    system = _create_folded_system(
        dxdt=eqs,
        states={f"x{i}": 0.5 for i in range(n)},
        parameters={f"p{j}": 1.0 for j in range(n_params)},
        constants=constants,
        drivers=[f"d{j}" for j in range(n_drivers)] or None,
        observables=[f"o{j}" for j in range(n_obs)] or None,
        precision=precision,
        name=(
            f"sweep_{n}s_{n_params}p_{n_drivers}d_{n_obs}o_"
            f"{consts_per_eq}c_{cfg['precision']}"
        ),
    )
    return system, precision


def baseline_kwargs(cfg):
    """Return the all-local baseline Solver kwargs for a config.

    auto_memory is off on both sides of every measurement: the
    sweep measures pure placements, and the heuristics under
    calibration must never leak into their own baseline.
    """
    kwargs = dict(
        dt=DT,
        step_controller="fixed",
        save_every=DURATION,
        output_types=["state"],
        time_logging_level="default",
        algorithm=cfg["algorithm"],
        auto_memory=False,
    )
    if cfg["algorithm"] in IMPLICIT:
        kwargs["krylov_max_iters"] = 100
    return kwargs


def declared_record(solver):
    """Return the registry-declared sizes the placement gates use.

    These are the exact quantities ``auto_memory_locations``
    measures at construction, captured through the same helper so
    the fit stage and the resolver can never disagree.
    """
    from cubie.cuda_simsafe import compute_capability_code
    from cubie.integrators.memory_heuristics import declared_sizes

    sizes = declared_sizes(solver.kernel.single_integrator)
    return dict(
        itemsize=sizes.itemsize,
        is_implicit=sizes.is_implicit,
        footprint_bytes=sizes.footprint_bytes,
        state_pair_bytes=sizes.state_pair_bytes,
        work_group_bytes=sizes.work_group_bytes,
        params_bytes=sizes.params_bytes,
        arch=compute_capability_code(),
    )


def solver_stats(solver):
    """Return launch-geometry and register stats for a built solver."""
    bsk = solver.kernel
    pad = 4 if bsk.shared_memory_needs_padding else 0
    bytes_per_run = bsk.shared_memory_bytes + pad
    smem = int(bytes_per_run * min(NRUNS, BLOCKSIZE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eff_blocksize, smem = bsk.limit_blocksize(
            BLOCKSIZE, smem, bytes_per_run, NRUNS
        )
    regs = lmem = None
    try:
        kernel = bsk.kernel
        regs = list(kernel.get_regs_per_thread().values())[0]
        lmem = list(kernel.get_local_mem_per_thread().values())[0]
    except Exception:
        pass
    return dict(
        regs_per_thread=regs,
        local_mem_per_thread=lmem,
        shared_bytes_per_run=int(bytes_per_run),
        eff_blocksize=int(eff_blocksize),
        dynamic_shared_per_block=int(smem),
    )


def timed_solve(solver, y0, p, solve_kwargs):
    """Run one solve and return its kernel CUDA-event total (ms)."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(y0, p, **solve_kwargs)
    events = solver.kernel._cuda_events
    ms = sum(
        e.elapsed_time_ms()
        for e in events
        if e.name.startswith("kernel_chunk")
    )
    return ms, result


def run_single(cfg):
    """Run one paired baseline/variant configuration; print JSON.

    The benchmark GPU also drives a desktop, and background load can
    inflate whole windows of measurements uniformly, which no
    per-config variance filter can detect. Timing is therefore
    paired: the all-local baseline solver and the variant solver
    alternate solves within the same process, and the reported ratio
    compares minima over interleaved samples, so slow drifts in
    outside load cancel instead of masquerading as placement
    effects.
    """
    from cubie.batchsolving.solver import Solver
    from cubie.time_logger import default_timelogger

    default_timelogger.set_verbosity("default")
    system, precision = make_system(cfg)

    loc_kwargs = variant_kwargs(
        cfg["algorithm"], cfg["variant"], cfg["n_drivers"],
        cfg["n_observables"],
    )
    if loc_kwargs is None or cfg["variant"] == "local":
        print(json.dumps({"skip": True}))
        return

    base_kwargs = baseline_kwargs(cfg)

    n = cfg["n_states"]
    y0 = {
        f"x{i}": np.full(NRUNS, 0.5, dtype=precision) for i in range(n)
    }
    p = {"p0": np.full(NRUNS, 1.0, dtype=precision)}
    solve_kwargs = dict(
        duration=DURATION, grid_type="verbatim", blocksize=BLOCKSIZE,
    )
    if cfg["n_drivers"]:
        t_samples = np.linspace(
            0.0, DURATION, 16, dtype=precision
        )
        drivers = {
            f"d{j}": np.sin(
                2 * np.pi * (j + 1) * t_samples / DURATION
            ).astype(precision)
            for j in range(cfg["n_drivers"])
        }
        drivers["driver_sample_period"] = precision(DURATION / 15)
        drivers["wrap"] = False
        solve_kwargs["drivers"] = drivers

    caught = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        solver_local = Solver(system, **base_kwargs)
        # Capture declared sizes at construction: the stage where
        # auto_memory_locations measures them. Builds and driver
        # updates can enlarge child registrations afterwards.
        declared = declared_record(solver_local)
        solver_variant = Solver(system, **base_kwargs, **loc_kwargs)
        # Warm-up both (JIT compilation / cache load) and capture
        # outputs for a placement-invariance check: buffer location
        # must never change results.
        _, result_local = timed_solve(
            solver_local, y0, p, solve_kwargs
        )
        _, result_variant = timed_solve(
            solver_variant, y0, p, solve_kwargs
        )
        caught = sorted({str(w.message)[:80] for w in wlist})
    compile_s = time.perf_counter() - t0

    out_local = np.asarray(result_local.time_domain_array)
    out_variant = np.asarray(result_variant.time_domain_array)
    if out_local.shape == out_variant.shape:
        diff = float(
            np.nanmax(np.abs(out_local - out_variant))
        )
    else:
        diff = float("nan")

    local_ms = []
    variant_ms = []
    while True:
        for _ in range(REPEATS):
            ms, _ = timed_solve(solver_local, y0, p, solve_kwargs)
            local_ms.append(ms)
            ms, _ = timed_solve(solver_variant, y0, p, solve_kwargs)
            variant_ms.append(ms)
        pair_ratios = np.asarray(variant_ms) / np.asarray(local_ms)
        if len(local_ms) >= 3 * REPEATS:
            break
        if pair_ratios.std(ddof=1) <= 0.05 * pair_ratios.mean():
            break

    local_arr = np.asarray(local_ms)
    variant_arr = np.asarray(variant_ms)
    record = dict(cfg)
    record.update(
        ratio_min=float(variant_arr.min() / local_arr.min()),
        ratio_median_pairwise=float(
            np.median(variant_arr / local_arr)
        ),
        local_ms_min=float(local_arr.min()),
        variant_ms_min=float(variant_arr.min()),
        local_ms=list(map(float, local_arr)),
        variant_ms=list(map(float, variant_arr)),
        compile_s=round(compile_s, 2),
        local_stats=solver_stats(solver_local),
        variant_stats=solver_stats(solver_variant),
        declared=declared,
        out_maxdiff=diff,
        warnings=caught,
    )
    print(json.dumps(record))


def core_matrix():
    """Yield the core sweep: sizes x algorithms x placement variants."""
    variants = [
        "local",
        "state_shared",
        "params_shared",
        "state_params_shared",
        "algo_shared",
        "solver_shared",
        "all_shared",
    ]
    # rosenbrock is excluded: auxiliary-cache planning
    # (_search_group_combinations) does not terminate for chain
    # systems at n_states >= 16, so its kernels cannot be built.
    algorithms = [
        "euler",
        "tsit5",
        "dirk",
        "firk",
        "backwards_euler",
    ]
    for n in [4, 16, 64, 128, 256]:
        for algorithm in algorithms:
            if algorithm == "firk" and n > 64:
                continue
            for variant in variants:
                yield dict(
                    phase="core",
                    algorithm=algorithm,
                    n_states=n,
                    n_params=2,
                    n_drivers=0,
                    n_observables=0,
                    consts_per_eq=3,
                    precision="f32",
                    variant=variant,
                )


def dims_matrix():
    """Yield the size-dimension sweep on two representative algorithms."""
    for algorithm in ["tsit5", "backwards_euler"]:
        for n in [16, 64, 128]:
            for n_params in [16, 64, 128]:
                for variant in ["local", "params_shared"]:
                    yield dict(
                        phase="dims",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=n_params,
                        n_drivers=0,
                        n_observables=0,
                        consts_per_eq=3,
                        precision="f32",
                        variant=variant,
                    )
            for n_drivers in [2, 8]:
                for variant in ["local", "drivers_shared"]:
                    yield dict(
                        phase="dims",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=2,
                        n_drivers=n_drivers,
                        n_observables=0,
                        consts_per_eq=3,
                        precision="f32",
                        variant=variant,
                    )
            for n_obs in [8, 32]:
                for variant in ["local", "obs_shared"]:
                    yield dict(
                        phase="dims",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=2,
                        n_drivers=0,
                        n_observables=n_obs,
                        consts_per_eq=3,
                        precision="f32",
                        variant=variant,
                    )


def consts_matrix():
    """Yield the baked-constants spot check (register pressure)."""
    for algorithm in ["tsit5", "backwards_euler"]:
        for n in [16, 64]:
            for consts_per_eq in [1, 8]:
                for variant in ["local", "state_params_shared"]:
                    yield dict(
                        phase="consts",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=2,
                        n_drivers=0,
                        n_observables=0,
                        consts_per_eq=consts_per_eq,
                        precision="f32",
                        variant=variant,
                    )


def f64_matrix():
    """Yield float64 spot checks."""
    for algorithm in ["euler", "tsit5", "backwards_euler"]:
        for n in [16, 128]:
            for variant in [
                "local",
                "state_shared",
                "params_shared",
                "all_shared",
            ]:
                yield dict(
                    phase="f64",
                    algorithm=algorithm,
                    n_states=n,
                    n_params=2,
                    n_drivers=0,
                    n_observables=0,
                    consts_per_eq=3,
                    precision="f64",
                    variant=variant,
                )


PHASES = {
    "core": core_matrix,
    "dims": dims_matrix,
    "consts": consts_matrix,
    "f64": f64_matrix,
}


def config_key(cfg):
    """Return the deduplication key for a configuration."""
    fields = [
        "algorithm", "n_states", "n_params", "n_drivers",
        "n_observables", "consts_per_eq", "precision", "variant",
    ]
    return tuple(cfg[f] for f in fields)


def drive(phases):
    """Run every configuration not yet in the results file."""
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "ratio_min" in rec or rec.get("error"):
                    done.add(config_key(rec))

    configs = [
        cfg for phase in phases for cfg in PHASES[phase]()
        if cfg["variant"] != "local"
    ]
    todo = [c for c in configs if config_key(c) not in done]
    print(f"{len(configs)} configs, {len(todo)} to run")

    env = dict(os.environ)
    for idx, cfg in enumerate(todo):
        label = "/".join(str(v) for v in config_key(cfg))
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--single",
                 json.dumps(cfg)],
                capture_output=True, text=True, timeout=900, env=env,
            )
        except subprocess.TimeoutExpired:
            record = dict(cfg)
            record["error"] = "timeout after 900 s"
            with open(RESULTS_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"[{idx + 1}/{len(todo)}] {label}: TIMEOUT")
            continue
        elapsed = time.perf_counter() - t0
        record = None
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(candidate, dict):
                record = candidate
                break
        if record is None:
            record = dict(cfg)
            record["error"] = (proc.stdout + proc.stderr)[-2000:]
        if record.get("skip"):
            print(f"[{idx + 1}/{len(todo)}] {label}: skipped")
            continue
        with open(RESULTS_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if "ratio_min" in record:
            vstats = record["variant_stats"]
            print(
                f"[{idx + 1}/{len(todo)}] {label}: "
                f"ratio {record['ratio_min']:.3f} "
                f"(local {record['local_ms_min']:.2f} ms, "
                f"variant {record['variant_ms_min']:.2f} ms, "
                f"bs {vstats['eff_blocksize']}, "
                f"sh {vstats['shared_bytes_per_run']} B/run, "
                f"maxdiff {record['out_maxdiff']:.2e}) "
                f"[{elapsed:.0f} s]"
            )
        else:
            print(f"[{idx + 1}/{len(todo)}] {label}: ERROR")
    print("SWEEP DONE")


# --- Threshold fitting ------------------------------------------------
#
# The fit stage captures the reasoning that turned raw sweep records
# into the MemoryThresholds constants, so recalibrating on another
# card is mechanical. Classification: a paired ratio <= WIN_RATIO is
# a win, >= LOSS_RATIO a loss, anything between is noise (the paired
# protocol's observed noise floor is ~5%).
#
# Size gates (state pair, work group, explicit work, params) are
# fitted per rule from that rule's own variant records. A measured
# size qualifies when its wins outnumber its losses across
# configurations - single-cell losses are outvoted when the size
# wins consistently elsewhere - and the gate is the qualifying
# boundary with every measured size inside it also qualifying, so
# one isolated far-out win can never drag the gate across a band of
# losses. Gates round to the GRID unless rounding would cross a
# disqualified measured size.
#
# The two footprint gates (heavy-spill and the explicit spill
# floor) couple every rule, so they are chosen by scanning the
# grid: for each candidate pair the size gates are refitted and
# every measured configuration is replayed through
# placement_candidates. The scan keeps the pair that fires on the
# fewest measured losses (zero, on the calibration dataset), then
# misses the fewest measured wins, then is most conservative.
# Isolated sub-spill wins (e.g. a 0.87 on a 236-byte footprint)
# are deliberately sacrificed by this objective: no footprint gate
# can capture them without also firing on the losses measured
# between them and the spilled region.
#
# A final validation pass reports every configuration's decision
# against its measured ratios, including placements that fire on
# territory the sweep never measured.

WIN_RATIO = 0.95
LOSS_RATIO = 1.05
GRID = 512

NEVER_AT_MOST = 0
NEVER_AT_LEAST = 1 << 62

DECLARED_KEY_FIELDS = [
    "algorithm", "n_states", "n_params", "n_drivers",
    "n_observables", "consts_per_eq", "precision",
]

VARIANT_GROUPS = {
    "state_shared": "state",
    "algo_shared": "work",
    "params_shared": "params",
}


def declared_key(cfg):
    """Return the per-system deduplication key (variant excluded)."""
    return tuple(cfg[f] for f in DECLARED_KEY_FIELDS)


def declared_for_config(cfg, cache):
    """Return declared sizes for a config, rebuilding when absent.

    Old sweep records predate the ``declared`` field; the sizes are
    registry declarations made at construction, so rebuilding the
    system and solver host-side (no kernel compilation) reproduces
    them exactly. Results are cached to DECLARED_CACHE_FILE.
    """
    key = declared_key(cfg)
    if key in cache:
        return cache[key]

    from cubie.batchsolving.solver import Solver

    system, _ = make_system(cfg)
    solver = Solver(system, **baseline_kwargs(cfg))
    declared = declared_record(solver)
    cache[key] = declared
    with open(DECLARED_CACHE_FILE, "a") as fh:
        fh.write(json.dumps({"key": list(key), "declared": declared})
                 + "\n")
    return declared


def load_declared_cache():
    """Load the declared-size cache written by earlier fits."""
    cache = {}
    if os.path.exists(DECLARED_CACHE_FILE):
        with open(DECLARED_CACHE_FILE) as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[tuple(entry["key"])] = entry["declared"]
    return cache


def qualifying_sizes(recs, field):
    """Vote each measured size: True when wins outnumber losses."""
    votes = {}
    for rec in recs:
        size = rec["declared"][field]
        win_count, loss_count = votes.get(size, (0, 0))
        if rec["ratio_min"] <= WIN_RATIO:
            win_count += 1
        elif rec["ratio_min"] >= LOSS_RATIO:
            loss_count += 1
        votes[size] = (win_count, loss_count)
    return {
        size: win_count > loss_count and win_count > 0
        for size, (win_count, loss_count) in votes.items()
    }


def fit_at_most_gate(recs, field):
    """Fit an 'at most' gate: fire when size <= threshold.

    The threshold is the largest qualifying measured size whose
    smaller measured sizes all qualify too, rounded up to the grid
    unless rounding would cross a disqualified size. Returns the
    never-fire sentinel when nothing qualifies.
    """
    votes = qualifying_sizes(recs, field)
    threshold = None
    for size in sorted(votes):
        if votes[size]:
            threshold = size
        else:
            break
    if threshold is None:
        return NEVER_AT_MOST
    rounded = -(-threshold // GRID) * GRID
    disqualified = [s for s, ok in votes.items() if not ok]
    if disqualified and rounded >= min(disqualified):
        return threshold
    return rounded


def fit_at_least_gate(recs, field):
    """Fit an 'at least' gate: fire when value >= threshold.

    The threshold is the smallest qualifying measured size whose
    larger measured sizes all qualify too, rounded down to the grid
    unless rounding would cross a disqualified size. Returns the
    never-fire sentinel when nothing qualifies.
    """
    votes = qualifying_sizes(recs, field)
    threshold = None
    for size in sorted(votes, reverse=True):
        if votes[size]:
            threshold = size
        else:
            break
    if threshold is None:
        return NEVER_AT_LEAST
    rounded = (threshold // GRID) * GRID
    disqualified = [s for s, ok in votes.items() if not ok]
    if disqualified and rounded <= max(disqualified):
        return threshold
    return rounded


def measured_family(algorithm):
    """Return True when the heuristics cover this algorithm family."""
    from cubie.integrators.algorithms import resolve_alias
    from cubie.integrators.memory_heuristics import MEASURED_STEP_TYPES

    step_type, _ = resolve_alias(algorithm)
    return issubclass(step_type, MEASURED_STEP_TYPES)


def load_fit_records(results_file):
    """Load measured records and attach declared sizes to each."""
    records = []
    with open(results_file) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ratio_min" in rec and not rec.get("error"):
                records.append(rec)

    cache = load_declared_cache()
    missing = {
        declared_key(rec): rec for rec in records
        if "declared" not in rec
    }
    if missing:
        print(
            f"reconstructing declared sizes for {len(missing)} "
            "configurations (host-side build, no kernel compile)"
        )
    for idx, (key, rec) in enumerate(sorted(missing.items())):
        declared_for_config(rec, cache)
        print(f"  [{idx + 1}/{len(missing)}] {'/'.join(map(str, key))}")
    for rec in records:
        if "declared" not in rec:
            rec["declared"] = cache[declared_key(rec)]
    return records


def gates_for(f32_records, heavy, floor):
    """Fit the four size gates for one footprint-gate candidate."""

    def cells(variant, implicit=None, band=None):
        out = []
        for rec in f32_records:
            if rec["variant"] != variant:
                continue
            dec = rec["declared"]
            if implicit is not None and dec["is_implicit"] != implicit:
                continue
            footprint = dec["footprint_bytes"]
            if band == "spilled" and footprint < heavy:
                continue
            if band == "mid" and not (floor <= footprint < heavy):
                continue
            out.append(rec)
        return out

    return dict(
        heavy_spill_bytes=heavy,
        spill_floor_bytes=floor,
        state_pair_max_bytes=fit_at_most_gate(
            cells("state_shared", band="spilled"),
            "state_pair_bytes",
        ),
        work_group_max_bytes=fit_at_most_gate(
            cells("algo_shared", implicit=True, band="spilled"),
            "work_group_bytes",
        ),
        explicit_work_max_bytes=fit_at_most_gate(
            cells("algo_shared", implicit=False, band="mid"),
            "work_group_bytes",
        ),
        params_min_bytes=fit_at_least_gate(
            cells("params_shared", band="spilled"),
            "params_bytes",
        ),
    )


def build_configs(records):
    """Group records into per-system configs with per-group ratios."""
    configs = {}
    for rec in records:
        entry = configs.setdefault(
            declared_key(rec),
            dict(
                declared=rec["declared"],
                algorithm=rec["algorithm"],
                ratios={},
            ),
        )
        group = VARIANT_GROUPS.get(rec["variant"])
        if group is not None:
            entry["ratios"][group] = rec["ratio_min"]
    return configs


def choose_group(config, thresholds):
    """Return the group the resolver would relocate for a config."""
    from cubie.integrators.memory_heuristics import (
        DeclaredSizes,
        PARAMS_KEYS,
        STATE_PAIR_KEYS,
        placement_candidates,
    )

    if not measured_family(config["algorithm"]):
        return None
    dec = config["declared"]
    sizes = DeclaredSizes(
        itemsize=dec["itemsize"],
        is_implicit=dec["is_implicit"],
        footprint_bytes=dec["footprint_bytes"],
        state_pair_bytes=dec["state_pair_bytes"],
        work_group_bytes=dec["work_group_bytes"],
        params_bytes=dec["params_bytes"],
        work_location_keys=("work_location",),
    )
    candidates = placement_candidates(sizes, thresholds)
    if not candidates:
        return None
    first = candidates[0]
    if first == STATE_PAIR_KEYS:
        return "state"
    if first == PARAMS_KEYS:
        return "params"
    return "work"


def score_thresholds(configs, thresholds):
    """Count fired losses and missed wins over every configuration."""
    fired_losses = missed_wins = 0
    for config in configs.values():
        chosen = choose_group(config, thresholds)
        ratios = config["ratios"]
        chosen_ratio = ratios.get(chosen) if chosen else None
        if chosen_ratio is not None and chosen_ratio >= LOSS_RATIO:
            fired_losses += 1
        has_win = any(r <= WIN_RATIO for r in ratios.values())
        chose_win = (
            chosen_ratio is not None and chosen_ratio <= WIN_RATIO
        )
        if has_win and not chose_win:
            missed_wins += 1
    return fired_losses, missed_wins


def fit_thresholds(records):
    """Derive MemoryThresholds values from measured records.

    Only f32 records fit the size gates: the f64 spot checks steer
    the structural rules (explicit-only state pair at scaled size),
    which live in placement_candidates, not in the constants. The
    footprint-gate scan replays configurations of both precisions.
    """
    from cubie.integrators.memory_heuristics import MemoryThresholds

    f32 = [r for r in records if r["precision"] == "f32"]
    configs = build_configs(records)

    footprints = sorted(
        {r["declared"]["footprint_bytes"] for r in records}
    )
    ceilings = {
        -(-footprint // GRID) * GRID for footprint in footprints
    }
    grid_stops = sorted(
        ceilings | {stop + GRID for stop in ceilings} | {GRID}
    )

    best = None
    for heavy in grid_stops:
        for floor in [g for g in grid_stops if g <= heavy]:
            fitted = gates_for(f32, heavy, floor)
            score = score_thresholds(
                configs, MemoryThresholds(**fitted)
            )
            ranking = (*score, -heavy, -floor)
            if best is None or ranking < best[0]:
                best = (ranking, fitted)

    fitted = best[1]
    fired_losses, missed_wins = best[0][:2]
    print(
        f"footprint-gate scan: heavy_spill_bytes="
        f"{fitted['heavy_spill_bytes']}, spill_floor_bytes="
        f"{fitted['spill_floor_bytes']} "
        f"({fired_losses} fired losses, {missed_wins} missed wins)"
    )
    return fitted


def validate_thresholds(records, fitted):
    """Replay every configuration through the fitted gates.

    Groups records by system configuration, asks
    placement_candidates which group the resolver would relocate,
    and compares that decision with the measured ratios for every
    variant of the configuration.
    """
    from cubie.integrators.memory_heuristics import MemoryThresholds

    thresholds = MemoryThresholds(**fitted)
    configs = build_configs(records)

    fired_losses = missed_wins = fired_unmeasured = 0
    print("\nvalidation (fitted gates vs measured ratios):")
    for key in sorted(configs):
        config = configs[key]
        chosen = choose_group(config, thresholds)
        ratios = config["ratios"]
        label = "/".join(str(k) for k in key)
        if chosen is not None:
            ratio = ratios.get(chosen)
            if ratio is None:
                fired_unmeasured += 1
                print(f"  {label}: fires {chosen} (UNMEASURED)")
            elif ratio >= LOSS_RATIO:
                fired_losses += 1
                print(
                    f"  {label}: fires {chosen} on a LOSS "
                    f"(ratio {ratio:.2f})"
                )
            elif ratio <= WIN_RATIO:
                print(
                    f"  {label}: fires {chosen} "
                    f"(win, ratio {ratio:.2f})"
                )
            else:
                print(
                    f"  {label}: fires {chosen} "
                    f"(neutral, ratio {ratio:.2f})"
                )
        else:
            wins = {
                group: ratio for group, ratio in ratios.items()
                if ratio <= WIN_RATIO
            }
            if wins:
                missed_wins += 1
                best = min(wins.items(), key=lambda kv: kv[1])
                print(
                    f"  {label}: no rule fires but {best[0]} "
                    f"measured {best[1]:.2f} (MISSED WIN)"
                )
    print(
        f"\nsummary: {fired_losses} fires on losses, "
        f"{missed_wins} missed wins, "
        f"{fired_unmeasured} fires on unmeasured groups"
    )


def fit(results_file, arch=None):
    """Derive, validate, and print the thresholds for one card."""
    records = load_fit_records(results_file)
    if not records:
        print("no measured records found")
        return
    if arch is None:
        arches = [
            r["declared"].get("arch") for r in records
            if r["declared"].get("arch")
        ]
        arch = arches[0] if arches else "unknown"

    fitted = fit_thresholds(records)
    validate_thresholds(records, fitted)

    never = {NEVER_AT_MOST, NEVER_AT_LEAST}
    if any(value in never for value in fitted.values()):
        print(
            "\nnote: gates without qualifying records emit "
            "never-fire values"
        )
    print("\npaste into cubie/integrators/memory_heuristics.py:")
    print(f'    "{arch}": MemoryThresholds(')
    for field, value in fitted.items():
        print(f"        {field}={value},")
    print("    ),")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=None)
    parser.add_argument(
        "--phase", default="all",
        choices=["all"] + list(PHASES),
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="derive MemoryThresholds from the recorded results",
    )
    parser.add_argument(
        "--results", default=RESULTS_FILE,
        help="results file to fit from (default: sweep output)",
    )
    parser.add_argument(
        "--arch", default=None,
        help="architecture code for the emitted entry "
             "(default: from the records or the current device)",
    )
    args = parser.parse_args()
    if args.fit:
        fit(args.results, args.arch)
    elif args.single:
        run_single(json.loads(args.single))
    else:
        phases = list(PHASES) if args.phase == "all" else [args.phase]
        drive(phases)


if __name__ == "__main__":
    main()
