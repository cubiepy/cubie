#!/usr/bin/env python
"""Mean-runtime benchmark for a large Lorenz ensemble.

Solves the same Lorenz ensemble as the GPUODEBenchmarks cubie runner
(``GPU_ODE_CUBIE/bench_cubie.py``) with the same solver settings and
drive pattern: one warm-up solve to absorb JIT compilation, then
``timeit.repeat`` with garbage collection enabled and one solve per
repeat. The first 20 post-warm-up solves are discarded — the GPU has
not reached steady state and they run slow — and the statistic covers
the following ``repeats`` solves.

Five configs run, each printing compile metrics and two statistics:

``fixed``
    classical-rk4 at ``2**22`` trajectories (or the positional
    ``n_runs``).
``adaptive``
    radau at ``2**22`` trajectories (or the positional ``n_runs``),
    sized so its kernel times sit far above noise.
``chunked``
    The fixed config at ``--chunked-runs`` trajectories (or the
    positional ``n_runs``), forced to split into a few run-axis
    chunks. Each of its three memory managers gets an absolute
    24 MiB cap at the default run count, scaled with smaller
    smoke-test counts and converted to a VRAM proportion for the
    active limit mode. An explicit ``--chunked-proportion``
    overrides that portable default. Its wall time is the
    chunk-path canary: critical-path transfer, staging, writeback,
    and lost overlap costs appear there but not in kernel events.
``wave``
    The fixed config sized to exactly **two full waves** of the
    card: ``2 * SMs * blocks_per_SM * runs_per_block`` trajectories,
    where ``blocks_per_SM`` is the occupancy of the compiled fixed
    kernel. At two waves any occupancy decrease forces a third wave
    and a step runtime increase, so occupancy regressions cannot
    hide behind compute saturation. Runs with ``duration = 10`` so
    each solve lasts tens of milliseconds.
``host_overhead``
    The fixed config at a constant **8 trajectories**, never scaled
    by the positional ``n_runs``. Its kernel runs in tens of
    microseconds, so its wall statistic is the per-call host cost
    of ``Solver.solve`` — the floor a user pays on every solve
    regardless of batch size.

Two statistics are reported per config. The **kernel** statistic is
the mean of the lowest ``min_count`` per-solve kernel times — the
per-chunk ``kernel_chunk_i`` CUDA events recorded on the GPU timeline
by ``BatchSolverKernel``. The lowest solves are those that ran at
full boost clock with no on-GPU contention, so they track the
compiled kernel's intrinsic cost and drift far less between
invocations than the mean (which the upper tail pulls around). The
**wall** statistic is the mean of the lowest ``min_count`` per-solve
host times around ``Solver.solve`` (which synchronises the stream
and waits for chunked writeback before returning). Host scatter is
one-sided — delays only ever add time — so the wall floor excludes
it while still catching changes that lengthen the total critical
path or destroy chunk overlap, which slow every solve. It carries
more host scatter than the kernel statistic and supports coarser
thresholds only.

Compile metrics include registers/thread; ptxas compile-time
spill-store and spill-load byte counts; static, dynamic, and constant
memory; exact post-limit launch geometry; occupancy; SM count; run
count; and chunk count. Spill counts come from the verbose log of the
link that emitted each cubin, keyed by cubin digest — they are
code-generation counts, not allocated bytes/thread or measured
runtime traffic. The kernels must therefore link in this process:
caches are cleared on startup unless ``--no-clear-cache`` is given
(``ab_gate.py`` passes it together with a fresh cache directory, so
its workers still compile once each). All metrics require a real GPU;
the script exits under the CUDA simulator.

Usage::

    python benchmarks/lorenz_mean_runtime.py [n_runs]
        [--repeats R] [--min-count K] [--grid-cache DIR]
        [--chunked-runs N] [--chunked-proportion P]
        [--no-clear-cache] [--worker]

``--worker`` turns the process into a persistent solve server for
``ab_gate.py``: after the solvers are built, the grids loaded, and
one JIT warm-up solve run per config, it prints one ``@META`` line
per config and then ``@READY <config>...``. The wave config is
excluded from ``@READY`` — its size must be imposed by the gate so
both sides run the same launch geometry. The worker then answers
commands on stdin until ``quit`` or EOF:

``wave <n_runs>``
    Build the wave config's grid at ``n_runs``, warm it up, and
    print its ``@META`` line (which doubles as the ready signal).
``run <config> <n>``
    Run ``n`` solves and print
    ``@TIMES <config> kernel <ms>... wall <ms>...``.

The chunked and wave configs reuse the fixed config's solver
settings, so their kernels are in-process cache hits after the fixed
config compiles. Input grids are cached per trajectory count, so the
chunked config shares the fixed config's grid file at the default
sizes.
"""

import argparse
import contextlib
import hashlib
import inspect
import io
import os
import re
import shutil
import sys
import timeit
from time import perf_counter

import numpy as np

import cubie as qb
from cubie.cache_root import get_cache_root
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda
from cubie.time_logger import default_timelogger

discarded_solves = 20
min_count = 5
precision = np.float32
blocksize = 64
initial_conditions = {"x": 1.0, "y": 0.0, "z": 0.0}
default_chunked_runs = 2**22
default_chunk_instance_cap = 24 * 2**20

# cubin digest -> verbose linker log, filled by install_spill_capture
_link_diagnostics = {}

def install_spill_capture():
    """Capture verbose ptxas diagnostics for each cubin produced.

    The CUDA driver attributes expose total local memory, not register
    spill code generation. ptxas reports spill-store and spill-load
    byte counts in the verbose log of the link that emits the cubin,
    so force that diagnostic on, bypass the nvJitLink cache so a log
    is always emitted, and retain each log under a digest of the
    returned cubin bytes.
    """
    linker_class = cuda.cudadrv.driver._Linker
    original_options = linker_class._get_linker_options
    original_complete = linker_class.complete

    def diagnostic_options(linker, ptx):
        options = original_options(linker, ptx)
        options.verbose = True
        options.no_cache = True
        return options

    def diagnostic_complete(linker):
        cubin = original_complete(linker)
        digest = hashlib.sha256(bytes(cubin.code)).hexdigest()
        _link_diagnostics[digest] = linker.info_log
        return cubin

    linker_class._get_linker_options = diagnostic_options
    linker_class.complete = diagnostic_complete


def _compiled_cubin(kern):
    """Return the loaded cubin bytes and kernel entry name."""
    library = kern._codelibrary
    if hasattr(library, "get_cubin"):
        cubin = bytes(library.get_cubin().code)
    else:
        cubin = library._cubin
    entry_name = (
        getattr(kern, "entry_name", None)
        or getattr(library, "_entry_name", None)
        or kern.cres.metadata["func_name"]
    )
    return cubin, entry_name


def parse_spill_diagnostics(log, entry_name):
    """Return ptxas spill-store/load bytes for ``entry_name``."""
    pattern = re.compile(
        r"Function properties for\s+" + re.escape(entry_name)
        + r"\s*\r?\n[^\r\n]*?(\d+) bytes spill stores,\s*"
        r"(\d+) bytes spill loads"
    )
    match = pattern.search(log)
    if match is None:
        raise SystemExit(
            f"no ptxas spill record for loaded entry {entry_name!r}"
        )
    return int(match.group(1)), int(match.group(2))


def collect_kernel_time(solver, kernel_ms):
    """Append one solve's kernel CUDA-event total (ms) to ``kernel_ms``.

    Sums the per-chunk ``kernel_chunk_i`` GPU-timeline times, which
    exclude host<->device transfer traffic.
    """
    kernel_ms.append(sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    ))


def resolve_n_runs(n_runs):
    """Return (fixed, adaptive) trajectory counts."""
    if n_runs is not None:
        return n_runs, n_runs
    return 2**22, 2**22


def resolve_chunk_settings(n_runs, chunked_runs, proportion,
                           total_memory):
    """Resolve the chunked config's run count and VRAM proportion."""
    runs = chunked_runs
    if runs is None:
        runs = n_runs if n_runs is not None else default_chunked_runs
    if proportion is None:
        cap = default_chunk_instance_cap * runs / default_chunked_runs
        proportion = cap / total_memory
    return runs, proportion


def build_fixed_style_solver(system, **memory_kwargs):
    """Build one solver with the fixed-config settings."""
    return qb.Solver(
        system,
        algorithm="classical-rk4",
        dt=0.001,
        save_every=1.0,
        step_controller="fixed",
        output_types=["state"],
        time_logging_level="default",
        **memory_kwargs,
    )


def build_solvers(n_fixed, n_adaptive, n_chunked, chunked_proportion):
    """Build the Lorenz system and all benchmark solvers.

    Returns a dict of ``key -> (label, solver, n_runs, duration)`` in
    the order the configs run. The wave config's ``n_runs`` is
    ``None`` until it is sized from the fixed config's occupancy (or
    by the gate's ``wave`` command in worker mode).
    """
    lorenz_system = qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"sigma": 10.0, "rho": 21.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=precision,
    )

    fixed_solver = build_fixed_style_solver(lorenz_system)
    # Declare the rho sweep so sigma and beta fold as literals.
    fixed_solver.set_swept_params(["rho"])

    # An implicit adaptive config so the gate exercises the Newton
    # and Krylov solvers, not just explicit tableaus. The truncated
    # Neumann preconditioner diverges on the coupled Radau stages of
    # this system, so the diagonal Jacobi preconditioner is used.
    adaptive_solver = qb.Solver(
        lorenz_system,
        algorithm="radau",
        preconditioner_type="jacobi",
        atol=1e-06,
        rtol=1e-06,
        save_every=1.0,
        dt_min=1e-12,
        dt_max=1e3,
        step_controller="pid",
        kp=6 / 5,
        kd=0.0,
        ki=0.0,
        max_gain=5.0,
        min_gain=0.1,
        output_types=["state"],
        time_logging_level="default",
    )

    chunked_solver = build_fixed_style_solver(
        lorenz_system,
        stream_group="chunked",
        mem_proportion=chunked_proportion,
    )
    # Chunk sizing sums the VRAM caps of every memory-manager
    # instance in the solver's stream group: the kernel (capped by
    # mem_proportion above) plus its input and output array
    # managers, which register as auto-pool instances with large
    # caps. Move both to the same small manual proportion so the
    # group cap is three times the proportion and the batch must
    # chunk. Caps derive from total (not free) VRAM, so the chunk
    # count is identical for both gate sides on one machine.
    for instance in (
        chunked_solver.kernel.input_arrays,
        chunked_solver.kernel.output_arrays,
    ):
        qb.default_memmgr.set_manual_proportion(
            instance, chunked_proportion
        )

    wave_solver = build_fixed_style_solver(lorenz_system)
    host_overhead_solver = build_fixed_style_solver(lorenz_system)

    return {
        "fixed": ("fixed (classical-rk4)", fixed_solver, n_fixed, 1.0),
        "adaptive": (
            "adaptive (radau)",
            adaptive_solver,
            n_adaptive,
            1.0,
        ),
        "host_overhead": (
            "host overhead (classical-rk4)",
            host_overhead_solver,
            8,
            1.0,
        ),
        "chunked": (
            "chunked (classical-rk4)",
            chunked_solver,
            n_chunked,
            1.0,
        ),
        "wave": ("wave (classical-rk4)", wave_solver, None, 10.0),
    }


def load_grid(solver, n_runs, grid_cache):
    """Load the input grid from cache, or build and save it.

    Grid contents depend only on the trajectory count, so the cache
    file is keyed by ``n_runs`` and shared between configs of the
    same size. The save is atomic (write then ``os.replace``) so
    concurrent workers sharing a cache directory cannot read a
    partial file.
    """
    gfile = (
        os.path.join(grid_cache, f"grid_{n_runs}.npz")
        if grid_cache is not None
        else None
    )
    if gfile is not None and os.path.exists(gfile):
        with np.load(gfile) as grid:
            return grid["inits"], grid["params"]
    parameters = {"rho": np.linspace(0.0, 21.0, n_runs)}
    inits, params = solver.build_grid(
        initial_values=initial_conditions, parameters=parameters
    )
    if gfile is not None:
        os.makedirs(grid_cache, exist_ok=True)
        tmp = f"{gfile}.{os.getpid()}.npz"
        np.savez(tmp, inits=inits, params=params)
        try:
            os.replace(tmp, gfile)
        except OSError:
            # Another worker raced us to the same grid (Windows
            # locks open files); ours is in memory, drop the temp.
            with contextlib.suppress(OSError):
                os.remove(tmp)
    return inits, params


# Both gate workers run this driver, so it must measure the lean
# result path on either API generation: trees with results_type get
# "raw"; trees without it return the zero-copy SolveResult by default.
_solve_extra_kwargs = (
    {"results_type": "raw"}
    if "results_type" in inspect.signature(qb.Solver.solve).parameters
    else {}
)


def solve_once(solver, inits, params, kernel_ms, wall_ms, duration):
    """Run one solve; record its kernel and wall times (ms)."""
    start = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(
            initial_values=inits,
            parameters=params,
            blocksize=blocksize,
            duration=duration,
            **_solve_extra_kwargs,
        )
    wall_ms.append(1000.0 * (perf_counter() - start))
    collect_kernel_time(solver, kernel_ms)


def compile_meta(solver):
    """Read compile metrics from the solver's compiled kernel.

    Returns a dict of integers: ptxas compile-time spill-store and
    spill-load byte counts plus resources and actual launch geometry.
    """
    (kern,) = solver.kernel.kernel.overloads.values()
    # The MLIR backend's kernel objects (CompileResult) resolve the
    # resource attributes from compile metadata, populated on demand;
    # numba-cuda kernels expose the same names as plain properties.
    # Both reach the loaded driver function via ``_codelibrary`` (the
    # public ``library`` is a numba-cuda-only spelling).
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()

    cubin, entry_name = _compiled_cubin(kern)
    digest = hashlib.sha256(cubin).hexdigest()
    log = _link_diagnostics.get(digest)
    if log is None:
        raise SystemExit(
            "no ptxas diagnostics were captured for the loaded cubin; "
            "rerun without --no-clear-cache so it links in this "
            "process"
        )
    spill_stores, spill_loads = parse_spill_diagnostics(
        log, entry_name
    )

    first_chunk_runs = int(solver.kernel.run_params[0].runs)
    pad = 4 if solver.kernel.shared_memory_needs_padding else 0
    padded_bytes = solver.kernel.shared_memory_bytes + pad
    dynshared = padded_bytes * min(first_chunk_runs, blocksize)
    actual_blocksize, dynshared = solver.kernel.limit_blocksize(
        blocksize,
        dynshared,
        padded_bytes,
        first_chunk_runs,
    )
    dynshared = max(4, dynshared)
    context = cuda.current_context()
    blocks_per_sm = context.get_active_blocks_per_multiprocessor(
        cufunc, actual_blocksize, dynshared
    )
    device = cuda.get_current_device()
    threads_per_loop = solver.kernel.single_integrator.threads_per_step
    return {
        "regs": int(kern.regs_per_thread),
        "spill_store_bytes": spill_stores,
        "spill_load_bytes": spill_loads,
        "shared": int(kern.shared_mem_per_block),
        "dynshared": int(dynshared),
        "const": int(kern.const_mem_size),
        "blocks_per_sm": int(blocks_per_sm),
        "sms": int(device.MULTIPROCESSOR_COUNT),
        "blocksize": int(actual_blocksize),
        "runs_per_block": int(actual_blocksize // threads_per_loop),
        "runs": int(solver.kernel.run_params.runs),
        "chunks": int(solver.chunks),
    }


def meta_line(key, meta):
    """Format one config's compile metrics as a parseable line."""
    fields = " ".join(f"{name}={value}" for name, value in meta.items())
    return f"@META {key} {fields}"


def check_chunks(key, solver):
    """Exit if a config's chunk count contradicts its purpose."""
    if key == "chunked":
        if solver.chunks < 2:
            raise SystemExit(
                "the chunked config ran in a single chunk; lower "
                "--chunked-proportion (or raise --chunked-runs) so "
                "its batch exceeds the group's VRAM cap"
            )
    elif solver.chunks != 1:
        raise SystemExit(
            f"config {key!r} unexpectedly chunked "
            f"({solver.chunks} chunks); its timings would not be "
            "comparable across sides"
        )


def wave_runs_from(meta):
    """Trajectory count filling exactly two waves at ``meta``'s
    occupancy."""
    return (
        2 * meta["sms"] * meta["blocks_per_sm"] * meta["runs_per_block"]
    )


def benchmark(key, label, solver, n_runs, duration, repeats, k,
              grid_cache=None):
    """Run ``repeats`` solves after a warm-up; print the statistics.

    Prints the config's ``@META`` compile-metrics line and the mean
    of the ``k`` lowest per-solve kernel and wall times (the module
    docstring explains both statistics), then parseable
    ``RESULT``/``RESULT_WALL`` lines. Returns the config's compile
    metrics.
    """
    inits, params = load_grid(solver, n_runs, grid_cache)
    kernel_ms = []
    wall_ms = []

    def run():
        solve_once(solver, inits, params, kernel_ms, wall_ms, duration)

    run()  # warm-up (JIT compilation)
    check_chunks(key, solver)
    meta = compile_meta(solver)
    print(meta_line(key, meta))
    kernel_ms.clear()
    wall_ms.clear()
    timeit.repeat(
        run,
        setup="gc.enable()",
        repeat=discarded_solves + repeats,
        number=1,
    )
    kernel_arr = np.sort(np.asarray(kernel_ms[discarded_solves:]))
    kernel_stat = kernel_arr[:k].mean()
    wall_arr = np.sort(np.asarray(wall_ms[discarded_solves:]))
    wall_stat = wall_arr[:k].mean()
    print(
        f"{label}: kernel {kernel_stat:.3f} ms, wall {wall_stat:.3f} "
        f"ms (means of the {k} lowest kernel and wall times over "
        f"{repeats} solves of {n_runs} trajectories)"
    )
    print(f"RESULT {key} {kernel_stat:.4f}")
    print(f"RESULT_WALL {key} {wall_stat:.4f}")
    return meta


def worker_loop(solvers, grid_cache):
    """Serve solve blocks over stdin/stdout for ``ab_gate.py``.

    Prints one ``@META`` line per config and ``@READY <config>...``
    once every non-wave config has its grid and a JIT warm-up solve
    behind it, then answers ``wave <n_runs>`` (size, warm up, and
    announce the wave config) and ``run <config> <n>`` (reply
    ``@TIMES <config> kernel <ms>... wall <ms>...``) until ``quit``
    or EOF.
    """
    ready = {}
    for key, (label, solver, n_runs, duration) in solvers.items():
        if key == "wave":
            continue
        inits, params = load_grid(solver, n_runs, grid_cache)
        solve_once(solver, inits, params, [], [], duration)
        check_chunks(key, solver)
        print(meta_line(key, compile_meta(solver)), flush=True)
        ready[key] = (solver, inits, params, duration)
    print("@READY " + " ".join(ready), flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "quit":
            break
        if parts[0] == "wave":
            label, solver, _, duration = solvers["wave"]
            inits, params = load_grid(
                solver, int(parts[1]), grid_cache
            )
            solve_once(solver, inits, params, [], [], duration)
            check_chunks("wave", solver)
            print(meta_line("wave", compile_meta(solver)), flush=True)
            ready["wave"] = (solver, inits, params, duration)
            continue
        if parts[0] != "run":
            continue
        key, count = parts[1], int(parts[2])
        solver, inits, params, duration = ready[key]
        kernel_ms = []
        wall_ms = []
        for _ in range(count):
            solve_once(
                solver, inits, params, kernel_ms, wall_ms, duration
            )
        print(
            "@TIMES " + key
            + " kernel " + " ".join(f"{v:.9g}" for v in kernel_ms)
            + " wall " + " ".join(f"{v:.9g}" for v in wall_ms),
            flush=True,
        )


def main():
    """Parse arguments and run the benchmark or the worker mode."""
    args = _parse_args()
    if CUDA_SIMULATION:
        raise SystemExit(
            "compile metrics come from the compiled cufunc; run this "
            "benchmark on a real GPU, not under NUMBA_ENABLE_CUDASIM"
        )

    cache_root = get_cache_root()
    if not args.no_clear_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)

    # Per-chunk CUDA events are recorded only while the time logger
    # is armed; Solver construction re-arms it from each solver's
    # time_logging_level argument.
    default_timelogger.set_verbosity("default")

    # Manual VRAM proportions only constrain allocation (and so
    # drive chunking) in active limit mode; passive mode measures
    # raw free VRAM and would never chunk the chunked config.
    qb.default_memmgr.set_limit_mode("active")

    install_spill_capture()
    n_fixed, n_adaptive = resolve_n_runs(args.n_runs)
    total_memory = int(cuda.current_context().get_memory_info()[1])
    chunked_runs, chunked_proportion = resolve_chunk_settings(
        args.n_runs,
        args.chunked_runs,
        args.chunked_proportion,
        total_memory,
    )
    solvers = build_solvers(
        n_fixed,
        n_adaptive,
        chunked_runs,
        chunked_proportion,
    )

    if args.worker:
        worker_loop(solvers, args.grid_cache)
        return

    metas = {}
    for key, (label, solver, n_runs, duration) in solvers.items():
        if key == "wave":
            n_runs = wave_runs_from(metas["fixed"])
        metas[key] = benchmark(
            key,
            label,
            solver,
            n_runs,
            duration,
            args.repeats,
            args.min_count,
            args.grid_cache,
        )


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Kernel-runtime benchmark (Lorenz ensemble). Prints "
        "compile metrics plus the mean of the lowest kernel times and "
        "the median wall time per config; drive A/B comparisons with "
        "ab_gate.py."
    )
    parser.add_argument("n_runs", nargs="?", type=int, default=None)
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Solves to time per config after the discard window.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=min_count,
        help="Number of lowest per-solve kernel times to average.",
    )
    parser.add_argument(
        "--grid-cache",
        default=None,
        metavar="DIR",
        help="Cache/load the input grid here to skip rebuilding it.",
    )
    parser.add_argument(
        "--chunked-runs",
        type=int,
        default=None,
        help="Trajectory count for the chunked config (defaults to "
        "the positional n_runs when supplied, otherwise 2**22).",
    )
    parser.add_argument(
        "--chunked-proportion",
        type=float,
        default=None,
        help="Manual VRAM proportion for each chunked memory manager; "
        "by default an absolute 24 MiB cap scales with run count.",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Reuse the compile cache instead of clearing it (for a "
        "fixed source tree across repeated invocations).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Serve solve blocks over stdin/stdout for ab_gate.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
