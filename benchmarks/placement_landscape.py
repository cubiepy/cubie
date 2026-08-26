#!/usr/bin/env python
"""Per-buffer shared-placement landscape sweep (GPU).

``--pool`` compiles variants ahead, ``--run`` times them, ``--report``
writes ``summary.md``; rows append to ``records.jsonl`` and re-runs
skip completed keys. ``--retake FILE`` drops the listed keys first.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import warnings
from importlib import util as importlib_util
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve()
OUT_DEFAULT = Path(
    r"C:\local_working_projects\cubie-notes\placement_landscape"
)
FABBRI_CELLML = (
    REPO / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)
PRECISION = np.float32
BLOCKSIZE = 256
ROUNDS = 3
ESCALATED_ROUNDS = 6
TWIN_TOLERANCE = 0.02
IQR_TOLERANCE = 0.05
BASELINE_TOLERANCE = 0.03
BASELINE_RETRIES = 3
TIMED_TASKS = ("single", "pair", "geometry", "blocksize", "padded")
WIN_RATIO = 0.95
GROUP_SIZE = 6
WORKERS = 4
MAX_SHARED_PER_BLOCK = 49152
SMOKE = os.environ.get("PL_SMOKE", "") == "1"
if SMOKE:
    ROUNDS = 1
    ESCALATED_ROUNDS = 2

# --- systems -----------------------------------------------------------


def _fixtures_module():
    spec = importlib_util.spec_from_file_location(
        "landscape_system_fixtures", REPO / "tests" / "system_fixtures.py"
    )
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_lorenz():
    from cubie import create_ODE_system

    return create_ODE_system(
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
    from cubie import create_ODE_system

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
    return create_ODE_system(
        "\n".join(lines),
        states=states,
        parameters={"F": 8.0},
        name=f"Lorenz96_{n}",
        precision=PRECISION,
    )


def build_chain(n, consts_per_eq, n_params=2):
    """Nonlinear nearest-neighbour ring chain (memory_location_sweep)."""
    from cubie import create_ODE_system

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
    return create_ODE_system(
        dxdt=eqs,
        states={f"x{i}": 0.5 for i in range(n)},
        parameters={f"p{j}": 1.0 for j in range(n_params)},
        constants=constants,
        precision=PRECISION,
        name=f"chain_{n}s_{consts_per_eq}c",
    )


def build_hodgkin_huxley():
    from cubie import create_ODE_system

    fixtures = _fixtures_module()
    constants = dict(fixtures.HODGKIN_HUXLEY_CONSTANTS)
    i_app = constants.pop("i_app")
    return create_ODE_system(
        fixtures.HODGKIN_HUXLEY_EQUATIONS,
        states=dict(fixtures.HODGKIN_HUXLEY_STATES),
        parameters={"i_app": i_app},
        constants=constants,
        precision=PRECISION,
        name="hodgkin_huxley_param",
    )


def build_diode_line():
    return _fixtures_module().build_diode_line_system(PRECISION)


def build_fabbri():
    from cubie import load_cellml_model

    return load_cellml_model(
        str(FABBRI_CELLML),
        precision=PRECISION,
        parameters=[
            "Rate_modulation_experiments_ACh",
            "Rate_modulation_experiments_Iso_cas",
        ],
        voltage_variable="Membrane$V_ode",
    )


def grid_param(name, low, high):
    def grid(solver, n_runs):
        return solver.build_grid(
            parameters={name: np.linspace(low, high, n_runs)}
        )

    return grid


def grid_chain(solver, n_runs):
    return solver.build_grid(
        parameters={
            "p0": np.linspace(0.9, 1.1, n_runs),
            "p1": np.linspace(1.1, 0.9, n_runs),
        }
    )


def grid_fabbri(solver, n_runs):
    side = int(np.ceil(np.sqrt(n_runs)))
    ach, iso = np.meshgrid(
        np.linspace(0.0, 2e-8, side), np.linspace(0.0, 1.0, side)
    )
    return solver.build_grid(
        parameters={
            "Rate_modulation_experiments_ACh": ach.ravel()[:n_runs],
            "Rate_modulation_experiments_Iso_cas": iso.ravel()[:n_runs],
        }
    )


TIGHT = {"atol": 1e-6, "rtol": 1e-6, "dt_min": 1e-12, "dt_max": 1e3}

SYSTEMS = {
    "lorenz": dict(
        build=build_lorenz, grid=grid_param("rho", 0.0, 21.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "lorenz96_10": dict(
        build=lambda: build_lorenz96(10), grid=grid_param("F", 0.0, 16.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "lorenz96_20": dict(
        build=lambda: build_lorenz96(20), grid=grid_param("F", 0.0, 16.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "lorenz96_40": dict(
        build=lambda: build_lorenz96(40), grid=grid_param("F", 0.0, 16.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "chain20": dict(
        build=lambda: build_chain(20, 3), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "chain32": dict(
        build=lambda: build_chain(32, 3), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "chain64": dict(
        build=lambda: build_chain(64, 3), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "chain32_c8": dict(
        build=lambda: build_chain(32, 8), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "hodgkin_huxley": dict(
        build=build_hodgkin_huxley, grid=grid_param("i_app", 5.0, 15.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "diode_line": dict(
        build=build_diode_line, grid=grid_param("amp", 0.5, 1.5),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT, time_all=True,
    ),
    "fabbri": dict(
        build=build_fabbri, grid=grid_fabbri,
        duration=0.2, n_runs=1 << 17,
        kwargs={"atol": 1e-6, "rtol": 1e-4, "dt_min": 1e-12,
                "dt_max": 1e-2},
        constants={"Rate_modulation_experiments_ANS": 1.0},
        time_all=False,
    ),
}
if SMOKE:
    for _spec in SYSTEMS.values():
        _spec["n_runs"] = 4096

# --- algorithms --------------------------------------------------------


def implicit(algorithm, variant="exact", correction="lu"):
    kwargs = dict(algorithm=algorithm, linear_correction_type=correction)
    if correction != "lu":
        kwargs["preconditioner_type"] = "jacobi"
    # exact/inexact: prefactored on/off; newton: full Newton.
    if variant == "inexact":
        kwargs["inexact_newton"] = True
        kwargs["prefactored"] = False
    elif variant == "newton":
        kwargs["inexact_newton"] = False
    return kwargs


ALGORITHMS = {
    "tsit5": dict(algorithm="tsit5"),
    "kvaerno3_exact": implicit("kvaerno3"),
    "kvaerno3_inexact": implicit("kvaerno3", "inexact"),
    "kvaerno3_newton": implicit("kvaerno3", "newton"),
    "radau_iia_5_exact": implicit("radau_iia_5"),
    "radau_iia_5_inexact": implicit("radau_iia_5", "inexact"),
    "radau_iia_5_newton": implicit("radau_iia_5", "newton"),
    "rosenbrock23": dict(algorithm="rosenbrock23",
                         linear_correction_type="lu"),
    # stage axis
    "sdirk_2_2_exact": implicit("sdirk_2_2"),
    "kvaerno5_exact": implicit("kvaerno5"),
    "radau_iia_3_exact": implicit("radau_iia_3"),
    "radau_iia_9_exact": implicit("radau_iia_9"),
    "bogacki-shampine-32": dict(algorithm="bogacki-shampine-32"),
    "dopri54": dict(algorithm="dopri54"),
    "vern7": dict(algorithm="vern7"),
    "ros3p": dict(algorithm="ros3p", linear_correction_type="lu"),
    "rodas3p": dict(algorithm="rodas3p", linear_correction_type="lu"),
    # bicgstab vectors
    "radau_iia_5_inexact_bicgstab": implicit(
        "radau_iia_5", "inexact", "bicgstab"
    ),
    "radau_iia_5_exact_bicgstab": implicit(
        "radau_iia_5", "exact", "bicgstab"
    ),
}

BASE_ALGOS = (
    "tsit5", "kvaerno3_exact", "kvaerno3_inexact",
    "radau_iia_5_exact", "radau_iia_5_inexact", "rosenbrock23",
)
STAGE_ALGOS = (
    "sdirk_2_2_exact", "kvaerno5_exact", "radau_iia_3_exact",
    "radau_iia_9_exact", "bogacki-shampine-32", "dopri54", "vern7",
    "ros3p", "rodas3p",
)
NEWTON_ALGOS = ("kvaerno3_newton", "radau_iia_5_newton")
SWEEP_SYSTEMS = (
    "chain20", "chain32", "lorenz96_20", "lorenz", "lorenz96_10",
    "hodgkin_huxley", "chain32_c8", "chain64", "diode_line",
    "lorenz96_40",
)

GEOMETRY_CONFIGS = (
    ("chain20", "kvaerno3_inexact"),
    ("chain32", "radau_iia_5_inexact"),
    ("lorenz96_20", "kvaerno3_exact"),
    ("fabbri", "radau_iia_5_exact"),
)
WAVES_CONFIG = ("chain32", "radau_iia_5_inexact")


def config_list():
    """Return ordered (phase, system, algorithm) triples."""
    phase_a = [
        ("A", "chain20", "kvaerno3_inexact"),
        ("A", "chain32", "radau_iia_5_inexact"),
        ("A", "lorenz96_20", "kvaerno3_exact"),
        ("A", "lorenz96_20", "radau_iia_5_exact"),
        ("A", "fabbri", "radau_iia_5_exact"),
    ]
    seen = {(s, a) for _, s, a in phase_a}
    phase_c = []
    for system in SWEEP_SYSTEMS:
        for algo in BASE_ALGOS:
            if system == "diode_line" and not algo.startswith(
                ("radau", "rosenbrock")
            ):
                continue
            if (system, algo) not in seen:
                seen.add((system, algo))
                phase_c.append(("C", system, algo))
    for system in ("chain32", "lorenz96_20"):
        for algo in STAGE_ALGOS:
            phase_c.append(("C", system, algo))
    phase_c.append(("C", "chain32", "radau_iia_5_inexact_bicgstab"))
    phase_c.append(("C", "fabbri", "radau_iia_5_exact_bicgstab"))
    phase_d = [
        ("D", "fabbri", "kvaerno3_exact"),
        ("D", "fabbri", "radau_iia_5_inexact"),
        ("D", "fabbri", "tsit5"),
    ]
    phase_e = [
        ("E", system, algo)
        for system in SWEEP_SYSTEMS + ("fabbri",)
        for algo in NEWTON_ALGOS
        if not (system == "diode_line" and not algo.startswith("radau"))
    ]
    return phase_a + phase_c + phase_d + phase_e


def task_list():
    """Return ordered tasks: (phase, kind, system, algorithm)."""
    configs = config_list()
    tasks = []
    for phase, system, algo in configs:
        if phase == "A":
            tasks.append(("A", "features", system, algo))
            tasks.append(("A", "singles", system, algo))
    for system, algo in GEOMETRY_CONFIGS:
        tasks.append(("B", "geometry", system, algo))
    for phase, system, algo in configs:
        if phase == "C":
            tasks.append(("C", "features", system, algo))
            tasks.append(("C", "singles", system, algo))
    for phase, system, algo in configs:
        if phase in ("A", "C"):
            tasks.append(("D", "pairs", system, algo))
            tasks.append(("D", "blocksize", system, algo))
    tasks.append(("D", "waves", *WAVES_CONFIG))
    for phase, system, algo in configs:
        if phase == "D":
            tasks.append(("D", "features", system, algo))
            tasks.append(("D", "singles", system, algo))
            tasks.append(("D", "pairs", system, algo))
    for phase, system, algo in configs:
        if phase in ("A", "C", "D"):
            tasks.append(("D", "padded", system, algo))
    for phase, system, algo in configs:
        if phase == "E":
            tasks.append(("E", "features", system, algo))
            tasks.append(("E", "singles", system, algo))
            tasks.append(("E", "pairs", system, algo))
            tasks.append(("E", "blocksize", system, algo))
            tasks.append(("E", "padded", system, algo))
    return tasks


def task_key(kind, system, algo, variant=""):
    return f"{kind}|{system}|{algo}|{variant}"


# --- records -----------------------------------------------------------


class Records:
    """Append-only JSONL store keyed by ``key``; ``extra`` paths are read."""

    def __init__(self, path, extra=()):
        self.path = Path(path)
        self.extra = [Path(p) for p in extra]
        self.rows = []
        self.keys = set()
        self.reload()

    def reload(self):
        self.rows = []
        self.keys = set()
        for path in [self.path] + self.extra:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.rows.append(row)
                    self.keys.add(row["key"])
        self.rows.sort(key=lambda row: row.get("time", 0.0))

    def has(self, key):
        return key in self.keys

    def get(self, key):
        for row in reversed(self.rows):
            if row["key"] == key:
                return row
        return None

    def append(self, row):
        row = dict(row)
        row["time"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        self.keys.add(row["key"])

    def select(self, **fields):
        return [
            row for row in self.rows
            if all(row.get(k) == v for k, v in fields.items())
        ]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --- solver construction -----------------------------------------------


def solver_kwargs(system_name, algo_name):
    spec = SYSTEMS[system_name]
    kwargs = dict(spec["kwargs"])
    kwargs.update(ALGORITHMS[algo_name])
    kwargs.update(
        output_types=["state"],
        save_every=spec["duration"],
        time_logging_level="default",
        auto_memory=False,
    )
    return kwargs


def make_solver(system, system_name, algo_name, placement=None,
                extra=None):
    from cubie import Solver

    kwargs = solver_kwargs(system_name, algo_name)
    if extra:
        kwargs.update(extra)
    solver = Solver(system, **kwargs)
    constants = SYSTEMS[system_name].get("constants")
    if constants:
        solver.update(constants)
    if placement:
        solver.update(placement)
    return solver


def placement_for(buffers):
    return {f"{name}_location": "shared" for name in buffers}


def kernel_ms(solver):
    return sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )


def solve_once(solver, inits, params, duration, blocksize=BLOCKSIZE):
    """Solve and return ``(kernel_ms, snapshot)``; the result is dropped."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(
            inits, params, duration=duration, blocksize=blocksize,
            grid_type="verbatim",
        )
    snapshot = dict(
        state_last=np.array(result.state[-1]),
        status_hist=status_histogram(result),
    )
    counters = result.iteration_counters
    if counters is not None:
        snapshot["counters"] = np.array(counters)
    del result
    return kernel_ms(solver), snapshot


def kernel_resources(solver):
    dispatcher = solver.kernel.kernel
    regs = list(dispatcher.get_regs_per_thread().values())[0]
    local_bytes = list(dispatcher.get_local_mem_per_thread().values())[0]
    return int(regs), int(local_bytes)


def candidate_buffers(solver):
    """Return every nonempty all-local buffer with a location setting."""
    from numpy import dtype as np_dtype

    from cubie.buffer_registry import buffer_registry

    out = []
    seen = set()
    for parent, group in buffer_registry._groups.items():
        config = getattr(parent, "compile_settings", None)
        for name in group.relocatable_names():
            entry = group.entries[name]
            if entry.size <= 0 or name in seen:
                continue
            if not hasattr(config, f"{name}_location"):
                continue
            if entry.location != "local":
                continue
            seen.add(name)
            out.append(
                dict(
                    name=name,
                    owner=type(parent).__name__,
                    elements=int(entry.size),
                    itemsize=int(np_dtype(entry.dtype).itemsize),
                    persistent=bool(getattr(entry, "persistent", False)),
                )
            )
    return out


def launch_geometry(solver, blocksize, n_runs, dynshared=None):
    """Return dict of blocks/SM, resident runs, waves at this launch."""
    from cubie.cuda_simsafe import cuda

    kernel = solver.kernel
    (kern,) = kernel.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    pad = 4 if kernel.shared_memory_needs_padding else 0
    bytes_per_run = kernel.shared_memory_bytes + pad
    if dynshared is None:
        dynshared = int(bytes_per_run * min(n_runs, blocksize))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocksize, dynshared = kernel.limit_blocksize(
                blocksize, dynshared, bytes_per_run, n_runs
            )
    dynshared = max(4, dynshared)
    context = cuda.current_context()
    blocks_per_sm = int(
        context.get_active_blocks_per_multiprocessor(
            cufunc, blocksize, dynshared
        )
    )
    device = cuda.get_current_device()
    sms = int(device.MULTIPROCESSOR_COUNT)
    threads_per_run = solver.kernel.single_integrator.threads_per_step
    runs_per_block = max(1, blocksize // threads_per_run)
    resident = max(1, blocks_per_sm * runs_per_block)
    return dict(
        blocksize=int(blocksize),
        dynshared=int(dynshared),
        bytes_per_run=int(bytes_per_run),
        blocks_per_sm=blocks_per_sm,
        resident_threads=blocks_per_sm * blocksize,
        waves=n_runs / (resident * sms),
    )


def pin_launch(solver, blocksize, dynshared):
    """Fix the launch geometry of one solver."""
    solver.kernel.limit_blocksize = (
        lambda bs, dyn, bpr, runs: (blocksize, dynshared)
    )


def status_histogram(result):
    from cubie.result_codes import CUBIE_RESULT_CODES

    codes = np.asarray(result.status_codes).ravel()
    flags = codes & 0xFFFF
    out = {"failed": int(np.count_nonzero(flags))}
    for member in CUBIE_RESULT_CODES:
        if member.value == 0:
            continue
        count = int(np.count_nonzero(flags & member.value))
        if count:
            out[member.name] = count
    return out


def compare_outputs(reference, result):
    a = reference["state_last"]
    b = result["state_last"]
    nan_match = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    per_run = np.abs(np.nan_to_num(a) - np.nan_to_num(b)).max(axis=0)
    return dict(
        max_abs_diff=float(per_run.max()),
        runs_differing=int(np.count_nonzero(per_run)),
        nan_match=nan_match,
    )


# --- timing ------------------------------------------------------------


def time_group(entries, inits, params, duration, blocksize=BLOCKSIZE,
               rounds=ROUNDS):
    """Interleave solves of ``entries`` (label -> solver); first is base.

    Returns label -> stats. A twin of the baseline is the null row.
    """
    labels = list(entries)
    base_label = labels[0]
    samples = {label: [] for label in labels}
    warm = {}
    warm_ms = {}
    for label in labels:
        warm_ms[label], result = solve_once(
            entries[label], inits, params, duration, blocksize
        )
        warm[label] = result
    block, min_count = block_plan(max(warm_ms.values()))
    reference = warm[base_label]
    checks = {
        label: dict(
            compare_outputs(reference, warm[label]),
            status_hist=warm[label]["status_hist"],
        )
        for label in labels
    }
    _, repeat = solve_once(
        entries[base_label], inits, params, duration, blocksize
    )
    repeat_check = compare_outputs(reference, repeat)
    for label in labels:
        checks[label]["reference_repeat_diff"] = repeat_check[
            "max_abs_diff"
        ]
    del warm, repeat

    def run_rounds(count):
        for index in range(count):
            order = labels if index % 2 == 0 else list(reversed(labels))
            for label in order:
                times = []
                for _ in range(block):
                    ms, _ = solve_once(
                        entries[label], inits, params, duration, blocksize
                    )
                    times.append(ms)
                samples[label].append(lowest_mean(times, min_count))
            time.sleep(random.uniform(0.2, 0.8))

    run_rounds(2 * rounds)
    stats = summarise(samples, base_label)
    if needs_escalation(stats, labels):
        run_rounds(2 * (ESCALATED_ROUNDS - rounds))
        stats = summarise(samples, base_label)
        for label in labels:
            stats[label]["escalated"] = True
    for label in labels:
        stats[label].update(checks[label])
        stats[label]["block"] = block
        stats[label]["min_count"] = min_count
    return stats


def baseline_reference(records, system_name, algo_name):
    """Lowest recorded baseline block median for this config."""
    values = [
        float(np.median(row["base_ms"]))
        for row in records.rows
        if row.get("system") == system_name
        and row.get("algo") == algo_name
        and row.get("task") in TIMED_TASKS
        and row.get("status") == "ok"
        and "n_runs" not in row
    ]
    return min(values) if values else None


def time_group_gated(records, system_name, algo_name, entries, inits,
                     params, duration, blocksize=BLOCKSIZE):
    """Time the group, retiming while its baseline sits above the reference."""
    reference = baseline_reference(records, system_name, algo_name)
    drift = 0.0
    for attempt in range(BASELINE_RETRIES + 1):
        stats = time_group(entries, inits, params, duration, blocksize)
        base_ms = float(np.median(stats["baseline"]["base_ms"]))
        if reference:
            drift = base_ms / reference - 1.0
        if drift <= BASELINE_TOLERANCE or attempt == BASELINE_RETRIES:
            break
        print(
            f"  baseline {base_ms:.3f} ms is {drift:+.1%} off "
            f"{reference:.3f} ms; retiming", flush=True,
        )
        time.sleep(20.0)
    for label in stats:
        stats[label]["baseline_reference"] = reference
        stats[label]["baseline_drift"] = drift
        stats[label]["baseline_retries"] = attempt
    return stats


def block_plan(warm_ms):
    """Solves per block and lowest-k count from the warm solve time."""
    if warm_ms < 1000.0:
        return 5, 3
    if warm_ms < 4000.0:
        return 3, 2
    return 1, 1


def lowest_mean(values, k):
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(ordered[:k].mean())


def summarise(samples, base_label):
    base = np.asarray(samples[base_label])
    out = {}
    for label, values in samples.items():
        arr = np.asarray(values)
        paired = arr / base
        q1, q3 = np.percentile(paired, [25, 75])
        out[label] = dict(
            ms=[round(float(v), 4) for v in values],
            base_ms=[round(float(v), 4) for v in base],
            ratio_median=float(np.median(paired)),
            ratio_min=float(arr.min() / base.min()),
            ratio_iqr=float(q3 - q1),
            escalated=False,
        )
    return out


def needs_escalation(stats, labels):
    twin = [label for label in labels if label.endswith("(twin)")]
    if twin and abs(stats[twin[0]]["ratio_median"] - 1.0) > TWIN_TOLERANCE:
        return True
    return any(
        stats[label]["ratio_iqr"] > IQR_TOLERANCE for label in labels
    )


# --- compile workers ---------------------------------------------------


def worker_main():
    """Compile one placement in this process and print its resources."""
    sys.path.insert(0, str(REPO / "benchmarks"))
    from lorenz_mean_runtime import (
        _compiled_cubin,
        _link_diagnostics,
        install_spill_capture,
        parse_spill_diagnostics,
    )

    install_spill_capture()
    job = json.loads(sys.stdin.read())
    spec = SYSTEMS[job["system"]]
    system = spec["build"]()
    solver = make_solver(
        system, job["system"], job["algo"],
        placement_for(job["buffers"]),
    )
    inits, params = spec["grid"](solver, 256)
    start = time.perf_counter()
    solver.compile(inits, params, duration=spec["duration"])
    compile_s = time.perf_counter() - start
    regs, local_bytes = kernel_resources(solver)
    (kern,) = solver.kernel.kernel.overloads.values()
    cubin, entry_name = _compiled_cubin(kern)
    log = _link_diagnostics.get(hashlib.sha256(cubin).hexdigest())
    spill_store = spill_load = None
    if log is not None:
        spill_store, spill_load = parse_spill_diagnostics(log, entry_name)
    print(
        "@RESULT "
        + json.dumps(
            dict(
                regs=regs, local_bytes=local_bytes,
                spill_store_bytes=spill_store,
                spill_load_bytes=spill_load,
                compile_s=round(compile_s, 2),
                cached=log is None,
            )
        ),
        flush=True,
    )


def compile_jobs(records, jobs, workers=WORKERS, phase=""):
    """Compile ``jobs`` (dicts: system, algo, buffers) in subprocesses."""
    pending = [
        job for job in jobs
        if not records.has(
            task_key("compile", job["system"], job["algo"],
                     "+".join(job["buffers"]))
        )
    ]
    running = []
    env = dict(os.environ)
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            process = subprocess.Popen(
                [sys.executable, str(BENCH), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env,
            )
            process.stdin.write(json.dumps(job))
            process.stdin.close()
            running.append((job, process, time.perf_counter()))
        still = []
        for job, process, started in running:
            if process.poll() is None:
                if time.perf_counter() - started > 1800:
                    process.kill()
                    records.append(
                        dict(
                            key=task_key(
                                "compile", job["system"], job["algo"],
                                "+".join(job["buffers"])
                            ),
                            task="compile", phase=phase, status="timeout",
                            **job,
                        )
                    )
                    continue
                still.append((job, process, started))
                continue
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            key = task_key(
                "compile", job["system"], job["algo"],
                "+".join(job["buffers"])
            )
            if records.has(key):
                continue
            payload = None
            for line in stdout.splitlines():
                if line.startswith("@RESULT "):
                    payload = json.loads(line[len("@RESULT "):])
            if payload is None:
                records.append(
                    dict(
                        key=key, task="compile", phase=phase,
                        status="error", error=stderr[-3000:], **job,
                    )
                )
            else:
                records.append(
                    dict(key=key, task="compile", phase=phase,
                         status="ok", **job, **payload)
                )
                print(
                    f"  compiled {job['system']}/{job['algo']}/"
                    f"{'+'.join(job['buffers']) or 'baseline'}: "
                    f"{payload['regs']} regs, {payload['local_bytes']} B "
                    f"local, {payload['compile_s']} s",
                    flush=True,
                )
        running = still
        if running:
            time.sleep(1.0)


def wait_for_compiles(records, jobs, timeout_s):
    """Block until every job has a compile row or ``timeout_s`` passes."""
    deadline = time.perf_counter() + timeout_s
    while True:
        records.reload()
        missing = [
            job for job in jobs
            if not records.has(
                task_key("compile", job["system"], job["algo"],
                         "+".join(job["buffers"]))
            )
        ]
        if not missing or time.perf_counter() > deadline:
            return missing
        print(
            f"  waiting on {len(missing)} compiles from the pool",
            flush=True,
        )
        time.sleep(15.0)


def compile_row(records, system_name, algo_name, buffers):
    return records.get(
        task_key("compile", system_name, algo_name, "+".join(buffers))
    )


# --- tasks -------------------------------------------------------------


def op_counts(path):
    """Count operations per function in a generated module."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        counts = dict(add=0, mul=0, div=0, pow=0, calls={}, loads={},
                      stores={}, statements=0)
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp):
                if isinstance(child.op, (ast.Add, ast.Sub)):
                    counts["add"] += 1
                elif isinstance(child.op, ast.Mult):
                    counts["mul"] += 1
                elif isinstance(child.op, ast.Div):
                    counts["div"] += 1
                elif isinstance(child.op, ast.Pow):
                    counts["pow"] += 1
            elif isinstance(child, ast.Call):
                name = getattr(child.func, "id", None) or getattr(
                    child.func, "attr", None
                )
                counts["calls"][name] = counts["calls"].get(name, 0) + 1
            elif isinstance(child, ast.Subscript):
                base = child.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                name = getattr(base, "id", None)
                if name is None:
                    continue
                bucket = (
                    "stores" if isinstance(child.ctx, ast.Store)
                    else "loads"
                )
                counts[bucket][name] = counts[bucket].get(name, 0) + 1
            elif isinstance(child, ast.stmt):
                counts["statements"] += 1
        out[node.name] = counts
    return out


def dynamics(system, system_name, algo_name, duration):
    """Mean per-run counters from a small counters-enabled solve."""
    solver = make_solver(
        system, system_name, algo_name,
        extra=dict(output_types=["state", "iteration_counters"]),
    )
    inits, params = SYSTEMS[system_name]["grid"](solver, 1024)
    _, snapshot = solve_once(solver, inits, params, duration)
    totals = snapshot["counters"].sum(axis=0)
    status = snapshot["status_hist"]
    solver.close()
    names = ["newton_iters", "krylov_iters", "steps", "rejected_steps"]
    return dict(
        {name: float(totals[i].mean()) for i, name in enumerate(names)
         if i < totals.shape[0]},
        status=status,
    )


def features_task(records, phase, system_name, algo_name):
    key = task_key("features", system_name, algo_name)
    if records.has(key):
        return
    sys.path.insert(0, str(REPO / "benchmarks"))
    from lorenz_mean_runtime import (
        _compiled_cubin,
        _link_diagnostics,
        install_spill_capture,
        parse_spill_diagnostics,
    )
    from cubie.odesystems.symbolic.engine import assignments

    try:
        from cubie.backend import _typed_block_scheduler
        block_log = _typed_block_scheduler.BLOCK_LOG
    except ImportError:
        block_log = []
    install_spill_capture()
    spec = SYSTEMS[system_name]
    start = time.perf_counter()
    system = spec["build"]()
    codegen_s = time.perf_counter() - start
    solver = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](solver, spec["n_runs"])
    start = time.perf_counter()
    solve_once(solver, inits, params, spec["duration"])
    first_solve_s = time.perf_counter() - start
    regs, local_bytes = kernel_resources(solver)
    (kern,) = solver.kernel.kernel.overloads.values()
    cubin, entry_name = _compiled_cubin(kern)
    log = _link_diagnostics.get(hashlib.sha256(cubin).hexdigest())
    spill = (None, None)
    if log is not None:
        spill = parse_spill_diagnostics(log, entry_name)
    step = solver.kernel.single_integrator._algo_step
    config = step.compile_settings
    tableau = getattr(config, "tableau", None)
    n_states = solver.kernel.single_integrator._loop.compile_settings.n_states
    geometry = launch_geometry(solver, BLOCKSIZE, spec["n_runs"])
    metadata = getattr(getattr(kern, "cres", None), "metadata", {}) or {}
    row = dict(
        key=key, task="features", phase=phase, status="ok",
        system=system_name, algo=algo_name,
        n_states=int(n_states),
        stage_count=(tableau.stage_count if tableau is not None else 1),
        solver_width=int(getattr(config, "solver_width", n_states)),
        is_implicit=bool(step.is_implicit),
        algorithm_settings={
            k: v for k, v in solver_kwargs(system_name, algo_name).items()
            if k not in ("output_types",)
        },
        regs=regs, local_bytes=local_bytes,
        spill_store_bytes=spill[0], spill_load_bytes=spill[1],
        shared_bytes_per_run=int(solver.kernel.shared_memory_bytes),
        geometry=geometry,
        candidates=candidate_buffers(solver),
        codegen_s=round(codegen_s, 2),
        first_solve_s=round(first_solve_s, 2),
        liveness=list(assignments.LIVENESS_LOG),
        block_liveness=list(block_log),
        scheduler_stats=metadata.get("typed_block_scheduler"),
        op_counts=op_counts(system.gen_file.file_path),
        dynamics=dynamics(system, system_name, algo_name,
                          spec["duration"]),
    )
    solver.close()
    records.append(row)
    print(
        f"  features {system_name}/{algo_name}: {regs} regs, "
        f"{local_bytes} B local, spill {spill}, "
        f"{len(row['candidates'])} candidates, "
        f"{geometry['waves']:.1f} waves", flush=True,
    )


def singles_task(records, phase, system_name, algo_name, pool_wait):
    spec = SYSTEMS[system_name]
    system = spec["build"]()
    base = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](base, spec["n_runs"])
    solve_once(base, inits, params, spec["duration"])
    base_regs, base_local = kernel_resources(base)
    candidates = candidate_buffers(base)
    jobs = [dict(system=system_name, algo=algo_name, buffers=[])] + [
        dict(system=system_name, algo=algo_name, buffers=[c["name"]])
        for c in candidates
    ]
    missing = wait_for_compiles(records, jobs, pool_wait)
    if missing:
        compile_jobs(records, missing, phase=phase)
    records.reload()
    base_row = compile_row(records, system_name, algo_name, [])
    variants = []
    for cand in candidates:
        row = compile_row(records, system_name, algo_name, [cand["name"]])
        if row is None or row.get("status") != "ok":
            records.append(
                dict(
                    key=task_key("single", system_name, algo_name,
                                 cand["name"]),
                    task="single", phase=phase, status="compile_failed",
                    system=system_name, algo=algo_name,
                    buffers=[cand["name"]], candidate=cand,
                )
            )
            continue
        delta = row["local_bytes"] - base_local
        variants.append((cand, row, delta))
    timed = variants
    if not spec["time_all"]:
        shrink = [v for v in variants if v[2] < 0]
        controls = sorted(
            [v for v in variants if v[2] >= 0], key=lambda v: v[2]
        )[:3]
        timed = shrink + controls
    timed = [
        v for v in timed
        if not records.has(
            task_key("single", system_name, algo_name, v[0]["name"])
        )
    ]
    for start in range(0, len(timed), GROUP_SIZE):
        chunk = timed[start:start + GROUP_SIZE]
        entries = {"baseline": base}
        entries["baseline (twin)"] = make_solver(
            system, system_name, algo_name
        )
        for cand, row, delta in chunk:
            entries[cand["name"]] = make_solver(
                system, system_name, algo_name,
                placement_for([cand["name"]]),
            )
        stats = time_group_gated(
            records, system_name, algo_name, entries, inits, params,
            spec["duration"],
        )
        geometry = {
            label: launch_geometry(solver, BLOCKSIZE, spec["n_runs"])
            for label, solver in entries.items()
        }
        twin = stats["baseline (twin)"]
        for cand, row, delta in chunk:
            stat = stats[cand["name"]]
            records.append(
                dict(
                    key=task_key("single", system_name, algo_name,
                                 cand["name"]),
                    task="single", phase=phase, status="ok",
                    system=system_name, algo=algo_name,
                    buffers=[cand["name"]], candidate=cand,
                    base_regs=base_regs, base_local_bytes=base_local,
                    base_spill=(
                        [base_row.get("spill_store_bytes"),
                         base_row.get("spill_load_bytes")]
                        if base_row else None
                    ),
                    regs=row["regs"], local_bytes=row["local_bytes"],
                    spill_store_bytes=row.get("spill_store_bytes"),
                    spill_load_bytes=row.get("spill_load_bytes"),
                    delta_local=delta, delta_regs=row["regs"] - base_regs,
                    no_effect=bool(row.get("cached")),
                    geometry=geometry[cand["name"]],
                    base_geometry=geometry["baseline"],
                    twin_ratio=twin["ratio_median"],
                    **stat,
                )
            )
            print(
                f"  {system_name}/{algo_name} {cand['name']}: local "
                f"{delta:+d} B, ratio {stat['ratio_median']:.3f} "
                f"(min {stat['ratio_min']:.3f}, twin "
                f"{twin['ratio_median']:.3f}, bs "
                f"{geometry[cand['name']]['blocksize']})", flush=True,
            )
        for label, solver in entries.items():
            if label != "baseline":
                solver.close()
    for cand, row, delta in variants:
        key = task_key("single", system_name, algo_name, cand["name"])
        if records.has(key):
            continue
        records.append(
            dict(
                key=key, task="single", phase=phase, status="untimed",
                system=system_name, algo=algo_name,
                buffers=[cand["name"]], candidate=cand,
                base_regs=base_regs, base_local_bytes=base_local,
                regs=row["regs"], local_bytes=row["local_bytes"],
                spill_store_bytes=row.get("spill_store_bytes"),
                spill_load_bytes=row.get("spill_load_bytes"),
                delta_local=delta, delta_regs=row["regs"] - base_regs,
            )
        )
    base.close()


def pairs_task(records, phase, system_name, algo_name):
    singles = [
        row for row in records.select(
            task="single", system=system_name, algo=algo_name, status="ok"
        )
    ]
    winners = sorted(
        [r for r in singles if r["ratio_median"] <= WIN_RATIO],
        key=lambda r: r["ratio_median"],
    )[:3]
    if len(winners) < 2:
        return
    spec = SYSTEMS[system_name]
    pairs = [
        sorted([a["buffers"][0], b["buffers"][0]])
        for a, b in combinations(winners, 2)
    ]
    pairs = [
        p for p in pairs
        if not records.has(task_key("pair", system_name, algo_name,
                                    "+".join(p)))
    ]
    if not pairs:
        return
    jobs = [dict(system=system_name, algo=algo_name, buffers=p)
            for p in pairs]
    compile_jobs(records, jobs, phase=phase)
    records.reload()
    system = spec["build"]()
    base = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](base, spec["n_runs"])
    entries = {
        "baseline": base,
        "baseline (twin)": make_solver(system, system_name, algo_name),
    }
    rows = {}
    for pair in pairs:
        row = compile_row(records, system_name, algo_name, pair)
        if row is None or row.get("status") != "ok":
            continue
        rows["+".join(pair)] = (pair, row)
        entries["+".join(pair)] = make_solver(
            system, system_name, algo_name, placement_for(pair)
        )
    solve_once(base, inits, params, spec["duration"])
    base_regs, base_local = kernel_resources(base)
    stats = time_group_gated(
        records, system_name, algo_name, entries, inits, params,
        spec["duration"],
    )
    single_delta = {
        r["buffers"][0]: r["delta_local"] for r in singles
    }
    for label, (pair, row) in rows.items():
        records.append(
            dict(
                key=task_key("pair", system_name, algo_name, label),
                task="pair", phase=phase, status="ok",
                system=system_name, algo=algo_name, buffers=pair,
                base_regs=base_regs, base_local_bytes=base_local,
                regs=row["regs"], local_bytes=row["local_bytes"],
                delta_local=row["local_bytes"] - base_local,
                summed_single_delta=sum(
                    single_delta.get(name, 0) for name in pair
                ),
                geometry=launch_geometry(
                    entries[label], BLOCKSIZE, spec["n_runs"]
                ),
                twin_ratio=stats["baseline (twin)"]["ratio_median"],
                **stats[label],
            )
        )
        print(
            f"  {system_name}/{algo_name} pair {label}: ratio "
            f"{stats[label]['ratio_median']:.3f}", flush=True,
        )
    for solver in entries.values():
        solver.close()


def pad_for_blocks(base, target_blocks, blocksize, n_runs):
    """Largest per-block dynamic shared (KiB steps) giving target blocks/SM."""
    best = None
    for kib in range(0, MAX_SHARED_PER_BLOCK // 1024 + 1):
        dynshared = max(4, kib * 1024)
        geometry = launch_geometry(base, blocksize, n_runs, dynshared)
        if geometry["blocks_per_sm"] == target_blocks:
            best = dynshared
        elif geometry["blocks_per_sm"] < target_blocks:
            break
    return best


def geometry_task(records, phase, system_name, algo_name):
    spec = SYSTEMS[system_name]
    system = spec["build"]()
    base = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](base, spec["n_runs"])
    solve_once(base, inits, params, spec["duration"])
    natural = launch_geometry(base, BLOCKSIZE, spec["n_runs"])
    plans = []
    for kib in (0, 2, 4, 8, 12):
        plans.append((f"carveout:{kib}KiB", 32, max(4, kib * 1024)))
    for target in (8, 6, 4, 3, 2):
        dynshared = pad_for_blocks(base, target, 32, spec["n_runs"])
        if dynshared is not None:
            plans.append((f"resident:{target}x32", 32, dynshared))
    for blocksize in (32, 64, 128, 256):
        plans.append((f"blocksize:{blocksize}", blocksize, 4))
    plans = [
        p for p in plans
        if not records.has(task_key("geometry", system_name, algo_name,
                                    p[0]))
    ]
    for start in range(0, len(plans), GROUP_SIZE):
        chunk = plans[start:start + GROUP_SIZE]
        entries = {
            "baseline": base,
            "baseline (twin)": make_solver(system, system_name, algo_name),
        }
        for label, blocksize, dynshared in chunk:
            solver = make_solver(system, system_name, algo_name)
            pin_launch(solver, blocksize, dynshared)
            entries[label] = solver
        stats = time_group_gated(
            records, system_name, algo_name, entries, inits, params,
            spec["duration"],
        )
        geometry = {
            label: launch_geometry(
                entries[label], blocksize, spec["n_runs"], dynshared
            )
            for label, blocksize, dynshared in chunk
        }
        for label, blocksize, dynshared in chunk:
            records.append(
                dict(
                    key=task_key("geometry", system_name, algo_name,
                                 label),
                    task="geometry", phase=phase, status="ok",
                    system=system_name, algo=algo_name,
                    requested_blocksize=blocksize,
                    requested_dynshared=dynshared,
                    geometry=geometry[label], base_geometry=natural,
                    twin_ratio=stats["baseline (twin)"]["ratio_median"],
                    **stats[label],
                )
            )
            print(
                f"  {system_name}/{algo_name} {label}: T="
                f"{geometry[label]['resident_threads']}, ratio "
                f"{stats[label]['ratio_median']:.3f}", flush=True,
            )
        for label, solver in entries.items():
            if label != "baseline":
                solver.close()
    base.close()


def blocksize_task(records, phase, system_name, algo_name):
    singles = records.select(
        task="single", system=system_name, algo=algo_name, status="ok"
    )
    if not singles:
        return
    best = min(singles, key=lambda r: r["ratio_median"])
    if best["ratio_median"] > WIN_RATIO:
        return
    name = best["buffers"][0]
    spec = SYSTEMS[system_name]
    system = spec["build"]()
    base = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](base, spec["n_runs"])
    solve_once(base, inits, params, spec["duration"])
    probe = make_solver(system, system_name, algo_name,
                        placement_for([name]))
    solve_once(probe, inits, params, spec["duration"])
    pad = 4 if probe.kernel.shared_memory_needs_padding else 0
    bytes_per_run = probe.kernel.shared_memory_bytes + pad
    probe.close()
    plans = []
    for blocksize in (32, 64, 128, 256):
        dynshared = bytes_per_run * blocksize
        if dynshared > MAX_SHARED_PER_BLOCK:
            continue
        label = f"{name}@bs{blocksize}"
        if records.has(task_key("blocksize", system_name, algo_name,
                                label)):
            continue
        plans.append((label, blocksize, max(4, dynshared)))
    if not plans:
        base.close()
        return
    entries = {
        "baseline": base,
        "baseline (twin)": make_solver(system, system_name, algo_name),
    }
    for label, blocksize, dynshared in plans:
        solver = make_solver(system, system_name, algo_name,
                             placement_for([name]))
        pin_launch(solver, blocksize, dynshared)
        entries[label] = solver
    stats = time_group_gated(
        records, system_name, algo_name, entries, inits, params,
        spec["duration"],
    )
    geometry = {
        label: launch_geometry(
            entries[label], blocksize, spec["n_runs"], dynshared
        )
        for label, blocksize, dynshared in plans
    }
    for label, blocksize, dynshared in plans:
        records.append(
            dict(
                key=task_key("blocksize", system_name, algo_name, label),
                task="blocksize", phase=phase, status="ok",
                system=system_name, algo=algo_name, buffers=[name],
                requested_blocksize=blocksize,
                geometry=geometry[label],
                twin_ratio=stats["baseline (twin)"]["ratio_median"],
                **stats[label],
            )
        )
        print(
            f"  {system_name}/{algo_name} {label}: ratio "
            f"{stats[label]['ratio_median']:.3f}", flush=True,
        )
    for solver in entries.values():
        solver.close()


def waves_task(records, phase, system_name, algo_name):
    singles = records.select(
        task="single", system=system_name, algo=algo_name, status="ok"
    )
    if len(singles) < 2:
        return
    best = min(singles, key=lambda r: r["ratio_median"])["buffers"][0]
    worst = max(singles, key=lambda r: r["ratio_median"])["buffers"][0]
    spec = SYSTEMS[system_name]
    system = spec["build"]()
    for n_runs in (1 << 16, 1 << 17, 1 << 18):
        labels = {
            f"{best}@{n_runs}": best, f"{worst}@{n_runs}": worst,
        }
        if all(
            records.has(task_key("waves", system_name, algo_name, label))
            for label in labels
        ):
            continue
        base = make_solver(system, system_name, algo_name)
        inits, params = spec["grid"](base, n_runs)
        entries = {
            "baseline": base,
            "baseline (twin)": make_solver(system, system_name, algo_name),
        }
        for label, name in labels.items():
            entries[label] = make_solver(
                system, system_name, algo_name, placement_for([name])
            )
        stats = time_group(entries, inits, params, spec["duration"])
        for label, name in labels.items():
            records.append(
                dict(
                    key=task_key("waves", system_name, algo_name, label),
                    task="waves", phase=phase, status="ok",
                    system=system_name, algo=algo_name, buffers=[name],
                    n_runs=n_runs,
                    geometry=launch_geometry(entries[label], BLOCKSIZE,
                                             n_runs),
                    twin_ratio=stats["baseline (twin)"]["ratio_median"],
                    **stats[label],
                )
            )
            print(
                f"  waves {label}: ratio "
                f"{stats[label]['ratio_median']:.3f}", flush=True,
            )
        for solver in entries.values():
            solver.close()


def padded_plans(records, system_name, algo_name, system, base, inits,
                 params):
    """Padding plans below natural occupancy for the all-local kernel and best single."""
    spec = SYSTEMS[system_name]
    plans = []
    natural = launch_geometry(base, 32, spec["n_runs"])
    for target in (6, 4, 3, 2):
        if target >= natural["blocks_per_sm"]:
            continue
        dynshared = pad_for_blocks(base, target, 32, spec["n_runs"])
        if dynshared is not None:
            plans.append((f"local:{target}x32", [], 32, dynshared))
    singles = records.select(
        task="single", system=system_name, algo=algo_name, status="ok"
    )
    if not singles:
        return plans
    best = min(singles, key=lambda r: r["ratio_median"])
    if best["ratio_median"] > WIN_RATIO:
        return plans
    name = best["buffers"][0]
    probe = make_solver(system, system_name, algo_name,
                        placement_for([name]))
    solve_once(probe, inits, params, spec["duration"])
    geometry = launch_geometry(probe, BLOCKSIZE, spec["n_runs"])
    blocksize = geometry["blocksize"]
    blocks = geometry["blocks_per_sm"]
    targets = sorted(
        {max(1, round(blocks * f)) for f in (0.75, 0.5, 0.33)} - {blocks},
        reverse=True,
    )
    for target in targets:
        dynshared = pad_for_blocks(probe, target, blocksize, spec["n_runs"])
        if dynshared is None or dynshared < geometry["dynshared"]:
            continue
        plans.append((f"{name}@{target}x{blocksize}", [name], blocksize,
                      dynshared))
    probe.close()
    return plans


def padded_task(records, phase, system_name, algo_name):
    spec = SYSTEMS[system_name]
    system = spec["build"]()
    base = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](base, spec["n_runs"])
    solve_once(base, inits, params, spec["duration"])
    natural = launch_geometry(base, BLOCKSIZE, spec["n_runs"])
    plans = [
        p for p in padded_plans(records, system_name, algo_name, system,
                                base, inits, params)
        if not records.has(task_key("padded", system_name, algo_name,
                                    p[0]))
    ]
    for start in range(0, len(plans), GROUP_SIZE):
        chunk = plans[start:start + GROUP_SIZE]
        entries = {
            "baseline": base,
            "baseline (twin)": make_solver(system, system_name, algo_name),
        }
        for label, buffers, blocksize, dynshared in chunk:
            solver = make_solver(
                system, system_name, algo_name,
                placement_for(buffers) if buffers else None,
            )
            pin_launch(solver, blocksize, dynshared)
            entries[label] = solver
        stats = time_group_gated(
            records, system_name, algo_name, entries, inits, params,
            spec["duration"],
        )
        for label, buffers, blocksize, dynshared in chunk:
            geometry = launch_geometry(
                entries[label], blocksize, spec["n_runs"], dynshared
            )
            records.append(
                dict(
                    key=task_key("padded", system_name, algo_name, label),
                    task="padded", phase=phase, status="ok",
                    system=system_name, algo=algo_name, buffers=buffers,
                    requested_blocksize=blocksize,
                    requested_dynshared=dynshared,
                    geometry=geometry, base_geometry=natural,
                    twin_ratio=stats["baseline (twin)"]["ratio_median"],
                    **stats[label],
                )
            )
            print(
                f"  {system_name}/{algo_name} {label}: T="
                f"{geometry['resident_threads']}, ratio "
                f"{stats[label]['ratio_median']:.3f}", flush=True,
            )
        for label, solver in entries.items():
            if label != "baseline":
                solver.close()
    base.close()


TASKS = dict(
    features=features_task, singles=singles_task, pairs=pairs_task,
    geometry=geometry_task, blocksize=blocksize_task, waves=waves_task,
    padded=padded_task,
)


def open_records(out):
    out = Path(out)
    return Records(out / "records.jsonl", extra=[out / "compiles.jsonl"])


def run_task(args):
    """Run one task in this process (spawned by the driver)."""
    records = open_records(args.out)
    phase, kind, system_name, algo_name = args.task
    if kind == "singles":
        singles_task(records, phase, system_name, algo_name,
                     args.pool_wait)
    else:
        TASKS[kind](records, phase, system_name, algo_name)


# --- driver ------------------------------------------------------------


def child_env(out, codegen_tag):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["CUBIE_CACHE_DIR"] = str(Path(out) / "codegen" / codegen_tag)
    env["CUBIE_KERNEL_CACHE_DIR"] = str(Path(out) / "kernel_cache")
    env["CUBIE_LIVENESS_LOG"] = "1"
    return env


def task_done(records, kind, system_name, algo_name):
    if kind == "features" and records.has(
        task_key("features", system_name, algo_name)
    ):
        return True
    if records.has(task_key("taskdone", system_name, algo_name, kind)):
        return True
    errors = records.select(
        task="taskerror", system=system_name, algo=algo_name
    )
    return len(errors) >= 2 or any(
        e["kind"] == kind for e in errors
    )


def drive(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = open_records(out)
    tasks = [
        t for t in task_list()
        if (args.phase is None or t[0] in args.phase)
        and (args.only is None or f"{t[2]}/{t[3]}" in args.only)
    ]
    print(f"{len(tasks)} tasks", flush=True)
    for phase, kind, system_name, algo_name in tasks:
        records.reload()
        if task_done(records, kind, system_name, algo_name):
            continue
        label = f"[{phase}] {kind} {system_name}/{algo_name}"
        print(f"{label} ...", flush=True)
        start = time.perf_counter()
        codegen_tag = (
            f"features_{system_name}_{algo_name}"
            if kind == "features" else "driver"
        )
        env = child_env(out, codegen_tag)
        if kind == "features":
            env["CUBIE_KERNEL_CACHE_DIR"] = str(
                out / "kernel_cache_features" / f"{system_name}_{algo_name}"
            )
        command = [
            sys.executable, "-u", str(BENCH), "--task", phase, kind,
            system_name, algo_name, "--out", str(out),
            "--pool-wait", str(args.pool_wait),
        ]
        log_path = out / "logs" / f"{kind}_{system_name}_{algo_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.task_timeout,
                )
                status = "ok" if process.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
        elapsed = time.perf_counter() - start
        records.reload()
        if status == "ok" and kind != "features":
            records.append(
                dict(
                    key=task_key("taskdone", system_name, algo_name, kind),
                    task="taskdone", phase=phase, kind=kind,
                    system=system_name, algo=algo_name, status=status,
                    elapsed_s=round(elapsed, 1),
                )
            )
        elif status != "ok":
            records.append(
                dict(
                    key=task_key("taskerror", system_name, algo_name,
                                 f"{kind}:{time.time():.0f}"),
                    task="taskerror", phase=phase, kind=kind,
                    system=system_name, algo=algo_name, status=status,
                    elapsed_s=round(elapsed, 1),
                )
            )
        tail = ""
        if status != "ok":
            tail = log_path.read_text(encoding="utf-8")[-800:]
        print(f"{label}: {status} in {elapsed:.0f} s {tail}", flush=True)
    print("DRIVER DONE", flush=True)


def pool(args):
    """Compile baseline and single-buffer variants ahead of the driver."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(child_env(out, "pool"))
    records = Records(out / "compiles.jsonl", extra=[out / "records.jsonl"])
    configs = [
        c for c in config_list()
        if (args.phase is None or c[0] in args.phase)
        and (args.only is None or f"{c[1]}/{c[2]}" in args.only)
    ]
    for phase, system_name, algo_name in configs:
        records.reload()
        spec = SYSTEMS[system_name]
        baseline = dict(system=system_name, algo=algo_name, buffers=[])
        if not records.has(task_key("compile", system_name, algo_name, "")):
            print(f"[pool] baseline {system_name}/{algo_name}", flush=True)
            compile_jobs(records, [baseline], workers=1, phase=phase)
        system = spec["build"]()
        try:
            solver = make_solver(system, system_name, algo_name)
            inits, params = spec["grid"](solver, 256)
            solver.compile(inits, params, duration=spec["duration"])
            candidates = candidate_buffers(solver)
            solver.close()
        except Exception as exc:
            print(f"[pool] {system_name}/{algo_name} failed: {exc}",
                  flush=True)
            continue
        jobs = [
            dict(system=system_name, algo=algo_name, buffers=[c["name"]])
            for c in candidates
        ]
        print(
            f"[pool] {system_name}/{algo_name}: {len(jobs)} singles",
            flush=True,
        )
        compile_jobs(records, jobs, workers=args.workers, phase=phase)
    print("POOL DONE", flush=True)


def retake(args):
    """Drop the listed row keys and their task markers from records."""
    out = Path(args.out)
    path = out / "records.jsonl"
    with open(args.retake, encoding="utf-8") as handle:
        keys = {line.strip() for line in handle if line.strip()}
    kinds = {"single": "singles", "pair": "pairs", "geometry": "geometry",
             "blocksize": "blocksize", "waves": "waves", "padded": "padded"}
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    markers = {
        task_key("taskdone", row["system"], row["algo"], kinds[row["task"]])
        for row in rows
        if row["key"] in keys and row["task"] in kinds
    }
    kept = [r for r in rows if r["key"] not in keys | markers]
    backup = out / f"records.jsonl.pre_retake_{int(time.time())}"
    path.replace(backup)
    with open(path, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, default=_json_default) + "\n")
    print(
        f"dropped {len(rows) - len(kept)} rows ({len(markers)} task "
        f"markers); backup {backup}"
    )


# --- report ------------------------------------------------------------


def report(args):
    out = Path(args.out)
    records = open_records(out)
    lines = ["# Placement landscape summary", ""]
    features = records.select(task="features")
    lines.append("## Kernels")
    lines.append("")
    lines.append(
        "| system | algo | n | stages | width | regs | local B | spill st/ld"
        " | shared B/run | waves | steps | newton | krylov |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in features:
        dyn = row.get("dynamics", {})
        lines.append(
            f"| {row['system']} | {row['algo']} | {row['n_states']} | "
            f"{row['stage_count']} | {row['solver_width']} | {row['regs']}"
            f" | {row['local_bytes']} | {row['spill_store_bytes']}/"
            f"{row['spill_load_bytes']} | {row['shared_bytes_per_run']} | "
            f"{row['geometry']['waves']:.1f} | {dyn.get('steps', 0):.1f} |"
            f" {dyn.get('newton_iters', 0):.1f} | "
            f"{dyn.get('krylov_iters', 0):.1f} |"
        )
    lines.append("")
    lines.append("## Singles")
    lines.append("")
    lines.append(
        "| system | algo | buffer | B/run | Δlocal | Δregs | Δspill st/ld |"
        " bs | ratio med | ratio min | IQR | twin | maxdiff | fails |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in sorted(
        records.select(task="single"),
        key=lambda r: (r["system"], r["algo"],
                       r.get("ratio_median", 9.0)),
    ):
        cand = row.get("candidate", {})
        if row["status"] != "ok":
            lines.append(
                f"| {row['system']} | {row['algo']} | {row['buffers'][0]} |"
                f" {cand.get('elements', 0) * cand.get('itemsize', 0)} | "
                f"{row.get('delta_local', '')} | {row.get('delta_regs', '')}"
                f" | | | {row['status']} | | | | | |"
            )
            continue
        base_spill = row.get("base_spill") or [None, None]
        spill = (
            f"{(row.get('spill_store_bytes') or 0) - (base_spill[0] or 0):+d}"
            f"/{(row.get('spill_load_bytes') or 0) - (base_spill[1] or 0):+d}"
            if row.get("spill_store_bytes") is not None
            and base_spill[0] is not None else ""
        )
        lines.append(
            f"| {row['system']} | {row['algo']} | {row['buffers'][0]} | "
            f"{cand['elements'] * cand['itemsize']} | "
            f"{row['delta_local']:+d} | {row['delta_regs']:+d} | {spill} | "
            f"{row['geometry']['blocksize']} | {row['ratio_median']:.3f} | "
            f"{row['ratio_min']:.3f} | {row['ratio_iqr']:.3f} | "
            f"{row['twin_ratio']:.3f} | {row['max_abs_diff']:.2e} | "
            f"{row['status_hist']['failed']} |"
        )
    for task in ("pair", "geometry", "blocksize", "waves", "padded"):
        rows = records.select(task=task)
        if not rows:
            continue
        lines.append("")
        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            "| system | algo | variant | T | bs | dynshared | ratio med |"
            " ratio min | IQR | twin | maxdiff |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            geometry = row.get("geometry", {})
            lines.append(
                f"| {row['system']} | {row['algo']} | "
                f"{row['key'].split('|')[-1]} | "
                f"{geometry.get('resident_threads', '')} | "
                f"{geometry.get('blocksize', '')} | "
                f"{geometry.get('dynshared', '')} | "
                f"{row['ratio_median']:.3f} | {row['ratio_min']:.3f} | "
                f"{row['ratio_iqr']:.3f} | {row['twin_ratio']:.3f} | "
                f"{row['max_abs_diff']:.2e} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


# --- entry -------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--task", nargs=4, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--phase", nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None,
                        help="system/algo labels to include")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--pool-wait", type=float, default=900.0,
                        help="seconds to wait for pool compiles")
    parser.add_argument("--task-timeout", type=float, default=5400.0)
    parser.add_argument("--retake", default=None,
                        help="file of row keys to drop before running")
    args = parser.parse_args(argv)
    if args.worker:
        worker_main()
    elif args.retake:
        retake(args)
    elif args.task:
        warnings.simplefilter("ignore")
        run_task(args)
    elif args.list:
        for task in task_list():
            print(task)
    elif args.pool:
        pool(args)
    elif args.report:
        report(args)
    elif args.run:
        drive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
