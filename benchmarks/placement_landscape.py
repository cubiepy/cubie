#!/usr/bin/env python
"""Per-buffer shared-placement time bank: one row per solve (GPU)."""

import argparse
import ast
import contextlib
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from importlib import util as importlib_util
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve()
OUT_DEFAULT = Path(
    r"C:\local_working_projects\cubie-notes\placement_landscape\post866"
)
FABBRI_CELLML = (
    REPO / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)
NVDISASM = os.environ.get(
    "NVDISASM",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
    r"\nvdisasm.exe",
)
NCU_PYTHON = os.environ.get(
    "NCU_PYTHON",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1"
    r"\extras\python",
)
PRECISION = np.float32
BLOCKSIZE = 64
BLOCKSIZES = (32, 64, 128, 256)
ROUNDS = 2
REPEATS = 3
MIN_SOLVE_MS = 20.0
MAX_DURATION_SCALE = 512
PROBE_SCALES = (4096, 1024, 64)
SOLVE_BUDGET_S = 10.0
SETTLE_S = 1.0
WORKERS = 4
WIN_RATIO = 0.95
PAIR_WINNERS = 3
COMPILE_TIMEOUT = 1800.0
SMOKE = os.environ.get("PL_SMOKE", "") == "1"

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
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "hodgkin_huxley": dict(
        build=build_hodgkin_huxley, grid=grid_param("i_app", 5.0, 15.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "lorenz96_10": dict(
        build=lambda: build_lorenz96(10), grid=grid_param("F", 0.0, 16.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "diode_line": dict(
        build=build_diode_line, grid=grid_param("amp", 0.5, 1.5),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "chain20": dict(
        build=lambda: build_chain(20, 3), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "lorenz96_20": dict(
        build=lambda: build_lorenz96(20), grid=grid_param("F", 0.0, 16.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "chain32": dict(
        build=lambda: build_chain(32, 3), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "chain32_c8": dict(
        build=lambda: build_chain(32, 8), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "fabbri": dict(
        build=build_fabbri, grid=grid_fabbri,
        duration=0.2, n_runs=1 << 17,
        kwargs={"atol": 1e-6, "rtol": 1e-4, "dt_min": 1e-12,
                "dt_max": 1e-2},
        constants={"Rate_modulation_experiments_ANS": 1.0},
    ),
    "lorenz96_40": dict(
        build=lambda: build_lorenz96(40), grid=grid_param("F", 0.0, 16.0),
        duration=1.0, n_runs=1 << 18, kwargs=TIGHT,
    ),
    "chain64": dict(
        build=lambda: build_chain(64, 3), grid=grid_chain,
        duration=0.05, n_runs=1 << 18, kwargs=TIGHT,
    ),
}
if SMOKE:
    for _spec in SYSTEMS.values():
        _spec["n_runs"] = 4096

# --- algorithms --------------------------------------------------------

EXPLICIT_TABLEAUS = ("bogacki-shampine-32", "tsit5", "vern7")
NEWTON_TABLEAUS = (
    "l_stable_dirk_3", "kvaerno3", "kvaerno5", "radau_iia_3",
    "radau_iia_5",
)
ROSENBROCK_TABLEAUS = ("rosenbrock23", "ros3p", "rodas3p")
TABLEAUS = EXPLICIT_TABLEAUS + NEWTON_TABLEAUS + ROSENBROCK_TABLEAUS
BICGSTAB_TABLEAUS = ("kvaerno3", "radau_iia_5", "rosenbrock23")
LU_ONLY_SYSTEMS = ("fabbri",)
NEWTON_SETTINGS = dict(inexact_newton=True, prefactored=True)


def algorithm_kwargs(algo_name):
    """Solver kwargs for an algorithm label (``<tableau>[_bicgstab]``)."""
    tableau = algo_name.partition("_bicgstab")[0]
    kwargs = dict(algorithm=tableau)
    if tableau in EXPLICIT_TABLEAUS:
        return kwargs
    if algo_name.endswith("_bicgstab"):
        kwargs.update(
            linear_correction_type="bicgstab", preconditioner_type="jacobi",
        )
    else:
        kwargs["linear_correction_type"] = "lu"
    if tableau in NEWTON_TABLEAUS:
        kwargs.update(NEWTON_SETTINGS)
    return kwargs


def family(algo_name):
    tableau = algo_name.partition("_bicgstab")[0]
    if tableau in EXPLICIT_TABLEAUS:
        return "ERK"
    if tableau.startswith("radau"):
        return "FIRK"
    if tableau in NEWTON_TABLEAUS:
        return "DIRK"
    return "ROS"


def config_list():
    """Return ordered (system, algorithm) pairs."""
    configs = []
    for system_name in SYSTEMS:
        for tableau in TABLEAUS:
            configs.append((system_name, tableau))
            if (
                tableau in BICGSTAB_TABLEAUS
                and system_name not in LU_ONLY_SYSTEMS
            ):
                configs.append((system_name, f"{tableau}_bicgstab"))
    return configs


# --- buffers -----------------------------------------------------------

BUFFERS = (
    "state", "proposed_state", "error", "previous_step_size",
    "predictor_transform", "predictor_previous_values",
    "cached_auxiliaries", "accumulator", "stage_accumulator",
    "stage_store", "stage_base", "stage_rhs", "stage_increment",
    "stage_state", "lu_factor", "delta", "residual",
    "bicg_r0_hat", "bicg_p", "bicg_v", "bicg_tmp", "bicg_s_hat",
    "init_base", "init_delta", "init_increment", "init_residual",
)
SETTING_NAMES = {
    "bicg_r0_hat": "r0_hat_location",
    "bicg_p": "p_location",
    "bicg_v": "v_location",
    "bicg_tmp": "tmp_location",
    "bicg_s_hat": "s_hat_location",
}


def setting_name(buffer_name):
    return SETTING_NAMES.get(buffer_name, f"{buffer_name}_location")


def placement_for(buffers):
    return {setting_name(name): "shared" for name in buffers}


def task_key(kind, system, algo, variant=""):
    return f"{kind}|{system}|{algo}|{variant}"


def safe_name(key):
    return key.replace("|", "__").replace("+", "_and_") or "baseline"


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


def open_records(out):
    out = Path(out)
    return Records(out / "records.jsonl", extra=[out / "compiles.jsonl"])


def open_compiles(out):
    out = Path(out)
    return Records(out / "compiles.jsonl")


# --- solvers -----------------------------------------------------------


def solver_kwargs(system_name, algo_name):
    spec = SYSTEMS[system_name]
    kwargs = dict(spec["kwargs"])
    kwargs.update(algorithm_kwargs(algo_name))
    kwargs.update(
        output_types=["state"],
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


def kernel_ms(solver):
    return sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )


def solve_once(solver, inits, params, duration, blocksize=BLOCKSIZE,
               snapshot=True):
    """Solve; return ``(kernel_ms, wall_ms, snapshot)``."""
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(
            inits, params, duration=duration, blocksize=blocksize,
            grid_type="verbatim",
        )
    wall = (time.perf_counter() - start) * 1e3
    out = None
    if snapshot:
        out = dict(
            state_last=np.array(result.state[-1]),
            status_hist=status_histogram(result),
        )
        counters = result.iteration_counters
        if counters is not None:
            out["counters"] = np.array(counters)
    del result
    return kernel_ms(solver), wall, out


def kernel_resources(solver):
    dispatcher = solver.kernel.kernel
    regs = list(dispatcher.get_regs_per_thread().values())[0]
    local_bytes = list(dispatcher.get_local_mem_per_thread().values())[0]
    return int(regs), int(local_bytes)


def candidate_buffers(solver):
    """Return the sweep buffers this solver registers at size > 0."""
    from numpy import dtype as np_dtype

    from cubie.buffer_registry import buffer_registry

    out = []
    seen = set()
    for parent, group in buffer_registry._groups.items():
        config = getattr(parent, "compile_settings", None)
        for name in group.relocatable_names():
            entry = group.entries[name]
            if name not in BUFFERS or name in seen or entry.size <= 0:
                continue
            if not hasattr(config, setting_name(name)):
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


# --- occupancy ---------------------------------------------------------

_NCU = None


def ncu_calculator():
    """Nsight Compute occupancy calculator, or None when unavailable."""
    global _NCU
    if _NCU is None:
        try:
            if NCU_PYTHON not in sys.path:
                sys.path.append(NCU_PYTHON)
            import ncu_occupancy

            from cubie.cuda_simsafe import cuda

            major, minor = cuda.get_current_device().compute_capability
            _NCU = (ncu_occupancy, ncu_occupancy.OccupancyCalculator(
                major, minor
            ))
        except Exception:
            _NCU = False
    return _NCU or None


def bytes_per_run(solver):
    kernel = solver.kernel
    pad = 4 if kernel.shared_memory_needs_padding else 0
    return int(kernel.shared_memory_bytes + pad)


def launch_geometry(solver, blocksize, n_runs, dynshared=None):
    """Blocks/SM, resident threads, waves and limiter at one launch."""
    from cubie.cuda_simsafe import cuda, max_shared_memory_per_block

    kernel = solver.kernel
    (kern,) = kernel.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    per_run = bytes_per_run(solver)
    if dynshared is None:
        dynshared = int(per_run * min(n_runs, blocksize))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocksize, dynshared = kernel.limit_blocksize(
                blocksize, dynshared, per_run, n_runs
            )
    dynshared = max(4, dynshared)
    if dynshared > max_shared_memory_per_block():
        return None
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
    limiters = None
    ncu_blocks = None
    ncu = ncu_calculator()
    if ncu is not None:
        module, calculator = ncu
        regs = list(kernel.kernel.get_regs_per_thread().values())[0]
        carveout = int(
            getattr(device, "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", 102400)
        )
        params = module.OccupancyParameters(
            threads_per_block=blocksize, registers_per_thread=int(regs),
            shared_mem_per_block=int(dynshared), shared_mem_size=carveout,
        )
        try:
            limiters = [
                limiter.name
                for limiter in calculator.get_occupancy_limiters(params)
            ]
            ncu_blocks = int(
                calculator.get_resource_utilization(params)[
                    "allocated_blocks"
                ]
            )
        except Exception:
            limiters = None
    return dict(
        blocksize=int(blocksize),
        dynshared=int(dynshared),
        bytes_per_run=per_run,
        blocks_per_sm=blocks_per_sm,
        resident_threads=blocks_per_sm * blocksize,
        waves=n_runs / (resident * sms),
        limiters=limiters,
        ncu_blocks=ncu_blocks,
    )


def occupancy_table(solver, n_runs):
    """Geometry at every block size plus the production default."""
    table = {}
    for blocksize in BLOCKSIZES:
        geometry = launch_geometry(
            solver, blocksize, n_runs,
            dynshared=bytes_per_run(solver) * min(n_runs, blocksize),
        )
        if geometry is not None:
            table[str(blocksize)] = geometry
    default = launch_geometry(solver, BLOCKSIZE, n_runs)
    return dict(default=default, by_blocksize=table)


def launch_plans(occupancy, equal_t_rows=False):
    """Launch rows for one kernel: default, higher-T sizes, equal-T sizes."""
    default = occupancy["default"]
    plans = [("default", default["blocksize"], default["dynshared"])]
    for blocksize, geometry in occupancy["by_blocksize"].items():
        if geometry["blocksize"] == default["blocksize"]:
            continue
        higher = geometry["resident_threads"] > default["resident_threads"]
        equal = geometry["resident_threads"] == default["resident_threads"]
        if higher or (equal_t_rows and equal):
            plans.append(
                (f"bs{blocksize}", geometry["blocksize"],
                 geometry["dynshared"])
            )
    return plans


def pin_launch(solver, blocksize, dynshared):
    """Fix the launch geometry of one solver."""
    solver.kernel.limit_blocksize = (
        lambda bs, dyn, bpr, runs: (blocksize, dynshared)
    )


# --- compiles ----------------------------------------------------------

_SOURCE_HASH = None


def source_hash():
    """Package source hash the kernel cache is keyed on."""
    global _SOURCE_HASH
    if _SOURCE_HASH is None:
        from cubie._utils import package_source_hash

        _SOURCE_HASH = package_source_hash()
    return _SOURCE_HASH


def spill_helpers():
    sys.path.insert(0, str(REPO / "benchmarks"))
    import lorenz_mean_runtime as helpers

    return helpers


def persist_kernel(out, key, cubin):
    """Write the cubin and its gzipped SASS text under ``out``."""
    out = Path(out)
    name = safe_name(key)
    cubin_dir = out / "cubins"
    cubin_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = cubin_dir / f"{name}.cubin"
    cubin_path.write_bytes(cubin)
    sass_dir = out / "sass"
    sass_dir.mkdir(parents=True, exist_ok=True)
    sass_path = sass_dir / f"{name}.sass.gz"
    if not Path(NVDISASM).exists():
        return dict(cubin=str(cubin_path), sass=None)
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tmp:
        tmp.write(cubin)
        tmp_path = tmp.name
    try:
        text = subprocess.run(
            [NVDISASM, "-c", tmp_path], capture_output=True, text=True,
            check=True,
        ).stdout
    finally:
        os.unlink(tmp_path)
    with gzip.open(sass_path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return dict(cubin=str(cubin_path), sass=str(sass_path))


def compile_payload(solver, helpers, out, key, compile_s):
    """Resources, spills, occupancy and artefact paths of a compiled kernel."""
    regs, local_bytes = kernel_resources(solver)
    (kern,) = solver.kernel.kernel.overloads.values()
    cubin, entry_name = helpers._compiled_cubin(kern)
    log = helpers._link_diagnostics.get(hashlib.sha256(cubin).hexdigest())
    spill_store = spill_load = None
    if log is not None:
        spill_store, spill_load = helpers.parse_spill_diagnostics(
            log, entry_name
        )
    n_runs = SYSTEMS[key.split("|")[1]]["n_runs"]
    return dict(
        regs=regs, local_bytes=local_bytes,
        spill_store_bytes=spill_store, spill_load_bytes=spill_load,
        shared_bytes_per_run=bytes_per_run(solver),
        occupancy=occupancy_table(solver, n_runs),
        compile_s=round(compile_s, 2),
        cached=log is None,
        source_hash=source_hash(),
        artefacts=persist_kernel(out, key, cubin),
    )


def worker_main(out):
    """Compile one placement in this process and print its row."""
    helpers = spill_helpers()
    helpers.install_spill_capture()
    job = json.loads(sys.stdin.read())
    spec = SYSTEMS[job["system"]]
    system = spec["build"]()
    solver = make_solver(
        system, job["system"], job["algo"], placement_for(job["buffers"])
    )
    inits, params = spec["grid"](solver, 256)
    start = time.perf_counter()
    solver.compile(inits, params, duration=spec["duration"])
    compile_s = time.perf_counter() - start
    key = task_key(
        "compile", job["system"], job["algo"], "+".join(job["buffers"])
    )
    payload = compile_payload(solver, helpers, out, key, compile_s)
    print("@RESULT " + json.dumps(payload, default=_json_default),
          flush=True)


def compile_row(compiles, system_name, algo_name, buffers):
    return compiles.get(
        task_key("compile", system_name, algo_name, "+".join(buffers))
    )


def compiled(compiles, system_name, algo_name, buffers):
    row = compile_row(compiles, system_name, algo_name, buffers)
    return row is not None and row.get("source_hash") == source_hash()


def compile_jobs(compiles, jobs, out, workers=WORKERS):
    """Compile ``jobs`` (dicts: system, algo, buffers) in subprocesses."""
    pending = [
        job for job in jobs
        if not compiled(compiles, job["system"], job["algo"],
                        job["buffers"])
    ]
    running = []
    env = dict(os.environ)
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            process = subprocess.Popen(
                [sys.executable, str(BENCH), "--worker", "--out",
                 str(out)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env,
            )
            process.stdin.write(json.dumps(job))
            process.stdin.close()
            running.append((job, process, time.perf_counter()))
        still = []
        for job, process, started in running:
            key = task_key(
                "compile", job["system"], job["algo"],
                "+".join(job["buffers"])
            )
            if process.poll() is None:
                if time.perf_counter() - started > COMPILE_TIMEOUT:
                    process.kill()
                    compiles.append(
                        dict(key=key, task="compile", status="timeout",
                             source_hash=source_hash(), **job)
                    )
                    continue
                still.append((job, process, started))
                continue
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            payload = None
            for line in stdout.splitlines():
                if line.startswith("@RESULT "):
                    payload = json.loads(line[len("@RESULT "):])
            if payload is None:
                compiles.append(
                    dict(key=key, task="compile", status="error",
                         error=stderr[-3000:], source_hash=source_hash(),
                         **job)
                )
            else:
                compiles.append(
                    dict(key=key, task="compile", status="ok", **job,
                         **payload)
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


# --- features ----------------------------------------------------------


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


def dynamics(system, system_name, algo_name, inits, params, duration,
             marker=None, probes=()):
    """Per-run counter means from one counters-enabled solve of the grid."""
    solver = make_solver(
        system, system_name, algo_name,
        extra=dict(output_types=["state", "iteration_counters"]),
    )
    solver.compile(inits, params, duration=duration)
    if marker is not None:
        marker.write_text(
            json.dumps(dict(scale="dynamics", duration=duration,
                            probes=list(probes), start=time.time())),
            encoding="utf-8",
        )
    _, _, snapshot = solve_once(solver, inits, params, duration)
    if marker is not None:
        marker.unlink()
    totals = snapshot["counters"].sum(axis=0)
    names = ["newton_iters", "krylov_iters", "steps", "rejected_steps"]
    out = dict(
        {name: float(totals[i].mean()) for i, name in enumerate(names)
         if i < totals.shape[0]},
        status=snapshot["status_hist"],
    )
    solver.close()
    return out


def features_row(records, system_name, algo_name, system, solver,
                 codegen_s, first_solve_s, candidates, dyn, duration,
                 probes):
    from cubie.odesystems.symbolic.engine import assignments

    try:
        from cubie.backend import _typed_block_scheduler
        block_log = list(_typed_block_scheduler.BLOCK_LOG)
    except ImportError:
        block_log = []
    step = solver.kernel.single_integrator._algo_step
    config = step.compile_settings
    tableau = getattr(config, "tableau", None)
    n_states = solver.kernel.single_integrator._loop.compile_settings.n_states
    (kern,) = solver.kernel.kernel.overloads.values()
    metadata = getattr(getattr(kern, "cres", None), "metadata", {}) or {}
    records.append(
        dict(
            key=task_key("features", system_name, algo_name),
            task="features", system=system_name, algo=algo_name,
            family=family(algo_name),
            n_states=int(n_states),
            stage_count=(tableau.stage_count if tableau is not None else 1),
            solver_width=int(getattr(config, "solver_width", n_states)),
            is_implicit=bool(step.is_implicit),
            algorithm_settings={
                k: v for k, v in solver_kwargs(system_name, algo_name).items()
                if k != "output_types"
            },
            n_runs=SYSTEMS[system_name]["n_runs"],
            duration=duration,
            probes=probes,
            candidates=candidates,
            codegen_s=round(codegen_s, 2),
            first_solve_s=round(first_solve_s, 2),
            liveness=list(assignments.LIVENESS_LOG),
            block_liveness=block_log,
            scheduler_stats=metadata.get("typed_block_scheduler"),
            op_counts=op_counts(system.gen_file.file_path),
            dynamics=dyn,
        )
    )


# --- banking -----------------------------------------------------------


def bank_wave(records, system_name, algo_name, wave, entries, inits,
              params, duration, n_runs):
    """Bank a warm solve then ROUNDS x REPEATS solves per launch row."""
    reference = None
    for label, buffers, solver, blocksize, dynshared in entries:
        pin_launch(solver, blocksize, dynshared)
        ms, wall, snapshot = solve_once(
            solver, inits, params, duration, blocksize
        )
        if reference is None:
            reference = snapshot
        check = compare_outputs(reference, snapshot)
        geometry = launch_geometry(solver, blocksize, n_runs, dynshared)
        records.append(
            dict(
                key=task_key("solve", system_name, algo_name,
                             f"{label}|{wave}|warm"),
                task="solve", system=system_name, algo=algo_name,
                wave=wave, label=label, buffers=buffers, warm=True,
                round=-1, rep=-1, kernel_ms=round(ms, 4),
                wall_ms=round(wall, 3), n_runs=n_runs, duration=duration,
                geometry=geometry, status_hist=snapshot["status_hist"],
                **check,
            )
        )
        del snapshot
    # Untimed solves until SETTLE_S has passed since the snapshot phase.
    label, buffers, solver, blocksize, dynshared = entries[0]
    pin_launch(solver, blocksize, dynshared)
    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < SETTLE_S:
        solve_once(solver, inits, params, duration, blocksize,
                   snapshot=False)
    for round_idx in range(ROUNDS):
        for label, buffers, solver, blocksize, dynshared in entries:
            pin_launch(solver, blocksize, dynshared)
            for rep in range(REPEATS):
                ms, wall, _ = solve_once(
                    solver, inits, params, duration, blocksize,
                    snapshot=False,
                )
                records.append(
                    dict(
                        key=task_key(
                            "solve", system_name, algo_name,
                            f"{label}|{wave}|r{round_idx}k{rep}",
                        ),
                        task="solve", system=system_name, algo=algo_name,
                        wave=wave, label=label, buffers=buffers,
                        warm=False, round=round_idx, rep=rep,
                        kernel_ms=round(ms, 4), wall_ms=round(wall, 3),
                        n_runs=n_runs, duration=duration,
                        blocksize=blocksize, dynshared=dynshared,
                    )
                )
        print(f"  wave {wave} round {round_idx} done", flush=True)


def kernel_entries(system, system_name, algo_name, buffers_list,
                   compiles, equal_t_for_baseline):
    """Build (label, buffers, solver, bs, dynshared) for compiled kernels."""
    entries = []
    for buffers in buffers_list:
        row = compile_row(compiles, system_name, algo_name, buffers)
        if row is None or row.get("status") != "ok":
            continue
        solver = make_solver(
            system, system_name, algo_name, placement_for(buffers)
        )
        base_label = "+".join(buffers) or "baseline"
        equal_rows = equal_t_for_baseline and not buffers
        for plan, blocksize, dynshared in launch_plans(
            row["occupancy"], equal_t_rows=equal_rows
        ):
            label = base_label if plan == "default" else (
                f"{base_label}@{plan}"
            )
            entries.append((label, buffers, solver, blocksize, dynshared))
    return entries


def close_entries(entries):
    """Close each distinct solver behind the launch rows once."""
    seen = set()
    for entry in entries:
        solver = entry[2]
        if id(solver) not in seen:
            seen.add(id(solver))
            solver.close()


def kernel_medians(records, system_name, algo_name, wave):
    """Median kernel ms per label over the timed solves of one wave."""
    samples = {}
    for row in records.select(
        task="solve", system=system_name, algo=algo_name, wave=wave,
        warm=False,
    ):
        samples.setdefault(row["label"], []).append(row["kernel_ms"])
    return {label: float(np.median(v)) for label, v in samples.items()}


def sweep_duration(records, system_name, algo_name):
    """Duration recorded in the features row, else the spec value."""
    row = records.get(task_key("features", system_name, algo_name))
    spec = SYSTEMS[system_name]
    if row is not None:
        return float(row.get("duration", spec["duration"]))
    return spec["duration"]


def run_config(out, system_name, algo_name, workers):
    """Compile, bank singles, then bank pairs for one configuration."""
    out = Path(out)
    records = open_records(out)
    compiles = open_compiles(out)
    helpers = spill_helpers()
    helpers.install_spill_capture()
    spec = SYSTEMS[system_name]
    n_runs = spec["n_runs"]
    duration = sweep_duration(records, system_name, algo_name)

    start = time.perf_counter()
    system = spec["build"]()
    codegen_s = time.perf_counter() - start
    base = make_solver(system, system_name, algo_name)
    inits, params = spec["grid"](base, n_runs)
    base_key = task_key("compile", system_name, algo_name, "")
    start = time.perf_counter()
    base.compile(inits, params, duration=duration)
    compile_s = time.perf_counter() - start
    if not compiled(compiles, system_name, algo_name, []):
        compiles.append(
            dict(key=base_key, task="compile", status="ok",
                 system=system_name, algo=algo_name, buffers=[],
                 **compile_payload(base, helpers, out, base_key,
                                   compile_s))
        )
    ramped = duration == spec["duration"]
    probes = []
    first_solve_s = None
    marker = probe_marker(out, system_name, algo_name)

    def guarded(solver, dur, scale, snapshot):
        # The driver kills the config when the marker outlives its budget.
        marker.write_text(
            json.dumps(dict(scale=scale, duration=dur, probes=probes,
                            start=time.time())),
            encoding="utf-8",
        )
        result = solve_once(solver, inits, params, dur, snapshot=snapshot)
        marker.unlink()
        return result

    if ramped:
        for scale in PROBE_SCALES:
            duration = spec["duration"] / scale
            start = time.perf_counter()
            _, _, probe = guarded(base, duration, scale, True)
            solve_s = time.perf_counter() - start
            if first_solve_s is None:
                first_solve_s = solve_s
            probes.append(dict(scale=scale, duration=duration,
                               solve_s=round(solve_s, 2),
                               status_hist=probe["status_hist"]))
            del probe
    else:
        start = time.perf_counter()
        guarded(base, duration, None, False)
        first_solve_s = time.perf_counter() - start
    ms, _, _ = guarded(base, duration, None, False)
    # Double the duration until a settled solve reaches MIN_SOLVE_MS.
    while ramped and ms < MIN_SOLVE_MS and duration < spec["duration"] * (
        MAX_DURATION_SCALE
    ):
        duration *= 2
        ms, _, _ = guarded(base, duration, None, False)
    candidates = candidate_buffers(base)
    base.close()
    print(
        f"  baseline compiled in {compile_s:.0f} s; "
        f"{len(candidates)} candidates", flush=True,
    )

    if not records.has(task_key("features", system_name, algo_name)):
        dyn = dynamics(system, system_name, algo_name, inits, params,
                       duration, marker=marker, probes=probes)
        probe = make_solver(system, system_name, algo_name)
        probe.compile(inits, params, duration=duration)
        features_row(records, system_name, algo_name, system, probe,
                     codegen_s, first_solve_s, candidates, dyn, duration,
                     probes)
        probe.close()

    singles = [[c["name"]] for c in candidates]
    compile_jobs(
        compiles,
        [dict(system=system_name, algo=algo_name, buffers=b)
         for b in singles],
        out, workers=workers,
    )
    compiles.reload()

    if not records.has(task_key("wavedone", system_name, algo_name,
                                "singles")):
        entries = kernel_entries(
            system, system_name, algo_name, [[]] + singles, compiles,
            equal_t_for_baseline=True,
        )
        print(f"  singles wave: {len(entries)} launch rows", flush=True)
        bank_wave(records, system_name, algo_name, "singles", entries,
                  inits, params, duration, n_runs)
        close_entries(entries)
        records.append(
            dict(key=task_key("wavedone", system_name, algo_name,
                              "singles"),
                 task="wavedone", system=system_name, algo=algo_name,
                 wave="singles")
        )

    medians = kernel_medians(records, system_name, algo_name, "singles")
    base_ms = medians.get("baseline")
    winners = sorted(
        [
            (medians[s[0]] / base_ms, s[0]) for s in singles
            if s[0] in medians and base_ms
            and medians[s[0]] / base_ms <= WIN_RATIO
        ]
    )[:PAIR_WINNERS]
    pairs = [
        sorted([a, b]) for (_, a), (_, b) in combinations(winners, 2)
    ]
    if pairs and not records.has(
        task_key("wavedone", system_name, algo_name, "pairs")
    ):
        compile_jobs(
            compiles,
            [dict(system=system_name, algo=algo_name, buffers=p)
             for p in pairs],
            out, workers=workers,
        )
        compiles.reload()
        entries = kernel_entries(
            system, system_name, algo_name, [[]] + pairs, compiles,
            equal_t_for_baseline=False,
        )
        print(f"  pairs wave: {len(entries)} launch rows", flush=True)
        bank_wave(records, system_name, algo_name, "pairs", entries,
                  inits, params, duration, n_runs)
        close_entries(entries)
        records.append(
            dict(key=task_key("wavedone", system_name, algo_name, "pairs"),
                 task="wavedone", system=system_name, algo=algo_name,
                 wave="pairs", pairs=pairs)
        )
    records.append(
        dict(key=task_key("configdone", system_name, algo_name),
             task="configdone", system=system_name, algo=algo_name)
    )


# --- driver ------------------------------------------------------------


def child_env(out):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["CUBIE_CACHE_DIR"] = str(Path(out) / "codegen")
    env["CUBIE_KERNEL_CACHE_DIR"] = str(Path(out) / "kernel_cache")
    env["CUBIE_LIVENESS_LOG"] = "1"
    return env


def probe_marker(out, system_name, algo_name):
    """Path of the file a config writes while a probe solve runs."""
    path = Path(out) / "logs" / f"{system_name}_{algo_name}.probe"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_child(command, env, log, timeout, marker):
    """Run a config; kill it when a probe outlives SOLVE_BUDGET_S."""
    process = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT
    )
    started = time.perf_counter()
    while process.poll() is None:
        if marker.exists():
            try:
                info = json.loads(marker.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                info = None
            if info and time.time() - info["start"] > SOLVE_BUDGET_S:
                kill_tree(process)
                marker.unlink(missing_ok=True)
                return "skipped", info
        if time.perf_counter() - started > timeout:
            kill_tree(process)
            return "timeout", None
        time.sleep(1.0)
    return ("ok" if process.returncode == 0 else "error"), None


def kill_tree(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
        )
    else:
        process.kill()
    process.wait()


def selected_configs(args):
    return [
        c for c in config_list()
        if args.only is None or f"{c[0]}/{c[1]}" in args.only
    ]


def drive(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = open_records(out)
    configs = selected_configs(args)
    print(f"{len(configs)} configs", flush=True)
    for system_name, algo_name in configs:
        records.reload()
        if records.has(task_key("configdone", system_name, algo_name)):
            continue
        if records.has(task_key("configskip", system_name, algo_name)):
            continue
        errors = records.select(
            task="configerror", system=system_name, algo=algo_name
        )
        if len(errors) >= 2:
            continue
        label = f"{system_name}/{algo_name}"
        print(f"{label} ...", flush=True)
        start = time.perf_counter()
        command = [
            sys.executable, "-u", str(BENCH), "--config", system_name,
            algo_name, "--out", str(out), "--workers", str(args.workers),
        ]
        log_path = out / "logs" / f"{system_name}_{algo_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        marker = probe_marker(out, system_name, algo_name)
        marker.unlink(missing_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            status, info = run_child(
                command, child_env(out), log, args.config_timeout, marker
            )
        elapsed = time.perf_counter() - start
        records.reload()
        if status == "skipped":
            records.append(
                dict(
                    key=task_key("configskip", system_name, algo_name),
                    task="configskip", system=system_name, algo=algo_name,
                    scale=info["scale"], duration=info["duration"],
                    probes=info["probes"], budget_s=SOLVE_BUDGET_S,
                )
            )
        elif status != "ok":
            records.append(
                dict(
                    key=task_key("configerror", system_name, algo_name,
                                 f"{time.time():.0f}"),
                    task="configerror", system=system_name,
                    algo=algo_name, status=status,
                    elapsed_s=round(elapsed, 1),
                )
            )
        tail = ""
        if status != "ok":
            tail = log_path.read_text(encoding="utf-8")[-800:]
        print(f"{label}: {status} in {elapsed:.0f} s {tail}", flush=True)
    print("DRIVER DONE", flush=True)


def drop(args):
    """Remove every row of the listed configs (backup kept)."""
    out = Path(args.out)
    targets = set(args.drop)
    for name in ("records.jsonl", "compiles.jsonl"):
        path = out / name
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")
                if l.strip()]
        kept = [
            r for r in rows
            if f"{r.get('system')}/{r.get('algo')}" not in targets
        ]
        backup = out / f"{name}.pre_drop_{int(time.time())}"
        path.replace(backup)
        with open(path, "w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, default=_json_default) + "\n")
        print(f"{name}: dropped {len(rows) - len(kept)} rows; "
              f"backup {backup}")


# --- report ------------------------------------------------------------


def report(args):
    out = Path(args.out)
    records = open_records(out)
    compiles = open_compiles(out)
    lines = ["# Placement time bank", ""]
    for system_name, algo_name in config_list():
        solves = records.select(
            task="solve", system=system_name, algo=algo_name, warm=False
        )
        if not solves:
            continue
        warm = {
            row["label"]: row for row in records.select(
                task="solve", system=system_name, algo=algo_name,
                warm=True,
            )
        }
        samples = {}
        for row in solves:
            samples.setdefault((row["wave"], row["label"]), []).append(
                row["kernel_ms"]
            )
        base = {
            wave: np.median(v) for (wave, label), v in samples.items()
            if label == "baseline"
        }
        lines.append(f"## {system_name} / {algo_name}")
        lines.append("")
        lines.append(
            "| wave | kernel | bs | T | regs | local B | spill st/ld |"
            " median ms | min ms | spread | ratio | maxdiff | fails |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (wave, label), values in sorted(samples.items()):
            arr = np.asarray(values)
            first = warm.get(label, {})
            geometry = first.get("geometry") or {}
            buffers = first.get("buffers") or []
            row = compile_row(compiles, system_name, algo_name, buffers)
            row = row or {}
            ratio = (
                f"{np.median(arr) / base[wave]:.3f}" if wave in base
                else ""
            )
            lines.append(
                f"| {wave} | {label} | {geometry.get('blocksize', '')} | "
                f"{geometry.get('resident_threads', '')} | "
                f"{row.get('regs', '')} | {row.get('local_bytes', '')} | "
                f"{row.get('spill_store_bytes', '')}/"
                f"{row.get('spill_load_bytes', '')} | "
                f"{np.median(arr):.3f} | {arr.min():.3f} | "
                f"{arr.max() / arr.min() - 1:.3f} | {ratio} | "
                f"{first.get('max_abs_diff', float('nan')):.2e} | "
                f"{(first.get('status_hist') or {}).get('failed', '')} |"
            )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


# --- entry -------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", nargs=2, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--only", nargs="+", default=None,
                        help="system/algo labels to include")
    parser.add_argument("--drop", nargs="+", default=None,
                        help="system/algo labels whose rows are removed")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--config-timeout", type=float, default=7200.0)
    args = parser.parse_args(argv)
    if args.worker:
        worker_main(args.out)
    elif args.drop:
        drop(args)
    elif args.config:
        warnings.simplefilter("ignore")
        run_config(args.out, args.config[0], args.config[1], args.workers)
    elif args.list:
        for config in config_list():
            print(config)
    elif args.report:
        report(args)
    elif args.run:
        drive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
