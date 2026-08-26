"""Per-buffer shared-memory placement chosen by measurement."""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from warnings import warn

from importlib.metadata import version as package_version

from attrs import define
from numpy import dtype as np_dtype, median as np_median, ndarray

from cubie.buffer_registry import buffer_registry
from cubie.cache_root import get_cache_root
from cubie.cuda_backend import CUDA_BACKEND
from cubie.cuda_simsafe import (
    CUDA_SIMULATION,
    compute_capability_code,
    cuda,
    max_shared_memory_per_block,
)
from cubie.time_logger import default_timelogger

WIN_RATIO = 0.95
"""Paired median ratio at or below which a placement counts as a win."""

MAX_TUNE_RUNS = 1 << 18
"""Largest batch slice used for timing."""

MIN_WAVES = 8
"""Waves below which the timing slice is reported as too small."""


@define(frozen=True)
class BufferCandidate:
    """One relocatable buffer and its shared-memory cost."""

    name: str
    owner: str
    elements: int
    itemsize: int

    @property
    def bytes_per_run(self) -> int:
        """Shared bytes one run adds when this buffer relocates."""
        return self.elements * self.itemsize

    @property
    def key(self) -> str:
        """The ``{name}_location`` setting that moves this buffer."""
        return f"{self.name}_location"


@define(frozen=True)
class KernelResources:
    """Compiled per-thread register and local-memory use."""

    regs: int
    local_bytes: int


@define
class PlacementTrial:
    """A placement with its compile and timing outcome."""

    names: Tuple[str, ...]
    resources: Optional[KernelResources] = None
    local_delta: Optional[int] = None
    ratio: Optional[float] = None
    error: Optional[str] = None

    @property
    def placement(self) -> Dict[str, str]:
        """Location settings that realise this trial."""
        return {f"{name}_location": "shared" for name in self.names}


@define
class TuneResult:
    """Outcome of one tuning run."""

    baseline: KernelResources
    candidates: List[BufferCandidate]
    trials: List[PlacementTrial]
    chosen: Dict[str, str]
    ratio: float
    runs: int
    waves: float
    cached: bool = False


def kernel_resources(solver) -> KernelResources:
    """Return the compiled kernel's registers and local bytes per thread."""
    dispatcher = solver.kernel.kernel
    regs = list(dispatcher.get_regs_per_thread().values())[0]
    local_bytes = list(dispatcher.get_local_mem_per_thread().values())[0]
    return KernelResources(regs=int(regs), local_bytes=int(local_bytes))


def candidate_buffers(solver) -> List[BufferCandidate]:
    """Return every nonempty buffer with a location setting, all-local."""
    shared_limit = max_shared_memory_per_block()
    current_shared = int(solver.kernel.shared_memory_bytes)
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
            candidate = BufferCandidate(
                name=name,
                owner=type(parent).__name__,
                elements=int(entry.size),
                itemsize=np_dtype(entry.dtype).itemsize,
            )
            if current_shared + candidate.bytes_per_run > shared_limit:
                continue
            seen.add(name)
            out.append(candidate)
    return out


def _tuning_key(solver) -> str:
    """Identity of the all-local kernel on this card and backend."""
    parts = (
        str(solver.kernel.config_hash),
        str(compute_capability_code()),
        CUDA_BACKEND,
        package_version("cubie"),
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _tuning_dir() -> Path:
    return Path(get_cache_root()) / "location_tuning"


def load_tuned_placement(solver) -> Optional[Dict[str, Any]]:
    """Return a persisted tuning record for this kernel, if any."""
    path = _tuning_dir() / f"{_tuning_key(solver)}.json"
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def save_tuned_placement(solver, record: Dict[str, Any]) -> Path:
    """Persist a tuning record for this kernel."""
    directory = _tuning_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_tuning_key(solver)}.json"
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return path


def _launch_geometry(solver, blocksize: int, runs: int) -> Tuple[int, float]:
    """Return (resident runs per SM, waves) for the compiled kernel."""
    kernel = solver.kernel
    (kern,) = kernel.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    pad = 4 if kernel.shared_memory_needs_padding else 0
    bytes_per_run = kernel.shared_memory_bytes + pad
    dynshared = int(bytes_per_run * min(runs, blocksize))
    actual_blocksize, dynshared = kernel.limit_blocksize(
        blocksize, dynshared, bytes_per_run, runs
    )
    dynshared = max(4, dynshared)
    context = cuda.current_context()
    blocks_per_sm = int(
        context.get_active_blocks_per_multiprocessor(
            cufunc, actual_blocksize, dynshared
        )
    )
    sms = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
    threads_per_run = solver.kernel.single_integrator.threads_per_step
    runs_per_block = max(1, actual_blocksize // threads_per_run)
    resident = max(1, blocks_per_sm * runs_per_block)
    return resident, runs / (resident * sms)


def _kernel_ms(solver) -> float:
    """Kernel-only CUDA-event time of the last solve, all chunks."""
    return sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )


def _worker_main() -> None:
    """Compile one placement variant in this process; report resources."""
    from cubie.batchsolving.solver import Solver

    job = pickle.load(sys.stdin.buffer)
    solver = Solver(job["system"], **job["kwargs"], auto_memory=False)
    for updates in job["updates"]:
        solver.update(updates)
    if job["placement"]:
        solver.update(job["placement"])
    solver.compile(
        job["inits"],
        job["params"],
        drivers=job["drivers"],
        duration=job["duration"],
        settling_time=job["settling_time"],
        t0=job["t0"],
        grid_type="verbatim",
    )
    resources = kernel_resources(solver)
    sys.stdout.write(
        json.dumps(
            {"regs": resources.regs, "local_bytes": resources.local_bytes}
        )
    )
    sys.stdout.flush()


def _run_worker(job: Dict[str, Any]) -> KernelResources:
    """Run one compile job in a child process and return its resources."""
    process = subprocess.run(
        [sys.executable, "-m", "cubie.batchsolving.location_tuning"],
        input=pickle.dumps(job),
        capture_output=True,
        env=dict(os.environ),
    )
    if process.returncode != 0:
        tail = process.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"placement compile worker failed:\n{tail}")
    payload = json.loads(process.stdout.decode().strip().splitlines()[-1])
    return KernelResources(
        regs=int(payload["regs"]), local_bytes=int(payload["local_bytes"])
    )


def _compile_trials(
    solver,
    trials: Sequence[PlacementTrial],
    job_base: Dict[str, Any],
    workers: int,
) -> None:
    """Fill each trial's compiled resources, using worker processes."""
    picklable = True
    try:
        pickle.dumps(dict(job_base, placement={}))
    except Exception:
        picklable = False
    if not picklable or workers <= 1 or CUDA_SIMULATION:
        for trial in trials:
            trial.resources = _compile_in_process(solver, trial, job_base)
        return

    def run(trial: PlacementTrial) -> None:
        job = dict(job_base, placement=trial.placement)
        try:
            trial.resources = _run_worker(job)
        except Exception as exc:
            trial.error = str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, trials))
    for trial in trials:
        if trial.resources is None:
            trial.error = None
            trial.resources = _compile_in_process(solver, trial, job_base)


def _variant_solver(solver, placement: Dict[str, str]):
    """Construct a solver with this placement from the recorded inputs."""
    from cubie.batchsolving.solver import Solver

    record = solver._construction_record
    variant = Solver(record["system"], **record["kwargs"], auto_memory=False)
    for updates in solver._update_record:
        variant.update(updates)
    if placement:
        variant.update(placement)
    return variant


def _compile_in_process(
    solver, trial: PlacementTrial, job_base: Dict[str, Any]
) -> KernelResources:
    """Compile a placement variant in this process and read resources."""
    variant = _variant_solver(solver, trial.placement)
    variant.compile(
        job_base["inits"],
        job_base["params"],
        drivers=job_base["drivers"],
        duration=job_base["duration"],
        settling_time=job_base["settling_time"],
        t0=job_base["t0"],
        grid_type="verbatim",
    )
    resources = kernel_resources(variant)
    variant.close()
    return resources


def _time_trials(
    solver,
    trials: Sequence[PlacementTrial],
    inits: ndarray,
    params: ndarray,
    solve_kwargs: Dict[str, Any],
    rounds: int,
) -> None:
    """Interleave solves of every trial against all-local; set ratios."""
    variants = {
        trial.names: _variant_solver(solver, trial.placement)
        for trial in trials
    }
    order = [None] + list(variants)
    times = {key: [] for key in order}

    def solve_once(key) -> None:
        target = solver if key is None else variants[key]
        target.solve(inits, params, **solve_kwargs)
        times[key].append(_kernel_ms(target))

    for key in order:
        solve_once(key)
    for key in order:
        times[key].clear()
    for _ in range(rounds):
        for key in order:
            solve_once(key)
        for key in reversed(order):
            solve_once(key)

    local = times[None]
    for trial in trials:
        paired = [
            variant / base
            for variant, base in zip(times[trial.names], local)
        ]
        trial.ratio = float(np_median(paired))
    for variant in variants.values():
        variant.close()


def tune_locations(
    solver,
    initial_values,
    parameters,
    drivers=None,
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    blocksize: int = 256,
    grid_type: str = "verbatim",
    workers: int = 4,
    rounds: int = 6,
    force: bool = False,
) -> TuneResult:
    """Measure per-buffer shared placements and apply the best one."""
    if CUDA_SIMULATION:
        raise RuntimeError(
            "Placement tuning measures kernel time on a GPU; it is not "
            "available under the CUDA simulator."
        )
    inits, params = solver.input_handler(
        states=initial_values, params=parameters, kind=grid_type
    )
    runs = min(int(inits.shape[1]), MAX_TUNE_RUNS)
    inits = inits[:, :runs]
    params = params[:, :runs]
    solve_kwargs = dict(
        drivers=drivers,
        duration=duration,
        settling_time=settling_time,
        t0=t0,
        blocksize=blocksize,
        grid_type="verbatim",
    )

    solver.compile(
        inits,
        params,
        drivers=drivers,
        duration=duration,
        settling_time=settling_time,
        t0=t0,
        grid_type="verbatim",
    )
    baseline = kernel_resources(solver)
    candidates = candidate_buffers(solver)
    _, waves = _launch_geometry(solver, blocksize, runs)
    if waves < MIN_WAVES:
        warn(
            f"Placement tuning is timing {runs} runs, {waves:.1f} waves "
            f"of the all-local kernel; ratios below {MIN_WAVES} waves "
            "carry wave-quantisation error. Tune on a larger batch.",
            UserWarning,
        )

    if not force:
        stored = load_tuned_placement(solver)
        if stored is not None:
            chosen = dict(stored["placement"])
            if chosen:
                solver._apply_placement(chosen)
            return TuneResult(
                baseline=baseline,
                candidates=candidates,
                trials=[],
                chosen=chosen,
                ratio=float(stored["ratio"]),
                runs=runs,
                waves=waves,
                cached=True,
            )

    record = solver._construction_record
    job_base = dict(
        system=record["system"],
        kwargs=record["kwargs"],
        updates=list(solver._update_record),
        inits=inits,
        params=params,
        drivers=drivers,
        duration=duration,
        settling_time=settling_time,
        t0=t0,
        placement={},
    )

    singles = [PlacementTrial(names=(c.name,)) for c in candidates]
    _log(
        f"Placement tuning: all-local kernel uses {baseline.regs} "
        f"registers and {baseline.local_bytes} B local; compiling "
        f"{len(singles)} single-buffer variants on {workers} workers."
    )
    _compile_trials(solver, singles, job_base, workers)
    for trial in singles:
        trial.local_delta = (
            trial.resources.local_bytes - baseline.local_bytes
        )
    timed = [t for t in singles if t.local_delta < 0]
    _log(
        f"Placement tuning: {len(timed)} variants shrink the local "
        f"frame; timing them over {rounds} interleaved rounds on "
        f"{runs} runs ({waves:.1f} waves)."
    )
    if timed:
        _time_trials(solver, timed, inits, params, solve_kwargs, rounds)
        _log_trials(timed)

    winners = [
        t for t in timed if t.ratio is not None and t.ratio <= WIN_RATIO
    ]
    pairs = [
        PlacementTrial(names=a.names + b.names)
        for a, b in combinations(winners, 2)
    ]
    if pairs:
        _log(f"Placement tuning: compiling and timing {len(pairs)} pairs.")
        _compile_trials(solver, pairs, job_base, workers)
        for trial in pairs:
            trial.local_delta = (
                trial.resources.local_bytes - baseline.local_bytes
            )
        _time_trials(solver, pairs, inits, params, solve_kwargs, rounds)
        _log_trials(pairs)

    trials = singles + pairs
    measured = [t for t in trials if t.ratio is not None]
    best = min(measured, key=lambda t: t.ratio, default=None)
    if best is not None and best.ratio <= WIN_RATIO:
        chosen = best.placement
        ratio = best.ratio
    else:
        chosen = {}
        ratio = 1.0

    save_tuned_placement(
        solver,
        dict(
            placement=chosen,
            ratio=ratio,
            runs=runs,
            waves=waves,
            baseline=dict(regs=baseline.regs, local_bytes=baseline.local_bytes),
            trials=[
                dict(
                    names=list(t.names),
                    regs=None if t.resources is None else t.resources.regs,
                    local_bytes=(
                        None if t.resources is None
                        else t.resources.local_bytes
                    ),
                    ratio=t.ratio,
                    error=t.error,
                )
                for t in trials
            ],
        ),
    )
    if chosen:
        solver._apply_placement(chosen)
        _log(
            f"Placement tuning: applying {sorted(chosen)} "
            f"(ratio {ratio:.3f} vs all-local)."
        )
    else:
        _log("Placement tuning: all-local is fastest; nothing applied.")
    return TuneResult(
        baseline=baseline,
        candidates=candidates,
        trials=trials,
        chosen=chosen,
        ratio=ratio,
        runs=runs,
        waves=waves,
    )


def _log(message: str) -> None:
    """Report tuning progress through the time logger."""
    default_timelogger.print_message(message, min_verbosity="default")


def _log_trials(trials: Sequence[PlacementTrial]) -> None:
    """Report each trial's local-frame change and timing ratio."""
    for trial in trials:
        _log(
            f"Placement tuning: {'+'.join(trial.names)} local "
            f"{trial.local_delta:+d} B, ratio {trial.ratio:.3f}"
        )


if __name__ == "__main__":
    _worker_main()
