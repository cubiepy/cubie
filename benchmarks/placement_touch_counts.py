#!/usr/bin/env python
"""Per-buffer access counts under the CUDA simulator (``--run``).

Rows append to ``touch.jsonl`` under ``--out``; re-runs skip done keys.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from importlib import util as importlib_util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve()
N_RUNS = 2
BLOCKSIZE = 32
MAX_EVENTS = 3_000_000
STEP_MARKER = "accept_step"
DURATIONS = {
    "lorenz": 0.1, "lorenz96_10": 0.1, "lorenz96_20": 0.1,
    "lorenz96_40": 0.1, "chain20": 0.02, "chain32": 0.02,
    "chain64": 0.02, "chain32_c8": 0.02, "hodgkin_huxley": 0.5,
    "diode_line": 0.1, "fabbri": 0.005,
}


def landscape():
    spec = importlib_util.spec_from_file_location(
        "placement_landscape", REPO / "benchmarks" / "placement_landscape.py"
    )
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- counting proxy ----------------------------------------------------


class Recorder:
    """Thread-local event streams plus global read/write totals."""

    def __init__(self):
        self.reads = defaultdict(int)
        self.writes = defaultdict(int)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.streams = {}

    def events(self):
        stream = getattr(self.local, "events", None)
        if stream is None:
            stream = []
            self.local.events = stream
            with self.lock:
                self.streams[threading.get_ident()] = stream
        return stream

    def read(self, name, index):
        with self.lock:
            self.reads[name] += 1
        stream = self.events()
        if len(stream) < MAX_EVENTS:
            stream.append((name, 0, index))

    def write(self, name, index):
        with self.lock:
            self.writes[name] += 1
        stream = self.events()
        if len(stream) < MAX_EVENTS:
            stream.append((name, 1, index))


RECORDER = Recorder()


def _index_key(index):
    try:
        return int(index)
    except (TypeError, ValueError):
        return str(index)


class CountingArray:
    """Array proxy counting element reads and writes under one name."""

    __slots__ = ("name", "array")

    def __init__(self, name, array):
        while isinstance(array, CountingArray):
            array = array.array
        self.name = name
        self.array = array

    def __getitem__(self, index):
        if isinstance(index, slice):
            return CountingArray(self.name, self.array[index])
        RECORDER.read(self.name, _index_key(index))
        return self.array[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            for i in range(*index.indices(len(self.array))):
                RECORDER.write(self.name, i)
        else:
            RECORDER.write(self.name, _index_key(index))
        self.array[index] = value

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        for i in range(len(self.array)):
            yield self[i]

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def size(self):
        return self.array.size

    @property
    def ndim(self):
        return self.array.ndim

    def view(self, dtype):
        return CountingArray(self.name, self.array.view(dtype))

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


def install_counting():
    from cubie.buffer_registry import BufferGroup

    original = BufferGroup.get_allocator

    def counting_get_allocator(self, name, zero=False, unroll=(True, None)):
        allocator = original(self, name, zero, unroll)

        def allocate(shared, persistent):
            return CountingArray(name, allocator(shared, persistent))

        return allocate

    BufferGroup.get_allocator = counting_get_allocator


# --- metrics -----------------------------------------------------------


def analyse_stream(events):
    """Per-buffer access statistics over one thread's event stream."""
    steps = sum(
        1 for name, kind, _ in events if name == STEP_MARKER and kind == 1
    )
    last_access = {}
    last_write = {}
    reuse = defaultdict(list)
    write_to_read = defaultdict(list)
    dead_writes = defaultdict(int)
    bursts = defaultdict(list)
    counts = defaultdict(lambda: [0, 0])
    previous_name = None
    burst = 0
    for position, (name, kind, index) in enumerate(events):
        key = (name, index)
        counts[name][kind] += 1
        if name == previous_name:
            burst += 1
        else:
            if previous_name is not None:
                bursts[previous_name].append(burst)
            previous_name = name
            burst = 1
        if key in last_access:
            reuse[name].append(position - last_access[key])
        if kind == 0 and key in last_write:
            write_to_read[name].append(position - last_write[key])
            del last_write[key]
        if kind == 1:
            if key in last_write:
                dead_writes[name] += 1
            last_write[key] = position
        last_access[key] = position
    if previous_name is not None:
        bursts[previous_name].append(burst)
    for (name, _), _ in last_write.items():
        dead_writes[name] += 1

    def stats(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return dict(
            n=int(arr.size), mean=float(arr.mean()),
            median=float(np.median(arr)), p90=float(np.percentile(arr, 90)),
            max=float(arr.max()),
        )

    out = {}
    for name, (reads, writes) in counts.items():
        out[name] = dict(
            reads=reads, writes=writes,
            reads_per_step=reads / steps if steps else None,
            writes_per_step=writes / steps if steps else None,
            reuse_distance=stats(reuse[name]),
            write_to_read=stats(write_to_read[name]),
            dead_writes=dead_writes[name],
            burst=stats(bursts[name]),
        )
    return steps, out


def run_config(system_name, algo_name, out):
    module = landscape()
    install_counting()
    spec = module.SYSTEMS[system_name]
    system = spec["build"]()
    duration = DURATIONS[system_name]
    solver = module.make_solver(
        system, system_name, algo_name,
        extra=dict(output_types=["state", "iteration_counters"],
                   save_every=duration),
    )
    inits, params = spec["grid"](solver, N_RUNS)
    start = time.perf_counter()
    _, _, snapshot = module.solve_once(
        solver, inits, params, duration, blocksize=BLOCKSIZE
    )
    elapsed = time.perf_counter() - start
    counters = snapshot["counters"].sum(axis=0)
    names = ["newton_iters", "krylov_iters", "steps", "rejected_steps"]
    dynamics = {
        name: float(counters[i].mean()) for i, name in enumerate(names)
        if i < counters.shape[0]
    }
    streams = [s for s in RECORDER.streams.values() if s]
    streams.sort(key=len, reverse=True)
    longest = streams[0] if streams else []
    steps, per_buffer = analyse_stream(longest)
    row = dict(
        key=f"touch|{system_name}|{algo_name}|",
        task="touch", system=system_name, algo=algo_name,
        n_runs=N_RUNS, duration=duration, elapsed_s=round(elapsed, 1),
        dynamics=dynamics, status_hist=snapshot["status_hist"],
        stream_steps=steps, stream_events=len(longest),
        truncated=len(longest) >= MAX_EVENTS,
        totals={
            name: dict(reads=RECORDER.reads[name],
                       writes=RECORDER.writes[name])
            for name in set(RECORDER.reads) | set(RECORDER.writes)
        },
        buffers=per_buffer,
        time=time.time(),
    )
    with open(Path(out) / "touch.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(
        f"  {system_name}/{algo_name}: {steps} steps in stream, "
        f"{len(longest)} events, {elapsed:.0f} s", flush=True,
    )


def done_keys(out):
    path = Path(out) / "touch.jsonl"
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def drive(args):
    module = landscape()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = [
        c for c in module.config_list()
        if args.only is None or f"{c[0]}/{c[1]}" in args.only
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["CUBIE_CUDA_BACKEND"] = "numba-cuda"
    env["CUBIE_CACHE_DIR"] = str(out / "codegen_touch")
    errors = defaultdict(int)
    for system_name, algo_name in configs:
        key = f"touch|{system_name}|{algo_name}|"
        if key in done_keys(out):
            continue
        print(f"touch {system_name}/{algo_name} ...", flush=True)
        log_path = out / "logs" / f"touch_{system_name}_{algo_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    [sys.executable, "-u", str(BENCH), "--config",
                     system_name, algo_name, "--out", str(out)],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                status = "ok" if process.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
        if status != "ok":
            errors[(system_name, algo_name)] += 1
            tail = log_path.read_text(encoding="utf-8")[-600:]
            print(f"  {status}: {tail}", flush=True)
    print("TOUCH DONE", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config", nargs=2, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--timeout", type=float, default=2400.0)
    args = parser.parse_args(argv)
    if args.config:
        run_config(args.config[0], args.config[1], args.out)
    elif args.run:
        drive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
