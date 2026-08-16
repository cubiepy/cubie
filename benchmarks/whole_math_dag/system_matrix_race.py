"""ABBA-race codegen orderings x block-schedule policies per system."""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

CONFIGS = {
    "A": ("kahn", "source"),
    "B": ("dfs", "source"),
    "C": ("kahn", "anchor_dfs"),
    "D": ("dfs", "anchor_dfs"),
    "E": ("liveness_auto", "anchor_dfs"),
}

SYSTEMS = [
    "lorenz_fixed",
    "lorenz_adaptive",
    "chains_serial",
    "chains_wide",
    "diamonds",
    "shared_prefix",
]


def start_worker(system, key, workdir, pythonpath):
    ordering, policy = CONFIGS[key]
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    env["PYTHONHASHSEED"] = "0"
    env["CUBIE_CUDA_BACKEND"] = "mlir"
    env["CUBIE_CACHE_DIR"] = str(workdir / f"cache_{system}_{key}")
    env["CUBIE_BLOCK_SCHEDULE"] = policy
    state_path = workdir / f"state_{system}_{key}.npy"
    process = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "system_matrix_probe.py"),
            "--system",
            system,
            "--ordering",
            ordering,
            "--save-state",
            str(state_path),
            "--serve",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    return process, state_path


def read_report(process, system, key):
    for line in process.stdout:
        if line.startswith("@REPORT"):
            return json.loads(line[len("@REPORT "):])
    raise RuntimeError(f"worker {system}/{key} died before @REPORT")


def solve(process):
    process.stdin.write("run\n")
    process.stdin.flush()
    for line in process.stdout:
        if line.startswith("@SOLVE"):
            return float(line.split()[1])
    raise RuntimeError("worker died mid-run")


def block(process, solves, lowest):
    times = sorted(solve(process) for _ in range(solves))
    return statistics.mean(times[:lowest])


def race(base, variant, pairs, solves, lowest):
    deltas = []
    for pair in range(pairs):
        first, second = (
            (base, variant) if pair % 2 == 0 else (variant, base)
        )
        time.sleep(random.uniform(0.3, 1.2))
        one = block(first, solves, lowest)
        time.sleep(random.uniform(0.3, 1.2))
        two = block(second, solves, lowest)
        a_ms, b_ms = (one, two) if pair % 2 == 0 else (two, one)
        deltas.append((b_ms - a_ms) / a_ms * 100.0)
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="*", default=SYSTEMS)
    parser.add_argument("--pythonpath", default=None)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--solves", type=int, default=6)
    parser.add_argument("--lowest", type=int, default=4)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--null", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    if args.null:
        CONFIGS.clear()
        CONFIGS.update(
            {"A": ("kahn", "source"), "B": ("kahn", "source")}
        )
    if args.variants:
        keep = {"A", *args.variants}
        for key in [k for k in CONFIGS if k not in keep]:
            del CONFIGS[key]

    random.seed(1234)
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="sysmatrix_"))
    workdir.mkdir(parents=True, exist_ok=True)

    for system in args.systems:
        workers = {}
        states = {}
        for key in CONFIGS:
            workers[key], states[key] = start_worker(
                system, key, workdir, args.pythonpath
            )
        reports = {}
        for key in CONFIGS:
            reports[key] = read_report(workers[key], system, key)
        base_state = np.load(states["A"])
        scale = float(np.abs(base_state).max()) or 1.0
        print(f"\n=== {system} ===", flush=True)
        for key, report in reports.items():
            ordering, policy = CONFIGS[key]
            kernel = report["kernel"]
            diff = float(
                np.abs(np.load(states[key]) - base_state).max()
            )
            sched = kernel.get("typed_block_scheduler") or {}
            print(
                f"{key} {ordering:>5}+{policy:<10} "
                f"regs={kernel['regs_per_thread']:>3} "
                f"local={kernel['local_mem_per_thread']:>5} "
                f"finite={report['state_finite']} "
                f"bad_status={report['status_nonzero']} "
                f"relerr_vs_A={diff / scale:.2e} "
                f"moved={sched.get('moved_statements', 0)} "
                f"compile={report['compile_wall_seconds']:.0f}s",
                flush=True,
            )
        for key in [k for k in CONFIGS if k != "A"]:
            deltas = race(
                workers["A"],
                workers[key],
                args.pairs,
                args.solves,
                args.lowest,
            )
            print(
                f"{key} vs A: median {statistics.median(deltas):+.2f}%  "
                f"pairs [{', '.join(f'{d:+.2f}%' for d in deltas)}]",
                flush=True,
            )
        for key in CONFIGS:
            try:
                workers[key].stdin.write("quit\n")
                workers[key].stdin.flush()
            except OSError:
                pass
        for key in CONFIGS:
            workers[key].wait(timeout=60)


if __name__ == "__main__":
    main()
