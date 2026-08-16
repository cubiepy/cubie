"""ABBA-interleaved paired timing of two fabbri_probe configurations.

Each side is ``ordering fused policy cache_dir``. Both workers stay
resident; solves ping-pong ABBA with randomised idle gaps. Verdict:
median paired delta of per-block means of the lowest-k times.
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def start_worker(name, ordering, fused, policy, cache,
                 python=None, pythonpath=None):
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    env["PYTHONHASHSEED"] = "0"
    env["CUBIE_CUDA_BACKEND"] = "mlir"
    env["CUBIE_CACHE_DIR"] = cache
    env["CUBIE_BLOCK_SCHEDULE"] = policy
    process = subprocess.Popen(
        [
            python or sys.executable,
            str(HERE / "fabbri_probe.py"),
            "--ordering",
            ordering,
            "--fused",
            fused,
            "--serve",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    for line in process.stdout:
        if line.startswith("@REPORT"):
            report = json.loads(line[len("@REPORT "):])
            kernel = report["kernel"]
            print(
                f"{name}: local={kernel['local_mem_per_thread']} "
                f"regs={kernel['regs_per_thread']} "
                f"cubin={kernel['cubin_sha256'][:12]} "
                f"status_nonzero={report['warmup']['status_nonzero']} "
                f"finite={report['warmup']['state_finite']}",
                flush=True,
            )
            return process
    raise RuntimeError(f"worker {name} died before @REPORT")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a", nargs=4, metavar=("ORD", "FUSED", "POLICY", "CACHE")
    )
    parser.add_argument(
        "--b", nargs=4, metavar=("ORD", "FUSED", "POLICY", "CACHE")
    )
    parser.add_argument("--python", default=None)
    parser.add_argument("--pythonpath", default=None)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--solves", type=int, default=3)
    parser.add_argument("--lowest", type=int, default=2)
    args = parser.parse_args()

    random.seed(1234)
    worker_a = start_worker(
        "A", *args.a, python=args.python, pythonpath=args.pythonpath
    )
    worker_b = start_worker(
        "B", *args.b, python=args.python, pythonpath=args.pythonpath
    )
    deltas = []
    for pair in range(args.pairs):
        first, second = (
            (worker_a, worker_b)
            if pair % 2 == 0
            else (worker_b, worker_a)
        )
        time.sleep(random.uniform(0.3, 1.2))
        one = block(first, args.solves, args.lowest)
        time.sleep(random.uniform(0.3, 1.2))
        two = block(second, args.solves, args.lowest)
        a_ms, b_ms = (one, two) if pair % 2 == 0 else (two, one)
        delta = (b_ms - a_ms) / a_ms * 100.0
        deltas.append(delta)
        print(
            f"pair {pair}: A={a_ms:.1f} ms B={b_ms:.1f} ms "
            f"delta={delta:+.2f}%",
            flush=True,
        )
    for worker in (worker_a, worker_b):
        worker.stdin.write("quit\n")
        worker.stdin.flush()
    print(
        "median paired delta (B vs A): %+.2f%%  "
        "spread [%+.2f%%, %+.2f%%]"
        % (statistics.median(deltas), min(deltas), max(deltas))
    )


if __name__ == "__main__":
    main()
