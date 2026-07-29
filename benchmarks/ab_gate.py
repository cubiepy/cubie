#!/usr/bin/env python
"""Block-interleaved A/B kernel-runtime gate: ``main`` vs the worktree.

For each installed CUDA backend the gate starts two persistent
``lorenz_mean_runtime.py --worker`` processes — A imports ``cubie``
from an ephemeral ``git worktree`` at ``--main`` (default
``origin/main``; removed afterwards), B from this repository — and
ping-pongs short solve blocks between them in a drift-balancing ABBA
order. Each block yields two statistics per config: the mean of its
lowest ``k`` per-solve kernel times (CUDA events, kernel only) and
the mean of its lowest ``k`` per-solve wall times (host clock around
``Solver.solve``). Host scatter is one-sided — delays only ever add
time — so the wall floor excludes it while still catching changes
that lengthen the end-to-end critical path or destroy chunk-transfer
overlap, which slow every solve. The two blocks of an ABBA
pair ran back to back, so each verdict statistic is the median of
the paired percent deltas — kernel deltas gate at ``--threshold``,
wall deltas at the coarser ``--wall-threshold``.

The workers also report compile metrics read from each config's
loaded cufunc and exact cubin link (``@META`` lines: registers, ptxas
spill-store/load byte counts, shared and constant memory, actual
launch geometry, occupancy in blocks/SM, run and chunk counts). The
gate prints an A-vs-B metrics table per backend and fails outright on
an occupancy decrease, a spill increase, or a chunk-count mismatch;
register deltas that leave occupancy and spill unchanged are
reported but not gated (their runtime effect, if any, is caught by
the timing rows).

Five configs run (see ``lorenz_mean_runtime.py``): ``fixed`` and
``adaptive`` as before; ``chunked``, whose small VRAM cap forces a
few run-axis chunks so chunk-path regressions surface in its wall
delta; ``wave``, sized by the gate to exactly two full waves of
side A's occupancy (``wave <n_runs>`` handshake after startup) so
any occupancy decrease on B forces a third wave and a step kernel
regression instead of hiding behind compute saturation; and
``host_overhead``, a constant eight trajectories whose
sub-millisecond wall statistic is the per-call host cost of
``Solver.solve``. Its wall delta gates in **absolute milliseconds**
(``--host-overhead-threshold``) — a fixed per-call cost is a
rounding error against the percent thresholds at the other configs'
sizes — and it reports no kernel row, since an eight-trajectory
kernel time is launch-dominated. Both workers run this repository's
benchmark script (only the ``cubie`` import differs per side); a
config announced by only one worker is skipped with a notice.

Why blocks: the floor of the kernel-time distribution tracks the
compiled kernel's intrinsic cost but wanders a few tenths of a
percent with GPU clock state. The blocks of a pair run seconds
apart, so they sample nearly the same clock state and the wander
cancels inside each pair — and each side pays its process startup,
JIT compile, and grid build once instead of once per sample. Verdict
reliability is judged from the per-pair deltas themselves: when at
least two pairs sit on each side of the regression threshold the
call is contested and the row is marked DISTRUST — rerun, with more
``--pairs`` or on a quieter GPU, before acting on it.
Constant background load inflates absolute times but cancels out of
the deltas. Both workers hold their device pools concurrently, which
is fine at the default sizes (~0.7 GB each); much larger ``--n-runs``
could change chunking between sides.

Usage::

    python benchmarks/ab_gate.py [--main REF] [--backends numba-cuda,mlir]
        [--pairs P] [--min-count K] [--threshold PCT]
        [--wall-threshold PCT] [--host-overhead-threshold MS]
        [--n-runs N] [--chunked-runs N]
        [--chunked-proportion P] [--calibrate] [--keep]

``--calibrate`` points B at ``main`` too (A-vs-A); rerun it a few
times to measure this machine's null |delta| for both statistics and
set ``--threshold``/``--wall-threshold`` to two to three times the
worst of it. Exit status is 1 for a regression, 2 for an otherwise
inconclusive DISTRUST result, and 0 for a trusted pass.
"""
import argparse
import importlib.util
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent / "lorenz_mean_runtime.py"

# Solves per block for every config. Fixed on purpose: the verdict
# statistics assume the same block shape on every run and machine,
# so solve counts are not a tuning lever. Fifteen solves keep the
# floor statistic well populated while the whole two-backend gate
# stays inside its five-minute budget with the implicit adaptive
# config included.
BLOCK_SOLVES = 15

# label -> (importable spec, CUBIE_CUDA_BACKEND value)
BACKENDS = {
    "numba-cuda": ("numba.cuda", "numba-cuda"),
    "mlir": ("numba_cuda_mlir", "mlir"),
}

META_FIELDS = (
    "regs",
    "spill_store_bytes",
    "spill_load_bytes",
    "shared",
    "dynshared",
    "const",
    "blocks_per_sm",
    "sms",
    "blocksize",
    "runs_per_block",
    "runs",
    "chunks",
)


def installed_backends():
    return [
        label
        for label, (spec, _) in BACKENDS.items()
        if importlib.util.find_spec(spec) is not None
    ]


def start_worker(tree, backend, cache_dir, grid_dir, args):
    """Start one persistent benchmark worker; return the process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(tree) / "src")
    env["CUBIE_CUDA_BACKEND"] = BACKENDS[backend][1]
    env["CUBIE_CACHE_DIR"] = str(cache_dir)
    cmd = [sys.executable, str(BENCH), "--worker",
           "--grid-cache", str(grid_dir), "--no-clear-cache"]
    if args.chunked_runs is not None:
        cmd.extend(("--chunked-runs", str(args.chunked_runs)))
    if args.chunked_proportion is not None:
        cmd.extend((
            "--chunked-proportion", str(args.chunked_proportion)
        ))
    if args.n_runs is not None:
        cmd.append(str(args.n_runs))
    return subprocess.Popen(
        cmd, env=env, cwd=str(tree), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )


def read_reply(proc, prefix, side):
    """Read worker stdout lines until one starts with ``prefix``."""
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(
                f"worker {side} exited (code {proc.poll()})"
            )
        line = line.strip()
        if line.startswith(prefix):
            return line


def parse_meta(line):
    """Parse ``@META <key> name=value ...`` into (key, metric dict)."""
    parts = line.split()
    meta = {}
    for token in parts[2:]:
        name, value = token.split("=")
        meta[name] = int(value)
    return parts[1], meta


def read_startup(proc, side):
    """Collect ``@META`` lines until ``@READY``; return both."""
    metas = {}
    while True:
        line = read_reply(proc, "@", side)
        if line.startswith("@META "):
            key, meta = parse_meta(line)
            metas[key] = meta
        elif line.startswith("@READY "):
            return line.split()[1:], metas


def run_block(proc, side, key, count):
    """Ask one worker for ``count`` solves; return kernel/wall ms."""
    try:
        proc.stdin.write(f"run {key} {count}\n")
        proc.stdin.flush()
    except OSError:
        raise SystemExit(
            f"worker {side} pipe closed (exit {proc.poll()})"
        )
    reply = read_reply(proc, "@TIMES ", side).split()
    if reply[1] != key:
        raise SystemExit(
            f"worker {side} answered for {reply[1]!r}, "
            f"expected {key!r}"
        )
    wall_at = reply.index("wall")
    kernel = [float(v) for v in reply[3:wall_at]]
    wall = [float(v) for v in reply[wall_at + 1:]]
    return kernel, wall


def stop_worker(proc):
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
    except OSError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


def add_main_worktree(main_ref):
    tree = Path(tempfile.mkdtemp(prefix="cubie_main_"))
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach",
         str(tree), main_ref],
        check=True,
    )
    return tree


def remove_worktree(tree):
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "remove", "--force",
         str(tree)],
        check=False,
    )


def abba_order(pairs):
    """Side sequence ABBA BAAB ... balancing time between A and B.

    Even ``pairs`` cancels linear drift in the side means; ``pairs``
    divisible by 4 also cancels quadratic drift (plain ABBA repetition
    always leaves B on the inner positions, which biases B under
    curving drift).
    """
    seq = []
    for i in range(pairs):
        seq.extend(("A", "B") if i % 4 in (0, 3) else ("B", "A"))
    return seq


def compare_meta(backend, metas, keys):
    """Print the A-vs-B compile-metrics table; return regression."""
    regressed = False
    print(f"\n[{backend}] compile metrics, A -> B")
    print(f"{'config':<15}{'metric':<20}{'A':>12}{'B':>12}  flags")
    for key in keys:
        a, b = metas["A"][key], metas["B"][key]
        flags = []
        # Fewer resident blocks per SM means less latency hiding
        # and, at the wave config's sizes, a whole extra wave.
        if b["blocks_per_sm"] < a["blocks_per_sm"]:
            flags.append("OCCUPANCY REGRESSION")
        # Spilled registers turn register traffic into local-memory
        # traffic; any increase is a compiled-code regression even
        # when the timing rows absorb it.
        if (
            b["spill_store_bytes"] > a["spill_store_bytes"]
            or b["spill_load_bytes"] > a["spill_load_bytes"]
        ):
            flags.append("SPILL REGRESSION")
        # A chunk-count mismatch means the sides ran different
        # transfer schedules, so their timings are not comparable
        # and the memory footprint itself changed.
        if b["chunks"] != a["chunks"]:
            flags.append("CHUNK MISMATCH")
        if b["sms"] != a["sms"] or b["runs"] != a["runs"]:
            flags.append("WORKLOAD MISMATCH")
        regressed = regressed or bool(flags)
        for index, field in enumerate(META_FIELDS):
            suffix = "  ".join(flags) if index == 0 else ""
            print(
                f"{key if index == 0 else '':<15}{field:<20}"
                f"{a[field]:>12}{b[field]:>12}  {suffix}"
            )
    return regressed


def classify_deltas(deltas, threshold):
    """Median delta, its verdict, and whether regression is contested.

    Only the ``+threshold`` boundary gates, so only a genuine pair
    split over it makes a row inconclusive: at least two pairs on
    each side. A single straggler follows the majority, and
    improvement-vs-ok scatter never flags.
    """
    delta = statistics.median(deltas)
    if delta > threshold:
        verdict = "REGRESSION"
    elif delta < -threshold:
        verdict = "improvement"
    else:
        verdict = "ok"
    over = sum(value > threshold for value in deltas)
    distrust = min(over, len(deltas) - over) >= 2
    return delta, verdict, distrust


def run_backend(backend, main_tree, b_tree, base, args):
    """Block-interleaved A/B for one backend.

    Returns (timing rows, compile-metrics regression flag).
    """
    grid_dir = base / "grid"
    workers = {}
    metas = {}
    try:
        for side, tree in (("A", main_tree), ("B", b_tree)):
            workers[side] = start_worker(
                tree, backend, base / f"{side}_{backend}", grid_dir,
                args)
        ready = {}
        for side in ("A", "B"):
            ready[side], metas[side] = read_startup(
                workers[side], side)
            print(f"[{backend}] {side} ready", file=sys.stderr)
        common = [key for key in ready["A"] if key in ready["B"]]
        if not common:
            raise SystemExit(
                f"no config keys shared between sides: "
                f"A={ready['A']} B={ready['B']}"
            )
        for side in ("A", "B"):
            skipped = [k for k in ready[side] if k not in common]
            if skipped:
                print(
                    f"[{backend}] configs only on side {side}, "
                    f"skipped: {', '.join(skipped)}",
                    file=sys.stderr,
                )

        # Size the wave config from side A's occupancy — "two full
        # waves at current levels" — and impose that count on both
        # sides, so a B-side occupancy drop must add a third wave.
        fixed_a = metas["A"]["fixed"]
        wave_runs = (2 * fixed_a["sms"] * fixed_a["blocks_per_sm"]
                     * fixed_a["runs_per_block"])
        print(
            f"[{backend}] wave config: {wave_runs} runs = 2 waves x "
            f"{fixed_a['sms']} SMs x {fixed_a['blocks_per_sm']} "
            f"blocks/SM x {fixed_a['runs_per_block']} runs/block",
            file=sys.stderr,
        )
        for side in ("A", "B"):
            workers[side].stdin.write(f"wave {wave_runs}\n")
            workers[side].stdin.flush()
        for side in ("A", "B"):
            line = read_reply(workers[side], "@META wave ", side)
            metas[side]["wave"] = parse_meta(line)[1]
        keys = common + ["wave"]

        def block(side, store=None):
            for key in keys:
                vals = run_block(
                    workers[side], side, key, BLOCK_SOLVES)
                if store is not None:
                    store[side][key].append(vals)
            # Idle gap: without it the GPU sits pinned at its power
            # limit and the kernel-time floor dithers block to block;
            # a rest lets it re-enter the repeatable boost state. The
            # duration is drawn fresh per block so a concurrent
            # periodic GPU load cannot phase-lock with the ping-pong
            # rhythm and bias one side coherently.
            time.sleep(random.uniform(*args.gap))

        # One throwaway ABBA round rides out the post-setup ramp.
        for side in ("A", "B", "B", "A"):
            block(side)
        order = abba_order(args.pairs)
        print(f"[{backend}] measuring {''.join(order)}",
              file=sys.stderr)
        collected = {s: {k: [] for k in keys} for s in ("A", "B")}
        for side in order:
            block(side, collected)
    finally:
        for proc in workers.values():
            stop_worker(proc)

    meta_regressed = compare_meta(backend, metas, keys)

    rows = []
    thresholds = {
        "kernel": args.threshold,
        "wall": args.wall_threshold,
    }
    for key in keys:
        k = min(args.min_count, BLOCK_SOLVES)
        # host_overhead gates on the wall statistic only.
        stats = (
            ("wall",) if key == "host_overhead"
            else ("kernel", "wall")
        )
        for stat in stats:
            stat_index = 0 if stat == "kernel" else 1
            floors = {
                side: [
                    statistics.fmean(sorted(blk[stat_index])[:k])
                    for blk in collected[side][key]
                ]
                for side in ("A", "B")
            }
            a = statistics.fmean(floors["A"])
            b = statistics.fmean(floors["B"])
            # The i-th A and B blocks ran back to back (one ABBA
            # pair), so they share clock state; the median paired
            # delta cancels wander that survives in the side means.
            if key == "host_overhead":
                # Paired deltas in absolute milliseconds.
                deltas = [
                    bf - af
                    for af, bf in zip(floors["A"], floors["B"])
                ]
                threshold = args.host_overhead_threshold
                unit = "ms"
            else:
                deltas = [
                    100.0 * (bf - af) / af
                    for af, bf in zip(floors["A"], floors["B"])
                ]
                threshold = thresholds[stat]
                unit = "%"
            delta, verdict, distrust = classify_deltas(
                deltas, threshold
            )
            rows.append(
                (backend, key, stat, a, b, delta, verdict,
                 distrust, unit)
            )
    return rows, meta_regressed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="origin/main",
                        help="A-side ref (default origin/main; the "
                             "local main branch is often stale).")
    parser.add_argument("--backends", default=None,
                        help="Comma-separated subset of: "
                             + ", ".join(BACKENDS))
    parser.add_argument("--pairs", type=int, default=4,
                        help="A/B block pairs per backend; even "
                             "cancels linear drift, multiples of 4 "
                             "also quadratic.")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Lowest per-solve kernel times averaged "
                             "into each block's kernel statistic.")
    parser.add_argument("--gap", type=float, nargs=2,
                        default=(1.5, 3.5),
                        metavar=("MIN", "MAX"),
                        help="Idle seconds after each block, drawn "
                             "uniformly per block. The rest lets the "
                             "GPU re-enter its repeatable boost state; "
                             "the jitter stops concurrent periodic GPU "
                             "loads phase-locking with the rhythm.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Trajectory count for the fixed and "
                             "adaptive configs (small values "
                             "smoke-test the harness).")
    parser.add_argument("--chunked-runs", type=int, default=None,
                        help="Trajectory count for the chunked "
                             "config (defaults to --n-runs when "
                             "supplied, otherwise 2**22).")
    parser.add_argument("--chunked-proportion", type=float,
                        default=None,
                        help="Manual VRAM proportion for each of the "
                             "chunked config's three memory-manager "
                             "instances; the batch chunks once it "
                             "exceeds three times this share of "
                             "total VRAM.")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Kernel-delta regression threshold in "
                             "percent (two to three times the "
                             "calibrated null).")
    parser.add_argument("--wall-threshold", type=float, default=10.0,
                        help="Wall-delta regression threshold in "
                             "percent; the wall floor still carries "
                             "host scatter from concurrent machine "
                             "load, so it is far coarser than "
                             "--threshold.")
    parser.add_argument("--host-overhead-threshold", type=float,
                        default=0.25,
                        help="Regression threshold for the "
                             "host_overhead config's wall delta, in "
                             "absolute milliseconds per solve.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Point B at main too (A-vs-A null).")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the ephemeral main tree and caches.")
    args = parser.parse_args()

    backends = installed_backends()
    if args.backends:
        wanted = [b.strip() for b in args.backends.split(",")]
        missing = [b for b in wanted if b not in backends]
        if missing:
            raise SystemExit(
                f"backend(s) not installed: {', '.join(missing)}"
            )
        backends = wanted
    if not backends:
        raise SystemExit("no CUDA backend is installed")

    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    a_sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", args.main],
        capture_output=True, text=True,
    ).stdout.strip()
    b_label = args.main if args.calibrate else f"{branch} (working tree)"
    print(f"A = {args.main} ({a_sha})   B = {b_label}   "
          f"backends: {', '.join(backends)}   "
          f"({args.pairs} block pairs x {BLOCK_SOLVES} "
          f"solves/block/side)\n")

    main_tree = add_main_worktree(args.main)
    base = Path(tempfile.mkdtemp(prefix="cubie_abgate_"))
    b_tree = main_tree if args.calibrate else REPO
    regressed = False
    distrusted = False
    try:
        for backend in backends:
            rows, meta_regressed = run_backend(
                backend, main_tree, b_tree, base, args)
            regressed = regressed or meta_regressed
            print()
            for row in rows:
                (bk, key, stat, a, b, delta, verdict, distrust,
                 unit) = row
                regressed = regressed or verdict == "REGRESSION"
                distrusted = distrusted or distrust
                flag = "  DISTRUST" if distrust else ""
                shown = (
                    f"{delta:+7.3f}ms" if unit == "ms"
                    else f"{delta:+6.2f}%"
                )
                print(f"{bk:<11}{key:<15}{stat:<8}"
                      f"A {a:9.3f}  B {b:9.3f}  "
                      f"{shown}  {verdict}{flag}")
    finally:
        if not args.keep:
            remove_worktree(main_tree)
            shutil.rmtree(base, ignore_errors=True)

    status = (
        "REGRESSION" if regressed
        else "INCONCLUSIVE" if distrusted
        else "PASS"
    )
    print(f"\nGATE: {status} "
          f"(kernel threshold {args.threshold:.2f}%, wall threshold "
          f"{args.wall_threshold:.2f}%, host-overhead threshold "
          f"{args.host_overhead_threshold:.2f} ms)")
    if distrusted:
        print("DISTRUST = the pairs split on the regression call, "
              "so that verdict is unreliable; rerun, with more "
              "--pairs or a quieter GPU, before acting on it.")
    if regressed:
        return 1
    return 2 if distrusted else 0


if __name__ == "__main__":
    raise SystemExit(main())
