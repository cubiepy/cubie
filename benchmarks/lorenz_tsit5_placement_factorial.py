#!/usr/bin/env python
"""Full factorial of the buffer placements on the Lorenz/tsit5 kernel.

The GPUODEBenchmarks Lorenz problem (``rho`` swept linearly over
``[0, 21]``, ``atol = rtol = 1e-5``, first step ``2**-10``, duration
1.0, one state save at the end) on ``tsit5``. Every relocatable buffer
of the kernel (``stage_rhs``, ``stage_accumulator``, ``state``,
``proposed_state``, ``error``) takes the levels local and shared: 32
kernels, each timed at block sizes 32, 64, 128 and 256 with every
loop group fully unrolled, unless ``--cross`` multiplies in the
unroll factorial of ``lorenz_tsit5_unroll_factorial.py`` (1024
kernels).

With ``rho`` below the Lorenz bifurcation every trajectory settles to
a fixed point and the kernel time is transient-dominated, so the run
count, not the duration, sets the solve length. The default ``2**24``
trajectories give solves of about 7 ms on an RTX 4070 SUPER (0.42 ms
at ``2**20``); ``2**26`` reaches the 20 ms floor of the landscape
banks at 1 GiB of device arrays.

Usage::

    python benchmarks/lorenz_tsit5_placement_factorial.py --out records.jsonl
        [--cross] [--n-runs 16777216] [--duration 1.0]
        [--blocksizes 32,64,128,256] [--score]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy_landscape as pl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pl.add_common_arguments(parser)
    parser.add_argument("--cross", action="store_true",
                        help="multiply in the unroll factorial")
    parser.add_argument("--n-runs", type=int, default=1 << 24)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.set_defaults(blocksizes="32,64,128,256")
    args = parser.parse_args()
    groups = pl.ERK_UNROLL_GROUPS if args.cross else ()
    buffers = pl.ERK_BUFFERS
    if args.score:
        for row in pl.load_rows(args.out):
            pl.score_factorial(row, groups, buffers)
        return
    log = pl.make_logger(args.log)
    pl.check_device(log)
    pl.apply_icache_override(
        None if args.icache_kib is None else args.icache_kib * 1024
    )
    arms = pl.factorial_arms(groups, buffers, pl.PUBLICATION_LORENZ)
    log(f"{len(arms)} kernels: groups {groups} buffers {buffers}")
    pl.run_jobs(
        [("lorenz", "tsit5", args.n_runs)],
        lambda system, algo: arms, args, log,
        duration_override=args.duration,
    )


if __name__ == "__main__":
    main()
