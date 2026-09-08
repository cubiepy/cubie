#!/usr/bin/env python
"""Full factorial of the buffer placements on the Lorenz/tsit5 kernel.

The GPUODEBenchmarks Lorenz problem (``rho`` over ``[0, 21]``,
``atol = rtol = 1e-5``, first step ``2**-10``, duration 1.0) on
``tsit5``: 32 kernels over the five relocatable buffers at block sizes
32 to 256; ``--cross`` multiplies in the unroll factorial. Solves are
transient-dominated, so the run count sets the solve length (about
7 ms at ``2**24`` on an RTX 4070 SUPER, 20 ms at ``2**26``).

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
    pl.apply_icache_override(pl.icache_bytes_from(args))
    arms = pl.factorial_arms(groups, buffers, pl.PUBLICATION_LORENZ)
    log(f"{len(arms)} kernels: groups {groups} buffers {buffers}")
    pl.run_jobs(
        [("lorenz", "tsit5", args.n_runs)],
        lambda system, algo: arms, args, log,
        duration_override=args.duration,
    )


if __name__ == "__main__":
    main()
