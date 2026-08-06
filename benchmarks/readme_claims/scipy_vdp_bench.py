"""Time SciPy RK45 on the 1024 x 1024 van der Pol grid.

Levers: plain solve_ivp, njit RHS, direct RK45 stepping, one solve_ivp over
a stacked block, and multiprocessing.  Timed on a random subsample and
scaled to 1,048,576 solves; --full integrates the whole grid.

Grid construction, sampling and worker startup all happen outside the timed
regions.

    python scipy_vdp_bench.py -n 20000 --block 256
    python scipy_vdp_bench.py --full
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from scipy.integrate import RK45, solve_ivp

N_X0 = 1024
N_MU = 1024
N_RUNS = N_X0 * N_MU
DURATION = 20.0
ATOL = 1e-6
RTOL = 1e-3

X0_VALUES = np.linspace(1.0, 2.0, N_X0)
MU_VALUES = np.linspace(1.0, 3.0, N_MU)


def rhs_python(t, y, mu):
    x, v = y
    return [v, mu * (1.0 - x * x) * v - x]


try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is optional here
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit(cache=True, fastmath=True)
    def rhs_numba(t, y, mu):
        out = np.empty(2)
        out[0] = y[1]
        out[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0]
        return out


def rhs_block(t, y, mu_block):
    """RHS for B systems stacked into one 2B-long state vector."""
    n = mu_block.size
    x = y[:n]
    v = y[n:]
    out = np.empty_like(y)
    out[:n] = v
    out[n:] = mu_block * (1.0 - x * x) * v - x
    return out


def sample_grid(n_sample, seed=0):
    """Draw n_sample (x0, mu) pairs from the full combinatorial grid."""
    rng = np.random.default_rng(seed)
    flat = rng.choice(N_RUNS, size=n_sample, replace=False)
    return X0_VALUES[flat // N_MU], MU_VALUES[flat % N_MU]


def full_grid():
    return np.repeat(X0_VALUES, N_MU), np.tile(MU_VALUES, N_X0)


def run_plain(x0s, mus):
    for x0, mu in zip(x0s, mus):
        solve_ivp(
            rhs_python,
            (0.0, DURATION),
            [x0, 0.0],
            method="RK45",
            atol=ATOL,
            rtol=RTOL,
            args=(mu,),
        )


def run_numba(x0s, mus):
    for x0, mu in zip(x0s, mus):
        solve_ivp(
            rhs_numba,
            (0.0, DURATION),
            np.array([x0, 0.0]),
            method="RK45",
            atol=ATOL,
            rtol=RTOL,
            args=(mu,),
        )


def run_direct(x0s, mus):
    fun = rhs_numba if HAVE_NUMBA else rhs_python
    for x0, mu in zip(x0s, mus):
        solver = RK45(
            lambda t, y, mu=mu: fun(t, y, mu),
            0.0,
            np.array([x0, 0.0]),
            DURATION,
            rtol=RTOL,
            atol=ATOL,
        )
        while solver.status == "running":
            solver.step()


def run_mega(x0s, mus, block):
    for start in range(0, x0s.size, block):
        xb = x0s[start : start + block]
        mb = np.ascontiguousarray(mus[start : start + block])
        y0 = np.concatenate([xb, np.zeros_like(xb)])
        solve_ivp(
            rhs_block,
            (0.0, DURATION),
            y0,
            method="RK45",
            atol=ATOL,
            rtol=RTOL,
            args=(mb,),
        )


def _worker(chunk):
    run_direct(*chunk)
    return chunk[0].size


def _worker_mega(chunk):
    x0s, mus, block = chunk
    run_mega(x0s, mus, block)
    return x0s.size


def _warm(_):
    """Import, JIT-cache load and one throwaway solve in each worker."""
    run_direct(X0_VALUES[:1], MU_VALUES[:1])
    return 0


def make_chunks(x0s, mus, n_chunks):
    return list(
        zip(np.array_split(x0s, n_chunks), np.array_split(mus, n_chunks))
    )


def per_solve_stats(x0s, mus, fun):
    times = np.empty(x0s.size)
    for i in range(x0s.size):
        t0 = time.perf_counter()
        fun(x0s[i : i + 1], mus[i : i + 1])
        times[i] = time.perf_counter() - t0
    return times


def report(name, total_s, n, times=None):
    per = total_s / n
    full = per * N_RUNS
    ci = ""
    if times is not None and times.size > 1:
        half = 1.96 * times.std(ddof=1) / np.sqrt(times.size)
        ci = f"  +/-{half * N_RUNS / 60.0:.2f} min (95% CI)"
    print(
        f"{name:<12} {per * 1e3:9.3f} ms/solve   "
        f"1,048,576 -> {full / 60.0:9.2f} min{ci}"
    )
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-sample", type=int, default=20000)
    ap.add_argument("--block", type=int, default=256, help="mega block size")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument(
        "--chunks-per-worker",
        type=int,
        default=4,
        help="pool load-balancing granularity",
    )
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated subset of plain,numba,direct,mega,pool",
    )
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    x0s, mus = full_grid() if args.full else sample_grid(
        args.n_sample, args.seed
    )
    n = x0s.size
    print(
        f"{'full grid' if args.full else 'sample'}: {n} solves, "
        f"duration={DURATION}, rtol={RTOL}, atol={ATOL}\n"
    )

    if HAVE_NUMBA:
        run_numba(x0s[:1], mus[:1])
        run_direct(x0s[:1], mus[:1])

    levers = [("plain", run_plain), ("numba", run_numba)]
    if HAVE_NUMBA:
        levers.append(("direct", run_direct))

    for name, fn in levers:
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        fn(x0s, mus)
        elapsed = time.perf_counter() - t0
        times = per_solve_stats(x0s, mus, fn) if n <= 512 else None
        report(name, elapsed, n, times)

    if not only or "mega" in only:
        t0 = time.perf_counter()
        run_mega(x0s, mus, args.block)
        report(f"mega/{args.block}", time.perf_counter() - t0, n)

    if not only or "pool" in only:
        chunks = make_chunks(
            x0s, mus, args.workers * args.chunks_per_worker
        )
        with mp.Pool(args.workers) as pool:
            t_start = time.perf_counter()
            pool.map(_warm, range(args.workers))
            print(
                f"(pool of {args.workers} started and warmed in "
                f"{time.perf_counter() - t_start:.1f} s, not timed)"
            )
            t0 = time.perf_counter()
            pool.map(_worker, chunks, chunksize=1)
            report(f"pool/{args.workers}", time.perf_counter() - t0, n)

            mega_chunks = [(a, b, args.block) for a, b in chunks]
            t0 = time.perf_counter()
            pool.map(_worker_mega, mega_chunks, chunksize=1)
            report(
                f"pool+mega/{args.workers}", time.perf_counter() - t0, n
            )


if __name__ == "__main__":
    main()
