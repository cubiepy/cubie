"""Settling-time analysis of ``settle.py`` output.

For each run, the steady cycle length is the median of the last
``--tail`` cycle lengths. A run has settled at tolerance ``tol`` from
the start of the first cycle after which every cycle length stays
within ``tol`` of the steady value. Prints the settling-time
distribution per tolerance and plots cycle length against time.

Usage::

    python benchmarks/fabbri_transients/settle_analysis.py NPZ
        [--tail 10]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TOLERANCES_MS = (1.0, 0.1, 0.01)


def settle_times(peak_times, tail, tol_s):
    """Return per-run settling time, steady cycle length and beats."""
    n_runs = peak_times.shape[0]
    settle = np.full(n_runs, np.nan)
    steady = np.full(n_runs, np.nan)
    for run in range(n_runs):
        times = peak_times[run]
        times = times[np.isfinite(times)]
        if times.size < tail + 2:
            continue
        cycle = np.diff(times)
        steady[run] = np.median(cycle[-tail:])
        outside = np.flatnonzero(np.abs(cycle - steady[run]) > tol_s)
        first = 0 if outside.size == 0 else outside[-1] + 1
        settle[run] = times[first]
    return settle, steady


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()
    data = np.load(args.npz)
    peak_times = data["peak_times"]
    ach, iso = data["ach"], data["iso"]
    duration = float(data["duration"])
    counts = np.isfinite(peak_times).sum(axis=1)
    print(
        f"{peak_times.shape[0]} runs, {duration:g} s; peaks per run "
        f"min {counts.min()} max {counts.max()}; runs with < "
        f"{args.tail + 2} peaks: {(counts < args.tail + 2).sum()}"
    )
    rows = []
    for tol_ms in TOLERANCES_MS:
        settle, steady = settle_times(
            peak_times, args.tail, tol_ms * 1e-3
        )
        tail_window = np.array([
            peak_times[run][np.isfinite(peak_times[run])][-args.tail - 1]
            if counts[run] > args.tail else np.nan
            for run in range(peak_times.shape[0])
        ])
        # Settled before the tail window starts, so the steady value
        # is not the settled cycles themselves.
        reliable = settle < tail_window
        rows.append((tol_ms, settle, steady))
        finite = settle[np.isfinite(settle)]
        print(
            f"tol {tol_ms:g} ms: settle time median "
            f"{np.median(finite):.1f} s, p90 "
            f"{np.percentile(finite, 90):.1f} s, max "
            f"{finite.max():.1f} s; settled before tail window: "
            f"{int(reliable.sum())}/{finite.size}"
        )
        worst = np.nanargmax(settle)
        print(
            f"  slowest run ACh {ach[worst]:.1f} nM, Iso "
            f"{iso[worst]:.1f} nM, steady CL "
            f"{steady[worst] * 1e3:.2f} ms"
        )
    _, steady = settle_times(peak_times, args.tail, 1e-3)
    print(
        f"steady CL range {np.nanmin(steady) * 1e3:.1f}-"
        f"{np.nanmax(steady) * 1e3:.1f} ms"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = np.argsort(np.nan_to_num(steady))
    picks = order[np.linspace(0, order.size - 1, 12).astype(int)]
    for run in picks:
        times = peak_times[run][np.isfinite(peak_times[run])]
        if times.size < 3:
            continue
        cycle = np.diff(times) * 1e3
        axes[0].plot(
            times[1:], cycle - cycle[-1], lw=0.8,
            label=f"ACh {ach[run]:.0f}, Iso {iso[run]:.0f}",
        )
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("cycle length - final (ms)")
    axes[0].legend(fontsize=6, ncol=2)
    for tol_ms, settle, _ in rows:
        finite = np.sort(settle[np.isfinite(settle)])
        axes[1].plot(
            finite, np.arange(1, finite.size + 1) / finite.size,
            label=f"{tol_ms:g} ms",
        )
    axes[1].set_xlabel("settling time (s)")
    axes[1].set_ylabel("fraction of runs settled")
    axes[1].legend(title="tolerance")
    fig.tight_layout()
    out = args.npz.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
