"""Settle the full grid and record its steady periodic state.

Phase 1 integrates ``--settle`` s from the CellML state in
``--segment``-s solves, each restarting at ``t0 = 0`` from the last
final state. Phase 2 runs ``--char`` s more, recording voltage peaks
and troughs. Writes the phase-2 final state and extremum times to
``results/steady_state.npz``.

Usage::

    python benchmarks/fabbri_transients/steady_state.py [--settle 600]
        [--segment 300] [--char 10] [--cadence 0.000244140625]
"""

import argparse
from time import perf_counter

import numpy as np

import common


def extrema_times(result, cadence, n_slots):
    """Return per-run peak and trough times from one summary window."""
    legend = [
        label for _, label in sorted(result.summaries_legend.items())
    ]
    summaries = np.asarray(result.summaries_array)
    times = {}
    for name in ("peaks", "negative_peaks"):
        first = next(
            i for i, label in enumerate(legend)
            if label.endswith(f" {name}_1")
        )
        # (window, slot, run) for the single summarised variable.
        slots = summaries[0, first : first + n_slots, :].T
        # Update k samples t = (k + 1) * cadence; 0 marks an empty slot.
        stamped = np.where(slots > 0, (slots + 1.0) * cadence, np.nan)
        stamped.sort(axis=1)
        times[name] = stamped
    return times["peaks"], times["negative_peaks"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settle", type=float, default=600.0)
    parser.add_argument("--segment", type=float, default=300.0)
    parser.add_argument("--char", type=float, default=10.0)
    parser.add_argument("--cadence", type=float, default=2.0**-12)
    parser.add_argument("--side", type=int, default=common.GRID_SIDE)
    args = parser.parse_args()

    system = common.build_system()
    ach, iso = common.grid_values(args.side)
    n_runs = ach.size
    params = common.parameter_array(system, ach, iso)
    state = common.initial_state_array(system, n_runs)
    failed = np.zeros(n_runs, dtype=bool)

    solver = common.make_solver(system, output_types=["state"])
    n_segments = int(round(args.settle / args.segment))
    start = perf_counter()
    for segment in range(n_segments):
        result, kernel_ms, _ = common.timed_solve(
            solver, state, params, args.segment,
            nan_error_trajectories=False,
        )
        bad = common.failed_runs(result)
        failed |= bad
        state = np.array(result.time_domain_array[-1], dtype=np.float32)
        del result
        print(
            f"settle segment {segment + 1}/{n_segments}: kernel "
            f"{kernel_ms / 1e3:.1f} s, failed {int(bad.sum())}, "
            f"elapsed {perf_counter() - start:.0f} s",
            flush=True,
        )
    solver.close()

    n_slots = int(np.ceil(args.char / 0.25)) + 4
    solver = common.make_solver(
        system,
        output_types=[
            "state", f"peaks[{n_slots}]", f"negative_peaks[{n_slots}]",
        ],
        summarise_variables=[common.VOLTAGE_LABEL],
        sample_summaries_every=args.cadence,
        summarise_every=args.char,
    )
    result, kernel_ms, _ = common.timed_solve(
        solver, state, params, args.char, nan_error_trajectories=False
    )
    bad = common.failed_runs(result)
    failed |= bad
    final_state = np.array(result.time_domain_array[-1], dtype=np.float32)
    peaks, troughs = extrema_times(result, args.cadence, n_slots)
    del result
    solver.close()
    print(
        f"characterisation {args.char:g} s: kernel {kernel_ms / 1e3:.1f} s,"
        f" failed {int(bad.sum())}; total failed {int(failed.sum())}"
    )

    cycle = np.diff(peaks, axis=1)
    counts = np.isfinite(peaks).sum(axis=1)
    steady_cl = np.nanmedian(cycle, axis=1)
    spread = np.nanmax(cycle, axis=1) - np.nanmin(cycle, axis=1)
    beating = counts >= 3
    print(
        f"beating runs {int(beating.sum())}/{n_runs}; steady CL "
        f"{np.nanmin(steady_cl) * 1e3:.1f}-{np.nanmax(steady_cl) * 1e3:.1f}"
        f" ms; CL spread within window: median "
        f"{np.nanmedian(spread) * 1e3:.3f} ms, p99 "
        f"{np.nanpercentile(spread, 99) * 1e3:.3f} ms, max "
        f"{np.nanmax(spread) * 1e3:.3f} ms"
    )
    common.RESULTS.mkdir(exist_ok=True)
    out = common.RESULTS / "steady_state.npz"
    np.savez_compressed(
        out,
        ach=ach,
        iso=iso,
        final_state=final_state,
        state_names=np.array(list(system.initial_values.names)),
        peak_times=peaks,
        trough_times=troughs,
        failed=failed,
        steady_cl=steady_cl,
        settle=args.settle,
        char=args.char,
        cadence=args.cadence,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
