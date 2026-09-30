"""Time to a steady periodic state from the model's initial conditions.

Integrates a ``side`` x ``side`` subset of the ACh/Iso grid (evenly
spaced indices of the full 256-point axes) from the CellML initial
state, saving membrane voltage densely. Peak times are the vertices of
parabolas through each sampled local maximum; cycle lengths are the
differences of successive peak times. Writes peak times per run to
``results/settle_peaks_<side>_<duration>s.npz``.

Usage::

    python benchmarks/fabbri_transients/settle.py [--side 16]
        [--duration 300] [--save-every 0.00048828125]
"""

import argparse

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--save-every", type=float, default=2.0**-11)
    args = parser.parse_args()

    system = common.build_system()
    pick = np.round(
        np.linspace(0, common.GRID_SIDE - 1, args.side)
    ).astype(int)
    ach_axis, iso_axis = common.axes()
    ach_grid, iso_grid = np.meshgrid(ach_axis[pick], iso_axis[pick])
    ach, iso = ach_grid.ravel(), iso_grid.ravel()
    n_runs = ach.size

    solver = common.make_solver(
        system,
        output_types=["state", "time"],
        save_variables=[common.VOLTAGE_LABEL],
        save_every=args.save_every,
        time_logging_level="default",
    )
    inits = common.initial_state_array(system, n_runs)
    params = common.parameter_array(system, ach, iso)
    result, kernel_ms, wall_s = common.timed_solve(
        solver, inits, params, args.duration
    )
    failed = common.failed_runs(result)
    print(
        f"runs {n_runs}, duration {args.duration} s, kernel "
        f"{kernel_ms / 1e3:.2f} s, wall {wall_s:.2f} s, failed "
        f"{int(failed.sum())}"
    )
    time = np.asarray(result.time[:, 0], dtype=np.float64)
    voltage = np.asarray(result.time_domain_array[:, 0, :])
    indices, peaks = common.dense_peaks(time, voltage)
    counts = np.array([p.size for p in peaks])
    padded = np.full((n_runs, counts.max()), np.nan)
    peak_voltage = np.full((n_runs, counts.max()), np.nan)
    for run, (idx, times) in enumerate(zip(indices, peaks)):
        padded[run, : times.size] = times
        peak_voltage[run, : times.size] = voltage[idx, run]
    common.RESULTS.mkdir(exist_ok=True)
    out = common.RESULTS / (
        f"settle_peaks_{args.side}_{args.duration:g}s.npz"
    )
    np.savez_compressed(
        out,
        ach=ach,
        iso=iso,
        peak_times=padded,
        peak_voltage=peak_voltage,
        failed=failed,
        final_voltage=voltage[-1],
        save_every=args.save_every,
        duration=args.duration,
        kernel_ms=kernel_ms,
    )
    print(f"peaks per run: min {counts.min()}, max {counts.max()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
