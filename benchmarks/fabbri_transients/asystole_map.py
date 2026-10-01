"""Map beating over log-spaced ACh/Iso; writes ``results/asystole_map.npz``."""

import argparse

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--decades", type=float, nargs=2, default=(-3, 4))
    parser.add_argument("--settle", type=float, default=600.0)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--cadence", type=float, default=2.0**-10)
    args = parser.parse_args()

    system = common.build_system()
    axis = np.logspace(*args.decades, args.side)
    ach_grid, iso_grid = np.meshgrid(axis, axis)
    ach, iso = ach_grid.ravel(), iso_grid.ravel()
    params = common.parameter_array(system, ach, iso)
    state = common.initial_state_array(system, ach.size)

    solver = common.make_solver(system, output_types=["state"])
    for _ in range(int(round(args.settle / 300.0))):
        result = solver.solve(
            initial_values=state, parameters=params, duration=300.0,
            nan_error_trajectories=False,
        )
        state = np.array(result.time_domain_array[-1], dtype=np.float32)
        del result
    solver.close()

    n_slots = int(np.ceil(args.window / 0.25)) + 4
    solver = common.make_solver(
        system,
        output_types=[f"peaks[{n_slots}]", "max", "min"],
        summarise_variables=[common.VOLTAGE_LABEL],
        save_variables=[],
        sample_summaries_every=args.cadence,
        summarise_every=args.window,
    )
    result = solver.solve(
        initial_values=state, parameters=params, duration=args.window,
        nan_error_trajectories=False,
    )
    failed = common.failed_runs(result)
    legend = [label for _, label in sorted(result.summaries_legend.items())]
    summaries = np.asarray(result.summaries_array)[0]
    first = next(i for i, s in enumerate(legend) if s.endswith(" peaks_1"))
    slots = summaries[first : first + n_slots].T
    v_max = summaries[next(i for i, s in enumerate(legend)
                           if s.endswith(" max"))]
    v_min = summaries[next(i for i, s in enumerate(legend)
                           if s.endswith(" min"))]
    del result
    solver.close()
    times = np.where(slots > 0, (slots + 1.0) * args.cadence, np.nan)
    times.sort(axis=1)
    counts = np.isfinite(times).sum(axis=1)
    cycle = np.nanmedian(np.diff(times, axis=1), axis=1)
    # A beat must overshoot 0 mV.
    beating = (counts >= 2) & (v_max > 0)
    print(
        f"{ach.size} runs, failed {int(failed.sum())}, beating "
        f"{int(beating.sum())}"
    )
    for row, iso_value in enumerate(axis):
        line = beating.reshape(args.side, args.side)[row]
        print(f"Iso {iso_value:10.4g} nM: " + "".join(
            "#" if b else "." for b in line))
    np.savez_compressed(
        common.RESULTS / "asystole_map.npz",
        axis=axis, ach=ach, iso=iso, beating=beating, cycle=cycle,
        counts=counts, v_max=v_max, v_min=v_min, failed=failed,
    )


if __name__ == "__main__":
    main()
