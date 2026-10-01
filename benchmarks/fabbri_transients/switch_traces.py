"""Dense voltage for chosen switches: peak times and voltages per target."""

import argparse

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--iso-index", type=int, nargs="+", required=True)
    parser.add_argument("--ach-index", type=int, nargs="+", required=True)
    parser.add_argument("--input", default="steady_state_log96.npz")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / args.input)
    side = int(np.sqrt(data["ach"].size))
    targets = np.array([
        r * side + c for r in args.iso_index for c in args.ach_index
    ])
    inits = np.repeat(
        data["final_state"][:, [args.source]], targets.size, axis=1
    )
    params = common.parameter_array(
        system, data["ach"][targets], data["iso"][targets]
    )
    solver = common.make_solver(
        system, output_types=["state", "time"],
        save_variables=[common.VOLTAGE_LABEL], save_every=2.0**-12,
    )
    result = solver.solve(
        initial_values=np.ascontiguousarray(inits, dtype=np.float32),
        parameters=params, duration=args.duration,
    )
    time = np.asarray(result.time[:, 0], dtype=np.float64)
    voltage = np.array(result.time_domain_array[:, 0, :])
    indices, refined = common.dense_peaks(time, voltage)
    for k, target in enumerate(targets):
        idx = indices[k][:6]
        peaks = ", ".join(
            f"{refined[k][j]:.3f} s ({voltage[i, k]:.1f} mV)"
            for j, i in enumerate(idx)
        )
        print(
            f"ACh {data['ach'][target]:7.1f} Iso {data['iso'][target]:6.2f}: "
            f"V range {voltage[:, k].min():.1f} to {voltage[:, k].max():.1f}"
            f" mV; maxima: {peaks}"
        )
    np.savez_compressed(
        common.RESULTS / f"switch_traces_{args.source}.npz",
        time=time, voltage=voltage, targets=targets,
    )


if __name__ == "__main__":
    main()
