"""Re-run failed source-target switches; report status codes and times."""

import argparse
from collections import Counter

import numpy as np

import common
from cubie.result_codes import decode_status_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--input", default="steady_state_log96.npz")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / args.input)
    pairs = np.load(args.pairs)
    source, target = pairs[:, 0], pairs[:, 1]
    inits = np.ascontiguousarray(data["final_state"][:, source])
    params = common.parameter_array(
        system, data["ach"][target], data["iso"][target]
    )
    solver = common.make_solver(
        system,
        output_types=["peaks[3]"],
        summarise_variables=[common.VOLTAGE_LABEL],
        save_variables=[],
        sample_summaries_every=2.0**-12,
        summarise_every=args.duration,
    )
    result = solver.solve(
        initial_values=inits, parameters=params, duration=args.duration,
        nan_error_trajectories=False,
    )
    codes = np.asarray(result.status_codes).ravel()
    failed = (codes & 0xFFFF) != 0
    print(f"{failed.sum()} of {codes.size} fail again")
    decoded = decode_status_codes(codes[failed].tolist())
    names = Counter(", ".join(v) for v in decoded.values())
    print("status:", dict(names))


if __name__ == "__main__":
    main()
