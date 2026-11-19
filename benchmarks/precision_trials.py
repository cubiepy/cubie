#!/usr/bin/env python
"""Time and score reduced-precision trial settings on Lorenz and Fabbri.

Each configuration builds one ``Solver``; each arm is a dict of solver
settings applied over it. ``measure`` times every arm with the
``ComparisonRunner`` protocol and scores its cost against the first
arm: status flags, step and Newton counters, and the end-state
deviation in units of the scaled tolerance. ``census`` runs one solve
between ``cuda.profile_start`` and ``cuda.profile_stop`` for an Nsight
Compute capture.

Usage::

    python benchmarks/precision_trials.py measure lorenz_tsit5 \
        --arms '[{}, {"lowp_norm": true}]'
    ncu --profile-from-start off ... python benchmarks/precision_trials.py \
        census fabbri_radau5
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import cubie
from cubie._cudasim_extensions import cuda
from cubie.backend.utils import kernel_resources
from cubie.batchsolving.comparison import Candidate, ComparisonRunner
from cubie.result_codes import CUBIE_RESULT_CODES

PRECISION = np.float32
FABBRI_CELLML = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)
FABBRI_PARAMETERS = (
    "Rate_modulation_experiments_ACh",
    "Rate_modulation_experiments_Iso_cas",
)


def build_lorenz():
    """Return the three-state Lorenz system."""
    return cubie.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": 21.0, "sigma": 10.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=PRECISION,
    )


def build_fabbri():
    """Return the 35-state Fabbri-Linder sinoatrial system with ANS on."""
    system = cubie.load_cellml_model(
        str(FABBRI_CELLML),
        precision=PRECISION,
        voltage_variable="Membrane$V_ode",
    )
    system.set_default_parameters({"Rate_modulation_experiments_ANS": 1.0})
    return system


def lorenz_grid(solver, n_runs):
    """Return a rho sweep over [0, 21]."""
    return solver.prepare_grid_arrays(
        parameters={"rho": np.linspace(0.0, 21.0, n_runs)},
    )


def fabbri_grid(solver, n_runs):
    """Return ``n_runs`` points spread over the ACh by Iso mesh."""
    side = int(np.ceil(np.sqrt(n_runs)))
    ach, iso = np.meshgrid(
        np.linspace(0.0, 2e-8, side), np.linspace(0.0, 1.0, side)
    )
    index = np.linspace(0, side * side - 1, n_runs).round().astype(int)
    return solver.prepare_grid_arrays(
        parameters={
            FABBRI_PARAMETERS[0]: ach.ravel()[index],
            FABBRI_PARAMETERS[1]: iso.ravel()[index],
        }
    )


LORENZ_TOLS = {"atol": 1e-6, "rtol": 1e-6, "dt_min": 1e-12, "dt_max": 1e3}
FABBRI_TOLS = {"atol": 1e-6, "rtol": 1e-4, "dt_min": 1e-12, "dt_max": 1e-2}

CONFIGS = {
    "lorenz_tsit5": dict(
        build=build_lorenz, grid=lorenz_grid, tols=LORENZ_TOLS,
        solver=dict(algorithm="tsit5"), duration=64.0,
    ),
    "lorenz_rk4_fixed": dict(
        build=build_lorenz, grid=lorenz_grid,
        tols={"atol": 1e-6, "rtol": 1e-6},
        solver=dict(algorithm="classical-rk4", step_controller="fixed",
                    dt=1e-3),
        duration=16.0, save_every=0.25,
    ),
    "lorenz_tsit5_late": dict(
        build=build_lorenz, grid=lorenz_grid, tols=LORENZ_TOLS,
        solver=dict(algorithm="tsit5"), duration=64.0, save_every=4.0,
        t0=1.0e4,
    ),
    "lorenz_kvaerno3": dict(
        build=build_lorenz, grid=lorenz_grid, tols=LORENZ_TOLS,
        solver=dict(algorithm="kvaerno3", linear_correction_type="lu"),
        duration=16.0,
    ),
    "lorenz_rosenbrock23": dict(
        build=build_lorenz, grid=lorenz_grid, tols=LORENZ_TOLS,
        solver=dict(algorithm="rosenbrock23", linear_correction_type="lu"),
        duration=64.0,
    ),
    "fabbri_radau5": dict(
        build=build_fabbri, grid=fabbri_grid, tols=FABBRI_TOLS,
        solver=dict(algorithm="radau_iia_5", linear_correction_type="lu"),
        duration=1.0,
    ),
    "fabbri_kvaerno3": dict(
        build=build_fabbri, grid=fabbri_grid, tols=FABBRI_TOLS,
        solver=dict(algorithm="kvaerno3", linear_correction_type="lu"),
        duration=1.0,
    ),
    "fabbri_rosenbrock23": dict(
        build=build_fabbri, grid=fabbri_grid, tols=FABBRI_TOLS,
        solver=dict(algorithm="rosenbrock23", linear_correction_type="lu"),
        duration=1.0,
    ),
}


def build_solver(name, settings=None):
    """Return the configured solver and its config entry."""
    entry = CONFIGS[name]
    kwargs = dict(entry["tols"])
    kwargs.update(entry["solver"])
    kwargs.update(
        output_types=["state", "iteration_counters"],
        save_every=entry.get("save_every", entry["duration"]),
        time_logging_level="silent",
    )
    kwargs.update(settings or {})
    solver = cubie.Solver(entry["build"](), **kwargs)
    return solver, entry


def solve(solver, entry, n_runs):
    """Solve ``n_runs`` trajectories saving only the end state."""
    inits, params = entry["grid"](solver, n_runs)
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve(
            initial_values=inits,
            parameters=params,
            duration=entry["duration"],
            t0=entry.get("t0", 0.0),
            save_every=entry.get("save_every", entry["duration"]),
        )


def end_state_and_counters(result):
    """Return the last saved state, the counters and the status codes."""
    # Copy: the result's buffers return to the solver for its next solve.
    state = np.array(result.time_domain_array[-1], copy=True)
    counters = np.array(result.iteration_counters, copy=True)
    codes = np.array(result.status_codes, copy=True).ravel()
    return state, counters, codes


def flag_histogram(codes):
    """Return ``{flag name: run count}`` for every flag that fired."""
    flags = codes & 0xFFFF
    histogram = {}
    for member in CUBIE_RESULT_CODES:
        if member.value and np.count_nonzero(flags & member.value):
            histogram[member.name] = int(np.count_nonzero(flags & member.value))
    return histogram


COUNTER_NAMES = ("newton", "krylov", "steps", "rejected")
SASS_OPCODES = (
    "DADD", "DFMA", "DMUL", "DSETP", "F2F", "MUFU", "HFMA2", "F2FP",
    "LDL", "STL", "FFMA", "FMUL", "FADD",
)


def nvdisasm_path():
    """Return the nvdisasm executable on PATH or under CUDA_PATH."""
    import os
    import shutil

    found = shutil.which("nvdisasm")
    if found:
        return found
    return os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "nvdisasm")


def sass_opcode_counts(solver):
    """Return static SASS counts of the opcodes in ``SASS_OPCODES``."""
    import os
    import subprocess
    import tempfile
    from collections import Counter

    from cubie.backend.utils import _compiled_cubin

    cubin = _compiled_cubin(solver.kernel.kernel, solver.kernel.signature)
    if not isinstance(cubin, (bytes, bytearray)):
        cubin = cubin.code
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        text = subprocess.run(
            [nvdisasm_path(), "-c", path], check=True, capture_output=True,
            text=True,
        ).stdout
    finally:
        os.unlink(path)
    counts = Counter()
    for line in text.splitlines():
        if "*/" not in line:
            continue
        tokens = line.split("*/", 1)[1].replace(";", " ").split()
        if tokens and tokens[0].startswith("@"):
            tokens = tokens[1:]
        if tokens and not tokens[0].startswith("/*"):
            counts[tokens[0].split(".")[0]] += 1
    return {op: counts.get(op, 0) for op in SASS_OPCODES}


def scaled_deviation(values, reference, tols, ok):
    """Return ``|values - reference| / (atol + rtol |reference|)``."""
    scale = tols["atol"] + tols["rtol"] * np.abs(reference)
    deviation = np.abs(values - reference) / scale
    return deviation[:, ok]


def describe(deviation):
    """Return the median, 99th percentile and maximum of ``deviation``."""
    if deviation.size == 0:
        return {"median": float("nan"), "p99": float("nan"),
                "max": float("nan")}
    return {
        "median": float(np.median(deviation)),
        "p99": float(np.percentile(deviation, 99)),
        "max": float(np.max(deviation)),
    }


def run_measure(name, arms, n_runs, rounds, target_ms):
    """Time every arm and score its cost against the first arm."""
    solver, entry = build_solver(name)
    tols = entry["tols"]
    inits, params = entry["grid"](solver, n_runs)
    candidates = [
        Candidate(label=json.dumps(arm, sort_keys=True), settings=arm)
        for arm in arms
    ]
    report = {"config": name, "arms": []}
    with ComparisonRunner(
        solver, inits, params, duration=entry["duration"],
        t0=entry.get("t0", 0.0),
    ) as runner:
        accepted = runner.compile(candidates)
        runner.warm()
        runner.size_batch(accepted, waves=4)
        if target_ms:
            measured = runner.solve_ms(None)
            runner.fit_batch(measured, target_ms, grow=True, shrink=True)
        timings = {candidate.label: [] for candidate in accepted}
        round_best = {candidate.label: [] for candidate in accepted}
        for _ in range(rounds):
            for timing in runner.time(accepted):
                timings[timing.candidate.label].extend(timing.times_ms)
                round_best[timing.candidate.label].append(timing.best_ms)
        batch = runner.runs
        waves = {}
        for candidate in accepted:
            runner.select(candidate)
            waves[candidate.label] = runner.geometry(None)
    report["runs_timed"] = batch

    # Score every arm on the same grid, plus a tolerance/10 reference.
    reference_solver, _ = build_solver(
        name, {"atol": tols["atol"] / 10, "rtol": tols["rtol"] / 10}
    )
    reference_state, _, reference_codes = end_state_and_counters(
        solve(reference_solver, entry, n_runs)
    )
    reference_solver.close()
    rows = []
    for candidate in accepted:
        solver.update(
            {key: None for arm in arms for key in arm}, silent=True
        )
        solver.update(candidate.settings, silent=True)
        state, counters, codes = end_state_and_counters(
            solve(solver, entry, n_runs)
        )
        rows.append((candidate, state, counters, codes,
                     sass_opcode_counts(solver),
                     kernel_resources(solver.kernel.kernel)))
    solver.close()

    base_state, base_counters, base_codes = rows[0][1:4]
    for candidate, state, counters, codes, sass, resources in rows:
        both_ok = (codes == 0) & (base_codes == 0) & (reference_codes == 0)
        totals = counters.sum(axis=0)
        base_totals = base_counters.sum(axis=0)
        times = timings[candidate.label]
        paired = 100.0 * (
            np.asarray(round_best[candidate.label])
            / np.asarray(round_best[accepted[0].label])
            - 1.0
        )
        blocks_per_sm, wave_count = waves[candidate.label]
        arm = {
            "settings": candidate.settings,
            "best_ms": min(times),
            "median_ms": float(np.median(times)),
            "times_ms": times,
            "paired_delta_pct": paired.tolist(),
            "median_paired_delta_pct": float(np.median(paired)),
            "blocks_per_sm": blocks_per_sm,
            "waves": wave_count,
            "registers": resources.registers_per_thread,
            "local_bytes": resources.local_bytes_per_thread,
            "sass": sass,
            "failed_runs": int(np.count_nonzero(codes)),
            "flags": flag_histogram(codes),
            "counters_mean": {
                counter: float(totals[i].mean())
                for i, counter in enumerate(COUNTER_NAMES)
                if i < totals.shape[0]
            },
            "counters_vs_base": {
                counter: float(totals[i].sum() / max(base_totals[i].sum(), 1))
                for i, counter in enumerate(COUNTER_NAMES)
                if i < totals.shape[0]
            },
            "deviation_vs_base": describe(
                scaled_deviation(state, base_state, tols, both_ok)
            ),
            "error_vs_reference": describe(
                scaled_deviation(state, reference_state, tols, both_ok)
            ),
        }
        report["arms"].append(arm)
    return report


def print_report(report):
    """Print one line per arm, timing relative to the first arm."""
    arms = report["arms"]
    print(f"== {report['config']}  runs={report['runs_timed']}")
    for arm in arms:
        paired = arm["paired_delta_pct"]
        print(
            f"  {json.dumps(arm['settings']):48s} best {arm['best_ms']:9.3f}"
            f" ms paired delta {arm['median_paired_delta_pct']:+6.2f}%"
            f" [{min(paired):+.2f}, {max(paired):+.2f}]"
            f" waves {arm['waves']:.2f} regs {arm['registers']}"
            f" local {arm['local_bytes']}"
        )
        print(
            f"      failed {arm['failed_runs']} {arm['flags']}"
            f" counters {arm['counters_mean']}"
            f" vs base {arm['counters_vs_base']}"
        )
        print(
            f"      deviation/tol vs base {arm['deviation_vs_base']}"
            f"  error/tol vs tol/10 reference {arm['error_vs_reference']}"
        )
        print(f"      sass {arm['sass']}")


def run_census(name, settings, n_runs):
    """Warm the kernel, then profile one solve."""
    solver, entry = build_solver(name, settings)
    solve(solver, entry, n_runs)
    resources = kernel_resources(solver.kernel.kernel)
    print("resources", resources, flush=True)
    cuda.synchronize()
    cuda.profile_start()
    solve(solver, entry, n_runs)
    cuda.synchronize()
    cuda.profile_stop()
    solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    census = sub.add_parser("census")
    census.add_argument("config", choices=sorted(CONFIGS))
    census.add_argument("--settings", default="{}")
    census.add_argument("--runs", type=int, default=32768)
    measure = sub.add_parser("measure")
    measure.add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    measure.add_argument("--arms", required=True,
                         help="JSON list of settings dicts; first is base")
    measure.add_argument("--runs", type=int, default=16384,
                         help="trajectories in the scoring solves")
    measure.add_argument("--rounds", type=int, default=3)
    measure.add_argument("--target-ms", type=float, default=200.0)
    measure.add_argument("--json", default=None,
                         help="append each report to this JSON-lines file")
    args = parser.parse_args()
    if args.mode == "census":
        run_census(args.config, json.loads(args.settings), args.runs)
    elif args.mode == "measure":
        arms = json.loads(args.arms)
        for config in args.configs:
            report = run_measure(
                config, arms, args.runs, args.rounds, args.target_ms
            )
            print_report(report)
            if args.json:
                with open(args.json, "a") as handle:
                    handle.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
