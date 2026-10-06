"""Race driver-derivative evaluator variants through cubie.Solver.

Usage: python race.py <system> <mode> <out_dir> [rounds] [block]

Checks each variant's states against ``columns``, dumps its kernel
SASS, and reports the median per-round delta against ``columns``.
"""

import collections
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import cubie as qb
from cubie.CUDAFactory import UnrollChoice
from cubie.array_interpolator import DriverSamples
from cubie._cudasim_extensions import cuda

TOOLS = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
VARIANTS = tuple(
    __import__("os").environ.get(
        "RACE_VARIANTS", "columns,combined,combined_selp,two_loop"
    ).split(",")
)
MODES = {"unrolled": UnrollChoice.FULL, "rolled": UnrollChoice.ROLLED}
precision = np.float32
n_samples = 127
period = 2.0 * np.pi / (n_samples - 1)
sample_times = np.arange(n_samples) * period


def periodic(values):
    values = np.asarray(values, dtype=float)
    values[-1] = values[0]
    return values


SYSTEMS = {
    "one_drive": dict(
        dxdt=["dx = y", "dy = w", "0 = x - drive", "dz = -z + y + w"],
        states={"z": 0.5},
        observables=["x", "y", "w"],
        drivers=["drive"],
        samples={"drive": periodic(np.sin(sample_times))},
    ),
    "three_drives": dict(
        dxdt=[
            "dx1 = y1",
            "0 = x1 - a",
            "dx2 = y2",
            "dy2 = w2",
            "0 = x2 - b",
            "dz = -z + y1 + w2 + c",
        ],
        states={"z": 0.5},
        observables=["x1", "y1", "x2", "y2", "w2"],
        drivers=["a", "b", "c"],
        samples={
            "a": periodic(np.sin(sample_times)),
            "b": periodic(np.cos(2.0 * sample_times)),
            "c": periodic(np.sin(3.0 * sample_times)),
        },
    ),
}


def sass_summary(solver, path):
    """Write the integration kernel's cubin and SASS; count ops."""
    (overload,) = solver.kernel.kernel.overloads.values()
    library = overload._codelibrary
    cubin = (
        bytes(library.get_cubin().code)
        if hasattr(library, "get_cubin")
        else library._cubin
    )
    path.write_bytes(cubin)
    sass = subprocess.run(
        [rf"{TOOLS}\nvdisasm.exe", "-c", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    path.with_suffix(".sass").write_text(sass)
    usage = subprocess.run(
        [rf"{TOOLS}\cuobjdump.exe", "-res-usage", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    regs = re.search(r"REG:(\d+)", usage).group(1)
    local = re.search(r"LOCAL:(\d+)", usage).group(1)
    stack = re.search(r"STACK:(\d+)", usage).group(1)
    ops = collections.Counter()
    for line in sass.splitlines():
        match = re.match(
            r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)", line
        )
        if match:
            ops[match.group(1)] += 1
    return {
        "regs": int(regs),
        "local_B": int(local),
        "stack_B": int(stack),
        "instr": sum(ops.values()),
        "FFMA": ops["FFMA"],
        "FMUL": ops["FMUL"],
        "FADD": ops["FADD"],
        "BRA": ops["BRA"],
        "FSEL": ops["FSEL"],
        "LDG": ops["LDG"],
        "LDC": ops["LDC"] + ops["ULDC"],
        "LDL": ops["LDL"],
        "STL": ops["STL"],
    }


def kernel_ms(solver):
    return sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )


def main(system_name, mode, out_dir, rounds=8, block=12):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = dict(SYSTEMS[system_name])
    samples = spec.pop("samples")
    system = qb.create_ODE_system(
        **spec, precision=precision, strict=True, name=system_name
    )
    print(
        f"SYSTEM {system_name} slots={system.driver_derivative_columns} "
        f"buffer={system.num_drivers}"
    )
    drivers = DriverSamples(samples, driver_sample_period=period)
    solvers = {}
    for variant in VARIANTS:
        solvers[variant] = qb.Solver(
            system,
            algorithm="rosenbrock",
            step_controller="fixed",
            dt=0.01,
            save_every=1.0,
            output_types=["state"],
            drivers=drivers,
            wrap=True,
            boundary_condition="periodic",
            driver_evaluation=variant,
            unroll_other_small=MODES[mode],
            time_logging_level="default",
        )
        settings = solvers[variant].kernel.driver_interpolator
        assert settings.compile_settings.driver_evaluation == variant
        assert (
            settings.compile_settings.unroll.unroll_other_small
            == MODES[mode].value
        )

    # Two or more full waves at every variant's occupancy.
    n_runs = 2**18
    duration = 10.0 * np.pi
    grid = solvers["columns"].build_grid(
        initial_values={"z": np.linspace(0.0, 1.0, n_runs)},
        parameters={},
    )
    finals = {}
    for variant, solver in solvers.items():
        with contextlib.redirect_stdout(io.StringIO()):
            result = solver.solve(*grid, duration=duration)
        finals[variant] = np.array(result.time_domain_array, copy=True)
        sass = sass_summary(
            solver, out_dir / f"{system_name}_{mode}_{variant}.cubin"
        )
        (overload,) = solver.kernel.kernel.overloads.values()
        cufunc = overload._codelibrary.get_cufunc()
        blocksize, dynshared = solver.kernel.launch_geometry(
            64, runs=n_runs
        )
        per_sm = cuda.current_context().get_active_blocks_per_multiprocessor(
            cufunc, blocksize, dynshared
        )
        sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
        runs_per_block = blocksize // (
            solver.kernel.single_integrator.threads_per_step
        )
        waves = n_runs / (sms * per_sm * runs_per_block)
        print(
            f"SASS {system_name} {mode} {variant} "
            + " ".join(f"{k}={v}" for k, v in sass.items())
            + f" blocks_per_sm={per_sm} waves={waves:.2f}"
        )
        if waves < 2.0:
            raise SystemExit(f"{variant}: fewer than two waves")
    reference = finals["columns"]
    for variant, final in finals.items():
        error = np.max(np.abs(final - reference) / (1.0 + np.abs(reference)))
        print(f"CHECK {system_name} {mode} {variant} max_rel={error:.2e}")

    if rounds == 0:
        return

    def run_block(solver):
        times = []
        for _ in range(block):
            with contextlib.redirect_stdout(io.StringIO()):
                solver.solve(*grid, duration=duration)
            times.append(kernel_ms(solver))
        return float(np.mean(np.sort(times)[:3]))

    for solver in solvers.values():
        run_block(solver)  # warm
    per_round = []
    order = list(VARIANTS)
    for round_index in range(rounds):
        shift = round_index % len(order)
        rotated = order[shift:] + order[:shift]
        if round_index % 2:
            rotated = rotated[::-1]
        stats = {variant: run_block(solvers[variant]) for variant in rotated}
        per_round.append(stats)
        print(
            f"ROUND {round_index} "
            + " ".join(f"{v}={stats[v]:.3f}" for v in VARIANTS)
        )
    for variant in VARIANTS:
        deltas = [
            100.0 * (r[variant] - r["columns"]) / r["columns"]
            for r in per_round
        ]
        floor = np.median([r[variant] for r in per_round])
        print(
            f"RESULT {system_name} {mode} {variant} "
            f"kernel_ms={floor:.3f} median_delta={np.median(deltas):+.2f}% "
            f"min={min(deltas):+.2f}% max={max(deltas):+.2f}%"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], args[2], *(int(a) for a in args[3:]))
