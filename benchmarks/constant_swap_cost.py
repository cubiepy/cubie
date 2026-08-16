"""Measure the per-constant-change cost of a cubie system.

Alternates one constant between two values on a single live solver
and records, per change, ``specialise_ms`` (set_constants),
``codegen_ms`` (TimeLogger codegen category), ``cold_wall_ms`` (first
solve after the change, which rebuilds through the factory chain),
``compile_ms`` (cold minus warm minus codegen), and warm wall/kernel
times. Models: ``ring-value``, ``ring-structural`` (needs the
specialisation architecture), ``fabbri-toggle``, ``fabbri-value``.

Usage::

    python benchmarks/constant_swap_cost.py --model ring-value \
        --out ring_value.json --cycles 4 --warm 3

Run once per architecture (PYTHONPATH selects the checkout) with a
fresh ``CUBIE_CACHE_DIR`` per run.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

RING_CONSTANTS = {
    "C": 1.6e-8,
    "Cp": 1.0e-8,
    "Lh": 4.45,
    "Ls1": 0.002,
    "Ls2": 5.0e-4,
    "Ls3": 5.0e-4,
    "gamma": 40.67286402e-9,
    "R": 25000.0,
    "Rp": 50.0,
    "Rg1": 36.3,
    "Rg2": 17.3,
    "Rg3": 17.3,
    "Ri": 50.0,
    "Rc": 600.0,
    "delta": 17.7493332,
    "w1": 6283.185307179586,
    "w2": 62831.85307179586,
}

RING_STATES = {
    name: 0.0
    for name in (
        "U1", "U2", "U3", "U4", "U5", "U6", "U7",
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8",
    )
}

RING_AUXILIARIES = """
    Uin1 = Uin1_amplitude * sin(w1 * t)
    Uin2 = 2.0 * sin(w2 * t)
    UD1 = U3 - U5 - U7 - Uin2
    UD2 = -U4 + U6 - U7 - Uin2
    UD3 = U4 + U5 + U7 + Uin2
    UD4 = -U3 - U6 + U7 + Uin2
    qD1 = gamma * (exp(delta * UD1) - 1.0)
    qD2 = gamma * (exp(delta * UD2) - 1.0)
    qD3 = gamma * (exp(delta * UD3) - 1.0)
    qD4 = gamma * (exp(delta * UD4) - 1.0)
"""

RING_COMMON = """
    dU1 = (I1 - 0.5 * I3 + 0.5 * I4 + I7 - U1 / R) / C
    dU2 = (I2 - 0.5 * I5 + 0.5 * I6 + I8 - U2 / R) / C
    dU7 = (-U7 / Rp + qD1 + qD2 - qD3 - qD4) / Cp
    dI1 = -U1 / Lh
    dI2 = -U2 / Lh
    dI3 = (0.5 * U1 - U3 - Rg2 * I3) / Ls2
    dI4 = (-0.5 * U1 + U4 - Rg3 * I4) / Ls3
    dI5 = (0.5 * U2 - U5 - Rg2 * I5) / Ls2
    dI6 = (-0.5 * U2 + U6 - Rg3 * I6) / Ls3
    dI7 = (-U1 + Uin1 - (Ri + Rg1) * I7) / Ls1
    dI8 = (-U2 - (Rc + Rg1) * I8) / Ls1
"""

RING_EXPLICIT = RING_AUXILIARIES + """
    0 = I3 - qD1 + qD4
    0 = -I4 + qD2 - qD3
    0 = I5 + qD1 - qD3
    0 = -I6 - qD2 + qD4
""" + RING_COMMON

RING_SCALED = RING_AUXILIARIES + """
    Cs * dU3 = I3 - qD1 + qD4
    Cs * dU4 = -I4 + qD2 - qD3
    Cs * dU5 = I5 + qD1 - qD3
    Cs * dU6 = -I6 - qD2 + qD4
""" + RING_COMMON


def build_ring(equations, constants, name):
    from cubie import create_ODE_system

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_ODE_system(
            equations,
            states=dict(RING_STATES),
            parameters={"Uin1_amplitude": 0.5},
            constants=constants,
            observables=["U3", "U4", "U6", "I3"],
            precision=np.float64,
            simplify=True,
            name=name,
        )


def build_fabbri():
    from cubie import load_cellml_model

    path = REPO_ROOT / "tests" / "fixtures" / "cellml"
    path = path / "Fabbri_Linder.cellml"
    return load_cellml_model(
        str(path),
        precision=np.float32,
        name="fabbri_swap_cost",
    )


MODELS = {
    "ring-value": {
        "build": lambda: build_ring(
            RING_EXPLICIT, dict(RING_CONSTANTS), "ring_swap_value"
        ),
        "constant": "R",
        "values": (25000.0, 26000.0),
        "solver": {
            "algorithm": "backwards_euler",
            "preconditioner_type": "jacobi",
            "linear_correction_type": "bicgstab",
            "dt": 1e-7,
        },
        "solve": {"duration": 2e-6},
        "runs": 8,
        "structural": False,
    },
    "ring-structural": {
        "build": lambda: build_ring(
            RING_SCALED,
            dict(RING_CONSTANTS, Cs=0.0),
            "ring_swap_structural",
        ),
        "constant": "Cs",
        "values": (0.0, 2e-9),
        "solver": {
            "algorithm": "backwards_euler",
            "preconditioner_type": "jacobi",
            "linear_correction_type": "bicgstab",
            "dt": 1e-7,
        },
        "solve": {"duration": 2e-6},
        "runs": 8,
        "structural": True,
    },
    "fabbri-toggle": {
        "build": build_fabbri,
        "constant": "Rate_modulation_experiments_Iso_1_uM",
        "values": (0.0, 1.0),
        "solver": {
            "algorithm": "backwards_euler",
            "preconditioner_type": "jacobi",
            "linear_correction_type": "bicgstab",
            "dt": 1e-5,
        },
        "solve": {"duration": 1e-3},
        "runs": 64,
        "structural": False,
    },
    "fabbri-value": {
        "build": build_fabbri,
        "constant": "Rate_modulation_experiments_ACh",
        "values": (0.0, 1e-8),
        "solver": {
            "algorithm": "backwards_euler",
            "preconditioner_type": "jacobi",
            "linear_correction_type": "bicgstab",
            "dt": 1e-5,
        },
        "solve": {"duration": 1e-3},
        "runs": 64,
        "structural": False,
    },
}


def make_inputs(system, runs):
    """Duplicate the system's defaults across the batch."""
    inits = {
        str(name): np.full(runs, float(value))
        for name, value in system.initial_values.values_dict.items()
    }
    params = {
        str(name): np.full(runs, float(value))
        for name, value in system.parameters.values_dict.items()
    }
    return inits, params


def kernel_ms(solver):
    """Sum kernel-only CUDA-event times for the last solve."""
    events = getattr(solver.kernel, "_cuda_events", None) or []
    return sum(
        event.elapsed_time_ms()
        for event in events
        if event.name.startswith("kernel_chunk")
    )


def codegen_snapshot(timelogger):
    return sum(
        timelogger.get_aggregate_durations("codegen").values()
    )


def run_model(model_key, cycles, warm, fresh_solver=False):
    from cubie import Solver
    from cubie.time_logger import default_timelogger

    default_timelogger.verbosity = "default"
    # Keep events across solves so codegen deltas are readable.
    default_timelogger._clear_events = lambda: None
    spec = MODELS[model_key]

    t0 = time.perf_counter()
    system = spec["build"]()
    build_ms = 1000.0 * (time.perf_counter() - t0)

    def new_solver():
        # time_logging_level keeps the TimeLogger recording.
        return Solver(
            system,
            time_logging_level="default",
            **spec["solver"],
        )

    def one_solve(solver, inits, params):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.solve(
                initial_values=inits,
                parameters=params,
                **spec["solve"],
            )
        return 1000.0 * (time.perf_counter() - start)

    solver = new_solver()
    inits, params = make_inputs(system, spec["runs"])
    baseline_cold_ms = one_solve(solver, inits, params)
    baseline_warm = [
        one_solve(solver, inits, params) for _ in range(warm)
    ]
    baseline_kernel = kernel_ms(solver)

    records = []
    constant = spec["constant"]
    values = spec["values"]
    for cycle in range(cycles):
        target = values[(cycle + 1) % 2]
        before = codegen_snapshot(default_timelogger)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({constant: target})
        specialise_ms = 1000.0 * (time.perf_counter() - start)

        # --fresh-solver counts construction into the cold wall.
        start = time.perf_counter()
        if fresh_solver:
            solver = new_solver()
        rebuild_ms = 1000.0 * (time.perf_counter() - start)
        inits, params = make_inputs(system, spec["runs"])

        cold_wall_ms = rebuild_ms + one_solve(solver, inits, params)
        codegen_delta_ms = 1000.0 * (
            codegen_snapshot(default_timelogger) - before
        )
        warm_walls = [
            one_solve(solver, inits, params) for _ in range(warm)
        ]
        warm_wall_ms = float(np.mean(warm_walls))
        warm_kernel = kernel_ms(solver)
        records.append(
            {
                "cycle": cycle,
                "constant": constant,
                "value": target,
                "specialise_ms": specialise_ms,
                "codegen_ms": codegen_delta_ms,
                "cold_wall_ms": cold_wall_ms,
                "compile_ms": (
                    cold_wall_ms
                    - warm_wall_ms
                    - codegen_delta_ms
                ),
                "warm_wall_ms": warm_wall_ms,
                "warm_kernel_ms": warm_kernel,
                "n_states": int(system.num_states),
                "fn_hash": system.fn_hash[:12],
            }
        )

    return {
        "model": model_key,
        "runs": spec["runs"],
        "build_ms": build_ms,
        "baseline_cold_ms": baseline_cold_ms,
        "baseline_warm_ms": float(np.mean(baseline_warm)),
        "baseline_kernel_ms": baseline_kernel,
        "changes": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument(
        "--label", default="", help="architecture label for the row"
    )
    parser.add_argument(
        "--fresh-solver",
        action="store_true",
        help=(
            "build a new solver per constant change (for "
            "architectures whose live solver misses the change)"
        ),
    )
    args = parser.parse_args()

    result = run_model(
        args.model,
        args.cycles,
        args.warm,
        fresh_solver=args.fresh_solver,
    )
    result["label"] = args.label
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
