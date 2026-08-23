#!/usr/bin/env python
"""Linear-solver panel evaluation on DAE benchmark systems.

Three systems: the NAND gate (Test Set for IVP Solvers; float32,
14 differential + 8 algebraic rows after simplification), the
index-2 ring modulator (Test Set II-3, Cs = 0; float64 only), and
its non-degenerate ODE twin (Cs > 0) for the ODE-vs-DAE comparison
on the same circuit. Per implicit algorithm the panel crosses the
linear correction (lu, bicgstab, minimal_residual) with the Newton
variants each family supports, all under jacobi-0 preconditioning
(neumann is rejected on singular mass), plus a krylov-cap axis on
BiCGSTAB where the stacked width exceeds the 50-iteration floor.

Configurations in one group block-interleave over a shared input
grid as in ``linear_solver_grid.py``: block statistic = mean of the
``min_count`` lowest per-solve kernel times, reported per config
with failure counts and a final-state deviation against the group's
reference config, so a fast-but-wrong candidate is visible. Rows
append to ``--csv`` as each group completes. Requires a real GPU;
exits under the CUDA simulator.

Usage::

    python benchmarks/dae_linear_solver_eval.py
        [--systems nand_gate ring_modulator_index2 ring_modulator]
        [--algorithms ...] [--rounds R] [--block N] [--min-count K]
        [--n-runs N] [--csv PATH] [--smoke]
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import cubie as qb
from cubie.cuda_simsafe import CUDA_SIMULATION

from linear_solver_grid import (
    achieved_waves,
    default_n_runs,
    failure_summary,
    lowest_mean,
    solve_once,
)

# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

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

RING_STATES = (
    "U1", "U2", "U3", "U4", "U5", "U6", "U7",
    "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8",
)

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


def build_ring_modulator_index2(precision):
    """Index-2 ring modulator, Cs = 0; the amplitude is swept."""
    equations = RING_AUXILIARIES + """
    0 = I3 - qD1 + qD4
    0 = -I4 + qD2 - qD3
    0 = I5 + qD1 - qD3
    0 = -I6 - qD2 + qD4
""" + RING_COMMON
    return qb.create_ODE_system(
        equations,
        states={name: 0.0 for name in RING_STATES},
        parameters={"Uin1_amplitude": 0.5},
        constants=dict(RING_CONSTANTS),
        name="ring_modulator_index2",
        precision=precision,
    )


def build_ring_modulator(precision):
    """Stiff ODE twin; the swept Cs stays above zero."""
    equations = RING_AUXILIARIES + """
    dU3 = (I3 - qD1 + qD4) / Cs
    dU4 = (-I4 + qD2 - qD3) / Cs
    dU5 = (I5 + qD1 - qD3) / Cs
    dU6 = (-I6 - qD2 + qD4) / Cs
""" + RING_COMMON
    return qb.create_ODE_system(
        equations,
        states={name: 0.0 for name in RING_STATES},
        parameters={"Cs": 2.0e-12},
        constants=dict(
            RING_CONSTANTS, Uin1_amplitude=0.5
        ),
        name="ring_modulator",
        precision=precision,
    )


# Diode ladder: float32-viable semi-explicit index-1 DAE. Eight RC
# nodes (differential v_i, spread time constants) each hang off an
# algebraic mid-node w_i carrying a diode pair to ground and
# resistive coupling along the line; the head is driven sinusoidally.
# The constraint Jacobian diagonal stays below -(1 + c) everywhere.
DIODE_N = 8

DIODE_CONSTANTS = {"gs": 0.1, "a": 3.0, "c": 0.5}


def _diode_line_equations(algebraic):
    """Return the ladder equations; stiff-capacitor rows otherwise."""
    lines = ["drive = amp * sin(6.283185307179586 * t)"]
    for i in range(1, DIODE_N + 1):
        tau = 10.0 ** (-2.0 * (i - 1) / (DIODE_N - 1))
        lines.append(f"dv{i} = (w{i} - v{i}) / {tau!r}")
    for i in range(1, DIODE_N + 1):
        upstream = f"w{i + 1}" if i < DIODE_N else "drive"
        residual = (
            f"(v{i} - w{i}) - gs * (exp(a * w{i}) - "
            f"exp(-a * w{i})) + c * ({upstream} - w{i})"
        )
        if algebraic:
            lines.append(f"0 = {residual}")
        else:
            lines.append(f"cw * dw{i} = {residual}")
    return "\n".join(lines)


def _build_diode_line(precision, algebraic, name):
    states = {f"v{i}": 0.0 for i in range(1, DIODE_N + 1)}
    states.update({f"w{i}": 0.0 for i in range(1, DIODE_N + 1)})
    constants = dict(DIODE_CONSTANTS)
    if not algebraic:
        constants["cw"] = 1.0e-6
    return qb.create_ODE_system(
        _diode_line_equations(algebraic),
        states=states,
        parameters={"amp": 1.0},
        constants=constants,
        name=name,
        precision=precision,
    )


def build_diode_line(precision):
    """Semi-explicit index-1 ladder; the amplitude is swept."""
    return _build_diode_line(precision, True, "diode_line")


def build_diode_line_ode(precision):
    """Stiff ODE twin with capacitance ``cw`` on the mid-nodes."""
    return _build_diode_line(precision, False, "diode_line_ode")

def ring_index2_grid(solver, n_runs):
    """Return the input-amplitude sweep for ``n_runs``."""
    return solver.build_grid(
        parameters={
            "Uin1_amplitude": np.linspace(0.0, 0.5, n_runs)
        },
    )


def ring_ode_grid(solver, n_runs):
    """Return the log Cs sweep for ``n_runs`` trajectories."""
    return solver.build_grid(
        parameters={
            "Cs": np.logspace(
                math.log10(2.0e-13), math.log10(2.0e-9), n_runs
            )
        },
    )


def diode_grid(solver, n_runs):
    """Return the drive-amplitude sweep for ``n_runs``."""
    return solver.build_grid(
        parameters={"amp": np.linspace(0.5, 2.0, n_runs)},
    )


DIODE_SOLVER_KWARGS = {
    "atol": 1e-6,
    "rtol": 1e-4,
    "dt": 1e-3,
    "dt_min": 1e-10,
    "dt_max": 0.1,
}

SYSTEMS = {
    "diode_line": {
        "build": build_diode_line,
        "grid": diode_grid,
        "precision": np.float32,
        "duration": 4.0,
        "solver_kwargs": dict(DIODE_SOLVER_KWARGS),
    },
    "diode_line_ode": {
        "build": build_diode_line_ode,
        "grid": diode_grid,
        "precision": np.float32,
        "duration": 4.0,
        "solver_kwargs": dict(DIODE_SOLVER_KWARGS),
    },
    "ring_modulator_index2": {
        "build": build_ring_modulator_index2,
        "grid": ring_index2_grid,
        "precision": np.float64,
        "duration": 1e-3,
        "solver_kwargs": {
            "atol": 1e-9,
            "rtol": 1e-7,
            "dt": 1e-7,
            "dt_min": 1e-13,
            "dt_max": 1e-4,
        },
    },
    "ring_modulator": {
        "build": build_ring_modulator,
        "grid": ring_ode_grid,
        "precision": np.float64,
        "duration": 1e-3,
        "solver_kwargs": {
            "atol": 1e-9,
            "rtol": 1e-7,
            "dt": 1e-9,
            "dt_min": 1e-14,
            "dt_max": 1e-4,
        },
    },
}

# backwards_euler is errorless (fixed dt) and impractical at these
# durations; crank_nicolson stands in for the one-stage family.
ALGORITHMS = (
    "radau_iia_5",
    "l_stable_dirk_3",
    "rosenbrock23",
    "crank_nicolson",
)


def panel(algorithm: str, solver_width: int) -> List[dict]:
    """Return the (label, settings) panel for one algorithm."""
    jacobi = {
        "preconditioner_type": "jacobi",
        "preconditioner_order": 0,
    }
    if algorithm == "rosenbrock23":
        return [
            {"label": "lu",
             "settings": {"linear_correction_type": "lu"}},
            {"label": "bicgstab jacobi-0",
             "settings": {"linear_correction_type": "bicgstab",
                          **jacobi}},
            {"label": "mr jacobi-0",
             "settings": {
                 "linear_correction_type": "minimal_residual",
                 **jacobi}},
        ]
    specs = [
        {"label": "lu exact",
         "settings": {"linear_correction_type": "lu",
                      "inexact_newton": False,
                      "prefactored": False}},
        {"label": "lu prefactored",
         "settings": {"linear_correction_type": "lu",
                      "inexact_newton": True,
                      "prefactored": True}},
        {"label": "bicgstab jacobi-0 exact",
         "settings": {"linear_correction_type": "bicgstab",
                      "inexact_newton": False, **jacobi}},
        {"label": "bicgstab jacobi-0 inexact",
         "settings": {"linear_correction_type": "bicgstab",
                      "inexact_newton": True, **jacobi}},
        {"label": "mr jacobi-0 exact",
         "settings": {
             "linear_correction_type": "minimal_residual",
             "inexact_newton": False, **jacobi}},
    ]
    if algorithm == "l_stable_dirk_3":
        specs.insert(
            2,
            {"label": "lu inexact",
             "settings": {"linear_correction_type": "lu",
                          "inexact_newton": True,
                          "prefactored": False}},
        )
    if 4 * solver_width > 50:
        specs.append(
            {"label": "bicgstab jacobi-0 exact cap4x",
             "settings": {"linear_correction_type": "bicgstab",
                          "inexact_newton": False,
                          "krylov_max_iters": 4 * solver_width,
                          **jacobi}},
        )
    return specs


def build_solver(system, algorithm, spec, settings):
    """Return one configured solver for a panel row."""
    kwargs = dict(spec["solver_kwargs"])
    kwargs.update(settings)
    kwargs.update(
        algorithm=algorithm,
        output_types=["state"],
        time_logging_level="default",
    )
    if algorithm == "backwards_euler":
        kwargs.pop("atol", None)
        kwargs.pop("rtol", None)
    return qb.Solver(system, **kwargs)


def final_state_deviation(result, reference) -> float:
    """Worst per-state scaled rms deviation between final states."""
    got = np.asarray(result.time_domain_array)[-1]
    ref = np.asarray(reference.time_domain_array)[-1]
    ok = np.isfinite(got) & np.isfinite(ref)
    if not ok.any():
        return float("nan")
    scale = np.nanmax(np.abs(ref), axis=-1, keepdims=True) + 1e-12
    dev = np.abs(got - ref) / scale
    return float(np.nanmax(np.where(ok, dev, 0.0)))


def run_group(system_name, spec, algorithm, args, writer, csv_file):
    """Measure one system x algorithm panel."""
    precision = spec["precision"]
    system = spec["build"](precision)
    n_runs = args.n_runs or default_n_runs()
    duration = spec["duration"]

    # Radau stacks three stages into one solve; others solve width n.
    n = system.sizes.states
    stacked_width = 3 * n if algorithm == "radau_iia_5" else n

    entries = []
    width = None
    inits = params = None
    for row in panel(algorithm, stacked_width):
        try:
            solver = build_solver(
                system, algorithm, spec, row["settings"]
            )
        except (ValueError, KeyError) as exc:
            print(f"  {row['label']}: rejected ({exc})")
            continue
        if width is None:
            width = int(
                solver.kernel.single_integrator._algo_step
                .compile_settings.solver_width
            )
            inits, params = spec["grid"](solver, n_runs)
        entries.append((row["label"], solver))
    if not entries:
        print(f"\n=== {system_name} x {algorithm}: no legal configs")
        return

    print(
        f"\n=== {system_name} x {algorithm} "
        f"(width {width}, {n_runs} runs) ==="
    )
    warm = {}
    for label, solver in entries:
        kernel_ms, wall_ms, result = solve_once(
            solver, inits, params, duration
        )
        waves = achieved_waves(solver, n_runs)
        failed, flags = failure_summary(result)
        warm[label] = (waves, failed, flags)
        print(
            f"  warmed {label}: {wall_ms / 1000.0:.1f} s, "
            f"{waves:.1f} waves, {failed} failed"
        )

    kernel_stats: Dict[str, List[float]] = {
        label: [] for label, _ in entries
    }
    deviation: Dict[str, float] = {}
    reference_result: Dict[int, object] = {}
    for round_index in range(args.rounds):
        ordered = (
            entries if round_index % 2 == 0
            else list(reversed(entries))
        )
        for label, solver in ordered:
            samples = []
            result = None
            for _ in range(args.block):
                kernel_ms, _, result = solve_once(
                    solver, inits, params, duration
                )
                samples.append(kernel_ms)
            kernel_stats[label].append(
                lowest_mean(samples, args.min_count)
            )
            if round_index == 0:
                if not reference_result:
                    reference_result[0] = result
                deviation[label] = final_state_deviation(
                    result, reference_result[0]
                )

    reference_ms = None
    for label, _ in entries:
        stat = float(np.median(kernel_stats[label]))
        if reference_ms is None:
            reference_ms = stat
        delta = 100.0 * (stat - reference_ms) / reference_ms
        waves, failed, flags = warm[label]
        flags_text = ",".join(
            f"{k}:{v}" for k, v in sorted(flags.items())
        )
        print(
            f"  {label:34s} {stat:10.2f} ms  {delta:+7.2f}%  "
            f"failed {failed:6d}  dev {deviation.get(label, 0.0):.2e}"
            f"  {flags_text}"
        )
        writer.writerow(
            {
                "system": system_name,
                "algorithm": algorithm,
                "config": label,
                "kernel_ms": stat,
                "delta_pct": delta,
                "failed": failed,
                "deviation": deviation.get(label, float("nan")),
                "waves": waves,
                "flags": flags_text,
            }
        )
        csv_file.flush()


def main(argv: Optional[Sequence[str]] = None) -> None:
    if CUDA_SIMULATION:
        print("Requires a real GPU; exiting under CUDASIM.")
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS),
        choices=list(SYSTEMS),
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=list(ALGORITHMS),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument(
        "--rtol", type=float, default=None,
        help="Override every system's rtol.",
    )
    parser.add_argument(
        "--atol", type=float, default=None,
        help="Override every system's atol.",
    )
    parser.add_argument(
        "--precision", choices=["float32", "float64"], default=None,
        help="Override every system's precision.",
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("dae_linear_solver_eval.csv"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.rounds = 1
        args.block = 2
        args.min_count = 1
        args.n_runs = args.n_runs or 128

    fresh = not args.csv.exists()
    with args.csv.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "system", "algorithm", "config", "kernel_ms",
                "delta_pct", "failed", "deviation", "waves",
                "flags",
            ],
        )
        if fresh:
            writer.writeheader()
        for system_name in args.systems:
            spec = dict(SYSTEMS[system_name])
            spec["solver_kwargs"] = dict(spec["solver_kwargs"])
            if args.rtol is not None:
                spec["solver_kwargs"]["rtol"] = args.rtol
            if args.atol is not None:
                spec["solver_kwargs"]["atol"] = args.atol
            if args.precision is not None:
                spec["precision"] = getattr(np, args.precision)
            for algorithm in args.algorithms:
                try:
                    run_group(
                        system_name, spec, algorithm, args,
                        writer, csv_file,
                    )
                except (ValueError, KeyError) as exc:
                    print(
                        f"\n=== {system_name} x {algorithm} "
                        f"rejected: {exc}"
                    )


if __name__ == "__main__":
    main()
