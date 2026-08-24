#!/usr/bin/env python
"""Collect calibration outcomes and system features for heuristics.

Runs :meth:`cubie.Solver.calibrate` on the benchmark systems and
appends every candidate measurement, joined with the system feature
record, to a CSV for offline feature-to-winner analysis.

Usage::

    python benchmarks/calibration_features.py [--systems ...]
        [--n-runs N] [--csv PATH]
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import cubie as qb
from cubie.cuda_simsafe import CUDA_SIMULATION

REPO = Path(__file__).resolve().parent.parent
FABBRI_CELLML = (
    REPO / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)

precision = np.float32

def build_lorenz_system():
    """Return the ab-gate Lorenz system."""
    return qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"sigma": 10.0, "rho": 21.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=precision,
    )


def build_lorenz96_system(n: int):
    """Return cyclic Lorenz 96 (GPUODEBenchmarks) with ``n`` states."""
    lines = []
    for i in range(1, n + 1):
        ip1 = i % n + 1
        im1 = (i - 2) % n + 1
        im2 = (i - 3) % n + 1
        lines.append(
            "dx{0} = (x{1} - x{2}) * x{3} - x{0} + F".format(
                i, ip1, im2, im1
            )
        )
    states = {
        "x{0}".format(i): 9.0 if i == 1 else 8.0
        for i in range(1, n + 1)
    }
    return qb.create_ODE_system(
        "\n".join(lines),
        states=states,
        parameters={"F": 8.0},
        name=f"Lorenz96_{n}",
        precision=precision,
    )


def build_fabbri_system():
    """Return the Fabbri-Linder sinoatrial system."""
    return qb.load_cellml_model(
        str(FABBRI_CELLML),
        precision=precision,
        voltage_variable="Membrane$V_ode",
    )


def lorenz_grid(n_runs: int):
    """Return the ab-gate rho sweep inputs."""
    return (
        {"x": 1.0, "y": 0.0, "z": 0.0},
        {"rho": np.linspace(0.0, 21.0, n_runs)},
    )


def lorenz96_grid(n_runs: int):
    """Return the GPUODEBenchmarks F sweep inputs."""
    return None, {"F": np.linspace(0.0, 16.0, n_runs)}


def fabbri_grid(n_runs: int):
    """Return an ACh/Iso grid for ``n_runs`` trajectories."""
    side = int(np.ceil(np.sqrt(n_runs)))
    ach, iso = np.meshgrid(
        np.linspace(0.0, 2e-8, side),
        np.linspace(0.0, 1.0, side),
    )
    return None, {
        "Rate_modulation_experiments_ACh": ach.ravel()[:n_runs],
        "Rate_modulation_experiments_Iso_cas": iso.ravel()[:n_runs],
    }


# Tolerances and durations match the linear-solver timing grid.
SYSTEMS = {
    "lorenz": {
        "build": build_lorenz_system,
        "grid": lorenz_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-06,
            "dt_min": 1e-12,
            "dt_max": 1e3,
        },
    },
    "lorenz96_10": {
        "build": lambda: build_lorenz96_system(10),
        "grid": lorenz96_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-06,
            "dt_min": 1e-12,
            "dt_max": 1e3,
        },
    },
    "lorenz96_20": {
        "build": lambda: build_lorenz96_system(20),
        "grid": lorenz96_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-06,
            "dt_min": 1e-12,
            "dt_max": 1e3,
        },
    },
    "fabbri": {
        "build": build_fabbri_system,
        "grid": fabbri_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-04,
            "dt_min": 1e-12,
            "dt_max": 1e-2,
        },
        "constants": {"Rate_modulation_experiments_ANS": 1.0},
    },
}


def append_records(path: Path, records: Sequence[dict]) -> None:
    """Append calibration records to the CSV, writing a header once."""
    if not records:
        return
    fields = sorted({key for record in records for key in record})
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args(argv: Optional[Sequence[str]] = None):
    """Return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(SYSTEMS),
        default=tuple(SYSTEMS),
    )
    parser.add_argument("--n-runs", type=int, default=4096)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("calibration_features.csv"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run calibration over the requested systems and append rows."""
    if CUDA_SIMULATION:
        raise SystemExit(
            "calibration_features.py measures kernel time on a real "
            "GPU; unset NUMBA_ENABLE_CUDASIM"
        )
    args = parse_args(argv)
    for name in args.systems:
        spec = SYSTEMS[name]
        print(f"=== {name} ===", flush=True)
        system = spec["build"]()
        solver = qb.Solver(
            system,
            algorithm="rosenbrock23",
            output_types=["state"],
            **spec["solver_kwargs"],
        )
        constants = spec.get("constants")
        if constants:
            solver.update(constants)
        initial_values, parameters = spec["grid"](args.n_runs)
        try:
            report = solver.calibrate(
                initial_values,
                parameters,
                duration=spec["duration"],
                grid_type="combinatorial",
                apply=False,
            )
        finally:
            solver.close()
        print(report.summary(), flush=True)
        append_records(args.csv, report.to_records())
    print(f"records appended to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
