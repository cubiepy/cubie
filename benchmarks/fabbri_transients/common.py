"""Shared setup for the Fabbri-Linder transient analysis.

ANS is on, so ``ACh_cas`` and ``Iso_cas`` (nM) drive the autonomic
branches; they are the swept parameters. Solves use rosenbrock23,
``atol = rtol = 1e-6``, float32.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

import cubie as qb

REPO = Path(__file__).resolve().parents[2]
FABBRI_CELLML = (
    REPO / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)
RESULTS = Path(__file__).resolve().parent / "results"

PRECISION = np.float32
ACH = "Rate_modulation_experiments_ACh_cas"
ISO = "Rate_modulation_experiments_Iso_cas"
ANS = "Rate_modulation_experiments_ANS"
VOLTAGE = "Membrane$V_ode"
VOLTAGE_LABEL = "Membrane_V_ode"

GRID_SIDE = 256
# Axes span ~2-3x the cascade half-saturations (43.5, 58.6 nM).
ACH_RANGE = (0.0, 100.0)
ISO_RANGE = (0.0, 200.0)

# Step bounds from linear_solver_grid.py's fabbri entry.
SOLVER_SETTINGS = {
    "algorithm": "rosenbrock23",
    "atol": 1e-6,
    "rtol": 1e-6,
    "dt_min": 1e-12,
    "dt_max": 1e-2,
}


def build_system():
    """Return the Fabbri-Linder system with ACh/Iso as parameters."""
    return qb.load_cellml_model(
        str(FABBRI_CELLML),
        precision=PRECISION,
        parameters=[ACH, ISO],
        voltage_variable=VOLTAGE,
    )


def axes(side: int = GRID_SIDE, ach_range=ACH_RANGE, iso_range=ISO_RANGE,
         log: bool = False):
    """Return the ACh and Iso axis values for a ``side`` x ``side`` grid."""
    space = np.geomspace if log else np.linspace
    return space(*ach_range, side), space(*iso_range, side)


def grid_values(side: int = GRID_SIDE, **axis_settings):
    """Return flattened ACh/Iso values, ACh varying fastest."""
    ach, iso = axes(side, **axis_settings)
    ach_grid, iso_grid = np.meshgrid(ach, iso)
    return ach_grid.ravel(), iso_grid.ravel()


def parameter_array(system, ach, iso):
    """Return a ``(n_parameters, n_runs)`` parameter array."""
    names = list(system.parameters.names)
    params = np.empty((len(names), ach.size), dtype=PRECISION)
    params[names.index(ACH)] = ach
    params[names.index(ISO)] = iso
    return params


def initial_state_array(system, n_runs: int):
    """Return the model's initial state repeated for ``n_runs`` runs."""
    values = np.asarray(system.initial_values.values_array, PRECISION)
    return np.repeat(values[:, None], n_runs, axis=1)


def make_solver(system, **settings):
    """Return a Solver with ANS switched on and the shared settings."""
    kwargs = dict(SOLVER_SETTINGS)
    kwargs.update(settings)
    solver = qb.Solver(system, **kwargs)
    solver.update({ANS: 1.0})
    return solver


def timed_solve(solver, inits, params, duration, **kwargs):
    """Run one solve; return ``(result, kernel_ms, wall_s)``."""
    start = perf_counter()
    result = solver.solve(
        initial_values=inits,
        parameters=params,
        duration=duration,
        **kwargs,
    )
    wall_s = perf_counter() - start
    kernel_ms = sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )
    return result, kernel_ms, wall_s


def failed_runs(result):
    """Return a boolean mask of runs with a nonzero status flag."""
    codes = np.asarray(result.status_codes).ravel()
    return (codes & 0xFFFF) != 0


def dense_peaks(time, voltage):
    """Return peak times of densely sampled traces, parabola-refined.

    Parameters
    ----------
    time
        ``(n_samples,)`` sample times shared by every run.
    voltage
        ``(n_samples, n_runs)`` sampled voltages.

    Returns
    -------
    list of ndarray
        Per run, local-maximum sample indices under the ``peaks``
        metric's rule, and parabola-refined peak times.
    """
    time = np.asarray(time, dtype=np.float64)
    indices = []
    refined = []
    for run in range(voltage.shape[1]):
        trace = voltage[:, run]
        ends = np.append(
            np.flatnonzero(trace[1:] != trace[:-1]), trace.size - 1
        )
        values = trace[ends]
        is_peak = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
        end = ends[1:-1][is_peak]
        start = ends[:-2][is_peak] + 1
        indices.append(end)
        refined.append(
            parabola_vertex(
                time[start - 1], 0.5 * (time[start] + time[end]),
                time[end + 1], trace[start - 1], trace[end],
                trace[end + 1],
            )
        )
    return indices, refined


def parabola_vertex(x0, x1, x2, y0, y1, y2):
    """Return the abscissa of the parabola vertex through three points."""
    y0, y1, y2 = (np.asarray(y, np.float64) for y in (y0, y1, y2))
    left = x1 - x0
    right = x1 - x2
    num = left**2 * (y1 - y2) - right**2 * (y1 - y0)
    den = left * (y1 - y2) - right * (y1 - y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(den != 0, 0.5 * num / den, 0.0)
    # Keep the vertex between the outer samples.
    return np.clip(x1 - shift, x0, x2)


def float32_sample_grid(interval, n_samples, t_start=0.0, t_end=None):
    """Return the time of each summary update ``k``.

    The loop accumulates ``min(next + interval, t_end)`` in float32.
    """
    step = np.float32(interval)
    end = np.float32(np.inf if t_end is None else t_end)
    times = np.empty(n_samples, dtype=np.float32)
    current = np.float32(t_start)
    for k in range(n_samples):
        current = np.minimum(np.float32(current + step), end)
        times[k] = current
    return times.astype(np.float64)
