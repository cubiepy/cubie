"""Symbolic system factories shared across the test-suite.

This module centralises the symbolic ODE problems used by the test fixtures so
that integrator tests, CPU reference implementations, and batch solver tests
operate on the same definitions.  Each system exposes lightweight NumPy helper
functions for evaluating derivatives and Jacobians.  These helpers provide
fast reference evaluations that mirror the behaviour of the compiled device
functions.
"""

import warnings
from math import cos, sin  # noqa: F401 — used inside ODE callables
from typing import Sequence, Union

import sympy as sp
from numpy import (
    asarray as np_asarray,
    dtype as np_dtype,
    floating as np_floating,
)
from numpy.typing import NDArray

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

Array = NDArray[np_floating]


def _as_array(vector: Union[Sequence[float], Array], dt: np_dtype) -> Array:
    """Return ``vector`` as a one-dimensional array of ``dt``.

    Parameters
    ----------
    vector
        Sequence of floats to convert.
    dt
        NumPy dtype for the output array.

    Returns
    -------
    Array
        One-dimensional array with dtype ``dt``.
    """

    arr = np_asarray(vector, dtype=dt)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return arr


THREE_STATE_LINEAR_EQUATIONS = [
    "dx0 = -x0",
    "dx1 = -x1/2",
    "dx2 = -x2/3",
    "o0 = dx0 * p0 + c0 + d0",
    "o1 = dx1 * p1 + c1 + d0",
    "o2 = dx2 * p2 + c2 + d0",
]

THREE_STATE_LINEAR_STATES = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
THREE_STATE_LINEAR_PARAMETERS = {"p0": 1.0, "p1": 2.0, "p2": 3.0}
THREE_STATE_LINEAR_CONSTANTS = {"c0": 0.5, "c1": 1.0, "c2": 2.0}
THREE_STATE_LINEAR_DRIVERS = ["d0"]
THREE_STATE_LINEAR_OBSERVABLES = ["o0", "o1", "o2"]


def build_three_state_linear_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic three-state linear system."""

    system = create_ODE_system(
        dxdt=THREE_STATE_LINEAR_EQUATIONS,
        states=THREE_STATE_LINEAR_STATES,
        parameters=THREE_STATE_LINEAR_PARAMETERS,
        constants=THREE_STATE_LINEAR_CONSTANTS,
        drivers=THREE_STATE_LINEAR_DRIVERS,
        observables=THREE_STATE_LINEAR_OBSERVABLES,
        precision=precision,
        name="three_state_linear",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three-state nonlinear system
# ---------------------------------------------------------------------------

THREE_STATE_NONLINEAR_EQUATIONS = [
    "dx0 = p0 * (x1 - x0**3) + d0",
    "dx1 = p1 * x0 * x2 - x1 + c1",
    "dx2 = -p2 * x2 + c2 * tanh(x0)",
    "o0 = x0 + c0",
    "o1 = x1**2 + p1",
    "o2 = x2 + d0",
]

THREE_STATE_NONLINEAR_STATES = {"x0": 0.5, "x1": -0.25, "x2": 1.2}
THREE_STATE_NONLINEAR_PARAMETERS = {"p0": 0.7, "p1": 0.9, "p2": 1.1}
THREE_STATE_NONLINEAR_CONSTANTS = {"c0": 0.5, "c1": -0.3, "c2": 0.25}
THREE_STATE_NONLINEAR_DRIVERS = ["d0"]
THREE_STATE_NONLINEAR_OBSERVABLES = ["o0", "o1", "o2"]


def build_three_state_nonlinear_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic three-state nonlinear system."""

    system = create_ODE_system(
        dxdt=THREE_STATE_NONLINEAR_EQUATIONS,
        states=THREE_STATE_NONLINEAR_STATES,
        parameters=THREE_STATE_NONLINEAR_PARAMETERS,
        constants=THREE_STATE_NONLINEAR_CONSTANTS,
        drivers=THREE_STATE_NONLINEAR_DRIVERS,
        observables=THREE_STATE_NONLINEAR_OBSERVABLES,
        precision=precision,
        name="three_state_nonlinear",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three chamber cardiovascular system (ThreeCM replacement)
# ---------------------------------------------------------------------------

THREE_CHAMBER_EQUATIONS = [
    "P_a = E_a * V_a",
    "P_v = E_v * V_v",
    "P_h = E_h * V_h * d1",
    "Q_i = (P_v - P_h) / R_i if P_v > P_h else 0",
    "Q_o = (P_h - P_a) / R_o if P_h > P_a else 0",
    "Q_c = (P_a - P_v) / R_c",
    "dV_h = Q_i - Q_o",
    "dV_a = Q_o - Q_c",
    "dV_v = Q_c - Q_i",
]

THREE_CHAMBER_STATES = {"V_h": 1.0, "V_a": 1.0, "V_v": 1.0}
THREE_CHAMBER_PARAMETERS = {
    "E_h": 0.52,
    "E_a": 0.0133,
    "E_v": 0.0624,
    "R_i": 0.012,
    "R_o": 1.0,
    "R_c": 1.0 / 114.0,
    "V_s3": 2.0,
}
THREE_CHAMBER_CONSTANTS: dict[str, float] = {}
THREE_CHAMBER_DRIVERS = ["d1"]
THREE_CHAMBER_OBSERVABLES = ["P_a", "P_v", "P_h", "Q_i", "Q_o", "Q_c"]


def build_three_chamber_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic three chamber cardiovascular system."""

    system = create_ODE_system(
        dxdt=THREE_CHAMBER_EQUATIONS,
        states=THREE_CHAMBER_STATES,
        parameters=THREE_CHAMBER_PARAMETERS,
        constants=THREE_CHAMBER_CONSTANTS,
        drivers=THREE_CHAMBER_DRIVERS,
        observables=THREE_CHAMBER_OBSERVABLES,
        precision=precision,
        name="three_chamber_system",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three-state very stiff nonlinear system
# ---------------------------------------------------------------------------

THREE_STATE_VERY_STIFF_EQUATIONS = [
    "dx0 = -k1 * (x0 - x1) - n0 * x0**3 + d0",
    "dx1 = k1 * (x0 - x1) - k2 * (x1 - x2) - n1 * x1**3",
    "dx2 = k2 * (x1 - x2) - k3 * (x2 - c0) - n2 * x2**3",
    "r0 = x0 - x1",
    "r1 = x1 - x2",
    "r2 = x0 + x1 + x2",
]

THREE_STATE_VERY_STIFF_STATES = {"x0": 0.5, "x1": 0.25, "x2": 0.1}
THREE_STATE_VERY_STIFF_PARAMETERS = {
    "k1": 150.0,
    "k2": 900.0,
    "k3": 1200.0,
    "n0": 40.0,
    "n1": 30.0,
    "n2": 20.0,
}
THREE_STATE_VERY_STIFF_CONSTANTS = {"c0": 0.5}
THREE_STATE_VERY_STIFF_DRIVERS = ["d0"]
THREE_STATE_VERY_STIFF_OBSERVABLES = ["r0", "r1", "r2"]


def build_three_state_very_stiff_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic very stiff nonlinear system."""

    system = create_ODE_system(
        dxdt=THREE_STATE_VERY_STIFF_EQUATIONS,
        states=THREE_STATE_VERY_STIFF_STATES,
        parameters=THREE_STATE_VERY_STIFF_PARAMETERS,
        constants=THREE_STATE_VERY_STIFF_CONSTANTS,
        drivers=THREE_STATE_VERY_STIFF_DRIVERS,
        observables=THREE_STATE_VERY_STIFF_OBSERVABLES,
        precision=precision,
        name="three_state_very_stiff",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Coupled nonlinear systems (20 and 100 states, same structure)
# ---------------------------------------------------------------------------


def _coupled_nonlinear_equations(size: int) -> list[str]:
    """Generate symbolic equations for a coupled nonlinear system."""

    equations = []
    for idx in range(size):
        nxt = (idx + 1) % size
        equations.append(
            "dx{idx} = -p{idx}*x{idx} + c{nxt}*sin(x{nxt}) + "
            "0.01*x{idx}*x{nxt} + d0/{denom}".format(
                idx=idx, nxt=nxt, denom=idx + 1
            )
        )
    return equations


def _coupled_nonlinear_values(size: int) -> tuple[dict, dict, dict]:
    states = {f"x{i}": 0.1 + 0.01 * i for i in range(size)}
    parameters = {f"p{i}": 0.5 + 0.005 * i for i in range(size)}
    constants = {
        f"c{i}": ((-1) ** i) * (0.01 + 0.002 * i) for i in range(size)
    }
    return states, parameters, constants


LARGE_SYSTEM_EQUATIONS = _coupled_nonlinear_equations(100)
(
    LARGE_SYSTEM_STATES,
    LARGE_SYSTEM_PARAMETERS,
    LARGE_SYSTEM_CONSTANTS,
) = _coupled_nonlinear_values(100)
LARGE_SYSTEM_DRIVERS = ["d0"]

MEDIUM_SYSTEM_EQUATIONS = _coupled_nonlinear_equations(20)
(
    MEDIUM_SYSTEM_STATES,
    MEDIUM_SYSTEM_PARAMETERS,
    MEDIUM_SYSTEM_CONSTANTS,
) = _coupled_nonlinear_values(20)
MEDIUM_SYSTEM_DRIVERS = ["d0"]


def build_medium_nonlinear_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic 20-state nonlinear system."""

    system = create_ODE_system(
        dxdt=MEDIUM_SYSTEM_EQUATIONS,
        states=MEDIUM_SYSTEM_STATES,
        parameters=MEDIUM_SYSTEM_PARAMETERS,
        constants=MEDIUM_SYSTEM_CONSTANTS,
        drivers=MEDIUM_SYSTEM_DRIVERS,
        precision=precision,
        name="medium_nonlinear_system",
        strict=True,
    )

    return system


HODGKIN_HUXLEY_EQUATIONS = [
    "alpha_m = 0.1*(vm + 40.0)/(1.0 - exp(-(vm + 40.0)/10.0))",
    "beta_m = 4.0*exp(-(vm + 65.0)/18.0)",
    "alpha_h = 0.07*exp(-(vm + 65.0)/20.0)",
    "beta_h = 1.0/(1.0 + exp(-(vm + 35.0)/10.0))",
    "alpha_n = 0.01*(vm + 55.0)/(1.0 - exp(-(vm + 55.0)/10.0))",
    "beta_n = 0.125*exp(-(vm + 65.0)/80.0)",
    "dvm = (i_app - g_na*m**3*hg*(vm - e_na)"
    " - g_k*n**4*(vm - e_k) - g_l*(vm - e_l))/c_m",
    "dm = alpha_m*(1.0 - m) - beta_m*m",
    "dhg = alpha_h*(1.0 - hg) - beta_h*hg",
    "dn = alpha_n*(1.0 - n) - beta_n*n",
]

HODGKIN_HUXLEY_STATES = {"vm": -62.0, "m": 0.07, "hg": 0.55, "n": 0.34}
HODGKIN_HUXLEY_CONSTANTS = {
    "i_app": 10.0,
    "g_na": 120.0,
    "e_na": 50.0,
    "g_k": 36.0,
    "e_k": -77.0,
    "g_l": 0.3,
    "e_l": -54.4,
    "c_m": 1.0,
}


def build_hodgkin_huxley_system(precision: np_dtype) -> BaseODE:
    """Return the 4-state Hodgkin-Huxley system with exp-heavy rates."""

    system = create_ODE_system(
        dxdt=HODGKIN_HUXLEY_EQUATIONS,
        states=HODGKIN_HUXLEY_STATES,
        constants=HODGKIN_HUXLEY_CONSTANTS,
        precision=precision,
        name="hodgkin_huxley",
    )

    return system


def build_large_nonlinear_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic 100-state nonlinear system."""

    system = create_ODE_system(
        dxdt=LARGE_SYSTEM_EQUATIONS,
        states=LARGE_SYSTEM_STATES,
        parameters=LARGE_SYSTEM_PARAMETERS,
        constants=LARGE_SYSTEM_CONSTANTS,
        drivers=LARGE_SYSTEM_DRIVERS,
        precision=precision,
        name="large_nonlinear_system",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three-state constant derivative system (all algorithms reduce to Euler)
# ---------------------------------------------------------------------------

THREE_STATE_CONSTANT_DERIV_EQUATIONS = [
    "dx0 = c0",
    "dx1 = c1",
    "dx2 = c2",
    "o0 = x0 + p0",
    "o1 = x1 + p1",
    "o2 = x2 + p2",
]

THREE_STATE_CONSTANT_DERIV_STATES = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
THREE_STATE_CONSTANT_DERIV_PARAMETERS = {"p0": 1.0, "p1": 2.0, "p2": 3.0}
THREE_STATE_CONSTANT_DERIV_CONSTANTS = {"c0": 1.0, "c1": 2.0, "c2": 3.0}
THREE_STATE_CONSTANT_DERIV_DRIVERS = []
THREE_STATE_CONSTANT_DERIV_OBSERVABLES = ["o0", "o1", "o2"]


def build_three_state_constant_deriv_system(precision: np_dtype) -> BaseODE:
    """Return a system with constant derivatives.

    For this system, dx/dt = constant (independent of state), which means
    all higher-order Taylor terms vanish. Therefore, all numerical
    integration algorithms (Euler, RK4, etc.) produce identical results,
    making it ideal for testing algorithm parity.
    """

    system = create_ODE_system(
        dxdt=THREE_STATE_CONSTANT_DERIV_EQUATIONS,
        states=THREE_STATE_CONSTANT_DERIV_STATES,
        parameters=THREE_STATE_CONSTANT_DERIV_PARAMETERS,
        constants=THREE_STATE_CONSTANT_DERIV_CONSTANTS,
        drivers=THREE_STATE_CONSTANT_DERIV_DRIVERS,
        observables=THREE_STATE_CONSTANT_DERIV_OBSERVABLES,
        precision=precision,
        name="three_state_constant_deriv",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Two-driver linear system
# ---------------------------------------------------------------------------

TWO_DRIVER_EQUATIONS = [
    "du0 = d_a",
    "du1 = d_b",
]

TWO_DRIVER_STATES = {"u0": 0.0, "u1": 0.0}
TWO_DRIVER_DRIVERS = ["d_a", "d_b"]


def build_two_driver_system(precision: np_dtype) -> BaseODE:
    """Return the symbolic two-driver linear system.

    Each state derivative tracks a distinct driver, so driver-to-column
    alignment is observable directly in the trajectories.
    """

    system = create_ODE_system(
        dxdt=TWO_DRIVER_EQUATIONS,
        states=TWO_DRIVER_STATES,
        drivers=TWO_DRIVER_DRIVERS,
        precision=precision,
        name="two_driver_linear",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Driverless decoupled system
# ---------------------------------------------------------------------------


def build_diagonally_dominant_system(precision: np_dtype) -> BaseODE:
    """Return a decoupled, strongly diagonal two-state system."""

    system = create_ODE_system(
        dxdt=["dx = -10.0 * x", "dy = -10.0 * y"],
        states={"x": 1.0, "y": 1.0},
        precision=precision,
        name="diagonally_dominant",
    )

    return system


# ---------------------------------------------------------------------------
# Solver-scaling constant collision system
# ---------------------------------------------------------------------------

COLLIDING_CONSTANTS_EQUATIONS = [
    "dx0 = -beta * x0 + gamma * x1",
    "dx1 = -gamma * x1",
]

COLLIDING_CONSTANTS_STATES = {"x0": 1.0, "x1": 2.0}
COLLIDING_CONSTANTS = {"beta": 2.5, "gamma": 0.75}


def build_colliding_constants_system(precision: np_dtype) -> BaseODE:
    """Return a system whose constants share solver-scaling names."""

    system = create_ODE_system(
        dxdt=COLLIDING_CONSTANTS_EQUATIONS,
        states=COLLIDING_CONSTANTS_STATES,
        constants=COLLIDING_CONSTANTS,
        precision=precision,
        name="colliding_constants",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Hostile-name system: one constant per class of generated binding
# ---------------------------------------------------------------------------

# Factory arguments (beta, gamma, order), factory locals (n, total_n,
# stage_width, beta_inv, h_eff_factor, precision), FIRK tableau
# metadata (c_0, a_0_0), and generated body locals (dx_0).
HOSTILE_NAME_CONSTANTS = {
    "beta": 8.0 / 3.0,
    "gamma": 0.5,
    "order": 2.0,
    "n": 1.5,
    "total_n": 0.25,
    "stage_width": 0.125,
    "beta_inv": 0.75,
    "h_eff_factor": 0.375,
    "precision": 1.25,
    "c_0": 0.6,
    "a_0_0": 0.3,
    "dx_0": 0.2,
}

HOSTILE_NAME_EQUATION = (
    "dx = -(beta + gamma + n + dx_0)*x - 0.1*x**order"
    " + c_0 + a_0_0 + beta_inv + h_eff_factor"
    " + total_n + stage_width + precision"
)

HOSTILE_NAME_STATES = {"x": 2.0}

SAFE_NAME_CONSTANTS = {
    f"k_{index}": value
    for index, value in enumerate(HOSTILE_NAME_CONSTANTS.values())
}

SAFE_NAME_EQUATION = (
    "dx = -(k_0 + k_1 + k_3 + k_11)*x - 0.1*x**k_2"
    " + k_9 + k_10 + k_6 + k_7 + k_4 + k_5 + k_8"
)


def build_hostile_names_system(precision: np_dtype) -> BaseODE:
    """Return a system whose constants shadow every generated binding."""

    system = create_ODE_system(
        HOSTILE_NAME_EQUATION,
        states=dict(HOSTILE_NAME_STATES),
        constants=dict(HOSTILE_NAME_CONSTANTS),
        precision=precision,
        name="hostile_names",
    )

    return system


def build_safe_names_system(precision: np_dtype) -> BaseODE:
    """Return the hostile-name system with unremarkable names."""

    system = create_ODE_system(
        SAFE_NAME_EQUATION,
        states=dict(HOSTILE_NAME_STATES),
        constants=dict(SAFE_NAME_CONSTANTS),
        precision=precision,
        name="safe_names",
    )

    return system


# ---------------------------------------------------------------------------
# Lorenz system pinned by the Julia golden-reference gate
# ---------------------------------------------------------------------------

LORENZ_JULIA_EQUATIONS = [
    "dx = sigma * (y - x)",
    "dy = x * (rho - z) - y",
    "dz = x * y - beta * z",
]

LORENZ_JULIA_STATES = {"x": 1.0, "y": 0.0, "z": 0.0}
LORENZ_JULIA_PARAMETERS = {"rho": 21.0}
LORENZ_JULIA_CONSTANTS = {"sigma": 10.0, "beta": 8.0 / 3.0}


def build_lorenz_julia_system(precision: np_dtype) -> BaseODE:
    """Return the Lorenz system used by the Julia reference gate."""

    system = create_ODE_system(
        dxdt=LORENZ_JULIA_EQUATIONS,
        states=LORENZ_JULIA_STATES,
        parameters=LORENZ_JULIA_PARAMETERS,
        constants=LORENZ_JULIA_CONSTANTS,
        precision=precision,
        name="lorenz_julia",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Coupled oscillator with piecewise damping (save-schedule hang test)
# ---------------------------------------------------------------------------


def _coupled_oscillator_dxdt(t, y, p):
    """Two coupled springs with piecewise velocity-dependent damping."""
    x1, v1, x2, v2 = y[0], y[1], y[2], y[3]
    k = p["k"]
    c_couple = p["c_couple"]
    omega = p["omega"]
    damp1 = 0.5 * v1 if v1 * v1 > 0.01 else 0.0
    damp2 = 0.5 * v2 if v2 * v2 > 0.01 else 0.0
    drive = sin(omega * t)
    dx1 = v1
    dv1 = -k * x1 - damp1 + c_couple * (x2 - x1) + drive
    dx2 = v2
    dv2 = -k * x2 - damp2 + c_couple * (x1 - x2)
    return [dx1, dv1, dx2, dv2]


def build_coupled_oscillator_system(precision: np_dtype) -> BaseODE:
    """Return the driven coupled oscillator with piecewise damping.

    An adaptive solver on this system is squeezed into very small
    steps right at a save boundary (with k=3.0 supplied at solve
    time), which none of the other shared systems provoke.
    """

    return create_ODE_system(
        dxdt=_coupled_oscillator_dxdt,
        states={"x1": 1.0, "v1": 0.0, "x2": -0.5, "v2": 0.0},
        parameters={"k": 4.0, "c_couple": 0.3, "omega": 2.5},
        precision=precision,
        name="coupled_oscillator",
    )


# ---------------------------------------------------------------------------
# Moderately stiff two-state system (status-staining recovery test)
# ---------------------------------------------------------------------------


def _status_staining_stiff_dxdt(t, y, p):
    """A stiff two-state system requiring an implicit inner solve."""
    x0, x1 = y[0], y[1]
    k = p["k"]
    dx0 = -k * (x0 - cos(x1))
    dx1 = -x1 + x0
    return [dx0, dx1]


def build_status_staining_stiff_system(precision: np_dtype) -> BaseODE:
    """Return the moderately stiff system for transient-recovery tests.

    The recovery scenario needs an oversized first step to exhaust a
    two-iteration Krylov budget while reduced steps converge. The
    nonlinear and three-chamber systems converge even at the initial
    step and the very stiff system never converges under the budget,
    so neither exhibits a recoverable transient failure.
    """

    return create_ODE_system(
        dxdt=_status_staining_stiff_dxdt,
        states={"x0": 1.0, "x1": 0.0},
        parameters={"k": 500.0},
        precision=precision,
        name="status_staining_stiff",
    )


# ---------------------------------------------------------------------------
# Sinusoid-driven twins: the drive as an equation and as a spline
# ---------------------------------------------------------------------------

# The two systems must stay identical up to where the drive comes
# from: the interpolated twin is compared against the function twin,
# so any other difference invalidates the comparison. The driver
# symbol name is part of the shared contract, because driver settings
# key off ``system.indices.driver_names``.
TIME_DRIVER_STATES = {"x": 0.5}
TIME_FUNCTION_DRIVER_EQUATIONS = ["dx = -x + sin(t)", "obs = x"]
TIME_ARRAY_DRIVER_EQUATIONS = ["dx = -x + drive", "obs = x"]


def build_time_function_driver_system(precision: np_dtype) -> BaseODE:
    """Return the twin whose sinusoid lives in the equations."""

    return create_ODE_system(
        dxdt=TIME_FUNCTION_DRIVER_EQUATIONS,
        states=dict(TIME_DRIVER_STATES),
        observables=["obs"],
        precision=precision,
        strict=True,
        name="time_function_driver",
    )


def build_time_array_driver_system(precision: np_dtype) -> BaseODE:
    """Return the twin whose sinusoid arrives as an interpolated
    driver."""

    return create_ODE_system(
        dxdt=TIME_ARRAY_DRIVER_EQUATIONS,
        states=dict(TIME_DRIVER_STATES),
        observables=["obs"],
        drivers=["drive"],
        precision=precision,
        strict=True,
        name="time_array_driver",
    )


__all__ = [
    "build_colliding_constants_system",
    "build_coupled_oscillator_system",
    "build_status_staining_stiff_system",
    "build_lorenz_julia_system",
    "build_two_driver_system",
    "build_three_state_linear_system",
    "build_three_state_nonlinear_system",
    "build_three_chamber_system",
    "build_three_state_very_stiff_system",
    "build_large_nonlinear_system",
    "build_three_state_constant_deriv_system",
    "build_diagonally_dominant_system",
    "build_hostile_names_system",
    "build_safe_names_system",
    "build_time_function_driver_system",
    "build_time_array_driver_system",
    "build_torn_driver_system",
    "build_torn_time_system",
    "build_torn_unsolvable_system",
]
# ---------------------------------------------------------------------------
# Torn DAE twins (mass diag(1, 0)); quintic residuals keep x1 torn
# ---------------------------------------------------------------------------

TORN_DRIVER_CONSTANTS = {"a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9}

TORN_TIME_CONSTANTS = {
    "a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9, "e": 0.8,
}


def build_torn_driver_system(precision: np_dtype) -> BaseODE:
    """Torn two-state DAE whose Jacobian depends on a driver."""

    return create_ODE_system(
        dxdt=[
            "dx0 = a*x0*x1 + b*x1 + d0*x0",
            "0 = c*x0*x0 + d*x1 + d0*x1 + x1**5",
        ],
        states=["x0", "x1"],
        constants=TORN_DRIVER_CONSTANTS,
        drivers=["d0"],
        precision=precision,
        name="torn_driver",
    )


def build_torn_time_system(precision: np_dtype) -> BaseODE:
    """Driverless torn DAE whose Jacobian depends on time."""

    return create_ODE_system(
        dxdt=[
            "dx0 = a*x0*x1 + b*x1 + e*t*x0",
            "0 = c*x0*x0 + d*x1 + x1**5",
        ],
        states=["x0", "x1"],
        constants=TORN_TIME_CONSTANTS,
        precision=precision,
        name="torn_time",
    )


def build_torn_unsolvable_system(precision: np_dtype) -> BaseODE:
    """Torn DAE whose constraint has no real root below x0 = 2."""

    return create_ODE_system(
        dxdt=[
            "dx0 = -x1",
            "0 = x1*x1 + cos(x1) + 1 - x0",
        ],
        states=["x0", "x1"],
        precision=precision,
        name="torn_unsolvable",
    )


RING_MODULATOR_CONSTANTS = {
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

RING_MODULATOR_STATES = (
    "U1", "U2", "U3", "U4", "U5", "U6", "U7",
    "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8",
)

RING_MODULATOR_AUXILIARIES = """
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

RING_MODULATOR_DIFFERENTIAL_ROWS = """
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

RING_MODULATOR_EQUATIONS = RING_MODULATOR_AUXILIARIES + """
0 = I3 - qD1 + qD4
0 = -I4 + qD2 - qD3
0 = I5 + qD1 - qD3
0 = -I6 - qD2 + qD4
""" + RING_MODULATOR_DIFFERENTIAL_ROWS

RING_MODULATOR_SCALED_EQUATIONS = RING_MODULATOR_AUXILIARIES + """
Cs * dU3 = I3 - qD1 + qD4
Cs * dU4 = -I4 + qD2 - qD3
Cs * dU5 = I5 + qD1 - qD3
Cs * dU6 = -I6 - qD2 + qD4
""" + RING_MODULATOR_DIFFERENTIAL_ROWS


def _build_ring_modulator(equations, constants, system_name, precision):
    return create_ODE_system(
        equations,
        states={name: 0.0 for name in RING_MODULATOR_STATES},
        parameters={"Uin1_amplitude": 0.5},
        constants=constants,
        observables=["U3", "U4", "U6", "I3"],
        precision=precision,
        name=system_name,
    )


def build_ring_modulator_index2_system(precision: np_dtype) -> BaseODE:
    """Index-2 ring modulator (Test Set II-3, Cs = 0)."""

    return _build_ring_modulator(
        RING_MODULATOR_EQUATIONS,
        RING_MODULATOR_CONSTANTS,
        "ring_modulator_index2",
        precision,
    )


def build_ring_modulator_index2_scaled_system(
    precision: np_dtype,
) -> BaseODE:
    """Ring modulator in ``Cs*dX = ...`` form with ``Cs = 0``."""

    return _build_ring_modulator(
        RING_MODULATOR_SCALED_EQUATIONS,
        dict(RING_MODULATOR_CONSTANTS, Cs=0.0),
        "ring_modulator_index2_scaled",
        precision,
    )


SCALED_CS_EQUATIONS = """
Cs*dU3 = I3 - 0.5*I1
dI1 = -U3 - 0.2*I1
dI3 = U3 - 0.1*I3
"""

SCALED_CS_STATES = {"U3": 0.0, "I1": 0.1, "I3": 0.2}


def build_scaled_cs_system(precision: np_dtype) -> BaseODE:
    """``Cs*dU3`` system at ``Cs = 0``; flips explicit when
    ``Cs`` is nonzero."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_ODE_system(
            SCALED_CS_EQUATIONS,
            states=dict(SCALED_CS_STATES),
            constants={"Cs": 0.0},
            precision=precision,
            name="scaled_cs",
        )


def build_amp_constant_system(precision: np_dtype) -> BaseODE:
    """One-state decay system with folded constant ``amp``."""

    return create_ODE_system(
        "dx = -k * x * (1.0 + amp)",
        states={"x": 1.0},
        parameters={"k": 0.5},
        constants={"amp": 2.0},
        precision=precision,
        name="amp_constant",
    )


DIODE_LINE_N = 8

DIODE_LINE_CONSTANTS = {"gs": 0.1, "a": 3.0, "c": 0.5}


def _diode_line_equations() -> str:
    lines = ["drive = amp * sin(6.283185307179586 * t)"]
    for i in range(1, DIODE_LINE_N + 1):
        tau = 10.0 ** (-2.0 * (i - 1) / (DIODE_LINE_N - 1))
        lines.append(f"dv{i} = (w{i} - v{i}) / {tau!r}")
    for i in range(1, DIODE_LINE_N + 1):
        upstream = f"w{i + 1}" if i < DIODE_LINE_N else "drive"
        lines.append(
            f"0 = (v{i} - w{i}) - gs * (exp(a * w{i}) - "
            f"exp(-a * w{i})) + c * ({upstream} - w{i})"
        )
    return "\n".join(lines)


def build_diode_line_system(precision: np_dtype) -> BaseODE:
    """Semi-explicit index-1 diode ladder; the chain-boundary
    constraint slot does not contain its own variable."""

    states = {f"v{i}": 0.0 for i in range(1, DIODE_LINE_N + 1)}
    states.update(
        {f"w{i}": 0.0 for i in range(1, DIODE_LINE_N + 1)}
    )
    return create_ODE_system(
        _diode_line_equations(),
        states=states,
        parameters={"amp": 1.0},
        constants=dict(DIODE_LINE_CONSTANTS),
        precision=precision,
        name="diode_line",
    )


TRANSAMP_CONSTANTS = {
    "ub": 6.0,
    "uf": 0.026,
    "alfa": 0.99,
    "bb": 1.0e-6,
    "r0": 1000.0,
    "r1": 9000.0,
    "r2": 9000.0,
    "r3": 9000.0,
    "r4": 9000.0,
    "r5": 9000.0,
    "r6": 9000.0,
    "r7": 9000.0,
    "r8": 9000.0,
    "r9": 9000.0,
    "c1": 1.0e-6,
    "c2": 2.0e-6,
    "c3": 3.0e-6,
    "c4": 4.0e-6,
    "c5": 5.0e-6,
}

TRANSAMP_EQUATIONS = """
Ue = 0.1 * sin(628.3185307179587 * t)
g23 = bb * (exp((y2 - y3) / uf) - 1.0)
g56 = bb * (exp((y5 - y6) / uf) - 1.0)
-c1*dy1 + c1*dy2 = -Ue / r0 + y1 / r0
c1*dy1 - c1*dy2 = -ub / r2 + y2 * (1.0/r1 + 1.0/r2) - (alfa - 1.0) * g23
-c2*dy3 = -g23 + y3 / r3
-c3*dy4 + c3*dy5 = -ub / r4 + y4 / r4 + alfa * g23
c3*dy4 - c3*dy5 = -ub / r6 + y5 * (1.0/r5 + 1.0/r6) - (alfa - 1.0) * g56
-c4*dy6 = -g56 + y6 / r7
-c5*dy7 + c5*dy8 = -ub / r8 + y7 / r8 + alfa * g56
c5*dy7 - c5*dy8 = y8 / r9
"""

TRANSAMP_DC_STATES = {
    "y1": 0.0,
    "y2": 3.0,
    "y3": 3.0,
    "y4": 6.0,
    "y5": 3.0,
    "y6": 3.0,
    "y7": 6.0,
    "y8": 0.0,
}


def build_transistor_amplifier_system(precision: np_dtype) -> BaseODE:
    """Test Set transistor amplifier (II-2) at its DC point."""

    return create_ODE_system(
        TRANSAMP_EQUATIONS,
        states=dict(TRANSAMP_DC_STATES),
        observables=["y1", "y4", "y7"],
        constants=dict(TRANSAMP_CONSTANTS),
        precision=precision,
        name="transistor_amplifier",
    )


def build_toggle_system(precision: np_dtype) -> BaseODE:
    """SymPy-input system whose constant ``tog`` picks a branch."""

    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True)
    tog = sp.Symbol("tog", real=True)
    dx = sp.Symbol("dx", real=True)
    equations = [
        (dx, sp.Piecewise((-k * x, tog > 0.5), (-2 * k * x, True)))
    ]
    return create_ODE_system(
        equations,
        states={"x": 1.0},
        parameters={"k": 0.3},
        constants={"tog": 1.0},
        precision=precision,
        name="toggle",
    )
