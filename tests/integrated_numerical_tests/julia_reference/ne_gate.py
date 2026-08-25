"""Protocol constants and verdict logic for the Julia reference gate.

Ports the numerical-equivalence protocol shared with GPUODEBenchmarks
(``runner_scripts/numerical_equivalence/ne_common.py`` and
``compare_numerical_equivalence.py``): cubie and raw
DifferentialEquations.jl both integrate the same Float32 Lorenz ensemble
and their final states are judged against a Float64 golden reference
(Vern9 at tol 1e-13) and against each other. The Julia and golden data
are vendored under ``data/`` by ``benchmarks/vendor_julia_reference.py``.

The vendored data covers full dt and tolerance sweeps; the gate solves
each algorithm once, at a single pinned dt (fixed tier) or tolerance
(adaptive tier) selected from the Julia data at the difficult end of
the convergence region. At that pin the truncation error is a steep
function of the method's order — a dropped order inflates it by
``(1/dt)`` per lost power, orders of magnitude beyond the ratio bound —
so one solve detects any drift from the textbook scheme that a full
sweep would.

Keep the constants in sync with the GPUODEBenchmarks protocol; the gate
is only meaningful while both sides integrate bit-identical inputs on
the same grids.
"""

import csv
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Dyadic dt grid, 2^-1 .. 2^-13: every dt, save and end boundary is exact
# in binary floating point (Float32 included). Order >= 5 methods hit the
# float32 error floor by dt ~ 1/32 on this problem, so the convergence
# region is only observable at coarse dt.
DTS_NE = [2.0 ** -k for k in range(1, 14)]

# Adaptive sweep: atol = rtol grid, 1e-2 .. 1e-6. Below 1e-6 a float32
# solve cannot honor the request (the error floor is ~1e-6 relative).
TOLS_NE = [10.0 ** -k for k in range(2, 7)]

# Shared adaptive-run pins (both stacks): initial dt and the dt clamps.
DT0_NE = 0.01
DT_MIN_NE = 1e-6
DT_MAX_NE = 0.5

N_NE = 1024

# Solve-request tolerances for the fixed sweep: OrdinaryDiffEq's
# defaults (abstol = 1e-6, reltol = 1e-3), which the paired Julia
# fixed-step runs used for their nonlinear stage solves.
ATOL_FIXED_NE = 1e-6
RTOL_FIXED_NE = 1e-3

# Error bounds (relative to the golden scale) for selecting reference
# points. Below FLOOR_REL the error is float32 roundoff, and above
# PIN_CAP_REL the selected point is not a useful accuracy comparison.
FLOOR_REL = 4e-6
PIN_CAP_REL = 3e-2

# Cubie and Julia final states must be within this multiple of Julia's
# golden-referenced error.
MATCH_FRACTION = 2.0

# Pins require the Julia error to clear the roundoff floor by this
# multiple, so the matching comparison does not measure floor noise.
PIN_MARGIN_MULT = 4.0


def load_algorithms():
    """Return the mutual algorithm table as a list of dicts.

    Keys: ``cubie_alias``, ``julia_expr``, ``order`` (int), ``family``,
    ``notes``.
    """
    path = os.path.join(DATA_DIR, "algorithms.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["order"] = int(row["order"])
    return rows


def load_controller_constants():
    """Julia's resolved default-controller constants by cubie alias."""
    path = os.path.join(DATA_DIR, "controller_constants.csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {"controller": row["controller"]}
            for key in ("beta1", "beta2", "qmin", "qmax", "gamma",
                        "qsteady_min", "qsteady_max", "order"):
                raw = row.get(key, "")
                entry[key] = float(raw) if raw not in ("", None) else None
            out[row["cubie_alias"]] = entry
    return out


def load_reference():
    """Load the vendored npz; returns the raw archive mapping."""
    path = os.path.join(DATA_DIR, "julia_reference_ne.npz")
    return np.load(path)


def julia_fixed_finals(reference, alias):
    """Vendored Julia fixed-sweep finals; dict {dt: (N_NE, 3) f8}."""
    dts = reference["fixed_{0}_dts".format(alias)]
    finals = reference["fixed_{0}_finals".format(alias)]
    return {float(dt): finals[i].astype(np.float64)
            for i, dt in enumerate(dts)}


def julia_adaptive_finals(reference, alias):
    """Vendored Julia adaptive finals; dict {tol: (N_NE, 3) f8}."""
    key = "adaptive_{0}_tols".format(alias)
    if key not in reference:
        return None
    tols = reference[key]
    finals = reference["adaptive_{0}_finals".format(alias)]
    return {float(tol): finals[i].astype(np.float64)
            for i, tol in enumerate(tols)}


def matched_controller_settings(constants, alias, order):
    """Cubie controller kwargs mirroring Julia's resolved defaults."""
    c = constants.get(alias)
    if c is None:
        return None
    if c["controller"] == "PIController":
        required = ("beta1", "beta2", "gamma", "qmin", "qmax",
                    "qsteady_min", "qsteady_max")
        missing = [key for key in required if c[key] is None]
        if missing:
            raise ValueError(
                "controller constants for '{0}' are missing {1}; "
                "regenerate data/ with "
                "benchmarks/vendor_julia_reference.py".format(
                    alias, ", ".join(missing)))
        return {
            "step_controller": "pi",
            # Julia applies beta as-is; cubie divides by (order+1).
            "filter_coefficients": (
                c["beta1"] * (order + 1),
                -c["beta2"] * (order + 1),
                0.0,
            ),
            "safety": c["gamma"],
            "min_step_factor": c["qmin"],
            "max_step_factor": c["qmax"],
            # Julia's qsteady deadband acts on q = 1/gain.
            "deadband_min": 1.0 / c["qsteady_max"],
            "deadband_max": 1.0 / c["qsteady_min"],
        }
    # PredictiveController maps to gustafsson, matched safety only.
    if c["controller"] == "PredictiveController":
        return {
            "step_controller": "gustafsson",
            "safety": c["gamma"],
        }
    return None


def ensemble_error(final_states, golden_states):
    """l2-at-final error over the ensemble, computed in float64."""
    diff = np.asarray(final_states, dtype=np.float64) - golden_states
    return float(np.sqrt(np.mean(diff ** 2)))


def rms_difference(cubie_final, julia_final):
    """Mutual rms distance between the two final-state ensembles."""
    diff = np.asarray(cubie_final, dtype=np.float64) - julia_final
    return float(np.sqrt(np.mean(diff ** 2)))


def fixed_pin(reference, alias, golden_states, scale):
    """Pinned dt for one algorithm's fixed-tier check.

    Walks the dt grid coarse to fine and returns the finest dt of the
    contiguous run where the Julia error sits inside the convergence
    region: at least PIN_MARGIN_MULT times the roundoff floor and
    below the pin cap. Later dts that dip back into range past a
    break belong to the flat roundoff-dominated tail, not to the
    asymptotic regime the pin must sample. Raises when no dt
    qualifies (the vendored data no longer covers the region).
    """
    floor = FLOOR_REL * scale
    cap = PIN_CAP_REL * scale
    errs = {dt: ensemble_error(final, golden_states)
            for dt, final in julia_fixed_finals(reference, alias).items()}
    pin = None
    for dt in DTS_NE:
        err = errs.get(dt)
        if (err is not None and np.isfinite(err)
                and PIN_MARGIN_MULT * floor <= err < cap):
            pin = dt
        elif pin is not None:
            break
    if pin is None:
        raise ValueError(
            "no in-region dt for '{0}'; regenerate data/ with "
            "benchmarks/vendor_julia_reference.py".format(alias))
    return pin


def adaptive_pin(reference, alias, golden_states, scale):
    """Pinned tolerance for one algorithm's adaptive-tier check.

    Returns the tightest tolerance where the Julia solve's own error
    clears the roundoff margin and is below the pin cap. Raises when no
    tolerance qualifies.
    """
    floor = FLOOR_REL * scale
    cap = PIN_CAP_REL * scale
    errs = {tol: ensemble_error(final, golden_states)
            for tol, final
            in julia_adaptive_finals(reference, alias).items()}
    in_range = [tol for tol in TOLS_NE
                if tol in errs and np.isfinite(errs[tol])
                and PIN_MARGIN_MULT * floor <= errs[tol] < cap]
    if not in_range:
        raise ValueError(
            "no in-range tolerance for '{0}'; regenerate data/ with "
            "benchmarks/vendor_julia_reference.py".format(alias))
    return min(in_range)


def point_matches_julia(cubie_final, julia_final, golden_states):
    """Return whether a pinned Cubie result matches its Julia result."""
    err_c = ensemble_error(cubie_final, golden_states)
    err_j = ensemble_error(julia_final, golden_states)
    rms_diff = rms_difference(cubie_final, julia_final)
    if (not np.isfinite(err_c) or not np.isfinite(err_j)
            or not np.isfinite(rms_diff) or err_j <= 0):
        return False, (
            "invalid comparison values (Julia error must be finite and "
            "positive): err_cubie={0:.3e} err_julia={1:.3e} "
            "rms_diff={2:.3e}".format(err_c, err_j, rms_diff))
    ratio = rms_diff / err_j
    report = (
        "err_cubie={0:.3e} err_julia={1:.3e} rms_diff={2:.3e} "
        "rms/err ratio {3:.3g}, limit {4:g}".format(
            err_c, err_j, rms_diff, ratio, MATCH_FRACTION))
    return ratio <= MATCH_FRACTION, report
