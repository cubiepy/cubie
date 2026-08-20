#!/usr/bin/env python
"""Calibrate dense-prediction step-ratio ceilings on a real GPU.

For every registered DIRK and FIRK tableau and every swept precision,
the sweep measures the largest step-size ratio ``next dt / previous
dt`` at which the compiled dense-prediction transform still produces
a better Newton seed than carrying the previous increments unchanged.
The results populate the ``dense_prediction_ratio_*`` fields on the
registered tableaus; checked-in values are reviewed data from this
script, not values computed at import time.

Per (tableau, precision, probe system, warmup ``dt``) case and forced
ratio, the sweep:

1. runs deterministic accepted steps on the GPU through the actual
   compiled step function, retaining the previous step's converged
   stage increments (the raw, untransformed prediction history);
2. runs one further step at ``ratio * dt`` and retains its converged
   increments as the target seed (solved to tight tolerances so the
   target does not depend on its own seed);
3. invokes the compiled predictor device function on a copy of the
   retained history with the forced ratio and ``apply_flag=True``,
   independent of any stored ceiling;
4. compares the predictor seed and the carry seed (the unchanged
   history) against the target over the rows prediction actually
   seeds (distinct-node implicit stages for DIRK, all stages for
   FIRK);
5. reruns the target step with prediction compiled in and compiled
   out and compares total Newton iterations as an end-to-end check.

A ratio passes when, across every informative probe case, the predictor seed
error does not exceed the carry seed error and prediction does not increase
Newton iterations by more than the convergence checker's one-iteration
quantisation (tolerated only when the predicted seed is strictly closer; solve
failures caused by prediction always fail). Cases where both seeds fail to
converge are uninformative about the seed and are skipped. The ceiling is the
highest ratio reachable from the lowest sampled ratio through passing samples
only, refined around the boundary so the result does not depend on the coarse
grid, then floored to two decimals. A tableau that never passes receives 0.0
(dense prediction disabled at that precision).

float16 is not swept: no implicit-solver path is exercised at
float16 and the seed comparison cannot discriminate at its
tolerances, so float16 ceilings stay 0.0 (disabled).

Usage::

    python benchmarks/dense_prediction_ratio_sweep.py
        [--tableaus NAME [NAME ...]] [--precisions {float32,float64}]
        [--output FILE.md]

Requires a real GPU; exits under the CUDA simulator.
"""

import argparse
import os
import sys
from pathlib import Path

import attrs
import numpy as np

if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    sys.exit("dense_prediction_ratio_sweep requires a real GPU.")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# The checkout's sources take priority over any installed cubie so
# the sweep always calibrates the code it sits next to.
sys.path.insert(0, str(REPO_ROOT / "src"))

from cubie.buffer_registry import buffer_registry  # noqa: E402
from cubie.cuda_simsafe import cuda, int32  # noqa: E402
from cubie.cuda_simsafe import numba_from_dtype as from_dtype  # noqa: E402
from cubie.integrators.algorithms.generic_dirk import DIRKStep  # noqa: E402
from cubie.integrators.algorithms.generic_dirk_tableaus import (  # noqa: E402
    DIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_firk import FIRKStep  # noqa: E402
from cubie.integrators.algorithms.generic_firk_tableaus import (  # noqa: E402
    FIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.stage_predictors import (  # noqa: E402
    DenseStagePredictor,
)
from cubie.memory import default_memmgr  # noqa: E402
from tests.system_fixtures import (  # noqa: E402
    build_lorenz_julia_system,
    build_three_state_very_stiff_system,
)

SWEEP_MAX_RATIO = 8.0
WARMUP_STEPS = 5
COARSE_GRID = [0.25, 0.5, 0.75, 0.9] + [
    1.0 + 0.5 * i for i in range(15)
]
REFINE_POINTS = 6

TIGHT_TOLERANCES = {
    np.float32: 1e-6,
    np.float64: 1e-10,
}

PROBES = {
    "lorenz": {
        "builder": build_lorenz_julia_system,
        "parameter_overrides": {"rho": 28.0},
        "dt0": [0.005, 0.01],
    },
    "very_stiff": {
        "builder": build_three_state_very_stiff_system,
        "parameter_overrides": {},
        "dt0": [0.0005, 0.002],
    },
}


def unique_tableaus(registry):
    """Return name -> tableau with alias duplicates removed."""

    seen = {}
    for name, tableau in registry.items():
        if id(tableau) not in {id(t) for t in seen.values()}:
            seen[name] = tableau
    return seen


def sweep_tableau(tableau):
    """Return the tableau with ceilings opened to the sweep bound."""

    return attrs.evolve(
        tableau,
        dense_prediction_ratio_float32=SWEEP_MAX_RATIO,
        dense_prediction_ratio_float64=SWEEP_MAX_RATIO,
    )


def compare_rows(kind, tableau):
    """Return the stage rows whose seeds derive from prediction."""

    stage_count = tableau.stage_count
    if kind == "firk":
        return list(range(stage_count))
    diagonal = [tableau.a[i][i] for i in range(stage_count)]
    latest = {}
    rows = []
    for stage, node in enumerate(tableau.c):
        repeated = node in latest
        latest[node] = stage
        if diagonal[stage] != 0.0 and not repeated:
            rows.append(stage)
    return rows


class ProbeContext:
    """Compiled step and predictor harnesses for one sweep case."""

    def __init__(self, kind, tableau, precision, system, params):
        self.kind = kind
        self.precision = precision
        self.system = system
        self.params = params
        n = int(system.sizes.states)
        self.n = n
        self.stage_count = tableau.stage_count
        opened = sweep_tableau(tableau)
        tight = TIGHT_TOLERANCES[precision]
        common = dict(
            precision=precision,
            n=n,
            evaluate_f=system.evaluate_f,
            evaluate_observables=system.evaluate_observables,
            get_solver_helper_fn=system.get_solver_helper,
            tableau=opened,
            n_drivers=int(system.sizes.drivers),
            newton_atol=tight,
            newton_rtol=tight,
            krylov_atol=tight,
            krylov_rtol=tight,
            newton_max_iters=50,
            krylov_max_iters=100,
        )
        step_cls = DIRKStep if kind == "dirk" else FIRKStep
        self.predicted_step = step_cls(
            attempt_dense_prediction=True, **common
        )
        self.carry_step = step_cls(
            attempt_dense_prediction=False, **common
        )
        history_name = (
            "stage_increment_history" if kind == "dirk"
            else "stage_increment"
        )
        self.history_len = self.stage_count * n
        self.run_predicted = self._build_runner(
            self.predicted_step, capture=history_name
        )
        self.run_carry = self._build_runner(self.carry_step)
        self.predictor = DenseStagePredictor(
            precision=precision,
            n=n,
            tableau=opened,
        )
        self.apply_prediction = self._build_predictor_runner()
        self.rows = compare_rows(kind, tableau)

    def _build_runner(self, step_object, capture=None):
        """Compile a one-thread kernel running a dt schedule."""

        precision = self.precision
        numba_precision = from_dtype(precision)
        step_fn = step_object.step_function
        shared_elems = int(step_object.shared_buffer_size)
        shared_bytes = precision(0).itemsize * max(shared_elems, 1)
        persistent_len = max(
            1, int(step_object.persistent_local_buffer_size)
        )
        n = self.n
        n_obs = max(1, int(self.system.sizes.observables))
        n_drv = max(1, int(self.system.sizes.drivers))
        history_len = self.history_len
        capture_history = capture is not None
        if capture_history:
            alloc_history = buffer_registry.get_allocator(
                capture, step_object
            )
        else:
            alloc_history = None

        @cuda.jit
        def kernel(
            state_vec,
            params_vec,
            driver_coeffs,
            dt_schedule,
            out_history,
            out_final,
            out_iters,
            status_vec,
        ):
            idx = cuda.grid(1)
            if idx > 0:
                return
            shared = cuda.shared.array(0, dtype=numba_precision)
            persistent = cuda.local.array(
                persistent_len, dtype=numba_precision
            )
            state = cuda.local.array(n, dtype=numba_precision)
            proposed = cuda.local.array(n, dtype=numba_precision)
            error = cuda.local.array(n, dtype=numba_precision)
            drivers = cuda.local.array(n_drv, dtype=numba_precision)
            proposed_drivers = cuda.local.array(
                n_drv, dtype=numba_precision
            )
            observables = cuda.local.array(
                n_obs, dtype=numba_precision
            )
            proposed_observables = cuda.local.array(
                n_obs, dtype=numba_precision
            )
            counters = cuda.local.array(2, dtype=int32)
            for i in range(persistent_len):
                persistent[i] = numba_precision(0.0)
            for i in range(n):
                state[i] = state_vec[i]
                proposed[i] = numba_precision(0.0)
                error[i] = numba_precision(0.0)
            for i in range(n_drv):
                drivers[i] = numba_precision(0.0)
                proposed_drivers[i] = numba_precision(0.0)
            for i in range(n_obs):
                observables[i] = numba_precision(0.0)
                proposed_observables[i] = numba_precision(0.0)
            counters[0] = 0
            counters[1] = 0
            time_value = numba_precision(0.0)
            status = int32(0)
            n_steps = dt_schedule.shape[0]
            for step_index in range(n_steps):
                if step_index == n_steps - 1:
                    if capture_history:
                        history = alloc_history(shared, persistent)
                        for i in range(history_len):
                            out_history[i] = history[i]
                    counters[0] = 0
                    counters[1] = 0
                first_flag = (
                    int32(1) if step_index == 0 else int32(0)
                )
                result = step_fn(
                    state,
                    proposed,
                    params_vec,
                    driver_coeffs,
                    drivers,
                    proposed_drivers,
                    observables,
                    proposed_observables,
                    error,
                    dt_schedule[step_index],
                    time_value,
                    first_flag,
                    int32(1),
                    shared,
                    persistent,
                    counters,
                )
                status = int32(status | result)
                time_value += dt_schedule[step_index]
                for i in range(n):
                    state[i] = proposed[i]
            if capture_history:
                history = alloc_history(shared, persistent)
                for i in range(history_len):
                    out_final[i] = history[i]
            out_iters[0] = counters[0]
            out_iters[1] = counters[1]
            status_vec[0] = status

        def run(state, dt_schedule):
            stream = default_memmgr.get_group_stream()
            d_state = cuda.to_device(state.astype(precision))
            d_params = cuda.to_device(self.params)
            d_coeffs = cuda.to_device(
                np.zeros((1, 1, 1), dtype=precision)
            )
            d_schedule = cuda.to_device(
                np.asarray(dt_schedule, dtype=precision)
            )
            d_history = cuda.to_device(
                np.zeros(history_len, dtype=precision)
            )
            d_final = cuda.to_device(
                np.zeros(history_len, dtype=precision)
            )
            d_iters = cuda.to_device(np.zeros(2, dtype=np.int32))
            d_status = cuda.to_device(np.zeros(1, dtype=np.int32))
            kernel[1, 1, stream, shared_bytes](
                d_state,
                d_params,
                d_coeffs,
                d_schedule,
                d_history,
                d_final,
                d_iters,
                d_status,
            )
            stream.synchronize()
            status = int(d_status.copy_to_host()[0])
            return (
                d_history.copy_to_host(),
                d_final.copy_to_host(),
                int(d_iters.copy_to_host()[0]),
                status,
            )

        return run

    def _build_predictor_runner(self):
        """Compile a one-thread wrapper around the predictor."""

        precision = self.precision
        numba_precision = from_dtype(precision)
        predict = self.predictor.device_function
        persistent_len = max(
            1,
            int(
                buffer_registry.persistent_local_buffer_size(
                    self.predictor
                )
            ),
        )

        @cuda.jit
        def kernel(vector, step_ratio):
            idx = cuda.grid(1)
            if idx > 0:
                return
            shared = cuda.shared.array(0, dtype=numba_precision)
            persistent = cuda.local.array(
                persistent_len, dtype=numba_precision
            )
            for i in range(persistent_len):
                persistent[i] = numba_precision(0.0)
            predict(vector, step_ratio, True, shared, persistent)

        def run(history, ratio):
            stream = default_memmgr.get_group_stream()
            d_vector = cuda.to_device(history.astype(precision))
            kernel[1, 1, stream](d_vector, precision(ratio))
            stream.synchronize()
            return d_vector.copy_to_host()

        return run

    def evaluate(self, state, dt0, ratio):
        """Return seed errors and Newton iterations for one ratio."""

        schedule = [dt0] * WARMUP_STEPS + [dt0 * ratio]
        history, target, iters_pred, status_pred = (
            self.run_predicted(state, schedule)
        )
        _, _, iters_carry, status_carry = self.run_carry(
            state, schedule
        )
        if status_pred and status_carry:
            # Both seeds fail: the solve limit is independent of the
            # seed, so the case is uninformative at this ratio.
            return {
                "skip": True,
                "pred_error": np.inf,
                "carry_error": np.inf,
                "iters_pred": iters_pred,
                "iters_carry": iters_carry,
            }
        if status_pred:
            # Prediction broke a solve that carry completes.
            return {
                "skip": False,
                "pred_error": np.inf,
                "carry_error": 0.0,
                "iters_pred": iters_pred,
                "iters_carry": iters_carry,
            }
        if status_carry:
            # Prediction rescued a solve that carry fails.
            return {
                "skip": False,
                "pred_error": 0.0,
                "carry_error": np.inf,
                "iters_pred": iters_pred,
                "iters_carry": iters_pred,
            }
        predicted = self.apply_prediction(history, ratio)
        n = self.n
        rows = np.asarray(self.rows, dtype=int)
        take = (
            rows[:, None] * n + np.arange(n)[None, :]
        ).ravel()
        target_rows = target.astype(np.float64)[take]
        pred_error = float(
            np.sqrt(
                np.mean(
                    (predicted.astype(np.float64)[take]
                     - target_rows) ** 2
                )
            )
        )
        carry_error = float(
            np.sqrt(
                np.mean(
                    (history.astype(np.float64)[take]
                     - target_rows) ** 2
                )
            )
        )
        return {
            "skip": False,
            "pred_error": pred_error,
            "carry_error": carry_error,
            "iters_pred": iters_pred,
            "iters_carry": iters_carry,
        }


def case_passes(result):
    """Return whether one probe case passes at a ratio.

    The seed-error comparison is primary. A single-iteration excess
    is tolerated when the predicted seed is strictly closer: on a
    settled system both seeds converge in one or two iterations and
    the residual-based convergence check quantises the count by one,
    which is checker noise rather than seed harm. Larger excesses
    (a genuinely harmful seed, or a solve prediction breaks) always
    fail.
    """

    if not result["pred_error"] <= result["carry_error"]:
        return False
    iteration_excess = result["iters_pred"] - result["iters_carry"]
    if iteration_excess > 1:
        return False
    if (
        iteration_excess == 1
        and result["pred_error"] >= result["carry_error"]
    ):
        return False
    return True


def ratio_passes(rows):
    """Return whether a ratio passes over its informative cases."""

    informative = [row for row in rows if not row["skip"]]
    if not informative:
        return False
    return all(case_passes(row) for row in informative)


def evaluate_ratio(contexts, ratio):
    """Evaluate every probe case at a ratio; return per-case rows."""

    rows = []
    for label, context, state, dt0 in contexts:
        result = context.evaluate(state, dt0, ratio)
        result["case"] = label
        rows.append(result)
    return rows


def find_ceiling(contexts, report_rows, name, precision_name):
    """Sweep the ratio grid and return the calibrated ceiling."""

    grid = sorted(COARSE_GRID)
    outcomes = {}
    for ratio in grid:
        rows = evaluate_ratio(contexts, ratio)
        outcomes[ratio] = rows
    contiguous = []
    for ratio in grid:
        if ratio_passes(outcomes[ratio]):
            contiguous.append(ratio)
        else:
            break
    if not contiguous:
        report_rows.append(
            (name, precision_name, 0.0, None, grid[0], outcomes)
        )
        return 0.0
    boundary = contiguous[-1]
    fail_index = len(contiguous)
    if fail_index < len(grid):
        first_fail = grid[fail_index]
        refine = np.linspace(
            boundary, first_fail, REFINE_POINTS + 2
        )[1:-1]
        for ratio in refine:
            ratio = float(ratio)
            rows = evaluate_ratio(contexts, ratio)
            outcomes[ratio] = rows
            if ratio_passes(rows):
                boundary = ratio
            else:
                first_fail = ratio
                break
    else:
        first_fail = None
    ceiling = float(np.floor(boundary * 100.0) / 100.0)
    report_rows.append(
        (name, precision_name, ceiling, boundary,
         first_fail, outcomes)
    )
    return ceiling


def format_report(report_rows):
    """Return the per-boundary evidence table as markdown."""

    lines = [
        "| tableau | precision | ceiling | boundary evidence "
        "(per case: pred err / carry err / iters P:C) | first fail |",
        "|---|---|---|---|---|",
    ]
    for name, precision_name, ceiling, boundary, fail, outcomes \
            in report_rows:
        if boundary is None:
            key = min(outcomes.keys())
            rows = outcomes[key]
            evidence = "; ".join(
                f"{r['case']}: skip"
                if r.get("skip")
                else f"{r['case']}: {r['pred_error']:.3e}/"
                f"{r['carry_error']:.3e}/"
                f"{r['iters_pred']}:{r['iters_carry']}"
                for r in rows
            )
            lines.append(
                f"| {name} | {precision_name} | 0.0 "
                f"| fails at {key}: {evidence} | {key} |"
            )
            continue
        rows = outcomes[boundary]
        evidence = "; ".join(
            f"{r['case']}: {r['pred_error']:.3e}/"
            f"{r['carry_error']:.3e}/"
            f"{r['iters_pred']}:{r['iters_carry']}"
            for r in rows
        )
        if fail is not None and fail in outcomes:
            fail_rows = [
                r for r in outcomes[fail] if not case_passes(r)
            ]
            fail_text = "; ".join(
                f"{r['case']}: {r['pred_error']:.3e}/"
                f"{r['carry_error']:.3e}/"
                f"{r['iters_pred']}:{r['iters_carry']}"
                for r in fail_rows
            )
            fail_cell = f"{fail:.3f}: {fail_text}"
        else:
            fail_cell = "none up to 8.0"
        lines.append(
            f"| {name} | {precision_name} | {ceiling} "
            f"| at {boundary:.3f}: {evidence} | {fail_cell} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tableaus", nargs="*", default=None)
    parser.add_argument(
        "--precisions",
        nargs="*",
        default=["float32", "float64"],
        choices=["float32", "float64"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    precisions = [
        {"float32": np.float32, "float64": np.float64}[p]
        for p in args.precisions
    ]
    targets = []
    for name, tableau in unique_tableaus(
        DIRK_TABLEAU_REGISTRY
    ).items():
        targets.append(("dirk", name, tableau))
    for name, tableau in unique_tableaus(
        FIRK_TABLEAU_REGISTRY
    ).items():
        targets.append(("firk", name, tableau))
    if args.tableaus:
        targets = [t for t in targets if t[1] in args.tableaus]

    report_rows = []
    ceilings = {}
    for precision in precisions:
        precision_name = np.dtype(precision).name
        systems = {}
        for probe_name, probe in PROBES.items():
            system = probe["builder"](precision)
            params = system.parameters.values_array.astype(
                precision
            ).copy()
            for key, value in probe["parameter_overrides"].items():
                params[
                    system.parameters.get_index_of_key(key)
                ] = value
            state = np.asarray(
                system.initial_values.values_array, dtype=precision
            ).copy()
            systems[probe_name] = (system, params, state, probe)
        for kind, name, tableau in targets:
            contexts = []
            for probe_name, (system, params, state, probe) \
                    in systems.items():
                context = ProbeContext(
                    kind, tableau, precision, system, params
                )
                for dt0 in probe["dt0"]:
                    contexts.append(
                        (f"{probe_name}@dt{dt0}", context,
                         state, dt0)
                    )
            ceiling = find_ceiling(
                contexts, report_rows, name, precision_name
            )
            ceilings[(name, precision_name)] = ceiling
            print(
                f"{name} {precision_name}: ceiling {ceiling}",
                flush=True,
            )

    report = format_report(report_rows)
    print()
    print(report)
    print()
    print("Ceilings for tableau registration:")
    for (name, precision_name), ceiling in sorted(ceilings.items()):
        print(f"  {name} {precision_name}: {ceiling}")
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
