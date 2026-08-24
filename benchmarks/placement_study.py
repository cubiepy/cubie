#!/usr/bin/env python
"""Liveness-grouped shared/local placement study for implicit solvers.

Paired all-local/one-group-shared timings over cadence-grouped buffers.

``--fit`` derives the ``implicit_*`` fields of ``MemoryThresholds``.

``--round2`` runs winner pairings; ``--verify`` runs held-out systems.

Trials append to STUDY_RESULTS (JSONL); re-running skips recorded ones.
"""

import itertools
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RESULTS_FILE = os.environ.get(
    "STUDY_RESULTS",
    os.path.join(HERE, "placement_study_results.jsonl"),
)
REPEATS = 5
NRUNS = int(os.environ.get("STUDY_NRUNS", "32768"))
BLOCKSIZE = 256
CHAIN_DURATION = 0.05
CHAIN_DT = 1e-3

FABBRI_CELLML = os.path.join(
    REPO, "tests", "fixtures", "cellml", "Fabbri_Linder.cellml"
)
FABBRI_DURATION = 0.2

ACH_CAS = "Rate_modulation_experiments_ACh_cas"
ISO_CAS = "Rate_modulation_experiments_Iso_cas"
ACH_DIRECT = "Rate_modulation_experiments_ACh"

WIN_RATIO = 0.95
LOSS_RATIO = 1.05
ITEM = 4

STAGE_KEYS = {
    "radau_iia_5": ["stage_increment_location",
                    "stage_driver_stack_location",
                    "stage_state_location"],
    "kvaerno3": ["stage_increment_location", "stage_base_location",
                 "accumulator_location", "stage_rhs_location"],
    "ros3p": ["stage_rhs_location", "stage_store_location"],
}

GROUPS = {
    "params": ["parameters_location"],
    "lu_factor": ["lu_factor_location"],
    "cached_aux": ["cached_auxiliaries_location"],
    "lu_cached": ["lu_factor_location", "cached_auxiliaries_location"],
    "params_cached": ["parameters_location", "lu_factor_location",
                      "cached_auxiliaries_location"],
    "newton": ["residual_location", "delta_location"],
    "stage": None,
    "cold": ["state_location", "proposed_state_location"],
    "pvec": ["preconditioned_vec_location"],
    "bicg": ["r0_hat_location", "p_location", "v_location",
             "s_hat_location"],
    "pvec_bicg": ["preconditioned_vec_location", "r0_hat_location",
                  "p_location", "v_location", "s_hat_location"],
    "scratch": ["temp_location", "tmp_location"],
    "stage_newton": ["@stage", "residual_location", "delta_location"],
    "stage_cached": ["@stage", "lu_factor_location",
                     "cached_auxiliaries_location"],
    "stage_cold": ["@stage", "state_location",
                   "proposed_state_location"],
    "newton_cold": ["residual_location", "delta_location",
                    "state_location", "proposed_state_location"],
}

LU_GROUPS = ["params", "lu_factor", "cached_aux", "lu_cached",
             "params_cached", "newton", "stage", "cold"]
BICG_GROUPS = ["pvec", "bicg", "pvec_bicg", "scratch", "params",
               "cached_aux", "newton", "stage", "cold"]
ROUND2_GROUPS = ["stage_newton", "stage_cached", "stage_cold",
                 "newton_cold"]

MODES = {
    "radau_exact": ("radau_iia_5", {"inexact_newton": False}),
    "radau_inexact": ("radau_iia_5", {"inexact_newton": True}),
    "kvaerno3_exact": ("kvaerno3", {"inexact_newton": False}),
    "kvaerno3_inexact": ("kvaerno3", {"inexact_newton": True}),
    "ros3p": ("ros3p", {}),
}

SYSTEMS = {
    "chain32": ("chain", 32),
    "chain64": ("chain", 64),
    "chain128": ("chain", 128),
    "fabbri": ("fabbri", 35),
    "pollu": ("pollu", 20),
    "ring": ("ring", 15),
}

# Correction "lu" keeps defaults; "bicgstab" pins jacobi + bicgstab.
MATRIX = (
    [(s, m, "lu") for s in ("chain32", "chain64") for m in MODES]
    + [("chain128", m, "lu")
       for m in ("kvaerno3_exact", "kvaerno3_inexact", "ros3p")]
    + [("fabbri", m, "bicgstab") for m in MODES]
    + [("chain64", m, "bicgstab")
       for m in ("radau_exact", "kvaerno3_exact")]
)

MATRIX2 = [
    ("chain64", "kvaerno3_exact", "lu"),
    ("chain64", "kvaerno3_inexact", "lu"),
    ("chain128", "kvaerno3_exact", "lu"),
    ("chain128", "kvaerno3_inexact", "lu"),
    ("chain128", "ros3p", "lu"),
    ("fabbri", "kvaerno3_inexact", "bicgstab"),
]

MATRIX_VERIFY = [
    (s, m, "lu")
    for s in ("pollu", "ring")
    for m in ("radau_exact", "kvaerno3_exact", "kvaerno3_inexact",
              "ros3p")
]

POLLU_EQUATIONS = """
    r1 = k1 * y1
    r2 = k2 * y2 * y4
    r3 = k3 * y5 * y2
    r4 = k4 * y7
    r5 = k5 * y7
    r6 = k6 * y7 * y6
    r7 = k7 * y9
    r8 = k8 * y9 * y6
    r9 = k9 * y11 * y2
    r10 = k10 * y11 * y1
    r11 = k11 * y13
    r12 = k12 * y10 * y2
    r13 = k13 * y14
    r14 = k14 * y1 * y6
    r15 = k15 * y3
    r16 = k16 * y4
    r17 = k17 * y4
    r18 = k18 * y16
    r19 = k19 * y16
    r20 = k20 * y17 * y6
    r21 = k21 * y19
    r22 = k22 * y19
    r23 = k23 * y1 * y4
    r24 = k24 * y19 * y1
    r25 = k25 * y20
    dy1 = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25
    dy2 = -r2 - r3 - r9 - r12 + r1 + r21
    dy3 = -r15 + r1 + r17 + r19 + r22
    dy4 = -r2 - r16 - r17 - r23 + r15
    dy5 = -r3 + 2.0 * r4 + r6 + r7 + r13 + r20
    dy6 = -r6 - r8 - r14 - r20 + r3 + 2.0 * r18
    dy7 = -r4 - r5 - r6 + r13
    dy8 = r4 + r5 + r6 + r7
    dy9 = -r7 - r8
    dy10 = -r12 + r7 + r9
    dy11 = -r9 - r10 + r8 + r11
    dy12 = r9
    dy13 = -r11 + r10
    dy14 = -r13 + r12
    dy15 = r14
    dy16 = -r18 - r19 + r16
    dy17 = -r20
    dy18 = r20
    dy19 = -r21 - r22 - r24 + r23 + r25
    dy20 = -r25 + r24
"""

POLLU_CONSTANTS = {
    "k2": 26.6, "k3": 1.23e4, "k4": 8.6e-4, "k5": 8.2e-4, "k6": 1.5e4,
    "k7": 1.3e-4, "k8": 2.4e4, "k9": 1.65e4, "k10": 9.0e3,
    "k11": 2.2e-2, "k12": 1.2e4, "k13": 1.88, "k14": 1.63e4,
    "k15": 4.8e6, "k16": 3.5e-4, "k17": 1.75e-2, "k18": 1.0e8,
    "k19": 4.44e11, "k20": 1.24e3, "k21": 2.1, "k22": 5.78,
    "k23": 4.74e-2, "k24": 1.78e3, "k25": 3.12,
}
POLLU_STATES = {f"y{i}": 0.0 for i in range(1, 21)}
POLLU_STATES.update(y2=0.2, y4=0.04, y7=0.1, y8=0.3, y9=0.01,
                    y17=0.007)
POLLU_DURATION = 60.0
POLLU_DT = 60.0 / 1024.0

RING_CONSTANTS = {
    "C": 1.6e-8, "Cp": 1.0e-8, "Lh": 4.45, "Ls1": 0.002,
    "Ls2": 5.0e-4, "Ls3": 5.0e-4, "gamma": 40.67286402e-9,
    "R": 25000.0, "Rp": 50.0, "Rg1": 36.3, "Rg2": 17.3, "Rg3": 17.3,
    "Ri": 50.0, "Rc": 600.0, "delta": 17.7493332,
    "w1": 6283.185307179586, "w2": 62831.85307179586,
    "Uin1_amplitude": 0.5,
}
RING_EQUATIONS = """
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
    dU3 = (I3 - qD1 + qD4) / Cs
    dU4 = (-I4 + qD2 - qD3) / Cs
    dU5 = (I5 + qD1 - qD3) / Cs
    dU6 = (-I6 - qD2 + qD4) / Cs
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
RING_STATES = {name: 0.0 for name in
               ("U1", "U2", "U3", "U4", "U5", "U6", "U7", "I1", "I2",
                "I3", "I4", "I5", "I6", "I7", "I8")}
RING_DURATION = 1.0e-3
RING_DT = 1.0e-3 / 1024.0

_SYSTEM_ROWS = {}


def groups_for(correction):
    return LU_GROUPS if correction == "lu" else BICG_GROUPS


def group_kwargs(group, algorithm):
    keys = GROUPS[group]
    if keys is None:
        keys = STAGE_KEYS[algorithm]
    expanded = []
    for key in keys:
        if key == "@stage":
            expanded.extend(STAGE_KEYS[algorithm])
        else:
            expanded.append(key)
    return {key: "shared" for key in expanded}


def build_system(sys_name):
    """Build the study system; caches the descriptor row."""
    from cubie import create_ODE_system, load_cellml_model

    kind, n = SYSTEMS[sys_name]
    if kind == "chain":
        rng = np.random.default_rng(1234)
        eqs = []
        constants = {}
        for i in range(n):
            im1 = (i - 1) % n
            ip1 = (i + 1) % n
            terms = [f"0.2*x{im1} + 0.3*x{ip1}"]
            for c in range(3):
                cname = f"k{i}_{c}"
                constants[cname] = float(rng.uniform(0.5, 5.0))
                if c == 0:
                    terms.append(f"-{cname}*x{i}")
                else:
                    terms.append(
                        f"+ {cname}*x{ip1}/(1.0 + x{i}*x{i})")
            terms.append(f"+ 0.05*p0*x{i}*x{ip1}")
            eqs.append(f"dx{i} = " + " ".join(terms))
        system = create_ODE_system(
            dxdt=eqs,
            states={f"x{i}": 0.5 for i in range(n)},
            parameters={"p0": 1.0, "p1": 1.0},
            constants=constants,
            precision=np.float32,
            name=f"placement_chain_{n}",
        )
        _SYSTEM_ROWS[sys_name] = dict(
            duration=CHAIN_DURATION, dt=CHAIN_DT, adaptive=False,
            y0={f"x{i}": 0.5 for i in range(n)},
            params={"p0": np.full(NRUNS, 1.0, dtype=np.float32)},
        )
        return system
    if kind == "fabbri":
        system = load_cellml_model(
            FABBRI_CELLML,
            precision=np.float32,
            parameters=[ACH_CAS, ISO_CAS, ACH_DIRECT],
            voltage_variable="Membrane$V_ode",
        )
        system.set_constants(
            {"Rate_modulation_experiments_ANS": 1.0})
        n_ach = int(np.floor(np.sqrt(NRUNS)))
        n_iso = max(1, NRUNS // n_ach)
        ach, iso = np.meshgrid(
            np.linspace(0.0, 100.0, n_ach),
            np.linspace(0.0, 1000.0, n_iso), indexing="ij")
        names = [str(s) for s in system.indices.states.index_map]
        values = np.asarray(system.initial_values.values_array)
        _SYSTEM_ROWS[sys_name] = dict(
            duration=FABBRI_DURATION, dt=None, adaptive=True,
            y0={name: float(v) for name, v in zip(names, values)},
            params={
                ACH_CAS: ach.ravel().astype(np.float32),
                ISO_CAS: iso.ravel().astype(np.float32),
                ACH_DIRECT: ach.ravel().astype(np.float32),
            },
        )
        return system
    if kind == "pollu":
        system = create_ODE_system(
            POLLU_EQUATIONS,
            states=dict(POLLU_STATES),
            parameters={"k1": 0.35},
            constants=dict(POLLU_CONSTANTS),
            precision=np.float32,
            name="placement_pollu",
        )
        _SYSTEM_ROWS[sys_name] = dict(
            duration=POLLU_DURATION, dt=POLLU_DT, adaptive=True,
            y0=dict(POLLU_STATES),
            params={"k1": np.geomspace(
                3.5e-2, 3.5, NRUNS).astype(np.float32)},
        )
        return system
    system = create_ODE_system(
        RING_EQUATIONS,
        states=dict(RING_STATES),
        parameters={"Cs": 2.0e-12},
        constants=dict(RING_CONSTANTS),
        precision=np.float32,
        name="placement_ring",
    )
    _SYSTEM_ROWS[sys_name] = dict(
        duration=RING_DURATION, dt=RING_DT, adaptive=True,
        y0=dict(RING_STATES),
        params={"Cs": np.geomspace(
            2.0e-13, 2.0e-9, NRUNS).astype(np.float32)},
    )
    return system


def make_solver(sys_name, system, algorithm, mode_kwargs, correction,
                loc_kwargs):
    from cubie import Solver

    row = _SYSTEM_ROWS[sys_name]
    kwargs = dict(
        algorithm=algorithm,
        save_every=row["duration"],
        output_types=["state", "iteration_counters"],
        time_logging_level="default",
        auto_memory=False,
        krylov_max_iters=100,
        **mode_kwargs,
    )
    if correction == "bicgstab":
        kwargs.update(preconditioner_type="jacobi",
                      linear_correction_type="bicgstab")
    if row["adaptive"]:
        tol = 1e-4 if sys_name == "fabbri" else 1e-5
        kwargs.update(rtol=tol, atol=tol)
        if row["dt"] is not None:
            kwargs.update(dt=row["dt"])
    else:
        kwargs.update(dt=row["dt"], step_controller="fixed")
    return Solver(system, **kwargs, **loc_kwargs)


def timed_solve(solver, y0, params, solve_kwargs):
    """Run one solve and return its kernel CUDA-event total (ms)."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(y0, params, **solve_kwargs)
    events = solver.kernel._cuda_events
    ms = sum(
        e.elapsed_time_ms()
        for e in events
        if e.name.startswith("kernel_chunk")
    )
    return ms, result


def solver_stats(solver):
    """Return launch-geometry and register stats for a built solver."""
    bsk = solver.kernel
    pad = 4 if bsk.shared_memory_needs_padding else 0
    bytes_per_run = bsk.shared_memory_bytes + pad
    smem = int(bytes_per_run * min(NRUNS, BLOCKSIZE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eff_blocksize, smem = bsk.limit_blocksize(
            BLOCKSIZE, smem, bytes_per_run, NRUNS
        )
    regs = lmem = None
    try:
        kernel = bsk.kernel
        regs = list(kernel.get_regs_per_thread().values())[0]
        lmem = list(kernel.get_local_mem_per_thread().values())[0]
    except Exception:
        pass
    return dict(
        regs_per_thread=regs,
        local_mem_per_thread=lmem,
        shared_bytes_per_run=int(bytes_per_run),
        eff_blocksize=int(eff_blocksize),
        dynamic_shared_per_block=int(smem),
    )


def shared_entries():
    from cubie.buffer_registry import buffer_registry

    names = []
    for parent, group in buffer_registry._groups.items():
        for name, entry in group.entries.items():
            if entry.location == "shared" and entry.size > 0:
                names.append(name)
    return sorted(set(names))


def counter_totals(result):
    try:
        counters = np.asarray(result.iteration_counters)
        if counters.ndim == 3:
            return [int(x) for x in counters.sum(axis=(0, 2))]
        return [int(x) for x in counters.sum(axis=0).ravel()[:8]]
    except Exception:
        return None


def run_single(sys_name, mode, correction, group):
    from cubie.time_logger import default_timelogger

    default_timelogger.set_verbosity("default")
    algorithm, mode_kwargs = MODES[mode]
    system = build_system(sys_name)
    loc_kwargs = group_kwargs(group, algorithm)

    row = _SYSTEM_ROWS[sys_name]
    y0, params = row["y0"], row["params"]
    solve_kwargs = dict(duration=row["duration"],
                        grid_type="verbatim", blocksize=BLOCKSIZE)

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        solver_local = make_solver(sys_name, system, algorithm,
                                   mode_kwargs, correction, {})
        solver_variant = make_solver(sys_name, system, algorithm,
                                     mode_kwargs, correction, {})
        # Keys the config lacks are dropped; the subset still runs.
        while True:
            try:
                solver_variant.update(**loc_kwargs)
                break
            except KeyError as exc:
                dropped = [k for k in list(loc_kwargs)
                           if k in str(exc)]
                if not dropped:
                    raise
                for key in dropped:
                    loc_kwargs.pop(key)
                if not loc_kwargs:
                    print(json.dumps({"skip": True,
                                      "reason": "no keys apply"}))
                    return
        _, result_local = timed_solve(solver_local, y0, params,
                                      solve_kwargs)
        _, result_variant = timed_solve(solver_variant, y0, params,
                                        solve_kwargs)
        caught = sorted({str(w.message)[:80] for w in wlist})
    compile_s = time.perf_counter() - t0

    applied = shared_entries()
    if not applied:
        print(json.dumps({"skip": True,
                          "reason": "placement not applied"}))
        return

    out_local = np.asarray(result_local.state[-1])
    out_variant = np.asarray(result_variant.state[-1])
    diff = float(np.nanmax(np.abs(np.nan_to_num(out_local)
                                  - np.nan_to_num(out_variant))))
    nan_match = bool(np.array_equal(np.isnan(out_local),
                                    np.isnan(out_variant)))
    counters_local = counter_totals(result_local)
    counters_variant = counter_totals(result_variant)

    local_ms, variant_ms = [], []
    for _ in range(REPEATS):
        ms, _ = timed_solve(solver_local, y0, params, solve_kwargs)
        local_ms.append(ms)
        ms, _ = timed_solve(solver_variant, y0, params, solve_kwargs)
        variant_ms.append(ms)

    local_arr = np.asarray(local_ms)
    variant_arr = np.asarray(variant_ms)
    record = dict(
        phase="study",
        system=sys_name,
        mode=mode,
        correction=correction,
        algorithm=algorithm,
        group=group,
        shared_keys=sorted(loc_kwargs),
        applied=applied,
        n_states=SYSTEMS[sys_name][1],
        n_runs=NRUNS,
        precision="f32",
        requested_blocksize=BLOCKSIZE,
        ratio_min=float(variant_arr.min() / local_arr.min()),
        ratio_median_pairwise=float(np.median(variant_arr / local_arr)),
        local_ms_min=float(local_arr.min()),
        variant_ms_min=float(variant_arr.min()),
        local_ms=list(map(float, local_arr)),
        variant_ms=list(map(float, variant_arr)),
        compile_s=round(compile_s, 2),
        local_stats=solver_stats(solver_local),
        variant_stats=solver_stats(solver_variant),
        counters_local=counters_local,
        counters_variant=counters_variant,
        out_maxdiff=diff,
        nan_pattern_match=nan_match,
        warnings=caught,
    )
    print(json.dumps(record))


def drive(matrix=MATRIX, group_list=None, only=None):
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "ratio_min" in rec or rec.get("error") \
                        or rec.get("skip"):
                    done.add((rec.get("system"), rec.get("mode"),
                              rec.get("correction"), rec.get("group")))

    trials = []
    for sys_name, mode, correction in matrix:
        for group in (group_list or groups_for(correction)):
            trial = (sys_name, mode, correction, group)
            if trial in done or trial in trials:
                continue
            if only is not None and "/".join(trial) not in only:
                continue
            trials.append(trial)
    print(f"study: {len(trials)} trials to run", flush=True)
    for idx, (sys_name, mode, correction, group) in enumerate(trials):
        label = f"{sys_name}/{mode}/{correction}/{group}"
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--single",
                 sys_name, mode, correction, group],
                capture_output=True, text=True, timeout=2400,
            )
        except subprocess.TimeoutExpired:
            record = dict(phase="study", system=sys_name, mode=mode,
                          correction=correction, group=group,
                          error="timeout after 2400 s")
            with open(RESULTS_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"[{idx + 1}/{len(trials)}] {label}: TIMEOUT",
                  flush=True)
            continue
        record = None
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(candidate, dict):
                record = candidate
                break
        if record is None:
            record = dict(phase="study", system=sys_name, mode=mode,
                          correction=correction, group=group,
                          error=(proc.stdout + proc.stderr)[-2000:])
        elif record.get("skip"):
            record.update(phase="study", system=sys_name, mode=mode,
                          correction=correction, group=group)
        with open(RESULTS_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        elapsed = time.perf_counter() - t0
        if "ratio_min" in record:
            vs = record["variant_stats"]
            print(f"[{idx + 1}/{len(trials)}] {label}: "
                  f"ratio {record['ratio_min']:.3f} "
                  f"(local {record['local_ms_min']:.2f} ms, "
                  f"bs {vs['eff_blocksize']}, "
                  f"sh {vs['shared_bytes_per_run']} B/run) "
                  f"[{elapsed:.0f} s]", flush=True)
        elif record.get("skip"):
            print(f"[{idx + 1}/{len(trials)}] {label}: inapplicable "
                  f"({record.get('reason', '')})", flush=True)
        else:
            print(f"[{idx + 1}/{len(trials)}] {label}: ERROR",
                  flush=True)
    print("STUDY DONE", flush=True)


# --- Implicit gate fitting -------------------------------------------

WIDTH_MULT = {"radau_iia_5": 3, "kvaerno3": 1, "ros3p": 1}
STAGE_ELEMS_PER_N = {"radau_iia_5": 4, "kvaerno3": 5, "ros3p": 4}
STAGES = {"radau_iia_5": 3, "kvaerno3": 4, "ros3p": 3}


def fit_declared(system, mode):
    """Declared byte features the implicit gates operate on."""
    algo = MODE_ALGOS[mode]
    n = SYSTEMS[system][1]
    n_params = {"fabbri": 3, "pollu": 1, "ring": 1}.get(system, 2)
    width = WIDTH_MULT[algo] * n
    pair = 2 * n * ITEM
    stage = STAGE_ELEMS_PER_N[algo] * n * ITEM
    newton = 2 * width * ITEM
    linear = 2 * width * ITEM
    footprint = (pair + stage + newton + linear + n * ITEM
                 + n_params * ITEM)
    return dict(algo=algo, n=n, width=width, pair_bytes=pair,
                stage_bytes=stage, newton_bytes=newton,
                footprint_bytes=footprint, stages=STAGES[algo])


MODE_ALGOS = {mode: algo for mode, (algo, _) in MODES.items()}


def fit_decide(dec, gates):
    """Replay the implicit placement rules for one configuration."""
    foot = dec["footprint_bytes"]
    if foot < gates["implicit_floor_bytes"]:
        return None
    narrow = dec["width"] == dec["n"]
    if foot >= gates["implicit_deep_bytes"] and narrow \
            and dec["pair_bytes"] <= gates["implicit_cold_max_bytes"]:
        return "cold"
    if dec["algo"] != "ros3p" and dec["stages"] >= 2 \
            and dec["stage_bytes"] <= gates["implicit_stage_max_bytes"]:
        return "stage"
    return None


def load_fit_configs(results_file):
    """Group valid measured trials by configuration."""
    configs = {}
    with open(results_file) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ratio_min" not in rec:
                continue
            cl = rec.get("counters_local")
            cv = rec.get("counters_variant")
            if cl and cv and cv[2] == 0 and cl[2] > 0:
                continue
            key = (rec["system"], rec["mode"], rec["correction"])
            configs.setdefault(key, {})[rec["group"]] = (
                rec["ratio_min"])
    return configs


def fit(results_file):
    """Scan gate candidates and print the implicit thresholds."""
    configs = load_fit_configs(results_file)
    print(f"{len(configs)} configs")
    best = None
    for floor, deep, cold, stage in itertools.product(
            [256, 512, 768, 1024], [1536, 2048, 2560],
            [512, 768, 1024, 1536], [512, 768, 1024]):
        gates = dict(implicit_floor_bytes=floor,
                     implicit_deep_bytes=deep,
                     implicit_cold_max_bytes=cold,
                     implicit_stage_max_bytes=stage)
        fired_losses = missed = wins = 0
        for (system, mode, correction), ratios in configs.items():
            choice = fit_decide(fit_declared(system, mode), gates)
            ratio = ratios.get(choice) if choice else None
            if ratio is not None and ratio >= LOSS_RATIO:
                fired_losses += 1
            elif ratio is not None and ratio <= WIN_RATIO:
                wins += 1
            elif ratio is None and any(
                    v <= WIN_RATIO for v in ratios.values()):
                missed += 1
        rank = (fired_losses, missed, -wins, -floor)
        if best is None or rank < best[0]:
            best = (rank, gates)
    gates = best[1]
    fired_losses, missed, wins = best[0][0], best[0][1], -best[0][2]
    print(f"wins {wins}, fired losses {fired_losses}, "
          f"missed wins {missed}")
    print("\nimplicit fields for cubie/integrators/"
          "memory_heuristics.py:")
    for field, value in gates.items():
        print(f"        {field}={value},")
    print("\nvalidation:")
    for key in sorted(configs):
        system, mode, correction = key
        choice = fit_decide(fit_declared(system, mode), gates)
        ratio = configs[key].get(choice) if choice else None
        r = "" if ratio is None else f"{ratio:.2f}"
        print(f"  {system:9s} {mode:17s} {correction:8s} "
              f"-> {choice or '-':6s} {r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        run_single(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) > 1 and sys.argv[1] == "--round2":
        drive(matrix=MATRIX2, group_list=ROUND2_GROUPS)
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        drive(matrix=MATRIX_VERIFY)
    elif len(sys.argv) > 1 and sys.argv[1] == "--fit":
        fit(sys.argv[2] if len(sys.argv) > 2 else RESULTS_FILE)
    elif len(sys.argv) > 2 and sys.argv[1] == "--only":
        drive(matrix=MATRIX + MATRIX2 + MATRIX_VERIFY,
              group_list=list(GROUPS),
              only=set(sys.argv[2].split(",")))
    else:
        drive()
