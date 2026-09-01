#!/usr/bin/env python
"""Unroll-policy time bank: every placement_landscape config x 15 UnrollFlags policies, all buffers local, bank protocol (GPU)."""

import argparse
import collections
import gzip
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import placement_landscape as pl

REPO = pl.REPO
BENCH = Path(__file__).resolve()
OUT_DEFAULT = Path(
    r"C:\local_working_projects\cubie-notes\unroll_landscape\post881"
)
BANK_RECORDS = pl.OUT_DEFAULT / "records.jsonl"
GROUPS = (
    "unroll_stage", "unroll_step_element", "unroll_accumulator",
    "unroll_solver_element", "unroll_norms", "unroll_other_small",
    "unroll_converged_exits",
)
FULL = True
ROLLED = (True, 1)
DEFAULT = False
POLICY_SETS = ("corners", "factorial")


def corner_policies():
    """Both binding corners, the libnvvm default, one-group deviations."""
    table = {
        "full": {g: FULL for g in GROUPS},
        "rolled": {g: ROLLED for g in GROUPS},
        "default": {g: DEFAULT for g in GROUPS},
    }
    for group in GROUPS:
        short = group[len("unroll_"):]
        table[f"full-{short}"] = {
            g: (ROLLED if g == group else FULL) for g in GROUPS
        }
        table[f"rolled+{short}"] = {
            g: (FULL if g == group else ROLLED) for g in GROUPS
        }
    return table


def factorial_policies():
    """Every full/rolled combination; label bit i is group i (1 = full)."""
    table = {}
    for code in range(2 ** len(GROUPS)):
        bits = format(code, f"0{len(GROUPS)}b")
        table["u" + bits] = {
            g: (FULL if bit == "1" else ROLLED)
            for g, bit in zip(GROUPS, bits)
        }
    return table


POLICIES = corner_policies()
POLICY_SET = "corners"


def set_policy_set(name):
    global POLICIES, POLICY_SET
    POLICY_SET = name
    POLICIES = (corner_policies() if name == "corners"
                else factorial_policies())


def full_label():
    """Label of the all-full policy in the active set."""
    return "full" if "full" in POLICIES else "u" + "1" * len(GROUPS)


def unroll_flags(policy):
    from cubie.cuda_simsafe import UnrollFlags

    return UnrollFlags(**POLICIES[policy])


def make_solver(system, system_name, algo_name, policy):
    return pl.make_solver(
        system, system_name, algo_name,
        extra=dict(unroll=unroll_flags(policy)),
    )


def open_records(out):
    out = Path(out)
    return pl.Records(out / "records.jsonl", extra=[out / "compiles.jsonl"])


def open_compiles(out):
    return pl.Records(Path(out) / "compiles.jsonl")


def compile_key(system_name, algo_name, policy):
    return pl.task_key("compile", system_name, algo_name, policy)


def compile_row(compiles, system_name, algo_name, policy):
    return compiles.get(compile_key(system_name, algo_name, policy))


def compiled(compiles, system_name, algo_name, policy):
    row = compile_row(compiles, system_name, algo_name, policy)
    return row is not None and row.get("source_hash") == pl.source_hash()


# --- compiles ----------------------------------------------------------


def worker_main(out):
    """Compile one policy in this process and print its row."""
    helpers = pl.spill_helpers()
    helpers.install_spill_capture()
    job = json.loads(sys.stdin.read())
    spec = pl.SYSTEMS[job["system"]]
    system = spec["build"]()
    solver = make_solver(system, job["system"], job["algo"], job["policy"])
    inits, params = spec["grid"](solver, 256)
    start = time.perf_counter()
    solver.compile(inits, params, duration=spec["duration"])
    compile_s = time.perf_counter() - start
    key = compile_key(job["system"], job["algo"], job["policy"])
    payload = compile_payload(solver, helpers, out, key, compile_s)
    print("@RESULT " + json.dumps(payload, default=pl._json_default),
          flush=True)


INSTR = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\.L_x_(\d+):")
COUNTED = ("BRA", "LDL", "STL", "LDS", "STS", "LDG", "STG", "MUFU",
           "FFMA", "FMUL", "FADD", "CALL", "VOTE")


def sass_counts(sass_path):
    """Whole-kernel SASS opcode counts and back-edge count."""
    if not sass_path:
        return None
    with gzip.open(sass_path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    instrs, labels = [], {}
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            labels[f".L_x_{m.group(1)}"] = len(instrs)
            continue
        m = INSTR.match(line)
        if m:
            body = m.group(2)
            if body.startswith("@"):
                body = body.split(None, 1)[1]
            instrs.append((body.split()[0].split(".")[0], body))
    back_edges = 0
    for k, (op, body) in enumerate(instrs):
        if op == "BRA":
            m = re.search(r"(\.L_x_\d+)", body)
            if m and m.group(1) in labels and labels[m.group(1)] <= k:
                back_edges += 1
    ops = collections.Counter(op for op, _ in instrs)
    out = {name.lower(): ops[name] for name in COUNTED}
    out["instructions"] = len(instrs)
    out["ldl_stl"] = ops["LDL"] + ops["STL"]
    out["back_edges"] = back_edges
    return out


def compile_payload(solver, helpers, out, key, compile_s):
    payload = pl.compile_payload(solver, helpers, out, key, compile_s)
    payload["sass_counts"] = sass_counts(payload["artefacts"].get("sass"))
    return payload


def compile_jobs(compiles, jobs, out, workers):
    """Compile ``jobs`` (dicts: system, algo, policy) in subprocesses."""
    pending = [
        job for job in jobs
        if not compiled(compiles, job["system"], job["algo"], job["policy"])
    ]
    running = []
    env = dict(os.environ)
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            process = subprocess.Popen(
                [sys.executable, str(BENCH), "--worker", "--out", str(out),
                 "--policy-set", POLICY_SET],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env,
            )
            process.stdin.write(json.dumps(job))
            process.stdin.close()
            running.append((job, process, time.perf_counter()))
        still = []
        for job, process, started in running:
            key = compile_key(job["system"], job["algo"], job["policy"])
            if process.poll() is None:
                if time.perf_counter() - started > pl.COMPILE_TIMEOUT:
                    process.kill()
                    compiles.append(
                        dict(key=key, task="compile", status="timeout",
                             source_hash=pl.source_hash(), **job)
                    )
                    continue
                still.append((job, process, started))
                continue
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            payload = None
            for line in stdout.splitlines():
                if line.startswith("@RESULT "):
                    payload = json.loads(line[len("@RESULT "):])
            if payload is None:
                compiles.append(
                    dict(key=key, task="compile", status="error",
                         error=stderr[-3000:], source_hash=pl.source_hash(),
                         **job)
                )
                print(f"  compile error {job['system']}/{job['algo']}/"
                      f"{job['policy']}: {stderr[-300:]!r}", flush=True)
            else:
                compiles.append(
                    dict(key=key, task="compile", status="ok", **job,
                         **payload)
                )
                print(
                    f"  compiled {job['system']}/{job['algo']}/"
                    f"{job['policy']}: {payload['regs']} regs, "
                    f"{payload['local_bytes']} B local, "
                    f"{payload['compile_s']} s",
                    flush=True,
                )
        running = still
        if running:
            time.sleep(1.0)


# --- banking -----------------------------------------------------------


def bank_duration(system_name, algo_name):
    """The placement bank's settled duration for a config, else None."""
    if not BANK_RECORDS.exists():
        return None
    with open(BANK_RECORDS, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get("task") == "features" and row["system"] == system_name
                    and row["algo"] == algo_name):
                return float(row["duration"])
    return None


def kernel_entries(system, system_name, algo_name, compiles):
    """Build (label, policy, solver, bs, dynshared) per compiled policy."""
    entries = []
    for policy in POLICIES:
        row = compile_row(compiles, system_name, algo_name, policy)
        if row is None or row.get("status") != "ok":
            continue
        # One system per solver (issue 685).
        solver = make_solver(pl.SYSTEMS[system_name]["build"](),
                             system_name, algo_name, policy)
        for plan, blocksize, dynshared in pl.launch_plans(
            row["occupancy"], equal_t_rows=False
        ):
            label = policy if plan == "default" else f"{policy}@{plan}"
            entries.append((label, policy, solver, blocksize, dynshared))
    return entries


def bank_wave(records, system_name, algo_name, entries, inits, params,
              duration, n_runs):
    """Warm solve per row, settle, then ROUNDS x REPEATS interleaved."""
    reference = None
    for label, policy, solver, blocksize, dynshared in entries:
        pl.pin_launch(solver, blocksize, dynshared)
        ms, wall, snapshot = pl.solve_once(
            solver, inits, params, duration, blocksize
        )
        if reference is None:
            reference = snapshot
        check = pl.compare_outputs(reference, snapshot)
        geometry = pl.launch_geometry(solver, blocksize, n_runs, dynshared)
        records.append(
            dict(
                key=pl.task_key("solve", system_name, algo_name,
                                f"{label}|warm"),
                task="solve", system=system_name, algo=algo_name,
                label=label, policy=policy, warm=True, round=-1, rep=-1,
                kernel_ms=round(ms, 4), wall_ms=round(wall, 3),
                n_runs=n_runs, duration=duration, geometry=geometry,
                status_hist=snapshot["status_hist"], **check,
            )
        )
        del snapshot
    label, policy, solver, blocksize, dynshared = entries[0]
    pl.pin_launch(solver, blocksize, dynshared)
    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < pl.SETTLE_S:
        pl.solve_once(solver, inits, params, duration, blocksize,
                      snapshot=False)
    for round_idx in range(pl.ROUNDS):
        for label, policy, solver, blocksize, dynshared in entries:
            pl.pin_launch(solver, blocksize, dynshared)
            for rep in range(pl.REPEATS):
                ms, wall, _ = pl.solve_once(
                    solver, inits, params, duration, blocksize,
                    snapshot=False,
                )
                records.append(
                    dict(
                        key=pl.task_key(
                            "solve", system_name, algo_name,
                            f"{label}|r{round_idx}k{rep}",
                        ),
                        task="solve", system=system_name, algo=algo_name,
                        label=label, policy=policy, warm=False,
                        round=round_idx, rep=rep, kernel_ms=round(ms, 4),
                        wall_ms=round(wall, 3), n_runs=n_runs,
                        duration=duration, blocksize=blocksize,
                        dynshared=dynshared,
                    )
                )
        print(f"  round {round_idx} done", flush=True)


def run_config(out, system_name, algo_name, workers):
    """Compile every policy, then bank one interleaved wave."""
    out = Path(out)
    records = open_records(out)
    compiles = open_compiles(out)
    helpers = pl.spill_helpers()
    helpers.install_spill_capture()
    spec = pl.SYSTEMS[system_name]
    n_runs = spec["n_runs"]
    features_key = pl.task_key("features", system_name, algo_name)
    row = records.get(features_key)
    duration = float(row["duration"]) if row else bank_duration(
        system_name, algo_name)

    start = time.perf_counter()
    system = spec["build"]()
    codegen_s = time.perf_counter() - start
    full = full_label()
    base = make_solver(system, system_name, algo_name, full)
    inits, params = spec["grid"](base, n_runs)
    base_key = compile_key(system_name, algo_name, full)
    start = time.perf_counter()
    base.compile(inits, params, duration=duration or spec["duration"])
    compile_s = time.perf_counter() - start
    if not compiled(compiles, system_name, algo_name, full):
        compiles.append(
            dict(key=base_key, task="compile", status="ok",
                 system=system_name, algo=algo_name, policy=full,
                 **compile_payload(base, helpers, out, base_key,
                                   compile_s))
        )
    marker = pl.probe_marker(out, system_name, algo_name)
    probes = []

    def guarded(solver, dur, scale, snapshot):
        pl.marker_set(marker, dict(scale=scale, duration=dur, probes=probes,
                                   start=time.time()))
        result = pl.solve_once(solver, inits, params, dur,
                               snapshot=snapshot)
        pl.marker_clear(marker)
        return result

    if duration is None:
        # No bank duration: ramp from the spec as the placement bank did.
        for scale in pl.PROBE_SCALES:
            dur = spec["duration"] / scale
            start = time.perf_counter()
            _, _, probe = guarded(base, dur, scale, True)
            probes.append(dict(scale=scale, duration=dur,
                               solve_s=round(time.perf_counter() - start,
                                             2),
                               status_hist=probe["status_hist"]))
            del probe
        duration = spec["duration"]
        ms, _, _ = guarded(base, duration, None, False)
        while ms < pl.MIN_SOLVE_MS and duration < spec["duration"] * (
            pl.MAX_DURATION_SCALE
        ):
            duration *= 2
            ms, _, _ = guarded(base, duration, None, False)
    else:
        guarded(base, duration, None, False)
    base.close()
    if row is None:
        records.append(
            dict(
                key=features_key, task="features", system=system_name,
                algo=algo_name, family=pl.family(algo_name),
                algorithm_settings={
                    k: v for k, v in pl.solver_kwargs(
                        system_name, algo_name).items()
                    if k != "output_types"
                },
                n_runs=n_runs, duration=duration, probes=probes,
                policies={k: {g: list(v) if isinstance(v, tuple) else v
                              for g, v in flags.items()}
                          for k, flags in POLICIES.items()},
                codegen_s=round(codegen_s, 2),
            )
        )
    print(f"  full compiled in {compile_s:.0f} s; duration {duration}",
          flush=True)

    compile_jobs(
        compiles,
        [dict(system=system_name, algo=algo_name, policy=p)
         for p in POLICIES],
        out, workers=workers,
    )
    compiles.reload()
    if not records.has(pl.task_key("wavedone", system_name, algo_name)):
        entries = kernel_entries(system, system_name, algo_name, compiles)
        print(f"  wave: {len(entries)} launch rows", flush=True)
        bank_wave(records, system_name, algo_name, entries, inits, params,
                  duration, n_runs)
        pl.close_entries(entries)
        records.append(
            dict(key=pl.task_key("wavedone", system_name, algo_name),
                 task="wavedone", system=system_name, algo=algo_name)
        )
    records.append(
        dict(key=pl.task_key("configdone", system_name, algo_name),
             task="configdone", system=system_name, algo=algo_name)
    )


# --- driver ------------------------------------------------------------


EXCLUDED_SYSTEMS = ("lorenz96_10",)


def selected_configs(args):
    return [c for c in pl.selected_configs(args)
            if c[0] not in EXCLUDED_SYSTEMS]


def drive(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = open_records(out)
    configs = selected_configs(args)
    print(f"{len(configs)} configs x {len(POLICIES)} policies", flush=True)
    for system_name, algo_name in configs:
        records.reload()
        if records.has(pl.task_key("configdone", system_name, algo_name)):
            continue
        if records.has(pl.task_key("configskip", system_name, algo_name)):
            continue
        errors = records.select(
            task="configerror", system=system_name, algo=algo_name
        )
        if len(errors) >= 2:
            continue
        label = f"{system_name}/{algo_name}"
        print(f"{label} ...", flush=True)
        start = time.perf_counter()
        command = [
            sys.executable, "-u", str(BENCH), "--config", system_name,
            algo_name, "--out", str(out), "--workers", str(args.workers),
            "--policy-set", POLICY_SET,
        ]
        log_path = out / "logs" / f"{system_name}_{algo_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        marker = pl.probe_marker(out, system_name, algo_name)
        marker.unlink(missing_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            status, info = pl.run_child(
                command, pl.child_env(out), log, args.config_timeout, marker
            )
        elapsed = time.perf_counter() - start
        records.reload()
        if status == "skipped":
            records.append(
                dict(
                    key=pl.task_key("configskip", system_name, algo_name),
                    task="configskip", system=system_name, algo=algo_name,
                    scale=info["scale"], duration=info["duration"],
                    probes=info["probes"], budget_s=pl.SOLVE_BUDGET_S,
                )
            )
        elif status != "ok":
            records.append(
                dict(
                    key=pl.task_key("configerror", system_name, algo_name,
                                    f"{time.time():.0f}"),
                    task="configerror", system=system_name,
                    algo=algo_name, status=status,
                    elapsed_s=round(elapsed, 1),
                )
            )
        tail = ""
        if status != "ok":
            tail = log_path.read_text(encoding="utf-8")[-800:]
        print(f"{label}: {status} in {elapsed:.0f} s {tail}", flush=True)
    print("DRIVER DONE", flush=True)


# --- report ------------------------------------------------------------


def report(args):
    out = Path(args.out)
    records = open_records(out)
    compiles = open_compiles(out)
    lines = ["# Unroll policy time bank", ""]
    for system_name, algo_name in pl.config_list():
        solves = records.select(
            task="solve", system=system_name, algo=algo_name, warm=False
        )
        if not solves:
            continue
        warm = {
            row["label"]: row for row in records.select(
                task="solve", system=system_name, algo=algo_name, warm=True,
            )
        }
        samples = {}
        for row in solves:
            samples.setdefault(row["label"], []).append(row["kernel_ms"])
        full = full_label()
        base = min(samples[full]) if full in samples else None
        lines.append(f"## {system_name} / {algo_name}")
        lines.append("")
        lines.append(
            "| policy | bs | T | regs | local B | spill st/ld | SASS |"
            " LDL+STL | back edges | min ms | median ms | spread |"
            " ratio to full | maxdiff | fails |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|"
                     "---|---|---|")
        for label, values in sorted(samples.items()):
            arr = np.asarray(values)
            first = warm.get(label, {})
            geometry = first.get("geometry") or {}
            policy = first.get("policy") or label.split("@")[0]
            row = compile_row(compiles, system_name, algo_name, policy) or {}
            counts = row.get("sass_counts") or {}
            ratio = f"{arr.min() / base:.3f}" if base else ""
            lines.append(
                f"| {label} | {geometry.get('blocksize', '')} | "
                f"{geometry.get('resident_threads', '')} | "
                f"{row.get('regs', '')} | {row.get('local_bytes', '')} | "
                f"{row.get('spill_store_bytes', '')}/"
                f"{row.get('spill_load_bytes', '')} | "
                f"{counts.get('instructions', '')} | "
                f"{counts.get('ldl_stl', '')} | "
                f"{counts.get('back_edges', '')} | "
                f"{arr.min():.3f} | {np.median(arr):.3f} | "
                f"{arr.max() / arr.min() - 1:.3f} | {ratio} | "
                f"{first.get('max_abs_diff', float('nan')):.2e} | "
                f"{(first.get('status_hist') or {}).get('failed', '')} |"
            )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


# --- entry -------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", nargs=2, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--only", nargs="+", default=None,
                        help="system/algo labels to include")
    parser.add_argument("--workers", type=int, default=pl.WORKERS)
    parser.add_argument("--config-timeout", type=float, default=7200.0)
    parser.add_argument("--policy-set", default="corners",
                        choices=POLICY_SETS)
    args = parser.parse_args(argv)
    set_policy_set(args.policy_set)
    if args.worker:
        worker_main(args.out)
    elif args.config:
        warnings.simplefilter("ignore")
        run_config(args.out, args.config[0], args.config[1], args.workers)
    elif args.list:
        for config in pl.config_list():
            if config[0] not in EXCLUDED_SYSTEMS:
                print(config)
        for label, flags in POLICIES.items():
            print(label, flags)
    elif args.report:
        report(args)
    elif args.run:
        drive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
