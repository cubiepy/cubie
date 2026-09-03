#!/usr/bin/env python
"""Unroll-policy time bank over each config's live UnrollFlags groups."""

import argparse
import collections
import gzip
import hashlib
import itertools
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
    r"C:\local_working_projects\cubie-notes\unroll_landscape\post882"
)
GROUPS = (
    "unroll_stage", "unroll_step_element", "unroll_accumulator",
    "unroll_solver_element", "unroll_norms", "unroll_other_small",
    "unroll_converged_exits",
)
FULL = True
ROLLED = (True, 1)
FULL_LABEL = "u" + "1" * len(GROUPS)
LIBNVVM_LABEL = "libnvvm"
DUPLICATE_SUFFIX = "#2"
PROBE_LIMITS = ((50, 50), (10, 10))
WAVE_TAGS = {"live": "", "single-false": "n", "fixed-four": "f"}
BLOCK_MIN_FREE_BYTES = 3 << 30
SYSTEM_LIST = ("lorenz", "lorenz96_20", "chain32", "fabbri")
TABLEAU_LIST = (
    "bogacki-shampine-32", "vern7",
    "kvaerno3", "kvaerno5", "kvaerno3_bicgstab",
    "radau_iia_3", "radau_iia_5", "radau_iia_5_bicgstab",
    "rosenbrock23", "rodas3p", "rosenbrock23_bicgstab",
)


# --- policies ----------------------------------------------------------


def config_list():
    configs = []
    for system_name in SYSTEM_LIST:
        for tableau in TABLEAU_LIST:
            if (tableau.endswith("_bicgstab")
                    and system_name in pl.LU_ONLY_SYSTEMS):
                continue
            configs.append((system_name, tableau))
    return configs


LEVELS = {"1": FULL, "0": ROLLED, "n": False}


def policy_flags(label):
    """UnrollFlags kwargs of a policy label (``u<levels>`` or ``libnvvm``)."""
    if label == LIBNVVM_LABEL:
        return {g: False for g in GROUPS}
    return {g: LEVELS[level] for g, level in zip(GROUPS, label[1:])}


FIXED_FOUR = (0, 1, 3, 4)
FREE_THREE = (2, 5, 6)


def fixed_four_labels():
    """Free-group combinations with a libnvvm level not yet compiled."""
    labels = []
    for combo in itertools.product("10n", repeat=len(FREE_THREE)):
        if "n" not in combo:
            continue
        if combo.count("n") == 1 and combo.count("1") == 2:
            continue
        bits = ["1"] * len(GROUPS)
        for index, level in zip(FREE_THREE, combo):
            bits[index] = level
        labels.append(bits_label(bits))
    return labels


def single_false_labels():
    """Per group: alone to libnvvm, alone full, alone rolled."""
    labels = []
    for group in GROUPS:
        labels.append(bits_label(
            "n" if g == group else "1" for g in GROUPS))
        for level in "10":
            labels.append(bits_label(
                level if g == group else "n" for g in GROUPS))
    return labels


def bits_label(bits):
    return "u" + "".join(bits)


def deviation_label(group):
    """All full except ``group`` rolled."""
    return bits_label("0" if g == group else "1" for g in GROUPS)


def factorial_labels(live):
    """Every full/rolled combination of ``live``; other groups full."""
    labels = []
    live_index = [GROUPS.index(g) for g in live]
    for combo in itertools.product("10", repeat=len(live)):
        bits = ["1"] * len(GROUPS)
        for index, bit in zip(live_index, combo):
            bits[index] = bit
        labels.append(bits_label(bits))
    return labels


def unroll_flags(label):
    from cubie.cuda_simsafe import UnrollFlags

    return UnrollFlags(**policy_flags(label))


def make_solver(system, system_name, algo_name, label):
    return pl.make_solver(
        system, system_name, algo_name,
        extra=dict(unroll=unroll_flags(label)),
    )


def open_records(out):
    out = Path(out)
    return pl.Records(out / "records.jsonl", extra=[out / "compiles.jsonl"])


def open_compiles(out):
    return pl.Records(Path(out) / "compiles.jsonl")


def compile_key(system_name, algo_name, label):
    return pl.task_key("compile", system_name, algo_name, label)


def compile_row(compiles, system_name, algo_name, label):
    return compiles.get(compile_key(system_name, algo_name, label))


def compiled(compiles, system_name, algo_name, label):
    row = compile_row(compiles, system_name, algo_name, label)
    return (row is not None and row.get("status") == "ok"
            and row.get("source_hash") == pl.source_hash())


# --- SASS --------------------------------------------------------------

INSTR = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\.L_x_(\d+):")
TARGET = re.compile(r"`\((\.L_x_\d+|[^)]+)\)")
COUNTED = ("BRA", "LDL", "STL", "LDS", "STS", "LDG", "STG", "MUFU",
           "FFMA", "FMUL", "FADD", "CALL", "VOTE", "ISETP", "FSETP",
           "SEL", "FSEL", "IMAD", "IADD3", "LEA", "MOV")
TERMINATORS = ("BRA", "EXIT", "RET", "BRX", "JMX")


def parse_sass(text):
    """Instructions as (pred, op, full op, body) and label -> index."""
    instrs = []
    labels = {}
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            labels[f".L_x_{m.group(1)}"] = len(instrs)
            continue
        m = INSTR.match(line)
        if m:
            body = m.group(2)
            pred = ""
            if body.startswith("@"):
                pred, body = body.split(None, 1)
            full = body.split()[0]
            instrs.append((pred, full.split(".")[0], full, body))
    return instrs, labels


def basic_blocks(instrs, labels):
    """Block (start, end) ranges and successor lists."""
    starts = {0} | set(labels.values())
    for k, (_, op, _, _) in enumerate(instrs):
        if op in TERMINATORS and k + 1 < len(instrs):
            starts.add(k + 1)
    starts = sorted(s for s in starts if s < len(instrs))
    index = {s: i for i, s in enumerate(starts)}
    ends = starts[1:] + [len(instrs)]
    succ = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        pred, op, full, body = instrs[e - 1]
        out = []
        fall = i + 1 if e < len(instrs) else None
        if op == "BRA":
            m = TARGET.search(body)
            if m and m.group(1) in labels:
                out.append(index[labels[m.group(1)]])
            if pred or ".DIV" in full or "BRA.U" in full:
                if fall is not None:
                    out.append(fall)
        elif op in ("EXIT", "RET"):
            if pred and fall is not None:
                out.append(fall)
        elif op in ("BRX", "JMX"):
            pass
        elif fall is not None:
            out.append(fall)
        succ.append(sorted(set(out)))
    return starts, ends, succ


def dominators(succ, entry=0):
    """Immediate dominators by the Cooper-Harvey-Kennedy iteration."""
    n = len(succ)
    preds = [[] for _ in range(n)]
    for u, outs in enumerate(succ):
        for v in outs:
            preds[v].append(u)
    order = []
    seen = [False] * n
    stack = [(entry, iter(succ[entry]))]
    seen[entry] = True
    while stack:
        node, it = stack[-1]
        advanced = False
        for v in it:
            if not seen[v]:
                seen[v] = True
                stack.append((v, iter(succ[v])))
                advanced = True
                break
        if not advanced:
            order.append(node)
            stack.pop()
    rpo = order[::-1]
    number = {b: i for i, b in enumerate(rpo)}
    idom = [None] * n
    idom[entry] = entry

    def intersect(a, b):
        while a != b:
            while number[a] > number[b]:
                a = idom[a]
            while number[b] > number[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            new = None
            for p in preds[b]:
                if p in number and idom[p] is not None:
                    new = p if new is None else intersect(p, new)
            if new is not None and idom[b] != new:
                idom[b] = new
                changed = True
    return idom, preds, number


def dominates(idom, a, b):
    while b is not None and b != a:
        if idom[b] == b:
            return False
        b = idom[b]
    return b == a


def natural_loops(succ):
    """Loop header -> body block set, plus the reachable block map."""
    idom, preds, reachable = dominators(succ)
    bodies = {}
    for u, outs in enumerate(succ):
        if u not in reachable:
            continue
        for h in outs:
            if dominates(idom, h, u):
                body = bodies.setdefault(h, {h})
                stack = [u]
                while stack:
                    x = stack.pop()
                    if x in body:
                        continue
                    body.add(x)
                    stack.extend(preds[x])
    return bodies, reachable


def loop_tree(bodies):
    """Parent header and depth of each loop by body containment."""
    headers = sorted(bodies, key=lambda h: (-len(bodies[h]), h))
    parent = {}
    for h in headers:
        best = None
        for g in headers:
            if g == h or len(bodies[g]) <= len(bodies[h]):
                continue
            if bodies[h] <= bodies[g]:
                if best is None or len(bodies[g]) < len(bodies[best]):
                    best = g
        parent[h] = best
    depth = {}
    for h in headers:
        d = 0
        g = h
        while parent[g] is not None:
            d += 1
            g = parent[g]
        depth[h] = d
    return parent, depth


def count_ops(instrs, rng):
    c = collections.Counter()
    for k in rng:
        pred, op, full, body = instrs[k]
        c["all"] += 1
        if op in COUNTED:
            c[op] += 1
        if op in ("LDL", "STL"):
            width = 4
            for part in full.split(".")[1:]:
                if part in ("64", "128", "32", "16", "8"):
                    width = int(part) // 8
            c[op + "_bytes"] += width
        if pred:
            c["predicated"] += 1
    return dict(c)


def sass_analysis(sass_path):
    """Whole-kernel opcode counts, back edges and natural loops."""
    if not sass_path:
        return None, None
    with gzip.open(sass_path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    instrs, labels = parse_sass(text)
    back_edges = 0
    for k, (_, op, _, body) in enumerate(instrs):
        if op == "BRA":
            m = TARGET.search(body)
            if m and m.group(1) in labels and labels[m.group(1)] <= k:
                back_edges += 1
    ops = collections.Counter(op for _, op, _, _ in instrs)
    counts = {name.lower(): ops[name] for name in COUNTED}
    counts["instructions"] = len(instrs)
    counts["ldl_stl"] = ops["LDL"] + ops["STL"]
    counts["back_edges"] = back_edges
    starts, ends, succ = basic_blocks(instrs, labels)
    bodies, reachable = natural_loops(succ)
    parent, depth = loop_tree(bodies)
    owner = [None] * len(starts)
    for h in sorted(bodies, key=lambda h: -len(bodies[h])):
        for b in bodies[h]:
            owner[b] = h
    loops = []
    for h in sorted(bodies, key=lambda h: (depth[h], starts[h])):
        exclusive = [k for b in bodies[h] if owner[b] == h
                     for k in range(starts[b], ends[b])]
        inclusive = sum(ends[b] - starts[b] for b in bodies[h])
        loops.append(dict(
            header=starts[h], depth=depth[h],
            parent=None if parent[h] is None else starts[parent[h]],
            blocks=len(bodies[h]), instructions=inclusive,
            span=[min(starts[b] for b in bodies[h]),
                  max(ends[b] for b in bodies[h])],
            exclusive=count_ops(instrs, exclusive),
        ))
    outside = [k for b in range(len(starts))
               if owner[b] is None and b in reachable
               for k in range(starts[b], ends[b])]
    counts["loops"] = len(loops)
    counts["outside"] = count_ops(instrs, outside)
    return counts, loops


# --- compiles ----------------------------------------------------------


def compile_payload(solver, helpers, out, key, compile_s):
    payload = pl.compile_payload(solver, helpers, out, key, compile_s)
    cubin = Path(payload["artefacts"]["cubin"]).read_bytes()
    payload["cubin_sha"] = hashlib.sha256(cubin).hexdigest()
    counts, loops = sass_analysis(payload["artefacts"].get("sass"))
    payload["sass_counts"] = counts
    payload["sass_loops"] = loops
    return payload


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
    path = result_path(out, key)
    path.write_text(json.dumps(payload, default=pl._json_default),
                    encoding="utf-8")
    print("@RESULT " + str(path), flush=True)


def result_path(out, key):
    """File a worker writes its compile payload to."""
    path = Path(out) / "results" / f"{pl.safe_name(key)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def compile_jobs(compiles, jobs, out, workers):
    """Compile ``jobs`` (dicts: system, algo, policy) in subprocesses."""
    pending = [
        job for job in jobs
        if not compiled(compiles, job["system"], job["algo"], job["policy"])
    ]
    running = []
    env = dict(os.environ)
    log_dir = Path(out) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            key = compile_key(job["system"], job["algo"], job["policy"])
            log = open(log_dir / f"{pl.safe_name(key)}.log", "w",
                       encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, str(BENCH), "--worker", "--out", str(out)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, text=True, env=env,
            )
            process.stdin.write(json.dumps(job))
            process.stdin.close()
            running.append((job, process, log, time.perf_counter()))
        still = []
        for job, process, log, started in running:
            key = compile_key(job["system"], job["algo"], job["policy"])
            if process.poll() is None:
                if time.perf_counter() - started > pl.COMPILE_TIMEOUT:
                    process.kill()
                    log.close()
                    compiles.append(
                        dict(key=key, task="compile", status="timeout",
                             source_hash=pl.source_hash(), **job)
                    )
                    continue
                still.append((job, process, log, started))
                continue
            stdout = process.stdout.read()
            log.close()
            stderr = log_dir.joinpath(f"{pl.safe_name(key)}.log").read_text(
                encoding="utf-8")
            payload = None
            for line in stdout.splitlines():
                if line.startswith("@RESULT "):
                    payload = json.loads(
                        Path(line[len("@RESULT "):]).read_text(
                            encoding="utf-8"))
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


def jobs_for(system_name, algo_name, labels):
    return [dict(system=system_name, algo=algo_name, policy=label)
            for label in labels]


def live_groups(compiles, system_name, algo_name):
    """Groups whose rolled deviation compiles to a different cubin."""
    full = compile_row(compiles, system_name, algo_name, FULL_LABEL)
    live = []
    shas = {}
    for group in GROUPS:
        row = compile_row(compiles, system_name, algo_name,
                          deviation_label(group))
        sha = row.get("cubin_sha") if row and row.get("status") == "ok" \
            else None
        shas[group] = sha
        if sha is None or sha != full["cubin_sha"]:
            live.append(group)
    return live, shas


def policy_labels(records, system_name, algo_name):
    """Wave policy labels of a config from its liveness row."""
    row = records.get(pl.task_key("liveness", system_name, algo_name))
    return list(row["policies"])


# --- banking -----------------------------------------------------------


def kernel_entries(system_name, algo_name, compiles, labels):
    """Build (label, policy, solver, bs, dynshared) per wave label."""
    entries = []
    for label in labels:
        policy = label.split(DUPLICATE_SUFFIX)[0]
        row = compile_row(compiles, system_name, algo_name, policy)
        if row is None or row.get("status") != "ok":
            continue
        # One system per solver (issue 685).
        solver = make_solver(pl.SYSTEMS[system_name]["build"](),
                             system_name, algo_name, policy)
        for plan, blocksize, dynshared in pl.launch_plans(
            row["occupancy"], equal_t_rows=False
        ):
            name = label if plan == "default" else f"{label}@{plan}"
            entries.append((name, policy, solver, blocksize, dynshared))
    return entries


def free_device_bytes():
    from cubie.cuda_simsafe import cuda

    return int(cuda.current_context().get_memory_info().free)


def probe_entry(solver, inits, params, duration, blocksize, floor):
    """Probe ms per fraction, and the fraction that caps the entry."""
    probes = {}
    for fraction, ratio in PROBE_LIMITS:
        ms, _, _ = pl.solve_once(solver, inits, params, duration / fraction,
                                 blocksize, snapshot=False)
        limit = floor.get(fraction)
        if limit is not None and ms > ratio * limit:
            again, _, _ = pl.solve_once(solver, inits, params,
                                        duration / fraction, blocksize,
                                        snapshot=False)
            ms = min(ms, again)
            if ms > ratio * limit:
                probes[fraction] = ms
                return probes, fraction
        probes[fraction] = ms
    return probes, None


def solver_groups(entries):
    """Consecutive rows that share a solver."""
    groups = []
    for entry in entries:
        if groups and groups[-1][0][2] is entry[2]:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def bank_wave(records, system_name, algo_name, entries, inits, params,
              duration, n_runs, block_solvers=None, wave=""):
    """Time rows in memory-sized blocks, each with the all-full reference."""
    groups = solver_groups(entries)
    reference_group = groups[0]
    pending = groups[1:]
    snapshot_reference = None
    floor = {}

    def admit(entry, block, capping):
        nonlocal snapshot_reference
        label, policy, solver, blocksize, dynshared = entry
        pl.pin_launch(solver, blocksize, dynshared)
        probes, capped_at = probe_entry(solver, inits, params, duration,
                                        blocksize, floor if capping else {})
        if capped_at is not None:
            records.append(
                dict(
                    key=pl.task_key("capped", system_name, algo_name, label),
                    task="capped", system=system_name, algo=algo_name,
                    label=label, policy=policy, wave=wave, block=block,
                    blocksize=blocksize, dynshared=dynshared,
                    probe_ms=round(probes[capped_at], 4),
                    probe_fraction=capped_at,
                    floor_probe_ms=round(floor[capped_at], 4),
                    floor_ms=round(floor["full"], 4), n_runs=n_runs,
                    duration=duration,
                )
            )
            print(f"  capped {label}: {probes[capped_at]:.1f} ms at "
                  f"duration/{capped_at} vs floor probe "
                  f"{floor[capped_at]:.1f} ms", flush=True)
            return False
        ms, wall, snapshot = pl.solve_once(
            solver, inits, params, duration, blocksize
        )
        if snapshot_reference is None:
            snapshot_reference = snapshot
        check = pl.compare_outputs(snapshot_reference, snapshot)
        geometry = pl.launch_geometry(solver, blocksize, n_runs, dynshared)
        records.append(
            dict(
                key=pl.task_key("solve", system_name, algo_name,
                                f"{label}|{wave}b{block}|warm"),
                task="solve", system=system_name, algo=algo_name,
                label=label, policy=policy, wave=wave, block=block,
                warm=True,
                round=-1, rep=-1, kernel_ms=round(ms, 4),
                wall_ms=round(wall, 3),
                probes={str(k): round(v, 4) for k, v in probes.items()},
                n_runs=n_runs, duration=duration, geometry=geometry,
                status_hist=snapshot["status_hist"], **check,
            )
        )
        del snapshot
        for fraction, value in probes.items():
            floor[fraction] = min(floor.get(fraction, value), value)
        floor["full"] = min(floor.get("full", ms), ms)
        return True

    block = 0
    while pending or block == 0:
        floor.clear()
        for entry in reference_group:
            admit(entry, block, False)
        members = []
        while pending and free_device_bytes() > BLOCK_MIN_FREE_BYTES and (
            block_solvers is None or len(members) < block_solvers
        ):
            group = pending.pop(0)
            kept = [entry for entry in group if admit(entry, block, True)]
            if kept:
                members.append((group[0][2], kept))
            else:
                group[0][2].close()
        rows = list(reference_group) + [
            entry for _, kept in members for entry in kept
        ]
        print(f"  block {block}: {len(rows)} rows, {len(pending)} solvers "
              f"pending, {free_device_bytes() >> 20} MB free", flush=True)
        label, policy, solver, blocksize, dynshared = reference_group[0]
        pl.pin_launch(solver, blocksize, dynshared)
        settle_start = time.perf_counter()
        while time.perf_counter() - settle_start < pl.SETTLE_S:
            pl.solve_once(solver, inits, params, duration, blocksize,
                          snapshot=False)
        for round_idx in range(pl.ROUNDS):
            for label, policy, solver, blocksize, dynshared in rows:
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
                                f"{label}|{wave}b{block}r{round_idx}k{rep}",
                            ),
                            task="solve", system=system_name,
                            algo=algo_name, label=label, policy=policy,
                            wave=wave, block=block, warm=False,
                            round=round_idx,
                            rep=rep, kernel_ms=round(ms, 4),
                            wall_ms=round(wall, 3), n_runs=n_runs,
                            duration=duration, blocksize=blocksize,
                            dynshared=dynshared,
                        )
                    )
            print(f"  block {block} round {round_idx} done", flush=True)
        for solver, _ in members:
            solver.close()
        block += 1
    reference_group[0][2].close()


def settled_ms(guarded, solver, duration):
    """Kernel ms of the faster of two solves at ``duration``."""
    first, _, _ = guarded(solver, duration, None, False)
    second, _, _ = guarded(solver, duration, None, False)
    return min(first, second)


def alias_rows(records, compiles, system_name, algo_name, labels):
    """Record labels sharing an earlier policy's cubin; return the rest."""
    seen = {}
    for row in compiles.select(task="compile", system=system_name,
                               algo=algo_name, status="ok"):
        if row["policy"] not in labels:
            seen.setdefault(row["cubin_sha"], row["policy"])
    distinct = []
    for label in labels:
        row = compile_row(compiles, system_name, algo_name, label)
        if row is None or row.get("status") != "ok":
            continue
        twin = seen.get(row["cubin_sha"])
        if twin is None:
            seen[row["cubin_sha"]] = label
            distinct.append(label)
            continue
        records.append(
            dict(key=pl.task_key("alias", system_name, algo_name, label),
                 task="alias", system=system_name, algo=algo_name,
                 label=label, equals=twin, cubin_sha=row["cubin_sha"])
        )
        print(f"  {label} compiles to the cubin of {twin}", flush=True)
    return distinct


def run_config(out, system_name, algo_name, workers, block_solvers=None,
               policy_set="live"):
    """Compile the liveness probe, the live factorial, then bank one wave."""
    out = Path(out)
    records = open_records(out)
    compiles = open_compiles(out)
    helpers = pl.spill_helpers()
    helpers.install_spill_capture()
    spec = pl.SYSTEMS[system_name]
    n_runs = spec["n_runs"]
    features_key = pl.task_key("features", system_name, algo_name)
    row = records.get(features_key)
    duration = float(row["duration"]) if row else None

    start = time.perf_counter()
    system = spec["build"]()
    codegen_s = time.perf_counter() - start
    base = make_solver(system, system_name, algo_name, FULL_LABEL)
    inits, params = spec["grid"](base, n_runs)
    base_key = compile_key(system_name, algo_name, FULL_LABEL)
    start = time.perf_counter()
    base.compile(inits, params, duration=duration or spec["duration"])
    compile_s = time.perf_counter() - start
    if not compiled(compiles, system_name, algo_name, FULL_LABEL):
        compiles.append(
            dict(key=base_key, task="compile", status="ok",
                 system=system_name, algo=algo_name, policy=FULL_LABEL,
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
        ms = settled_ms(guarded, base, duration)
        while ms < pl.MIN_SOLVE_MS and duration < spec["duration"] * (
            pl.MAX_DURATION_SCALE
        ):
            duration *= 2
            ms = settled_ms(guarded, base, duration)
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
                groups=list(GROUPS), codegen_s=round(codegen_s, 2),
            )
        )
    print(f"  full compiled in {compile_s:.0f} s; duration {duration}",
          flush=True)

    liveness_key = pl.task_key("liveness", system_name, algo_name)
    if not records.has(liveness_key):
        probe_labels = [deviation_label(g) for g in GROUPS]
        compile_jobs(compiles, jobs_for(system_name, algo_name,
                                        probe_labels + [LIBNVVM_LABEL]),
                     out, workers=workers)
        compiles.reload()
        live, shas = live_groups(compiles, system_name, algo_name)
        labels = factorial_labels(live)
        wave = ([FULL_LABEL, FULL_LABEL + DUPLICATE_SUFFIX, LIBNVVM_LABEL]
                + [label for label in labels if label != FULL_LABEL])
        records.append(
            dict(key=liveness_key, task="liveness", system=system_name,
                 algo=algo_name, live=live,
                 dead=[g for g in GROUPS if g not in live],
                 deviation_sha=shas, policies=wave)
        )
        print(f"  live groups {live}: {len(labels)} policies", flush=True)
    wave = WAVE_TAGS[policy_set]
    if policy_set == "live":
        labels = policy_labels(records, system_name, algo_name)
    elif policy_set == "single-false":
        labels = single_false_labels()
    else:
        labels = fixed_four_labels()
    compile_jobs(
        compiles,
        jobs_for(system_name, algo_name,
                 sorted({label.split(DUPLICATE_SUFFIX)[0]
                         for label in labels})),
        out, workers=workers,
    )
    compiles.reload()
    if policy_set != "live":
        labels = [FULL_LABEL, FULL_LABEL + DUPLICATE_SUFFIX] + alias_rows(
            records, compiles, system_name, algo_name, labels)
    if not records.has(pl.task_key("wavedone", system_name, algo_name,
                                   wave)):
        entries = kernel_entries(system_name, algo_name, compiles, labels)
        print(f"  wave: {len(entries)} launch rows", flush=True)
        bank_wave(records, system_name, algo_name, entries, inits, params,
                  duration, n_runs, block_solvers, wave)
        records.append(
            dict(key=pl.task_key("wavedone", system_name, algo_name, wave),
                 task="wavedone", system=system_name, algo=algo_name,
                 wave=wave)
        )
    records.append(
        dict(key=pl.task_key("configdone", system_name, algo_name, wave),
             task="configdone", system=system_name, algo=algo_name,
             wave=wave)
    )


# --- driver ------------------------------------------------------------


def selected_configs(args):
    return [
        c for c in config_list()
        if args.only is None or f"{c[0]}/{c[1]}" in args.only
    ]


def check_import_path():
    """Refuse to drive with a cubie that is not this worktree's source."""
    import cubie

    package = Path(cubie.__file__).resolve()
    expected = (REPO / "src" / "cubie").resolve()
    if package.parent != expected:
        sys.exit(f"cubie imports from {package.parent}, expected "
                 f"{expected}; set PYTHONPATH={REPO / 'src'}")


def drive(args):
    check_import_path()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = open_records(out)
    configs = selected_configs(args)
    wave = WAVE_TAGS[args.policy_set]
    print(f"{len(configs)} configs; policy set {args.policy_set}",
          flush=True)
    for system_name, algo_name in configs:
        records.reload()
        if records.has(pl.task_key("configdone", system_name, algo_name,
                                   wave)):
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
        ]
        if args.block_solvers:
            command += ["--block-solvers", str(args.block_solvers)]
        command += ["--policy-set", args.policy_set]
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
    for system_name, algo_name in config_list():
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
            key = (row["label"], f"{row.get('wave', '')}{row.get('block', 0)}")
            samples.setdefault(key, []).append(row["kernel_ms"])
        reference = {block: min(values) for (label, block), values
                     in samples.items() if label == FULL_LABEL}
        ratios = {key: min(values) / reference[key[1]]
                  for key, values in samples.items() if key[1] in reference}
        best = min(ratios.values())
        liveness = records.get(
            pl.task_key("liveness", system_name, algo_name)) or {}
        lines.append(f"## {system_name} / {algo_name}")
        lines.append("")
        lines.append(f"live groups: {', '.join(liveness.get('live', []))}; "
                     f"all-full floor per block: " + ", ".join(
                         f"b{block} {ms:.3f} ms"
                         for block, ms in sorted(reference.items())))
        lines.append("")
        lines.append(
            "| policy | block | bs | T | regs | local B | spill st/ld |"
            " SASS | LDL+STL | back edges | loops | min ms | spread |"
            " ratio to full | ratio to best | maxdiff | fails |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|"
                     "---|---|---|---|---|")
        ranked = sorted(ratios, key=ratios.get)
        for label, block in ranked:
            arr = np.asarray(samples[(label, block)])
            first = warm.get(label, {})
            geometry = first.get("geometry") or {}
            policy = first.get("policy") or label.split("@")[0]
            row = compile_row(compiles, system_name, algo_name, policy) or {}
            counts = row.get("sass_counts") or {}
            ratio = ratios[(label, block)]
            lines.append(
                f"| {label} | {block} | {geometry.get('blocksize', '')} | "
                f"{geometry.get('resident_threads', '')} | "
                f"{row.get('regs', '')} | {row.get('local_bytes', '')} | "
                f"{row.get('spill_store_bytes', '')}/"
                f"{row.get('spill_load_bytes', '')} | "
                f"{counts.get('instructions', '')} | "
                f"{counts.get('ldl_stl', '')} | "
                f"{counts.get('back_edges', '')} | "
                f"{counts.get('loops', '')} | "
                f"{arr.min():.3f} | {arr.max() / arr.min() - 1:.3f} | "
                f"{ratio:.3f} | {ratio / best:.3f} | "
                f"{first.get('max_abs_diff', float('nan')):.2e} | "
                f"{(first.get('status_hist') or {}).get('failed', '')} |"
            )
        aliases = records.select(task="alias", system=system_name,
                                 algo=algo_name)
        if aliases:
            lines.append("")
            lines.append("same cubin as an earlier policy: " + ", ".join(
                f"{row['label']} = {row['equals']}"
                for row in sorted(aliases, key=lambda r: r["label"])))
        capped = records.select(task="capped", system=system_name,
                                algo=algo_name)
        if capped:
            lines.append("")
            lines.append("capped (probe over the floor probe): " + ", ".join(
                f"{row['label']} {row['probe_ms']:.0f} ms at "
                f"1/{row.get('probe_fraction', 10)} vs "
                f"{row.get('floor_probe_ms', row['floor_ms']):.0f} ms"
                for row in sorted(capped, key=lambda r: r["label"])))
        lines.append("")
    summary = out / "summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--block-solvers", type=int, default=None,
                        help="solvers per wave block besides the reference")
    parser.add_argument("--policy-set", default="live",
                        choices=tuple(WAVE_TAGS))
    args = parser.parse_args(argv)
    if args.worker:
        worker_main(args.out)
    elif args.config:
        warnings.simplefilter("ignore")
        run_config(args.out, args.config[0], args.config[1], args.workers,
                   args.block_solvers, args.policy_set)
    elif args.list:
        for config in config_list():
            print(config)
        print(f"groups {GROUPS}; deviations "
              f"{[deviation_label(g) for g in GROUPS]}; "
              f"references {FULL_LABEL}{DUPLICATE_SUFFIX}, {LIBNVVM_LABEL}")
    elif args.report:
        report(args)
    elif args.run:
        drive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
