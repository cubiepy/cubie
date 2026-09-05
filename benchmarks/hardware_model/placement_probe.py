"""Collect isolated, matched local/shared placement contrasts on a GPU.

Host construction is the default. Only ``--execute`` compiles or solves.
Each persistent worker owns one Solver and its own buffer registry.
"""

import argparse
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from cubie.cuda_simsafe import cuda
from cubie.buffer_registry import buffer_registry

import placement_landscape as pl
import unroll_landscape as ul
from hardware_model.workload import describe_workload
from lorenz_mean_runtime import _compiled_cubin


CASES = {
    "chain32-kvaerno3-stage_base-bs64": (
        "chain32",
        "kvaerno3",
        "stage_base",
        64,
    ),
    "chain64-radau5-delta-bs32": (
        "chain64",
        "radau_iia_5",
        "delta",
        32,
    ),
    "chain32-vern7-stage_accumulator-bs32": (
        "chain32",
        "vern7",
        "stage_accumulator",
        32,
    ),
}
ROLES = ("baseline", "duplicate", "shared")
SCRIPT = Path(__file__).resolve()


def digest(value):
    """Hash an exact JSON protocol or configuration."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def array_receipt(value):
    """Record a raw array's shape, dtype and value identity."""
    value = np.asarray(value)
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    )


def write_json(path, value):
    """Write strict JSON with no silent coercion of unknown objects."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def manifest_for(case, policy, cohort, blocks):
    """Describe the exact fresh source, compiler and paired protocol."""
    system, algo, buffer, blocksize = CASES[case]
    flags = ul.policy_flags(policy)
    if set(flags) != set(ul.GROUPS):
        raise ValueError("A policy must specify all eight unroll groups")
    base = ul.target_manifest(system, algo, [policy], cohort)["manifest"]
    if base["compiler_identity"]["cuda_simulation"]:
        raise ValueError("Placement probes require the real CUDA backend")
    if pl.ROUNDS * pl.REPEATS < 6 or pl.MIN_SOLVE_MS < 20:
        raise ValueError("Harness requires at least six samples of 20 ms")
    return dict(
        schema_version=1,
        kind="matched_placement_cohort",
        case=case,
        cohort=cohort,
        system=system,
        algo=algo,
        buffer=buffer,
        unroll_policy=policy,
        unroll_flags=flags,
        requested_blocksize=blocksize,
        source_and_compiler=base,
        probe_source_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        protocol=dict(
            roles=list(ROLES),
            paired_blocks=blocks,
            rounds=pl.ROUNDS,
            repeats=pl.REPEATS,
            samples_per_role_per_block=pl.ROUNDS * pl.REPEATS,
            minimum_kernel_ms=pl.MIN_SOLVE_MS,
            initial_duration=pl.SYSTEMS[system]["duration"],
            maximum_duration=(
                pl.SYSTEMS[system]["duration"] * pl.MAX_DURATION_SCALE
            ),
            duration_rule="double duration until all samples reach minimum",
            minimum_occupancy_waves=2,
            order="baseline/shared/duplicate, reversed on alternate samples",
            settle_s=pl.SETTLE_S,
            n_runs=pl.SYSTEMS[system]["n_runs"],
            geometry="pin exact requested block size; reject limited geometry",
            registry_isolation="one Solver per persistent worker process",
            link_diagnostics="unchanged backend; no global diagnostic hook",
            clock_and_load="external clock and GPU load are not measured",
        ),
    )


class ProbeWorker:
    """One persistent process isolates a Solver and its compiler state."""

    def __init__(self, job, output):
        self.log = None
        self.process = None
        try:
            self.log = (output / f"{job['role']}.log").open(
                "w", encoding="utf-8"
            )
            self.process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
            )
            self.ready = self.request(dict(command="construct", job=job))
        except BaseException:
            cleanup = self.close()
            print(json.dumps(dict(worker_cleanup=cleanup)), file=sys.stderr)
            raise

    def request(self, message):
        """Run one sequential worker command and return its exact receipt."""
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker exited; inspect {self.log.name}")
        result = json.loads(line)
        if result["status"] != "ok":
            raise RuntimeError(result["error"] + f"; log {self.log.name}")
        return result["payload"]

    def close(self):
        """Release each owned handle and bound worker shutdown waits."""
        result = dict(errors=[], forced_shutdown=False, returncode=None)

        def close_handle(handle):
            if handle is None or handle.closed:
                return
            try:
                handle.close()
            except BrokenPipeError:
                pass
            except Exception as error:
                result["errors"].append(
                    f"close: {type(error).__name__}: {error}"
                )

        process = self.process
        try:
            if process is not None:
                close_handle(process.stdin)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    result["forced_shutdown"] = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                result["returncode"] = process.returncode
        except Exception as error:
            result["errors"].append(f"wait: {type(error).__name__}: {error}")
        finally:
            if process is not None:
                close_handle(process.stdout)
            close_handle(self.log)
        return result


def construct(job):
    """Construct and inspect one solver without requesting a batch kernel."""
    manifest = job["manifest"]
    current = manifest_for(
        manifest["case"],
        manifest["unroll_policy"],
        manifest["cohort"],
        manifest["protocol"]["paired_blocks"],
    )
    if digest(current) != digest(manifest):
        raise ValueError("Worker source/compiler/protocol differs from parent")
    system_name, algo = manifest["system"], manifest["algo"]
    system = pl.SYSTEMS[system_name]["build"]()
    location = "shared" if job["role"] == "shared" else "local"
    solver = pl.make_solver(
        system,
        system_name,
        algo,
        placement={pl.setting_name(manifest["buffer"]): location},
        extra=dict(unroll=ul.unroll_flags(manifest["unroll_policy"])),
    )
    descriptor = describe_workload(solver)
    entries = [
        entry
        for entry in descriptor["buffers"]
        if entry["name"] == manifest["buffer"] and entry["elements"] > 0
    ]
    if not entries or any(entry["location"] != location for entry in entries):
        raise ValueError(
            "Requested named buffer placement was not constructed"
        )
    inputs = pl.SYSTEMS[system_name]["grid"](
        solver,
        manifest["protocol"]["n_runs"],
    )
    source = Path(system.gen_file.file_path)
    receipt = dict(
        role=job["role"],
        location=location,
        buffers=entries,
        resolved_buffers=resolved_buffers(solver, entries, location),
        config_hash=solver.kernel.config_hash,
        inputs={
            name: array_receipt(value)
            for name, value in zip(
                ("inits", "params"),
                inputs,
            )
        },
        generated_source=dict(
            path=str(source.resolve()),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            fn_hash=system.fn_hash,
        ),
        workload=descriptor["workload"],
        compilation_check=descriptor["compilation_check"],
    )
    return solver, inputs, receipt


def resolved_buffers(solver, entries, expected):
    """Record alias-resolved host layouts and allocator memory branches."""
    result = []

    def slice_values(value):
        return None if value is None else [value.start, value.stop, value.step]

    for entry in entries:
        owner = solver.kernel.single_integrator._algo_step
        for attribute in entry["owner"].split(".")[1:]:
            owner = getattr(owner, attribute)
        group = buffer_registry._groups[owner]
        name = entry["name"]
        shared = group.shared_layout.get(name)
        persistent = group.persistent_layout.get(name)
        local = group.local_sizes.get(name)
        location = (
            "shared"
            if shared is not None
            else "persistent_local"
            if persistent is not None
            else "local"
        )
        allocator = group.get_allocator(name)
        closure = inspect.getclosurevars(allocator.py_func).nonlocals
        alias = entry["alias_target"]
        record = dict(
            owner=entry["owner"],
            owner_type=entry["owner_type"],
            name=name,
            resolved_location=location,
            shared_slice=slice_values(shared),
            persistent_slice=slice_values(persistent),
            local_size=local,
            alias_target=alias,
            alias_parent_location=(
                group.entries[alias].location if alias else None
            ),
            alias_parent_shared_slice=slice_values(
                group.shared_layout.get(alias)
            ),
            allocator_use_shared=bool(closure["_use_shared"]),
            allocator_use_persistent=bool(closure["_use_persistent"]),
            allocator_local_size=int(closure["_local_size"]),
            native_overloads=len(allocator.overloads),
        )
        if (
            location != expected
            or record["allocator_use_shared"] != (expected == "shared")
            or record["allocator_use_persistent"]
            or (expected == "local" and local is None)
            or record["native_overloads"] != 0
        ):
            raise ValueError(f"Unexpected resolved allocator layout: {record}")
        result.append(record)
    return result


def compile_solver(solver, inputs, job):
    """Compile and record requested/limited geometry before any solve."""
    manifest = job["manifest"]
    output = Path(job["output"])
    solver.compile(*inputs, duration=manifest["protocol"]["initial_duration"])
    dispatcher = solver.kernel.kernel
    (kernel,) = dispatcher.overloads.values()
    cubin, entry = _compiled_cubin(kernel)
    cubin_sha = hashlib.sha256(cubin).hexdigest()
    key = pl.task_key(
        "placement_probe", manifest["system"], manifest["algo"], job["role"]
    )
    artifacts = pl.persist_kernel(output, key, cubin)
    assemblies = dispatcher.inspect_asm()
    if len(assemblies) != 1:
        raise ValueError("Expected one PTX specialization")
    (ptx,) = assemblies.values()
    ptx_path = output / f"{job['role']}.ptx"
    ptx_path.write_text(ptx, encoding="utf-8")
    artifacts["ptx"] = str(ptx_path)
    if artifacts["sass"] is None:
        raise ValueError("nvdisasm is required to retain SASS")
    n_runs = manifest["protocol"]["n_runs"]
    blocksize = manifest["requested_blocksize"]
    limited = pl.launch_geometry(solver, blocksize, n_runs)
    requested = pl.launch_geometry(
        solver,
        blocksize,
        n_runs,
        dynshared=max(4, pl.bytes_per_run(solver) * min(n_runs, blocksize)),
    )
    reasons = []
    if limited is None or requested is None:
        reasons.append("no_valid_launch_geometry")
    else:
        if limited["blocksize"] != blocksize:
            reasons.append("production_limit_changes_requested_blocksize")
        if requested["blocks_per_sm"] <= 0:
            reasons.append("zero_resident_blocks")
        if requested["waves"] < 2:
            reasons.append("fewer_than_two_occupancy_waves")
    registers, local_bytes = pl.kernel_resources(solver)
    result = dict(
        role=job["role"],
        source_hash=pl.source_hash(),
        compiler_identity=ul.compiler_identity(),
        config_hash=solver.kernel.config_hash,
        cubin_sha256=cubin_sha,
        entry_name=entry,
        artifacts=artifacts,
        ptx_stage="dispatcher diagnostic PTX; cubin may be linked from LTO IR",
        requested_blocksize=blocksize,
        requested_geometry=requested,
        production_limited_geometry=limited,
        blocked_reasons=reasons,
        registers_per_thread=registers,
        local_bytes_per_thread=local_bytes,
        shared_bytes_per_run=pl.bytes_per_run(solver),
        multiprocessor_count=int(
            cuda.get_current_device().MULTIPROCESSOR_COUNT
        ),
        compute_capability=list(cuda.get_current_device().compute_capability),
        actual_shared_carveout=None,
        carveout_reason="Launch occupancy does not reveal actual carveout",
    )
    if not reasons:
        pl.pin_launch(solver, blocksize, requested["dynshared"])
        actual = pl.launch_geometry(solver, blocksize, n_runs)
        result["actual_pinned_geometry"] = actual
        if actual is None:
            reasons.append("no_valid_pinned_geometry")
        else:
            for field in (
                "blocksize",
                "dynshared",
                "bytes_per_run",
                "blocks_per_sm",
                "resident_threads",
                "waves",
            ):
                if actual[field] != requested[field]:
                    reasons.append(f"pinned_geometry_differs_{field}")
            if actual["blocksize"] != blocksize:
                reasons.append("pinned_geometry_changes_requested_blocksize")
            if actual["blocks_per_sm"] <= 0 or actual["waves"] < 2:
                reasons.append("pinned_geometry_fails_residency_or_waves")
    else:
        result["actual_pinned_geometry"] = None
    np.savez_compressed(
        output / f"{job['role']}-inputs.npz", inits=inputs[0], params=inputs[1]
    )
    return result


def solve_sample(solver, inputs, job, message):
    """Retain every raw timing and optional numerical snapshot."""
    duration = message["duration"]
    blocksize = job["manifest"]["requested_blocksize"]
    start = time.perf_counter()
    result = solver.solve(
        *inputs, duration=duration, blocksize=blocksize, grid_type="verbatim"
    )
    wall_ms = (time.perf_counter() - start) * 1000
    record = dict(
        role=job["role"],
        kernel_ms=pl.kernel_ms(solver),
        wall_ms=wall_ms,
        duration=duration,
        n_runs=job["manifest"]["protocol"]["n_runs"],
        chunks=int(solver.chunks),
        blocksize=blocksize,
        status_hist=pl.status_histogram(result),
        finite_state=bool(np.all(np.isfinite(result.state))),
    )
    if message.get("snapshot"):
        path = Path(job["output"]) / (
            f"{job['role']}-{message['sample_id']}.npz"
        )
        arrays = dict(
            state=np.array(result.state),
            status_codes=np.array(result.status_codes),
        )
        np.savez_compressed(path, **arrays)
        record["snapshot"] = str(path)
        record["arrays"] = {
            name: array_receipt(value) for name, value in arrays.items()
        }
    return record


def worker_main():
    """Serve host/compile/solve requests with one isolated solver."""
    solver = None
    try:
        for line in sys.stdin:
            message = json.loads(line)
            command = message["command"]
            with contextlib.redirect_stdout(sys.stderr):
                if command == "construct":
                    if solver is not None:
                        raise ValueError(
                            "Each worker may construct one Solver"
                        )
                    job = message["job"]
                    solver, inputs, payload = construct(job)
                elif command == "compile":
                    payload = compile_solver(solver, inputs, job)
                elif command == "solve":
                    payload = solve_sample(solver, inputs, job, message)
                elif command == "close":
                    solver.close()
                    solver = None
                    payload = dict(closed=True)
                else:
                    raise ValueError(f"Unknown command: {command}")
            print(json.dumps(dict(status="ok", payload=payload)), flush=True)
            if command == "close":
                break
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps(
                dict(status="failed", error=f"{type(error).__name__}: {error}")
            ),
            flush=True,
        )
    finally:
        if solver is not None:
            solver.close()


def compare_snapshots(reference, candidate):
    """Retain exact state/status differences without a fitted tolerance."""
    with (
        np.load(reference["snapshot"]) as a,
        np.load(candidate["snapshot"]) as b,
    ):
        if a["state"].shape != b["state"].shape:
            return dict(shape_match=False, exact=False)
        state_a, state_b = a["state"], b["state"]
        delta = np.abs(state_a.astype(np.float64) - state_b.astype(np.float64))
        finite = bool(
            np.all(np.isfinite(state_a)) and np.all(np.isfinite(state_b))
        )
        return dict(
            shape_match=True,
            finite=finite,
            exact=bool(np.array_equal(state_a, state_b)),
            status_exact=bool(
                np.array_equal(a["status_codes"], b["status_codes"])
            ),
            maximum_absolute_difference=float(delta.max()) if finite else None,
            differing_values=int(np.count_nonzero(state_a != state_b)),
            differing_runs=int(
                np.count_nonzero(np.any(state_a != state_b, axis=(0, 1)))
            ),
        )


def measure(workers, manifest, output, receipt):
    """Collect a complete repeated cohort at a common adequate duration."""
    records = output / "timings.jsonl"
    sequence = 0

    def take(role, duration, phase, attempt, block, index, snapshot=False):
        nonlocal sequence
        key = f"a{attempt}-b{block}-{phase}-{index}-{sequence}"
        sequence += 1
        row = workers[role].request(
            dict(
                command="solve",
                duration=duration,
                snapshot=snapshot,
                sample_id=key,
            )
        )
        row.update(
            key=key,
            phase=phase,
            attempt=attempt,
            paired_block=block,
            sample=index,
            manifest_sha256=receipt["manifest_sha256"],
            source_hash=manifest["source_and_compiler"]["source_hash"],
            cubin_sha256=receipt["compiles"][role]["cubin_sha256"],
        )
        with records.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if (
            row["chunks"] != 1
            or row["status_hist"]["failed"]
            or not row["finite_state"]
        ):
            raise ValueError(f"Invalid solve in retained record {key}")
        return row

    duration = manifest["protocol"]["initial_duration"]
    maximum = manifest["protocol"]["maximum_duration"]
    attempt = 0
    receipt["attempts"] = []
    while duration <= maximum:
        current = dict(attempt=attempt, duration=duration, blocks=[])
        receipt["attempts"].append(current)
        pilot = [
            take(role, duration, "pilot", attempt, -1, 0) for role in ROLES
        ]
        if min(row["kernel_ms"] for row in pilot) < pl.MIN_SOLVE_MS:
            current["status"] = "duration_too_short"
            duration *= 2
            attempt += 1
            continue
        short = False
        for block in range(manifest["protocol"]["paired_blocks"]):
            warm = {
                role: take(role, duration, "warm", attempt, block, 0, True)
                for role in ROLES
            }
            checks = {
                role: compare_snapshots(warm["baseline"], warm[role])
                for role in ROLES
            }
            block_receipt = dict(
                block=block, warm=warm, numerical_checks=checks
            )
            current["blocks"].append(block_receipt)
            if not all(
                all(
                    check.get(key, False)
                    for key in (
                        "shape_match",
                        "finite",
                        "exact",
                        "status_exact",
                    )
                )
                for check in checks.values()
            ):
                raise ValueError(
                    "Warm state/status differences retained; "
                    "cohort numerical equivalence failed"
                )
            start = time.perf_counter()
            settle_index = 0
            while time.perf_counter() - start < pl.SETTLE_S:
                for role in ROLES:
                    take(
                        role, duration, "settle", attempt, block, settle_index
                    )
                settle_index += 1
            samples = []
            for index in range(pl.ROUNDS * pl.REPEATS):
                order = ("baseline", "shared", "duplicate")
                if index % 2:
                    order = tuple(reversed(order))
                samples.extend(
                    take(role, duration, "measurement", attempt, block, index)
                    for role in order
                )
            block_receipt["measurement_keys"] = [row["key"] for row in samples]
            short |= min(row["kernel_ms"] for row in samples) < pl.MIN_SOLVE_MS
        current["status"] = "duration_too_short" if short else "complete"
        if not short:
            receipt.update(
                duration=duration,
                accepted_attempt=attempt,
                timing_records=str(records),
                status="complete",
            )
            return
        duration *= 2
        attempt += 1
    raise ValueError(
        "No complete >=20ms cohort within the recorded duration bound"
    )


def run_case(args):
    """Prepare or execute one immutable named placement cohort."""
    manifest = manifest_for(args.case, args.policy, args.cohort, args.blocks)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(manifest=manifest, manifest_sha256=digest(manifest))
    write_json(output / "manifest.json", receipt)
    workers = {}
    try:
        receipt["construction"] = {}
        for role in ROLES:
            workers[role] = ProbeWorker(
                dict(role=role, manifest=manifest, output=str(output)), output
            )
            receipt["construction"][role] = workers[role].ready
        if (
            len(
                {
                    digest(row["inputs"])
                    for row in receipt["construction"].values()
                }
            )
            != 1
        ):
            raise ValueError("Worker input grids differ")
        if (
            workers["baseline"].ready["config_hash"]
            != (workers["duplicate"].ready["config_hash"])
        ):
            raise ValueError(
                "Independent local baseline configurations differ"
            )
        if not args.execute:
            receipt.update(
                status="prepared",
                kernel_compilation=False,
                gpu_execution=False,
            )
            return
        receipt["compiles"] = {}
        for role, worker in workers.items():
            receipt["compiles"][role] = worker.request(dict(command="compile"))
        if any(row["blocked_reasons"] for row in receipt["compiles"].values()):
            receipt["status"] = "blocked_geometry"
            raise ValueError("Exact requested geometry is not comparable")
        shas = {
            role: row["cubin_sha256"]
            for role, row in receipt["compiles"].items()
        }
        if shas["baseline"] != shas["duplicate"]:
            raise ValueError("Independent baseline cubin identities differ")
        receipt["physical_code_identities"] = len(set(shas.values()))
        receipt["shared_aliases_baseline"] = shas["shared"] == shas["baseline"]
        measure(workers, manifest, output, receipt)
    except Exception as error:
        receipt.setdefault("status", "failed")
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        receipt["worker_cleanup"] = {}
        for role, worker in workers.items():
            try:
                receipt["worker_cleanup"][role] = worker.close()
            except Exception as error:
                receipt["worker_cleanup"][role] = dict(
                    errors=[f"{type(error).__name__}: {error}"],
                    forced_shutdown=True,
                )
        cleanup_failed = any(
            row["errors"]
            or row["forced_shutdown"]
            or row.get("returncode", 0) not in (0, None)
            for row in receipt["worker_cleanup"].values()
        )
        if cleanup_failed:
            receipt["status_before_cleanup_failure"] = receipt.get("status")
            receipt["status"] = "failed_cleanup"
        write_json(output / "result.json", receipt)
        if cleanup_failed and not unwinding:
            raise RuntimeError("Worker cleanup failed; inspect result.json")


def main():
    """Construct host probes by default; GPU execution requires --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--policy")
    parser.add_argument("--cohort")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main()
        return
    if (
        not all((args.case, args.policy, args.cohort, args.out))
        or args.blocks < 1
    ):
        parser.error(
            "--case, --policy, --cohort, --out and positive --blocks required"
        )
    run_case(args)
    print(f"Placement cohort written: {args.out.resolve()}")


if __name__ == "__main__":
    main()
