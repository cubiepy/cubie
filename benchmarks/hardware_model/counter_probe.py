"""Collect policy-specific iteration labels beside a completed timing bank.

The default command only constructs host objects and generated helpers.
``--execute`` compiles and solves on a GPU; it never collects timings.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cubie.cuda_simsafe import cuda

import placement_landscape as pl
import unroll_landscape as ul
from hardware_model.bank_analysis import analyse_config, compile_identity_key
from hardware_model.workload import describe_workload
from lorenz_mean_runtime import _compiled_cubin


COUNTER_NAMES = (
    "newton_iterations",
    "linear_solver_iterations",
    "attempted_steps",
    "rejected_steps",
)


def digest_json(value):
    """Hash the canonical JSON representation used by cohort manifests."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def read_rows(path):
    """Read original records with file, line and record-key receipts."""
    path = Path(path).resolve()
    rows = []
    for line, text in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if text.strip():
            row = json.loads(text)
            row["_receipt"] = dict(
                path=str(path),
                line=line,
                record_key=row["key"],
            )
            rows.append(row)
    return rows


def one_record(rows, task):
    """Require an unambiguous protocol record."""
    matches = [row for row in rows if row.get("task") == task]
    if len(matches) != 1:
        raise ValueError(f"Expected one {task} record, found {len(matches)}")
    return matches[0]


def load_cohort(bank, system, algo, cohort, policies=None):
    """Validate a completed targeted cohort and join eligible observations.

    Parameters
    ----------
    bank : str or Path
        Existing unroll bank directory; it is only read.
    system, algo, cohort : str
        Exact configuration and cohort identifiers from the bank.
    policies : sequence of str, optional
        Selected policy labels. Defaults to every unique cohort policy.

    Returns
    -------
    dict
        Original manifest, protocol and per-policy timing receipts.
    """
    bank = Path(bank).resolve()
    wave = ul.cohort_wave(cohort)
    rows = [
        row
        for row in read_rows(bank / "records.jsonl")
        if (row.get("system"), row.get("algo"), row.get("wave"))
        == (system, algo, wave)
    ]
    manifest_row = one_record(rows, "cohort_manifest")
    manifest = manifest_row["manifest"]
    digest = digest_json(manifest)
    if manifest_row["manifest_sha256"] != digest:
        raise ValueError("Cohort manifest digest does not match its payload")
    for task in ("cohort_protocol", "wavedone", "configdone"):
        if one_record(rows, task).get("manifest_sha256") != digest:
            raise ValueError(f"{task} does not identify this manifest")
    protocol = one_record(rows, "cohort_protocol")
    if (manifest["system"], manifest["algo"], manifest["wave"]) != (
        system,
        algo,
        wave,
    ):
        raise ValueError("Manifest configuration differs from record labels")
    if protocol["selected_policies"] != manifest["launch_policies"]:
        raise ValueError("Protocol policies differ from the manifest")
    if protocol["n_runs"] != manifest["n_runs"]:
        raise ValueError("Protocol run count differs from the manifest")
    if not np.isfinite(protocol["duration"]) or protocol["duration"] <= 0:
        raise ValueError("Cohort has no positive finite duration")
    available = list(
        dict.fromkeys(
            label.split("#")[0] for label in protocol["selected_policies"]
        )
    )
    policies = list(dict.fromkeys(policies or available))
    if set(policies) - set(available):
        raise ValueError("Requested policy was not in the completed cohort")
    compiles = [
        row
        for row in read_rows(bank / "compiles.jsonl")
        if (row.get("system"), row.get("algo")) == (system, algo)
    ]
    audit = analyse_config(
        (system, algo),
        rows,
        compiles,
        manifest["groups"],
        include_observations=True,
    )
    plans = []
    for policy in policies:
        ul.policy_flags(policy)
        candidates = [
            row
            for row in compiles
            if row.get("policy") == policy and row.get("status") == "ok"
        ]
        identities = {compile_identity_key(row) for row in candidates}
        if len(identities) != 1:
            raise ValueError(f"Missing/ambiguous compile identity: {policy}")
        compiled = candidates[-1]
        if compiled["source_hash"] != manifest["source_hash"] or (
            compiled["compiler_identity"] != manifest["compiler_identity"]
        ):
            raise ValueError(
                f"Compile and manifest identities differ: {policy}"
            )
        observations = [
            row
            for row in audit["observations"]
            if row["eligible"]
            and row["compile_identity"] == compile_identity_key(compiled)
        ]
        if not observations:
            raise ValueError(f"No eligible cohort timing for {policy}")
        if any(
            (row["duration"], row["n_runs"])
            != (protocol["duration"], protocol["n_runs"])
            for row in observations
        ):
            raise ValueError(f"Timing and protocol inputs differ: {policy}")
        # One label sample at each timed block size. Identical native
        # kernels may supply timing receipts, never another policy's counts.
        for blocksize in sorted({row["blocksize"] for row in observations}):
            matched = [
                row for row in observations if row["blocksize"] == blocksize
            ]
            plans.append(
                dict(
                    policy=policy,
                    blocksize=blocksize,
                    bank_compile=compiled,
                    timing_receipts=[
                        receipt
                        for row in matched
                        for receipt in row["sample_receipts"]
                    ],
                    timing_policies=sorted({row["policy"] for row in matched}),
                    bank_geometries=[row["resources"] for row in matched],
                )
            )
    return dict(
        bank=str(bank),
        manifest=manifest,
        manifest_sha256=digest,
        manifest_receipt=manifest_row["_receipt"],
        protocol=protocol,
        plans=plans,
    )


def check_environment(cohort):
    """Reject source, compiler, fixture or protocol drift before GPU work."""
    manifest = cohort["manifest"]
    current = ul.target_manifest(
        manifest["system"],
        manifest["algo"],
        manifest["requested_policies"],
        manifest["cohort"],
        manifest["protocol"]["block_solvers"],
    )["manifest"]
    mismatches = [key for key in manifest if current.get(key) != manifest[key]]
    if mismatches:
        raise ValueError(
            "Imported source/harness/environment differs from cohort: "
            + ", ".join(mismatches)
            + ". Use the original frozen tree and recorded environment."
        )
    if current["compiler_identity"]["cuda_simulation"]:
        raise ValueError("Counter labels require the real CUDA backend")
    return current


def array_identity(array):
    """Describe an exact array without replacing its saved values."""
    array = np.asarray(array)
    return dict(
        shape=list(array.shape),
        dtype=array.dtype.str,
        sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    )


def construct_pair(cohort, policy):
    """Construct matching host solvers and helpers without compilation.

    Returns
    -------
    tuple
        State-only solver, instrumented solver, inputs and provenance.
        The caller owns both solvers and closes them after use.
    """
    manifest = cohort["manifest"]
    system_name, algo = manifest["system"], manifest["algo"]
    system = pl.SYSTEMS[system_name]["build"]()
    reference = pl.make_solver(
        system,
        system_name,
        algo,
        extra=dict(unroll=ul.unroll_flags(policy)),
    )
    instrumented = pl.make_solver(
        system,
        system_name,
        algo,
        extra=dict(
            unroll=ul.unroll_flags(policy),
            output_types=["state", "iteration_counters"],
        ),
    )
    inputs = pl.SYSTEMS[system_name]["grid"](
        reference,
        cohort["protocol"]["n_runs"],
    )
    descriptions = [
        describe_workload(solver) for solver in (reference, instrumented)
    ]
    path = Path(system.gen_file.file_path).resolve()
    metadata = dict(
        policy=policy,
        unroll_flags=ul.policy_flags(policy),
        algorithm_settings=manifest["algorithm_settings"],
        generated_source=dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fn_hash=system.fn_hash,
        ),
        input_arrays={
            name: array_identity(value)
            for name, value in zip(("inits", "params"), inputs)
        },
        solvers={
            name: dict(
                config_hash=solver.kernel.config_hash,
                compilation_check=description["compilation_check"],
                workload=description["workload"],
            )
            for name, solver, description in zip(
                ("reference", "instrumented"),
                (reference, instrumented),
                descriptions,
            )
        },
    )
    return reference, instrumented, inputs, metadata


def compiled_identity(solver, out, name, blocksize, n_runs):
    """Persist native identity and require at least two occupancy waves."""
    geometry = pl.launch_geometry(solver, blocksize, n_runs)
    if geometry is None or geometry["blocks_per_sm"] <= 0:
        raise ValueError("Compiled kernel has no valid resident launch")
    if geometry["waves"] < 2:
        raise ValueError("Recorded n_runs provides fewer than two GPU waves")
    pl.pin_launch(solver, geometry["blocksize"], geometry["dynshared"])
    (kernel,) = solver.kernel.kernel.overloads.values()
    cubin, entry_name = _compiled_cubin(kernel)
    path = out / f"{name}.cubin"
    path.write_bytes(cubin)
    registers, local_bytes = pl.kernel_resources(solver)
    device = cuda.get_current_device()
    threads_per_run = int(solver.kernel.single_integrator.threads_per_step)
    sms = int(device.MULTIPROCESSOR_COUNT)
    minimum_runs = (
        2
        * sms
        * geometry["blocks_per_sm"]
        * (geometry["blocksize"] // threads_per_run)
    )
    if n_runs < minimum_runs:
        raise ValueError("Recorded grid is smaller than two resident waves")
    return dict(
        config_hash=solver.kernel.config_hash,
        cubin_path=str(path),
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        entry_name=entry_name,
        geometry=geometry,
        requested_blocksize=blocksize,
        registers_per_thread=registers,
        local_bytes_per_thread=local_bytes,
        multiprocessor_count=sms,
        threads_per_run=threads_per_run,
        minimum_runs_for_two_waves=minimum_runs,
        device_name=str(device.name),
        compute_capability=list(device.compute_capability),
    )


def sample_arrays(solver, inputs, duration, blocksize):
    """Copy every raw output before closing/reusing its solver."""
    result = solver.solve(
        *inputs,
        duration=duration,
        blocksize=blocksize,
        grid_type="verbatim",
    )
    if tuple(result._stride_order) != ("time", "variable", "run"):
        raise ValueError(
            f"Unsupported output stride order {result._stride_order}"
        )
    arrays = dict(
        state=np.array(result.state),
        status_codes=np.array(result.status_codes),
    )
    if result.iteration_counters is not None:
        arrays["iteration_counters"] = np.array(result.iteration_counters)
    return arrays, pl.status_histogram(result)


def summarize_counters(counters, n_runs, state_rows):
    """Preserve raw int32 counts and produce exact per-run distributions."""
    if counters.shape != (state_rows, 4, n_runs):
        raise ValueError(f"Unexpected counter shape: {counters.shape}")
    if counters.dtype != np.dtype("int32"):
        raise ValueError(f"Unexpected counter dtype: {counters.dtype}")
    if np.any(counters < 0):
        raise ValueError("Negative counters: invalid output or int32 overflow")
    if np.any(counters[:, 3, :] > counters[:, 2, :]):
        raise ValueError("Rejected steps exceed attempted steps")
    totals = counters.sum(axis=0, dtype=np.int64)
    distributions = {}
    for index, name in enumerate(COUNTER_NAMES):
        values, counts = np.unique(totals[index], return_counts=True)
        distributions[name] = dict(
            total=int(totals[index].sum()),
            min=int(values.min()),
            max=int(values.max()),
            mean=float(totals[index].mean()),
            values=values.tolist(),
            counts=counts.tolist(),
        )
    return totals, distributions


def execute_plan(cohort, plan, output, samples):
    """Collect labels with state-only equivalence checks, never timings."""
    protocol = cohort["protocol"]
    name = f"{plan['policy']}-bs{plan['blocksize']}"
    out = Path(output).resolve() / name
    out.mkdir(parents=True, exist_ok=False)
    reference, instrumented, inputs, preparation = construct_pair(
        cohort,
        plan["policy"],
    )
    metadata = dict(
        schema_version=1,
        kind="policy_iteration_labels",
        timing_role="none",
        predictor_role="validation_labels_only",
        cohort=cohort,
        plan=plan,
        preparation=preparation,
        samples=[],
        source_hash=pl.source_hash(),
        compiler_identity=ul.compiler_identity(),
        duration=protocol["duration"],
        n_runs=protocol["n_runs"],
        counter_columns=list(COUNTER_NAMES),
        axes=["save_row", "counter", "run"],
        limitations=[
            "Counts sum solves within save intervals, not per-step traces.",
            "Per-run totals do not determine warp-voted loop maxima.",
            "LU contributes one linear-solver iteration per solve.",
            "Counters include rejected attempts and t0 DAE initialization.",
            "Failed trajectories may have unsaved final interval counts.",
            "Raw int32 overflow can wrap; negative checks are partial.",
            "Instrumented code and occupancy can differ from the timing bank.",
        ],
    )
    try:
        compiled = {}
        for role, solver in (
            ("reference", reference),
            ("instrumented", instrumented),
        ):
            solver.compile(*inputs, duration=protocol["duration"])
            compiled[role] = compiled_identity(
                solver,
                out,
                role,
                plan["blocksize"],
                protocol["n_runs"],
            )
        metadata["compiled"] = compiled
        if (
            compiled["reference"]["cubin_sha256"]
            != (plan["bank_compile"]["cubin_sha"])
        ):
            raise ValueError(
                "State-only reference cubin differs from the bank"
            )
        np.savez_compressed(
            out / "inputs.npz", inits=inputs[0], params=inputs[1]
        )
        metadata["input_artifact"] = str(out / "inputs.npz")
        for sample in range(samples):
            ref, ref_status = sample_arrays(
                reference,
                inputs,
                protocol["duration"],
                compiled["reference"]["geometry"]["blocksize"],
            )
            raw, status = sample_arrays(
                instrumented,
                inputs,
                protocol["duration"],
                compiled["instrumented"]["geometry"]["blocksize"],
            )
            arrays = {f"reference_{key}": value for key, value in ref.items()}
            arrays.update(raw)
            artifact = out / f"sample-{sample:03d}.npz"
            # Persist first: failing checks must not discard raw evidence.
            np.savez_compressed(artifact, **arrays)
            entry = dict(
                index=sample,
                artifact=str(artifact),
                status_hist=status,
                reference_status_hist=ref_status,
                arrays={
                    key: array_identity(value) for key, value in arrays.items()
                },
            )
            metadata["samples"].append(entry)
            totals, distributions = summarize_counters(
                raw["iteration_counters"],
                protocol["n_runs"],
                raw["state"].shape[0],
            )
            arrays["per_run_totals"] = totals
            np.savez_compressed(artifact, **arrays)
            entry["arrays"]["per_run_totals"] = array_identity(totals)
            entry["distributions"] = distributions
            entry["checks"] = dict(
                state_exact=bool(np.array_equal(ref["state"], raw["state"])),
                state_finite=bool(np.all(np.isfinite(raw["state"]))),
                status_exact=bool(
                    np.array_equal(
                        ref["status_codes"],
                        raw["status_codes"],
                    )
                ),
                all_success=status["failed"] == ref_status["failed"] == 0,
                reference_single_chunk=int(reference.chunks) == 1,
                instrumented_single_chunk=int(instrumented.chunks) == 1,
            )
            entry["launch_chunks"] = dict(
                reference=int(reference.chunks),
                instrumented=int(instrumented.chunks),
            )
            entry["eligible_validation_label"] = all(entry["checks"].values())
        metadata["status"] = (
            "ok"
            if all(
                row["eligible_validation_label"] for row in metadata["samples"]
            )
            else "state_or_status_mismatch"
        )
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "labels.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        reference.close()
        instrumented.close()
    return metadata


def main():
    """Validate/construct by default; execute only with the explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--policy", action="append")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    cohort = load_cohort(
        args.bank,
        args.system,
        args.algo,
        args.cohort,
        args.policy,
    )
    check_environment(cohort)
    output = args.out.resolve()
    bank = args.bank.resolve()
    if output == bank or bank in output.parents:
        parser.error("--out must be outside the timing bank")
    output.mkdir(parents=True, exist_ok=True)
    if args.execute:
        results = [
            execute_plan(cohort, plan, output, args.samples)
            for plan in cohort["plans"]
        ]
        if any(row["status"] != "ok" for row in results):
            raise SystemExit(
                "Raw labels saved; state/status equivalence failed"
            )
    else:
        prepared = []
        for policy in dict.fromkeys(
            plan["policy"] for plan in cohort["plans"]
        ):
            ref, instrumented, _, metadata = construct_pair(cohort, policy)
            prepared.append(metadata)
            ref.close()
            instrumented.close()
        (output / "prepared.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    kind="counter_probe_preparation",
                    cohort=cohort,
                    solvers=prepared,
                    kernel_compilation=False,
                    gpu_execution=False,
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    print(
        f"Counter probe {'executed' if args.execute else 'prepared'}: {output}"
    )


if __name__ == "__main__":
    main()
