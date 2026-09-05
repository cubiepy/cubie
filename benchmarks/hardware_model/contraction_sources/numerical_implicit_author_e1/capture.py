"""Capture independent implicit source baselines and their original IR."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def asset(path):
    path = Path(path).resolve()
    return dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def checked(record):
    path = Path(record["path"])
    require(
        asset(path)["sha256"] == record["sha256"],
        "Changed input: " + str(path),
    )
    return path


def read(path):
    return json.loads(Path(path).read_text())


def load(record, name):
    spec = importlib.util.spec_from_file_location(name, checked(record))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(left, right, tolerance):
    require(
        left.shape == right.shape and left.dtype == right.dtype,
        "Comparison schema",
    )
    bad = ~np.isclose(right, left, equal_nan=False, **tolerance)
    return dict(
        bitwise_equal=left.tobytes() == right.tobytes(),
        original_gate_pass=bool(not bad.any()),
        failing_elements=int(bad.sum()),
        failing_trajectories=int(np.any(bad, axis=0).sum()),
        max_absolute_difference=float(np.max(np.abs(left - right))),
    )


def run(plan_path, output):
    plan = read(plan_path)
    require(asset(__file__) == plan["capture_runner"], "Capture epoch differs")
    for binding in plan["bindings"]:
        checked(binding)
    sys.path.insert(0, str(Path(plan["wrapper"]["path"]).parent))
    wrapper = load(plan["wrapper"], "implicit_source_wrapper")
    previous = load(plan["previous_author"], "implicit_factory_walker")
    prepared = wrapper.load_prepared(checked(plan["prepared"]))
    request = read(wrapper.checked(prepared["request"]))
    native_identity = read(wrapper.checked(prepared["native_identity"]))
    original_manifest = read(wrapper.checked(prepared["original_manifest"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="STARTED",
        source=asset(__file__),
        plan=asset(plan_path),
        prepared=plan["prepared"],
        records=[],
        comparisons=[],
        launches=0,
        scope="Own source-contract-disabled references and original IR; "
        "functional only, no timings, flags/defaults or model changes.",
    )
    solver = None
    saved = {}
    original_saved = {}
    try:
        receipt["device"] = wrapper.verify_device(native_identity)
        for case_id in plan["cases"]:
            case = prepared["cases"][case_id]
            workload_id = case_id[:12]
            workload = next(
                x
                for x in original_manifest["workloads"]
                if x["id"] == workload_id
            )
            tolerance = workload["numerical_tolerances"]
            kwargs, constants = wrapper.case_kwargs(
                request, case["workload"], case["candidate"]
            )
            require(
                wrapper.compiler_record(kwargs) == case["constructor_kwargs"],
                "Changed source constructor",
            )
            folder = output / case_id
            folder.mkdir()
            wrapper.set_cache_root(folder / "codegen")
            solver = wrapper.Solver(
                wrapper.landscape.SYSTEMS[request["system"]]["build"](),
                **kwargs,
            )
            if constants:
                solver.update(constants)
            require(
                "contract" in solver.update(contract=False),
                "Public contract flag not recognized",
            )
            solver.kernel.single_integrator.device_function
            baseline_compile = read(wrapper.checked(case["compile"]))
            expected_flags = set(
                baseline_compile["compiler_kwargs"]["fastmath"]["members"]
            ) - {"contract"}
            flags = []
            for name, factory in previous.factories(solver.kernel):
                require(
                    set(factory.jit_kwargs["fastmath"]) == expected_flags,
                    "Source factory contraction flags differ: " + name,
                )
                flags.append(
                    dict(
                        path=name,
                        kwargs=wrapper.compiler_record(
                            dict(factory.jit_kwargs)
                        ),
                    )
                )
            require(
                wrapper.landscape.bytes_per_run(solver)
                == case["shared_stride_bytes"],
                "Changed source placement",
            )
            with np.load(
                wrapper.checked(case["grid"]), allow_pickle=False
            ) as grid:
                inits = grid["initial_values"].copy()
                params = grid["parameters"].copy()
            require(
                inits.dtype == params.dtype == np.float32, "Changed FP32 grid"
            )
            protocol = prepared["protocol"]
            solver.compile(
                inits,
                params,
                duration=protocol["duration"],
                t0=protocol["t0"],
                grid_type="verbatim",
            )
            dispatcher = solver.kernel.kernel
            ((signature, specialization),) = dispatcher.overloads.items()
            library = specialization._codelibrary
            image = bytes(library._cubin)
            ir = bytes(specialization.metadata["ltoir"])
            require(
                bool(image) and bool(ir), "Missing original cubin or LTOIR"
            )
            image_path, ir_path = (
                folder / "kernel.cubin",
                folder / "cached_ltoir.bin",
            )
            image_path.write_bytes(image)
            ir_path.write_bytes(ir)
            process = subprocess.run(
                [
                    str(wrapper.checked(prepared["disassembler"])),
                    "-c",
                    str(image_path),
                ],
                capture_output=True,
                check=True,
            )
            (folder / "kernel.sass").write_bytes(process.stdout)
            resources = {
                name: int(getattr(dispatcher, method)(signature))
                for name, method in wrapper.RESOURCE_METHODS.items()
            }
            reference = wrapper.reference_arrays(case["reference"])
            original_saved[case_id] = reference
            record = dict(
                case=case_id,
                source=case_id,
                workload=workload_id,
                source_policy=case_id.split("_source_")[1][:4],
                tolerance=tolerance,
                constructor=case["constructor_kwargs"],
                constants=wrapper.compiler_record(constants),
                factory_flags=flags,
                original_compile=case["compile"],
                original_reference=case["reference"],
                original_cross_candidate_passed=case.get(
                    "original_cross_candidate_passed"
                ),
                signature=str(signature),
                cubin=asset(image_path),
                cached_ltoir=asset(ir_path),
                native=dict(cubin=asset(image_path), resources=resources),
                solves=[],
                public_source_contract=False,
            )
            receipt["records"].append(record)
            first = None
            for phase in ("warmup", "capture"):
                result = solver.solve(
                    inits,
                    params,
                    duration=protocol["duration"],
                    t0=protocol["t0"],
                    grid_type="verbatim",
                    blocksize=case["candidate"]["geometry"]["block_threads"],
                    nan_error_trajectories=False,
                )
                receipt["launches"] += 1
                arrays = dict(
                    state=np.array(result.state[-1]),
                    status=np.array(result.status_codes),
                )
                array_path = folder / (phase + ".npz")
                np.savez_compressed(array_path, **arrays)
                geometry = wrapper.landscape.launch_geometry(
                    solver, 128, protocol["n_runs"]
                )
                fingerprint = wrapper.array_digest(arrays)
                if first is None:
                    first = fingerprint
                checks = dict(
                    fp32=arrays["state"].dtype == np.float32,
                    finite=bool(np.isfinite(arrays["state"]).all()),
                    success=bool(np.all(arrays["status"] == 0)),
                    counter_free=result.iteration_counters is None,
                    exact_repeat=first == fingerprint,
                    original_array_schema=all(
                        arrays[k].shape == reference[k].shape
                        and arrays[k].dtype == reference[k].dtype
                        for k in arrays
                    ),
                    same_source_image=bytes(library._cubin) == image,
                    same_source_ir=bytes(specialization.metadata["ltoir"])
                    == ir,
                    same_dispatcher=solver.kernel.kernel is dispatcher,
                    one_specialization=len(dispatcher.overloads) == 1,
                    same_geometry=geometry["blocksize"] == 128
                    and geometry["dynshared"]
                    == max(4, case["shared_stride_bytes"] * 128)
                    and solver.kernel.run_params.num_chunks == 1
                    and int(solver.kernel.run_params[0].runs)
                    == protocol["n_runs"],
                    two_waves=geometry["waves"] >= 2,
                )
                record["solves"].append(
                    dict(
                        phase=phase,
                        arrays=asset(array_path),
                        content_sha256=fingerprint,
                        geometry=geometry,
                        checks=checks,
                        original_own_comparison=compare(
                            reference["state"], arrays["state"], tolerance
                        ),
                    )
                )
                wrapper.write(output / "receipt.json", receipt)
                require(
                    all(checks.values()),
                    "Source reference functional gate failed",
                )
                saved[case_id] = arrays
                del result
            solver.close()
            solver = None
        for workload_id in plan["workloads"]:
            keys = [x for x in plan["cases"] if x.startswith(workload_id)]
            full = next(x for x in keys if "source_0002_" in x)
            rolled = next(x for x in keys if "source_0000_" in x)
            tolerance = next(
                x["numerical_tolerances"]
                for x in original_manifest["workloads"]
                if x["id"] == workload_id
            )
            receipt["comparisons"].append(
                dict(
                    workload=workload_id,
                    tolerance=tolerance,
                    original_fastmath_bank=compare(
                        original_saved[full]["state"],
                        original_saved[rolled]["state"],
                        tolerance,
                    ),
                    public_contract_false=compare(
                        saved[full]["state"], saved[rolled]["state"], tolerance
                    ),
                )
            )
        receipt["status"] = (
            "IMPLICIT_CONTRACT_FALSE_OWN_REFERENCES_AND_IR_COMPLETE"
        )
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        if solver is not None:
            solver.close()
    print(
        json.dumps(
            dict(
                status=receipt["status"],
                launches=receipt["launches"],
                comparisons=receipt["comparisons"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
