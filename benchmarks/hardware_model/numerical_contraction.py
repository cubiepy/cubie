"""Compare a single compile-flag intervention on frozen FP32 RK23 inputs."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np

from cubie import Solver
from cubie.CUDAFactory import CUDAFactory
from cubie.cache_root import set_cache_root
from benchmarks import placement_landscape as landscape


def require(condition, message):
    """Refuse an unmet diagnostic contract."""
    if not condition:
        raise ValueError(message)


def factories(root):
    """Visit concrete cached-compilation children without building them."""
    pending = [("kernel", root)]
    seen = set()
    rows = []
    while pending:
        name, item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        rows.append((name, item))
        for key, value in vars(item).items():
            if isinstance(value, CUDAFactory):
                pending.append((name + "." + key, value))
    return rows


def run(prepared_path, wrapper_path, output):
    """Retain original failures and test contraction as one causal input."""
    spec = importlib.util.spec_from_file_location(
        "frozen_wrapper", wrapper_path
    )
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    prepared = wrapper.load_prepared(prepared_path)
    request = wrapper.read(wrapper.checked(prepared["request"]))
    identity = wrapper.read(wrapper.checked(prepared["native_identity"]))
    manifest = wrapper.read(wrapper.checked(prepared["original_manifest"]))
    workload = next(
        x for x in manifest["workloads"] if x["id"] == "workload_006"
    )
    tolerance = workload["numerical_tolerances"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="STARTED",
        started_utc=datetime.now(timezone.utc).isoformat(),
        source=wrapper.asset(__file__),
        wrapper=wrapper.asset(wrapper_path),
        prepared=wrapper.asset(prepared_path),
        tolerance=tolerance,
        records=[],
        comparisons=[],
        scope="Functional FP32 contraction intervention only. No timing, "
        "global accuracy, policy admission or original gate replacement.",
    )
    solver = None
    saved = {}
    try:
        receipt["device"] = wrapper.verify_device(identity)
        for mode in ("original", "contract_false"):
            for source in ("0002", "0000"):
                case_id = "workload_006_source_" + source + "_b128_s102400"
                case = prepared["cases"][case_id]
                folder = output / (mode + "_" + source)
                folder.mkdir()
                kwargs, constants = wrapper.case_kwargs(
                    request, case["workload"], case["candidate"]
                )
                require(
                    wrapper.compiler_record(kwargs)
                    == case["constructor_kwargs"],
                    "Original constructor changed",
                )
                require("jit_flags" not in kwargs, "Unexpected existing flags")
                set_cache_root(folder / "codegen")
                solver = Solver(
                    landscape.SYSTEMS[request["system"]]["build"](), **kwargs
                )
                if constants:
                    solver.update(constants)
                if mode == "contract_false":
                    require(
                        "contract" in solver.update(contract=False),
                        "Public contraction update was not recognized",
                    )
                solver.kernel.single_integrator.device_function
                require(not solver.kernel.kernel.overloads, "Early compile")
                require(
                    landscape.bytes_per_run(solver)
                    == case["shared_stride_bytes"],
                    "Source placement changed",
                )
                baseline_flags = set(
                    wrapper.read(wrapper.checked(case["compile"]))[
                        "compiler_kwargs"
                    ]["fastmath"]["members"]
                )
                expected_flags = baseline_flags - (
                    {"contract"} if mode == "contract_false" else set()
                )
                factory_flags = []
                for name, factory in factories(solver.kernel):
                    actual = dict(factory.jit_kwargs)
                    require(set(actual["fastmath"]) == expected_flags, name)
                    factory_flags.append(
                        dict(path=name, kwargs=wrapper.compiler_record(actual))
                    )
                with np.load(
                    wrapper.checked(case["grid"]), allow_pickle=False
                ) as grid:
                    inits = grid["initial_values"].copy()
                    params = grid["parameters"].copy()
                protocol = prepared["protocol"]
                record = dict(
                    mode=mode,
                    case=case_id,
                    factory_flags=factory_flags,
                    solves=[],
                )
                receipt["records"].append(record)
                if mode == "original":
                    _, record["native"] = wrapper.compile_exact(
                        solver, case, protocol, inits, params, prepared, folder
                    )
                else:
                    solver.compile(
                        inits,
                        params,
                        duration=protocol["duration"],
                        t0=protocol["t0"],
                        grid_type="verbatim",
                    )
                    dispatcher = solver.kernel.kernel
                    ((signature, specialization),) = (
                        dispatcher.overloads.items()
                    )
                    library = specialization._codelibrary
                    library.get_cufunc().set_shared_memory_carveout(
                        wrapper.read(wrapper.checked(case["compile"]))[
                            "carveout"
                        ]["integer_percent"]
                    )
                    cubin = (
                        bytes(library.get_cubin().code)
                        if hasattr(library, "get_cubin")
                        else bytes(library._cubin)
                    )
                    image_path = folder / "kernel.cubin"
                    image_path.write_bytes(cubin)
                    command = [
                        str(wrapper.checked(prepared["disassembler"])),
                        "-c",
                        str(image_path),
                    ]
                    process = subprocess.run(
                        command, capture_output=True, text=True
                    )
                    wrapper.write(
                        folder / "disassembly.json",
                        dict(
                            command=command,
                            returncode=process.returncode,
                            stdout=process.stdout,
                            stderr=process.stderr,
                        ),
                    )
                    process.check_returncode()
                    record["native"] = dict(
                        cubin=wrapper.asset(image_path),
                        compiler_kwargs=wrapper.compiler_record(
                            dict(solver.kernel.jit_kwargs)
                        ),
                        resources={
                            name: int(getattr(dispatcher, method)(signature))
                            for name, method in wrapper.RESOURCE_METHODS.items()
                        },
                    )
                first = None
                require(
                    protocol["n_runs"]
                    >= 2
                    * receipt["device"]["capacities"]["multiprocessor_count"]
                    * receipt["device"]["capacities"]["max_threads_per_sm"],
                    "Batch is below two hardware thread-capacity waves",
                )
                reference = wrapper.reference_arrays(case["reference"])
                for phase in ("warmup", "capture"):
                    result = solver.solve(
                        inits,
                        params,
                        duration=protocol["duration"],
                        t0=protocol["t0"],
                        grid_type="verbatim",
                        blocksize=case["candidate"]["geometry"][
                            "block_threads"
                        ],
                        nan_error_trajectories=False,
                    )
                    arrays = dict(
                        state=np.array(result.state[-1]),
                        status=np.array(result.status_codes),
                    )
                    array_path = folder / (phase + ".npz")
                    np.savez_compressed(array_path, **arrays)
                    geometry = landscape.launch_geometry(
                        solver, 128, protocol["n_runs"]
                    )
                    checks = dict(
                        fp32=arrays["state"].dtype == np.float32,
                        finite=bool(np.isfinite(arrays["state"]).all()),
                        success=bool(np.all(arrays["status"] == 0)),
                        counter_free=result.iteration_counters is None,
                        two_waves=geometry["waves"] >= 2,
                        array_schema=all(
                            arrays[k].shape == reference[k].shape
                            and arrays[k].dtype == reference[k].dtype
                            for k in arrays
                        ),
                    )
                    if mode == "original":
                        ref = wrapper.reference_arrays(case["reference"])
                        checks["own_original_exact"] = all(
                            arrays[k].shape == ref[k].shape
                            and arrays[k].dtype == ref[k].dtype
                            and arrays[k].tobytes() == ref[k].tobytes()
                            for k in arrays
                        )
                    content = wrapper.array_digest(arrays)
                    if first is None:
                        first = content
                    checks["repeat_exact"] = content == first
                    record["solves"].append(
                        dict(
                            phase=phase,
                            arrays=wrapper.asset(array_path),
                            content_sha256=content,
                            geometry=geometry,
                            checks=checks,
                        )
                    )
                    wrapper.write(output / "receipt.json", receipt)
                    require(
                        all(checks.values()), "Functional/identity gate failed"
                    )
                    saved[(mode, source)] = arrays
                    del result
                solver.close()
                solver = None
            full = saved[(mode, "0002")]["state"]
            rolled = saved[(mode, "0000")]["state"]
            bad = ~np.isclose(rolled, full, equal_nan=False, **tolerance)
            receipt["comparisons"].append(
                dict(
                    mode=mode,
                    bitwise_equal=full.tobytes() == rolled.tobytes(),
                    original_gate_pass=bool(not bad.any()),
                    failing_elements=int(bad.sum()),
                    failing_trajectories=int(np.any(bad, axis=0).sum()),
                    max_absolute_difference=float(
                        np.max(np.abs(full - rolled))
                    ),
                )
            )
        receipt["status"] = (
            "FUNCTIONAL_INTERVENTION_COMPLETE_NOT_POLICY_ADMISSION"
        )
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        wrapper.write(output / "receipt.json", receipt)
        if solver is not None:
            solver.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.prepared, arguments.wrapper, arguments.output),
            indent=2,
        )
    )
