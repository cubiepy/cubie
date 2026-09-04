"""Explicit native worker for the separately prepared arithmetic probe."""

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
REQUEST = json.loads((HERE / "request.json").read_text())
spec = importlib.util.spec_from_file_location(
    "arithmetic_controller", HERE / "benchmark_source.py"
)
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)
control.validate_preparation(HERE)
helper = control.helpers(HERE / "latency_source.py")
sys.path.insert(0, REQUEST["research_root"])
# The isolated worker verifies retained source before native-module import.
import cubie  # noqa: E402
import cubie.cuda_simsafe as cuda_helpers  # noqa: E402
from cubie._utils import package_source_hash  # noqa: E402
from benchmarks import unroll_landscape  # noqa: E402
from benchmarks.hardware_model import hardware_probes as hardware  # noqa: E402


def normalized(value):
    """Normalize immutable flags before source/compiler comparisons."""

    def visit(item):
        if isinstance(item, (set, frozenset)):
            return sorted(
                (visit(x) for x in item),
                key=lambda x: json.dumps(x, sort_keys=True),
            )
        if isinstance(item, dict):
            return {key: visit(x) for key, x in item.items()}
        if isinstance(item, (tuple, list)):
            return [visit(x) for x in item]
        return item

    return json.loads(json.dumps(visit(value), default=hardware._json_default))


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=control.digest(path))


def identity(manifest):
    """Record actual imported source and effective compiler configuration."""
    if len(manifest["clocks"].get("devices", [])) != 1:
        raise ValueError("Need one identified GPU")
    return normalized(
        dict(
            actual_cubie_root=str(Path(cubie.__file__).resolve().parent),
            actual_cubie_source_hash=package_source_hash(),
            compiler=unroll_landscape.compiler_identity(),
            versions=dict(
                manifest["versions"], numba=importlib.metadata.version("numba")
            ),
            imported_sources={
                name: record(inspect.getfile(module))
                for name, module in (
                    ("hardware", hardware),
                    ("cuda_helpers", cuda_helpers),
                    ("unroll", unroll_landscape),
                    ("cubin", hardware._compiled_cubin),
                )
            },
            jit_kwargs=hardware.get_jit_kwargs(),
            device_name=manifest["device_name"],
            device_attributes=manifest["device_attributes"],
            compute_capability=manifest["compute_capability"],
            gpu_uuid=manifest["clocks"]["devices"][0]["uuid"],
            nvdisasm=manifest["nvdisasm"],
        )
    )


def native(kernel, compiled, function, geometry):
    """Check the same CUfunc, binary and queried residency per launch."""
    cubin, entry = hardware._compiled_cubin(compiled)
    resident = int(
        hardware.cuda.current_context().get_active_blocks_per_multiprocessor(
            function, geometry["block_size"], 0
        )
    )
    value = dict(
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        entry=entry,
        overloads=len(kernel.overloads),
        handle=str(function.handle),
        resident_blocks_per_sm=resident,
        geometry=geometry,
    )
    if value["overloads"] != 1 or resident != 1:
        raise ValueError("Native specialization/residency changed")
    return value


def main():
    """Compile once, then run only the explicitly requested experiment."""
    result = dict(
        status="validating",
        kernel_compilation=False,
        gpu_execution=False,
        samples=[],
        cleanup_errors=[],
    )
    try:
        if not hardware.IS_MLIR or hardware.CUDA_SIMULATION:
            raise ValueError("Requires the actual installed MLIR backend")
        if control.digest(hardware.__file__) != control.HARDWARE_SHA:
            raise ValueError("Frozen hardware source differs")
        nvdisasm = hardware._tool("nvdisasm", REQUEST["nvdisasm"])
        cuobjdump = hardware._tool("cuobjdump", REQUEST["cuobjdump"])
        manifest = normalized(
            hardware._manifest(SimpleNamespace(**REQUEST), nvdisasm)
        )
        control.write_json(HERE / "manifest.json", manifest)
        if control.plan(manifest) != REQUEST["hardware_plan"]:
            raise ValueError("Live hardware capacity differs")
        result["compilation_identity"] = identity(manifest)
        result["compilation_identity"]["binary_tools"] = dict(
            nvdisasm=record(nvdisasm), cuobjdump=record(cuobjdump)
        )
        module = control.load_module(HERE / "kernel.py", "arithmetic_kernel")
        kernel = hardware.cuda.jit(**hardware.get_jit_kwargs())(module.probe)
        result["kernel_compilation"] = True
        hardware.compile_kernel_specialization(
            kernel,
            (np.uint64(0),) * 4 + (np.uint32(1),) * 5,
        )
        (compiled,) = kernel.overloads.values()
        cubin, entry = hardware._compiled_cubin(compiled)
        (HERE / "kernel.cubin").write_bytes(cubin)
        ptx = kernel.inspect_asm()
        if isinstance(ptx, dict):
            ptx = "\n".join(ptx.values())
        (HERE / "kernel.ptx").write_text(ptx)
        target = control.OPERATIONS[REQUEST["operation"]][0]
        total = REQUEST["body_operations"]
        total += control.TRACE_STEPS if REQUEST["operation"] == "rcp" else 0
        retained_ptx_targets = len(re.findall(re.escape(target), ptx))
        command = hardware._command([nvdisasm, "-c", HERE / "kernel.cubin"])
        control.write_json(HERE / "disassembly_command.json", command)
        if command["returncode"]:
            raise ValueError(command["stderr"])
        (HERE / "kernel.sass").write_text(command["stdout"])
        parsed = hardware._parse_sass(command["stdout"], entry)
        elf = hardware._command(
            [cuobjdump, "--dump-elf", HERE / "kernel.cubin"]
        )
        control.write_json(HERE / "native_elf_command.json", elf)
        if elf["returncode"]:
            raise ValueError(elf["stderr"])
        (HERE / "native_elf.txt").write_text(elf["stdout"])
        if retained_ptx_targets != total:
            raise ValueError("Compiled PTX target inventory differs")
        abi = control.parameter_abi(elf["stdout"], entry)
        admission = control.check_native(
            *parsed,
            REQUEST["operation"],
            REQUEST["body_operations"],
            abi,
            helper,
        )
        result["native_admission"] = admission
        control.write_json(HERE / "sass_analysis.json", admission)
        compiled._ensure_kernel_attrs()
        resources = normalized(
            dict(
                entry=entry,
                cubin_sha256=hashlib.sha256(cubin).hexdigest(),
                registers=int(
                    next(iter(kernel.get_regs_per_thread().values()))
                ),
                local_bytes_per_thread=int(
                    next(iter(kernel.get_local_mem_per_thread().values()))
                ),
                static_shared_bytes=int(
                    next(iter(kernel.get_shared_mem_per_block().values()))
                ),
                jit_kwargs=hardware.get_jit_kwargs(),
            )
        )
        if (
            resources["local_bytes_per_thread"]
            or resources["static_shared_bytes"]
        ):
            raise ValueError("Arithmetic probe has memory-frame allocation")
        function = compiled._codelibrary.get_cufunc()
        device = hardware.cuda.get_current_device()
        context = hardware.cuda.current_context()
        occupancy_query = context.get_active_blocks_per_multiprocessor
        resident = int(occupancy_query(function, REQUEST["block_size"], 0))
        if resident != 1:
            raise ValueError("Need one maximal block per SM")
        geometry = dict(
            block_size=REQUEST["block_size"],
            grid_blocks=REQUEST["waves"] * int(device.MULTIPROCESSOR_COUNT),
            resident_blocks_per_sm=1,
            sms=int(device.MULTIPROCESSOR_COUNT),
            resident_warps_per_sm=32,
            timed_warp_choices=[1, 32],
            waves=REQUEST["waves"],
            dynamic_shared_bytes=0,
            thread_capacity_blocks=int(device.MAX_THREADS_PER_MULTIPROCESSOR)
            // REQUEST["block_size"],
            carveout_preference_set=False,
            actual_shared_capacity=None,
        )
        if geometry["waves"] < 2 or geometry["thread_capacity_blocks"] != 1:
            raise ValueError("Need two full occupancy waves")
        result.update(resources=resources, geometry=geometry)
        initial = native(kernel, compiled, function, geometry)
        result["initial_native"] = initial
        result["artifacts"] = {
            name: control.digest(HERE / name) for name in control.ARTIFACTS
        }
        if REQUEST["mode"] == "compile_only":
            result["final_native"] = initial
            result["status"] = "compile_only_complete"
            return
        prior = None
        if REQUEST["mode"] == "profile":
            prior, evidence = control.load_ordinary(REQUEST["ordinary_dir"])
            control.match_native_bank(prior, result, REQUEST)
            result["ordinary_evidence"] = evidence
        seeds = np.load(HERE / "seeds.npy", allow_pickle=False)
        seed_gpu = hardware.cupy.asarray(seeds)
        expected_gpu = hardware.cupy.empty(seeds.shape, dtype=np.uint32)
        shape = (geometry["grid_blocks"], geometry["block_size"])
        output = hardware.cupy.empty(shape + (8,), dtype=np.uint64)
        trace = hardware.cupy.empty(
            shape + (control.TRACE_STEPS + 1,), dtype=np.uint32
        )
        cycles = prior.get("reciprocal_cycles") if prior else None
        result["gpu_execution"] = True

        def launch(
            iterations,
            warps,
            phase,
            block=-1,
            index=-1,
            multiple=1,
            tracing=False,
        ):
            before_native = native(kernel, compiled, function, geometry)
            if before_native != initial:
                raise ValueError("Native identity changed before launch")
            expected = (
                seeds
                if tracing
                else control.expected_bits(
                    REQUEST["operation"],
                    seeds,
                    REQUEST["coefficients"],
                    iterations * REQUEST["body_operations"],
                    cycles,
                )
            )
            expected_gpu.set(expected)
            output.fill(np.uint64(2**64 - 1))
            trace.fill(np.uint32(2**32 - 1))
            before = hardware._clocks()
            event = hardware._timed_launch(
                kernel,
                geometry,
                (
                    np.uint64(output.data.ptr),
                    np.uint64(seed_gpu.data.ptr),
                    np.uint64(expected_gpu.data.ptr),
                    np.uint64(trace.data.ptr),
                    np.uint32(iterations),
                    np.uint32(warps),
                    np.uint32(control.TRACE_STEPS if tracing else 0),
                    np.uint32(REQUEST["coefficients"]["m"]),
                    np.uint32(REQUEST["coefficients"]["a"]),
                ),
            )
            after = hardware._clocks()
            after_native = native(kernel, compiled, function, geometry)
            if after_native != initial:
                raise ValueError("Native identity changed after launch")
            values, trace_values = map(hardware.cupy.asnumpy, (output, trace))
            if not (
                np.array_equal(hardware.cupy.asnumpy(seed_gpu), seeds)
                and np.array_equal(
                    hardware.cupy.asnumpy(expected_gpu), expected
                )
            ):
                raise ValueError("Runtime source operands changed")
            if tracing:
                checks = control.validate_trace(values, trace_values, seeds)
                chain_ms = None
            else:
                if not np.all(trace_values == np.uint32(2**32 - 1)):
                    raise ValueError(
                        "Ordinary launch entered functional trace"
                    )
                checks = control.validate_output(
                    values,
                    expected,
                    iterations,
                    warps,
                    geometry,
                    REQUEST["body_operations"],
                )
                chain_ms = helper.chain_milliseconds(
                    checks,
                    before,
                    after,
                    event,
                    result["compilation_identity"]["gpu_uuid"],
                )
            name = f"sample_{len(result['samples']):04d}.npz"
            np.savez_compressed(
                HERE / name,
                values=values,
                trace=trace_values,
                seeds=seeds,
                expected=expected,
            )
            row = normalized(
                dict(
                    phase=phase,
                    block=block,
                    index=index,
                    multiple=multiple,
                    iterations=iterations,
                    warps=warps,
                    event_ms=event,
                    minimum_chain_ms_at_observed_max_clock=chain_ms,
                    clocks_before=before,
                    clocks_after=after,
                    output_checks=checks,
                    array_file=name,
                    array_sha256=control.digest(HERE / name),
                    native_before=before_native,
                    native_after=after_native,
                )
            )
            result["samples"].append(row)
            with (HERE / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            return row

        if REQUEST["operation"] == "rcp":
            if prior is None:
                row = launch(0, 32, "functional_trace", tracing=True)
                cycles = row["output_checks"]["cycles"]
            result["reciprocal_cycles"] = cycles
        iterations = prior["iterations"] if prior else REQUEST["iterations"]
        if prior:
            result["iterations"] = iterations
            row = launch(
                iterations * REQUEST["profile_multiplier"],
                REQUEST["profile_warps"],
                "profile",
                multiple=REQUEST["profile_multiplier"],
            )
            result["status"] = "profile_complete_service_unassigned"
        else:
            for trial in range(25):
                while not control.valid_repeats(
                    REQUEST["operation"],
                    seeds,
                    REQUEST["coefficients"],
                    iterations,
                    REQUEST["body_operations"],
                    cycles,
                ):
                    iterations += 1
                    if iterations >= 2**30:
                        raise ValueError("Repeat counter overflow")
                rows = [
                    launch(iterations, warps, "calibration", index=trial)
                    for warps in (1, 32)
                ]
                if all(control.duration_ok(row) for row in rows):
                    break
                iterations = iterations * 2 + 1
            else:
                raise ValueError("Both populations failed duration gate")
            result["iterations"] = iterations
            for block, index, warps, multiple in control.sample_order():
                time.sleep(0.05)
                row = launch(
                    iterations * multiple,
                    warps,
                    "measurement",
                    block,
                    index,
                    multiple,
                )
                if not control.duration_ok(row):
                    raise ValueError("Ordinary sample fell below 20 ms")
            result["status"] = "ordinary_complete"
        result["final_native"] = native(kernel, compiled, function, geometry)
        if result["final_native"] != initial:
            raise ValueError("Final native identity differs")
    except Exception as error:
        result.update(
            status="failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["cleanup_scope"] = (
            "Isolated worker exit releases allocations/context; "
            "no device attributes changed"
        )
        control.write_json(HERE / "result.json", normalized(result))


if __name__ == "__main__":
    main()
