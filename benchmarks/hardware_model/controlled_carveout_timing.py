"""Measure one fixed native image under interleaved function preferences."""

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import numpy as np
from cuda.bindings import driver
from cubie.cuda_simsafe import compile_kernel_specialization
from cubie.outputhandling.output_sizes import BatchOutputSizes
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import LTOIR
from numba_cuda_mlir.numba_cuda.core.options import FastMathOptions


ANCHOR = 16
PREFERENCES = (0, 8, 32, 64, 100)
BLOCK_SOLVES = 15
PAIRS = 4
MIN_COUNT = 5
IDLE_SECONDS = (1.5, 3.5)
RNG_SEED = 20260905
GATE_SOURCE = Path(
    "C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/benchmarks/ab_gate.py"
)


def block_schedule():
    """Return the exact predeclared ABBA ramp and ABBA BAAB design."""
    result = []
    random_source = random.Random(RNG_SEED)
    for preference in PREFERENCES:
        measured = []
        for pair in range(PAIRS):
            measured.extend(("A", "B") if pair % 4 in (0, 3) else ("B", "A"))
        for stage, order in (
            ("ramp", ("A", "B", "B", "A")),
            ("measured", measured),
        ):
            for index, side in enumerate(order):
                result.append(
                    dict(
                        comparison=preference,
                        stage=stage,
                        block_index=index,
                        side=side,
                        preference=ANCHOR if side == "A" else preference,
                        solves=BLOCK_SOLVES,
                        idle_seconds=random_source.uniform(*IDLE_SECONDS),
                    )
                )
    return result


def timing_summary(blocks):
    """Use the source gate's lowest-five means and paired median delta."""
    result = {}
    for preference in PREFERENCES:
        sides = {
            side: [
                entry["kernel_floor_ms"]
                for entry in blocks
                if entry["comparison"] == preference
                and entry["stage"] == "measured"
                and entry["side"] == side
            ]
            for side in ("A", "B")
        }
        if not all(len(values) == PAIRS for values in sides.values()):
            raise ValueError("Incomplete paired timing blocks")
        deltas = [
            100.0 * (b / a - 1.0) for a, b in zip(sides["A"], sides["B"])
        ]
        result[str(preference)] = dict(
            anchor_percent=ANCHOR,
            block_floors=sides,
            paired_percent_deltas=deltas,
            median_paired_percent_delta=statistics.median(deltas),
            anchor_mean_ms=statistics.fmean(sides["A"]),
            preference_mean_ms=statistics.fmean(sides["B"]),
        )
    return result


CASE = "workload_006_source_0000_b128_s102400"
CARRIER = {
    "path": "C:/local_working_projects/cubie-notes/hardware_unroll_placement/controlled_carveout_native_e3/20260905T125730_097891Z_pid79300_c4034f4b10844f9da8f4ee7088ba620c/diagnostic_cached_ltoir.bin",
    "sha256": "63f4a9375dd0176661656dd55354f9eff4be557ff8ca8236908350f108ad53c3",
}


def asset(path):
    """Bind exact bytes of a source or retained artifact."""
    path = Path(path).resolve()
    return dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def targetoptions_record(value):
    """Encode exact backend options without retaining custom objects."""
    if isinstance(value, FastMathOptions):
        return dict(type="FastMathOptions", flags=sorted(value.flags))
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {
            str(key): targetoptions_record(item) for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        return dict(
            type=type(value).__name__,
            members=sorted(targetoptions_record(x) for x in value),
        )
    if isinstance(value, (list, tuple)):
        return [targetoptions_record(x) for x in value]
    if is_dataclass(value):
        return dict(
            type=type(value).__name__,
            fields=targetoptions_record(asdict(value)),
        )
    raise TypeError(
        f"Unsupported recorded option type: {type(value).__name__}"
    )


def disabled_counter_contract(solver, prepared):
    """Require disabled source counters while retaining their ABI placeholder."""
    kernel = solver.kernel
    requested = prepared["protocol"]["diagnostic_counters"]
    flags = dict(
        prepared_diagnostic_counters=requested,
        source_save_counters=bool(kernel.compile_flags.save_counters),
        selected_counter_output=bool(kernel.active_outputs.iteration_counters),
        integrator_save_counters=bool(kernel.single_integrator.save_counters),
    )
    if requested is not False or any(flags.values()):
        raise ValueError(
            "Activated counter collection is not this frozen case"
        )
    sizes = BatchOutputSizes.from_solver(kernel)
    if tuple(sizes.iteration_counters) != (0, 0, 0):
        raise ValueError("Disabled source counter dimensions differ")
    host = kernel.iteration_counters
    device = kernel.device_iteration_counters
    if device.ndim != 3 or np.dtype(device.dtype) != np.dtype(np.int32):
        raise ValueError(
            "Counter ABI placeholder must remain int32 rank three"
        )
    return dict(
        disabled=True,
        flags=flags,
        source_dimensions=list(sizes.iteration_counters),
        allocated_source_dimensions=list(sizes.nonzero.iteration_counters),
        host_placeholder=None
        if host is None
        else dict(
            shape=list(host.shape),
            dtype=str(host.dtype),
            strides=list(host.strides),
        ),
        device_placeholder=dict(
            shape=list(device.shape),
            dtype=str(device.dtype),
            strides=list(device.strides),
        ),
        contract="Raw kernel buffer remains an ABI placeholder. SolveResult returns None from inactive output selection; source counter writes are disabled and native image identity is separately required.",
    )


def load_wrapper(path, expected):
    """Load the exact reviewed constructor and original-image gate."""
    if asset(path)["sha256"] != expected:
        raise ValueError("Reviewed wrapper source differs")
    spec = importlib.util.spec_from_file_location("frozen_profile", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(prepared_path, wrapper_path, percent, output_root):
    """Gate a separate dispatcher before warmup and one capture launch."""
    if percent != ANCHOR:
        raise ValueError("This timing design starts at anchor 16")
    output = Path(output_root) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + f"_pid{os.getpid()}_"
        + uuid.uuid4().hex
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="STARTED",
        source=asset(__file__),
        prepared=asset(prepared_path),
        wrapper=asset(wrapper_path),
        requested_percent=percent,
        case=CASE,
        driver_calls=[],
        callbacks=[],
        phases=[],
        launches=0,
        scope="Ordinary CUDA-event timing of one fixed native image; validation only, no predictor fitting",
        design=dict(
            anchor=ANCHOR,
            preferences=PREFERENCES,
            solves=BLOCK_SOLVES,
            pairs=PAIRS,
            minimum_count=MIN_COUNT,
            idle_seconds=IDLE_SECONDS,
            rng_seed=RNG_SEED,
            gate_source=asset(GATE_SOURCE),
        ),
        timing_blocks=[],
        schedule=block_schedule(),
    )
    solver = None
    try:
        raw = json.loads(Path(prepared_path).read_text())
        wrapper = load_wrapper(wrapper_path, raw["wrapper_sha256"])
        prepared = wrapper.load_prepared(prepared_path)
        case = prepared["cases"][CASE]
        request = wrapper.read(wrapper.checked(prepared["request"]))
        identity = wrapper.read(wrapper.checked(prepared["native_identity"]))
        receipt["device"] = wrapper.verify_device(identity)
        reference = wrapper.reference_arrays(case["reference"])
        receipt["reference"] = case["reference"]
        receipt["original_cross_candidate_passed"] = case[
            "original_cross_candidate_passed"
        ]
        kwargs, constants = wrapper.case_kwargs(
            request, case["workload"], case["candidate"]
        )
        if (
            wrapper.compiler_record(kwargs) != case["constructor_kwargs"]
            or wrapper.compiler_record(constants)
            != case["constructor_constants"]
        ):
            raise ValueError("Frozen public constructor differs")
        wrapper.set_cache_root(output / "codegen")
        solver = wrapper.Solver(
            wrapper.landscape.SYSTEMS[request["system"]]["build"](), **kwargs
        )
        if constants:
            solver.update(constants)
        solver.kernel.single_integrator.device_function
        if solver.kernel.kernel.overloads:
            raise ValueError("Unexpected early native specialization")
        if wrapper.landscape.bytes_per_run(solver) != 0:
            raise ValueError("RK23 diagnostic requires the frozen local case")
        with np.load(wrapper.checked(case["grid"]), allow_pickle=False) as d:
            inits, params = d["initial_values"], d["parameters"]
        if inits.dtype != np.float32 or params.dtype != np.float32:
            raise ValueError("Frozen inputs must remain FP32")
        original_dir = output / "original"
        original_dir.mkdir()
        compiled, receipt["original"] = wrapper.compile_exact(
            solver,
            case,
            prepared["protocol"],
            inits,
            params,
            prepared,
            original_dir,
        )
        kernel = solver.kernel
        receipt["disabled_counter_contract"] = disabled_counter_contract(
            solver, prepared
        )
        original = kernel.kernel
        ((original_signature, original_compiled),) = original.overloads.items()
        original_bytes = bytes(original_compiled._codelibrary._cubin)
        original_options = targetoptions_record(dict(original.targetoptions))
        effective = dict(kernel.jit_kwargs)
        if kernel.compile_settings.max_registers:
            effective["max_registers"] = kernel.compile_settings.max_registers
        if wrapper.compiler_record(effective) != compiled["compiler_kwargs"]:
            raise ValueError("Effective source compiler kwargs differ")
        if "link" in effective or effective.get("lto") is not True:
            raise ValueError("Expected explicit LTO without existing links")
        receipt["effective_original_kwargs"] = wrapper.compiler_record(
            effective
        )
        receipt["original_targetoptions"] = original_options
        receipt["signature"] = str(original_signature)
        receipt["py_func"] = dict(
            name=original.py_func.__name__,
            qualname=original.py_func.__qualname__,
            filename=original.py_func.__code__.co_filename,
            first_line=original.py_func.__code__.co_firstlineno,
        )
        geometry = wrapper.geometry(
            solver, case, compiled, prepared["protocol"]
        )
        actual = geometry[0]["actual"]
        runs = geometry[0]["runs"]
        block, shared = actual["blocksize"], actual["dynshared"]
        blocks = (runs + block - 1) // block
        sm_count = receipt["device"]["capacities"]["multiprocessor_count"]
        thread_limit = receipt["device"]["capacities"]["max_threads_per_sm"]
        if runs < 2 * sm_count * (thread_limit // block) * block:
            raise ValueError(
                "Grid lacks two waves at the thread capacity ceiling"
            )
        receipt["launch_geometry"] = dict(
            original=geometry,
            blocks=blocks,
            threads=(1, block),
            dynamic_shared=shared,
            hardware_thread_ceiling_two_waves=True,
        )
        resolved = {}

        def checked(name, *args):
            result = getattr(driver, name)(*args)
            receipt["driver_calls"].append(
                dict(
                    name=name,
                    args=[str(x) for x in args],
                    result=[str(x) for x in result],
                )
            )
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"{name}: {result}")
            return result[1] if len(result) == 2 else result[1:]

        preference = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT

        def attributes(function):
            return {
                name: int(
                    checked(
                        "cuFuncGetAttribute",
                        getattr(driver.CUfunction_attribute, name),
                        function,
                    )
                )
                for name in (
                    "CU_FUNC_ATTRIBUTE_NUM_REGS",
                    "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
                    "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
                    "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
                    "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
                )
            }

        def setup(module):
            if not resolved.get("prelaunch_gate") or receipt["callbacks"]:
                raise RuntimeError(
                    "Unexpected callback timing or multiplicity"
                )
            ckernel = checked(
                "cuLibraryGetKernel", module.handle, resolved["name"].encode()
            )
            function = checked("cuKernelGetFunction", ckernel)
            module_handle = checked("cuFuncGetModule", function)
            callback_module = checked("cuLibraryGetModule", module.handle)
            if int(module_handle) != int(callback_module):
                raise ValueError("Callback does not own actual function")
            before = attributes(function)
            expected = compiled["native_attributes"]
            for attr, key in (
                ("CU_FUNC_ATTRIBUTE_NUM_REGS", "regs"),
                ("CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES", "shared"),
                ("CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES", "local"),
                ("CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK", "maxthreads"),
            ):
                if before[attr] != expected[key]:
                    raise ValueError(
                        "Actual launch resource attributes differ"
                    )
            checked("cuFuncSetAttribute", function, preference, percent)
            after = attributes(function)
            if (
                after["CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT"]
                != percent
            ):
                raise ValueError("Actual function preference query differs")
            resident = int(
                checked(
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                    function,
                    block,
                    shared,
                )
            )
            if resident < 1 or blocks < 2 * sm_count * resident:
                raise ValueError("Actual function lacks two occupancy waves")
            resolved["function"] = function
            receipt["callbacks"].append(
                dict(
                    library=int(module.handle),
                    kernel=int(ckernel),
                    function=int(function),
                    module=int(module_handle),
                    callback_module=int(callback_module),
                    before=before,
                    after=after,
                    active_blocks_per_sm=resident,
                    two_waves=True,
                )
            )

        carrier_bytes = wrapper.checked(CARRIER).read_bytes()
        receipt["carrier_input"] = CARRIER
        (output / "linked_kernel.ltoir").write_bytes(carrier_bytes)
        link = LTOIR(
            carrier_bytes, name="carveout_kernel.ltoir", setup_callback=setup
        )
        diagnostic_kwargs = dict(effective, link=[link])
        diagnostic = wrapper.cuda.jit(**diagnostic_kwargs)(original.py_func)
        if diagnostic.py_func is not original.py_func:
            raise ValueError(
                "Diagnostic does not own the exact Python function"
            )
        stream = kernel.stream
        manager = kernel.memory_manager
        manager.begin_work(kernel)
        try:
            args = kernel._kernel_launch_args(kernel.run_params[0])
            compile_kernel_specialization(diagnostic, args)
            ((signature, specialization),) = diagnostic.overloads.items()
            if signature != original_signature:
                raise ValueError("Separate dispatcher ABI differs")
            library = specialization._codelibrary
            cubin = bytes(library._cubin)
            path = output / "diagnostic.cubin"
            path.write_bytes(cubin)
            resolved["name"] = library._func_name
            process = subprocess.run(
                [
                    str(wrapper.checked(prepared["disassembler"])),
                    "-c",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            wrapper.write(
                output / "disassembly.json",
                dict(
                    command=process.args,
                    returncode=process.returncode,
                    stdout=process.stdout,
                    stderr=process.stderr,
                ),
            )
            process.check_returncode()
            plan = specialization.metadata.get("link_plan")
            external = specialization.metadata.get(
                "linked_external_link_items"
            )
            if not plan.compile_new_inputs_as_ltoir or tuple(external) != (
                link,
            ):
                raise ValueError(
                    "Expected the exact single supplied LTOIR input"
                )
            fresh_ltoir = bytes(specialization.metadata["ltoir"])
            if fresh_ltoir != carrier_bytes:
                raise ValueError(
                    "Fresh kernel LTOIR differs from bound carrier"
                )
            base_linker = specialization.metadata["linker"]
            if base_linker._pending_cu:
                raise ValueError("Unexpected pending CUDA translation input")
            base_inputs = list(base_linker._object_codes)
            if len(base_inputs) != 1 or base_inputs[0].code_type != "ltoir":
                raise ValueError("Expected one prelinked kernel LTOIR object")
            if bytes(base_inputs[0].code) != fresh_ltoir:
                raise ValueError("Base linker object differs from kernel IR")
            reconstructed = base_linker.recreate_with_lto(
                kernel_ltoir=fresh_ltoir
            )
            final_inputs = list(reconstructed._object_codes)
            if len(final_inputs) != 1 or final_inputs[0].code_type != "ltoir":
                raise ValueError(
                    "Source final-link reconstruction is not singleton"
                )
            if bytes(final_inputs[0].code) != fresh_ltoir:
                raise ValueError("Source reconstructed final input differs")
            receipt["callback_carrier"] = dict(
                type=type(link).__name__,
                kind=str(link.kind),
                name=link.name,
                source_sha256=hashlib.sha256(link.data).hexdigest(),
                fresh_kernel_bytes_equal=True,
                base_ltoir_objects=1,
                reconstructed_final_ltoir_objects=1,
                proof="Installed mlir_optimization.py817 invokes this kernel-first reconstruction; linker.py159-225 deduplicates equal bytes through its hash map. Reconstructed input inventory, not an observation of the expired final-linker local variable.",
            )
            receipt["link_plans"] = {
                "original": targetoptions_record(
                    original_compiled.metadata.get("link_plan")
                ),
                "diagnostic": targetoptions_record(
                    specialization.metadata.get("link_plan")
                ),
            }
            for label, item in (
                ("original", original_compiled),
                ("diagnostic", specialization),
            ):
                blob = item.metadata.get("ltoir")
                if blob:
                    ir_path = output / (label + "_cached_ltoir.bin")
                    ir_path.write_bytes(bytes(blob))
                    receipt[label + "_ltoir"] = asset(ir_path)
            receipt["diagnostic_cubin"] = asset(path)
            wrapper.write(output / "receipt.json", receipt)
            comparison = wrapper.compare_cubins(
                original_bytes,
                cubin,
                prepared["cubin_equivalence"]["naming_source"]["sha256"],
            )
            resources = {
                name: int(getattr(diagnostic, method)(signature))
                for name, method in wrapper.RESOURCE_METHODS.items()
            }
            old_disassembly = wrapper.read(original_dir / "disassembly.json")[
                "stdout"
            ]
            checks = dict(
                image=comparison["admitted"],
                sass=process.stdout == old_disassembly,
                resources=resources == compiled["native_attributes"],
                original_dispatcher=kernel.kernel is original,
                original_cubin=bytes(original_compiled._codelibrary._cubin)
                == original_bytes,
                original_options=targetoptions_record(
                    dict(original.targetoptions)
                )
                == original_options,
                zero_prior_launches=receipt["launches"] == 0,
                no_prior_callback=len(receipt["callbacks"]) == 0,
            )
            receipt["diagnostic_gate"] = dict(
                cubin=asset(path),
                comparison=comparison,
                resources=resources,
                checks=checks,
                linked_ltoir=asset(output / "linked_kernel.ltoir"),
            )
            wrapper.write(output / "receipt.json", receipt)
            if not all(checks.values()):
                raise ValueError("Diagnostic prelaunch identity gate failed")
            resolved["prelaunch_gate"] = True
            snapshots = output / "snapshots"
            snapshots.mkdir()
            start_event = wrapper.cuda.event(timing=True)
            end_event = wrapper.cuda.event(timing=True)

            def launch_once(label, timed):
                kernel.input_arrays.initialise(0, stream=stream)
                kernel.output_arrays.initialise(0, stream=stream)
                stream.synchronize()
                launch_args = kernel._kernel_launch_args(kernel.run_params[0])
                launcher = diagnostic[blocks, (1, block), stream, shared]
                if timed:
                    start_event.record(stream)
                launcher(*launch_args)
                if timed:
                    end_event.record(stream)
                receipt["launches"] += 1
                milliseconds = None
                if timed:
                    end_event.synchronize()
                    milliseconds = float(
                        wrapper.cuda.event_elapsed_time(start_event, end_event)
                    )
                    if not math.isfinite(milliseconds) or milliseconds <= 0:
                        raise ValueError("Invalid CUDA event interval")
                kernel.input_arrays.finalise(0, stream=stream)
                kernel.output_arrays.finalise(0, stream=stream)
                stream.synchronize()
                kernel.output_arrays.wait_pending()
                arrays = dict(
                    state=np.array(kernel.state[-1]),
                    status=np.array(kernel.status_codes),
                )
                content = wrapper.array_digest(arrays)
                path = snapshots / (content + ".npz")
                if not path.exists():
                    np.savez_compressed(path, **arrays)
                checks = {
                    name: array.dtype == reference[name].dtype
                    and array.shape == reference[name].shape
                    and array.tobytes(order="C")
                    == reference[name].tobytes(order="C")
                    for name, array in arrays.items()
                }
                counter_contract = disabled_counter_contract(solver, prepared)
                checks["counter_free"] = counter_contract["disabled"]
                checks["callback_once"] = len(receipt["callbacks"]) == 1
                checks["untouched_original"] = kernel.kernel is original
                phase = dict(
                    phase=label,
                    timed=timed,
                    kernel_ms=milliseconds,
                    arrays=asset(path),
                    checks=checks,
                    content_sha256=content,
                    disabled_counter_contract=counter_contract,
                    actual_attributes=attributes(resolved["function"]),
                    original_geometry=wrapper.geometry(
                        solver, case, compiled, prepared["protocol"]
                    ),
                )
                receipt["phases"].append(phase)
                wrapper.write(output / "receipt.json", receipt)
                if not all(checks.values()):
                    raise ValueError("Own-candidate exact array check failed")
                return milliseconds

            launch_once("initial_warmup", False)
            fixed_function = int(resolved["function"])
            for serial, block_spec in enumerate(receipt["schedule"]):
                stream.synchronize()
                function = resolved["function"]
                if int(function) != fixed_function:
                    raise ValueError(
                        "Resolved actual function identity changed"
                    )
                previous = attributes(function)
                preference = block_spec["preference"]
                checked(
                    "cuFuncSetAttribute",
                    function,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    preference,
                )
                current = attributes(function)
                if (
                    current[
                        "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT"
                    ]
                    != preference
                ):
                    raise ValueError("Actual preference setter/query mismatch")
                for key in current:
                    if (
                        key
                        != "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT"
                        and current[key] != previous[key]
                    ):
                        raise ValueError("Actual native resources changed")
                resident = int(
                    checked(
                        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                        function,
                        block,
                        shared,
                    )
                )
                if resident < 1 or blocks < 2 * sm_count * resident:
                    raise ValueError(
                        "Changed function lacks two occupancy waves"
                    )
                block_receipt = dict(
                    block_spec,
                    serial=serial,
                    actual_function=fixed_function,
                    previous_attributes=previous,
                    current_attributes=current,
                    active_blocks_per_sm=resident,
                    two_waves=True,
                    kernel_ms=[],
                )
                receipt["timing_blocks"].append(block_receipt)
                for index in range(BLOCK_SOLVES):
                    elapsed = launch_once(
                        f"block{serial:03d}_solve{index:02d}", True
                    )
                    block_receipt["kernel_ms"].append(elapsed)
                block_receipt["kernel_floor_ms"] = statistics.fmean(
                    sorted(block_receipt["kernel_ms"])[:MIN_COUNT]
                )
                before_idle = time.perf_counter()
                time.sleep(block_spec["idle_seconds"])
                block_receipt["actual_idle_seconds"] = (
                    time.perf_counter() - before_idle
                )
                wrapper.write(output / "receipt.json", receipt)
            receipt["timing_summary"] = timing_summary(
                receipt["timing_blocks"]
            )
            if (
                bytes(original_compiled._codelibrary._cubin) != original_bytes
                or bytes(library._cubin) != cubin
            ):
                raise ValueError("Retained native image changed during timing")
            receipt["same_native_image_after_timing"] = True
        finally:
            stream.synchronize()
            kernel.output_arrays.wait_pending()
            manager.end_work(kernel, stream)
        receipt["status"] = (
            "ACTUAL_FUNCTION_ORDINARY_TIMING_AND_EXACT_OUTPUT_PASS"
        )
        receipt["physical_partition"] = (
            "Requires separate matching Nsight capture"
        )
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        if solver is not None:
            solver.close()
    return dict(output=str(output), **receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--percent", default=ANCHOR, type=int)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.prepared,
                arguments.wrapper,
                arguments.percent,
                arguments.output,
            ),
            indent=2,
        )
    )
