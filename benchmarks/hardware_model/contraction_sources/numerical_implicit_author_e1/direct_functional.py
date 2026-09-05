"""Execute retained arithmetic interventions through the CUDA Driver API."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from cuda.bindings import driver


ROOT = Path(__file__).parent


def require(condition, message):
    """Refuse an unsupported source, ABI or functional observation."""
    if not condition:
        raise ValueError(message)


def asset(path):
    """Hash exact retained bytes."""
    path = Path(path).resolve()
    return dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def checked(record):
    """Resolve a content-bound input."""
    path = Path(record["path"])
    require(asset(path) == record, "Changed input: " + str(path))
    return path


def load(record, name):
    """Import a verified source without replacing an existing module."""
    spec = importlib.util.spec_from_file_location(name, checked(record))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    """Read a retained JSON record."""
    return json.loads(Path(path).read_text())


def scalar_type(name):
    """Map only the actual FP32/int32/int64 scalar kernel ABI."""
    mapping = {
        "float32": ctypes.c_float,
        "int32": ctypes.c_int32,
        "int64": ctypes.c_int64,
    }
    require(name in mapping, "Unsupported scalar ABI: " + name)
    return mapping[name]


def flatten_values(records):
    """Construct exact MLIR scalar parameters from recorded source values."""
    storage = []
    layout = []
    cursor = 0

    def append(source, field, kind, value):
        nonlocal cursor
        ctype = ctypes.c_uint64 if kind == "pointer" else scalar_type(kind)
        size = ctypes.sizeof(ctype)
        alignment = ctypes.alignment(ctype)
        cursor = (cursor + alignment - 1) // alignment * alignment
        item = ctype(value)
        require(item.value == value, "Scalar conversion is not exact")
        layout.append(
            dict(
                index=len(storage),
                source_argument=source,
                field=field,
                kind=kind,
                offset=cursor,
                size=size,
                value=value,
                bytes=bytes(item).hex(),
            )
        )
        storage.append(item)
        cursor += size

    for index, record in enumerate(records):
        if record["kind"] == "array":
            append(index, "allocated_pointer", "pointer", record["pointer"])
            append(index, "aligned_pointer", "pointer", record["pointer"])
            append(index, "offset", "int64", 0)
            for axis, value in enumerate(record["shape"]):
                append(index, "shape_" + str(axis), "int64", value)
            for axis, value in enumerate(record["element_strides"]):
                append(index, "stride_" + str(axis), "int64", value)
        else:
            require(record["kind"] == "scalar", "Unsupported argument kind")
            append(index, "scalar", record["dtype"], record["value"])
    pointers = (ctypes.c_void_p * len(storage))(
        *(ctypes.addressof(item) for item in storage)
    )
    return storage, pointers, layout


def argument_records(args, signature):
    """Bind actual device-array metadata and scalar values to source types."""
    require(len(args) == len(signature) == 15, "Expected 15 source arguments")
    records = []
    expected_ranks = (2, 2, 3, 3, 3, 3, 3, 3, 1)
    for index, (arg, source_type) in enumerate(zip(args, signature)):
        if index < 9:
            dtype = np.dtype(arg.dtype)
            expected = np.dtype("float32" if index < 7 else "int32")
            shape = tuple(int(x) for x in arg.shape)
            byte_strides = tuple(int(x) for x in arg.strides)
            interface = arg.__cuda_array_interface__
            require(dtype == expected, "Changed array dtype")
            require(arg.ndim == expected_ranks[index], "Changed array rank")
            require(
                source_type.ndim == arg.ndim
                and str(source_type.dtype) == str(dtype),
                "Compiled array type differs from source argument",
            )
            require(
                tuple(interface["shape"]) == shape
                and np.dtype(interface["typestr"]) == dtype,
                "Array interface differs from owning array",
            )
            require(
                interface["strides"] is None
                or tuple(interface["strides"]) == byte_strides,
                "Array stride ownership differs",
            )
            require(
                all(x % dtype.itemsize == 0 for x in byte_strides),
                "Array has fractional element stride",
            )
            pointer = int(interface["data"][0])
            require(
                pointer > 0 and not interface["data"][1],
                "Expected writable device pointer",
            )
            records.append(
                dict(
                    kind="array",
                    dtype=str(dtype),
                    shape=list(shape),
                    byte_strides=list(byte_strides),
                    pointer=pointer,
                    element_strides=[
                        x // dtype.itemsize for x in byte_strides
                    ],
                    source_type=str(source_type),
                    interface_stream=interface.get("stream"),
                )
            )
        else:
            expected = (
                "float32",
                "float32",
                "float32",
                "int32",
                "int32",
                "int64",
            )[index - 9]
            require(str(source_type) == expected, "Changed scalar signature")
            value = float(arg) if expected == "float32" else int(arg)
            require(np.isfinite(value), "Nonfinite scalar argument")
            require(
                scalar_type(expected)(value).value == value,
                "Source scalar does not fit exact compiled type",
            )
            records.append(
                dict(
                    kind="scalar",
                    dtype=expected,
                    value=value,
                    python_type=type(arg).__name__,
                    source_type=str(source_type),
                )
            )
    return records


def compare_arrays(left, right, tolerance):
    """Retain the original cross-policy gate without changing tolerance."""
    require(
        left.shape == right.shape and left.dtype == right.dtype,
        "Comparison schema changed",
    )
    bad = ~np.isclose(right, left, equal_nan=False, **tolerance)
    return dict(
        bitwise_equal=left.tobytes() == right.tobytes(),
        original_gate_pass=bool(not bad.any()),
        failing_elements=int(bad.sum()),
        failing_trajectories=int(np.any(bad, axis=0).sum()),
        max_absolute_difference=float(np.max(np.abs(left - right))),
    )


def run(manifest_path, output):
    """Reproduce each baseline before running a separate arithmetic image."""
    manifest = read(manifest_path)
    for value in manifest["bindings"]:
        checked(value)
    require(asset(__file__) == manifest["runner"], "Runner epoch differs")
    wrapper_dir = str(Path(manifest["wrapper"]["path"]).parent)
    sys.path.insert(0, wrapper_dir)
    wrapper = load(manifest["wrapper"], "direct_frozen_wrapper")
    prior_author = load(manifest["prior_author"], "direct_prior_author")
    functional = load(manifest["functional"], "direct_counter_contract")
    prepared = wrapper.load_prepared(checked(manifest["prepared"]))
    request = read(wrapper.checked(prepared["request"]))
    capture = read(checked(manifest["capture"]))
    prior = capture
    require(
        capture["status"]
        == "IMPLICIT_CONTRACT_FALSE_OWN_REFERENCES_AND_IR_COMPLETE",
        "Own-reference capture is incomplete",
    )
    links = read(checked(manifest["links"]))
    frozen = read(wrapper.checked(prepared["original_manifest"]))
    tolerances = {
        x["id"]: x["numerical_tolerances"]
        for x in frozen["workloads"]
        if x["id"] in manifest["workloads"]
    }
    require(
        set(tolerances) == set(manifest["workloads"]),
        "Workload tolerance identity",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="STARTED",
        manifest=asset(manifest_path),
        source=asset(__file__),
        tolerances=tolerances,
        original_comparisons=capture["comparisons"],
        records=[],
        comparisons=[],
        driver_calls=[],
        launches=0,
        scope="Functional arithmetic intervention; no timings or "
        "policy admission. Original numerical failures retained.",
    )
    solver = None
    module = None
    saved = {}

    def call(name, *args):
        result = getattr(driver, name)(*args)
        receipt["driver_calls"].append(
            dict(
                name=name,
                args=[str(x) for x in args],
                result=[str(x) for x in result],
            )
        )
        require(
            result[0] == driver.CUresult.CUDA_SUCCESS,
            name + ": " + str(result),
        )
        return result[1] if len(result) == 2 else result[1:]

    try:
        identity = read(wrapper.checked(prepared["native_identity"]))
        receipt["device"] = wrapper.verify_device(identity)
        for source in manifest["cases"]:
            case_id = source
            case = prepared["cases"][case_id]
            retained = next(
                x for x in prior["records"] if x["case"] == case_id
            )
            source_record = next(
                x for x in capture["records"] if x["source"] == source
            )
            folder = output / ("source_" + source)
            folder.mkdir()
            kwargs, constants = wrapper.case_kwargs(
                request, case["workload"], case["candidate"]
            )
            require(
                wrapper.compiler_record(kwargs) == case["constructor_kwargs"],
                "Changed constructor",
            )
            wrapper.set_cache_root(folder / "codegen")
            solver = wrapper.Solver(
                wrapper.landscape.SYSTEMS[request["system"]]["build"](),
                **kwargs,
            )
            if constants:
                solver.update(constants)
            require(
                "contract" in solver.update(contract=False),
                "Public contract update failed",
            )
            solver.kernel.single_integrator.device_function
            factory_flags = [
                dict(
                    path=name,
                    kwargs=wrapper.compiler_record(dict(item.jit_kwargs)),
                )
                for name, item in prior_author.factories(solver.kernel)
            ]
            require(
                factory_flags
                == retained["factory_flags"]
                == source_record["factory_flags"],
                "Actual source factory flags differ",
            )
            with np.load(
                wrapper.checked(case["grid"]), allow_pickle=False
            ) as grid:
                inits = grid["initial_values"].copy()
                params = grid["parameters"].copy()
            require(
                inits.dtype == params.dtype == np.float32,
                "Changed FP32 source inputs",
            )
            protocol = prepared["protocol"]
            solver.compile(
                inits,
                params,
                duration=protocol["duration"],
                t0=protocol["t0"],
                grid_type="verbatim",
            )
            kernel = solver.kernel
            original_dispatcher = kernel.kernel
            ((signature, specialization),) = (
                original_dispatcher.overloads.items()
            )
            library = specialization._codelibrary
            original_bytes = bytes(library._cubin)
            fresh_path = folder / "source.cubin"
            fresh_path.write_bytes(original_bytes)
            equality = wrapper.compare_cubins(
                wrapper.checked(retained["native"]["cubin"]).read_bytes(),
                original_bytes,
                manifest["naming_provider"]["sha256"],
            )
            require(
                str(signature) == source_record["signature"],
                "Retained source signature changed",
            )
            require(
                bytes(specialization.metadata["ltoir"]),
                "Missing freshly compiled source IR",
            )
            # Native equality binds the constructor; saved IR separately binds
            # both offline images. Global symbol numbering can change fresh IR.
            record = dict(
                source=source,
                source_image=asset(fresh_path),
                source_native_equality=equality,
                source_signature=str(signature),
                factory_flags=factory_flags,
                source_ir_sha256=hashlib.sha256(
                    bytes(specialization.metadata["ltoir"])
                ).hexdigest(),
                images=[],
            )
            receipt["records"].append(record)
            reference_path = wrapper.checked(retained["solves"][-1]["arrays"])
            with np.load(reference_path, allow_pickle=False) as arrays:
                reference = {k: arrays[k].copy() for k in ("state", "status")}
            require(
                kernel.run_params.num_chunks == 1
                and int(kernel.single_integrator.threads_per_step) == 1,
                "Unsupported source launch geometry",
            )
            block = case["candidate"]["geometry"]["block_threads"]
            runs = int(kernel.run_params[0].runs)
            geometry = wrapper.landscape.launch_geometry(solver, block, runs)
            require(
                geometry["blocksize"] == block
                and runs == protocol["n_runs"]
                and geometry["dynshared"]
                == max(4, wrapper.landscape.bytes_per_run(solver) * block),
                "Source geometry differs",
            )
            shared = geometry["dynshared"]
            blocks = (runs + block - 1) // block
            stream = kernel.stream
            stream_handle = driver.CUstream(int(stream.handle))
            manager = kernel.memory_manager
            manager.begin_work(kernel)
            try:
                for fma in (True, False):
                    linked = next(
                        x
                        for x in links["records"]
                        if x["source"] == source and x["fma"] == fma
                    )
                    require(
                        linked["ir"] == source_record["cached_ltoir"],
                        "Offline image IR differs from source capture",
                    )
                    require(
                        linked["options"]["fma"] is fma,
                        "Link arithmetic option differs",
                    )
                    require(
                        linked["native_ffma_sites"] == 0
                        if not fma
                        else linked["exact_original_bytes"],
                        "Expected baseline or zero-FFMA intervention",
                    )
                    binary = wrapper.checked(linked["cubin"]).read_bytes()
                    if fma:
                        require(
                            binary
                            == wrapper.checked(
                                source_record["cubin"]
                            ).read_bytes(),
                            "Baseline image differs from source capture",
                        )
                    image_dir = folder / ("fma_" + str(fma))
                    image_dir.mkdir()
                    image_path = image_dir / "kernel.cubin"
                    image_path.write_bytes(binary)
                    process = subprocess.run(
                        [
                            str(wrapper.checked(prepared["disassembler"])),
                            "-c",
                            str(image_path),
                        ],
                        capture_output=True,
                        check=True,
                    )
                    (image_dir / "kernel.sass").write_bytes(process.stdout)
                    image_buffer = ctypes.create_string_buffer(binary)
                    module = call(
                        "cuModuleLoadData", ctypes.addressof(image_buffer)
                    )
                    function = call(
                        "cuModuleGetFunction",
                        module,
                        library._func_name.encode(),
                    )
                    require(
                        int(call("cuFuncGetModule", function)) == int(module),
                        "Direct function module ownership differs",
                    )
                    attributes = {
                        name: int(call("cuFuncGetAttribute", attr, function))
                        for name, attr in (
                            (
                                "regs",
                                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            ),
                            (
                                "local",
                                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            ),
                            (
                                "shared",
                                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            ),
                            (
                                "const",
                                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                            ),
                            (
                                "maxthreads",
                                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            ),
                        )
                    }
                    if fma:
                        require(
                            all(
                                attributes[k] == v
                                for k, v in retained["native"][
                                    "resources"
                                ].items()
                                if k in attributes
                            ),
                            "Baseline resource mismatch",
                        )
                    resident = int(
                        call(
                            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                            function,
                            block,
                            shared,
                        )
                    )
                    sm_count = receipt["device"]["capacities"][
                        "multiprocessor_count"
                    ]
                    require(
                        resident > 0 and blocks >= 2 * sm_count * resident,
                        "Direct image lacks two full occupancy waves",
                    )
                    image_record = dict(
                        fma=fma,
                        cubin=asset(image_path),
                        link=linked,
                        resources=attributes,
                        module=int(module),
                        function=int(function),
                        resident_blocks_per_sm=resident,
                        geometry=dict(
                            blocks=blocks,
                            block=[1, block, 1],
                            dynamic_shared=shared,
                        ),
                        phases=[],
                    )
                    record["images"].append(image_record)
                    first = None
                    for phase in ("warmup", "capture"):
                        kernel.input_arrays.initialise(0, stream=stream)
                        kernel.output_arrays.initialise(0, stream=stream)
                        args = kernel._kernel_launch_args(kernel.run_params[0])
                        argument_data = argument_records(args, signature)
                        storage, pointers, layout = flatten_values(
                            argument_data
                        )
                        count = int(call("cuFuncGetParamCount", function))
                        require(
                            count == len(layout),
                            "Native parameter count differs",
                        )
                        native_layout = [
                            list(call("cuFuncGetParamInfo", function, i))
                            for i in range(count)
                        ]
                        require(
                            native_layout
                            == [[x["offset"], x["size"]] for x in layout],
                            "Native flattened parameter layout differs",
                        )
                        require(
                            kernel.kernel is original_dispatcher
                            and bytes(library._cubin) == original_bytes,
                            "Original dispatcher changed",
                        )
                        counters = functional.disabled_counter_contract(
                            solver, prepared
                        )
                        call(
                            "cuLaunchKernel",
                            function,
                            blocks,
                            1,
                            1,
                            1,
                            block,
                            1,
                            shared,
                            stream_handle,
                            ctypes.addressof(pointers),
                            0,
                        )
                        receipt["launches"] += 1
                        kernel.input_arrays.finalise(0, stream=stream)
                        kernel.output_arrays.finalise(0, stream=stream)
                        stream.synchronize()
                        kernel.output_arrays.wait_pending()
                        arrays = dict(
                            state=np.array(kernel.state[-1]),
                            status=np.array(kernel.status_codes),
                        )
                        array_path = image_dir / (phase + ".npz")
                        np.savez_compressed(array_path, **arrays)
                        content = wrapper.array_digest(arrays)
                        if first is None:
                            first = content
                        checks = dict(
                            source_array_schema=all(
                                arrays[k].shape == reference[k].shape
                                and arrays[k].dtype == reference[k].dtype
                                for k in arrays
                            ),
                            finite_fp32=arrays["state"].dtype == np.float32
                            and bool(np.isfinite(arrays["state"]).all()),
                            success=bool(np.all(arrays["status"] == 0)),
                            exact_repeat=content == first,
                            counter_free=counters["disabled"],
                        )
                        if fma:
                            checks["baseline_own_output_exact"] = all(
                                arrays[k].tobytes() == reference[k].tobytes()
                                for k in arrays
                            )
                        image_record["phases"].append(
                            dict(
                                phase=phase,
                                arguments=argument_data,
                                flattened=layout,
                                native_layout=native_layout,
                                arrays=asset(array_path),
                                content_sha256=content,
                                checks=checks,
                                disabled_counter_contract=counters,
                            )
                        )
                        wrapper.write(output / "receipt.json", receipt)
                        require(all(checks.values()), "Functional gate failed")
                        saved[(source, fma)] = arrays
                        # The owning arrays, scalar storage and pointer table stay
                        # live through synchronization and output retention.
                        del args, storage, pointers
                    call("cuModuleUnload", module)
                    module = None
            finally:
                stream.synchronize()
                kernel.output_arrays.wait_pending()
                if module is not None:
                    call("cuModuleUnload", module)
                    module = None
                manager.end_work(kernel, stream)
            solver.close()
            solver = None
        for workload in manifest["workloads"]:
            keys = [x for x in manifest["cases"] if x.startswith(workload)]
            full = next(x for x in keys if "source_0002_" in x)
            rolled = next(x for x in keys if "source_0000_" in x)
            for fma in (True, False):
                receipt["comparisons"].append(
                    dict(
                        workload=workload,
                        final_fma=fma,
                        tolerance=tolerances[workload],
                        **compare_arrays(
                            saved[(full, fma)]["state"],
                            saved[(rolled, fma)]["state"],
                            tolerances[workload],
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
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        if solver is not None:
            solver.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    result = run(options.manifest, options.output)
    print(
        json.dumps(dict(status=result["status"], launches=result["launches"]))
    )
