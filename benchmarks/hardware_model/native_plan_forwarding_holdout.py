"""Freeze ERK predictions and observe fresh public-kernel compilations.

Preparation and source checks are CPU-only. The explicit compile command
starts a separate worker, compiles once and records native artifacts; it
never solves or pins a launch. Predictions precede every native label.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.hardware_model import native_plan as model
from benchmarks.hardware_model import native_plan_forwarding as forwarding


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
CASE_NAMES = tuple(
    f"chain{size}_{location}"
    for size in (21, 22)
    for location in ("local", "shared")
)


def canonical(value):
    """Return finite JSON-compatible closure/configuration values."""
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    return value


def identity(value):
    """Hash a canonical finite JSON value."""
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    """Bind an existing artifact's exact bytes and absolute path."""
    path = Path(path).resolve()
    return dict(path=str(path), sha256=model.digest(path))


def read_record(record):
    """Read a JSON artifact only after checking its retained byte hash."""
    if model.digest(record["path"]) != record["sha256"]:
        raise ValueError(f"Artifact bytes changed: {record['path']}")
    return json.loads(Path(record["path"]).read_text())


def adapter_sources():
    """Bind installed MLIR extraction interfaces without importing CUDA."""
    specification = importlib.util.find_spec("numba_cuda_mlir")
    if specification is None or specification.origin is None:
        raise ValueError("The holdout requires the installed MLIR backend")
    package = Path(specification.origin).resolve().parent
    return {
        name: file_record(package / (name + ".py"))
        for name in ("compiler", "descriptor")
    }


def compiled_payload(kernel, dispatcher, expected_sources):
    """Join natural compiled metadata to the same live MLIR library.

    No compilation, linking or module loading occurs here. The returned
    library supplies the existing function-handle interface in the
    explicit native worker.
    """
    library = kernel._codelibrary
    objects = {
        "result": (kernel, "compiler"),
        "library": (library, "compiler"),
        "dispatcher": (dispatcher, "descriptor"),
    }
    classes = {}
    for name, (value, source_name) in objects.items():
        source = file_record(inspect.getsourcefile(type(value)))
        if source != expected_sources[source_name]:
            raise ValueError("Unexpected installed native interface: " + name)
        classes[name] = type(value).__module__ + "." + type(value).__qualname__
    metadata = kernel.cres.metadata
    cubin = metadata["cubin"]
    entry = metadata["func_name"]
    if (
        not isinstance(cubin, bytes)
        or not cubin.startswith(b"\x7fELF")
        or not isinstance(entry, str)
        or not entry
        or cubin != library._cubin
        or entry != library._func_name
        or not callable(library.get_cufunc)
    ):
        raise ValueError("Compiled metadata and live library disagree")
    return cubin, entry, library, dict(
        classes=classes,
        installed_sources=expected_sources,
        extraction="cres.metadata cubin/func_name == live library payload",
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        entry_name=entry,
        function_handle_source="same live library.get_cufunc()",
        native_relink=False,
    )


def graph_identity(graph):
    """Compare complete graphs across caches by exact source-file bytes.

    Only the NativePlan construction envelope and source file locations
    differ between isolated caches. File identity, definition lines and
    all graph/closure/value/control/allocation data remain in this hash.
    """

    source_files = {}

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key == "native_plan_construction":
                continue
            if key in ("source_path", "path"):
                if item not in source_files:
                    if not isinstance(item, str) or not Path(item).is_file():
                        raise ValueError("Graph source location is not a file")
                    source_files[item] = model.digest(item)
                result[key] = dict(source_file_sha256=source_files[item])
            else:
                result[key] = visit(item)
        return result

    return identity(visit(graph))


def validate_request(path, expected_sha):
    """Admit only the exact prepared request and its immutable inputs."""
    request = read_record(dict(path=str(path), sha256=expected_sha))
    if request["schema"] != 1 or request["cases_order"] != list(CASE_NAMES):
        raise ValueError("Unexpected holdout cohort")
    if request["native_adapter_sources"] != adapter_sources():
        raise ValueError("Installed MLIR extraction interfaces changed")
    for record in request["observers"]:
        if model.digest(record["path"]) != record["sha256"]:
            raise ValueError("Observer source changed")
    if request["observer_sha256"] != model.digest(SCRIPT):
        raise ValueError("Worker imported a different holdout observer")
    if request["forwarding_sha256"] != model.digest(forwarding.SCRIPT):
        raise ValueError("Forwarding estimator source changed")
    if request["model_sha256"] != model.digest(model.SCRIPT):
        raise ValueError("Estimator source differs from frozen predictions")
    if model.digest(request["worker"]["path"]) != request["worker"]["sha256"]:
        raise ValueError("Worker source changed")
    if set(request["cases"]) != set(CASE_NAMES):
        raise ValueError("Holdout case membership changed")
    hardware = model.hardware_model(read_record(request["hardware_manifest"]))
    if hardware != request["hardware"]:
        raise ValueError("Hardware inputs differ from their source query")
    validate_selection(request)
    if request["n_runs"] != (
        2
        * hardware["MULTIPROCESSOR_COUNT"]
        * hardware["MAX_THREADS_PER_MULTIPROCESSOR"]
    ):
        raise ValueError(
            "Batch size differs from two full thread-capacity waves"
        )
    if request["requested_block"] != 64 or request["duration"] != 1.0:
        raise ValueError("Compile input protocol changed")
    origin = request.get("original_prediction_request")
    earlier = read_record(origin) if origin is not None else None
    if earlier is not None and (
        earlier["forwarding_sha256"] != request["forwarding_sha256"]
        or earlier["model_sha256"] != request["model_sha256"]
        or earlier["hardware"] != hardware
        or earlier["cases"] != request["cases"]
    ):
        raise ValueError("Reused prediction identities changed")
    for name, case in request["cases"].items():
        graph = read_record(case["graph"])
        model.validate_graph(graph)
        construction = model.validate_construction(graph)
        if (
            construction != case["construction"]
            or graph_identity(graph) != case["graph_identity"]
            or name != construction["system"] + "_" + construction["placement"]
            or construction["algo"] != "vern7"
        ):
            raise ValueError("Original generated workload identity changed")
        schedules = (
            ("early", "late")
            if construction["placement"] == "shared"
            else (None,)
        )
        estimator_sha = (
            request["forwarding_sha256"]
            if construction["placement"] == "shared"
            else request["model_sha256"]
        )
        expected = {
            (block, mode, contract, schedule)
            for block in (32, 64)
            for mode in ("promote", "addressable")
            for contract in (False, True)
            for schedule in schedules
        }
        actual = set()
        for prediction in case["predictions"]:
            plan = read_record(prediction)
            candidate = plan["candidate"]
            if model.hardware_model(read_record(
                plan["provenance"]["hardware_manifest"]
            )) != hardware:
                raise ValueError("Forecast hardware query differs")
            actual.add(
                (
                    candidate["block_size"],
                    candidate["materialization"],
                    candidate["contraction"],
                    candidate.get("shared_final_store_schedule"),
                )
            )
            if (
                plan["provenance"]["model_source_sha256"]
                != estimator_sha
                or plan["provenance"]["construction"] != construction
                or plan["hardware"] != hardware
                or candidate["actual_placement"] != construction["placement"]
                or plan["provenance"]["input_graph"] != case["graph"]
            ):
                raise ValueError("Frozen prediction identity changed")
        if actual != expected or len(case["predictions"]) != len(expected):
            raise ValueError(
                "Incomplete pre-native geometry/scenario coverage"
            )
    return request


def validate_selection(request):
    """Join the source-only dimension decision to actual generated sizes."""
    selection = read_record(request["source_selection"])
    receipt = read_record(request["source_construction_receipt"])
    hardware = request["hardware"]
    if (
        selection["status"]
        != "SOURCE_SELECTED_BEFORE_CONSTRUCTION_OR_NATIVE_LABELS"
        or selection["systems"] != ["chain21", "chain22"]
        or selection["placements"] != ["local", "shared"]
        or selection["constructor_sha256"] != request["model_sha256"]
        or selection["block_size"] != 32
        or selection["algorithm"] != "vern7"
        or selection["shared_allocation_quantum"]
        != hardware["shared_unit_bytes"]
        or selection["reserved_per_block"]
        != hardware["RESERVED_SHARED_MEMORY_PER_BLOCK"]
        or selection["shared_capacity_per_sm"]
        != hardware["MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"]
        or receipt["status"] != "ROOT_FRESH_SOURCE_CONSTRUCTIONS_PASS"
        or receipt["selection_sha256"]
        != request["source_selection"]["sha256"]
        or receipt["hardware_manifest_sha256"]
        != request["hardware_manifest"]["sha256"]
        or receipt["native_compilations"] != 0
        or receipt["kernel_launches"] != 0
    ):
        raise ValueError("Source-only transition selection differs")
    rows = {row["case"]: row for row in receipt["cases"]}
    if set(rows) != set(CASE_NAMES) or len(rows) != len(receipt["cases"]):
        raise ValueError("Source construction receipt has different cases")
    for name, case in request["cases"].items():
        row = rows[name]
        if (
            row["graph_path"] != case["graph"]["path"]
            or row["graph_sha256"] != case["graph"]["sha256"]
            or row["construction"] != case["construction"]
            or row["native_overloads"] != 0
            or row["selection_involves_native_label"] is not False
        ):
            raise ValueError("Source-only construction identity differs")
    observed = []
    for system in selection["systems"]:
        construction = request["cases"][system + "_shared"]["construction"]
        stride = construction["shared_stride_bytes"]
        geometry = model.residency(hardware, 255, 32, stride * 32)
        observed.append((stride, geometry["shared_allocated_bytes"],
                         geometry["resident_blocks"]))
    if (
        [item[0] for item in observed]
        != selection["hypothesized_stride_bytes"]
        or [item[1] for item in observed]
        != selection["hypothesized_allocation_bytes"]
        or [4 * item[1] for item in observed]
        != selection["four_blocks_bytes"]
        or [item[2] for item in observed] != [4, 3]
    ):
        raise ValueError("Actual source sizes do not prove the selected cut")


def validate_source(record, case, request_sha):
    """Verify the completed CPU source result and its retained files."""
    source = read_record(record)
    graph = read_record(source["graph"])
    model.validate_graph(graph)
    if (
        source["status"] != "COMPLETE_SOURCE_IDENTITY_PASS"
        or source["request_sha256"] != request_sha
        or source["native_overloads"] != 0
        or source["graph_identity"] != case["graph_identity"]
        or graph_identity(graph) != case["graph_identity"]
    ):
        raise ValueError("Prepared source identity differs")
    for artifact in [source["inputs"]] + source["source_snapshots"]:
        if model.digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Prepared source artifact changed")
    with model.np.load(source["inputs"]["path"]) as data:
        observed = [
            dict(
                shape=list(data[key].shape),
                dtype=str(data[key].dtype),
                sha256=hashlib.sha256(data[key].tobytes()).hexdigest(),
            )
            for key in ("inits", "params")
        ]
    if observed != source["input_arrays"]:
        raise ValueError("Prepared input arrays differ from their receipt")
    return source


def run_worker(request_path, case, output, command, source=None):
    """Run one isolated worker and retain exact command and all output."""
    output.mkdir(parents=True, exist_ok=False)
    arguments = [
        sys.executable,
        str(request_path.parent / "worker.py"),
        str(request_path.resolve()),
        model.digest(request_path),
        case,
        str(output.resolve()),
        command,
    ]
    if source is not None:
        arguments.extend([source["path"], source["sha256"]])
    result = subprocess.run(
        arguments,
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    model.write_json(
        output / "worker.json",
        dict(
            command=arguments,
            cwd=str(REPO),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ),
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return file_record(output / "source.json")


def prepare(args):
    """Freeze both public geometry alternatives before any native work."""
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (SCRIPT, model.SCRIPT, forwarding.SCRIPT):
        (args.output / path.name).write_bytes(path.read_bytes())
    worker = args.output / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    hardware = model.hardware_model(json.loads(args.hardware.read_text()))
    origin = None
    if args.reuse_predictions is not None:
        if args.reuse_predictions_sha is None:
            raise ValueError("Reusing predictions requires their request SHA")
        origin = dict(
            path=str(args.reuse_predictions.resolve()),
            sha256=args.reuse_predictions_sha,
        )
        earlier = read_record(origin)
        if (
            earlier["forwarding_sha256"] != model.digest(forwarding.SCRIPT)
            or earlier["model_sha256"] != model.digest(model.SCRIPT)
            or earlier["hardware"] != hardware
        ):
            raise ValueError("Earlier prediction model/hardware differs")
    cases = {}
    for name in CASE_NAMES:
        graph_path = args.graphs / name / "graph.json"
        graph = json.loads(graph_path.read_text())
        model.validate_graph(graph)
        construction = model.validate_construction(graph)
        directory = args.output / "predictions" / name
        directory.mkdir(parents=True, exist_ok=False)
        predictions = []
        if origin is not None:
            old = earlier["cases"][name]
            if (
                old["construction"] != construction
                or old["graph_identity"] != graph_identity(graph)
                or old["graph"] != file_record(graph_path)
            ):
                raise ValueError("Earlier prediction source differs")
            cases[name] = old
            print(
                json.dumps(dict(reused=name, predictions=len(
                    old["predictions"]
                ))), flush=True,
            )
            continue
        schedules = (
            ("early", "late")
            if construction["placement"] == "shared"
            else (None,)
        )
        for block in (32, 64):
            for mode in ("promote", "addressable"):
                for contract in (False, True):
                    for schedule in schedules:
                        if schedule is None:
                            plan = model.predict(
                                graph, hardware, mode, contract, block=block
                            )
                        else:
                            plan = forwarding.predict(
                                graph, hardware, mode, contract, block=block,
                                store_schedule=schedule,
                            )
                        plan["provenance"]["input_graph"] = file_record(
                            graph_path
                        )
                        plan["provenance"]["hardware_manifest"] = file_record(
                            args.hardware
                        )
                        label = f"b{block}_{mode}_{contract}_{schedule}.json"
                        path = directory / label
                        model.write_json(path, plan)
                        predictions.append(file_record(path))
        cases[name] = dict(
            graph=file_record(graph_path),
            graph_identity=graph_identity(graph),
            construction=construction,
            predictions=predictions,
        )
        print(
            json.dumps(dict(frozen=name, predictions=len(predictions))),
            flush=True,
        )
    observers = [
        SCRIPT,
        model.SCRIPT,
        forwarding.SCRIPT,
        REPO / "benchmarks/placement_landscape.py",
        REPO / "benchmarks/hardware_model/operation_translation.py",
    ]
    request = dict(
        schema=1,
        kind="pre_native_erk_forwarding_holdout",
        cases=cases,
        cases_order=list(CASE_NAMES),
        hardware=hardware,
        hardware_manifest=file_record(args.hardware),
        source_selection=file_record(args.graphs / "selection.json"),
        source_construction_receipt=file_record(
            args.graphs / "source_receipt.json"
        ),
        observer_sha256=model.digest(SCRIPT),
        model_sha256=model.digest(model.SCRIPT),
        forwarding_sha256=model.digest(forwarding.SCRIPT),
        observers=[file_record(path) for path in observers],
        native_adapter_sources=adapter_sources(),
        worker=file_record(worker),
        requested_block=64,
        duration=1.0,
        n_runs=2
        * hardware["MULTIPROCESSOR_COUNT"]
        * hardware["MAX_THREADS_PER_MULTIPROCESSOR"],
        native_labels_read=False,
        kernel_launches_requested=0,
        nvdisasm=file_record(args.nvdisasm),
        original_prediction_request=origin,
    )
    request_path = args.output / "request.json"
    model.write_json(request_path, request)
    validate_request(request_path, model.digest(request_path))
    sources = {}
    for name in CASE_NAMES:
        sources[name] = run_worker(
            request_path,
            name,
            args.output / "source" / name,
            "source",
        )
        validate_source(sources[name], cases[name], model.digest(request_path))
        print(json.dumps(dict(source_gate=name, status="PASS")), flush=True)
    manifest = dict(
        schema=1,
        status="PRE_NATIVE_PREDICTIONS_AND_SOURCE_PASS",
        request=file_record(request_path),
        sources=sources,
        worker_receipts={
            name: file_record(args.output / "source" / name / "worker.json")
            for name in CASE_NAMES
        },
        native_compilations=0,
        kernel_launches_requested=0,
    )
    model.write_json(args.output / "manifest.json", manifest)
    print(json.dumps(file_record(args.output / "manifest.json")), flush=True)


WORKER = r"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from cubie import Solver
from cubie._utils import package_source_hash
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cubie_cache import toolchain_fingerprint
from cubie.cuda_simsafe import cuda
from benchmarks import placement_landscape as placement
from benchmarks.hardware_model import native_plan as model
from benchmarks.hardware_model import (
    native_plan_forwarding_holdout as observer)
from benchmarks.hardware_model.source_value_graph import describe_source_values
from benchmarks.hardware_model.operation_translation import inspect_sass

request_path, request_sha, name, output_path, command = sys.argv[1:6]
request = observer.validate_request(Path(request_path), request_sha)
output = Path(output_path)
case = request['cases'][name]
expected = case['construction']
cache = output/'codegen'
cache.mkdir(exist_ok=False)
previous = get_cache_root_override()
solver = None
def array_record(array):
    array = np.asarray(array)
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())
try:
    set_cache_root(cache)
    system = placement.build_chain(expected['actual_system_size'],3)
    kwargs = placement.solver_kwargs('chain32',expected['algo'])
    kwargs['stage_accumulator_location'] = expected['placement']
    solver = Solver(system,**kwargs)
    kernel_cache=output/'kernel_cache'
    kernel_cache.mkdir(exist_ok=False)
    solver.kernel.set_cache_dir(kernel_cache)
    cache_settings=solver.kernel.compile_settings.cache
    cache_policy=dict(enabled=bool(cache_settings.cache_enabled),
                      mode=cache_settings.cache_mode,
                      max_entries=int(cache_settings.max_cache_entries))
    graph = describe_source_values(solver,max_states=1)
    source_graph = output/'graph.json'
    model.write_json(source_graph,graph)
    observed_sources=model.validate_graph(graph)
    observed_sources[expected['shared_stride_source']['path']]=(
        expected['shared_stride_source']['sha256'])
    snapshots=output/'source_snapshots'
    snapshots.mkdir(exist_ok=False)
    source_snapshots=[]
    for index,(path,sha) in enumerate(sorted(observed_sources.items())):
        original=Path(path)
        if model.digest(original)!=sha:
            raise ValueError('Observed source changed before snapshot')
        target=snapshots/(str(index)+'_'+original.name)
        target.write_bytes(original.read_bytes())
        source_snapshots.append(dict(original_path=path,
                                    **observer.file_record(target)))
    metadata = dict(
        actual_system_size=int(graph['workload']['n_states']),
        shared_stride_bytes=4*(int(solver.kernel.shared_memory_elements)+
                              int(solver.kernel.shared_memory_needs_padding)),
        shared_elements_per_run=int(solver.kernel.shared_memory_elements),
        shared_padding_elements=int(solver.kernel.shared_memory_needs_padding),
        jit_kwargs=observer.canonical(solver.kernel.jit_kwargs),
        toolchain_fingerprint=toolchain_fingerprint(),
        package_source_hash=package_source_hash(),
    )
    if any(expected[key]!=value for key,value in metadata.items()):
        raise ValueError('Actual compiler/dimension/shared metadata differs')
    if observer.graph_identity(graph)!=case['graph_identity']:
        raise ValueError('Actual complete graph differs from frozen workload')
    inputs = placement.grid_chain(solver,request['n_runs'])
    np.savez_compressed(output/'inputs.npz',inits=inputs[0],params=inputs[1])
    source = dict(
        status='COMPLETE_SOURCE_IDENTITY_PASS',case=name,
        request_sha256=request_sha,graph=observer.file_record(source_graph),
        graph_identity=observer.graph_identity(graph),metadata=metadata,
        config_hash=solver.kernel.config_hash,fn_hash=system.fn_hash,
        kernel_cache_policy=cache_policy,kernel_cache_dir=str(kernel_cache),
        input_arrays=[array_record(array) for array in inputs],
        inputs=observer.file_record(output/'inputs.npz'),
        source_snapshots=source_snapshots,
        native_overloads=graph['compilation_check']['native_overloads'],
        kernel_launches_requested=0,
    )
    model.write_json(output/'source.json',source)
    if command=='compile':
        original = observer.read_record(
            dict(path=sys.argv[6],sha256=sys.argv[7]))
        for key in ('status','case','request_sha256','graph_identity',
                    'metadata','config_hash','fn_hash','input_arrays',
                    'native_overloads','kernel_cache_policy'):
            if source[key]!=original[key]:
                raise ValueError('Pre-native construction differs: '+key)
        device = cuda.get_current_device()
        observed = {key:int(getattr(device,key)) for key in request['hardware']
                    if key.isupper()}
        if any(value!=request['hardware'][key]
               for key,value in observed.items()):
            raise ValueError('Actual device attributes differ from prediction')
        if list(device.compute_capability)!=[8,9]:
            raise ValueError('Actual compute capability differs')
        start=time.perf_counter()
        solver.compile(*inputs,duration=request['duration'],grid_type='verbatim')
        seconds=time.perf_counter()-start
        after=dict(
            actual_system_size=int(
                solver.kernel.single_integrator._algo_step.compile_settings.n),
            shared_stride_bytes=4*(int(solver.kernel.shared_memory_elements)+
                int(solver.kernel.shared_memory_needs_padding)),
            shared_elements_per_run=int(solver.kernel.shared_memory_elements),
            shared_padding_elements=int(solver.kernel.shared_memory_needs_padding),
            jit_kwargs=observer.canonical(solver.kernel.jit_kwargs),
            toolchain_fingerprint=toolchain_fingerprint(),
            package_source_hash=package_source_hash(),
        )
        changed=[]
        if after!=metadata:
            changed.append('metadata')
        if solver.kernel.config_hash!=source['config_hash']:
            changed.append('config_hash')
        if system.fn_hash!=source['fn_hash']:
            changed.append('fn_hash')
        if Path(solver.kernel.compile_settings.cache.cache_dir)!=kernel_cache:
            changed.append('isolated_kernel_cache')
        source_hashes={}
        for path,sha in observed_sources.items():
            source_hashes[path]=model.digest(path)
            if source_hashes[path]!=sha:
                changed.append('source:'+path)
        model.write_json(output/'post_compile_source.json',dict(
            metadata=after,config_hash=solver.kernel.config_hash,
            fn_hash=system.fn_hash,source_hashes=source_hashes,
            changed=changed,kernel_cache_dir=str(
                solver.kernel.compile_settings.cache.cache_dir)))
        dispatcher=solver.kernel.kernel
        kernel,=dispatcher.overloads.values()
        cubin,entry,library,adapter=observer.compiled_payload(
            kernel,dispatcher,request['native_adapter_sources'])
        model.write_json(output/'native_adapter.json',adapter)
        cubin_path=output/'kernel.cubin'
        cubin_path.write_bytes(cubin)
        disassembler=request['nvdisasm']
        if model.digest(disassembler['path'])!=disassembler['sha256']:
            raise ValueError('Native disassembler bytes changed')
        disasm_command=[disassembler['path'],'-c',str(cubin_path)]
        disasm=subprocess.run(disasm_command,capture_output=True,text=True,
                              check=True)
        sass_path=output/'kernel.sass'
        sass_path.write_text(disasm.stdout,encoding='utf-8')
        native=inspect_sass(disasm.stdout,entry)
        model.write_json(output/'native_categories.json',native)
        if changed:
            raise ValueError('Public compile changed source/configuration')
        requested=request['requested_block']
        stride=metadata['shared_stride_bytes']
        run_params=solver.kernel.run_params
        chunk_runs=[int(run_params[index].runs)
                    for index in range(run_params.num_chunks)]
        if sum(chunk_runs)!=request['n_runs'] or any(n<=0 for n in chunk_runs):
            raise ValueError('Actual chunk run counts differ from inputs')
        first_runs=chunk_runs[0]
        actual,dynamic=solver.kernel.limit_blocksize(
            requested,stride*min(first_runs,requested),stride,first_runs)
        dynamic=max(4,int(dynamic))
        if actual not in (32,64):
            raise ValueError('Public geometry has no frozen prediction')
        function=library.get_cufunc()
        resident=int(cuda.current_context().get_active_blocks_per_multiprocessor(
            function,actual,dynamic))
        threads_per_step=int(solver.kernel.single_integrator.threads_per_step)
        if threads_per_step!=1 or resident<=0:
            raise ValueError('Unsupported actual run geometry')
        required=2*int(device.MULTIPROCESSOR_COUNT)*resident*actual
        chunk_geometry=[dict(
            index=index,runs=runs,grid_blocks=(runs+actual-1)//actual,
            active_run_waves=runs/(int(device.MULTIPROCESSOR_COUNT)*resident*actual),
            at_least_two_full_waves=runs>=required,
        ) for index,runs in enumerate(chunk_runs)]
        selected=[]
        for item in case['predictions']:
            prediction=observer.read_record(item)
            if prediction['candidate']['block_size']==actual:
                selected.append(item)
        if len(selected)!=(8 if expected['placement']=='shared' else 4):
            raise ValueError('Actual geometry has incomplete frozen scenarios')
        result=dict(
            status='PUBLIC_KERNEL_COMPILE_ONLY_PASS',case=name,
            request_sha256=request_sha,source=observer.file_record(output/'source.json'),
            original_source=dict(path=sys.argv[6],sha256=sys.argv[7]),
            compilation_seconds=seconds,config_hash=solver.kernel.config_hash,
            device_attributes=observed,compute_capability=list(device.compute_capability),
            registers_per_thread=int(next(iter(dispatcher.get_regs_per_thread().values()))),
            local_bytes_per_thread=int(next(iter(dispatcher.get_local_mem_per_thread().values()))),
            static_shared_bytes=int(next(iter(dispatcher.get_shared_mem_per_block().values()))),
            requested_block=requested,production_block=int(actual),
            dynamic_shared_bytes=dynamic,shared_stride_bytes=stride,
            occupancy_query_blocks_per_sm=resident,
            run_params=dict(runs=int(run_params.runs),
                num_chunks=int(run_params.num_chunks),
                chunk_length=int(run_params.chunk_length),
                duration=float(run_params.duration),
                warmup=float(run_params.warmup),
                t0=float(run_params.t0),precision=np.dtype(run_params.precision).name),
            per_chunk_geometry=chunk_geometry,
            prospective_timing_eligible=all(
                item['at_least_two_full_waves'] for item in chunk_geometry),
            n_runs=request['n_runs'],entry_name=entry,
            cubin=observer.file_record(cubin_path),sass=observer.file_record(sass_path),
            categories=observer.file_record(output/'native_categories.json'),
            native_adapter=observer.file_record(output/'native_adapter.json'),
            disassembler=disassembler,disassembler_command=disasm_command,
            matched_pre_native_predictions=selected,
            kernel_launches_requested=0,launch_pinning=False,
            scope='Static whole public kernel; model describes one ERK step',
            local_bytes_are_not_spill_bytes=True,
            actual_carveout=None,actual_runtime_residency=None,
            original_ptx_not_observed=True,
        )
        model.write_json(output/'compile.json',result)
    elif command!='source':
        raise ValueError('Unknown worker command')
    print(json.dumps(dict(case=name,status='PASS',command=command)))
finally:
    if solver is not None:
        solver.close()
    set_cache_root(previous)
"""


def main():
    """Prepare CPU predictions/source or explicitly compile one holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--graphs", type=Path, required=True)
    setup.add_argument("--hardware", type=Path, required=True)
    setup.add_argument("--nvdisasm", type=Path, required=True)
    setup.add_argument("--output", type=Path, required=True)
    setup.add_argument("--reuse-predictions", type=Path)
    setup.add_argument("--reuse-predictions-sha")
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--prepared", type=Path, required=True)
    compile_parser.add_argument("--manifest-sha", required=True)
    compile_parser.add_argument("--case", choices=CASE_NAMES, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    manifest = read_record(
        dict(
            path=str(args.prepared / "manifest.json"),
            sha256=args.manifest_sha,
        )
    )
    if manifest["status"] != "PRE_NATIVE_PREDICTIONS_AND_SOURCE_PASS":
        raise ValueError("Prepared source gate did not complete")
    request_path = Path(manifest["request"]["path"])
    validate_request(request_path, manifest["request"]["sha256"])
    for name in CASE_NAMES:
        source = validate_source(
            manifest["sources"][name],
            read_record(manifest["request"])["cases"][name],
            manifest["request"]["sha256"],
        )
        worker = read_record(manifest["worker_receipts"][name])
        expected_command = [
            sys.executable,
            str(request_path.parent / "worker.py"),
            str(request_path.resolve()),
            manifest["request"]["sha256"],
            name,
            str(Path(manifest["sources"][name]["path"]).parent),
            "source",
        ]
        if (
            source["native_overloads"] != 0
            or worker["returncode"] != 0
            or worker["command"] != expected_command
            or worker["cwd"] != str(REPO)
        ):
            raise ValueError("Prepared source worker did not close cleanly")
    run_worker(
        request_path,
        args.case,
        args.output,
        "compile",
        manifest["sources"][args.case],
    )
    print(json.dumps(file_record(args.output / "compile.json")))


if __name__ == "__main__":
    main()
