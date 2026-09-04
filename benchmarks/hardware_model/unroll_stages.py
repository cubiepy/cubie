"""Prepare exact cache controls and explicitly gated native observations.

Preparation and verification use only the standard library. The observe
subcommand imports the installed compiler only after its explicit native
gate. Pickle instructions are interpreted as inert data; serialized
constructors, reducers and class bodies are never executed.
"""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickletools
import re
import sys
import traceback
from types import SimpleNamespace


SCHEMA = 1
FASTMATH_CLASS = (
    "numba_cuda_mlir.numba_cuda.core.options", "FastMathOptions"
)
DISTRIBUTIONS = (
    "cubie-numba-cuda-mlir", "numba-cuda", "numba", "llvmlite",
    "numpy", "cuda-core", "cuda-bindings", "nvidia-cuda-nvcc",
    "nvidia-cuda-nvrtc", "nvidia-nvjitlink",
)
MODULE_ROOTS = ("numba_cuda_mlir", "numba", "llvmlite", "cuda")
COMPILER_ENVIRONMENT = (
    "LIBLLVM7", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "NUMBA_ENABLE_CUDASIM", "CUBIE_CUDA_BACKEND",
    "NUMBA_CUDA_MLIR_DUMP_NVVM",
)
MARKER = object()


class Symbol:
    """An inert serialized constructor, reduction or object instance."""

    def __init__(self, kind, values):
        self.kind = kind
        self.values = values
        self.state = None


def digest(data: bytes) -> str:
    """Return the SHA-256 of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path) -> dict:
    """Bind a file path to its exact contents."""
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": digest(data), "bytes": len(data)}


def check_file(record: dict) -> bytes:
    """Read an artifact only if its bytes match its receipt."""
    data = Path(record["path"]).read_bytes()
    if digest(data) != record["sha256"]:
        raise ValueError(f"File identity mismatch: {record['path']}")
    if "bytes" in record and len(data) != record["bytes"]:
        raise ValueError(f"File size mismatch: {record['path']}")
    return data


def save_json(path: Path, value: dict) -> None:
    """Save JSON with exact UTF-8 LF bytes."""
    Path(path).write_bytes((json.dumps(value, indent=2) + "\n").encode())


def annotation_inventory(data: bytes) -> dict:
    """Retain text annotation lines without decoding bitcode or loop bodies."""
    if data.startswith(b"BC\xc0\xde"):
        return {"format": "bitcode", "text_inspection": "unavailable"}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"format": "binary", "text_inspection": "unavailable"}
    return {
        "format": "utf8", "lines": len(text.splitlines()),
        "annotation_lines": [
            {"line": index, "text": line}
            for index, line in enumerate(text.splitlines(), 1)
            if any(token in line for token in (
                "llvm.loop", "loop_annotation", "loop_unroll"
            ))
        ],
        "scope": "Literal annotation lines; no native loop-copy inference",
    }


def read_cache(data: bytes) -> tuple:
    """Interpret pickle data without executing reducers or constructors.

    Parameters
    ----------
    data : bytes
        Exact serialized cache bytes.

    Returns
    -------
    tuple
        Inert root value and opcode receipt. Unsupported instructions fail
        closed; all GLOBAL, REDUCE, NEWOBJ and BUILD effects stay symbolic.
    """
    stack = []
    memo = {}
    receipt = []

    def marked():
        position = max(i for i, value in enumerate(stack)
                       if value is MARKER)
        values = stack[position + 1:]
        del stack[position:]
        return values

    for opcode, argument, offset in pickletools.genops(data):
        name = opcode.name
        receipt.append({"offset": offset, "opcode": name})
        if name in ("PROTO", "FRAME"):
            continue
        if name == "MARK":
            stack.append(MARKER)
        elif name in ("NONE", "NEWTRUE", "NEWFALSE"):
            stack.append({"NONE": None, "NEWTRUE": True,
                          "NEWFALSE": False}[name])
        elif name in (
            "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
            "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8",
            "BININT", "BININT1", "BININT2", "LONG1", "LONG4",
            "BINFLOAT", "INT", "LONG", "FLOAT", "UNICODE",
        ):
            stack.append(argument)
        elif name in ("EMPTY_DICT", "EMPTY_LIST", "EMPTY_SET",
                      "EMPTY_TUPLE"):
            stack.append({"EMPTY_DICT": dict, "EMPTY_LIST": list,
                          "EMPTY_SET": set, "EMPTY_TUPLE": tuple}[name]())
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[int(argument)] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo[int(argument)])
        elif name in ("TUPLE", "FROZENSET", "LIST", "DICT"):
            values = marked()
            if name == "DICT":
                stack.append(dict(zip(values[::2], values[1::2])))
            else:
                stack.append({"TUPLE": tuple, "FROZENSET": frozenset,
                              "LIST": list}[name](values))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            count = int(name[-1])
            values = tuple(stack[-count:])
            del stack[-count:]
            stack.append(values)
        elif name in ("APPENDS", "ADDITEMS", "SETITEMS"):
            values = marked()
            target = stack[-1]
            if name == "SETITEMS":
                if len(values) % 2:
                    raise ValueError("Odd dictionary item count")
                target.update(zip(values[::2], values[1::2]))
            elif name == "APPENDS":
                target.extend(values)
            else:
                target.update(values)
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            stack[-1][key] = value
        elif name == "APPEND":
            value = stack.pop()
            stack[-1].append(value)
        elif name == "STACK_GLOBAL":
            value, module = stack.pop(), stack.pop()
            stack.append(Symbol("global", (module, value)))
        elif name == "GLOBAL":
            stack.append(Symbol("global", tuple(argument.split(" "))))
        elif name in ("REDUCE", "NEWOBJ"):
            arguments, constructor = stack.pop(), stack.pop()
            stack.append(Symbol(name, (constructor, arguments)))
        elif name == "NEWOBJ_EX":
            keywords, arguments, constructor = (
                stack.pop(), stack.pop(), stack.pop()
            )
            stack.append(Symbol(name, (constructor, arguments, keywords)))
        elif name == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Symbol):
                raise ValueError("BUILD target must remain symbolic")
            stack[-1].state = state
        elif name == "POP":
            stack.pop()
        elif name == "POP_MARK":
            marked()
        elif name == "DUP":
            stack.append(stack[-1])
        elif name == "STOP":
            if len(stack) != 1 or offset + 1 != len(data):
                raise ValueError("Ambiguous pickle root or trailing bytes")
            return stack[0], receipt
        else:
            raise ValueError(f"Unsupported inert opcode {name} at {offset}")
    raise ValueError("Missing pickle STOP")


def plain_options(options: dict) -> dict:
    """Convert only whitelisted target-option values to JSON data."""
    result = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise ValueError("Non-string target option")
        if key == "fastmath" and isinstance(value, Symbol):
            constructor, arguments = value.values
            if not (
                value.kind == "NEWOBJ" and arguments == ()
                and isinstance(constructor, Symbol)
                and constructor.kind == "global"
                and constructor.values == FASTMATH_CLASS
                and set(value.state) == {"flags"}
                and isinstance(value.state["flags"], set)
                and all(isinstance(x, str) for x in value.state["flags"])
            ):
                raise ValueError("Unsupported serialized fastmath options")
            result[key] = {"serialized_class": list(FASTMATH_CLASS),
                           "flags": sorted(value.state["flags"])}
        elif type(value) in (str, int, float, bool, type(None)):
            result[key] = value
        elif isinstance(value, (list, tuple)) and not value:
            result[key] = []
        else:
            raise ValueError(f"Nonliteral target option: {key}")
    return result


def versions() -> dict:
    """Read installed distribution versions without importing packages."""
    result = {}
    for name in DISTRIBUTIONS:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def compiler_inventory(wheel_root: Path, libraries: list) -> list:
    """Hash installed compiler code and explicitly supplied native inputs."""
    roots = [Path(wheel_root).parent / name for name in MODULE_ROOTS]
    paths = {Path(path).resolve() for path in libraries}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Missing compiler package: {root}")
        paths.update(path.resolve() for path in root.rglob("*")
                     if path.suffix.lower() in (".py", ".dll", ".pyd",
                                                ".so", ".bc"))
    return [file_record(path) for path in sorted(paths)]


def prepare(receipt_path: Path, wheel_root: Path, libraries: list,
            output: Path) -> dict:
    """Prepare immutable cache controls without compiler imports.

    Parameters
    ----------
    receipt_path : pathlib.Path
        Schema-2 counted-unroll extraction receipt.
    wheel_root : pathlib.Path
        Installed numba_cuda_mlir package directory.
    libraries : list
        Exact LLVM, libnvvm, libdevice and nvJitLink library paths.
    output : pathlib.Path
        Fresh output directory.

    Returns
    -------
    dict
        Manifest binding original and prepared artifact identities.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Preparation output must be fresh")
    receipt = json.loads(Path(receipt_path).read_bytes())
    if receipt["schema_version"] != 2:
        raise ValueError("Schema-2 original literal/file hashes required")
    inventory = compiler_inventory(Path(wheel_root), libraries)
    indexed = {str(Path(item["path"]).resolve()): item for item in inventory}
    for path, expected in receipt["installed_file_sha256"].items():
        if indexed[str(Path(path).resolve())]["sha256"] != expected:
            raise ValueError(f"Original compiler anchor changed: {path}")
    cases = []
    pending = []
    for case in receipt["cases"]:
        name = case["case"]
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError("Unsafe case name")
        cache_record = file_record(Path(case["cache_path"]))
        if cache_record["sha256"] != case["cache_sha256"]:
            raise ValueError(f"Original cache changed: {name}")
        payload, opcodes = read_cache(check_file(cache_record))
        options = plain_options(payload["targetoptions"])
        target = payload["gpu_target"]
        linker_cc = target["linker_cc"]
        if not (type(linker_cc) is tuple and len(linker_cc) == 2
                and all(type(x) is int for x in linker_cc)):
            raise ValueError("Linker CC is not an exact literal pair")
        if not (options["chip"] == target["linker_arch"] == "sm_89"
                and linker_cc == (8, 9) and options["lto"] is True
                and options["link"] == [] and not payload["needs_nrt"]):
            raise ValueError("Replay supports SM89 LTO without links/NRT")
        original = case["compile_receipt"]
        bank_bytes = Path(original["path"]).read_bytes()
        bank_line = bank_bytes.splitlines()[original["line"] - 1]
        bank = json.loads(bank_line)
        for field in ("policy", "algo", "compile_s", "cached",
                      "source_hash", "compiler_identity"):
            if bank[field] != original[field]:
                raise ValueError(f"Original compile row differs: {field}")
        if bank["status"] != "ok" or bank["cached"] is not False:
            raise ValueError("An original successful uncached row is required")
        mlir = payload["mlir_module_optimized"].encode("utf-8")
        lto = payload["ltoir"]
        cubin = payload["cubin"]
        for data, expected in (
            (mlir, case["mlir_literal_utf8_sha256"]),
            (lto, case["ltoir_sha256"]),
            (cubin, case["cubin_sha256"]),
            (cubin, bank["cubin_sha256"]),
        ):
            if digest(data) != expected:
                raise ValueError(f"Original artifact hash mismatch: {name}")
        export = Path(case["mlir_path"]).read_bytes()
        if (digest(export) != case["mlir_export_file_sha256"]
                or export.replace(b"\r\n", b"\n") != mlir):
            raise ValueError("Original MLIR export/literal relation changed")
        if Path(case["ltoir_path"]).read_bytes() != lto:
            raise ValueError("Original LTO export changed")
        if payload["ptx"] != "":
            raise ValueError("Expected empty natural PTX in LTO cache")
        if Path(bank["artefacts"]["cubin"]).read_bytes() != cubin:
            raise ValueError("Original bank cubin differs from cached cubin")
        artifacts = {}
        for suffix, data in (("mlir", mlir), ("ltoir", lto),
                             ("cubin", cubin), ("ptx", b"")):
            path = output / f"{name}.original.{suffix}"
            artifacts[suffix] = {"path": str(path),
                                 "sha256": digest(data), "bytes": len(data)}
            pending.append((path, data))
        cases.append({
            "case": name, "cache": cache_record,
            "original_extraction": case,
            "compile_row": {"path": original["path"],
                            "line": original["line"],
                            "line_sha256": digest(bank_line)},
            "original_compile": bank, "targetoptions": options,
            "linker_target": {"cc": list(linker_cc),
                              "arch": target["linker_arch"]},
            "needs_nrt": payload["needs_nrt"],
            "nrt_inline": payload["nrt_inline"],
            "artifacts": artifacts, "inert_pickle_opcodes": opcodes,
            "original_mlir_annotations": annotation_inventory(mlir),
        })
    if len({case["case"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identities")
    manifest = {
        "schema_version": SCHEMA,
        "route": "saved optimized MLIR -> pre-codegen -> LLVM70 LTO",
        "preparation_native_imports": 0,
        "preparation_kernel_launches": 0,
        "input_receipt": file_record(receipt_path),
        "tool_source": file_record(Path(__file__)),
        "python": file_record(Path(sys.executable)),
        "python_version": sys.version, "versions": versions(),
        "wheel_root": str(Path(wheel_root).resolve()),
        "libraries": [str(Path(path).resolve()) for path in libraries],
        "compiler_inventory": inventory, "cases": cases,
        "limits": [
            "Original linked helper objects are not serialized.",
            "Replay equality confirms original outputs, not unsaved inputs.",
            "Inspection LLVM uses a separate gen_llvmir route.",
            "Diagnostic PTX is a separate -ptx re-link of original LTO.",
            "No native loop-copy or pass attribution from whole-file hashes.",
        ],
    }
    output.mkdir(parents=True)
    for path, data in pending:
        path.write_bytes(data)
    source = check_file(manifest["tool_source"])
    observer = output / "observer.py"
    observer.write_bytes(source)
    manifest["observer"] = file_record(observer)
    save_json(output / "manifest.json", manifest)
    return manifest


def verify(manifest_path: Path) -> dict:
    """Recheck every saved control and installed compiler input on CPU."""
    manifest = json.loads(Path(manifest_path).read_bytes())
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("Unsupported observer schema")
    for field in ("input_receipt", "tool_source", "observer", "python"):
        check_file(manifest[field])
    if digest(Path(__file__).read_bytes()) != manifest["observer"]["sha256"]:
        raise ValueError("Executing observer differs from prepared source")
    if (file_record(Path(sys.executable)) != manifest["python"]
            or sys.version != manifest["python_version"]
            or versions() != manifest["versions"]):
        raise ValueError("Python or distribution identity changed")
    current = compiler_inventory(Path(manifest["wheel_root"]),
                                 manifest["libraries"])
    if current != manifest["compiler_inventory"]:
        raise ValueError("Installed compiler inventory changed")
    extraction = json.loads(check_file(manifest["input_receipt"]))
    if extraction["schema_version"] != 2:
        raise ValueError("Original extraction schema changed")
    originals = {item["case"]: item for item in extraction["cases"]}
    if (len(originals) != len(extraction["cases"])
            or [item["case"] for item in manifest["cases"]]
            != [item["case"] for item in extraction["cases"]]):
        raise ValueError("Prepared original case membership/order differs")
    indexed = {item["path"]: item for item in current}
    for path, expected in extraction["installed_file_sha256"].items():
        if indexed[str(Path(path).resolve())]["sha256"] != expected:
            raise ValueError("Original compiler anchor changed")
    for case in manifest["cases"]:
        original = originals[case["case"]]
        if case["original_extraction"] != original:
            raise ValueError("Prepared extraction differs from input receipt")
        if (case["cache"]["path"] != original["cache_path"]
                or case["cache"]["sha256"] != original["cache_sha256"]):
            raise ValueError("Prepared cache differs from original extraction")
        payload, opcodes = read_cache(check_file(case["cache"]))
        target = payload["gpu_target"]
        linker_target = {"cc": list(target["linker_cc"]),
                         "arch": target["linker_arch"]}
        derived = {
            "targetoptions": plain_options(payload["targetoptions"]),
            "linker_target": linker_target,
            "needs_nrt": payload["needs_nrt"],
            "nrt_inline": payload["nrt_inline"],
            "inert_pickle_opcodes": opcodes,
        }
        for name, value in derived.items():
            if case[name] != value:
                raise ValueError(f"Prepared cache field differs: {name}")
        options = derived["targetoptions"]
        if not (options["chip"] == linker_target["arch"] == "sm_89"
                and linker_target["cc"] == [8, 9]
                and options["lto"] is True and options["link"] == []
                and not derived["needs_nrt"] and payload["ptx"] == ""):
            raise ValueError("Original cache is outside supported replay")
        mlir = payload["mlir_module_optimized"].encode("utf-8")
        raw = {"mlir": mlir, "ltoir": payload["ltoir"],
               "cubin": payload["cubin"], "ptx": payload["ptx"].encode()}
        if set(case["artifacts"]) != set(raw):
            raise ValueError("Prepared artifact membership differs")
        for name, data in raw.items():
            if check_file(case["artifacts"][name]) != data:
                raise ValueError(
                    f"Prepared artifact differs from cache: {name}"
                )
        for name, field in (("mlir", "mlir_literal_utf8_sha256"),
                            ("ltoir", "ltoir_sha256"),
                            ("cubin", "cubin_sha256")):
            if digest(raw[name]) != original[field]:
                raise ValueError("Original artifact/extraction hash differs")
        if case["original_mlir_annotations"] != annotation_inventory(mlir):
            raise ValueError("Original MLIR annotation receipt differs")
        exported = Path(original["mlir_path"]).read_bytes()
        if (digest(exported) != original["mlir_export_file_sha256"]
                or exported.replace(b"\r\n", b"\n") != mlir
                or Path(original["ltoir_path"]).read_bytes() != raw["ltoir"]):
            raise ValueError("Original literal/export relation changed")
        reference = case["compile_row"]
        original_compile = original["compile_receipt"]
        if any(reference[key] != original_compile[key]
               for key in ("path", "line")):
            raise ValueError("Original compilation row membership differs")
        line = Path(reference["path"]).read_bytes().splitlines()[
            reference["line"] - 1]
        if (digest(line) != reference["line_sha256"]
                or json.loads(line) != case["original_compile"]):
            raise ValueError("Original compile line identity changed")
        bank = json.loads(line)
        for field in ("policy", "algo", "compile_s", "cached",
                      "source_hash", "compiler_identity"):
            if bank[field] != original_compile[field]:
                raise ValueError(f"Original compile field differs: {field}")
        if (bank["status"] != "ok" or bank["cached"] is not False
                or bank["cubin_sha256"] != digest(raw["cubin"])
                or Path(bank["artefacts"]["cubin"]).read_bytes()
                != raw["cubin"]):
            raise ValueError("Original bank/cached cubin identity differs")
    return manifest


def loaded_libraries() -> list:
    """Return actual loaded Windows native module paths."""
    if os.name != "nt":
        raise ValueError("This native observer requires Windows DLL receipts")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    process_status = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    process = kernel.GetCurrentProcess()
    handles = (ctypes.c_void_p * 4096)()
    needed = ctypes.c_ulong()
    process_status.EnumProcessModulesEx.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong,
    )
    if not process_status.EnumProcessModulesEx(
        process, handles, ctypes.sizeof(handles), ctypes.byref(needed), 3
    ) or needed.value > ctypes.sizeof(handles):
        raise ValueError("Could not enumerate all loaded DLLs")
    process_status.GetModuleFileNameExW.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong,
    )
    result = []
    for handle in handles[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        buffer = ctypes.create_unicode_buffer(32768)
        length = process_status.GetModuleFileNameExW(
            process, handle, buffer, len(buffer)
        )
        if not length or length == len(buffer):
            raise ValueError("Incomplete loaded DLL path")
        result.append(str(Path(buffer.value).resolve()))
    return sorted(set(result))


def observe(args: argparse.Namespace) -> dict:
    """Replay one exact native control only under explicit authorization.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including execute_native and a fresh output directory.

    Returns
    -------
    dict
        Raw stage receipts and independent byte-equality gates. This action
        invokes native compilation/linking but never launches a kernel.
    """
    if not args.execute_native:
        raise ValueError("observe requires explicit --execute-native")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Observation output must be fresh")
    dump = output / "nvvm_input"
    configured = os.environ.get("NUMBA_CUDA_MLIR_DUMP_NVVM", "")
    if not configured or Path(configured).resolve() != dump:
        raise ValueError(f"Set external NVVM dump environment to {dump}")
    if os.environ.get("NUMBA_ENABLE_CUDASIM") != "0":
        raise ValueError("External NUMBA_ENABLE_CUDASIM=0 is required")
    manifest = verify(Path(args.manifest))
    case = next(item for item in manifest["cases"]
                if item["case"] == args.case)
    output.mkdir(parents=True)
    dump.mkdir()
    result = {
        "schema_version": SCHEMA, "case": args.case,
        "manifest": file_record(Path(args.manifest)),
        "observer": file_record(Path(__file__)),
        "pid": os.getpid(), "argv": sys.argv, "cwd": str(Path.cwd()),
        "environment": {key: os.environ.get(key)
                        for key in COMPILER_ENVIRONMENT},
        "status": "started", "kernel_launches": 0, "stages": {},
        "gates": {}, "original_outputs_reproduced": False,
        "stage_scope": (
            "Observed replay input; the original libnvvm input was not "
            "saved. Equal outputs do not prove equal unsaved inputs."
        ),
    }

    def save_stage(name, data):
        path = output / name
        path.write_bytes(data)
        result["stages"][name] = file_record(path)
        if path.suffix in (".ll", ".mlir", ".ptx"):
            result["stages"][name]["annotations"] = annotation_inventory(data)
        return data

    try:
        # Import only in the separately invoked, opted-in native process.
        compiler = importlib.import_module("numba_cuda_mlir")
        optimization = importlib.import_module(
            "numba_cuda_mlir.mlir_optimization"
        )
        linker_module = importlib.import_module("numba_cuda_mlir.linker")
        fastmath = importlib.import_module("numba_cuda_mlir.fastmath")
        compiler_tools = importlib.import_module("numba_cuda_mlir.tools")
        libs = importlib.import_module(
            "numba_cuda_mlir.numba_cuda.cudadrv.libs"
        )
        if Path(compiler.__file__).resolve().parent != Path(
            manifest["wheel_root"]
        ):
            raise ValueError("Imported compiler root differs")
        if tuple(compiler_tools.get_gpu_compute_capability(tuple)) != (8, 9):
            raise ValueError("Replay requires the original SM89 target")
        input_paths = [compiler_tools.get_llvm70_capi_path(),
                       optimization._get_libnvvm_path().decode(),
                       libs.get_libdevice()]
        llvm_override = os.environ.get("LIBLLVM7")
        if llvm_override:
            input_paths.append(llvm_override)
        else:
            llvm_directory = Path(optimization.__file__).resolve().parent
            candidates = [llvm_directory / "lib" / name
                          for name in ("LLVM-C.dll", "LLVM.dll")]
            llvm_runtime = next(
                (path for path in candidates if path.is_file()), None
            )
            if llvm_runtime is None:
                raise ValueError("No bound bundled LLVM70 runtime")
            input_paths.append(str(llvm_runtime))
        inventory = {str(Path(item["path"]).resolve()): item
                     for item in manifest["compiler_inventory"]}
        result["resolved_native_inputs"] = []
        for path in input_paths:
            record = file_record(Path(path))
            if inventory.get(record["path"]) != record:
                raise ValueError(f"Unbound native input: {path}")
            result["resolved_native_inputs"].append(record)
        options = dict(case["targetoptions"])
        fast = options["fastmath"]
        if isinstance(fast, dict):
            options["fastmath"] = set(fast["flags"])
        result["effective_targetoptions"] = case["targetoptions"]
        original_mlir = check_file(case["artifacts"]["mlir"])
        with optimization.context.get_context():
            module = optimization.ir.Module.parse(original_mlir.decode())
            optimization.run_pre_codegen_patterns(module)
            save_stage("after_pre_codegen.mlir", str(module).encode())
            lto = optimization._call_llvm70_capi(
                module, options, gen_lto=True
            )
        save_stage("replayed.ltoir", lto)
        result["gates"]["lto_equal_original"] = (
            lto == check_file(case["artifacts"]["ltoir"])
        )
        dumps = sorted(dump.iterdir())
        if len(dumps) != 1 or not dumps[0].is_file():
            raise ValueError("Exactly one natural libnvvm input required")
        result["stages"]["natural_replay_libnvvm_input"] = file_record(
            dumps[0]
        )
        result["stages"]["natural_replay_libnvvm_input"]["annotations"] = (
            annotation_inventory(dumps[0].read_bytes())
        )
        module_flags = fastmath.nvvm_fastmath_options(options["fastmath"])
        linker_options = {
            "cc": tuple(case["linker_target"]["cc"]),
            "arch": case["linker_target"]["arch"],
            "lto": True, "optimize_unused_variables": True,
            "verbose": options["dump"], "debug": options["debug"],
            "lineinfo": options["lineinfo"],
            "optimization_level": int(options["opt_level"]),
            "ptxas_options": options["ptxas_options"],
            "max_registers": options["max_registers"],
            **{name: module_flags.get(name)
               for name in ("ftz", "fma", "prec_div", "prec_sqrt")},
        }
        result["effective_linker_options"] = linker_options
        linker = linker_module.Linker(**linker_options)
        original_lto = check_file(case["artifacts"]["ltoir"])
        linker.add_ltoir(original_lto)
        cubin = bytes(linker.complete().code)
        save_stage("original_lto_relinked.cubin", cubin)
        result["gates"]["original_lto_relink_cubin_equal"] = (
            cubin == check_file(case["artifacts"]["cubin"])
        )
        cres = SimpleNamespace(metadata={
            "mlir_module_optimized": original_mlir.decode(),
            "ltoir": original_lto, "targetoptions": options,
            "needs_nrt": False, "nrt_inline": case["nrt_inline"],
        })
        if args.diagnostic_llvm:
            text = optimization.get_llvmir(cres, options)
            save_stage("diagnostic_gen_llvmir.ll", text.encode())
        if args.diagnostic_ptx:
            text = optimization.get_lto_ptx(cres, linker, options)
            save_stage("diagnostic_original_lto_relink.ptx", text.encode())
        result["loaded_libraries"] = loaded_libraries()
        result["loaded_compiler_libraries"] = []
        for path in result["loaded_libraries"]:
            if any(word in Path(path).name.lower()
                   for word in ("llvm", "nvvm", "nvjitlink")):
                record = file_record(Path(path))
                result["loaded_compiler_libraries"].append(record)
                if inventory.get(record["path"]) != record:
                    raise ValueError(f"Unbound loaded compiler DLL: {path}")
        if not any("nvjitlink" in Path(item["path"]).name.lower()
                   for item in result["loaded_compiler_libraries"]):
            raise ValueError("Actual nvJitLink DLL identity not captured")
        result["imported_compiler_modules"] = []
        for name, imported in sorted(sys.modules.items()):
            if name.split(".")[0] not in MODULE_ROOTS:
                continue
            path = getattr(imported, "__file__", None)
            if path is None:
                continue
            record = file_record(Path(path))
            if inventory.get(record["path"]) != record:
                raise ValueError(f"Unbound imported compiler module: {name}")
            result["imported_compiler_modules"].append(
                {"module": name, **record}
            )
        result["gates"]["loaded_compiler_inputs_bound"] = True
        result["original_outputs_reproduced"] = all(result["gates"].values())
        result["effective_environment"] = {
            key: os.environ.get(key) for key in COMPILER_ENVIRONMENT
        }
        result["status"] = (
            "matched" if result["original_outputs_reproduced"]
            else "replay_mismatch"
        )
    except Exception:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
    finally:
        result["retained_nvvm_dumps"] = [
            file_record(path) for path in sorted(dump.iterdir())
            if path.is_file()
        ]
        save_json(output / "result.json", result)
    return result


def main() -> None:
    """Run preparation, CPU verification or explicitly opted-in replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--receipt", type=Path, required=True)
    prepare_parser.add_argument("--wheel-root", type=Path, required=True)
    prepare_parser.add_argument("--library", action="append", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    observe_parser = commands.add_parser("observe")
    observe_parser.add_argument("--manifest", type=Path, required=True)
    observe_parser.add_argument("--case", required=True)
    observe_parser.add_argument("--output", type=Path, required=True)
    observe_parser.add_argument("--execute-native", action="store_true")
    observe_parser.add_argument("--diagnostic-llvm", action="store_true")
    observe_parser.add_argument("--diagnostic-ptx", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.receipt, args.wheel_root, args.library,
                         args.output)
        print(json.dumps({"prepared_cases": len(result["cases"]),
                          "output": str(args.output.resolve())}))
    elif args.command == "verify":
        result = verify(args.manifest)
        print(json.dumps({"verified_cases": len(result["cases"]),
                          "native_imports": 0}))
    else:
        result = observe(args)
        print(json.dumps({"status": result["status"],
                          "gates": result["gates"]}))
        if result["status"] != "matched":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
