"""Prepare same-cell memory composites with explicit native review.

The measured sequence includes store, optional shared fence, load and
loop administration. No isolated store-completion latency is inferred.
"""

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
ARITHMETIC_SHA = (
    "8283f1a5ded6bd14c1b0e27d2cd67b7ca48523dc617d3a366bf45399300d9008"
)
LATENCY_SHA = (
    "bd6172f8e924583fabed2d5dd621da7824fdad46aa9dc5730eb36b0f663c76f0"
)
HARDWARE_SHA = (
    "758b54944fe2bd88b6778df59277b89b8ea66d0107bf1feea3825dfd25ea898e"
)
POISON = 0xA5A5A5A5
TRACE_STEPS = 1
IDENTITY_FIELDS = (
    "space",
    "fence",
    "body_operations",
    "block_size",
    "waves",
    "seeds_sha256",
    "coefficients",
    "kernel_sha256",
    "generator_sha256",
    "worker_sha256",
    "arithmetic_sha256",
    "latency_sha256",
    "hardware_source_sha256",
    "hardware_manifest_sha256",
    "native_certificate_sha256",
)
ARTIFACTS = (
    "kernel.cubin",
    "kernel.ptx",
    "kernel.sass",
    "native_elf.txt",
    "kernel.py",
    "primitive.ptx",
    "benchmark_source.py",
    "worker.py",
    "seeds.npy",
    "arithmetic_source.py",
    "latency_source.py",
    "hardware_source.py",
    "hardware_manifest.json",
)


def digest(path):
    """Return an exact file-byte digest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write a finite machine-readable record."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_module(path, name):
    """Load a previously verified source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def arithmetic(directory=None):
    """Use the frozen CPU helpers, without importing a CUDA package."""
    path = (
        Path(directory) / "arithmetic_source.py"
        if directory
        else (
            SCRIPT.with_name("arithmetic_source.py")
            if SCRIPT.with_name("arithmetic_source.py").exists()
            else SCRIPT.with_name("arithmetic_service_probe.py")
        )
    )
    if digest(path) != ARITHMETIC_SHA:
        raise ValueError("Frozen arithmetic helper differs")
    return load_module(path, "store_arithmetic_cpu")


def helpers(path):
    """Load the frozen CPU control-flow helper."""
    if digest(path) != LATENCY_SHA:
        raise ValueError("Frozen latency helper differs")
    return load_module(path, "store_latency_cpu")


def plan(manifest):
    """Bind maximal CTA geometry to the queried hardware thread limit."""
    if manifest["device_attributes"]["MULTIPROCESSOR_COUNT"] != 56:
        raise ValueError("This protocol binds the queried56-SM target")
    return arithmetic().plan(manifest)


def inputs(block, body):
    """Return lane-dependent 32-bit payloads and distinct runtime offsets."""
    seeds = np.arange(block, dtype=np.uint32) % np.uint32(32) + np.uint32(1)
    return seeds, dict(m=0, a=0, body=body)


def expected_bits(operation, seeds, coefficients, count, cycles=None):
    """Compute the endpoint after one increment per complete pair body."""
    del operation, cycles
    body = coefficients["body"]
    if type(count) is not int or count <= 0 or count % body:
        raise ValueError("Need an exact positive number of pair bodies")
    return seeds + np.uint32((count // body) % 2**32)


def valid_repeats(operation, seeds, coefficients, repeats, body, cycles=None):
    """Require bounded N/2N counters and distinct exact endpoints."""
    del operation, cycles
    if (
        type(repeats) is not int
        or not 0 < repeats < 2**30
        or repeats % 2 == 0
        or coefficients != dict(m=0, a=0, body=body)
    ):
        return False
    for multiple in (1, 2):
        expected = seeds + np.uint32(repeats * multiple)
        checksum = seeds ^ expected ^ np.uint32(repeats * multiple)
        checksum ^= np.uint32(POISON)
        if np.any(checksum == np.uint32(2**32 - 1)):
            return False
    return True


def source_text(space, fence, body):
    """Emit runtime-unknown equal offsets and a dependent memory sequence."""
    if space not in ("shared", "local") or body not in (33, 257):
        raise ValueError("Unsupported memory space or body size")
    if fence not in ("none", "cta") or space == "local" and fence != "none":
        raise ValueError("The fence contrast is shared-memory-only")
    storage = (
        ".shared .align 4 .b8 cells[4096];"
        if space == "shared"
        else ".local .align 4 .b8 cells[4];"
    )
    pair = [f"st.{space}.u32 [write_address], x;"]
    if fence == "cta":
        pair.append("membar.cta;")
    pair.append(f"ld.{space}.u32 x, [read_address];")
    ptx = [
        "{",
        storage,
        ".reg .pred inactive, again, bad;",
        ".reg .b32 tid, warp, sm0, sm1, count, x, expected, ready;",
        ".reg .b32 base, write_address, read_address, first, final;",
        ".reg .u64 address, begin, end, delta, total;",
        "mov.u32 tid, %tid.x;",
        "shr.u32 warp, tid, 5;",
        "setp.ge.u32 inactive, warp, $6;",
        "@inactive bra finished;",
        "mov.u32 sm0, %smid;",
        "ld.global.u32 x, [$2];",
        "ld.global.u32 expected, [$3];",
        "mov.u32 base, cells;",
    ]
    if space == "shared":
        ptx += ["mad.lo.u32 base, tid, 4, base;"]
    ptx += [
        "add.u32 write_address, base, $8;",
        "add.u32 read_address, base, $9;",
        f"mov.u32 first, {POISON};",
        f"st.{space}.u32 [write_address], first;",
        f"ld.{space}.u32 first, [read_address];",
        "st.global.u32 [$4], first;",
        "mov.u32 count, $5;",
        "xor.b32 ready, x, expected;",
        "xor.b32 ready, ready, first;",
        "xor.b32 ready, ready, count;",
        "xor.b32 ready, ready, $8;",
        "xor.b32 ready, ready, $9;",
        "setp.eq.u32 bad, ready, 0xffffffff;",
        "@bad bra invalid;",
        "mov.u64 begin, %clock64;",
        "timed_pairs:",
    ]
    ptx += pair * body
    ptx += [
        "add.u32 x, x, 1;",
        "add.u32 count, count, -1;",
        "setp.ne.u32 again, count, 0;",
        "@again bra.uni timed_pairs;",
        "setp.ne.u32 bad, x, expected;",
        "@bad bra invalid;",
        "mov.u64 end, %clock64;",
        "sub.u64 delta, end, begin;",
        "mov.u32 sm1, %smid;",
        f"ld.{space}.u32 final, [read_address];",
        "st.global.u32 [$4+4], final;",
        f"mul.wide.u32 total, $5, {body};",
        "st.global.u64 [$1+0], begin;",
        "st.global.u64 [$1+8], end;",
        "st.global.u64 [$1+16], delta;",
        "cvt.u64.u32 address, x;",
        "st.global.u64 [$1+24], address;",
        "cvt.u64.u32 address, sm0;",
        "st.global.u64 [$1+32], address;",
        "cvt.u64.u32 address, sm1;",
        "st.global.u64 [$1+40], address;",
        "mov.u64 address, 1;",
        "st.global.u64 [$1+48], address;",
        "st.global.u64 [$1+56], total;",
        "bra finished;",
        "invalid:",
        "mov.u64 address, 0;",
        "st.global.u64 [$1+48], address;",
        "finished:",
        "bar.sync 0;",
        "mov.u32 $0, 0;",
        "}",
    ]
    asm = "\n".join(ptx)
    escaped = asm.replace("\\", "\\5C").replace('"', "\\22")
    escaped = escaped.replace("\n", "\\0A")
    names = (
        "out",
        "seed",
        "expected",
        "memory",
        "count",
        "warps",
        "reserved",
        "write_offset",
        "read_offset",
    )
    types = ["i64"] * 4 + ["i32"] * 5
    signature = ", ".join(f"%{n}: {t}" for n, t in zip(names, types))
    arguments = ", ".join("%" + n for n in names)
    intrinsic = (
        f"func.func private @store_chain({signature}) -> i32 "
        "attributes {always_inline} {\n"
        f'  %answer = "llvm.inline_asm"({arguments}) {{\n'
        f'    asm_string = "{escaped}",\n'
        '    constraints = "=r,l,l,l,l,r,r,r,r,r,~{memory}", '
        "has_side_effects\n"
        f"  }} : ({', '.join(types)}) -> i32\n"
        "  return %answer : i32\n}\n"
    )
    source = (
        "from numpy import uint32, uint64\n"
        "from cubie.cuda_simsafe import cuda\n\n"
        f"store_chain = cuda.intrin.define({intrinsic!r})\n\n"
        "def probe(output, seeds, expected, memory, iterations, warps, "
        "reserved, write_offset, read_offset):\n"
        "    tid = uint64(cuda.threadIdx.x)\n"
        "    index = uint64(cuda.blockIdx.x) * "
        "uint64(cuda.blockDim.x) + tid\n"
        "    store_chain(output + index * uint64(64), "
        "seeds + tid * uint64(4), expected + tid * uint64(4), "
        "memory + index * uint64(8), uint32(iterations), uint32(warps), "
        "uint32(reserved), uint32(write_offset), uint32(read_offset))\n"
    )
    compile(source, "<store-kernel>", "exec")
    return source, asm


def validate_memory(values, expected, seeds, warps):
    """Require both actual poison read and final stored memory per lane."""
    if values.dtype != np.uint32 or values.shape != (112, 1024, 2):
        raise ValueError("Memory observation type/shape differs")
    active = values[:, : 32 * warps]
    if not (
        np.all(active[:, :, 0] == POISON)
        and np.all(active[:, :, 1] == expected[: 32 * warps] - np.uint32(1))
        and np.all(values[:, 32 * warps :] == np.uint32(2**32 - 1))
        and np.all(seeds != POISON)
    ):
        raise ValueError("Initial/final memory or inactive sentinel differs")


def validate_output(values, expected, repeats, warps, geometry, body):
    """Check every endpoint, timestamp, SMID and full-warp population."""
    return arithmetic().validate_output(
        values, expected, repeats, warps, geometry, body
    )


def duration_ok(row):
    """Apply the finite 20 ms event and observed-clock duration gates."""
    return arithmetic().duration_ok(row)


def sample_order():
    """Use two mirrored blocks of six samples per N/population."""
    return arithmetic().sample_order()


def parameter_abi(text, entry):
    """Bind four uint64 addresses and five uint32 scalar parameters."""
    return arithmetic().parameter_abi(text, entry)


def parse_native(directory, entry):
    """Parse retained native text through the frozen CPU parser."""
    return arithmetic(directory).parse_native(directory, entry)


def check_native(
    instructions, loops, labels, space, body, abi, helper, fence="none"
):
    """Prove the counted pair recurrence before independent whole-code review.

    Address definitions, complete endpoint stores and convergence-stack
    semantics additionally require a retained independent certificate.
    That certificate is bound to this exact native inventory before any
    ordinary or profiled launch is permitted.
    """
    store, load = ("STS", "LDS") if space == "shared" else ("STL", "LDL")
    clocks = [i for i, x in enumerate(instructions) if x["opcode"] == "CS2R"]
    if len(clocks) != 2:
        raise ValueError("Need exactly two clock reads")
    start, end = clocks
    for i in clocks:
        if (
            instructions[i]["predicate"]
            or helper.operands(instructions[i])[1:] != ["SR_CLOCKLO"]
            or len(helper.written_registers(instructions[i])) != 2
        ):
            raise ValueError("Unproved clock pair")
    candidates = [
        x
        for x in loops
        if start < x["start_index"] <= x["end_index"] < end
        and x["opcounts"].get(load) == body
        and x["opcounts"].get(store) == body
    ]
    if len(candidates) != 1:
        raise ValueError("Need one exact counted store/load body")
    (region,) = candidates
    hot = instructions[region["start_index"] : region["end_index"] + 1]
    memory = [x for x in hot if x["opcode"] in (store, load, "MEMBAR")]
    stride = 3 if fence == "cta" else 2
    if len(memory) != stride * body:
        raise ValueError("Pair/fence count differs")
    edges, previous = [], None
    addresses = {}
    for offset in range(0, len(memory), stride):
        group = memory[offset : offset + stride]
        first, last = group[0], group[-1]
        if (
            first["full_opcode"] != store
            or last["full_opcode"] != load
            or any(x["predicate"] for x in group)
        ):
            raise ValueError("Need unconditional scalar32 store/load pairs")
        if fence == "cta" and (
            group[1]["full_opcode"] != "MEMBAR.CTA"
            or helper.operands(group[1])
        ):
            raise ValueError("Shared fence is not an explicit CTA MEMBAR")
        left, right = helper.operands(first), helper.operands(last)
        if len(left) != 2 or len(right) != 2:
            raise ValueError("Unproved scalar pair operands")
        address_pattern = r"\[(R[0-9]+)(?:\+(0x[0-9a-f]+))?\]"
        for name, operand in (("write", left[0]), ("read", right[1])):
            match = re.fullmatch(address_pattern, operand)
            if match is None:
                raise ValueError("Indirect/compound memory address")
            if name in addresses and addresses[name] != operand:
                raise ValueError("A pair changed its address")
            addresses[name] = operand
        if any(
            re.fullmatch(r"R[0-9]+", x) is None for x in (left[1], right[0])
        ):
            raise ValueError("Pair value is not a scalar GPR")
        if previous is not None and left[1] != previous:
            raise ValueError("Store does not consume the previous load")
        edges.append(
            dict(
                store=first,
                load=last,
                fence=group[1:-1],
                input_register=left[1],
                output_register=right[0],
            )
        )
        previous = right[0]
    if addresses["write"] == addresses["read"]:
        raise ValueError("Compiler merged the two unknown runtime addresses")
    address_words = {
        re.match(r"\[(R[0-9]+)", a)[1] for a in addresses.values()
    }
    increments = [
        x
        for x in hot
        if x["full_opcode"] == "IADD3"
        and helper.operands(x)[1:] == [previous, "0x1", "RZ"]
        and not x["predicate"]
    ]
    if len(increments) != 1:
        raise ValueError("Need one dependent endpoint increment per body")
    (increment,) = increments
    endpoint = helper.operands(increment)[0]
    if endpoint != edges[0]["input_register"]:
        raise ValueError("Increment does not close the pair recurrence")
    last_load = edges[-1]["load"]["address"]
    if increment["address"] <= last_load:
        raise ValueError("Increment precedes final dependent load")
    target_addresses = {x["address"] for x in memory + [increment]}
    transformed = copy.deepcopy(instructions)
    for x in transformed:
        if x["address"] in target_addresses:
            x["opcode"] = "COMPOSITE_TARGET"
    admin, counter, calls, terminal = helper.loop_administration(
        transformed, region, labels, "COMPOSITE_TARGET"
    )
    dominance = helper.dominators(instructions, labels, calls)
    exit_index = region["end_index"] - bool(calls)
    if (
        start not in dominance[region["start_index"]]
        or exit_index not in dominance[end]
        or any(
            i not in dominance[exit_index]
            for i in range(region["start_index"], exit_index)
        )
    ):
        raise ValueError("Control bypasses counted memory work")
    origin = helper.count_origin(
        instructions, counter, start, abi["count_constant_offset"], dominance
    )
    begin = helper.written_registers(instructions[start])
    destinations = {x["output_register"] for x in edges} | {endpoint}
    if (
        destinations & (address_words | {counter} | begin)
        or begin & address_words
        or any(
            helper.written_registers(x) & (begin | address_words) for x in hot
        )
    ):
        raise ValueError("Body clobbers address/count/start clock")
    barriers = [i for i, x in enumerate(instructions) if x["opcode"] == "BAR"]
    if len(barriers) != 1 or barriers[0] <= end:
        raise ValueError("Need one final full-CTA barrier")
    barrier = instructions[barriers[0]]
    if (
        barrier["predicate"]
        or barrier["full_opcode"]
        not in ("BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING")
        or helper.operands(barrier) != ["0x0"]
        or any(
            barriers[0] not in dominance[i]
            for i in dominance
            if instructions[i]["opcode"] == "EXIT"
        )
    ):
        raise ValueError("An exit can release warps before the final barrier")
    store_count = sum(x["opcode"] == store for x in instructions)
    load_count = sum(x["opcode"] == load for x in instructions)
    scalar_global_stores = [
        x for x in instructions if x["full_opcode"] == "STG.E"
    ]
    final_value = edges[-1]["output_register"]
    final_observations = [
        x
        for x in scalar_global_stores
        if x["address"] > instructions[end]["address"]
        and not x["predicate"]
        and len(helper.operands(x)) == 2
        and helper.operands(x)[1] == final_value
    ]
    if (
        store_count != body + 1
        or load_count != body + 1
        or len(scalar_global_stores) != 2
        or len(final_observations) != 1
    ):
        raise ValueError("Initialization/final memory inventory differs")
    (final_observation,) = final_observations
    last_load_index = next(
        index
        for index, item in enumerate(instructions)
        if item["address"] == edges[-1]["load"]["address"]
    )
    final_observation_index = instructions.index(final_observation)
    if (
        last_load_index >= end
        or end >= final_observation_index
        or any(
            final_value in helper.written_registers(item)
            for item in instructions[
                last_load_index + 1 : final_observation_index
            ]
        )
    ):
        raise ValueError("Final timed-load value is not live through output")
    return dict(
        status="counted_pairs_require_independent_native_certificate",
        region=region,
        edges=edges,
        addresses=addresses,
        endpoint_increment=increment,
        administration=admin,
        count_origin=origin,
        terminal=terminal,
        clock_instructions=[instructions[i] for i in clocks],
        final_barrier=barrier,
        final_memory_observation=dict(
            form="forwarded_final_timed_load",
            final_timed_load=edges[-1]["load"],
            live_register=final_value,
            observation_store=final_observation,
            separate_post_clock_reload=False,
            scope=(
                "The observation is not an independent post-clock "
                "memory reread"
            ),
        ),
        native_parameter_abi=abi,
        complete_instruction_count=len(instructions),
        complete_opcounts=dict(Counter(x["opcode"] for x in instructions)),
        certificate_obligations=[
            "Both addresses resolve to the same private cell at the "
            "recorded runtime offsets",
            "Poison load and all input operands complete before "
            "starting clock",
            "Final load then increment and exact expected guard dominate "
            "ending clock",
            "Every raw output word has the documented address and "
            "value source",
            "The final memory record address is the documented trace word "
            "and consumes the admitted live final timed-load value",
            "Full warp selection and every convergence token preserve "
            "final CTA barrier",
        ],
    )


def native_certificate(directory, result, request):
    """Require a retained independent whole-native proof before launch."""
    path = Path(directory) / "native_certificate.json"
    if not request.get("native_certificate_sha256"):
        raise ValueError("Ordinary/profile execution requires native review")
    if digest(path) != request["native_certificate_sha256"]:
        raise ValueError("Native review certificate changed")
    value = json.loads(path.read_text())
    if (
        value.get("kind") != "independent_store_composite_native_review"
        or value.get("status") != "PASS"
        or not value.get("reviewer")
        or value.get("reviewed_generator_sha256")
        != request["generator_sha256"]
        or value.get("reviewed_worker_sha256") != request["worker_sha256"]
    ):
        raise ValueError("Native review identity/status differs")
    for name in (
        "kernel.py",
        "kernel.cubin",
        "kernel.ptx",
        "kernel.sass",
        "native_elf.txt",
        "primitive.ptx",
    ):
        if value["artifacts"][name] != digest(Path(directory) / name):
            raise ValueError("Native review does not bind this artifact")
    if value["native_admission"] != result["native_admission"]:
        raise ValueError("Native review admission differs")
    obligations = result["native_admission"]["certificate_obligations"]
    proofs = value.get("obligations", [])
    if [x.get("claim") for x in proofs] != obligations:
        raise ValueError("Native review does not cover all obligations")
    saved_addresses = re.findall(
        r"/\*([0-9a-f]+)\*/", (Path(directory) / "kernel.sass").read_text()
    )
    instruction_addresses = {
        f"0x{int(address, 16):x}" for address in saved_addresses
    }
    if (
        len(saved_addresses)
        != result["native_admission"]["complete_instruction_count"]
        or len(instruction_addresses) != len(saved_addresses)
    ):
        raise ValueError("Native review instruction inventory differs")
    for proof in proofs:
        pcs = proof.get("native_pcs")
        if (
            not isinstance(pcs, list)
            or not pcs
            or len(set(pcs)) != len(pcs)
            or any(
                not isinstance(pc, str)
                or re.fullmatch(r"0x[0-9a-f]+", pc) is None
                or f"0x{int(pc, 16):x}" != pc
                or pc not in instruction_addresses
                for pc in pcs
            )
            or not proof.get("reasoning")
        ):
            raise ValueError("Native review lacks PC witnesses and reasoning")
    return value


def match_native_bank(prior, current, request):
    """Bind the same inputs, helpers, native code and physical geometry."""
    for key in IDENTITY_FIELDS:
        if prior["request"][key] != request[key]:
            raise ValueError("Ordinary/profile input differs: " + key)
    for key in (
        "compilation_identity",
        "resources",
        "geometry",
        "native_admission",
        "artifacts",
    ):
        if prior[key] != current[key]:
            raise ValueError(
                "Ordinary/profile native identity differs: " + key
            )


def process_ledger(directory, exit_record):
    """Rederive the exact isolated-worker command and clean child exit."""
    directory = Path(directory).resolve()
    worker_command = json.loads(
        (directory / "worker_command.json").read_text()
    )
    expected_command = [
        sys.executable,
        str((directory / "worker.py").resolve()),
    ]
    expected = dict(command=expected_command, cwd=str(directory))
    if (
        worker_command != expected
        or exit_record.get("command") != expected_command
        or exit_record.get("returncode") != 0
        or exit_record.get("forced") is not False
        or type(exit_record.get("pid")) is not int
        or exit_record["pid"] <= 0
    ):
        raise ValueError("Isolated worker execution ledger differs")
    return expected


def validate_preparation(directory):
    """Rebuild the exact source, ABI input constants and copied helpers."""
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text())
    pairs = (
        ("benchmark_source.py", "generator_sha256"),
        ("worker.py", "worker_sha256"),
        ("kernel.py", "kernel_sha256"),
        ("seeds.npy", "seeds_sha256"),
        ("arithmetic_source.py", "arithmetic_sha256"),
        ("latency_source.py", "latency_sha256"),
        ("hardware_source.py", "hardware_source_sha256"),
        ("hardware_manifest.json", "hardware_manifest_sha256"),
    )
    for name, key in pairs:
        if digest(directory / name) != request[key]:
            raise ValueError("Prepared bytes changed: " + name)
    if (
        request["arithmetic_sha256"] != ARITHMETIC_SHA
        or request["latency_sha256"] != LATENCY_SHA
        or request["hardware_source_sha256"] != HARDWARE_SHA
    ):
        raise ValueError("A frozen helper identity differs")
    source, ptx = source_text(
        request["space"], request["fence"], request["body_operations"]
    )
    if (directory / "kernel.py").read_text() != source or (
        directory / "primitive.ptx"
    ).read_text() != ptx:
        raise ValueError("Emitted source differs from declared pair protocol")
    seeds, coefficients = inputs(1024, request["body_operations"])
    saved = np.load(directory / "seeds.npy", allow_pickle=False)
    if (
        saved.dtype != np.uint32
        or not np.array_equal(saved, seeds)
        or request["coefficients"] != coefficients
        or request["block_size"] != 1024
        or request["waves"] != 2
        or request["space"] != request["operation"]
        or request["trace_steps"] != TRACE_STEPS
        or request["hardware_plan"]
        != plan(json.loads((directory / "hardware_manifest.json").read_text()))
    ):
        raise ValueError("Prepared endpoint/address/geometry inputs differ")
    return request


def load_ordinary(directory):
    """Revalidate all source, native, raw-memory and paired timing evidence."""
    directory = Path(directory).resolve()
    request = validate_preparation(directory)
    value = json.loads((directory / "result.json").read_text())
    exit_record = json.loads((directory / "process_exit.json").read_text())
    process_ledger(directory, exit_record)
    if (
        request["mode"] != "ordinary"
        or value["status"] != "ordinary_complete"
        or not value["gpu_execution"]
        or not value["kernel_compilation"]
        or value["cleanup_errors"]
        or exit_record["returncode"]
        or exit_record["forced"]
    ):
        raise ValueError("Incomplete ordinary run or cleanup")
    if set(value["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Artifact membership differs")
    for name, checksum in value["artifacts"].items():
        if digest(directory / name) != checksum:
            raise ValueError("Ordinary artifact changed: " + name)
    native_certificate(directory, value, request)
    geometry, initial = value["geometry"], value["initial_native"]
    resources = value["resources"]
    if (
        initial != value["final_native"]
        or initial["overloads"] != 1
        or initial["resident_blocks_per_sm"] != 1
        or geometry["waves"] < 2
        or geometry["grid_blocks"] != 112
        or geometry["block_size"] != 1024
        or geometry["dynamic_shared_bytes"] != 0
        or initial["geometry"] != geometry
        or initial["cubin_sha256"] != value["artifacts"]["kernel.cubin"]
        or resources["cubin_sha256"] != initial["cubin_sha256"]
        or resources["entry"] != initial["entry"]
        or geometry["resident_blocks_per_sm"] != 1
        or geometry["resident_warps_per_sm"] != 32
        or geometry["thread_capacity_blocks"] != 1
    ):
        raise ValueError("Native geometry or in-process handle differs")
    if (
        resources["static_shared_bytes"]
        != (4096 if request["space"] == "shared" else 0)
        or request["space"] == "shared"
        and resources["local_bytes_per_thread"]
        or request["space"] == "local"
        and resources["local_bytes_per_thread"] < 4
    ):
        raise ValueError("Actual memory frame differs from the protocol")
    helper = helpers(directory / "latency_source.py")
    admission = check_native(
        *parse_native(directory, resources["entry"]),
        request["space"],
        request["body_operations"],
        parameter_abi(
            (directory / "native_elf.txt").read_text(), resources["entry"]
        ),
        helper,
        request["fence"],
    )
    if (
        json.loads(json.dumps(admission)) != value["native_admission"]
        or json.loads((directory / "sass_analysis.json").read_text())
        != value["native_admission"]
    ):
        raise ValueError("Saved native admission does not rederive")
    raw = [
        json.loads(x)
        for x in (directory / "samples.jsonl").read_text().splitlines()
    ]
    measured = [x for x in raw if x["phase"] == "measurement"]
    if (
        raw != value["samples"]
        or [
            (x["block"], x["index"], x["warps"], x["multiple"])
            for x in measured
        ]
        != sample_order()
    ):
        raise ValueError("Need the complete48 mirrored measurement samples")
    seeds = np.load(directory / "seeds.npy", allow_pickle=False)
    calibration = [x for x in raw if x["phase"] == "calibration"]
    if (
        len(calibration) < 2
        or len(calibration) % 2
        or raw != calibration + measured
    ):
        raise ValueError("Need complete paired calibration before measurement")
    for index in range(len(calibration) // 2):
        pair = calibration[2 * index : 2 * index + 2]
        if (
            [x["warps"] for x in pair] != [1, 32]
            or any(
                x["block"] != -1 or x["index"] != index or x["multiple"] != 1
                for x in pair
            )
            or pair[0]["iterations"] != pair[1]["iterations"]
        ):
            raise ValueError("Calibration pair differs")
    if any(
        x["iterations"] != value["iterations"] or not duration_ok(x)
        for x in calibration[-2:]
    ):
        raise ValueError("Final calibration is not the measured common N")
    for ordinal, row in enumerate(raw):
        if row["phase"] not in ("calibration", "measurement"):
            raise ValueError("Unknown ordinary phase")
        if row["array_file"] != f"sample_{ordinal:04d}.npz":
            raise ValueError("Array order differs")
        path = directory / row["array_file"]
        if digest(path) != row["array_sha256"]:
            raise ValueError("Raw array changed")
        expected = expected_bits(
            "",
            seeds,
            request["coefficients"],
            row["iterations"] * request["body_operations"],
        )
        with np.load(path, allow_pickle=False) as arrays:
            if (
                set(arrays.files) != {"values", "trace", "seeds", "expected"}
                or arrays["seeds"].dtype != np.uint32
                or arrays["expected"].dtype != np.uint32
                or not np.array_equal(arrays["seeds"], seeds)
                or not np.array_equal(arrays["expected"], expected)
            ):
                raise ValueError("Raw runtime input differs")
            checks = validate_output(
                arrays["values"],
                expected,
                row["iterations"],
                row["warps"],
                geometry,
                request["body_operations"],
            )
            validate_memory(arrays["trace"], expected, seeds, row["warps"])
        if (
            checks != row["output_checks"]
            or row["native_before"] != initial
            or row["native_after"] != initial
        ):
            raise ValueError("Output/native evidence differs")
        milliseconds = helper.chain_milliseconds(
            checks,
            row["clocks_before"],
            row["clocks_after"],
            row["event_ms"],
            value["compilation_identity"]["gpu_uuid"],
        )
        if milliseconds != row["minimum_chain_ms_at_observed_max_clock"]:
            raise ValueError("Retained interval differs from raw clocks")
        if row["phase"] == "measurement" and (
            not duration_ok(row)
            or row["iterations"] != value["iterations"] * row["multiple"]
        ):
            raise ValueError("Measurement duration/count differs")
    value["request"] = request
    return value, dict(
        path=str(directory),
        result_sha256=digest(directory / "result.json"),
        request_sha256=digest(directory / "request.json"),
        accepted_measurements=48,
        scope="Exact composite workload",
    )


def main():
    """Prepare sources by default; native execution is always explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hardware-manifest", type=Path, required=True)
    parser.add_argument("--space", choices=("shared", "local"), required=True)
    parser.add_argument("--fence", choices=("none", "cta"), default="none")
    parser.add_argument(
        "--body-operations", type=int, choices=(33, 257), default=257
    )
    parser.add_argument("--iterations", type=int, default=32769)
    parser.add_argument("--native-certificate", type=Path)
    parser.add_argument("--ordinary-dir", type=Path)
    parser.add_argument("--profile-warps", type=int, choices=(1, 32))
    parser.add_argument("--nvdisasm")
    parser.add_argument("--cuobjdump")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--profile-multiplier", type=int, choices=(1, 2))
    args = parser.parse_args()
    if not (
        bool(args.profile_multiplier)
        == bool(args.profile_warps)
        == bool(args.ordinary_dir)
    ):
        parser.error("Profile count/population/ordinary reference are paired")
    if (
        args.execute or args.profile_multiplier
    ) and not args.native_certificate:
        parser.error("An independent native-review certificate is required")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, primitive = source_text(
        args.space, args.fence, args.body_operations
    )
    (output / "kernel.py").write_text(source)
    (output / "primitive.ptx").write_text(primitive)
    copies = {
        "benchmark_source.py": SCRIPT,
        "worker.py": SCRIPT.with_name("store_composite_worker.py"),
        "arithmetic_source.py": SCRIPT.with_name(
            "arithmetic_service_probe.py"
        ),
        "latency_source.py": SCRIPT.with_name("latency_probe.py"),
        "hardware_source.py": SCRIPT.with_name("hardware_probes.py"),
        "hardware_manifest.json": args.hardware_manifest,
    }
    for name, path in copies.items():
        (output / name).write_bytes(path.read_bytes())
    hardware_plan = plan(json.loads(args.hardware_manifest.read_text()))
    seeds, coefficients = inputs(1024, args.body_operations)
    np.save(output / "seeds.npy", seeds, allow_pickle=False)
    certificate = None
    if args.native_certificate:
        (output / "native_certificate.json").write_bytes(
            args.native_certificate.read_bytes()
        )
        certificate = digest(output / "native_certificate.json")
    request = dict(
        schema=1,
        research_root=str(REPO),
        mode="profile"
        if args.profile_multiplier
        else "ordinary"
        if args.execute
        else "compile_only"
        if args.compile_only
        else "source_only",
        operation=args.space,
        space=args.space,
        fence=args.fence,
        block_size=1024,
        waves=2,
        iterations=args.iterations,
        body_operations=args.body_operations,
        trace_steps=TRACE_STEPS,
        coefficients=coefficients,
        hardware_plan=hardware_plan,
        profile_multiplier=args.profile_multiplier,
        profile_warps=args.profile_warps,
        ordinary_dir=str(args.ordinary_dir.resolve())
        if args.ordinary_dir
        else None,
        native_certificate_sha256=certificate,
        nvdisasm=args.nvdisasm,
        cuobjdump=args.cuobjdump,
    )
    for name, key in (
        ("benchmark_source.py", "generator_sha256"),
        ("worker.py", "worker_sha256"),
        ("kernel.py", "kernel_sha256"),
        ("seeds.npy", "seeds_sha256"),
        ("arithmetic_source.py", "arithmetic_sha256"),
        ("latency_source.py", "latency_sha256"),
        ("hardware_source.py", "hardware_source_sha256"),
        ("hardware_manifest.json", "hardware_manifest_sha256"),
    ):
        request[key] = digest(output / name)
    write_json(output / "request.json", request)
    compile((output / "worker.py").read_text(), "<store-worker>", "exec")
    validate_preparation(output)
    if request["mode"] == "source_only":
        print(
            json.dumps(
                dict(
                    status="prepared",
                    out=str(output),
                    native_compilation=False,
                )
            )
        )
        return
    command = [sys.executable, str(output / "worker.py")]
    write_json(
        output / "worker_command.json", dict(command=command, cwd=str(output))
    )
    forced = False
    with (
        (output / "worker.stdout.log").open("w") as stdout,
        (output / "worker.stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command, cwd=output, stdout=stdout, stderr=stderr
        )
        try:
            process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            forced = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            write_json(
                output / "process_exit.json",
                dict(
                    returncode=process.returncode,
                    forced=forced,
                    command=command,
                    pid=process.pid,
                ),
            )
    if process.returncode or forced:
        raise SystemExit(process.returncode or 1)
    print(json.dumps(dict(status="worker_complete", out=str(output))))


if __name__ == "__main__":
    main()
