"""Prepare direct-address dependent loads; native work is explicit.

Default execution writes inspectable source, ring topology and a request.
Only --compile-only, --execute or --profile-multiplier starts the isolated
native worker. No hardware latency is assigned from another GPU model.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
HARDWARE_SHA = (
    "758b54944fe2bd88b6778df59277b89b8ea66d0107bf1feea3825dfd25ea898e"
)
FOOTPRINTS = (
    "l1_quarter",
    "l1_double",
    "l2_quarter",
    "l2_double",
    "shared8",
    "shared16",
    "shared32",
)
IDENTITY_FIELDS = (
    "space",
    "cache",
    "footprint",
    "footprint_bytes",
    "stride_bytes",
    "body_loads",
    "seed",
    "block_size",
    "waves",
    "nodes",
    "topology_sha256",
    "cycle_sha256",
    "kernel_source_sha256",
    "generator_sha256",
    "worker_sha256",
    "hardware_manifest_sha256",
)


def digest(path):
    """Hash retained file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write an inspectable record with explicit real newlines."""
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def plan(manifest, footprint):
    """Derive a byte window from queried limits and published Ada L1."""
    attrs = manifest["device_attributes"]
    if manifest["compute_capability"] != [8, 9]:
        raise ValueError("This bounded protocol requires queried SM89")
    l2 = attrs["L2_CACHE_SIZE"]
    if (
        attrs["MAX_THREADS_PER_MULTIPROCESSOR"]
        // attrs["MAX_THREADS_PER_BLOCK"]
        != 1
    ):
        raise ValueError(
            "Max-size blocks must exclude a second block by threads"
        )
    unified = 128 * 1024
    choices = dict(
        l1_quarter=unified // 4,
        l1_double=unified * 2,
        l2_quarter=l2 // 4,
        l2_double=l2 * 2,
        shared8=8 * 1024,
        shared16=16 * 1024,
        shared32=32 * 1024,
    )
    size = choices[footprint]
    if size % 32 or l2 <= 0:
        raise ValueError("A complete 32-byte-sector window is required")
    if (
        footprint.startswith("shared")
        and size > (attrs["MAX_SHARED_MEMORY_PER_BLOCK"])
    ):
        raise ValueError("Shared window exceeds queried static block limit")
    return dict(
        footprint_bytes=size,
        stride_bytes=32,
        nodes=size // 32,
        block_size=attrs["MAX_THREADS_PER_BLOCK"],
        max_threads_per_sm=attrs["MAX_THREADS_PER_MULTIPROCESSOR"],
        thread_capacity_blocks=1,
        queried_l2_bytes=l2,
        published_unified_l1_texture_shared_bytes=unified,
        shared_supported_kib=[0, 8, 16, 32, 64, 100],
        capacity_source=(
            "https://docs.nvidia.com/cuda/ada-tuning-guide/index.html"
            "#unified-shared-memory-l1-texture-cache"
        ),
        stride_source=(
            "https://docs.nvidia.com/nsight-compute/ProfilingGuide/"
            "index.html#memory-workload-analysis"
        ),
        scope=(
            "Global grid shares one device address window. Shared has one "
            "window per block. Nominal capacity does not assert cache hits."
        ),
    )


def topology(nodes, stride, seed):
    """Build one reproducible randomized cycle spanning every sector."""
    order = np.random.default_rng(seed).permutation(nodes).astype(np.uint32)
    offsets = np.empty(nodes, dtype=np.uint32)
    offsets[order] = np.roll(order, -1) * np.uint32(stride)
    return offsets, order


def validate_topology(offsets, order, nodes, stride):
    """Prove exact permutation membership and every directed cycle edge."""
    if (
        offsets.dtype != np.uint32
        or order.dtype != np.uint32
        or offsets.shape != (nodes,)
        or order.shape != (nodes,)
        or not np.array_equal(
            np.sort(order), np.arange(nodes, dtype=np.uint32)
        )
        or not np.array_equal(offsets[order], np.roll(order, -1) * stride)
    ):
        raise ValueError("Saved ring is not the declared single cycle")


def final_offset(order, loads, stride):
    """Traverse the validated cycle algebraically from byte offset zero."""
    start = int(np.flatnonzero(order == 0)[0])
    return int(order[(start + loads) % len(order)]) * stride


def valid_repeats(value, body, nodes):
    """Require distinct nonzero N/2N final offsets, with bounded counters."""
    return (
        0 < value < 2**30
        and body * value > nodes
        and (body * value) % nodes != 0
        and (body * value * 2) % nodes != 0
    )


def chain_milliseconds(checks, before, after, event_ms, expected_uuid):
    """Derive finite duration from raw cycles and unit-qualified clocks."""
    if not math.isfinite(event_ms) or event_ms <= 0:
        raise ValueError("Event duration is not finite and positive")
    clocks = []
    for snapshot in (before, after):
        if (
            snapshot.get("returncode") != 0
            or snapshot.get("units", {}).get("clocks.current.sm") != "MHz"
            or len(snapshot.get("devices", [])) != 1
        ):
            raise ValueError("Need one GPU's successful MHz clock query")
        device = snapshot["devices"][0]
        value = float(device["clocks.current.sm"])
        if (
            device["uuid"] != expected_uuid
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Clock identity/value differs")
        clocks.append(value)
    cycles = checks["minimum_cycles"]
    if not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("Raw cycle duration is not a positive integer")
    return cycles / (max(clocks) * 1000)


def source_text(space, cache, footprint_bytes, body):
    """Emit one side-effecting PTX region with raw scalar addresses."""
    pointer = "u64" if space == "global" else "u32"
    load = (
        f"ld.global.{cache}.u64 p, [p];"
        if space == "global"
        else ("ld.shared.u32 p, [p];")
    )
    ptx = [
        "{",
        ".reg .pred inactive, again, bad;",
        ".reg .u32 tid, nt, sm0, sm1, count, low, base, i, val;",
        ".reg .u64 address, begin, end, delta, offset, total;",
        f".reg .{pointer} p;",
    ]
    if space == "shared":
        ptx.append(f".shared .align 4 .b8 latency_ring[{footprint_bytes}];")
    ptx += ["mov.u32 tid, %tid.x;", "mov.u32 nt, %ntid.x;"]
    if space == "shared":
        ptx += [
            "mov.u32 base, latency_ring;",
            "mov.u32 i, tid;",
            "copy_ring:",
            "mul.wide.u32 address, i, 4;",
            "add.u64 address, address, $1;",
            "ld.global.u32 val, [address];",
            "add.u32 val, val, base;",
            "mad.lo.u32 low, i, 4, base;",
            "st.shared.u32 [low], val;",
            "add.u32 i, i, nt;",
            f"setp.lt.u32 again, i, {footprint_bytes // 4};",
            "@again bra copy_ring;",
            "bar.sync 0;",
        ]
    ptx += [
        "setp.ne.u32 inactive, tid, 0;",
        "@inactive bra finished;",
        "mov.u32 sm0, %smid;",
        "ld.global.u32 low, [$3];",
        "cvt.u64.u32 address, low;"
        if space == "global"
        else "mov.u32 p, base;",
        "add.u64 p, $1, address;"
        if space == "global"
        else "add.u32 p, p, low;",
        "mov.u32 count, $5;",
        "prime_ring:",
        load,
        "add.u32 count, count, -1;",
        "setp.ne.u32 again, count, 0;",
        "@again bra.uni prime_ring;",
        "cvt.u32.u64 low, p;" if space == "global" else "mov.u32 low, p;",
        "setp.eq.u32 bad, low, 0xffffffff;",
        "@bad bra invalid_pointer;",
        "mov.u32 count, $4;",
        "mov.u64 begin, %clock64;",
        "timed_chain:",
    ]
    ptx += [load] * body
    ptx += [
        "add.u32 count, count, -1;",
        "setp.ne.u32 again, count, 0;",
        "@again bra.uni timed_chain;",
        "cvt.u32.u64 low, p;" if space == "global" else "mov.u32 low, p;",
        "setp.eq.u32 bad, low, 0xffffffff;",
        "@bad bra invalid_pointer;",
        "mov.u64 end, %clock64;",
        "sub.u64 delta, end, begin;",
        "sub.u64 offset, p, $1;"
        if space == "global"
        else ("sub.u32 low, p, base;"),
    ]
    if space == "shared":
        ptx.append("cvt.u64.u32 offset, low;")
    ptx += [
        "mov.u32 sm1, %smid;",
        "mul.wide.u32 total, $4, " + str(body) + ";",
        "st.global.u64 [$2+0], begin;",
        "st.global.u64 [$2+8], end;",
        "st.global.u64 [$2+16], delta;",
        "st.global.u64 [$2+24], offset;",
        "cvt.u64.u32 address, sm0;",
        "st.global.u64 [$2+32], address;",
        "cvt.u64.u32 address, sm1;",
        "st.global.u64 [$2+40], address;",
        "mov.u64 address, 1;",
        "st.global.u64 [$2+48], address;",
        "st.global.u64 [$2+56], total;",
        "bra finished;",
        "invalid_pointer:",
        "mov.u64 address, 0;",
        "st.global.u64 [$2+48], address;",
        "finished:",
        "bar.sync 0;",
        "mov.u32 $0, 0;",
        "}",
    ]
    asm = "\n".join(ptx)
    escaped = asm.replace("\\", "\\5C").replace('"', "\\22")
    escaped = escaped.replace("\n", "\\0A")
    intrinsic = (
        "func.func private @direct_chain(%ring: i64, %out: i64, %start: i64, "
        "%count: i32, %nodes: i32) -> i32 attributes {always_inline} {\n"
        '  %answer = "llvm.inline_asm"'
        "(%ring, %out, %start, %count, %nodes) {\n"
        '    asm_string = "' + escaped + '",\n'
        '    constraints = "=r,l,l,l,r,r,~{memory}", has_side_effects\n'
        "  } : (i64, i64, i64, i32, i32) -> i32\n"
        "  return %answer : i32\n}\n"
    )
    source = (
        "from numpy import uint32, uint64\n"
        "from cubie.cuda_simsafe import cuda\n\n"
        f"direct_chain = cuda.intrin.define({intrinsic!r})\n\n"
        "def probe(ring, output, starts, iterations, nodes):\n"
        "    address = output + uint64(cuda.blockIdx.x) * uint64(64)\n"
        "    start = starts + uint64(cuda.blockIdx.x) * uint64(4)\n"
        "    direct_chain(ring, address, start, "
        "uint32(iterations), uint32(nodes))\n"
    )
    compile(source, "<latency-kernel>", "exec")
    return source, asm


def operands(instruction):
    """Split the bounded scalar SASS operand grammar."""
    if " " not in instruction["text"]:
        return []
    return [
        item.strip()
        for item in instruction["text"].split(None, 1)[1].split(",")
    ]


def parameter_abi(elf_text, entry):
    """Bind the five scalar parameters to cuobjdump's native ABI output."""
    marker = ".nv.info." + entry
    if elf_text.count(marker) != 2:
        # One section-table occurrence and one expanded metadata section.
        raise ValueError("Unexpected native parameter metadata membership")
    section = elf_text.rsplit(marker, 1)[1]
    pairs = re.findall(
        r"Ordinal\s*:\s*(0x[0-9a-f]+)\s+Offset\s*:\s*(0x[0-9a-f]+)"
        r"\s+Size\s*:\s*(0x[0-9a-f]+)",
        section,
    )
    observed = sorted(tuple(int(value, 16) for value in row) for row in pairs)
    if observed != [(0, 0, 8), (1, 8, 8), (2, 16, 8), (3, 24, 4), (4, 28, 4)]:
        raise ValueError("Native parameter layout is not the five scalar ABI")
    cbank = re.search(
        r"EIATTR_PARAM_CBANK\s+Format:\s+EIFMT_SVAL\s+Value:\s+"
        r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)",
        section,
    )
    if cbank is None:
        raise ValueError("Missing native parameter constant-bank descriptor")
    packed = int(cbank.group(2), 16)
    base, size = packed & 0xFFFF, packed >> 16
    if size != 32:
        raise ValueError("Native parameter extent differs")
    return dict(
        parameters=observed,
        constant_base=base,
        parameter_bytes=size,
        count_constant_offset=base + 24,
    )


def written_registers(item):
    """Include implicit high words for bounded native register writes."""
    if " " not in item["text"]:
        return set()
    destination = operands(item)[0]
    if re.fullmatch(r"U?R[0-9]+", destination) is None:
        return set()
    result = {destination}
    if (
        item["opcode"] == "CS2R"
        or "64" in item["full_opcode"].split(".")
        or "WIDE" in item["full_opcode"].split(".")
    ):
        prefix = "UR" if destination.startswith("UR") else "R"
        result.add(prefix + str(int(destination[len(prefix) :]) + 1))
    return result


def control_target(item, labels):
    """Resolve one direct native control target to an instruction index."""
    match = re.fullmatch(
        re.escape(item["full_opcode"]) + r" `\((\.L[\w.$]+)\)",
        item["text"],
    )
    if match is None or match.group(1) not in labels:
        raise ValueError("Unresolved native control edge")
    return labels[match.group(1)]


def terminal_tail(instructions, labels, first):
    """Prove a tail target reaches EXIT through forward direct paths."""
    pending = [first]
    visited = set()
    exits = set()
    while pending:
        index = pending.pop()
        if index in visited:
            continue
        if index >= len(instructions):
            raise ValueError("Tail path falls off native code")
        visited.add(index)
        item = instructions[index]
        if item["opcode"] in ("CALL", "RET", "BRX", "JMP", "JMX", "BRXU"):
            raise ValueError("Tail exit contains a call/return/indirect edge")
        if item["opcode"] == "EXIT":
            if item["predicate"]:
                raise ValueError("Tail EXIT is predicated")
            exits.add(index)
            continue
        if item["opcode"] == "BRA":
            target = control_target(item, labels)
            if target <= index:
                raise ValueError("Tail exit returns to an earlier address")
            pending.append(target)
            if not item["predicate"]:
                continue
        pending.append(index + 1)
    if len(exits) != 1:
        raise ValueError("Tail exit does not have one terminal EXIT")
    return dict(
        start_index=first,
        visited_indices=sorted(visited),
        exit_index=next(iter(exits)),
    )


def dominators(instructions, labels, tail_calls=None):
    """Prove reaching definitions over the exact direct-control graph."""
    tail_calls = tail_calls or {}
    successors = {}
    for index, item in enumerate(instructions):
        next_index = [index + 1] if index + 1 < len(instructions) else []
        if item["opcode"] == "EXIT":
            successors[index] = next_index if item["predicate"] else []
        elif item["opcode"] == "BRA" or index in tail_calls:
            successors[index] = [control_target(item, labels)]
            if index in tail_calls and successors[index] != [
                tail_calls[index]
            ]:
                raise ValueError("Tail-call proof/control mismatch")
            if item["predicate"]:
                successors[index] += next_index
        else:
            successors[index] = next_index
    reachable = set()
    pending = [0]
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        if node not in tail_calls and instructions[node]["opcode"] in (
            "CALL",
            "RET",
            "BRX",
            "JMP",
            "JMX",
            "BRXU",
        ):
            raise ValueError("Unproved reachable call/indirect control")
        reachable.add(node)
        pending.extend(successors[node])
    predecessors = {node: set() for node in reachable}
    for node in reachable:
        for target in successors[node]:
            if target in reachable:
                predecessors[target].add(node)
    result = {node: set(reachable) for node in reachable}
    result[0] = {0}
    changed = True
    while changed:
        changed = False
        for node in sorted(reachable - {0}):
            common = set.intersection(*(result[p] for p in predecessors[node]))
            value = common | {node}
            if value != result[node]:
                result[node] = value
                changed = True
    return result


def count_origin(instructions, register, before, expected_offset, dominance):
    """Trace only exact parameter loads/register copies into loop count."""
    chain = []
    while True:
        writes = []
        for index, item in enumerate(instructions[:before]):
            if register in written_registers(item):
                writes.append((index, item))
        if not writes:
            raise ValueError("Counter has no proved reaching definition")
        index, item = writes[-1]
        if index not in dominance[before]:
            raise ValueError("Counter definition does not dominate its use")
        if item["predicate"]:
            raise ValueError("Counter initialization is predicated")
        parts = operands(item)
        if parts[0] != register or written_registers(item) != {register}:
            raise ValueError("Counter origin is not an explicit scalar write")
        if item["opcode"] in ("MOV", "UMOV", "ULDC") and len(parts) == 2:
            source = parts[1]
        elif item["full_opcode"] == "IMAD.MOV.U32" and parts[1:3] == [
            "RZ",
            "RZ",
        ]:
            source = parts[3]
        else:
            raise ValueError("Counter has unproved native initialization")
        chain.append(item)
        constant = re.fullmatch(r"c\[0x0\]\[(0x[0-9a-f]+)\]", source)
        if constant:
            if int(constant.group(1), 16) != expected_offset:
                raise ValueError("Counter reads a different native parameter")
            return chain
        if not re.fullmatch(r"U?R[0-9]+", source):
            raise ValueError("Counter copy does not reach the parameter bank")
        register, before = source, index


def direct_load(instruction, space):
    """Describe one exact-width direct load without indexed addressing."""
    op = "LDG" if space == "global" else "LDS"
    if instruction["opcode"] != op or instruction["predicate"]:
        raise ValueError("Target load is predicated or in another space")
    parts = operands(instruction)
    if len(parts) != 2 or not re.fullmatch(r"R[0-9]+", parts[0]):
        raise ValueError("Unexpected target load operand grammar")
    suffix = r"(?:\.64)?" if space == "global" else ""
    address = re.fullmatch(r"\[(R[0-9]+)" + suffix + r"\]", parts[1])
    if address is None:
        raise ValueError("Target load contains address arithmetic")
    if space == "global" and not re.fullmatch(
        r"LDG\.E\.64(?:\.STRONG\.(?:SM|GPU|SYS))?",
        instruction["full_opcode"],
    ):
        raise ValueError("Global pointer load is not a bounded 64-bit form")
    if space == "shared" and instruction["full_opcode"] != "LDS":
        raise ValueError("Shared pointer load is not the exact 32-bit form")
    width = 2 if space == "global" else 1
    return dict(
        instruction_address=instruction["address"],
        destination=parts[0],
        address_register=address.group(1),
        result_words=[f"R{int(parts[0][1:]) + word}" for word in range(width)],
        address_words=[
            f"R{int(address.group(1)[1:]) + word}" for word in range(width)
        ],
    )


def load_chain(loads, space, transport=None):
    """Prove every result supplies the next address, including backedge."""
    edges = [direct_load(item, space) for item in loads]
    if len({item["full_opcode"] for item in loads}) != 1:
        raise ValueError("The native pointer-load/cache opcode changes")
    previous = edges[0]
    for current in edges[1:]:
        if current["address_words"] != previous["result_words"]:
            raise ValueError("Dependent load edge differs")
        previous = current
    if transport is None:
        if edges[0]["address_words"] != edges[-1]["result_words"]:
            raise ValueError("Direct loop-carried pointer tie differs")
    elif transport != dict(
        zip(edges[0]["address_words"], edges[-1]["result_words"])
    ):
        raise ValueError("Transported loop-carried pointer words differ")
    return edges


def pointer_entry(body, space, target):
    """Prove the observed global loop-entry pair copies and fixed work."""
    loads = [item for item in body if item["opcode"] == target]
    first, last = direct_load(loads[0], space), direct_load(loads[-1], space)
    prefix = body[: body.index(loads[0])]
    extras = [
        item
        for item in prefix
        if item["opcode"] in ("MOV", "IMAD", "YIELD", "ULDC")
    ]
    if not extras:
        return load_chain(loads, space), dict(instructions=[], mapping={})
    if space != "global" or len(extras) != 4:
        raise ValueError("Unproved pointer-entry administration")
    copies = [item for item in extras if item["opcode"] in ("MOV", "IMAD")]
    yields = [item for item in extras if item["opcode"] == "YIELD"]
    uniform = [item for item in extras if item["opcode"] == "ULDC"]
    if (
        len(copies) != 2
        or len(yields) != 1
        or len(uniform) != 1
        or any(item["predicate"] for item in extras)
        or yields[0]["full_opcode"] != "YIELD"
        or operands(yields[0])
        or uniform[0]["full_opcode"] != "ULDC.64"
        or operands(uniform[0]) != ["UR4", "c[0x0][0x118]"]
    ):
        raise ValueError("Global entry is not two copies/YIELD/fixed ULDC")
    mapping = {}
    for item in copies:
        args = operands(item)
        if item["full_opcode"] == "MOV" and len(args) == 2:
            destination, source = args
        elif (
            item["full_opcode"] == "IMAD.MOV.U32"
            and len(args) == 4
            and args[1:3] == ["RZ", "RZ"]
        ):
            destination, source = args[0], args[3]
        else:
            raise ValueError("Pointer transport is not a scalar register copy")
        if (
            re.fullmatch(r"R[0-9]+", destination) is None
            or re.fullmatch(r"R[0-9]+", source) is None
            or destination in mapping
            or written_registers(item) != {destination}
        ):
            raise ValueError("Pointer transport repeats or changes a word")
        mapping[destination] = source
    if set(mapping) & set(last["result_words"]):
        raise ValueError("Entry copies overwrite the loop-carried input pair")
    expected = dict(zip(first["address_words"], last["result_words"]))
    if mapping != expected:
        raise ValueError("Entry copies do not preserve the low/high pointer")
    for item in prefix:
        if item not in copies and written_registers(item) & (
            set(mapping) | set(last["result_words"])
        ):
            raise ValueError("Entry work clobbers a pointer transport word")
    return load_chain(loads, space, mapping), dict(
        instructions=extras,
        mapping=mapping,
        scalar_copies=copies,
        yield_instruction=yields[0],
        uniform_constant_load=uniform[0],
        scope="Per-loop native work; no intrinsic or zero-cost assumption",
    )


def loop_administration(instructions, region, labels, target, entry=None):
    """Prove decrement/test and conditional or terminal-tail backedge."""
    body = instructions[region["start_index"] : region["end_index"] + 1]
    excluded = entry["instructions"] if entry else []
    admin = [
        item
        for item in body
        if item["opcode"] != target and item not in excluded
    ]
    if (
        len(admin) not in (3, 4)
        or admin[0]["full_opcode"] not in ("IADD3", "UIADD3")
        or admin[1]["opcode"] != "ISETP"
        or any(item["predicate"] for item in admin[:2])
        or body[-1] != admin[-1]
        or admin[-1]["full_opcode"] != "BRA"
        or control_target(admin[-1], labels) != region["start_index"]
        or admin[0]["address"] >= admin[1]["address"]
    ):
        raise ValueError("Loop administration is not decrement/test/backedge")
    dec, pred = operands(admin[0]), operands(admin[1])
    zero = "URZ" if dec[0].startswith("UR") else "RZ"
    if (
        len(dec) != 4
        or re.fullmatch(r"U?R[0-9]+", dec[0]) is None
        or dec[0] != dec[1]
        or dec[2:] != ["-0x1", zero]
        or admin[1]["full_opcode"] not in ("ISETP.NE.U32.AND", "ISETP.NE.AND")
        or len(pred) != 5
        or re.fullmatch(r"P[0-6]", pred[0]) is None
        or pred[1] != "PT"
        or pred[4] != "PT"
        or pred[2:4] not in ([dec[0], "RZ"], ["RZ", dec[0]])
    ):
        raise ValueError("Loop counter/control operands differ")
    if any(dec[0] in written_registers(item) for item in excluded):
        raise ValueError("Pointer-entry administration clobbers loop count")
    calls = {}
    proof = None
    if len(admin) == 3:
        if admin[-1]["predicate"] != "@" + pred[0]:
            raise ValueError("Backedge does not test nonzero count")
    else:
        call, back = admin[-2:]
        if (
            body[-2:] != [call, back]
            or call["full_opcode"] != "CALL.REL.NOINC"
            or call["predicate"] != "@!" + pred[0]
            or back["predicate"]
            or control_target(call, labels) != region["end_index"] + 1
        ):
            raise ValueError("Unproved terminal CALL/backedge form")
        first = region["end_index"] + 1
        proof = terminal_tail(instructions, labels, first)
        calls[region["end_index"] - 1] = first
    if region["predicate"] != admin[-1]["predicate"]:
        raise ValueError("Parser/backedge predicate mismatch")
    return admin, dec[0], calls, proof


def pointer_guard(tail, pointer, labels, clock_index, clock):
    """Prove the final pointer decides whether a clock can execute."""
    if len(tail) not in (1, 2) or tail[0]["opcode"] != "ISETP":
        raise ValueError("Clock lacks the exact final-pointer guard")
    guard = tail[0]
    args = operands(guard)
    if (
        guard["predicate"]
        or len(args) != 5
        or args[1:] != ["PT", pointer, "-0x1", "PT"]
        or guard["full_opcode"]
        not in (
            "ISETP.EQ.U32.AND",
            "ISETP.NE.U32.AND",
            "ISETP.EQ.AND",
            "ISETP.NE.AND",
        )
    ):
        raise ValueError("Clock guard does not consume final pointer bits")
    equality = guard["full_opcode"].startswith("ISETP.EQ")
    if len(tail) == 1:
        good = "@" + ("!" if equality else "") + args[0]
        if clock["predicate"] != good:
            raise ValueError("Clock predicate does not wait for the pointer")
    else:
        branch = tail[1]
        bad = "@" + ("" if equality else "!") + args[0]
        if (
            branch["full_opcode"] != "BRA"
            or branch["predicate"] != bad
            or control_target(branch, labels) <= clock_index
            or clock["predicate"]
        ):
            raise ValueError("Pointer guard does not dominate the clock")


def prime_completion(
    instructions,
    loops,
    labels,
    dominance,
    start,
    space,
    pointer,
    target_opcode,
    abi,
    measured_origin,
    body_loads,
):
    """Prove the complete priming traversal finishes before the first clock."""
    target = "LDG" if space == "global" else "LDS"
    candidates = [
        loop
        for loop in loops
        if loop["end_index"] < start and loop["opcounts"].get(target) == 1
    ]
    if len(candidates) != 1:
        raise ValueError("Need one proved direct-load priming loop")
    loop = candidates[0]
    body = instructions[loop["start_index"] : loop["end_index"] + 1]
    loads = [item for item in body if item["opcode"] == target]
    edges, entry = pointer_entry(body, space, target)
    if (
        edges[-1]["destination"] != pointer
        or loads[0]["full_opcode"] != target_opcode
        or len(body) != 4 + len(entry["instructions"])
        or loop["end_index"] not in dominance[start]
        or any(
            index not in dominance[loop["end_index"]]
            for index in range(loop["start_index"], loop["end_index"])
        )
    ):
        raise ValueError("Priming pointer/path differs from timed chain")
    admin, counter, calls, _ = loop_administration(
        instructions, loop, labels, target, entry
    )
    if calls or counter in (
        set(edges[0]["address_words"]) | set(edges[-1]["result_words"])
    ):
        raise ValueError("Priming count clobbers pointer or uses a tail call")
    origin = count_origin(
        instructions,
        counter,
        loop["start_index"],
        abi["count_constant_offset"] + 4,
        dominance,
    )
    tail = instructions[loop["end_index"] + 1 : start]
    pointer_guard(tail[:2], pointer, labels, start, instructions[start])
    copies = tail[2:]
    auxiliary = []
    for item in copies:
        destination = operands(item)[0]
        if written_registers(item) & written_registers(loads[0]):
            raise ValueError("Priming tail clobbers the final pointer")
        index = instructions.index(item)
        chain = count_origin(
            instructions,
            destination,
            index + 1,
            abi["count_constant_offset"],
            dominance,
        )
        if chain[0] != item:
            raise ValueError("Unproved runtime-count copy before clock")
        if item not in measured_origin:
            # One compiler copy retains the original repeat count for the
            # emitted uint64 total = repeats * body output calculation.
            users = []
            for value in instructions[start:]:
                if re.search(
                    r"\b" + destination + r"\b", ",".join(operands(value)[1:])
                ):
                    users.append(value)
                # A wide multiply may reuse the scalar count's register
                # for the resulting output pair. Later reads then use
                # that product, not the original parameter value.
                if destination in written_registers(value):
                    break
            if (
                len(users) != 1
                or users[0]["full_opcode"] != "IMAD.WIDE.U32"
                or users[0]["predicate"]
                or operands(users[0])[1:]
                != [destination, hex(body_loads), "RZ"]
                or users[0]["address"]
                <= max(
                    value["address"]
                    for value in instructions
                    if value["opcode"] == "CS2R"
                )
                or any(
                    destination in written_registers(value)
                    for value in instructions[
                        start : instructions.index(users[0])
                    ]
                )
            ):
                raise ValueError("Auxiliary count copy has unproved use")
            auxiliary.append(dict(initialization=chain, output_use=users[0]))
    if any(item not in copies for item in measured_origin):
        raise ValueError("Measured count is not initialized after priming")
    if len(auxiliary) > 1:
        raise ValueError("Multiple unproved auxiliary count outputs")
    return dict(
        region=loop,
        loads=loads,
        administration=admin,
        count_initialization=origin,
        completion_guard=tail[:2],
        output_count_copies=auxiliary,
        pointer_entry=entry,
    )


def check_native(instructions, loops, labels, space, body_loads, abi):
    """Admit direct pointer dataflow, bounded controls and clock endpoints."""
    clocks = [
        index
        for index, item in enumerate(instructions)
        if item["opcode"] == "CS2R" and "SR_CLOCKLO" in item["text"]
    ]
    if len(clocks) != 2 or instructions[clocks[0]]["predicate"]:
        raise ValueError("Need exactly two 64-bit clock reads")
    start, end = clocks
    for index in clocks:
        if (
            instructions[index]["full_opcode"] != "CS2R"
            or len(operands(instructions[index])) != 2
            or re.fullmatch(r"R[0-9]+", operands(instructions[index])[0])
            is None
            or operands(instructions[index])[1] != "SR_CLOCKLO"
            or len(written_registers(instructions[index])) != 2
        ):
            raise ValueError("Unexpected clock operand/register-pair form")
    target = "LDG" if space == "global" else "LDS"
    candidates = [
        loop
        for loop in loops
        if start < loop["start_index"]
        and loop["end_index"] < end
        and loop["opcounts"].get(target) == body_loads
    ]
    if len(candidates) != 1:
        raise ValueError("Need one exact repeated chain between clock reads")
    region = candidates[0]
    hot = instructions[region["start_index"] : region["end_index"] + 1]
    loads = [item for item in hot if item["opcode"] == target]
    edges, entry = pointer_entry(hot, space, target)
    pointer = edges[-1]["destination"]
    words = set().union(*(set(edge["result_words"]) for edge in edges))
    admin, counter, calls, terminal = loop_administration(
        instructions, region, labels, target, entry
    )
    if counter in words:
        raise ValueError("Loop counter clobbers a pointer-chain word")
    beginning = written_registers(instructions[start])
    if beginning & (set(edges[-1]["result_words"]) | {counter}):
        raise ValueError("Starting clock clobbers initial pointer or count")
    if any(written_registers(item) & beginning for item in hot):
        raise ValueError("Measured work clobbers the starting timestamp")
    if written_registers(instructions[end]) & (
        beginning | set(edges[-1]["result_words"])
    ):
        raise ValueError("Ending clock clobbers start timestamp/final pointer")
    dominance = dominators(instructions, labels, calls)
    if start not in dominance or end not in dominance:
        raise ValueError("Clock reads are not reachable from kernel entry")
    if start not in dominance[region["start_index"]]:
        raise ValueError("Starting clock does not dominate the measured chain")
    # Every instruction in this straight-line body executes before the
    # forward terminal exit or conditional backedge. No side entry may
    # bypass a load, the decrement, or the predicate reaching its branch.
    exit_index = region["end_index"] - (1 if calls else 0)
    if any(
        index not in dominance[exit_index]
        for index in range(region["start_index"], exit_index)
    ):
        raise ValueError("A control edge bypasses measured chain work")
    if exit_index not in dominance[end]:
        raise ValueError("Ending clock bypasses the measured loop exit")
    if instructions[start + 1 : region["start_index"]]:
        raise ValueError("Unproved work between starting clock and first load")
    origin = count_origin(
        instructions, counter, start, abi["count_constant_offset"], dominance
    )
    priming = prime_completion(
        instructions,
        loops,
        labels,
        dominance,
        start,
        space,
        pointer,
        loads[0]["full_opcode"],
        abi,
        origin,
        body_loads,
    )
    tail = instructions[region["end_index"] + 1 : end]
    pointer_guard(
        tail, edges[-1]["destination"], labels, end, instructions[end]
    )
    barriers = [
        index
        for index, item in enumerate(instructions)
        if index > end and item["opcode"] == "BAR"
    ]
    if len(barriers) != 1:
        raise ValueError("Need one final CTA barrier after measured interval")
    barrier_index = barriers[0]
    barrier = instructions[barrier_index]
    if any(
        item["predicate"]
        or item["full_opcode"] not in ("BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING")
        or operands(item) != ["0x0"]
        for item in instructions
        if item["opcode"] == "BAR"
    ):
        raise ValueError("Unproved full-CTA synchronization barrier")
    if (
        barrier["predicate"]
        or barrier["full_opcode"]
        not in ("BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING")
        or operands(barrier) != ["0x0"]
        or any(
            barrier_index not in dominance[index]
            for index in dominance
            if instructions[index]["opcode"] == "EXIT"
        )
    ):
        raise ValueError("Final uniform CTA barrier does not hold every warp")
    convergence = []
    matched_syncs = set()
    for index, item in enumerate(instructions):
        if item["opcode"] != "BSSY":
            continue
        match = re.fullmatch(
            r"BSSY (B[0-9]+), `\((\.L[\w.$]+)\)", item["text"]
        )
        if item["predicate"] or match is None:
            raise ValueError("Unproved convergence-token setup")
        destination = labels[match.group(2)]
        sync = instructions[destination - 1]
        if (
            destination <= index + 1
            or sync["full_opcode"] != "BSYNC"
            or sync["predicate"]
            or operands(sync) != [match.group(1)]
            or instructions[destination]["opcode"] != "BAR"
            or any(
                value["opcode"] == "BSSY"
                and operands(value)[0] == match.group(1)
                for value in instructions[index + 1 : destination - 1]
            )
        ):
            raise ValueError("Convergence token does not join before barrier")
        matched_syncs.add(destination - 1)
        convergence.append(
            dict(setup=item, join=sync, barrier=instructions[destination])
        )
    if matched_syncs != {
        index
        for index, item in enumerate(instructions)
        if item["opcode"] == "BSYNC"
    }:
        raise ValueError("Unmatched native convergence join")
    warp_syncs = [
        index
        for index, item in enumerate(instructions)
        if item["opcode"] == "WARPSYNC"
    ]
    if (warp_syncs or entry["instructions"]) and (
        space != "global"
        or warp_syncs != [barrier_index - 1]
        or instructions[warp_syncs[0]]["full_opcode"] != "WARPSYNC"
        or instructions[warp_syncs[0]]["predicate"]
        or operands(instructions[warp_syncs[0]]) != ["0xffffffff"]
    ):
        raise ValueError("Unproved final full-mask warp synchronization")
    return dict(
        status="direct_chain_and_endpoint_admitted",
        measured_region=region,
        priming=priming,
        final_residency_barrier=barrier,
        convergence=convergence,
        pointer_register=pointer,
        pointer_entry=entry,
        pointer_chain_edges=edges,
        starting_clock_live_words=sorted(beginning),
        target_loads=loads,
        administration=admin,
        terminal_call_exit=terminal,
        endpoint_guard=tail,
        count_initialization=origin,
        native_parameter_abi=abi,
        clock_instructions=[instructions[index] for index in clocks],
        complete_instruction_count=len(instructions),
        complete_opcounts=dict(
            Counter(item["opcode"] for item in instructions)
        ),
        units="elapsed thread cycles; target count excludes priming",
        cache_level=None,
    )


WORKER = r'''"""Explicit native worker; preparation is CPU-only."""

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
    "latency_controller", HERE / "benchmark_source.py"
)
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)
if (
    control.digest(HERE / "benchmark_source.py") != REQUEST["generator_sha256"]
    or control.digest(__file__) != REQUEST["worker_sha256"]
):
    raise ValueError("Prepared instrument bytes changed")
sys.path.insert(0, REQUEST["research_root"])
import cubie
import cubie.cuda_simsafe as cuda_helpers
from cubie._utils import package_source_hash
from benchmarks import unroll_landscape
from benchmarks.hardware_model import hardware_probes as hardware


def normalized(value):
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
    if len(manifest["clocks"].get("devices", [])) != 1:
        raise ValueError("Need one identified GPU in manifest clocks")
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
    cubin, entry = hardware._compiled_cubin(compiled)
    resident = (
        hardware.cuda.current_context().get_active_blocks_per_multiprocessor(
            function, geometry["block_size"], 0
        )
    )
    result = dict(
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        entry=entry,
        overloads=len(kernel.overloads),
        handle=str(function.handle),
        resident_blocks_per_sm=int(resident),
        geometry=geometry,
    )
    if (
        result["overloads"] != 1
        or resident != geometry["resident_blocks_per_sm"]
    ):
        raise ValueError("Native signature or driver residency changed")
    return result


def validate_output(values, iterations, order, geometry):
    loads = REQUEST["body_loads"] * iterations
    positions = (
        np.arange(geometry["grid_blocks"], dtype=np.uint64)
        * len(order)
        // geometry["grid_blocks"]
    )
    expected = (
        order[(positions + loads) % len(order)] * REQUEST["stride_bytes"]
    )
    if values.dtype != np.uint64 or values.shape != (
        geometry["grid_blocks"],
        8,
    ):
        raise ValueError("Clock/output shape or type differs")
    if not (
        np.all(values[:, 6] == 1)
        and np.all(values[:, 7] == loads)
        and np.all(values[:, 3] == expected)
        and np.all(values[:, 1] > values[:, 0])
        and np.all(values[:, 2] == values[:, 1] - values[:, 0])
        and np.all(values[:, 4] == values[:, 5])
    ):
        raise ValueError(
            "Final pointer, dependent clock, SMID or load count differs"
        )
    return dict(
        expected_final_offsets=expected.tolist(),
        loads_per_active_thread=loads,
        unique_nodes_per_active_thread=min(loads, len(order)),
        traversed_sector_bytes_per_active_thread=min(loads, len(order))
        * REQUEST["stride_bytes"],
        full_cycles=loads // len(order),
        remainder_loads=loads % len(order),
        prime_unique_nodes_per_active_thread=len(order),
        minimum_cycles=int(values[:, 2].min()),
        maximum_cycles=int(values[:, 2].max()),
        median_cycles=float(np.median(values[:, 2])),
        distinct_observed_smids=np.unique(values[:, 4]).tolist(),
    )


def load_ordinary(directory, current_identity, resources, geometry, order):
    result = json.loads((directory / "result.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if result["status"] != "ordinary_complete" or result["cleanup_errors"]:
        raise ValueError("Ordinary bank did not complete")
    process = json.loads((directory / "process_exit.json").read_text())
    if process != {"returncode": 0}:
        raise ValueError("Ordinary isolated worker did not exit cleanly")
    for field in control.IDENTITY_FIELDS:
        if request[field] != REQUEST[field]:
            raise ValueError("Ordinary request differs: " + field)
    if (
        result["compilation_identity"] != current_identity
        or result["resources"] != resources
        or result["geometry"] != geometry
    ):
        raise ValueError("Ordinary native/source/compiler identity differs")
    for name, sha in result["artifacts"].items():
        if control.digest(directory / name) != sha:
            raise ValueError("Ordinary artifact changed: " + name)
    ring_record = result["ring"]
    ring_path = directory / ring_record["raw_file"]
    if (
        not ring_path.resolve().is_relative_to(directory)
        or control.digest(ring_path) != ring_record["sha256"]
    ):
        raise ValueError("Ordinary source address ring changed")
    stored_ring = np.load(ring_path, allow_pickle=False)
    expected_dtype = np.uint64 if REQUEST["space"] == "global" else np.uint32
    expected_ring = np.zeros(
        REQUEST["footprint_bytes"] // np.dtype(expected_dtype).itemsize,
        dtype=expected_dtype,
    )
    saved_offsets = np.load(directory / "next_offsets.npy", allow_pickle=False)
    saved_cycle = np.load(directory / "cycle.npy", allow_pickle=False)
    control.validate_topology(
        saved_offsets, saved_cycle, REQUEST["nodes"], REQUEST["stride_bytes"]
    )
    expected_ring[
        :: REQUEST["stride_bytes"] // np.dtype(expected_dtype).itemsize
    ] = saved_offsets.astype(expected_dtype) + (
        ring_record["base_address"] if REQUEST["space"] == "global" else 0
    )
    if (
        not np.array_equal(stored_ring, expected_ring)
        or stored_ring.dtype != expected_dtype
    ):
        raise ValueError("Ordinary raw address ring differs from topology")
    starts_path = directory / result["starts"]["file"]
    positions = (
        np.arange(geometry["grid_blocks"], dtype=np.uint64)
        * len(order)
        // geometry["grid_blocks"]
    )
    expected_starts = order[positions] * REQUEST["stride_bytes"]
    if control.digest(starts_path) != result["starts"][
        "sha256"
    ] or not np.array_equal(
        np.load(starts_path, allow_pickle=False), expected_starts
    ):
        raise ValueError("Ordinary CTA starting phases changed")
    if not control.valid_repeats(
        result["iterations"], REQUEST["body_loads"], REQUEST["nodes"]
    ):
        raise ValueError("Ordinary N/2N outputs are not nonfull controls")
    if result["initial_native"] != result["final_native"]:
        raise ValueError("Ordinary native identity changed")
    rows = [
        json.loads(line)
        for line in (directory / "samples.jsonl").read_text().splitlines()
    ]
    if rows != result["samples"]:
        raise ValueError("Ordinary raw sample membership differs")
    measured = [row for row in rows if row["phase"] == "measurement"]
    expected = [
        (block, index, multiple)
        for block in range(2)
        for index in range(6)
        for multiple in ((1, 2) if index % 2 == 0 else (2, 1))
    ]
    if [
        (row["block"], row["index"], row["multiple"]) for row in measured
    ] != expected:
        raise ValueError("Ordinary mirrored membership differs")
    for row in rows:
        file = directory / row["array_file"]
        if (
            not file.resolve().is_relative_to(directory)
            or control.digest(file) != row["array_sha256"]
        ):
            raise ValueError("Ordinary raw output changed")
        with np.load(file, allow_pickle=False) as saved:
            if saved.files != ["values"]:
                raise ValueError("Ordinary output archive membership differs")
            checks = validate_output(
                saved["values"], row["iterations"], order, geometry
            )
        if checks != row["output_checks"]:
            raise ValueError("Ordinary output checks differ")
        duration = control.chain_milliseconds(
            checks,
            row["clocks_before"],
            row["clocks_after"],
            row["event_ms"],
            current_identity["gpu_uuid"],
        )
        if duration != row["minimum_chain_ms_at_observed_max_clock"]:
            raise ValueError(
                "Stored chain duration differs from raw derivation"
            )
        if (
            row["native_before"] != result["initial_native"]
            or row["native_after"] != result["initial_native"]
        ):
            raise ValueError("Ordinary sample native identity differs")
        if (
            row["phase"] not in ("calibration", "measurement")
            or row["prime_loads_per_active_thread"] != REQUEST["nodes"]
            or row["measured_loads_per_active_thread"]
            != REQUEST["body_loads"] * row["iterations"]
        ):
            raise ValueError("Ordinary raw work differs")
        if row["phase"] == "measurement" and (
            row["iterations"] != result["iterations"] * row["multiple"]
            or duration < 20
            or row["event_ms"] < 20
        ):
            raise ValueError("Ordinary work or duration differs")
    return result, dict(
        result_sha256=control.digest(directory / "result.json"),
        samples_sha256=control.digest(directory / "samples.jsonl"),
    )


def main():
    result = dict(
        status="validating",
        kernel_compilation=False,
        gpu_execution=False,
        samples=[],
        actual_cache_level=None,
        cleanup_errors=[],
    )
    try:
        if not hardware.IS_MLIR or hardware.CUDA_SIMULATION:
            raise ValueError("Requires actual installed MLIR backend")
        if control.digest(hardware.__file__) != control.HARDWARE_SHA:
            raise ValueError("Frozen hardware helper differs")
        for filename, field in (
            ("kernel.py", "kernel_source_sha256"),
            ("next_offsets.npy", "topology_sha256"),
            ("cycle.npy", "cycle_sha256"),
        ):
            if control.digest(HERE / filename) != REQUEST[field]:
                raise ValueError("Prepared artifact changed: " + filename)
        offsets = np.load(HERE / "next_offsets.npy", allow_pickle=False)
        order = np.load(HERE / "cycle.npy", allow_pickle=False)
        control.validate_topology(
            offsets, order, REQUEST["nodes"], REQUEST["stride_bytes"]
        )
        nvdisasm = hardware._tool("nvdisasm", REQUEST["nvdisasm"])
        cuobjdump = hardware._tool("cuobjdump", REQUEST["cuobjdump"])
        manifest = normalized(
            hardware._manifest(SimpleNamespace(**REQUEST), nvdisasm)
        )
        control.write_json(HERE / "manifest.json", manifest)
        queried = control.plan(manifest, REQUEST["footprint"])
        if queried != REQUEST["hardware_plan"]:
            raise ValueError(
                "Live hardware capacity query differs from preparation"
            )
        result["compilation_identity"] = identity(manifest)
        result["compilation_identity"]["binary_tools"] = dict(
            nvdisasm=record(nvdisasm), cuobjdump=record(cuobjdump)
        )
        spec = importlib.util.spec_from_file_location(
            "latency_generated", HERE / "kernel.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        kernel = hardware.cuda.jit(**hardware.get_jit_kwargs())(module.probe)
        result["kernel_compilation"] = True
        hardware.compile_kernel_specialization(
            kernel,
            (
                np.uint64(0),
                np.uint64(0),
                np.uint64(0),
                np.uint32(1),
                np.uint32(REQUEST["nodes"]),
            ),
        )
        (compiled,) = kernel.overloads.values()
        cubin, entry = hardware._compiled_cubin(compiled)
        (HERE / "kernel.cubin").write_bytes(cubin)
        ptx = kernel.inspect_asm()
        if isinstance(ptx, dict):
            ptx = "\n".join(ptx.values())
        (HERE / "kernel.ptx").write_text(ptx)
        load_pattern = (
            (r"ld\.global\." + REQUEST["cache"] + r"\.u64\s+p,\s*\[p\];")
            if REQUEST["space"] == "global"
            else r"ld\.shared\.u32\s+p,\s*\[p\];"
        )
        if len(re.findall(load_pattern, ptx)) != REQUEST["body_loads"] + 1:
            raise ValueError(
                "Compiled PTX does not retain the exact direct-load primitive"
            )
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
        abi = control.parameter_abi(elf["stdout"], entry)
        admission = control.check_native(
            *parsed, REQUEST["space"], REQUEST["body_loads"], abi
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
        if resources["local_bytes_per_thread"]:
            raise ValueError(
                "Local frame confounds the direct load instrument"
            )
        expected_static = (
            REQUEST["footprint_bytes"] if REQUEST["space"] == "shared" else 0
        )
        if resources["static_shared_bytes"] != expected_static:
            raise ValueError(
                "Static shared allocation differs from emitted ring"
            )
        function = compiled._codelibrary.get_cufunc()
        device = hardware.cuda.get_current_device()
        resident = int(
            hardware.cuda.current_context().get_active_blocks_per_multiprocessor(
                function, REQUEST["block_size"], 0
            )
        )
        if resident != 1:
            raise ValueError("Need one max-thread block per SM")
        geometry = dict(
            block_size=REQUEST["block_size"],
            active_lanes=1,
            grid_blocks=REQUEST["waves"]
            * int(device.MULTIPROCESSOR_COUNT)
            * resident,
            resident_blocks_per_sm=resident,
            sms=int(device.MULTIPROCESSOR_COUNT),
            resident_warps_per_sm=resident
            * REQUEST["block_size"]
            // int(device.WARP_SIZE),
            timed_chain_warps_per_sm_upper_bound=1,
            waves=REQUEST["waves"],
            dynamic_shared_bytes=0,
            thread_capacity_blocks=int(device.MAX_THREADS_PER_MULTIPROCESSOR)
            // REQUEST["block_size"],
            carveout_preference_set=False,
            actual_shared_capacity=None,
        )
        if geometry["waves"] < 2 or geometry["thread_capacity_blocks"] != 1:
            raise ValueError("Need two complete queried occupancy waves")
        result.update(resources=resources, geometry=geometry)
        initial = native(kernel, compiled, function, geometry)
        result["initial_native"] = initial
        result["artifacts"] = {
            name: control.digest(HERE / name)
            for name in (
                "kernel.cubin",
                "kernel.ptx",
                "kernel.sass",
                "kernel.py",
                "benchmark_source.py",
                "worker.py",
                "cycle.npy",
                "next_offsets.npy",
                "native_elf.txt",
            )
        }
        if REQUEST["mode"] == "compile_only":
            result["status"] = "compile_only_complete"
            return
        prior = None
        if REQUEST["mode"] == "profile":
            prior, result["ordinary_evidence"] = load_ordinary(
                Path(REQUEST["ordinary_dir"]).resolve(),
                result["compilation_identity"],
                resources,
                geometry,
                order,
            )
        dtype = np.uint64 if REQUEST["space"] == "global" else np.uint32
        ring = hardware.cupy.zeros(
            REQUEST["footprint_bytes"] // np.dtype(dtype).itemsize, dtype=dtype
        )
        host_ring = np.zeros(ring.shape, dtype=dtype)
        host_ring[:: REQUEST["stride_bytes"] // np.dtype(dtype).itemsize] = (
            offsets.astype(dtype)
            + (int(ring.data.ptr) if REQUEST["space"] == "global" else 0)
        )
        ring.set(host_ring)
        np.save(HERE / "raw_ring.npy", host_ring, allow_pickle=False)
        result["ring"] = dict(
            base_address=int(ring.data.ptr),
            raw_file="raw_ring.npy",
            sha256=control.digest(HERE / "raw_ring.npy"),
            space=REQUEST["space"],
            dtype=str(host_ring.dtype),
        )
        output = hardware.cupy.empty(
            (geometry["grid_blocks"], 8), dtype=np.uint64
        )
        if REQUEST["nodes"] < geometry["grid_blocks"]:
            raise ValueError("Need a distinct cycle phase for every CTA")
        positions = (
            np.arange(geometry["grid_blocks"], dtype=np.uint64)
            * len(order)
            // geometry["grid_blocks"]
        )
        host_starts = order[positions] * REQUEST["stride_bytes"]
        starts = hardware.cupy.asarray(host_starts)
        np.save(HERE / "start_offsets.npy", host_starts, allow_pickle=False)
        result["starts"] = dict(
            file="start_offsets.npy",
            sha256=control.digest(HERE / "start_offsets.npy"),
            policy="Evenly spaced saved randomized-cycle positions per CTA",
        )
        result["gpu_execution"] = True

        def sample(iterations, phase, block=-1, index=-1, multiple=1):
            before_native = native(kernel, compiled, function, geometry)
            if before_native != initial:
                raise ValueError("Native identity changed before launch")
            output.fill(np.uint64(2**64 - 1))
            before = hardware._clocks()
            event = hardware._timed_launch(
                kernel,
                geometry,
                (
                    np.uint64(ring.data.ptr),
                    np.uint64(output.data.ptr),
                    np.uint64(starts.data.ptr),
                    np.uint32(iterations),
                    np.uint32(REQUEST["nodes"]),
                ),
            )
            after = hardware._clocks()
            after_native = native(kernel, compiled, function, geometry)
            if after_native != initial:
                raise ValueError("Native identity changed after launch")
            values = hardware.cupy.asnumpy(output)
            checks = validate_output(values, iterations, order, geometry)
            if not np.array_equal(hardware.cupy.asnumpy(ring), host_ring):
                raise ValueError("Source pointer ring changed")
            if not np.array_equal(hardware.cupy.asnumpy(starts), host_starts):
                raise ValueError("CTA starting phase input changed")
            chain_ms = control.chain_milliseconds(
                checks,
                before,
                after,
                event,
                result["compilation_identity"]["gpu_uuid"],
            )
            name = f"sample_{len(result['samples']):04d}.npz"
            np.savez(HERE / name, values=values)
            row = dict(
                phase=phase,
                block=block,
                index=index,
                multiple=multiple,
                iterations=iterations,
                prime_loads_per_active_thread=REQUEST["nodes"],
                measured_loads_per_active_thread=REQUEST["body_loads"]
                * iterations,
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
            result["samples"].append(row)
            with (HERE / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(normalized(row)) + "\n")
            return row

        iterations = (
            prior["iterations"]
            if prior
            else max(
                REQUEST["iterations"],
                REQUEST["nodes"] // REQUEST["body_loads"] + 1,
            )
        )
        if prior:
            result["iterations"] = iterations
            sample(
                iterations * REQUEST["profile_multiplier"],
                "profile",
                multiple=REQUEST["profile_multiplier"],
            )
            result["status"] = "profile_complete_cache_level_unassigned"
        else:
            for trial in range(25):
                while not control.valid_repeats(
                    iterations, REQUEST["body_loads"], REQUEST["nodes"]
                ):
                    iterations += 1
                    if iterations >= 2**30:
                        raise ValueError("Repeat counter overflow")
                row = sample(iterations, "calibration", index=trial)
                if (
                    row["event_ms"] >= 20
                    and row["minimum_chain_ms_at_observed_max_clock"] >= 20
                ):
                    break
                iterations = iterations * 2 + 1
            else:
                raise ValueError("Measured chain duration did not reach 20 ms")
            result["iterations"] = iterations
            for block in range(2):
                for index in range(6):
                    for multiple in (1, 2) if index % 2 == 0 else (2, 1):
                        time.sleep(0.05)
                        row = sample(
                            iterations * multiple,
                            "measurement",
                            block,
                            index,
                            multiple,
                        )
                        if (
                            row["event_ms"] < 20
                            or row["minimum_chain_ms_at_observed_max_clock"]
                            < 20
                        ):
                            raise ValueError(
                                "Ordinary chain measurement fell below 20 ms"
                            )
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
'''


def main():
    """Prepare one case, or explicitly run its retained native worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hardware-manifest", type=Path, required=True)
    parser.add_argument(
        "--footprint", choices=FOOTPRINTS, default="l1_quarter"
    )
    parser.add_argument("--cache", choices=("ca", "cg"), default="ca")
    parser.add_argument(
        "--body-loads", type=int, choices=(33, 257), default=257
    )
    parser.add_argument("--iterations", type=int, default=32769)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--nvdisasm")
    parser.add_argument("--cuobjdump")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--profile-multiplier", type=int, choices=(1, 2))
    parser.add_argument("--ordinary-dir", type=Path)
    args = parser.parse_args()
    if args.waves < 2 or not (0 < args.iterations < 2**30):
        parser.error("Need >=2 waves and a positive bounded repeat count")
    if bool(args.profile_multiplier) != bool(args.ordinary_dir):
        parser.error("Profile multiplier and ordinary directory are paired")
    manifest = json.loads(args.hardware_manifest.read_text())
    hardware_plan = plan(manifest, args.footprint)
    space = "shared" if args.footprint.startswith("shared") else "global"
    if space == "shared" and args.cache != "ca":
        parser.error("Shared loads have no ca/cg qualifier; use default ca")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, ptx = source_text(
        space, args.cache, hardware_plan["footprint_bytes"], args.body_loads
    )
    (output / "kernel.py").write_text(source, encoding="utf-8")
    (output / "primitive.ptx").write_text(ptx, encoding="utf-8")
    (output / "worker.py").write_text(WORKER, encoding="utf-8")
    (output / "benchmark_source.py").write_bytes(SCRIPT.read_bytes())
    offsets, order = topology(
        hardware_plan["nodes"], hardware_plan["stride_bytes"], args.seed
    )
    validate_topology(
        offsets, order, hardware_plan["nodes"], hardware_plan["stride_bytes"]
    )
    np.save(output / "next_offsets.npy", offsets, allow_pickle=False)
    np.save(output / "cycle.npy", order, allow_pickle=False)
    request = dict(
        schema=1,
        mode="profile"
        if args.profile_multiplier
        else "ordinary"
        if args.execute
        else "compile_only"
        if args.compile_only
        else "source_only",
        research_root=str(REPO),
        generator_sha256=digest(SCRIPT),
        worker_sha256=digest(output / "worker.py"),
        kernel_source_sha256=digest(output / "kernel.py"),
        topology_sha256=digest(output / "next_offsets.npy"),
        cycle_sha256=digest(output / "cycle.npy"),
        hardware_manifest_path=str(args.hardware_manifest.resolve()),
        hardware_manifest_sha256=digest(args.hardware_manifest),
        hardware_plan=hardware_plan,
        space=space,
        cache=args.cache,
        footprint=args.footprint,
        footprint_bytes=hardware_plan["footprint_bytes"],
        stride_bytes=hardware_plan["stride_bytes"],
        nodes=hardware_plan["nodes"],
        block_size=hardware_plan["block_size"],
        waves=args.waves,
        body_loads=args.body_loads,
        seed=args.seed,
        iterations=args.iterations,
        profile_multiplier=args.profile_multiplier,
        ordinary_dir=str(args.ordinary_dir.resolve())
        if args.ordinary_dir
        else None,
        nvdisasm=args.nvdisasm,
        cuobjdump=args.cuobjdump,
    )
    write_json(output / "request.json", request)
    compile(WORKER, "<latency-worker>", "exec")
    if request["mode"] == "source_only":
        print(
            json.dumps(
                dict(
                    status="source_only",
                    output=str(output),
                    source_sha256=digest(SCRIPT),
                )
            )
        )
        return
    with (output / "worker.stdout.log").open("w") as stdout:
        with (output / "worker.stderr.log").open("w") as stderr:
            completed = subprocess.run(
                [sys.executable, str(output / "worker.py")],
                cwd=REPO,
                stdout=stdout,
                stderr=stderr,
                timeout=7200,
            )
    write_json(
        output / "process_exit.json", dict(returncode=completed.returncode)
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
