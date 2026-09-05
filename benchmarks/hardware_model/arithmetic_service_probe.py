"""Prepare counted arithmetic with exact oracles and separate populations.

Source preparation imports no CUDA modules. Explicit native modes use an
isolated worker; compiled admission precedes every target launch.
"""

import argparse
import ast
import collections
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
LATENCY_SHA = (
    "bd6172f8e924583fabed2d5dd621da7824fdad46aa9dc5730eb36b0f663c76f0"
)
HARDWARE_SHA = (
    "758b54944fe2bd88b6778df59277b89b8ea66d0107bf1feea3825dfd25ea898e"
)
OPERATIONS = {
    "fadd": ("add.rn.f32 x, -x, a;", "FADD"),
    "fmul": ("mul.rn.f32 x, x, m;", "FMUL"),
    "ffma": ("fma.rn.f32 x, x, m, a;", "FFMA"),
    "iadd3": ("add.u32 x, x, a;", "IADD3"),
    "imad": ("mad.lo.u32 x, x, m, a;", "IMAD"),
    "rcp": ("rcp.approx.ftz.f32 x, x;", "MUFU"),
    "mov": ("mov.b32 x, x;", "MOV"),
}
TRACE_STEPS = 64
IDENTITY_FIELDS = (
    "operation",
    "body_operations",
    "block_size",
    "waves",
    "seeds_sha256",
    "coefficients",
    "trace_steps",
    "kernel_sha256",
    "generator_sha256",
    "worker_sha256",
    "latency_sha256",
    "hardware_source_sha256",
    "hardware_manifest_sha256",
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
    "latency_source.py",
    "hardware_manifest.json",
    "hardware_source.py",
)


def digest(path):
    """Hash the bytes actually retained on disk."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write finite JSON without silently normalizing unsupported values."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_module(path, name):
    """Load an explicitly bound local Python source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers(path):
    """Load only the frozen CPU arithmetic/control proof helpers."""
    if digest(path) != LATENCY_SHA:
        raise ValueError("Frozen native-proof helper bytes differ")
    return load_module(path, "arithmetic_frozen_latency_helpers")


def plan(manifest):
    """Bind maximal blocks to an architecture-qualified thread capacity."""
    attrs = manifest["device_attributes"]
    block = attrs["MAX_THREADS_PER_BLOCK"]
    if (
        manifest["compute_capability"] != [8, 9]
        or block != 1024
        or attrs["WARP_SIZE"] != 32
        or attrs["MAX_THREADS_PER_MULTIPROCESSOR"] // block != 1
    ):
        raise ValueError("Need queried SM89 with one maximal block/SM")
    return dict(
        block_size=block,
        warp_size=32,
        max_threads_per_sm=attrs["MAX_THREADS_PER_MULTIPROCESSOR"],
        thread_capacity_blocks=1,
        timed_warp_choices=[1, 32],
        scope="Final CTA barrier retains every allocated warp",
    )


def inputs(operation, block):
    """Construct lane-dependent exact bits and runtime coefficients."""
    lane = np.arange(block, dtype=np.uint32) % np.uint32(32)
    if operation in ("fadd", "ffma"):
        values = (2 * lane + 1).astype(np.float32) / np.float32(128)
        seeds = values.view(np.uint32)
        coefficients = dict(m=0xBF800000, a=0x3F800000)
    elif operation == "fmul":
        values = np.float32(1) + lane.astype(np.float32) / np.float32(64)
        seeds = values.view(np.uint32)
        coefficients = dict(m=0xBF800000, a=0)
    elif operation == "rcp":
        values = np.float32(1.25) + lane.astype(np.float32) / np.float32(128)
        seeds = values.view(np.uint32)
        coefficients = dict(m=0, a=0)
    else:
        seeds = lane + np.uint32(1)
        coefficients = dict(m=5 if operation == "imad" else 0, a=1)
    return seeds.copy(), coefficients


def affine_power(multiplier, addend, count):
    """Compose an unsigned affine map exactly modulo 2**32."""
    mask = 2**32 - 1
    left, right = 1, 0
    while count:
        if count & 1:
            left = multiplier * left & mask
            right = (multiplier * right + addend) & mask
        addend = addend * (multiplier + 1) & mask
        multiplier = multiplier * multiplier & mask
        count >>= 1
    return left, right


def expected_bits(operation, seeds, coefficients, count, cycles=None):
    """Return exact FP32 bits or integer modular results for each lane."""
    if type(count) is not int or count < 0:
        raise ValueError("Need a nonnegative integer operation count")
    if operation in ("fadd", "ffma"):
        values = seeds.view(np.float32)
        answer = (np.float32(1) - values) if count & 1 else values
        return answer.copy().view(np.uint32)
    if operation == "fmul":
        return seeds ^ np.uint32(0x80000000 if count & 1 else 0)
    if operation in ("iadd3", "imad"):
        multiplier = 1 if operation == "iadd3" else coefficients["m"]
        left, right = affine_power(multiplier, coefficients["a"], count)
        return np.array(
            [(left * int(seed) + right) & (2**32 - 1) for seed in seeds],
            dtype=np.uint32,
        )
    if operation == "mov":
        return seeds.copy()
    if operation != "rcp" or cycles is None:
        raise ValueError("Reciprocal needs an admitted functional cycle")
    answer = []
    for seed in seeds:
        record = cycles[str(int(seed))]
        sequence, start = record["sequence"], record["cycle_start"]
        index = (
            count
            if count < start
            else (start + (count - start) % record["cycle_length"])
        )
        answer.append(sequence[index])
    return np.array(answer, dtype=np.uint32)


def normal_fraction(bits):
    """Decode a normal finite FP32 value using integer arithmetic."""
    bits = int(bits)
    exponent = (bits >> 23) & 255
    if not 0 < exponent < 255:
        raise ValueError("Functional reciprocal trace is not normal")
    mantissa = (bits & (2**23 - 1)) + 2**23
    shift = exponent - 150
    value = Fraction(mantissa) * Fraction(2) ** shift
    return -value if bits >> 31 else value


def reciprocal_cycles(trace, seeds):
    """Validate exact rational error bounds and retained closed cycles."""
    if trace.dtype != np.uint32 or trace.shape != (
        len(seeds),
        TRACE_STEPS + 1,
    ):
        raise ValueError("Functional trace shape/type differs")
    result = {}
    for seed, row in zip(seeds, trace):
        if int(row[0]) != int(seed):
            raise ValueError("Functional trace starts at a different seed")
        sequence = list(map(int, row))
        for previous, following in zip(sequence, sequence[1:]):
            exact = 1 / normal_fraction(previous)
            magnitude = abs(exact)
            exponent = magnitude.numerator.bit_length()
            exponent -= magnitude.denominator.bit_length()
            if magnitude < Fraction(2) ** exponent:
                exponent -= 1
            ulp = Fraction(2) ** (exponent - 23)
            if abs(normal_fraction(following) - exact) > ulp:
                raise ValueError("Native reciprocal exceeds its one-ulp bound")
        first = {}
        cycle = None
        for index, bits in enumerate(sequence):
            if bits in first:
                start = first[bits]
                length = index - start
                if length <= 1:
                    raise ValueError(
                        "Fixed-point seed cannot distinguish N/2N"
                    )
                cycle = dict(
                    sequence=sequence[:index],
                    cycle_start=start,
                    cycle_length=length,
                )
                if any(
                    value != sequence[start + (offset - start) % length]
                    for offset, value in enumerate(sequence[index:], index)
                ):
                    raise ValueError(
                        "Observed reciprocal cycle does not close"
                    )
                break
            first[bits] = index
        if cycle is None:
            raise ValueError("Bounded reciprocal trace has no closed cycle")
        key = str(int(seed))
        if key in result and result[key] != cycle:
            raise ValueError("Same seed has inconsistent device transitions")
        result[key] = cycle
    return result


def source_text(operation, body):
    """Emit one primitive with a runtime one/full-warp population switch."""
    target = OPERATIONS[operation][0]
    ptx = [
        "{",
        ".reg .pred inactive, again, bad, tracing;",
        ".reg .b32 tid, warp, sm0, sm1, count, x, m, a, expected, ready;",
        ".reg .u64 address, begin, end, delta, total;",
        "mov.u32 tid, %tid.x;",
        "shr.u32 warp, tid, 5;",
        "setp.ge.u32 inactive, warp, $6;",
        "@inactive bra finished;",
        "mov.u32 sm0, %smid;",
        "ld.global.u32 x, [$2];",
        "ld.global.u32 expected, [$3];",
        "mov.b32 m, $8;",
        "mov.b32 a, $9;",
    ]
    if operation == "rcp":
        ptx += [
            "setp.ne.u32 tracing, $7, 0;",
            "@tracing bra trace_begin;",
        ]
    ptx += (
        [
            "mov.u32 count, $5;",
            "xor.b32 ready, x, m;",
            "xor.b32 ready, ready, a;",
            "xor.b32 ready, ready, expected;",
            "xor.b32 ready, ready, count;",
            "setp.eq.u32 bad, ready, 0xffffffff;",
            "@bad bra invalid;",
            "mov.u64 begin, %clock64;",
            "timed_chain:",
        ]
        + [target] * body
        + [
            "add.u32 count, count, -1;",
            "setp.ne.u32 again, count, 0;",
            "@again bra.uni timed_chain;",
            "setp.ne.u32 bad, x, expected;",
            "@bad bra invalid;",
            "mov.u64 end, %clock64;",
            "sub.u64 delta, end, begin;",
            "mov.u32 sm1, %smid;",
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
        ]
    )
    if operation == "rcp":
        ptx += [
            "trace_begin:",
            "mov.u64 address, $4;",
            "st.global.u32 [address], x;",
        ]
        for index in range(TRACE_STEPS):
            ptx += [target, f"st.global.u32 [address+{4 * (index + 1)}], x;"]
        ptx += [
            "cvt.u64.u32 address, sm0;",
            "st.global.u64 [$1+32], address;",
            "mov.u32 sm1, %smid;",
            "cvt.u64.u32 address, sm1;",
            "st.global.u64 [$1+40], address;",
            "mov.u64 address, 2;",
            "st.global.u64 [$1+48], address;",
            "bra finished;",
        ]
    ptx += [
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
        "trace",
        "count",
        "warps",
        "trace_steps",
        "m",
        "a",
    )
    types = ["i64"] * 4 + ["i32"] * 5
    signature = ", ".join(f"%{n}: {t}" for n, t in zip(names, types))
    arguments = ", ".join("%" + name for name in names)
    intrinsic = (
        f"func.func private @arithmetic_chain({signature}) -> i32 "
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
        f"arithmetic_chain = cuda.intrin.define({intrinsic!r})\n\n"
        "def probe(output, seeds, expected, trace, iterations, warps, "
        "trace_steps, m, a):\n"
        "    tid = uint64(cuda.threadIdx.x)\n"
        "    index = uint64(cuda.blockIdx.x) * "
        "uint64(cuda.blockDim.x) + tid\n"
        "    arithmetic_chain(output + index * uint64(64), "
        "seeds + tid * uint64(4), expected + tid * uint64(4), "
        f"trace + index * uint64({4 * (TRACE_STEPS + 1)}), "
        "uint32(iterations), uint32(warps), uint32(trace_steps), "
        "uint32(m), uint32(a))\n"
    )
    compile(source, "<arithmetic-kernel>", "exec")
    return source, asm


def valid_repeats(operation, seeds, coefficients, repeats, body, cycles=None):
    """Require bounded counters and distinct exact nonidentity outputs."""
    if type(repeats) is not int or not 0 < repeats < 2**30 or repeats % 2 == 0:
        return False
    for multiple in (1, 2):
        expected = expected_bits(
            operation,
            seeds,
            coefficients,
            body * repeats * multiple,
            cycles,
        )
        ready = seeds ^ expected ^ np.uint32(coefficients["m"])
        ready ^= np.uint32(coefficients["a"])
        ready ^= np.uint32(repeats * multiple)
        if np.any(ready == np.uint32(2**32 - 1)):
            return False
    first = expected_bits(
        operation, seeds, coefficients, body * repeats, cycles
    )
    second = expected_bits(
        operation, seeds, coefficients, body * repeats * 2, cycles
    )
    return operation == "mov" or np.all(first != second).item()


def validate_output(values, expected, repeats, warps, geometry, body):
    """Check every active lane and retain exact per-CTA clock envelopes."""
    blocks, block = geometry["grid_blocks"], geometry["block_size"]
    if values.dtype != np.uint64 or values.shape != (blocks, block, 8):
        raise ValueError("Arithmetic output type/shape differs")
    if warps not in (1, 32) or block != 1024:
        raise ValueError("Unproved population")
    active = values[:, : warps * 32, :]
    inactive = values[:, warps * 32 :, :]
    if not (
        np.all(inactive == np.uint64(2**64 - 1))
        and np.all(active[:, :, 6] == 1)
        and np.all(active[:, :, 7] == body * repeats)
        and np.all(active[:, :, 3] == expected[: warps * 32])
        and np.all(active[:, :, 1] > active[:, :, 0])
        and np.all(active[:, :, 2] == active[:, :, 1] - active[:, :, 0])
        and np.all(active[:, :, 4] == active[:, :, 5])
        and np.all(active[:, :, 4] == active[:, :1, 4])
    ):
        raise ValueError("Result bits, clocks, SMID or population differ")
    envelopes = active[:, :, 1].max(axis=1) - active[:, :, 0].min(axis=1)
    return dict(
        minimum_cycles=int(active[:, :, 2].min()),
        median_cycles=float(np.median(active[:, :, 2])),
        maximum_cycles=int(active[:, :, 2].max()),
        cta_envelope_cycles=list(map(int, envelopes)),
        sum_cta_envelope_sm_cycles=sum(map(int, envelopes)),
        operations_per_lane=body * repeats,
        target_warp_instructions=blocks * warps * body * repeats,
        target_thread_instructions=blocks * warps * 32 * body * repeats,
        timed_warps_per_cta=warps,
        observed_smids=np.unique(active[:, :, 4]).tolist(),
        final_bits_sha256=hashlib.sha256(
            active[:, :, 3].tobytes()
        ).hexdigest(),
        scope=(
            "CTA envelopes contain target work and administration; "
            "no inferred pipeline map"
        ),
    )


def parameter_abi(text, entry):
    """Bind the emitted nine-scalar argument positions and native extent."""
    marker = ".nv.info." + entry
    if text.count(marker) != 2:
        raise ValueError("Unexpected parameter section membership")
    section = text.rsplit(marker, 1)[1]
    rows = re.findall(
        r"Ordinal\s*:\s*(0x[0-9a-f]+)\s+Offset\s*:\s*(0x[0-9a-f]+)"
        r"\s+Size\s*:\s*(0x[0-9a-f]+)",
        section,
    )
    actual = sorted(tuple(int(x, 16) for x in row) for row in rows)
    expected = [(i, 8 * i, 8) for i in range(4)]
    expected += [(i + 4, 32 + 4 * i, 4) for i in range(5)]
    if actual != expected:
        raise ValueError("Native ABI differs from the emitted nine scalars")
    match = re.search(
        r"EIATTR_PARAM_CBANK\s+Format:\s+EIFMT_SVAL\s+Value:\s+"
        r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)",
        section,
    )
    if match is None:
        raise ValueError("Missing native parameter constant-bank extent")
    packed = int(match.group(2), 16)
    base, size = packed & 65535, packed >> 16
    if size not in (52, 56):
        raise ValueError("Unproved parameter extent/alignment padding")
    return dict(
        parameters=actual,
        constant_base=base,
        parameter_bytes=size,
        trailing_alignment_padding=size - 52,
        count_constant_offset=base + 32,
        warps_constant_offset=base + 36,
        trace_constant_offset=base + 40,
        coefficient_offsets=dict(m=base + 44, a=base + 48),
    )


def register_operand(value):
    """Recognize a scalar GPR operand and its explicit sign modifier."""
    match = re.fullmatch(r"(-?)(R[0-9]+)(?:\.reuse)?", value)
    return (match.group(2), bool(match.group(1))) if match else None


def target_instruction(item, operation, helper):
    """Distinguish the target native form from moves and loop counters."""
    opcode = item["full_opcode"]
    allowed = {
        "fadd": ("FADD", "FADD.FTZ"),
        "fmul": ("FMUL", "FMUL.FTZ"),
        "ffma": ("FFMA", "FFMA.FTZ"),
        "iadd3": ("IADD3",),
        "imad": ("IMAD", "IMAD.U32"),
        "rcp": ("MUFU.RCP",),
        "mov": ("MOV", "IMAD.MOV.U32"),
    }[operation]
    if opcode not in allowed or item["predicate"]:
        return False
    values = helper.operands(item)
    if not values or register_operand(values[0]) is None:
        return False
    if operation == "iadd3" and len(values) == 4:
        if values[0] == values[1] and values[2:] == ["-0x1", "RZ"]:
            return False
    return True


def coefficient_origin(value, instructions, before, offset, dominance, helper):
    """Require a direct runtime coefficient or a dominating scalar copy."""
    if value == f"c[0x0][0x{offset:x}]":
        return dict(operand=value, constant_offset=offset)
    register = register_operand(value)
    if register is None or register[1]:
        raise ValueError(
            "Target coefficient is not the declared runtime value"
        )
    return helper.count_origin(
        instructions,
        register[0],
        before,
        offset,
        dominance,
    )


def reaching_value(value, instructions, before, dominance, helper, seen=None):
    """Trace scalar copies/XORs to actual loads, parameters or thread ID."""
    seen = set() if seen is None else set(seen)
    key = (value, before)
    if key in seen:
        raise ValueError("Cyclic pre-clock operand definition")
    seen.add(key)
    parameter = re.fullmatch(r"c\[0x0\]\[(0x[0-9a-f]+)\]", value)
    if parameter:
        return {"parameter:" + str(int(parameter.group(1), 16))}
    if value in ("RZ", "URZ", "0x0"):
        return set()
    if re.fullmatch(r"U?R[0-9]+", value) is None:
        raise ValueError("Unproved pre-clock scalar operand: " + value)
    definitions = [
        i
        for i in range(before)
        if value in helper.written_registers(instructions[i])
    ]
    if not definitions:
        raise ValueError("Missing scalar definition")
    index = definitions[-1]
    item, args = instructions[index], helper.operands(instructions[index])
    if (
        index not in dominance.get(before, set())
        or item["predicate"]
        or args[0] != value
    ):
        raise ValueError("Scalar definition is not explicit and dominating")
    if item["full_opcode"] == "LDG.E" and len(args) == 2:
        return {"load:" + str(index)}
    if item["full_opcode"] == "S2R" and args[1:] == ["SR_TID.X"]:
        return {"thread_index"}
    if item["full_opcode"] in ("MOV", "ULDC", "LDC") and len(args) == 2:
        sources = args[1:]
    elif item["full_opcode"] == "IMAD.MOV.U32" and args[1:3] == ["RZ", "RZ"]:
        sources = args[3:]
    elif item["full_opcode"] == "LOP3.LUT" and args[4:] == ["0x96", "!PT"]:
        sources = args[1:4]
    else:
        raise ValueError("Unproved pre-clock copy/XOR: " + item["text"])
    roots = set()
    for source in sources:
        roots ^= reaching_value(
            source, instructions, index, dominance, helper, seen
        )
    return roots


def initial_guard(
    instructions, start, end, initial, expected, abi, dominance, helper, labels
):
    """Require a checksum guard depending on every timed input and count."""
    if start < 2:
        raise ValueError("Missing initial completion guard")
    compare, branch = instructions[start - 2 : start]
    args = helper.operands(compare)
    if (
        compare["full_opcode"] != "ISETP.EQ.U32.AND"
        or compare["predicate"]
        or len(args) != 5
        or args[1] != "PT"
        or args[3:] != ["-0x1", "PT"]
        or branch["full_opcode"] != "BRA"
        or branch["predicate"] != "@" + args[0]
        or helper.control_target(branch, labels) <= end
    ):
        raise ValueError("Starting clock lacks the initial checksum guard")
    roots = reaching_value(args[2], instructions, start - 2, dominance, helper)
    wanted = reaching_value(
        initial, instructions, start - 2, dominance, helper
    )
    wanted |= reaching_value(
        expected, instructions, start - 2, dominance, helper
    )
    wanted |= {
        "parameter:" + str(offset)
        for offset in (
            abi["count_constant_offset"],
            *abi["coefficient_offsets"].values(),
        )
    }
    if (
        roots != wanted
        or len([item for item in roots if item.startswith("load:")]) != 2
    ):
        raise ValueError(
            "Initial guard does not consume both inputs and coefficients/count"
        )
    return dict(compare=compare, branch=branch, data_roots=sorted(roots))


def population_gate(instructions, start, end, abi, dominance, helper, labels):
    """Prove a runtime comparison against floor(threadIdx.x/32)."""
    matches = []
    for index in range(start - 1):
        item, branch = instructions[index : index + 2]
        args = helper.operands(item)
        if (
            item["full_opcode"] != "ISETP.GE.U32.AND"
            or item["predicate"]
            or len(args) != 5
            or args[1] != "PT"
            or args[-1] != "PT"
            or branch["full_opcode"] != "BRA"
            or branch["predicate"] != "@" + args[0]
            or helper.control_target(branch, labels) <= end
            or index + 1 not in dominance[start]
        ):
            continue
        coefficient_origin(
            args[3],
            instructions,
            index,
            abi["warps_constant_offset"],
            dominance,
            helper,
        )
        definitions = [
            i
            for i in range(index)
            if args[2] in helper.written_registers(instructions[i])
        ]
        if not definitions:
            raise ValueError("Missing warp-index definition")
        shift_index = definitions[-1]
        shift = instructions[shift_index]
        values = helper.operands(shift)
        if (
            shift["full_opcode"] != "SHF.R.U32.HI"
            or shift["predicate"]
            or len(values) != 4
            or values[:3] != [args[2], "RZ", "0x5"]
            or shift_index not in dominance[index]
            or reaching_value(
                values[3], instructions, shift_index, dominance, helper
            )
            != {"thread_index"}
        ):
            raise ValueError("Warp index is not the exact native tid/32 form")
        matches.append(dict(compare=item, branch=branch, warp_index=shift))
    if len(matches) != 1:
        raise ValueError("Need one dominating full-warp population gate")
    return matches[0]


def check_native(instructions, loops, labels, operation, body, abi, helper):
    """Admit a counted GPR recurrence and its observable clock interval."""
    if operation == "ffma" and body == 257:
        return observed_ffma257(instructions, loops, labels, abi, helper)
    clocks = [
        i for i, item in enumerate(instructions) if item["opcode"] == "CS2R"
    ]
    if len(clocks) != 2:
        raise ValueError("Need exactly two native clock reads")
    start, end = clocks
    for index in clocks:
        item = instructions[index]
        values = helper.operands(item)
        if (
            item["predicate"]
            or item["full_opcode"] != "CS2R"
            or len(values) != 2
            or values[1] != "SR_CLOCKLO"
            or len(helper.written_registers(item)) != 2
        ):
            raise ValueError("Unproved clock register pair")
    candidates = []
    for loop in loops:
        if not start < loop["start_index"] <= loop["end_index"] < end:
            continue
        hot = instructions[loop["start_index"] : loop["end_index"] + 1]
        targets = [
            item for item in hot if target_instruction(item, operation, helper)
        ]
        if len(targets) == body:
            candidates.append((loop, hot, targets))
    if len(candidates) != 1:
        raise ValueError("No unique retained counted target recurrence")
    region, hot, targets = candidates[0]
    transformed = [dict(item) for item in instructions]
    target_addresses = {item["address"] for item in targets}
    for item in transformed:
        if item["address"] in target_addresses:
            item["opcode"] = "ARITHMETIC_TARGET"
    admin, counter, calls, terminal = helper.loop_administration(
        transformed,
        region,
        labels,
        "ARITHMETIC_TARGET",
    )
    dominance = helper.dominators(instructions, labels, calls)
    exit_index = region["end_index"] - bool(calls)
    if (
        start not in dominance.get(region["start_index"], set())
        or exit_index not in dominance.get(end, set())
        or any(
            i not in dominance[exit_index]
            for i in range(region["start_index"], exit_index)
        )
        or region["start_index"] != start + 1
    ):
        raise ValueError("An edge bypasses the counted body or starting clock")
    origin = helper.count_origin(
        instructions,
        counter,
        start,
        abi["count_constant_offset"],
        dominance,
    )
    previous = helper.operands(targets[-1])[0]
    initial = previous
    edges, coefficients = [], []
    for item in targets:
        values = helper.operands(item)
        destination, reads = values[0], values[1:]
        if operation == "fadd":
            matches = [
                i
                for i, value in enumerate(reads)
                if register_operand(value) == (previous, True)
            ]
            needed = ["a"]
        elif operation in ("fmul", "ffma", "imad"):
            matches = [
                i
                for i, value in enumerate(reads[:2])
                if register_operand(value) == (previous, False)
            ]
            needed = ["m"] + (["a"] if operation in ("ffma", "imad") else [])
        elif operation == "iadd3":
            reads = [value for value in reads if value != "RZ"]
            matches = [
                i
                for i, value in enumerate(reads)
                if register_operand(value) == (previous, False)
            ]
            needed = ["a"]
        else:
            if item["full_opcode"] == "IMAD.MOV.U32":
                if reads[:2] != ["RZ", "RZ"]:
                    raise ValueError("Move has extra arithmetic operands")
                reads = reads[2:]
            matches = [
                i
                for i, value in enumerate(reads)
                if register_operand(value) == (previous, False)
            ]
            needed = []
        if len(matches) != 1:
            raise ValueError("Target does not consume the preceding result")
        others = reads[: matches[0]] + reads[matches[0] + 1 :]
        if len(others) != len(needed):
            raise ValueError("Target operand cardinality differs")
        proofs = [
            coefficient_origin(
                value,
                instructions,
                start,
                abi["coefficient_offsets"][name],
                dominance,
                helper,
            )
            for name, value in zip(needed, others)
        ]
        coefficients += [
            register_operand(value)[0]
            for value in others
            if register_operand(value)
        ]
        edges.append(
            dict(
                address=item["address"],
                destination=destination,
                predecessor=previous,
                instruction=item,
                coefficient_origins=proofs,
            )
        )
        previous = destination
    if previous != initial:
        raise ValueError("The recurrence does not close at its backedge")
    destinations = {edge["destination"] for edge in edges}
    begin_words = helper.written_registers(instructions[start])
    if (
        counter in destinations
        or set(coefficients) & destinations
        or begin_words & (destinations | set(coefficients) | {counter})
        or any(helper.written_registers(item) & begin_words for item in hot)
        or helper.written_registers(instructions[end])
        & (begin_words | {previous})
    ):
        raise ValueError("Measured work clobbers a live operand or timestamp")
    tail = instructions[region["end_index"] + 1 : end]
    if len(tail) != 2:
        raise ValueError("Need an immediate final-result guard before end")
    compare, branch = tail
    values = helper.operands(compare)
    if (
        compare["full_opcode"] != "ISETP.NE.U32.AND"
        or compare["predicate"]
        or len(values) != 5
        or values[1] != "PT"
        or values[-1] != "PT"
        or previous not in values[2:4]
        or branch["full_opcode"] != "BRA"
        or branch["predicate"] != "@" + values[0]
        or helper.control_target(branch, labels) <= end
    ):
        raise ValueError("Ending clock is not guarded by the final result")
    expected = next(value for value in values[2:4] if value != previous)
    if register_operand(expected) is None:
        raise ValueError("Expected output must be a preloaded scalar register")
    if any(expected in helper.written_registers(item) for item in hot):
        raise ValueError("Measured body clobbers the expected output")
    if expected in begin_words:
        raise ValueError("Starting clock clobbers the expected output")
    initial_proof = initial_guard(
        instructions,
        start,
        end,
        initial,
        expected,
        abi,
        dominance,
        helper,
        labels,
    )
    population = population_gate(
        instructions,
        start,
        end,
        abi,
        dominance,
        helper,
        labels,
    )
    barriers = [
        i for i, item in enumerate(instructions) if item["opcode"] == "BAR"
    ]
    if len(barriers) != 1 or barriers[0] <= end:
        raise ValueError("Need one final CTA synchronization")
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
        raise ValueError(
            "Not every exit retains the full CTA until completion"
        )
    convergence = []
    syncs = set()
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
            or sync["predicate"]
            or sync["full_opcode"] != "BSYNC"
            or helper.operands(sync) != [match.group(1)]
            or destination != barriers[0]
        ):
            raise ValueError(
                "Convergence token does not join at final barrier"
            )
        syncs.add(destination - 1)
        convergence.append(dict(setup=item, join=sync))
    if syncs != {
        i for i, item in enumerate(instructions) if item["opcode"] == "BSYNC"
    }:
        raise ValueError("Unmatched native convergence join")
    warp_syncs = [
        i
        for i, item in enumerate(instructions)
        if item["opcode"] == "WARPSYNC"
    ]
    if warp_syncs and (
        warp_syncs != [barriers[0] - 1]
        or instructions[warp_syncs[0]]["predicate"]
        or helper.operands(instructions[warp_syncs[0]]) != ["0xffffffff"]
    ):
        raise ValueError("Unproved full-mask warp synchronization")
    functional = None
    outside = [
        item
        for i, item in enumerate(instructions)
        if not start < i < end and target_instruction(item, operation, helper)
    ]
    if operation == "rcp":
        functional = trace_native(
            instructions,
            outside,
            start,
            end,
            abi,
            dominance,
            helper,
            labels,
            reaching_value(
                initial, instructions, start - 2, dominance, helper
            ),
            calls,
        )
    elif operation not in ("mov", "iadd3", "imad") and outside:
        raise ValueError("Unaccounted target arithmetic outside timed body")
    return dict(
        status="arithmetic_recurrence_admitted",
        measured_region=region,
        target_edges=edges,
        administration=admin,
        count_origin=origin,
        terminal_exit=terminal,
        end_guard=tail,
        expected_register=expected,
        initial_guard=initial_proof,
        population_gate=population,
        convergence=convergence,
        clock_instructions=[instructions[i] for i in clocks],
        starting_clock_live_words=sorted(begin_words),
        final_barrier=barrier,
        native_parameter_abi=abi,
        complete_instruction_count=len(instructions),
        functional_trace=functional,
        complete_opcounts=dict(
            Counter(item["opcode"] for item in instructions)
        ),
        scope=(
            "Exact recurrence and interval; first compiled control/ABI "
            "artifact needs independent review"
        ),
    )


def observed_ffma257(instructions, loops, labels, abi, helper):
    """Prove the complete retained FFMA257 ABI, values and control template.

    This form is the actual installed-backend lowering of the generated
    primitive. Every instruction and target is checked. BREAK cancels
    B1 participation for successful lanes; it is not a branch to B1.
    NVIDIA's saved-cubin basic-block CFG independently confirms the
    explicit branch edges recorded here.
    """
    prefix = """MOV R1, c[0x0][0x28]
S2R R4, SR_TID.X
MOV R5, RZ
BSSY B0, `(.L_x_0)
MOV R9, 0x4
S2R R7, SR_CTAID.X
SHF.R.U32.HI R0, RZ, 0x5, R4
ISETP.GE.U32.AND P0, PT, R0, c[0x0][0x184], PT
IMAD.WIDE.U32 R6, R7, c[0x0][0x0], R4
LEA R2, P1, R6, c[0x0][0x160], 0x6
LEA.HI.X R3, R6, c[0x0][0x164], R7, 0x6, P1
IMAD.WIDE.U32 R6, R4, R9, c[0x0][0x168]
IMAD.WIDE.U32 R8, R4, R9, c[0x0][0x170]
@P0 BRA `(.L_x_1)
ULDC.64 UR4, c[0x0][0x118]
LDG.E R6, [R6.64]
LDG.E R5, [R8.64]
BSSY B1, `(.L_x_2)
S2R R4, SR_VIRTUALSMID
LOP3.LUT R0, R6, c[0x0][0x18c], RZ, 0x3c, !PT
LOP3.LUT R0, R5, c[0x0][0x190], R0, 0x96, !PT
LOP3.LUT R0, R0, c[0x0][0x180], RZ, 0x3c, !PT
ISETP.NE.U32.AND P0, PT, R0, -0x1, PT
@!P0 BRA `(.L_x_3)
MOV R9, R6
ULDC UR4, c[0x0][0x180]
MOV R0, c[0x0][0x180]
CS2R R6, SR_CLOCKLO""".splitlines()
    body = (
        [
            "MOV R8, c[0x0][0x18c]",
            "UIADD3 UR4, UR4, -0x1, URZ",
            "FFMA R9, R9, R8.reuse, c[0x0][0x190]",
            "ISETP.NE.U32.AND P0, PT, RZ, UR4, PT",
        ]
        + ["FFMA R9, R9, R8, c[0x0][0x190]"] * 256
        + [
            "@!P0 CALL.REL.NOINC `(.L_x_4)",
            "BRA `(.L_x_5)",
        ]
    )
    tail = (
        """NOP
ISETP.NE.U32.AND P0, PT, R9, R5, PT
@!P0 BREAK B1
@P0 BRA `(.L_x_3)
CS2R R12, SR_CLOCKLO
S2UR UR4, SR_VIRTUALSMID
IADD3 R14, P0, -R6, R12, RZ
ULDC.64 UR6, c[0x0][0x118]
MOV R16, 0x1
STG.E.64 [R2.64], R6
IADD3.X R15, ~R7, R13, RZ, P0, !PT
MOV R17, 0x0
STG.E.64 [R2.64+0x8], R12
MOV R5, RZ
MOV R11, RZ
STG.E.64 [R2.64+0x30], R16
STG.E.64 [R2.64+0x20], R4
MOV R6, R9
IMAD.WIDE.U32 R8, R0, 0x101, RZ
MOV R7, RZ
STG.E.64 [R2.64+0x10], R14
MOV R10, UR4
STG.E.64 [R2.64+0x18], R6
STG.E.64 [R2.64+0x38], R8
STG.E.64 [R2.64+0x28], R10
BRA `(.L_x_1)
BSYNC B1
ULDC.64 UR4, c[0x0][0x118]
STG.E.64 [R2.64+0x30], RZ
BSYNC B0
WARPSYNC 0xffffffff
BAR.SYNC.DEFER_BLOCKING 0x0
EXIT
BRA `(.L_x_6)""".splitlines()
        + ["NOP"] * 12
    )
    template = prefix + body + tail
    expected_labels = {
        ".L_x_5": 28,
        ".L_x_4": 290,
        ".L_x_3": 316,
        ".L_x_2": 317,
        ".L_x_1": 319,
        ".L_x_0": 320,
        ".L_x_6": 323,
        ".L_x_7": 336,
    }
    if len(instructions) != 336 or len(template) != 336:
        raise ValueError("Observed FFMA257 complete instruction count differs")
    if labels != expected_labels:
        raise ValueError("Observed FFMA257 control destinations differ")
    for index, (expected, item) in enumerate(zip(template, instructions)):
        predicated = expected.startswith("@")
        predicate, text = (
            expected.split(" ", 1) if predicated else ("", expected)
        )
        if (
            item["address"] != index * 16
            or item["predicate"] != predicate
            or item["text"] != text
            or item["full_opcode"] != text.split()[0]
            or item["opcode"] != text.split()[0].split(".")[0]
        ):
            raise ValueError(
                f"FFMA257 exact instruction differs at 0x{index * 16:x}"
            )
    if (
        abi["constant_base"] != 0x160
        or abi["count_constant_offset"] != 0x180
        or abi["warps_constant_offset"] != 0x184
        or abi["trace_constant_offset"] != 0x188
        or abi["coefficient_offsets"] != dict(m=0x18C, a=0x190)
        or abi["parameters"]
        != [(i, 8 * i, 8) for i in range(4)]
        + [(i + 4, 32 + 4 * i, 4) for i in range(5)]
    ):
        raise ValueError("FFMA257 native parameter ABI differs")
    if (
        len(loops) != 1
        or loops[0]["start_index"] != 28
        or loops[0]["end_index"] != 289
        or loops[0]["opcounts"]
        != dict(MOV=1, UIADD3=1, FFMA=257, ISETP=1, CALL=1, BRA=1)
    ):
        raise ValueError("FFMA257 actual hot-loop inventory differs")
    calls = {288: 290}
    dominance = helper.dominators(instructions, labels, calls)
    if (
        any(27 not in dominance[i] for i in range(28, 294))
        or any(i not in dominance[288] for i in range(28, 288))
        or 293 not in dominance[294]
        or 321 not in dominance[322]
    ):
        raise ValueError("FFMA257 clock/body/end/barrier domination differs")
    begin_words = {"R6", "R7"}
    if any(
        helper.written_registers(item) & begin_words
        for item in instructions[28:294]
    ):
        raise ValueError("FFMA257 starting timestamp is overwritten")
    if any(
        "R0" in helper.written_registers(item) for item in instructions[27:308]
    ):
        raise ValueError("Original repeat count is not preserved for output")
    target_indices = [30] + list(range(32, 288))
    target_edges = [
        dict(
            instruction=instructions[i],
            destination="R9",
            predecessor="R9",
            multiplier="R8",
            multiplier_load=instructions[28],
            addend_constant_offset=0x190,
        )
        for i in target_indices
    ]
    return dict(
        status="arithmetic_recurrence_admitted",
        form="complete_observed_ffma257_template",
        native_parameter_abi=abi,
        measured_region=loops[0],
        complete_instruction_count=336,
        complete_opcounts=dict(Counter(x["opcode"] for x in instructions)),
        target_edges=target_edges,
        administration=[instructions[i] for i in (28, 29, 31, 288, 289)],
        clock_instructions=[instructions[27], instructions[294]],
        starting_clock_live_words=sorted(begin_words),
        population_gate=dict(
            compare=instructions[7],
            branch=instructions[13],
            warp_index=instructions[6],
            predicate_preserving_address_work=instructions[8:13],
        ),
        initial_guard=dict(
            checksum=instructions[19:22],
            compare=instructions[22],
            branch=instructions[23],
            completion_copies=instructions[24:27],
            expression="seed XOR multiplier XOR addend XOR expected XOR N",
            seed_load=instructions[15],
            expected_load=instructions[16],
        ),
        count_origin=dict(
            decremented="UR4",
            original_output_copy="R0",
            definitions=instructions[25:27],
        ),
        end_guard=instructions[290:294],
        terminal_exit=dict(
            call=instructions[288],
            backedge=instructions[289],
            destination=290,
            returns=[],
        ),
        convergence=[
            dict(
                setup=instructions[3],
                join=instructions[319],
                scope="All lanes, including idle warps",
            ),
            dict(
                setup=instructions[17],
                join=instructions[316],
                cancellation=instructions[292],
                scope="Successful lanes cancel B1; invalid lanes join B1",
            ),
        ],
        final_barrier=instructions[321],
        full_warp_join=instructions[320],
        address_proof=dict(
            output=instructions[8:11],
            seeds=instructions[11],
            expected=instructions[12],
            equations=[
                "row=blockIdx.x*blockDim.x+threadIdx.x",
                "output=argument0+64*row",
                "seeds=argument1+4*tid",
                "expected=argument2+4*tid",
            ],
        ),
        output_proof=dict(
            instructions=instructions[295:316],
            words=[
                "clock_begin_R6:R7",
                "clock_end_R12:R13",
                "end-minus-begin_R14:R15",
                "result_R9_zero_extended",
                "entry_SMID_R4_zero_extended",
                "exit_SMID_UR4_zero_extended",
                "success_one_64bit",
                "original_N_times_257_unsigned64",
            ],
            store_offsets=[0, 8, 16, 24, 32, 40, 48, 56],
            failure_status_store=instructions[318],
        ),
        unreachable_footer=instructions[323:],
        functional_trace=None,
        control_provenance=[
            (
                "https://docs.nvidia.com/cuda/cuda-binary-utilities/"
                "index.html#nvidia-ampere-gpu-and-ada-instruction-set"
            ),
            "https://arxiv.org/html/2407.02944v1#S5.SS5",
        ],
        scope=(
            "Exact observed FFMA257 instructions/ABI/dataflow/token form; "
            "MOV constant reload and all control remain measured work"
        ),
    )


def trace_native(
    instructions,
    targets,
    start,
    end,
    abi,
    dominance,
    helper,
    labels,
    initial_roots,
    terminal_calls,
):
    """Prove the retained reciprocal trace uses the timed native operation."""
    if len(targets) != TRACE_STEPS:
        raise ValueError("Need exactly 64 functional reciprocal operations")
    by_address = {item["address"]: i for i, item in enumerate(instructions)}
    indices = [by_address[item["address"]] for item in targets]
    first, last = indices[0] - 1, indices[-1] + 1
    region = instructions[first : last + 1]
    if first <= end or len(region) != 2 * TRACE_STEPS + 1:
        raise ValueError("Trace must be a separate linear load/store sequence")
    stores = region[::2]
    if any(
        item["full_opcode"] != "STG.E" or item["predicate"] for item in stores
    ):
        raise ValueError("Functional trace needs unpredicated 32-bit stores")
    addresses, values = [], []
    for item in stores:
        args = helper.operands(item)
        match = (
            re.fullmatch(r"\[(R[0-9]+)\.64(?:\+(0x[0-9a-f]+))?\]", args[0])
            if len(args) == 2
            else None
        )
        if match is None or register_operand(args[1]) is None:
            raise ValueError("Unproved functional trace store width/address")
        addresses.append((match.group(1), int(match.group(2) or "0", 16)))
        values.append(args[1])
    if addresses != [(addresses[0][0], 4 * i) for i in range(TRACE_STEPS + 1)]:
        raise ValueError("Functional trace byte offsets differ")
    if any(
        helper.written_registers(item)
        & {
            addresses[0][0],
            "R" + str(int(addresses[0][0][1:]) + 1),
        }
        for item in region
    ):
        raise ValueError("Functional trace clobbers its output address")
    edges = []
    for index, item in enumerate(targets):
        args = helper.operands(item)
        if (
            item["full_opcode"] != "MUFU.RCP"
            or item["predicate"]
            or args != [values[index + 1], values[index]]
        ):
            raise ValueError("Functional trace does not witness every result")
        edges.append(
            dict(
                instruction=item,
                predecessor=values[index],
                result_store=stores[index + 1],
            )
        )
    if any(i not in dominance[last] for i in range(first, last)):
        raise ValueError("A control edge bypasses a functional trace step")
    definitions = reaching_definitions(
        instructions,
        labels,
        terminal_calls,
        values[0],
        first,
        helper,
    )
    if len(definitions) != 1:
        raise ValueError("Functional trace seed differs")
    seed_index = next(iter(definitions))
    if (
        initial_roots != {"load:" + str(seed_index)}
        or instructions[seed_index]["full_opcode"] != "LDG.E"
        or helper.operands(instructions[seed_index])[0] != values[0]
    ):
        raise ValueError("Trace and counted body do not share the seed load")
    branches = []
    for i in range(start - 1):
        compare, branch = instructions[i : i + 2]
        args = helper.operands(compare)
        if (
            compare["full_opcode"] != "ISETP.NE.U32.AND"
            or compare["predicate"]
            or len(args) != 5
            or args[1] != "PT"
            or args[3:] != ["RZ", "PT"]
            or branch["full_opcode"] != "BRA"
            or branch["predicate"] != "@" + args[0]
            or i not in dominance[first]
            or i not in dominance[start]
        ):
            continue
        destination = helper.control_target(branch, labels)
        if destination > first or destination <= end:
            continue
        coefficient_origin(
            args[2],
            instructions,
            i,
            abi["trace_constant_offset"],
            dominance,
            helper,
        )
        branches.append(dict(compare=compare, branch=branch))
    if len(branches) != 1:
        raise ValueError("Functional trace lacks its runtime mode gate")
    return dict(
        target_edges=edges,
        stores=stores,
        trace_mode=branches[0],
        scope="Retained MUFU.RCP transition sequence; not a timing sample",
    )


def reaching_definitions(
    instructions, labels, calls, register, before, helper
):
    """Propagate all physical definitions over the admitted control graph."""
    incoming = [set() for _ in instructions]
    incoming[0] = {-1}
    changed = True
    while changed:
        changed = False
        for index, item in enumerate(instructions):
            if not incoming[index]:
                continue
            writes = register in helper.written_registers(item)
            outgoing = {index} if writes else set(incoming[index])
            if writes and item["predicate"]:
                outgoing |= incoming[index]
            targets = []
            if item["opcode"] == "BRA":
                targets.append(helper.control_target(item, labels))
            elif item["opcode"] == "CALL":
                if index not in calls:
                    raise ValueError("Unknown call in definition graph")
                targets.append(calls[index])
            if (
                item["opcode"] not in ("BRA", "CALL", "EXIT")
                or item["predicate"]
            ):
                if index + 1 < len(instructions):
                    targets.append(index + 1)
            for target in targets:
                difference = outgoing - incoming[target]
                if difference:
                    incoming[target] |= difference
                    changed = True
    return incoming[before]


def parse_native(directory, entry):
    """Use the exact frozen text parser without importing its CUDA module."""
    directory = Path(directory)
    source = directory / "hardware_source.py"
    if digest(source) != HARDWARE_SHA:
        raise ValueError("Saved native parser differs")
    names = {"INSTRUCTION_BYTES", "INSTRUCTION", "LABEL", "SECTION", "TARGET"}
    selected = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_sass":
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
    namespace = dict(re=re, collections=collections)
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    return namespace["_parse_sass"](
        (directory / "kernel.sass").read_text(),
        entry,
    )


def validate_trace(values, trace, seeds):
    """Reconstruct native reciprocal cycles from every retained CTA/lane."""
    if (
        values.dtype != np.uint64
        or values.ndim != 3
        or values.shape[1:] != (1024, 8)
        or trace.dtype != np.uint32
        or trace.shape != values.shape[:2] + (TRACE_STEPS + 1,)
    ):
        raise ValueError("Functional arrays have wrong dimensions/types")
    if not (
        np.all(values[:, :, 6] == 2)
        and np.all(values[:, :, 4] == values[:, :, 5])
        and np.all(values[:, :, 4] == values[:, :1, 4])
        and np.all(values[:, :, [0, 1, 2, 3, 7]] == np.uint64(2**64 - 1))
        and np.all(trace == trace[:1, :, :])
    ):
        raise ValueError("Functional status, SMID or CTA determinism differs")
    cycles = reciprocal_cycles(trace[0], seeds)
    return dict(
        cycles=cycles,
        steps=TRACE_STEPS,
        observed_smids=np.unique(values[:, :, 4]).tolist(),
        trace_sha256=hashlib.sha256(trace.tobytes()).hexdigest(),
    )


def duration_ok(row):
    """Require finite positive measured chain and event durations."""
    return all(
        isinstance(row.get(key), (int, float))
        and math.isfinite(row[key])
        and row[key] >= 20
        for key in ("event_ms", "minimum_chain_ms_at_observed_max_clock")
    )


def sample_order():
    """Mirror both population and count across two six-sample blocks."""
    first = [(1, 1), (1, 2), (32, 2), (32, 1)]
    return [
        (block, index, warps, multiple)
        for block in range(2)
        for index in range(6)
        for warps, multiple in (
            first if (block + index) % 2 == 0 else first[::-1]
        )
    ]


def validate_preparation(directory):
    """Rederive source, inputs and hardware plan from retained artifacts."""
    directory = Path(directory).resolve()
    request = json.loads((directory / "request.json").read_text())
    for filename, key in (
        ("benchmark_source.py", "generator_sha256"),
        ("worker.py", "worker_sha256"),
        ("kernel.py", "kernel_sha256"),
        ("seeds.npy", "seeds_sha256"),
        ("latency_source.py", "latency_sha256"),
        ("hardware_source.py", "hardware_source_sha256"),
        ("hardware_manifest.json", "hardware_manifest_sha256"),
    ):
        if digest(directory / filename) != request[key]:
            raise ValueError("Prepared bytes differ: " + filename)
    if request["latency_sha256"] != LATENCY_SHA:
        raise ValueError("Prepared helper is not the frozen helper")
    if request["hardware_source_sha256"] != HARDWARE_SHA:
        raise ValueError("Prepared native parser differs")
    if request["generator_sha256"] != digest(SCRIPT):
        raise ValueError("Reader differs from the measured source epoch")
    operation, body = request["operation"], request["body_operations"]
    if operation not in OPERATIONS or body not in (33, 257):
        raise ValueError("Unsupported prepared recurrence")
    source, primitive = source_text(operation, body)
    if (directory / "kernel.py").read_text(encoding="utf-8") != source or (
        directory / "primitive.ptx"
    ).read_text() != primitive:
        raise ValueError("Emitted kernel is not the declared construction")
    seeds, coefficients = inputs(operation, 1024)
    if (
        request["coefficients"] != coefficients
        or not np.array_equal(
            np.load(directory / "seeds.npy", allow_pickle=False), seeds
        )
        or np.load(directory / "seeds.npy", allow_pickle=False).dtype
        != np.uint32
    ):
        raise ValueError("Prepared operands differ from the exact oracle")
    manifest = json.loads((directory / "hardware_manifest.json").read_text())
    if (
        request["hardware_plan"] != plan(manifest)
        or request["block_size"] != 1024
        or type(request["waves"]) is not int
        or request["waves"] < 2
        or request["trace_steps"] != TRACE_STEPS
    ):
        raise ValueError("Prepared hardware/population plan differs")
    return request


def match_native_bank(prior, current, request):
    """Bind a fresh profile to the exact ordinary native/workload epoch."""
    previous_request = prior["request"]
    for field in IDENTITY_FIELDS:
        if request[field] != previous_request[field]:
            raise ValueError("Ordinary/profile source/input differs: " + field)
    for field in (
        "compilation_identity",
        "resources",
        "geometry",
        "native_admission",
        "artifacts",
    ):
        if current[field] != prior[field]:
            raise ValueError(
                "Ordinary/profile native identity differs: " + field
            )


def load_ordinary(directory):
    """Admit raw ordinary arrays, duration, work and native identity."""
    directory = Path(directory).resolve()
    request = validate_preparation(directory)
    value = json.loads((directory / "result.json").read_text())
    exit_record = json.loads((directory / "process_exit.json").read_text())
    if (
        request["mode"] != "ordinary"
        or value["status"] != "ordinary_complete"
        or not value["gpu_execution"]
        or not value["kernel_compilation"]
        or value["cleanup_errors"]
        or exit_record["returncode"] != 0
        or exit_record["forced"]
    ):
        raise ValueError("Ordinary process is incomplete or cleanup failed")
    if set(value["artifacts"]) != set(ARTIFACTS):
        raise ValueError(
            "Incomplete ordinary native/source artifact inventory"
        )
    for filename, checksum in value["artifacts"].items():
        if digest(directory / filename) != checksum:
            raise ValueError("Ordinary artifact bytes changed: " + filename)
    geometry, resources = value["geometry"], value["resources"]
    initial = value["initial_native"]
    identity = value["compilation_identity"]
    if (
        initial != value["final_native"]
        or initial["overloads"] != 1
        or initial["resident_blocks_per_sm"] != 1
        or initial["geometry"] != geometry
        or initial["cubin_sha256"] != value["artifacts"]["kernel.cubin"]
        or resources["cubin_sha256"] != initial["cubin_sha256"]
        or resources["entry"] != initial["entry"]
        or resources["local_bytes_per_thread"] != 0
        or resources["static_shared_bytes"] != 0
        or geometry["grid_blocks"] != geometry["sms"] * request["waves"]
        or geometry["block_size"] != request["block_size"]
        or geometry["waves"] != request["waves"]
        or geometry["resident_blocks_per_sm"] != 1
        or geometry["resident_warps_per_sm"] != 32
        or geometry["dynamic_shared_bytes"] != 0
        or geometry["thread_capacity_blocks"] != 1
        or identity["compute_capability"] != [8, 9]
        or identity["device_attributes"]["MULTIPROCESSOR_COUNT"]
        != geometry["sms"]
    ):
        raise ValueError("Ordinary native geometry/resources differ")
    raw = [
        json.loads(line)
        for line in (directory / "samples.jsonl").read_text().splitlines()
    ]
    if raw != value["samples"]:
        raise ValueError("Embedded and append-only sample inventories differ")
    measurements = [row for row in raw if row["phase"] == "measurement"]
    if [
        (row["block"], row["index"], row["warps"], row["multiple"])
        for row in measurements
    ] != sample_order():
        raise ValueError("Need all 48 ordered mirrored ordinary samples")
    calibration = [row for row in raw if row["phase"] == "calibration"]
    functional = [row for row in raw if row["phase"] == "functional_trace"]
    expected_trace = 1 if request["operation"] == "rcp" else 0
    if (
        len(functional) != expected_trace
        or not calibration
        or len(calibration) % 2
        or raw != functional + calibration + measurements
    ):
        raise ValueError("Ordinary phase membership differs")
    for index in range(len(calibration) // 2):
        pair = calibration[2 * index : 2 * index + 2]
        if (
            [row["warps"] for row in pair] != [1, 32]
            or any(
                row["index"] != index
                or row["block"] != -1
                or row["multiple"] != 1
                for row in pair
            )
            or pair[0]["iterations"] != pair[1]["iterations"]
        ):
            raise ValueError("Calibration does not contain paired populations")
    if any(
        row["iterations"] != value["iterations"] or not duration_ok(row)
        for row in calibration[-2:]
    ):
        raise ValueError("Final calibration differs from measured N")
    seeds = np.load(directory / "seeds.npy", allow_pickle=False)
    helper = helpers(directory / "latency_source.py")
    abi = parameter_abi(
        (directory / "native_elf.txt").read_text(), resources["entry"]
    )
    admission = check_native(
        *parse_native(directory, resources["entry"]),
        request["operation"],
        request["body_operations"],
        abi,
        helper,
    )
    admission = json.loads(json.dumps(admission))
    if admission != value["native_admission"] or admission != json.loads(
        (directory / "sass_analysis.json").read_text()
    ):
        raise ValueError("Saved native admission does not rederive")
    cycles = None
    for ordinal, row in enumerate(raw):
        if row["array_file"] != f"sample_{ordinal:04d}.npz":
            raise ValueError("Raw array order/path differs")
        path = directory / row["array_file"]
        if digest(path) != row["array_sha256"]:
            raise ValueError("Raw array bytes differ")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"values", "trace", "seeds", "expected"}:
                raise ValueError("Incomplete output/input arrays")
            if arrays["seeds"].dtype != np.uint32 or not np.array_equal(
                arrays["seeds"], seeds
            ):
                raise ValueError("Raw input seeds differ")
            if row["phase"] == "functional_trace":
                if (
                    row["iterations"] != 0
                    or row["warps"] != 32
                    or row["block"] != -1
                    or row["index"] != -1
                    or row["multiple"] != 1
                    or row["minimum_chain_ms_at_observed_max_clock"]
                    is not None
                ):
                    raise ValueError("Functional trace treated as timing")
                checks = validate_trace(
                    arrays["values"], arrays["trace"], seeds
                )
                cycles = checks["cycles"]
                expected = seeds
            else:
                if (
                    type(row["iterations"]) is not int
                    or not 0 < row["iterations"] < 2**31
                    or row["warps"] not in (1, 32)
                ):
                    raise ValueError("Invalid runtime work/population")
                expected = expected_bits(
                    request["operation"],
                    seeds,
                    request["coefficients"],
                    row["iterations"] * request["body_operations"],
                    cycles,
                )
                checks = validate_output(
                    arrays["values"],
                    expected,
                    row["iterations"],
                    row["warps"],
                    geometry,
                    request["body_operations"],
                )
                if (
                    arrays["trace"].dtype != np.uint32
                    or arrays["trace"].shape
                    != (geometry["grid_blocks"], 1024, TRACE_STEPS + 1)
                    or not np.all(arrays["trace"] == np.uint32(2**32 - 1))
                ):
                    raise ValueError("Timed launch has a functional trace")
                milliseconds = helper.chain_milliseconds(
                    checks,
                    row["clocks_before"],
                    row["clocks_after"],
                    row["event_ms"],
                    identity["gpu_uuid"],
                )
                if (
                    milliseconds
                    != row["minimum_chain_ms_at_observed_max_clock"]
                ):
                    raise ValueError(
                        "Stored duration is not raw-clock derived"
                    )
                if row["phase"] == "measurement" and (
                    row["iterations"] != value["iterations"] * row["multiple"]
                    or not duration_ok(row)
                ):
                    raise ValueError("Measurement duration/count differs")
            if arrays["expected"].dtype != np.uint32 or not np.array_equal(
                arrays["expected"], expected
            ):
                raise ValueError("Expected bits do not match the exact oracle")
        if (
            checks != row["output_checks"]
            or row["native_before"] != initial
            or row["native_after"] != initial
        ):
            raise ValueError("Raw output checks/native identity differ")
    if value.get("reciprocal_cycles") != cycles:
        raise ValueError("Ordinary cycles differ from the functional evidence")
    if not valid_repeats(
        request["operation"],
        seeds,
        request["coefficients"],
        value["iterations"],
        request["body_operations"],
        cycles,
    ):
        raise ValueError("N/2N outputs are not distinct bounded controls")
    value["request"] = request
    return value, dict(
        path=str(directory),
        result_sha256=digest(directory / "result.json"),
        request_sha256=digest(directory / "request.json"),
        samples_sha256=digest(directory / "samples.jsonl"),
        accepted_measurements=len(measurements),
        scope=(
            "Exact ordinary source/native/arrays/work; "
            "profiles remain separate"
        ),
    )


def main():
    """Prepare CPU artifacts, or explicitly execute their isolated worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hardware-manifest", type=Path, required=True)
    parser.add_argument("--operation", choices=OPERATIONS, default="ffma")
    parser.add_argument(
        "--body-operations", type=int, choices=(33, 257), default=257
    )
    parser.add_argument("--iterations", type=int, default=32769)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--nvdisasm")
    parser.add_argument("--cuobjdump")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--profile-multiplier", type=int, choices=(1, 2))
    parser.add_argument("--profile-warps", type=int, choices=(1, 32))
    parser.add_argument("--ordinary-dir", type=Path)
    args = parser.parse_args()
    if args.waves < 2 or not 0 < args.iterations < 2**30:
        parser.error("Need >=2 waves and a positive bounded count")
    if not (
        bool(args.profile_multiplier)
        == bool(args.profile_warps)
        == bool(args.ordinary_dir)
    ):
        parser.error("Profile count, population and ordinary bank are paired")
    manifest = json.loads(args.hardware_manifest.read_text())
    hardware_plan = plan(manifest)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, primitive = source_text(args.operation, args.body_operations)
    (output / "kernel.py").write_text(source, encoding="utf-8")
    (output / "primitive.ptx").write_text(primitive, encoding="utf-8")
    worker = SCRIPT.with_name("arithmetic_service_worker.py")
    (output / "worker.py").write_bytes(worker.read_bytes())
    (output / "benchmark_source.py").write_bytes(SCRIPT.read_bytes())
    frozen = SCRIPT.with_name("latency_probe.py")
    if digest(frozen) != LATENCY_SHA:
        raise ValueError("Prepared frozen helper differs")
    (output / "latency_source.py").write_bytes(frozen.read_bytes())
    parser_source = SCRIPT.with_name("hardware_probes.py")
    if digest(parser_source) != HARDWARE_SHA:
        raise ValueError("Frozen native parser source differs")
    (output / "hardware_source.py").write_bytes(parser_source.read_bytes())
    (output / "hardware_manifest.json").write_bytes(
        args.hardware_manifest.read_bytes()
    )
    seeds, coefficients = inputs(args.operation, hardware_plan["block_size"])
    np.save(output / "seeds.npy", seeds, allow_pickle=False)
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
        operation=args.operation,
        generator_sha256=digest(output / "benchmark_source.py"),
        worker_sha256=digest(output / "worker.py"),
        kernel_sha256=digest(output / "kernel.py"),
        latency_sha256=digest(output / "latency_source.py"),
        hardware_source_sha256=digest(output / "hardware_source.py"),
        hardware_manifest_sha256=digest(output / "hardware_manifest.json"),
        seeds_sha256=digest(output / "seeds.npy"),
        coefficients=coefficients,
        hardware_plan=hardware_plan,
        block_size=hardware_plan["block_size"],
        body_operations=args.body_operations,
        waves=args.waves,
        iterations=args.iterations,
        trace_steps=TRACE_STEPS,
        profile_multiplier=args.profile_multiplier,
        profile_warps=args.profile_warps,
        ordinary_dir=str(args.ordinary_dir.resolve())
        if args.ordinary_dir
        else None,
        nvdisasm=args.nvdisasm,
        cuobjdump=args.cuobjdump,
    )
    write_json(output / "request.json", request)
    compile(worker.read_text(), str(worker), "exec")
    validate_preparation(output)
    if request["mode"] == "source_only":
        print(
            json.dumps(
                dict(
                    status="prepared",
                    out=str(output),
                    native_compilation=False,
                    gpu_execution=False,
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
            code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            forced = True
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=10)
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
    if code or forced:
        raise SystemExit(code or 1)
    print(json.dumps(dict(status="worker_complete", out=str(output))))


if __name__ == "__main__":
    main()
