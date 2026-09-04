"""Construct and gate two separately addressed CUDA instruction streams.

The default writes source only, without importing CUDA. Explicit
``--compile-only`` admits native structure without launching kernels.
``--execute`` runs mirrored ordinary samples at N and 2N repeats.
``--profile-mode`` requires a completed ordinary cohort and launches
exactly one selected arm, retaining its arrays and original binary gates.
No timing coefficient or physical instruction-cache domain is inferred.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
MODES = {"all_a": 0, "all_b": 1, "mixed": 2}
MIRROR = ("all_a", "all_b", "mixed", "mixed", "all_b", "all_a")
REGISTER = re.compile(r"\b(?:UR|UP|R|P)[0-9]+\b")
LABEL = re.compile(r"(\.L[\w.$]+)")


def digest(path):
    """Return an exact file-byte identity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write finite JSON to a retained evidence file."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def mlir_asm(text):
    """Escape a PTX fragment as one MLIR string attribute."""
    return (
        text.replace("\\", "\\5C")
        .replace('"', "\\22")
        .replace(
            "\n",
            "\\0A",
        )
    )


def make_source(operations):
    """Emit identical recurrent FFMAs at two requested PTX addresses.

    Inline PTX keeps the selector and repeated bodies inspectable. Only the
    subsequent disassembly gate can establish that native duplication and
    the dependency spacing survived compilation.
    """
    if operations <= 0 or operations % 8:
        raise ValueError("Each body needs a positive multiple of eight FFMAs")
    ptx = [
        "{",
        ".reg .f32 v<8>;",
        ".reg .u32 count;",
        ".reg .pred choose, again;",
    ]
    ptx += [f"mov.f32 v{i}, ${i + 1};" for i in range(8)]
    ptx += [
        "mov.u32 count, $11;",
        "setp.eq.u32 choose, $12, 0;",
        "@!choose bra stream_b;",
    ]
    for name in ("a", "b"):
        ptx.append(f"stream_{name}:")
        ptx += [
            f"fma.rn.f32 v{i % 8}, v{i % 8}, $9, $10;"
            for i in range(operations)
        ]
        ptx += [
            "add.u32 count, count, -1;",
            "setp.ne.u32 again, count, 0;",
            f"@again bra.uni stream_{name};",
        ]
        if name == "a":
            ptx.append("bra streams_done;")
    ptx.append("streams_done:")
    ptx += [f"add.rn.f32 v0, v0, v{i};" for i in range(1, 8)]
    ptx += ["mov.f32 $0, v0;", "}"]
    parameters = [f"%v{i}: f32" for i in range(8)]
    parameters += [
        "%multiplier: f32",
        "%increment: f32",
        "%iterations: i32",
        "%selection: i32",
    ]
    operands = [f"%v{i}" for i in range(8)]
    operands += ["%multiplier", "%increment", "%iterations", "%selection"]
    types = ["f32"] * 10 + ["i32"] * 2
    intrinsic = "\n".join(
        [
            "func.func private @dual_stream("
            + ", ".join(parameters)
            + ") -> f32 attributes {always_inline} {",
            '  %result = "llvm.inline_asm"(' + ", ".join(operands) + ") {",
            '    asm_string = "' + mlir_asm("\n".join(ptx)) + '",',
            '    constraints = "=f,'
            + ",".join(["f"] * 10 + ["r", "r"])
            + '", has_side_effects',
            "  } : (" + ", ".join(types) + ") -> f32",
            "  return %result : f32",
            "}",
        ]
    )
    smid = "\n".join(
        [
            "func.func private @read_smid() -> i32",
            "attributes {always_inline} {",
            '  %result = "llvm.inline_asm"() {',
            '    asm_string = "mov.u32 $0, %smid;",',
            '    constraints = "=r", has_side_effects',
            "  } : () -> i32",
            "  return %result : i32",
            "}",
        ]
    )
    lines = [
        "from numpy import uint32",
        "from cubie.cuda_simsafe import cuda, float32",
        "",
        f"read_smid = cuda.intrin.define({smid!r})",
        f"dual_stream = cuda.intrin.define({intrinsic!r})",
        "",
        "def probe(output, entry_smid, exit_smid, selections,",
        "          iterations, mode, mask, multiplier, increment):",
        "    thread = cuda.grid(1)",
        "    entry = uint32(read_smid())",
        "    chosen = uint32(0)",
        "    if mode == uint32(1):",
        "        chosen = uint32(1)",
        "    elif mode == uint32(2):",
        "        chosen = uint32((entry & mask) != uint32(0))",
    ]
    lines += [
        f"    value{i} = float32({i + 1}) "
        "+ float32(thread & 7) * float32(0.001)"
        for i in range(8)
    ]
    lines += [
        "    answer = dual_stream("
        + ", ".join([f"value{i}" for i in range(8)])
        + ",",
        "                         multiplier, increment, iterations, chosen)",
        "    exit_id = uint32(read_smid())",
        "    output[thread] = answer",
        "    entry_smid[thread] = entry",
        "    exit_smid[thread] = exit_id",
        "    selections[thread] = chosen",
    ]
    source = "\n".join(lines) + "\n"
    compile(source, "<instruction-sharing-source>", "exec")
    return source


def canonical_body(body, labels):
    """Retain native operand roles while removing register/label names."""
    registers = {}
    family_counts = Counter()

    def rename(match):
        token = match.group()
        if token not in registers:
            family = re.sub(r"[0-9]+", "", token)
            registers[token] = family + ":" + str(family_counts[family])
            family_counts[family] += 1
        return registers[token]

    result = []
    for instruction in body:
        text = instruction["predicate"] + " " + instruction["text"]
        text = REGISTER.sub(rename, text)
        text = LABEL.sub(
            lambda match: (
                "LOOP"
                if labels[match.group(1)] == body[0]["address"]
                else "OTHER"
            ),
            text,
        )
        result.append(text)
    return result


def native_operands(instruction):
    """Return operand tokens, retaining memory and immediate operands."""
    return [part.strip() for part in
            instruction["text"].split(None, 1)[1].split(",")]


def direct_target(instruction, address_labels):
    """Resolve one complete direct branch operand without indirect edges."""
    match = re.fullmatch(
        r"(?:BRA|CALL\.REL\.NOINC) `\((\.L[\w.$]+)\)",
        instruction["text"],
    )
    if match is None or match.group(1) not in address_labels:
        raise ValueError("Unresolved or unsupported direct control target")
    return address_labels[match.group(1)]


def check_counted_region(body, region, address_labels, operations):
    """Prove the observed constant-operand countdown and forward exit."""
    if len(body) != operations + 5:
        raise ValueError("Countdown region has unexpected instruction work")
    move = body[0]
    if move["full_opcode"] != "MOV" or move["predicate"]:
        raise ValueError("Region must begin with its unconditional MOV")
    multiplier = native_operands(move)
    if (len(multiplier) != 2
            or not re.fullmatch(r"R[0-9]+", multiplier[0])
            or multiplier[1] != "c[0x0][0x20c]"):
        raise ValueError("Multiplier MOV differs from the kernel parameter")
    increments = [item for item in body
                  if item["full_opcode"] == "UIADD3"]
    comparisons = [item for item in body
                   if item["full_opcode"] == "ISETP.NE.U32.AND"]
    if len(increments) != 1 or len(comparisons) != 1:
        raise ValueError("One exact countdown decrement and test required")
    decrement, comparison = increments[0], comparisons[0]
    counter = native_operands(decrement)[0]
    if (not re.fullmatch(r"UR[0-9]+", counter)
            or native_operands(decrement) != [counter, counter, "-0x1", "URZ"]
            or decrement["predicate"]):
        raise ValueError("Countdown must subtract one unconditionally")
    predicate = native_operands(comparison)[0]
    if (not re.fullmatch(r"P[0-9]+", predicate)
            or native_operands(comparison)
            != [predicate, "PT", "RZ", counter, "PT"]
            or comparison["predicate"]):
        raise ValueError("Countdown must compare updated count with zero")
    exit_control, backedge = body[-2:]
    if (exit_control["full_opcode"] != "CALL.REL.NOINC"
            or exit_control["predicate"] != "@!" + predicate
            or backedge["full_opcode"] != "BRA"
            or backedge["predicate"]
            or direct_target(backedge, address_labels)
            != region["start_address"]
            or not decrement["address"] < comparison["address"]
            < exit_control["address"]):
        raise ValueError("Countdown exit/backedge dependency differs")
    allowed_objects = {id(item) for item in
                       (move, decrement, comparison, exit_control, backedge)}
    ffmas = [item for item in body if id(item) not in allowed_objects]
    if len(ffmas) != operations:
        raise ValueError("Unexpected countdown region instructions")
    destinations = []
    for item in ffmas:
        if item["full_opcode"] != "FFMA" or item["predicate"]:
            raise ValueError("Repeated arithmetic must be plain ungated FFMA")
        operands = [value.removesuffix(".reuse")
                    for value in native_operands(item)]
        if (len(operands) != 4
                or not re.fullmatch(r"R[0-9]+", operands[0])
                or operands != [operands[0], operands[0], multiplier[0],
                                "c[0x0][0x210]"]):
            raise ValueError("FFMA accumulator/parameter operand differs")
        destinations.append(operands[0])
    first = destinations[:8]
    if (operations <= 0 or operations % 8 or len(set(first)) != 8
            or destinations != first * (operations // 8)
            or multiplier[0] in first):
        raise ValueError("Native recurrence spacing is not eight chains")
    return {
        "counter": counter, "predicate": predicate,
        "recurrence_registers": first,
        "multiplier_move": move, "decrement": decrement,
        "zero_comparison": comparison, "exit_control": exit_control,
        "backedge": backedge,
        "exit_target": direct_target(exit_control, address_labels),
        "constant_operands_per_body_traversal": {
            multiplier[1]: 1, "c[0x0][0x210]": operations,
        },
    }


def check_stream_paths(instructions, regions, controls, address_labels):
    """Prove the common initialized counter and nonreturning exit paths."""
    first, second = regions
    prefix = instructions[:first["start_index"]]
    counter = controls[0]["counter"]
    if controls[1]["counter"] != counter:
        raise ValueError("Streams do not use the same initialized counter")
    if (controls[0]["recurrence_registers"]
            != controls[1]["recurrence_registers"]):
        raise ValueError("Streams do not share their input/output registers")
    for item in prefix:
        if item["opcode"] not in {
            "MOV", "S2R", "ISETP", "IMAD", "LOP3", "I2FP", "BRA",
            "SEL", "ULDC", "FFMA",
        }:
            raise ValueError("Unsupported instruction in the dispatch prefix")
        if item["full_opcode"] == "ULDC.64":
            destination = native_operands(item)[0]
            if re.fullmatch(r"UR[0-9]+", destination):
                high = "UR" + str(int(destination[2:]) + 1)
                if counter in (destination, high):
                    raise ValueError("Wide parameter load clobbers counter")
    counter_uses = [item for item in prefix
                    if counter in REGISTER.findall(item["text"])]
    if (len(counter_uses) != 1
            or counter_uses[0]["full_opcode"] != "ULDC"
            or counter_uses[0]["predicate"]
            or native_operands(counter_uses[0])
            != [counter, "c[0x0][0x200]"]):
        raise ValueError("Counter lacks one unchanged uniform parameter load")
    initializer = counter_uses[0]
    dispatch = prefix[-1]
    if (dispatch["full_opcode"] != "BRA" or not dispatch["predicate"]
            or direct_target(dispatch, address_labels)
            != second["start_address"]):
        raise ValueError("Expected the sole two-stream dispatch branch")
    indices = {item["address"]: index
               for index, item in enumerate(instructions)}
    for index, item in enumerate(prefix):
        if item["opcode"] == "BRA":
            target = indices[direct_target(item, address_labels)]
            if (target <= index or (target >= len(prefix)
                                    and item is not dispatch)):
                raise ValueError("Alternative prefix/body control edge")
    pending = [(0, False)]
    visited = set()
    entries = set()
    while pending:
        index, initialized = pending.pop()
        if (index, initialized) in visited:
            continue
        visited.add((index, initialized))
        if index in (first["start_index"], second["start_index"]):
            if not initialized:
                raise ValueError("A stream can bypass counter initialization")
            entries.add(index)
            continue
        if not 0 <= index < len(prefix):
            raise ValueError("Prefix has an alternative body or exit entry")
        item = instructions[index]
        initialized = initialized or item is initializer
        if item["opcode"] == "BRA":
            if item["full_opcode"] != "BRA":
                raise ValueError("Unsupported dispatch branch modifier")
            target = indices[direct_target(item, address_labels)]
            if target <= index:
                raise ValueError("Prefix contains recurrent control")
            if (target in (first["start_index"], second["start_index"])
                    and item is not dispatch):
                raise ValueError("Alternative stream entry branch")
            pending.append((target, initialized))
            if item["predicate"]:
                pending.append((index + 1, initialized))
        elif item["opcode"] in {
            "CALL", "RET", "EXIT", "BRX", "JMP", "JMX", "BSSY",
            "BSYNC", "BREAK", "BMOV", "YIELD", "WARPSYNC",
        }:
            raise ValueError("Unsupported prefix control")
        else:
            pending.append((index + 1, initialized))
    if entries != {first["start_index"], second["start_index"]}:
        raise ValueError("Both native stream entries must remain reachable")
    common = second["end_index"] + 1
    bridge = instructions[first["end_index"] + 1:second["start_index"]]
    if (len(bridge) != 1 or bridge[0]["full_opcode"] != "BRA"
            or bridge[0]["predicate"]
            or controls[0]["exit_target"] != bridge[0]["address"]
            or direct_target(bridge[0], address_labels)
            != instructions[common]["address"]
            or controls[1]["exit_target"] != instructions[common]["address"]):
        raise ValueError("Stream exits do not join the same forward epilogue")
    epilogue = []
    for item in instructions[common:]:
        epilogue.append(item)
        if item["opcode"] == "EXIT":
            if item["predicate"] or item["full_opcode"] != "EXIT":
                raise ValueError("Common epilogue exit must be unconditional")
            break
        if item["opcode"] not in {
            "FADD", "SHF", "MOV", "IMAD", "S2R", "LEA", "IADD3", "STG"
        } or item["predicate"]:
            raise ValueError("Common epilogue is not the linear output path")
    if not epilogue or epilogue[-1]["opcode"] != "EXIT":
        raise ValueError("Common epilogue has no terminating EXIT")
    if Counter(item["opcode"] for item in epilogue)["FADD"] != 7:
        raise ValueError("Common epilogue lacks the eight-value reduction")
    if Counter(item["opcode"] for item in epilogue)["STG"] != 4:
        raise ValueError("Common epilogue lacks the four output stores")
    footer = instructions[common + len(epilogue):]
    if (not footer or footer[0]["full_opcode"] != "BRA"
            or footer[0]["predicate"]
            or direct_target(footer[0], address_labels) != footer[0]["address"]
            or any(item["full_opcode"] != "NOP" or item["predicate"]
                   for item in footer[1:])):
        raise ValueError("Unexpected code after the terminal EXIT")
    return {
        "counter_initializer": initializer, "dispatch": dispatch,
        "counter_constant_parameter": "c[0x0][0x200]",
        "common_epilogue_start": instructions[common]["address"],
        "common_epilogue_instructions": len(epilogue),
        "common_epilogue_opcounts": dict(Counter(
            item["opcode"] for item in epilogue)),
        "common_exit": epilogue[-1], "first_stream_exit_bridge": bridge,
        "fixed_extra_exit_branches": [1, 0],
        "prefix_instructions": prefix,
        "prefix_static_pc_union": len(prefix),
        "prefix_path_scope": "Acyclic dispatch; branch predicates may "
                             "select different fixed paths. No common "
                             "dynamic prefix count is assumed.",
        "logical_traversals": "N per selected lane for initial 0 < N < 2^32",
        "per_selected_lane": {
            "counter_decrements": "N", "zero_tests": "N",
            "exit_guard_visits": "N", "exit_transfers": 1,
            "taken_backedges": "N - 1",
        },
        "scope": "Logical CFG counts; SourceCounters must verify native "
                 "warp/thread units and predication. Prefix selection and "
                 "one first-stream exit BRA are fixed path differences.",
    }


def check_native(instructions, loops, labels, operations):
    """Require two disjoint repeated native streams with equal work.

    The parser arguments come from the frozen hardware probe's SASS parser.
    Each native countdown must visit eight independent FFMA chains and
    terminate at the same linear output path. Constant parameter reads
    and the first stream's fixed exit bridge remain explicit work.
    """
    if (not instructions or instructions[0]["address"] != 0
            or any(right["address"] != left["address"] + 16
                   for left, right in zip(instructions, instructions[1:]))):
        raise ValueError("Expected contiguous 16-byte native instructions")
    regions = [loop for loop in loops if loop["opcounts"].get("FFMA")]
    if len(regions) != 2:
        raise ValueError("Native code must retain exactly two FFMA loops")
    regions.sort(key=lambda region: region["start_address"])
    if regions[0]["end_address_exclusive"] > regions[1]["start_address"]:
        raise ValueError("Native streams overlap or contain nested loops")
    address_labels = {
        label: instructions[index]["address"]
        for label, index in labels.items()
        if index < len(instructions)
    }
    canonical = []
    admitted = []
    controls = []
    for region in regions:
        body = instructions[region["start_index"] : region["end_index"] + 1]
        if (len(body) != region["instructions"]
                or dict(Counter(item["opcode"] for item in body))
                != region["opcounts"]
                or body[0]["address"] != region["start_address"]
                or body[-1]["address"] + 16
                != region["end_address_exclusive"]
                or region["bytes"] != len(body) * 16):
            raise ValueError("Parsed region and actual native bytes differ")
        control = check_counted_region(
            body, region, address_labels, operations
        )
        controls.append(control)
        canonical.append(canonical_body(body, address_labels))
        admitted.append(
            dict(region, **control, verified_ffmas=operations)
        )
    if canonical[0] != canonical[1]:
        raise ValueError("Native bodies differ after register-role renaming")
    paths = check_stream_paths(instructions, regions, controls, address_labels)
    outside_ffmas = (
        sum(instruction["opcode"] == "FFMA" for instruction in instructions)
        - 2 * operations
    )
    smids = [
        instruction
        for instruction in instructions
        if any(
            name in instruction["text"]
            for name in ("SR_SMID", "SR_VIRTUALSMID")
        )
    ]
    if (
        len(smids) != 2
        or smids[0]["address"] >= regions[0]["start_address"]
        or smids[1]["address"] < regions[1]["end_address_exclusive"]
    ):
        raise ValueError("Native SMID reads do not bracket both streams")
    return dict(
        admitted=True,
        bodies=admitted,
        countdown_and_exit_proof=paths,
        smid_reads=smids,
        canonical_body_sha256=hashlib.sha256(
            "\n".join(canonical[0]).encode()
        ).hexdigest(),
        whole_opcounts=dict(Counter(item["opcode"] for item in instructions)),
        whole_encoded_bytes=len(instructions) * 16,
        fixed_outside_ffmas=outside_ffmas,
        workload_denominator="Hot warp FFMAs; fixed prologue/output excluded",
        instruction_bytes_basis="16-byte SM89 native encoding",
        native_stream_to_runtime_arm=(
            "Address order alone does not label A/B; profile executed PCs "
            "must bind each runtime selection to one admitted stream."
        ),
    )


def main():
    """Write source, or explicitly request a compiled/real-GPU worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--profile-mode", choices=MODES)
    parser.add_argument("--ordinary-dir", type=Path)
    parser.add_argument(
        "--body-kib",
        type=int,
        default=80,
        help="Requested FFMA bytes/stream; actual hot span gated",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4096,
        help="Initial runtime repeat count; ordinary may grow it",
    )
    parser.add_argument(
        "--smid-mask",
        type=int,
        default=1,
        help="Recorded runtime SMID bit mask, not a topology map",
    )
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument(
        "--blocks",
        type=int,
        default=2,
        help="Paired measurement blocks, six samples/arm/block",
    )
    parser.add_argument("--nvdisasm")
    args = parser.parse_args()
    if not (
        1 <= args.body_kib <= 256
        and 0 < args.iterations < 2**30
        and 0 < args.smid_mask < 2**32
        and args.waves >= 2
        and args.blocks >= 2
    ):
        parser.error(
            "Require body 1..256 KiB, repeats 1..2^30-1, positive "
            "mask and >=2 waves/blocks"
        )
    if bool(args.profile_mode) != bool(args.ordinary_dir):
        parser.error("--profile-mode and --ordinary-dir are required together")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    operations = args.body_kib * 1024 // 16
    source = make_source(operations)
    (output / "kernel.py").write_text(source, encoding="utf-8")
    worker = SCRIPT.with_name("instruction_sharing_worker.py")
    (output / "worker.py").write_bytes(worker.read_bytes())
    (output / "benchmark_source.py").write_bytes(SCRIPT.read_bytes())
    request = dict(
        schema=1,
        mode="profile"
        if args.profile_mode
        else "ordinary"
        if args.execute
        else "compile_only"
        if args.compile_only
        else "source_only",
        research_root=str(REPO),
        generator_path=str(SCRIPT),
        generator_sha256=digest(SCRIPT),
        worker_sha256=digest(worker),
        kernel_source_sha256=digest(output / "kernel.py"),
        requested_body_kib=args.body_kib,
        operations_per_body=operations,
        chains=8,
        block_size=256,
        active_lanes=256,
        waves=args.waves,
        blocks=args.blocks,
        iterations=args.iterations,
        smid_mask=args.smid_mask,
        minimum_ms=20.0,
        profile_mode=args.profile_mode,
        ordinary_dir=str(args.ordinary_dir.resolve())
        if args.ordinary_dir
        else None,
        nvdisasm=args.nvdisasm,
        semantics={
            "repeat": "Runtime loop count, independent of body bytes",
            "smid": "Observed IDs can be noncontiguous or migrate",
            "admission": "Native duplication/coverage/equal work required",
            "cache_domain": "Unknown; neither parity nor capacity assumed",
        },
    )
    write_json(output / "request.json", request)
    if request["mode"] == "source_only":
        print(
            json.dumps(
                dict(
                    status="source_only",
                    out=str(output),
                    operations_per_body=operations,
                )
            )
        )
        return
    with (output / "worker.stdout.log").open("w", encoding="utf-8") as out:
        with (output / "worker.stderr.log").open("w", encoding="utf-8") as err:
            completed = subprocess.run(
                [sys.executable, str(output / "worker.py")],
                cwd=REPO,
                stdout=out,
                stderr=err,
                timeout=14400,
            )
    print(json.dumps(dict(out=str(output), returncode=completed.returncode)))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
