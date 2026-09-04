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
            f"@again bra stream_{name};",
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


def check_native(instructions, loops, labels, operations):
    """Require two disjoint repeated native streams with equal work.

    The parser arguments come from the frozen hardware probe's SASS parser.
    Every FFMA must update one of eight distinct recurrence registers;
    both native operand sequences must agree after register renaming.
    """
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
    for region in regions:
        body = instructions[region["start_index"] : region["end_index"] + 1]
        ffmas = [
            instruction
            for instruction in body
            if instruction["opcode"] == "FFMA"
        ]
        if len(ffmas) != operations:
            raise ValueError("Native FFMA count differs from requested body")
        allowed = {"FFMA", "IADD3", "UIADD3", "ISETP", "UISETP", "BRA"}
        if any(instruction["opcode"] not in allowed for instruction in body):
            raise ValueError("Repeated body has unapproved control or traffic")
        branches = [
            instruction
            for instruction in body
            if instruction["opcode"] == "BRA"
        ]
        if len(branches) != 1 or branches[0] is not body[-1]:
            raise ValueError("Body must contain only its final loop branch")
        branch = branches[0]
        target = LABEL.search(branch["text"])
        if (
            not branch["predicate"]
            or target is None
            or address_labels[target.group(1)] != region["start_address"]
        ):
            raise ValueError(
                "Final branch must conditionally repeat this body"
            )
        destinations = []
        for instruction in ffmas:
            if instruction["predicate"]:
                raise ValueError("FFMA work must be unconditional")
            operands = [
                part.strip()
                for part in instruction["text"].split(None, 1)[1].split(",")
            ]
            if len(operands) != 4 or operands[0] not in operands[1:3]:
                raise ValueError(
                    "FFMA does not preserve the accumulator chain"
                )
            destinations.append(operands[0])
        first = destinations[:8]
        if len(set(first)) != 8 or destinations != first * (operations // 8):
            raise ValueError("Native recurrence spacing is not eight chains")
        accumulators = set(first)
        for instruction in ffmas:
            operands = [
                part.strip()
                for part in instruction["text"].split(None, 1)[1].split(",")
            ]
            inputs = operands[1:]
            if sum(operand in accumulators for operand in inputs) != 1:
                raise ValueError(
                    "FFMA mixes the independent accumulator chains"
                )
        canonical.append(canonical_body(body, address_labels))
        admitted.append(
            dict(region, recurrence_registers=first, verified_ffmas=operations)
        )
    if canonical[0] != canonical[1]:
        raise ValueError("Native bodies differ after register-role renaming")
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
