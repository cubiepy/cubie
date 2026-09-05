"""Measure repeated complete clock/victim/checksum instruction regions."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from benchmarks.hardware_model.collective_service_probe import (
    exact_median,
    sha,
    write,
)


CHAINS = 8
SENTINEL = 0xDEADBEEF
ENTRY_PADDING_FFMAS = (0, 8, 16, 32, 64, 128, 256, 512, 1024)
RUNTIME_PACK_FIELDS = (
    "endpoint_pointer", "timestamp_pointer", "iterations",
    "multiplier_bits", "increment_bits", "endpoint_bound", "warm_mode",
)


def fmas(count):
    """Generate eight distinct observable FP32 recurrence chains."""
    if count < CHAINS or count % CHAINS:
        raise ValueError("Body FFMA count must be a positive multiple of 8")
    return "\n".join(
        f"    fma.rn.f32 x{i % CHAINS}, x{i % CHAINS}, fm, fa;"
        for i in range(count)
    )


def checksum(guard=True):
    """Bind all chains to the output and the ending clock control path."""
    lines = [
        "    add.rn.f32 total, x0, x1;",
        *[f"    add.rn.f32 total, total, x{i};" for i in range(2, CHAINS)],
        "    cvt.rzi.u32.f32 value, total;",
    ]
    if guard:
        lines.extend(["    setp.gt.u32 bad, value, bound;",
                      "    @bad bra INVALID;"])
    return "\n".join(lines)


def source(kind, body_ffmas, victim_blocks, padding_ffmas,
           entry_padding_ffmas=1024):
    """Return PTX with explicit timed-region labels and register operands."""
    if kind != "victim" or victim_blocks != 1:
        raise ValueError("Complete-region source requires one victim")
    header = """.version 8.0
.target sm_89
.address_size 64
.visible .entry instruction_probe(
    .param .u64 runtime_pack, .param .u32 active_warps,
    .param .u32 seed)
.maxntid 1024, 1, 1
{
    .reg .pred inactive, again, bad, lane_nonzero, priming;
    .reg .u32 tid, cta, lane, warp, n, count, population;
    .reg .u32 initial, value, bound, thread_index, warp_index, time_index;
    .reg .u32 multiplier, increment, temporary, warm, phase;
    .reg .u64 out, times, address, begin, end, ticks, byte_offset;
    .reg .u64 controls;
    .reg .f32 x0, x1, x2, x3, x4, x5, x6, x7, fm, fa, total;
    ld.param.u64 controls, [runtime_pack];
    ld.volatile.global.u64 out, [controls];
    ld.volatile.global.u64 times, [controls+8];
    ld.volatile.global.u32 n, [controls+16];
    ld.volatile.global.u32 multiplier, [controls+24];
    ld.volatile.global.u32 increment, [controls+32];
    ld.volatile.global.u32 bound, [controls+40];
    ld.volatile.global.u32 warm, [controls+48];
    ld.param.u32 population, [active_warps];
    ld.param.u32 initial, [seed];
    mov.u32 tid, %tid.x;
    mov.u32 cta, %ctaid.x;
    and.b32 lane, tid, 31;
    shr.u32 warp, tid, 5;
    // Volatile global values remain register operands in the hot region.
    mov.b32 fm, multiplier;
    mov.b32 fa, increment;
    setp.ge.u32 inactive, warp, population;
    bar.sync 0;
    @inactive bra JOIN;
"""
    initial = "\n".join(
        f"    add.u32 temporary, initial, {i};\n"
        f"    cvt.rn.f32.u32 x{i}, temporary;"
        for i in range(CHAINS)
    )
    index = """
    mad.lo.u32 thread_index, cta, 1024, tid;
    mad.lo.u32 warp_index, cta, 32, warp;
    mov.u32 count, 0;
"""
    if kind == "stream":
        body = """
TIMED_ENTRY:
    mov.u64 begin, %clock64;
STREAM:
""" + fmas(body_ffmas) + """
    add.u32 count, count, 1;
    setp.lt.u32 again, count, n;
    @again bra STREAM;
""" + checksum() + """
    mov.u64 end, %clock64;
    sub.u64 ticks, end, begin;
    mov.u32 time_index, warp_index;
    setp.ne.u32 lane_nonzero, lane, 0;
    @lane_nonzero bra OUTPUT;
    mul.wide.u32 byte_offset, time_index, 8;
    add.u64 address, times, byte_offset;
    st.global.u64 [address], ticks;
    bra OUTPUT;
"""
    elif kind == "victim":
        chunks = []
        for block in range(victim_blocks):
            chunks.append(f"VICTIM_{block}:\n" + fmas(CHAINS))
            if block + 1 < victim_blocks:
                chunks.append(
                    "    cvt.rzi.u32.f32 value, x0;\n"
                    "    setp.le.u32 bad, value, bound;\n"
                    f"    @bad bra VICTIM_{block + 1};\n"
                    f"PADDING_{block}:\n" + fmas(padding_ffmas)
                    + "\n" + checksum(False) + "\n    bra OUTPUT;"
                )
        victim = "\n".join(chunks)
        entry_padding = (fmas(entry_padding_ffmas)
                         if entry_padding_ffmas else "")
        body = """
TRIAL:
AGGRESSOR:
""" + fmas(body_ffmas) + """
    mov.u32 phase, warm;
TIMED_ENTRY:
    mov.u64 begin, %clock64;
    setp.le.u32 bad, warm, 1;
    @bad bra VICTIM_0;
    bra CONTROLLER_PADDING;
CONTROLLER_PADDING:
""" + entry_padding + "\n" + checksum(False) + """
    bra OUTPUT;
""" + victim + """
""" + checksum() + """
    mov.u64 end, %clock64;
    // Discard one complete interval in warm mode, then retain the repeat.
    setp.ne.u32 priming, phase, 0;
    mov.u32 phase, 0;
    @priming bra TIMED_ENTRY;
    sub.u64 ticks, end, begin;
    mad.lo.u32 time_index, warp_index, n, count;
    setp.ne.u32 lane_nonzero, lane, 0;
    @lane_nonzero bra NEXT_TRIAL;
    mul.wide.u32 byte_offset, time_index, 8;
    add.u64 address, times, byte_offset;
    st.global.u64 [address], ticks;
NEXT_TRIAL:
    add.u32 count, count, 1;
    setp.lt.u32 again, count, n;
    @again bra TRIAL;
"""
    else:
        raise ValueError("Unknown instruction probe kind")
    footer = """
OUTPUT:
    mul.wide.u32 byte_offset, thread_index, 4;
    add.u64 address, out, byte_offset;
    st.global.u32 [address], value;
    bra JOIN;
INVALID:
    trap;
JOIN:
    // Park inactive warps and retain the complete 1024-thread CTA.
    bar.sync 0;
    ret;
}
"""
    return header + initial + index + body + footer


def parameters(args):
    """Bind requested FFMA payload bytes separately from actual hot span."""
    if args.body_kib < 1:
        raise ValueError("Positive FFMA payload size required")
    if args.kind != "victim" or args.victim_blocks != 1:
        raise ValueError("Complete-region probe requires one eight-FFMA victim")
    if args.padding_ffmas < 8 or args.padding_ffmas % 8:
        raise ValueError("Padding must contain aligned eight-chain groups")
    if args.entry_padding_ffmas not in ENTRY_PADDING_FFMAS:
        raise ValueError("Entry padding must be one of the declared gap cases")
    return dict(kind=args.kind, body_ffmas=args.body_kib * 64,
                requested_ffma_payload_bytes=args.body_kib * 1024,
                victim_blocks=args.victim_blocks,
                victim_ffmas=8 * args.victim_blocks,
                padding_ffmas=args.padding_ffmas,
                entry_padding_ffmas=args.entry_padding_ffmas,
                requested_entry_padding_bytes=16 * args.entry_padding_ffmas,
                chains=CHAINS, optimization_level=args.optimization_level,
                warm_contract="repeat_complete_timed_region",
                clocks_per_trial_by_warm=[1, 2], retained_intervals_per_trial=1)


def native_layout(sass):
    """Record observed PCs and control forms without issuing a native PASS."""
    instructions = {int(pc, 16): text.strip() for pc, text in re.findall(
        r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;', sass)}
    clocks = [pc for pc, text in instructions.items()
              if text.startswith("CS2R ") and "SR_CLOCKLO" in text]
    if len(clocks) != 2:
        raise ValueError("Complete-region probe needs two static native clocks")
    ffmas = [pc for pc, text in instructions.items()
             if clocks[0] < pc < clocks[1] and text.startswith("FFMA ")]
    if len(ffmas) < 8:
        raise ValueError("Native clock span has fewer than eight victim FFMAs")
    controls = [dict(pc=pc, instruction=text)
                for pc, text in instructions.items()
                if clocks[0] <= pc <= clocks[1]
                and any(op in text for op in ("BRA ", "CALL.", "ISETP.", "PLOP3."))]
    after = []
    for pc, text in instructions.items():
        if pc <= clocks[1]:
            continue
        after.append(dict(pc=pc, instruction=text))
        if "BRA " in text or "CALL." in text:
            break
    return dict(
        status="observed_layout_requires_independent_control_review",
        clock_pcs=clocks, candidate_victim_pcs=ffmas[-8:],
        candidate_victim_gap_bytes=ffmas[-8] - clocks[0],
        static_ffmas_in_clock_span=len(ffmas),
        control_in_clock_span=controls, instructions_after_end_to_branch=after,
        interpretation="Static PCs are observed; executed padding and warm "
        "phase ordering require the independent native path proof",
    )


def prepare(args):
    """Write source, optionally assemble and disassemble without launching."""
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = parameters(args)
    ptx = root / "kernel.ptx"
    ptx.write_text(source(spec["kind"], spec["body_ffmas"],
                          spec["victim_blocks"], spec["padding_ffmas"],
                          spec["entry_padding_ffmas"]))
    shutil.copyfile(__file__, root / "probe_source.py")
    write(root / "source_specification.json", dict(
        **spec, generator_sha256=sha(__file__), ptx_sha256=sha(ptx),
        launched=False,
        runtime_pack=dict(fields=RUNTIME_PACK_FIELDS, stride_bytes=8,
                          pointer_load_bytes=8, scalar_load_bytes=4),
        timed_scope=("stream loop plus endpoint/control administration"
                     if spec["kind"] == "stream" else
                     "TIMED_ENTRY to ending clock through identical victim "
                     "PCs; warm mode discards a complete first interval and retains the repeat"),
        native_review_requirements=[
            "Every FFMA uses the two global-load-produced register operands",
            "Eight distinct recurrences survive and feed exact endpoint guard",
            "Hot bodies contain no recurrent IMC/LDC/data/spill source",
            "Warm and cold traverse identical begin/victim/checksum/end PCs",
            "Both native clocks execute N*(1+warm) times per active warp",
            "Warm toggle and repeat branch occur after the ending clock",
            "Native target gaps and actual hot span are measured, not assumed",
            "Timed controller/victim native PCs and their gap are recorded",
            "Padding output path executes zero times for admitted inputs",
            "End clock is reached only after the endpoint-dependent branch",
            "1024-thread CTA has zero local bytes and one resident block/SM",
        ],
    ))
    if args.mode == "emit":
        print(root)
        return
    assembler = args.ptxas or shutil.which("ptxas")
    disassembler = args.nvdisasm or shutil.which("nvdisasm")
    if not assembler or not disassembler:
        raise ValueError("Set --ptxas and --nvdisasm or place them on PATH")
    commands = []
    for command in (
        [str(assembler), "-arch=sm_89", f"-O{args.optimization_level}", "-v", str(ptx), "-o",
         str(root / "kernel.cubin")],
        [str(disassembler), "-c", str(root / "kernel.cubin")],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        commands.append(dict(command=command, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
        write(root / "commands.json", commands)
        result.check_returncode()
    (root / "kernel.sass").write_text(commands[-1]["stdout"])
    write(root / "preparation.json", dict(
        **spec, generator_sha256=sha(__file__), ptx_sha256=sha(ptx),
        cubin_sha256=sha(root / "kernel.cubin"),
        sass_sha256=sha(root / "kernel.sass"),
        commands_sha256=sha(root / "commands.json"),
        exact_assembler_command=commands[0]["command"],
        native_layout_observation=native_layout(commands[-1]["stdout"]),
        assembler_sha256=sha(assembler), disassembler_sha256=sha(disassembler),
        native_review_required=True, gpu_execution=False,
        service_assignment="unassigned_pending_native_and_counter_review",
    ))
    print(root)


def expected_value(preparation, iterations, seed, warm):
    """Compute an exact integer endpoint in the FP32 consecutive range."""
    per_trial = preparation["body_ffmas"]
    if preparation["kind"] == "victim":
        per_trial += preparation["victim_ffmas"] * (1 + warm)
    expected = CHAINS * seed + 28 + iterations * per_trial
    if expected >= 2**24:
        raise ValueError("Probe would leave the exactly represented FP32 range")
    return expected


def run(args):
    """Launch only a hash-bound, independently inspected native image."""
    from cubie.cuda_simsafe import cupy

    root = args.out.resolve()
    prep = json.loads((root / "preparation.json").read_text())
    review = json.loads(args.review.read_text())
    if (review["status"] != "PASS"
            or review["cubin_sha256"] != sha(root / "kernel.cubin")
            or review["sass_sha256"] != sha(root / "kernel.sass")
            or review["ptx_sha256"] != prep["ptx_sha256"]
            or prep["generator_sha256"] != sha(__file__)
            or review.get("actual_hot_body") is None):
        raise ValueError("Independent native/source review identity differs")
    if not 1 <= args.iterations <= 2**31 - 1 or args.pairs < 1:
        raise ValueError("Positive uint32-safe N and pair count required")
    populations = [int(value) for value in args.populations.split(",")]
    if any(value not in (1, 4, 8, 16, 32) for value in populations):
        raise ValueError("Population must be 1, 4, 8, 16 or 32")
    warms = (0,) if prep["kind"] == "stream" else (0, 1)
    device = cupy.cuda.Device()
    attrs = device.attributes
    if device.compute_capability != "89":
        raise ValueError("Instruction probes require SM89")
    module = cupy.RawModule(path=str(root / "kernel.cubin"))
    kernel = module.get_function("instruction_probe")
    resident = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        kernel.kernel.ptr, 1024, 0)
    if resident != 1 or kernel.attributes["local_size_bytes"] != 0:
        raise ValueError("Expected one resident CTA and no local allocation")
    blocks = 2 * attrs["MultiProcessorCount"]
    output = root / args.label
    output.mkdir(exist_ok=False)
    write(output / "launch.json", dict(
        preparation=prep, review_sha256=sha(args.review),
        attributes=attrs, kernel_attributes=kernel.attributes,
        block_threads=1024, grid_blocks=blocks, resident_blocks_per_sm=resident,
        allocated_warps_per_cta=32, full_occupancy_waves=2,
        timed_populations=populations, iterations=args.iterations,
        multiplier_bits=0x3F800000, increment_bits=0x3F800000,
        endpoint_bound=2**24 - 1, warm_modes=warms,
        runtime_pack=dict(fields=RUNTIME_PACK_FIELDS, stride_bytes=8,
                          pointer_load_bytes=8, scalar_load_bytes=4),
        hot_scope=review["actual_hot_body"],
        service_assignment="unassigned",
    ))
    if args.profile_once:
        jobs = [(populations[0], args.warm, 1, args.factor)]
    else:
        jobs = [(population, warm, pair, factor)
                for population in populations for pair in range(args.pairs + 1)
                for warm in (warms if pair % 2 == 0 else tuple(reversed(warms)))
                for factor in ((1, 2) if pair % 2 == 0 else (2, 1))]
    records = []
    for population, warm, pair, factor in jobs:
        count = args.iterations * factor
        seed = pair % 7
        expected = expected_value(prep, count, seed, warm)
        samples = count if prep["kind"] == "victim" else 1
        if blocks * 32 * samples >= 2**32:
            raise ValueError("Timestamp index exceeds uint32 address product")
        endpoints = cupy.full((blocks, 1024), SENTINEL, dtype=np.uint32)
        ticks = cupy.zeros((blocks, 32, samples), dtype=np.uint64)
        controls = cupy.asarray(np.array([
            endpoints.data.ptr, ticks.data.ptr, count,
            0x3F800000, 0x3F800000, 2**24 - 1, warm,
        ], dtype=np.uint64))
        begin, end = cupy.cuda.Event(), cupy.cuda.Event()
        begin.record()
        kernel((blocks,), (1024,), (
            controls, np.uint32(population), np.uint32(seed),
        ))
        end.record()
        end.synchronize()
        actual, clocks = endpoints.get(), ticks.get()
        wanted = np.full((blocks, 1024), SENTINEL, dtype=np.uint32)
        wanted[:, :population * 32] = expected
        np.testing.assert_array_equal(actual, wanted)
        if not np.all(clocks[:, :population, :] > 0):
            raise ValueError("Every active warp/trial must report elapsed clocks")
        np.testing.assert_array_equal(
            clocks[:, population:, :],
            np.zeros((blocks, 32 - population, samples), dtype=np.uint64))
        name = f"w{population}_warm{warm}_p{pair}_f{factor}"
        path = output / (name + ".npz")
        np.savez(path, endpoints=actual, clocks=clocks)
        median = exact_median(clocks[:, :population, :])
        records.append(dict(
            name=name, population=population, warm=warm, pair=pair,
            factor=factor, iterations=count, seed=seed, expected=expected,
            warmup=pair == 0 and not args.profile_once,
            event_ms=cupy.cuda.get_elapsed_time(begin, end),
            median_clocks=[median.numerator, median.denominator],
            array_sha256=sha(path), all_endpoints_exact=True,
        ))
        write(output / "records.json", records)
        if not args.profile_once:
            time.sleep(0.1)
    summaries = []
    if not args.profile_once and prep["kind"] == "stream":
        for population in populations:
            for pair in range(1, args.pairs + 1):
                rows = {row["factor"]: row for row in records
                        if row["population"] == population
                        and row["pair"] == pair}
                interval = (Fraction(*rows[2]["median_clocks"])
                            - Fraction(*rows[1]["median_clocks"])) / (
                                args.iterations * prep["body_ffmas"])
                summaries.append(dict(
                    population=population, pair=pair,
                    cycles_per_warp_ffma_including_control=[
                        interval.numerator, interval.denominator],
                    service_assignment="stream_composite_not_refill_latency",
                ))
    elif not args.profile_once:
        for population in populations:
            for pair in range(1, args.pairs + 1):
                for factor in (1, 2):
                    rows = {row["warm"]: row for row in records
                            if row["population"] == population
                            and row["pair"] == pair
                            and row["factor"] == factor}
                    delta = (Fraction(*rows[0]["median_clocks"])
                             - Fraction(*rows[1]["median_clocks"]))
                    summaries.append(dict(
                        population=population, pair=pair, factor=factor,
                        cold_minus_warm_victim_composite=[
                            delta.numerator, delta.denominator],
                    service_assignment="eviction_and_L2_state_unverified",
                    ))
    write(output / "summary.json", summaries)
    print(json.dumps(summaries))


def main():
    """Keep source emission, native preparation and GPU measurement separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("emit", "prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("victim",), default="victim")
    parser.add_argument("--body-kib", type=int, default=32)
    parser.add_argument("--victim-blocks", type=int, default=1)
    parser.add_argument("--padding-ffmas", type=int, default=16)
    parser.add_argument("--entry-padding-ffmas", type=int, default=1024,
                        choices=ENTRY_PADDING_FFMAS)
    parser.add_argument("--ptxas", type=Path)
    parser.add_argument("--optimization-level", type=int, choices=(0, 3), default=3)
    parser.add_argument("--nvdisasm", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--populations", default="1")
    parser.add_argument("--label", default="ordinary_e1")
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument("--factor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warm", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.mode == "run" and args.review is None:
        parser.error("GPU run requires an independent --review receipt")
    (run if args.mode == "run" else prepare)(args)


if __name__ == "__main__":
    main()
