"""Measure a retained dependent predicate-vote chain on native SM89.

PTX assembly and native inspection precede the separate measurement mode.
This probe measures a repeated motif, including its loop administration.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def exact_median(values):
    ordered = sorted(int(item) for item in np.asarray(values).flat)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle])
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def source(body, form="vote_all"):
    motif = {
        "vote_all": "    vote.sync.all.pred p, !p, 0xffffffff;",
        "ballot_compare": (
            "    vote.sync.ballot.b32 voted, p, 0xffffffff;\n"
            "    setp.eq.u32 p, voted, 0;"
        ),
    }[form]
    votes = "\n".join(
        motif for _ in range(body)
    )
    return """.version 8.0
.target sm_89
.address_size 64
.visible .entry collective_probe(
    .param .u64 endpoints, .param .u64 elapsed,
    .param .u32 iterations, .param .u32 active_warps,
    .param .u32 seed, .param .u32 endpoint_bound)
.maxntid 1024, 1, 1
{
    .reg .pred p, inactive, again, bad, lane_nonzero;
    .reg .u32 tid, cta, lane, warp, n, count, population;
    .reg .u32 initial, value, bound, thread_index, warp_index, voted;
    .reg .u64 out, times, address, begin, end, ticks, byte_offset;
    ld.param.u64 out, [endpoints];
    ld.param.u64 times, [elapsed];
    ld.param.u32 n, [iterations];
    ld.param.u32 population, [active_warps];
    ld.param.u32 initial, [seed];
    ld.param.u32 bound, [endpoint_bound];
    mov.u32 tid, %tid.x;
    mov.u32 cta, %ctaid.x;
    and.b32 lane, tid, 31;
    shr.u32 warp, tid, 5;
    setp.ge.u32 inactive, warp, population;
    bar.sync 0;
    @inactive bra JOIN;
    setp.ne.u32 p, initial, 0;
    mov.u32 count, 0;
    mov.u64 begin, %clock64;
LOOP:
""" + votes + """
    add.u32 count, count, 1;
    setp.lt.u32 again, count, n;
    @again bra LOOP;
    selp.u32 value, 1, 0, p;
    setp.gt.u32 bad, value, bound;
    @bad bra INVALID;
    mov.u64 end, %clock64;
    sub.u64 ticks, end, begin;
    mad.lo.u32 thread_index, cta, 1024, tid;
    mul.wide.u32 byte_offset, thread_index, 4;
    add.u64 address, out, byte_offset;
    st.global.u32 [address], value;
    setp.ne.u32 lane_nonzero, lane, 0;
    @lane_nonzero bra JOIN;
    mad.lo.u32 warp_index, cta, 32, warp;
    mul.wide.u32 byte_offset, warp_index, 8;
    add.u64 address, times, byte_offset;
    st.global.u64 [address], ticks;
    bra JOIN;
INVALID:
    trap;
JOIN:
    bar.sync 0;
    ret;
}
"""


def prepare(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    ptx = root / "kernel.ptx"
    ptx.write_text(source(args.body, args.form))
    records = []
    for command in (
        [shutil.which("ptxas"), "-arch=sm_89", "-O3", "-v",
         str(ptx), "-o", str(root / "kernel.cubin")],
        [shutil.which("nvdisasm"), "-c", str(root / "kernel.cubin")],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        records.append(dict(command=command, returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        write(root / "commands.json", records)
        result.check_returncode()
    (root / "kernel.sass").write_text(records[-1]["stdout"])
    shutil.copyfile(__file__, root / "probe_source.py")
    write(root / "preparation.json", dict(
        body=args.body, form=args.form, generator_sha256=sha(__file__),
        ptx_sha256=sha(ptx), cubin_sha256=sha(root / "kernel.cubin"),
        sass_sha256=sha(root / "kernel.sass"),
        native_review_required=True, gpu_execution=False,
        assembler_sha256=sha(records[0]["command"][0]),
        disassembler_sha256=sha(records[1]["command"][0]),
        scope=("Dependent vote motif with retained loop administration; "
               "ballot_compare also includes one dependent integer compare"),
    ))
    print(root)


def run(args):
    # Native mode imports the project's single CUDA/CuPy import hub.
    from cubie.cuda_simsafe import cupy

    root = args.out.resolve()
    prep = json.loads((root / "preparation.json").read_text())
    review = json.loads(args.review.read_text())
    if (review["status"] != "PASS"
            or review["cubin_sha256"] != sha(root / "kernel.cubin")
            or review["body_votes"] != prep["body"]
            or prep["generator_sha256"] != sha(__file__)):
        raise ValueError("Independent native review/source identity differs")
    device = cupy.cuda.Device()
    attrs = device.attributes
    if device.compute_capability != "89":
        raise ValueError("Probe requires SM89")
    module = cupy.RawModule(path=str(root / "kernel.cubin"))
    kernel = module.get_function("collective_probe")
    function = kernel.kernel
    resident = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        function.ptr, 1024, 0)
    if resident != 1 or kernel.attributes["local_size_bytes"] != 0:
        raise ValueError("Expected one resident block and zero local bytes")
    blocks = 2 * attrs["MultiProcessorCount"] * resident
    if not 1 <= args.iterations <= 0xFFFFFFFF // 2 or args.pairs < 1:
        raise ValueError("Positive counts and uint32-safe doubled N required")
    output = root / args.label
    output.mkdir(exist_ok=False)
    telemetry = subprocess.run([
        shutil.which("nvidia-smi"), "--query-gpu=uuid,name,driver_version,"
        "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
        "--format=csv"], capture_output=True, text=True)
    write(output / "launch.json", dict(
        preparation=prep, review_sha256=sha(args.review),
        attributes=attrs, kernel_attributes=kernel.attributes,
        blocks=blocks, block_threads=1024, resident_blocks_per_sm=resident,
        full_occupancy_waves=2, populations=[1, 32],
        iterations=args.iterations, pairs=args.pairs,
        logical_oracle="seed xor ((body*iterations) mod 2)",
        telemetry_before=telemetry.stdout,
    ))
    records = []
    # Alternate N/2N order across pairs; save every clock and endpoint array.
    for population in (1, 32):
        endpoints = cupy.empty((blocks, 1024), dtype=np.uint32)
        ticks = cupy.empty((blocks, 32), dtype=np.uint64)
        for pair in range(args.pairs + 1):
            order = (1, 2) if pair % 2 == 0 else (2, 1)
            for factor in order:
                count = factor * args.iterations
                seed = pair % 2
                endpoints.fill(np.uint32(0xDEADBEEF))
                ticks.fill(np.uint64(0))
                start, end = cupy.cuda.Event(), cupy.cuda.Event()
                start.record()
                kernel((blocks,), (1024,), (
                    endpoints, ticks, np.uint32(count), np.uint32(population),
                    np.uint32(seed), np.uint32(1)))
                end.record()
                end.synchronize()
                actual = endpoints.get()
                clocks = ticks.get()
                expected = seed ^ ((prep["body"] * count) & 1)
                np.testing.assert_array_equal(
                    actual[:, :population * 32],
                    np.full((blocks, population * 32), expected, np.uint32))
                if not np.all(clocks[:, :population] > 0):
                    raise ValueError("Every active warp must report clocks")
                name = f"w{population}_p{pair}_f{factor}"
                np.savez(output / (name + ".npz"), endpoints=actual,
                         clocks=clocks)
                median = exact_median(clocks[:, :population])
                records.append(dict(
                    name=name, population=population, pair=pair,
                    warmup=pair == 0, factor=factor, count=count, seed=seed,
                    event_ms=cupy.cuda.get_elapsed_time(start, end),
                    median_warp_clocks=[median.numerator, median.denominator],
                    arrays_sha256=sha(output / (name + ".npz")),
                    all_active_endpoints_exact=True,
                ))
                write(output / "records.json", records)
                time.sleep(0.1)
    summary = []
    for population in (1, 32):
        for pair in range(1, args.pairs + 1):
            rows = {r["factor"]: r for r in records if
                    r["population"] == population and r["pair"] == pair}
            interval = (
                Fraction(*rows[2]["median_warp_clocks"])
                - Fraction(*rows[1]["median_warp_clocks"])
            ) / (prep["body"] * args.iterations)
            summary.append(dict(
                population=population, pair=pair,
                cycles_per_vote_including_administration=[
                    interval.numerator, interval.denominator],
                motif_form=prep.get("form", "vote_all"),
                paired_compare_included=prep.get("form") == "ballot_compare"))
    write(output / "summary.json", summary)
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--body", type=int, default=257)
    parser.add_argument("--form", choices=("vote_all", "ballot_compare"),
                        default="vote_all")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--iterations", type=int, default=65539)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--label", default="ordinary_e1")
    args = parser.parse_args()
    if args.body < 1:
        parser.error("body must be positive")
    if args.mode == "run" and args.review is None:
        parser.error("run requires independent --review")
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
