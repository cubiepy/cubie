"""Measure reviewed instruction streams at actual occupancy limits."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cubie.cuda_simsafe import cupy


def sha(path):
    """Hash exact retained bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median(values):
    """Calculate the exact median of unsigned clock samples."""
    ordered = sorted(int(value) for value in values.ravel())
    middle = len(ordered) // 2
    return (Fraction(ordered[middle]) if len(ordered) % 2 else
            Fraction(ordered[middle - 1] + ordered[middle], 2))


def encoded(value):
    """Serialize exact fractions without rounding."""
    return [value.numerator, value.denominator]


def main():
    """Run identical reviewed images with multiple resident CTAs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", default="256,512,768,1024")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    root = args.prepared.resolve()
    prep = json.loads((root / "preparation.json").read_text())
    review = json.loads((root / "native_review.json").read_text())
    if prep["kind"] != "stream" or review["status"] != "PASS":
        raise ValueError("Reviewed register-only stream required")
    for kind in ("ptx", "cubin", "sass"):
        actual = sha(root / ("kernel." + kind))
        if (actual != prep[kind + "_sha256"]
                or actual != review[kind + "_sha256"]):
            raise ValueError("Reviewed image identity differs")
    blocksizes = [int(value) for value in args.blocks.split(",")]
    if (any(block not in (128, 256, 512, 768, 1024)
            for block in blocksizes) or args.iterations <= 0
            or args.pairs <= 0):
        raise ValueError("Positive counts and full-warp legal blocks required")
    if 8 * args.pairs + 28 + 2 * args.iterations * prep["body_ffmas"] >= 2**24:
        raise ValueError("Endpoint leaves exact FP32 consecutive integers")
    module = cupy.RawModule(path=str(root / "kernel.cubin"))
    kernel = module.get_function("instruction_probe")
    device = cupy.cuda.Device()
    if device.compute_capability != "89":
        raise ValueError("Reviewed SM89 image required")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    specification = dict(
        prepared=str(root), preparation_sha256=sha(root / "preparation.json"),
        review_sha256=sha(root / "native_review.json"),
        attributes=device.attributes, kernel_attributes=kernel.attributes,
        block_sizes=blocksizes, iterations=args.iterations, pairs=args.pairs,
        native_compilations=0, binary_change=False,
        geometry="Native slot strides stay 1024 threads and32warps per CTA; "
        "only actual launch size changes. All actual warps participate.",
        actual_hot_body=review["actual_hot_body"],
        service_assignment="occupancy-dependent stream composite",
    )
    (args.out / "launch.json").write_text(json.dumps(specification, indent=2))
    records = []
    summaries = []
    for block in blocksizes:
        resident = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
            kernel.kernel.ptr, block, 0)
        if resident <= 0 or kernel.attributes["local_size_bytes"] != 0:
            raise ValueError("Expected legal register-only occupancy")
        grid = 2 * device.attributes["MultiProcessorCount"] * resident
        warps = block // 32
        medians = {}
        for pair in range(args.pairs + 1):
            for factor in ((1, 2) if pair % 2 == 0 else (2, 1)):
                count = args.iterations * factor
                endpoints = cupy.full((grid, 1024), 0xdeadbeef, np.uint32)
                clocks = cupy.zeros((grid, 32), np.uint64)
                controls = cupy.asarray(np.array([
                    endpoints.data.ptr, clocks.data.ptr, count, 0x3f800000,
                    0x3f800000, 2**24 - 1, 0,
                ], np.uint64))
                begin, end = cupy.cuda.Event(), cupy.cuda.Event()
                begin.record()
                kernel((grid,), (block,), (
                    controls, np.uint32(warps), np.uint32(pair),
                ))
                end.record()
                end.synchronize()
                actual, ticks = endpoints.get(), clocks.get()
                expected = 8 * pair + 28 + count * prep["body_ffmas"]
                wanted = np.full((grid, 1024), 0xdeadbeef, np.uint32)
                wanted[:, :block] = expected
                np.testing.assert_array_equal(actual, wanted)
                if not np.all(ticks[:, :warps] > 0):
                    raise ValueError("Every active warp requires elapsed clocks")
                np.testing.assert_array_equal(
                    ticks[:, warps:], np.zeros((grid, 32 - warps), np.uint64))
                name = f"b{block}_p{pair}_f{factor}"
                path = args.out / (name + ".npz")
                np.savez(path, endpoints=actual, clocks=ticks)
                value = median(ticks[:, :warps])
                medians[pair, factor] = value
                records.append(dict(
                    name=name, block_threads=block, grid_blocks=grid,
                    resident_blocks=resident, resident_warps=resident * warps,
                    full_occupancy_waves=2, pair=pair, factor=factor,
                    iterations=count, seed=pair, expected=expected,
                    median_clocks=encoded(value), warmup=pair == 0,
                    event_ms=cupy.cuda.get_elapsed_time(begin, end),
                    array_sha256=sha(path), all_endpoints_exact=True,
                ))
                (args.out / "records.json").write_text(
                    json.dumps(records, indent=2))
                time.sleep(0.1)
            if pair:
                interval = (medians[pair, 2] - medians[pair, 1]) / (
                    args.iterations * prep["body_ffmas"])
                summaries.append(dict(
                    block_threads=block, resident_warps=resident * warps,
                    pair=pair, cycles_per_warp_ffma=encoded(interval),
                    scope="Inclusive stream service; not intrinsic refill "
                    "latency or an independently saturated bandwidth proof",
                ))
        print(block, resident, [float(Fraction(*row["cycles_per_warp_ffma"]))
                               for row in summaries
                               if row["block_threads"] == block], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
