"""Launch one reviewed instruction-fill image for counter attribution."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cubie.cuda_simsafe import cupy


def sha(path):
    """Return the exact artifact digest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Retain exact endpoints and profile identity for one native launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warps", type=int, choices=(1, 4, 8, 16, 32),
                        required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warm", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup-launch", action="store_true")
    parser.add_argument("--application-replay-output", action="store_true")
    args = parser.parse_args()
    root = args.prepared.resolve()
    prep = json.loads((root / "preparation.json").read_text())
    review = json.loads((root / "native_review.json").read_text())
    if review["status"] != "PASS":
        raise ValueError("Native review has not passed")
    for kind in ("cubin", "sass", "ptx"):
        actual = sha(root / ("kernel." + kind))
        if (actual != prep[kind + "_sha256"]
                or actual != review[kind + "_sha256"]):
            raise ValueError("Native image differs from reviewed bytes")
    if not 1 <= args.iterations < 2**31:
        raise ValueError("Positive bounded iteration count required")
    if prep["kind"] == "stream" and args.warm:
        raise ValueError("Stream has no separate warm mode")
    count = args.iterations
    expected = 36 + count * (prep["body_ffmas"] + (
        prep["victim_ffmas"] * (1 + args.warm)
        if prep["kind"] == "victim" else 0))
    if expected >= 2**24:
        raise ValueError("Endpoint leaves exact FP32 integer range")
    module = cupy.RawModule(path=str(root / "kernel.cubin"))
    kernel = module.get_function("instruction_probe")
    device = cupy.cuda.Device()
    resident = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        kernel.kernel.ptr, 1024, 0)
    if (device.compute_capability != "89" or resident != 1
            or kernel.attributes["local_size_bytes"] != 0):
        raise ValueError("Reviewed launch geometry differs")
    blocks = 2 * device.attributes["MultiProcessorCount"]
    samples = count if prep["kind"] == "victim" else 1
    endpoints = cupy.full((blocks, 1024), 0xdeadbeef, dtype=np.uint32)
    ticks = cupy.zeros((blocks, 32, samples), dtype=np.uint64)
    controls = cupy.asarray(np.array([
        endpoints.data.ptr, ticks.data.ptr, count, 0x3f800000,
        0x3f800000, 2**24 - 1, args.warm,
    ], dtype=np.uint64))
    if args.application_replay_output:
        args.out.mkdir(parents=True, exist_ok=True)
        args.out = args.out / f"process_{os.getpid()}"
    args.out.mkdir(parents=True, exist_ok=False)
    if args.warmup_launch:
        kernel((blocks,), (1024,), (
            controls, np.uint32(args.warps), np.uint32(1),
        ))
        cupy.cuda.get_current_stream().synchronize()
        endpoints.fill(np.uint32(0xdeadbeef))
        ticks.fill(np.uint64(0))
    kernel((blocks,), (1024,), (
        controls, np.uint32(args.warps), np.uint32(1),
    ))
    cupy.cuda.get_current_stream().synchronize()
    actual, clocks = endpoints.get(), ticks.get()
    wanted = np.full((blocks, 1024), 0xdeadbeef, np.uint32)
    wanted[:, :args.warps * 32] = expected
    np.testing.assert_array_equal(actual, wanted)
    if not np.all(clocks[:, :args.warps, :] > 0):
        raise ValueError("Every active warp/trial requires a clock interval")
    np.testing.assert_array_equal(
        clocks[:, args.warps:, :],
        np.zeros((blocks, 32 - args.warps, samples), np.uint64),
    )
    arrays = args.out / "arrays.npz"
    np.savez(arrays, endpoints=actual, clocks=clocks)
    receipt = dict(
        status="FUNCTIONAL_PASS_COUNTER_AUDIT_PENDING", prepared=str(root),
        preparation_sha256=sha(root / "preparation.json"),
        review_sha256=sha(root / "native_review.json"),
        profile_source_sha256=sha(__file__), arrays_sha256=sha(arrays),
        iterations=count, population=args.warps, warm=args.warm, seed=1,
        expected=expected, block_threads=1024, grid_blocks=blocks,
        resident_blocks_per_sm=resident, full_occupancy_waves=2,
        warmup_launches=int(args.warmup_launch),
        native_launches=1 + int(args.warmup_launch),
        capture_requires_filtered_launch_skip=int(args.warmup_launch),
        application_replay_output=args.application_replay_output,
        kernel_attributes=kernel.attributes, attributes=device.attributes,
        service_assignment="unassigned_pending_counter_conservation",
    )
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
