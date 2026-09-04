"""Launch one reviewed collective-vote binary for native counter capture."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cubie.cuda_simsafe import cupy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warps", type=int, choices=(1, 32), required=True)
    parser.add_argument("--iterations", type=int, default=4099)
    args = parser.parse_args()
    root = args.prepared.resolve()
    review = json.loads((root / "independent_native_review.json").read_text())
    prep = json.loads((root / "preparation.json").read_text())
    binary = root / "kernel.cubin"
    if (review["status"] != "PASS"
            or review["cubin_sha256"] != digest(binary)
            or prep["cubin_sha256"] != digest(binary)
            or review["body_votes"] != prep["body"]):
        raise ValueError("Reviewed binary identity differs")
    if not 1 <= args.iterations <= 0xFFFFFFFF:
        raise ValueError("Positive uint32 iteration count required")
    args.out.mkdir(parents=True, exist_ok=False)
    module = cupy.RawModule(path=str(binary))
    kernel = module.get_function("collective_probe")
    device = cupy.cuda.Device()
    resident = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        kernel.kernel.ptr, 1024, 0)
    if (device.compute_capability != "89" or resident != 1
            or kernel.attributes["local_size_bytes"] != 0):
        raise ValueError("Expected reviewed SM89 geometry")
    blocks = 2 * device.attributes["MultiProcessorCount"] * resident
    endpoints = cupy.empty((blocks, 1024), np.uint32)
    clocks = cupy.empty((blocks, 32), np.uint64)
    kernel((blocks,), (1024,), (
        endpoints, clocks, np.uint32(args.iterations), np.uint32(args.warps),
        np.uint32(0), np.uint32(1)))
    cupy.cuda.get_current_stream().synchronize()
    observed, ticks = endpoints.get(), clocks.get()
    expected = (prep["body"] * args.iterations) & 1
    np.testing.assert_array_equal(
        observed[:, :args.warps * 32],
        np.full((blocks, args.warps * 32), expected, np.uint32))
    if not np.all(ticks[:, :args.warps] > 0):
        raise ValueError("Every participating warp must report clocks")
    np.savez(args.out / "outputs.npz", endpoints=observed, clocks=ticks)
    receipt = dict(
        status="PASS", cubin_sha256=digest(binary),
        profile_source_sha256=digest(__file__),
        review_sha256=digest(root / "independent_native_review.json"),
        preparation_sha256=digest(root / "preparation.json"),
        attributes=device.attributes, kernel_attributes=kernel.attributes,
        resident_blocks_per_sm=resident, blocks=blocks, block_threads=1024,
        full_occupancy_waves=2, active_warps=args.warps,
        iterations=args.iterations, body_votes=prep["body"],
        expected_target_warp_votes=blocks * args.warps * args.iterations
        * prep["body"], exact_active_endpoints=blocks * args.warps * 32,
        outputs_sha256=digest(args.out / "outputs.npz"),
        profile_clocks_used_as_ordinary_measurement=False,
    )
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
