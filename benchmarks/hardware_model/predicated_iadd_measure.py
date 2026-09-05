"""Measure the inspected predicated-IADD chain, retaining complete arrays."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np


def asset(path):
    path = Path(path)
    return dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def median(values):
    values = sorted(int(value) for value in values.flat)
    middle = len(values) // 2
    return (
        Fraction(values[middle])
        if len(values) % 2
        else Fraction(values[middle - 1] + values[middle], 2)
    )


def checked(record):
    path = Path(record["path"])
    if asset(path)["sha256"] != record["sha256"]:
        raise ValueError("Changed bound artifact: " + str(path))
    return path


def run(prepared_path, review_path, output, iterations, pairs):
    from cubie.cuda_simsafe import cupy

    prepared = json.loads(prepared_path.read_text())
    review = json.loads(review_path.read_text())
    if review["status"] != "PASS":
        raise ValueError("Independent native and runner admission required")
    if checked(review["measurement_source"]) != Path(__file__).resolve():
        raise ValueError("Wrong reviewed measurement source")
    if checked(review["preparation"]) != prepared_path.resolve():
        raise ValueError("Wrong reviewed preparation")
    if iterations < 1 or pairs < 1:
        raise ValueError("Positive iteration and pair counts required")
    output.mkdir(exist_ok=False)
    device = cupy.cuda.Device()
    if device.compute_capability != "89":
        raise ValueError("SM89 required")
    properties = device.attributes
    stream = cupy.cuda.Stream(non_blocking=True)
    receipt = dict(
        status="STARTED",
        source=asset(__file__),
        preparation=asset(prepared_path),
        review=asset(review_path),
        device=properties,
        records=[],
        summaries=[],
        scope="Dependent predicated-IADD stream and exact loop administration; "
        "not a MOV measurement or a standalone instruction latency. "
        "Allocated32warps/CTA, active populations1/8/32; issue/control mask "
        "differs from data predicate. Event times retained as diagnostics only.",
    )
    try:
        for case in prepared["cases"]:
            if case["form"] != "predicated_move" or case["optimization"] != 3:
                continue
            admitted = next(
                item
                for item in review["cases"]
                if item["cubin"]["sha256"] == case["cubin"]["sha256"]
            )
            if admitted["native_form"] != "predicated_iadd_chain":
                raise ValueError("Wrong native motif admission")
            cubin = checked(case["cubin"])
            body = case["body"]
            initial = 7
            if initial + 2 * iterations * body > 0xFFFFFFFF:
                raise ValueError(
                    "Arithmetic oracle exceeds exact uint32 range"
                )
            module = cupy.RawModule(path=str(cubin))
            kernel = module.get_function("predicated_update_probe")
            function = kernel.kernel
            attrs = kernel.attributes
            resident = (
                cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                    function.ptr, 1024, 0
                )
            )
            if resident != 1 or attrs["local_size_bytes"] != 0:
                raise ValueError("Expected one CTA/SM without local memory")
            if attrs["num_regs"] != admitted["registers"]:
                raise ValueError("Native register admission differs")
            blocks = 2 * properties["MultiProcessorCount"] * resident
            for population in (1, 8, 32):
                for predicate_lanes in (0, 16, 32):
                    endpoints = cupy.empty((blocks, 1024), dtype=np.uint32)
                    ticks = cupy.empty((blocks, 32), dtype=np.uint64)
                    for pair in range(pairs + 1):
                        for factor in (1, 2) if pair % 2 == 0 else (2, 1):
                            count = iterations * factor
                            bound = initial + body * count
                            with stream:
                                endpoints.fill(np.uint32(0xDEADBEEF))
                                ticks.fill(np.uint64(0))
                                start, end = (
                                    cupy.cuda.Event(),
                                    cupy.cuda.Event(),
                                )
                                start.record(stream)
                                kernel(
                                    (blocks,),
                                    (1024,),
                                    (
                                        endpoints,
                                        ticks,
                                        np.uint32(count),
                                        np.uint32(population),
                                        np.uint32(initial),
                                        np.uint32(predicate_lanes),
                                        np.uint32(bound),
                                    ),
                                    stream=stream,
                                )
                                end.record(stream)
                            end.synchronize()
                            actual = endpoints.get(stream=stream)
                            clocks = ticks.get(stream=stream)
                            stream.synchronize()
                            expected = np.full(
                                (blocks, 1024), 0xDEADBEEF, np.uint32
                            )
                            lane = np.arange(population * 32) % 32
                            expected[:, : population * 32] = np.where(
                                lane < predicate_lanes, bound, initial
                            ).astype(np.uint32)
                            name = f"b{body}_w{population}_m{predicate_lanes}_p{pair}_f{factor}"
                            path = output / (name + ".npz")
                            np.savez_compressed(
                                path, endpoints=actual, clocks=clocks
                            )
                            good = bool(np.array_equal(actual, expected))
                            clocks_good = bool(
                                np.all(clocks[:, :population] > 0)
                                and np.all(clocks[:, population:] == 0)
                            )
                            value = median(clocks[:, :population])
                            receipt["records"].append(
                                dict(
                                    name=name,
                                    arrays=asset(path),
                                    cubin=asset(cubin),
                                    body=body,
                                    population=population,
                                    predicate_lanes=predicate_lanes,
                                    pair=pair,
                                    warmup=pair == 0,
                                    factor=factor,
                                    iterations=count,
                                    initial=initial,
                                    expected_bound=bound,
                                    blocks=blocks,
                                    threads=1024,
                                    resident_blocks_per_sm=resident,
                                    waves=2,
                                    native_attributes=attrs,
                                    all_endpoints_exact=good,
                                    all_clock_slots_valid=clocks_good,
                                    median_warp_cycles=[
                                        value.numerator,
                                        value.denominator,
                                    ],
                                    event_ms=cupy.cuda.get_elapsed_time(
                                        start, end
                                    ),
                                )
                            )
                            (output / "receipt.json").write_text(
                                json.dumps(receipt, indent=2)
                            )
                            if not good or not clocks_good:
                                raise ValueError(
                                    "Whole-array or clock-slot gate failed"
                                )
                            time.sleep(0.1)
        for body in (33, 257):
            for population in (1, 8, 32):
                for predicate_lanes in (0, 16, 32):
                    for pair in range(1, pairs + 1):
                        rows = {
                            row["factor"]: row
                            for row in receipt["records"]
                            if (
                                row["body"],
                                row["population"],
                                row["predicate_lanes"],
                                row["pair"],
                            )
                            == (body, population, predicate_lanes, pair)
                        }
                        interval = (
                            Fraction(*rows[2]["median_warp_cycles"])
                            - Fraction(*rows[1]["median_warp_cycles"])
                        ) / (body * iterations)
                        receipt["summaries"].append(
                            dict(
                                body=body,
                                population=population,
                                predicate_lanes=predicate_lanes,
                                pair=pair,
                                cycles_per_target_including_administration=[
                                    interval.numerator,
                                    interval.denominator,
                                ],
                            )
                        )
        receipt["status"] = (
            "AUTHOR_ARRAY_AND_INTERVAL_COMPLETE_REQUIRES_REVIEW"
        )
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        stream.synchronize()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt["summaries"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=65539)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    run(args.prepared, args.review, args.output, args.iterations, args.pairs)
