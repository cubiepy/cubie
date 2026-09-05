"""Rebuild placement-specific allocation and physical memory demand on CPU."""

import argparse
from collections import Counter, OrderedDict
from fractions import Fraction
import itertools
import json
from pathlib import Path

from benchmarks.hardware_model.candidate_selection import (
    canonical, file_digest, positive_int, residency, validate_hardware,
)
from benchmarks.hardware_model.implicit_native_lowering import make_plan


def load_bound(record, label):
    path = Path(record["path"]).resolve()
    if file_digest(path) != record["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return json.loads(path.read_text()), path


def named_registry(graph, names):
    records = {}
    for record in graph["registry"]:
        name = record["owner"] + ":" + record["name"]
        if name in names:
            if name in records:
                raise ValueError(f"Ambiguous registry identity {name}")
            positive_int(record["bytes"], "named buffer extent", True)
            if record["dtype"] not in ("float32", "int32", "uint32"):
                raise ValueError("Placement needs typed four-byte buffers")
            records[name] = record
    if set(records) != set(names):
        raise ValueError("Named buffer is absent from actual registry")
    return records


def admit_graph(graph, placement):
    """Bind target settings, actual layouts and full construction identity."""
    if graph.get("kind") != "typed_implicit_execution_region":
        raise ValueError("Placement requires an actual implicit source graph")
    construction = graph["placement_construction"]
    if construction["placement_identity"] != placement:
        raise ValueError("Source graph belongs to another placement")
    common = graph["candidate_construction"]
    if (common["workload_identity"] != construction["workload_identity"] or
            common["precision"] != construction["precision"] or
            common["shared_stride_bytes"] != construction["shared_stride_bytes"]):
        raise ValueError("Candidate construction binding differs")
    actual_placement = {item["owner"] + ":" + item["name"]:
                        item["declared_location"] for item in graph["registry"]}
    if graph["placement_identity"] != actual_placement:
        raise ValueError("Full registry placement binding differs")
    for key in ("layout_source", "constructor"):
        record = construction[key]
        if file_digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Placement {key} source bytes changed")
    if construction["precision"] != "float32":
        raise ValueError("Only actual FP32 shared layouts are admitted")
    elements = positive_int(construction["shared_elements_per_run"],
                            "shared elements", True)
    padding = int(elements > 0 and elements % 2 == 0)
    stride = 4 * (elements + padding)
    if (construction["shared_padding_elements"] != padding or
            construction["shared_stride_bytes"] != stride):
        raise ValueError("Shared stride disagrees with actual FP32 padding")
    records = named_registry(graph, placement)
    for name, space in placement.items():
        record = records[name]
        if record["declared_location"] != space:
            raise ValueError("Requested placement did not reach the owner")
        expected = "persistent_local" if record["persistent"] else "local"
        if record["resolved_location"] != (
                "shared" if space == "shared" else expected):
            raise ValueError("Resolved registry placement disagrees")
    for allocation in graph["allocations"]:
        view = allocation["view"]
        if (view["storage"] == "caller:shared_scratch" and
                view["offset"] + view["bytes"] > stride):
            raise ValueError("Actual shared view exceeds captured run stride")
    return records, stride, construction["workload_identity"]


def memory_events(plan):
    """Retain fresh lowering's named accesses and allocator spill traffic."""
    result = []
    step_mask = plan["dynamic_work"]["iterations"]["step_entry_mask"]
    for event in plan["allocation"]["events"]:
        memory = event.get("memory")
        if event["kind"] in ("spill", "reload"):
            memory = {"space": "local", "kind": "spill",
                      "access": "write" if event["kind"] == "spill"
                      else "read", "offset": event["offset"], "bytes": 4}
        if memory is not None:
            if memory["bytes"] != 4 or memory["offset"] % 4:
                raise ValueError("Memory model needs aligned typed32 accesses")
            position = event.get("node", event["source_position"])
            node = plan["lowering"]["nodes"][position]
            masks = {context.get("runtime_region", {}).get(
                "entry_mask", step_mask) for context in node["source_contexts"]}
            if len(masks) != 1:
                raise ValueError("Memory operation has ambiguous issue mask")
            issue_mask = masks.pop()
            if (type(issue_mask) is not int or not 0 < issue_mask < 2**32 or
                    issue_mask & ~step_mask):
                raise ValueError("Memory issue mask exceeds step entry")
            result.append({"allocation_event": event["id"],
                           "issue_mask": issue_mask, **memory})
    return result


def allocation_regime_support(plan):
    """State the positive mask condition proved by scalar-tag allocation."""
    step_mask = plan["dynamic_work"]["iterations"]["step_entry_mask"]
    masks = {step_mask}
    for node in plan["lowering"]["nodes"]:
        masks.update(context.get("runtime_region", {}).get(
            "entry_mask", step_mask) for context in node["source_contexts"])
    coherent = masks == {2**32 - 1}
    return {
        "status": "supported_coherent_full_warp" if coherent else
        "unsupported_subgroup_allocation",
        "proved_source_issue_masks": sorted(masks),
        "finite_ranking_supported": coherent,
        "proof_condition": "every source operation has full-warp issue mask",
        "limitation": (
            "Scalar tag replay does not prove per-lane register/spill "
            "conservation across subgroup entry/exit. Subgroup memory counts "
            "describe conditional issued accesses only."),
    }


def memory_demand(plan, geometry, stride, hardware, scenario):
    """Execute an explicit sector-LRU and shared-bank demand hypothesis."""
    events = memory_events(plan)
    warps = geometry["resident_warps_per_sm"]
    waves = positive_int(scenario["waves"], "modeled waves")
    if waves < 2:
        raise ValueError("Physical stream requires at least two full waves")
    backing = scenario["local_backing"]
    if backing not in ("resident_slots", "trajectory_unique"):
        raise ValueError("Unknown local backing hypothesis")
    if scenario["cache_policy"] != "cold_fully_associative_sector_lru":
        raise ValueError("Unknown cache policy hypothesis")
    store_policy = scenario["local_store_policy"]
    if store_policy not in ("l1_write_through_l2_write_back",
                            "l1_write_back_l2_write_back"):
        raise ValueError("Local stores require an explicit downstream policy")
    through = store_policy == "l1_write_through_l2_write_back"
    if (hardware["data_sector_bytes"] != 32 or
            hardware["shared_banks"] != 32 or
            hardware["shared_bank_bytes"] != 4):
        raise ValueError("Admitted local/shared layout is 32B/32x4B")
    l1_capacity = ((hardware["unified_data_bytes"] -
                    scenario["shared_carveout_bytes"]) // 32)
    # Other SMs can compete for unified L2. The request supplies a partition
    # scenario, bounded by published L2; it is not fitted to observed hits.
    l2_bytes = positive_int(scenario["l2_bytes_available_per_sm"],
                            "available L2 bytes", True)
    if l2_bytes > hardware["l2_bytes"]:
        raise ValueError("L2 partition exceeds physical capacity")
    caches = [OrderedDict(), OrderedDict()]
    capacities = [l1_capacity, l2_bytes // 32]
    counts = Counter()

    def access(level, sector, write):
        cache, capacity = caches[level], capacities[level]
        label = "l1" if level == 0 else "l2"
        counts[label + ("_write_requests" if write else "_read_requests")] += 1
        if sector in cache:
            dirty = cache.pop(sector)
            if level == 0 and through and write:
                access(1, sector, True)
            cache[sector] = dirty or (write and not (level == 0 and through))
            counts[label + "_hits"] += 1
            return
        counts[label + "_misses"] += 1
        if capacity == 0:
            if level == 0:
                access(1, sector, write)
            else:
                counts["dram_read_sectors"] += 1
                counts["dram_write_sectors"] += int(write)
            return
        if level == 0:
            access(1, sector, write and through)
        else:
            counts["dram_read_sectors"] += 1
        if len(cache) == capacity:
            evicted, dirty = cache.popitem(last=False)
            if dirty:
                counts[label + "_dirty_evictions"] += 1
                if level == 0:
                    access(1, evicted, True)
                else:
                    counts["dram_write_sectors"] += 1
        cache[sector] = write and not (level == 0 and through)

    frame = plan["allocation"]["local_frame_bytes"]
    if frame % 4:
        raise ValueError("Local frame is not word aligned")
    for wave in range(waves):
        for event in events:
            for warp in range(warps):
                space, operation = event["space"], event["access"]
                counts[space + "_" + operation + "_warp_instructions"] += 1
                if space == "shared":
                    addresses = [lane * stride + event["offset"]
                                 for lane in range(32)
                                 if event["issue_mask"] & (1 << lane)]
                    # Equal-address reads broadcast; distinct words serialize.
                    banks = [set() for _ in range(32)]
                    for address in addresses:
                        banks[(address // 4) % 32].add(address)
                    counts["shared_bank_wavefronts"] += max(map(len, banks))
                elif space == "local":
                    if event["offset"] + 4 > frame:
                        raise ValueError("Local event exceeds allocated frame")
                    slot = warp + (wave * warps
                                   if backing == "trajectory_unique" else 0)
                    start = slot * frame + event["offset"]
                    sectors = {start + lane // 8 for lane in range(32)
                               if event["issue_mask"] & (1 << lane)}
                    counts["local_" + operation + "_sectors"] += len(sectors)
                    for sector in sorted(sectors):
                        access(0, sector, operation == "write")
                else:
                    raise ValueError("Unknown lowered memory space")
    return {
        "events": events, "counts": dict(sorted(counts.items())),
        "waves": waves, "resident_warps_per_sm": warps,
        "local_frame_bytes_per_thread": frame,
        "resident_local_backing_bytes_per_sm": frame * warps * 32,
        "l1_capacity_bytes": l1_capacity * 32,
        "l2_available_bytes_per_sm": l2_bytes,
        "retained_dirty_sectors": [sum(cache.values()) for cache in caches],
        "allocation_regime_support": allocation_regime_support(plan),
        "assumptions": {
            "interleaving": "cyclic warps at each memory-event ordinal",
            "local_backing": backing, "cache_policy": scenario["cache_policy"],
            "writes": "write-allocate; " + store_policy,
            "l2_sharing": "explicit available-capacity partition scenario",
            "scope": "one step per warp per wave; no native hit-class claim",
            "issue_masks": "source runtime entry or actual step-entry mask",
        },
    }


def physical_coefficients(plan, demand):
    """Return work per completed warp for a symbolic service comparison."""
    denominator = demand["waves"] * demand["resident_warps_per_sm"]
    result = {"memory:" + name: Fraction(value, denominator)
              for name, value in demand["counts"].items()}
    counts = Counter(event["opcode"] for event in plan["allocation"]["events"]
                     if event.get("opcode"))
    result.update({"instruction:" + key: Fraction(value)
                   for key, value in counts.items()})
    return {key: [value.numerator, value.denominator]
            for key, value in sorted(result.items())}


def enumerate_placements(request):
    """Build both compiler materialization hypotheses for every placement."""
    if (request.get("schema_version") != 2 or
            request.get("kind") != "named_buffer_placement_request"):
        raise ValueError("Placement requires v2 actual source graphs")
    hardware = request["hardware"]
    validate_hardware(hardware)
    positive_int(hardware["unified_data_bytes"], "unified capacity")
    positive_int(hardware["l2_bytes"], "L2 capacity")
    names = request["named_buffers"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Named buffers must be unique and nonempty")
    block = positive_int(request["block_threads"], "block threads")
    static = positive_int(request["static_shared_bytes"], "static shared", True)
    scenario = request["cache_scenario"]
    carveout = positive_int(scenario["shared_carveout_bytes"], "carveout", True)
    if carveout > hardware["unified_data_bytes"]:
        raise ValueError("Shared carveout exceeds unified capacity")
    if scenario["instruction_delivery"] != "symbolic":
        raise ValueError("Instruction delivery needs separately qualified data")
    modes = request["materialization_scenarios"]
    if set(modes) != {"promote", "addressable"} or len(modes) != 2:
        raise ValueError("Preserve promoted and addressable compiler scenarios")
    placements = [dict(zip(names, spaces)) for spaces in itertools.product(
        ("local", "shared"), repeat=len(names))]
    if set(request["source_graphs"]) != {canonical(item) for item in placements}:
        raise ValueError("Every actual placement requires its own source graph")
    rows = []
    construction_identity = None
    execution_identity = None
    for placement in placements:
        graph_record = request["source_graphs"][canonical(placement)]
        graph, graph_path = load_bound(graph_record, "placement source graph")
        records, stride, construction = admit_graph(graph, placement)
        if construction_identity is None:
            construction_identity = construction
        if construction != construction_identity:
            raise ValueError("Placement graph construction workloads differ")
        execution = {"regime": graph["regime"],
                     "branch_choices": graph["branch_choices"],
                     "semantic_contract": graph["semantic_contract"]}
        if execution_identity is None:
            execution_identity = execution
        if execution != execution_identity:
            raise ValueError("Placement execution regimes or branches differ")
        for mode in modes:
            # No externally supplied register count or allocation is accepted.
            plan = make_plan(graph, request["architecture"],
                             request["compiler"], mode)
            plan["provenance"]["graph"] = {
                "path": str(graph_path), "sha256": graph_record["sha256"]}
            plan["placement_identity"] = graph["placement_identity"]
            plan["candidate_construction"] = graph["candidate_construction"]
            plan["materialization_scenario"] = mode
            support = allocation_regime_support(plan)
            plan["allocation_regime_support"] = support
            registers = plan["allocation"]["peak_resident"]["R"]
            dynamic = max(4, stride * block)
            geometry = residency(hardware, registers, block, static,
                                 dynamic, carveout)
            demand = None
            if geometry["legal"]:
                demand = memory_demand(plan, geometry, stride,
                                       hardware, scenario)
            rows.append({
                "placement_identity": placement,
                "materialization_scenario": mode,
                "source_graph": graph_record, "registry": records,
                "native_plan": plan,
                "allocation_regime_support": support,
                "full_placement_identity": graph["placement_identity"],
                "launch_geometry": {
                    "block_threads": block, "static_shared_bytes": static,
                    "dynamic_shared_bytes": dynamic,
                    "shared_carveout_bytes": carveout,
                    "hardware_identity": hardware["identity"],
                },
                "shared_stride_bytes": stride,
                "dynamic_shared_bytes_per_block": dynamic,
                "geometry": geometry, "memory_demand": demand,
                "physical_coefficients": None if demand is None else
                physical_coefficients(plan, demand),
            })
    return {
        "schema_version": 2,
        "kind": "conditional_named_buffer_placement_decision",
        "construction_identity": construction_identity,
        "execution_identity": execution_identity,
        "hardware": hardware, "cache_scenario": scenario,
        "placements": rows,
        "selection_status": "qualified_memory_and_instruction_services_needed",
        "comparison_contract": (
            "Physical coefficients are work demands, not additive runtime. "
            "Compare each materialization scenario using a service scheduler "
            "with its own residency and dependency trace; promote/addressable "
            "are compiler uncertainty, not selectable user settings."),
        "timing_labels_consumed": False,
        "native_compilation_performed": False,
        "source_liveness_used_as_native_registers": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = enumerate_placements(json.loads(args.request.read_text()))
    result["provenance"] = {
        "request_path": str(args.request.resolve()),
        "request_sha256": file_digest(args.request),
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": file_digest(__file__),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
