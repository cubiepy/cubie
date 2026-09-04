"""Schedule conditional typed events using nominal physical services.

This CPU-only model schedules one coherent resident wave. It keeps
register readiness, store input consumption, memory visibility and issue
capacity separate. The supplied catalog and scenario remain hypotheses.
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from benchmarks.hardware_model.nominal_data_cache import NominalDataCache


FULL_MASK = 0xFFFFFFFF
PARTITIONS = 4
WARP_SIZE = 32


def rational(value, name, positive=True):
    """Read an exact rational, rejecting implicit float approximations."""
    if isinstance(value, dict):
        if set(value) != {"numerator", "denominator"}:
            raise ValueError(f"{name}: expected a rational pair")
        numerator, denominator = value["numerator"], value["denominator"]
        if type(numerator) is not int or type(denominator) is not int:
            raise ValueError(f"{name}: rational components must be integers")
        if denominator <= 0:
            raise ValueError(f"{name}: denominator must be positive")
        result = Fraction(numerator, denominator)
    elif type(value) in (int, str, Fraction):
        result = Fraction(value)
    else:
        raise ValueError(f"{name}: use an integer or exact rational")
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{name}: expected positive cycles")
    return result


def integer(value, name, minimum=1):
    """Read an integer geometry or instruction-count field."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return value


def encoded(value):
    """Encode exact rational values recursively for a JSON result."""
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator)
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded(item) for item in value]
    return value


def identity(value):
    """Hash the exact normalized JSON input, without consuming labels."""
    payload = json.dumps(
        encoded(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def qualified(record, name):
    """Require an explicit physical source and transfer assumption."""
    if not isinstance(record, dict):
        raise ValueError(f"{name}: expected a qualified scenario record")
    if not record.get("provenance") or not record.get("assumption"):
        raise ValueError(f"{name}: provenance and assumption are required")
    return record


def register_keys(refs):
    """Return allocated register identities, including predicate banks."""
    result = set()
    for ref in refs:
        if ref.get("register") is None:
            continue
        bank = ref.get("bank")
        if bank not in ("R", "P"):
            raise ValueError("Only allocated R/P register banks are supported")
        result.add((bank, integer(ref["register"], "register", 0)))
    return result


def executable_events(events):
    """Retain actual instructions and check their allocation interfaces."""
    result = []
    seen = set()
    for event in events:
        if event.get("opcode") is None:
            if event.get("kind") not in ("release", "free_home"):
                raise ValueError("Unknown non-executable event")
            continue
        key = integer(event.get("id"), "event id", 0)
        if key in seen:
            raise ValueError("Executable event IDs must be unique")
        seen.add(key)
        if event.get("kind") in ("spill", "reload"):
            event = dict(event, memory=dict(
                kind="spill_frame", space="local",
                access="write" if event["kind"] == "spill" else "read",
                offset=event["offset"], bytes=event["bytes"],
            ))
        register_keys(event.get("reads", []))
        register_keys(event.get("writes", []))
        semantic = event.get("semantics", {})
        for field in ("declared_active_entry_mask", "participating_mask"):
            if field in semantic and semantic[field] != FULL_MASK:
                raise ValueError("This scheduler requires coherent full warps")
        detail = event.get("memory")
        if detail is not None:
            if detail.get("space") not in ("local", "shared", "constant"):
                raise ValueError("Unsupported memory address space")
            if detail.get("access") not in ("read", "write"):
                raise ValueError("Memory access must be read or write")
            integer(detail.get("offset"), "memory byte offset", 0)
            size = integer(detail.get("bytes"), "memory bytes")
            if size % 4:
                raise ValueError("Memory services require scalar word units")
            if detail.get("cross_warp_alias") not in (None, False, True):
                raise ValueError("cross_warp_alias must be Boolean")
            if detail["space"] == "local" and detail.get("cross_warp_alias"):
                raise ValueError("Thread-local frames cannot alias warps")
            if detail["space"] == "constant":
                if (
                    event["opcode"] != "LDC"
                    or detail.get("kind") != "immutable_constant"
                    or detail["access"] != "read"
                    or size != 4
                    or not detail.get("table_id")
                    or detail.get("offset_is_execution_witness") is not True
                    or detail.get("broadcast_regime")
                    != "uniform_indices_over_declared_active_warp"
                    or not detail.get("address_affine")
                ):
                    raise ValueError("LDC needs an immutable broadcast witness")
        if event["opcode"] == "LDC" and (
            detail is None or detail["space"] != "constant"
        ):
            raise ValueError("LDC requires separate immutable constant memory")
        if event["opcode"] in ("STS", "STL"):
            if detail is None or detail["access"] != "write":
                raise ValueError("Store requires an explicit memory write")
            if event.get("writes"):
                raise ValueError("STS/STL must be resultless")
        if event["opcode"] in ("LDS", "LDL"):
            if detail is None or detail["access"] != "read":
                raise ValueError("Load requires an explicit memory read")
        result.append(event)
    if not result:
        raise ValueError("At least one executable event is required")
    return result


def common_work(wave_cycles, resident_warps, work):
    """Normalize complete synchronized waves to common useful work."""
    if work.get("kind") != "synchronized_full_waves":
        raise ValueError("Common work must use synchronized_full_waves")
    attempts = integer(work.get("warp_attempts_per_sm"), "warp attempts")
    if attempts % resident_warps:
        raise ValueError("Common work must contain complete resident waves")
    waves = attempts // resident_warps
    if waves < 2:
        raise ValueError("Common work must contain at least two full waves")
    return dict(
        warp_attempts_per_sm=attempts,
        resident_waves=waves,
        cycles=wave_cycles * waves,
        cycles_per_warp_attempt=wave_cycles / resident_warps,
    )


def service_for(event, catalog, scenario, missing):
    """Resolve one physical service without filling unknown delays."""
    opcode = event["opcode"]
    base = catalog["instructions"].get(opcode)
    if base is None or opcode == "CAPTURED_LOOKUP":
        missing.append(dict(
            event_id=event["id"], opcode=opcode,
            fields=["concrete_allocated_native_expansion"],
            reason="Expand the source operation before finite scheduling",
        ))
        return None
    form = event.get("semantics", {}).get("vote_operation")
    if form in base.get("forms", {}):
        base = dict(base, **base["forms"][form])
    override = scenario.get("service_overrides", {}).get(opcode, {})
    local = scenario.get("event_overrides", {}).get(str(event["id"]), {})
    for name, item in ((opcode, override), (str(event["id"]), local)):
        if item:
            qualified(item, name)
    values = dict(override)
    values.update(local)
    output = dict(opcode=opcode, event_id=event["id"])
    absent = []

    def cycles(field, fallback=None):
        value = values.get(field, fallback)
        if value is None:
            absent.append(field)
            return None
        return rational(value, f"{opcode}.{field}")

    dependence = base.get("dependent_result")
    fallback = dependence["cycles"] if dependence else None
    consumption = base.get("source_consumed")
    consume_fallback = consumption["cycles"] if consumption else None
    if opcode in ("STS", "STL"):
        output.update(
            result=None,
            source_consumed=cycles(
                "store_source_consumed_cycles", consume_fallback
            ),
            visible=cycles("store_visible_cycles"),
            complete=cycles("store_complete_cycles"),
        )
        if not absent and not (
            output["source_consumed"] <= output["visible"]
            <= output["complete"]
        ):
            raise ValueError("Store consumption <= visibility <= completion")
    elif opcode == "BRA":
        output.update(result=None, warp_block=cycles("warp_block_cycles"))
    else:
        output["result"] = cycles("result_latency_cycles", fallback)
        if consume_fallback is not None or "source_consumed_cycles" in values:
            output["source_consumed"] = cycles(
                "source_consumed_cycles", consume_fallback
            )
    initiation = base["initiation"]
    output["resource"] = values.get("resource", initiation["resource"])
    interval = values.get(
        "initiation_cycles", initiation.get(
            "sm_cycles_per_full_warp",
            initiation.get("subpartition_cycles_per_full_warp"),
        )
    )
    output["initiation"] = (
        rational(interval, "initiation") if interval is not None else None
    )
    if output["resource"] not in {
        "fp32", "integer_like", "sfu", "vote", "lsu_scalar", "dispatch",
        "constant_broadcast",
    }:
        raise ValueError("Unsupported execution resource")
    if output["resource"] != "dispatch" and output["initiation"] is None:
        absent.append("initiation_cycles")
    if output["resource"] == "fp32" and output["initiation"] != Fraction(1, 4):
        raise ValueError("FP32 topology uses the published 128-result rate")
    if (
        output["resource"] == "integer_like"
        and output["initiation"] is not None
        and output["initiation"] < Fraction(1, 2)
    ):
        raise ValueError("Integer-only shared route cannot exceed 64 results")
    detail = event.get("memory")
    if detail and detail["space"] == "constant":
        hypothesis = scenario.get("constant_cache")
        if not hypothesis:
            absent.append("scenario.constant_cache")
        else:
            qualified(hypothesis, "constant_cache")
            if hypothesis.get("kind") != "immutable_broadcast_hit":
                raise ValueError("Explicit constant broadcast hit scenario required")
    elif detail:
        hypothesis = scenario.get("memory_issue")
        if not hypothesis:
            absent.append("scenario.memory_issue")
        else:
            qualified(hypothesis, "memory_issue")
            if hypothesis.get("kind") != "scalar_full_warp_wavefronts":
                raise ValueError("Explicit scalar memory issue model required")
            wavefronts = hypothesis.get("event_wavefronts", {}).get(
                str(event["id"]), hypothesis.get("default_wavefronts")
            )
            output["wavefronts"] = integer(wavefronts, "memory wavefronts")
            output["initiation"] *= detail["bytes"] // 4
            output["initiation"] *= output["wavefronts"]
    if absent:
        missing.append(dict(
            event_id=event["id"], opcode=opcode, fields=absent,
            reason=base.get("missing_reason", "Explicit scenario required"),
        ))
    return output


def memory_identity(detail, warp, warps_per_block):
    """Identify overlapping bytes in a frame, independent of cell labels."""
    if detail.get("cross_warp_alias"):
        owner = ("block", warp // warps_per_block)
    else:
        owner = ("warp", warp)
    return (detail["space"], owner), detail["offset"], detail["bytes"]


def memory_hazard(history, detail, warp, warps_per_block):
    """Enforce byte-overlap RAW/WAR/WAW; independent reads may overlap."""
    key, offset, size = memory_identity(detail, warp, warps_per_block)
    ready = Fraction(0)
    for address in range(offset, offset + size):
        previous = history.get(key, {}).get(address, {})
        ready = max(ready, previous.get("write", 0))
        if detail["access"] == "write":
            ready = max(ready, previous.get("read", 0))
    return ready


def operations(events, services, scenario, missing):
    """Build single instructions and explicitly contracted ST/LD motifs."""
    motifs = {}
    occupied = set()
    indexes = {event["id"]: index for index, event in enumerate(events)}
    for record in scenario.get("store_load_motifs", []):
        qualified(record, "store_load_motif")
        pair = record.get("event_ids")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("A motif must name exactly two event IDs")
        if any(key not in indexes or key in occupied for key in pair):
            raise ValueError("Motif IDs are unknown or overlap another motif")
        first, second = [indexes[key] for key in pair]
        if second != first + 1:
            raise ValueError("A contracted motif must be adjacent instructions")
        store, load = events[first], events[second]
        if (store["opcode"], load["opcode"]) not in {
            ("STS", "LDS"), ("STL", "LDL")
        }:
            raise ValueError("Motif must be one same-space store/load pair")
        for field in ("space", "offset", "bytes", "cross_warp_alias"):
            if store["memory"].get(field) != load["memory"].get(field):
                raise ValueError("Motif must access exactly the same bytes")
        if record.get("issue_model") != "noninterleaved_reservation":
            raise ValueError("Motif must state its issue reservation model")
        interval = rational(record["pair_cycles"], "pair interval")
        completion = record.get("store_complete_cycles")
        if completion is None:
            missing.append(dict(
                event_id=pair[0], opcode=store["opcode"],
                fields=["motif.store_complete_cycles"],
                reason="Pair readback does not identify final store drain",
            ))
        motifs[first] = dict(
            events=[store, load], services=[services[first], services[second]],
            pair=interval,
            complete=(rational(completion, "motif store completion")
                      if completion is not None else None),
        )
        occupied.update(pair)
    result = []
    for index, event in enumerate(events):
        if index in motifs:
            result.append(motifs[index])
        elif event["id"] not in occupied:
            result.append(dict(events=[event], services=[services[index]]))
    return result


def reservations(operation, partition):
    """Return alternative jointly reserved dispatch/execution resources."""
    service = operation["services"][0]
    dispatch = f"dispatch:{partition}"
    resource = service["resource"]
    if "pair" in operation:
        if any(item["resource"] != "lsu_scalar"
               for item in operation["services"]):
            raise ValueError("Pair motif requires two scalar LSU operations")
        first, second = [item["initiation"]
                         for item in operation["services"]]
        second_issue = max(Fraction(1), first)
        spans = {dispatch: second_issue + 1,
                 "lsu_scalar": second_issue + second}
        if operation["pair"] < max(spans.values()):
            raise ValueError("Motif interval is shorter than its issue span")
        return [spans]
    if resource == "fp32":
        return [
            {dispatch: Fraction(1), f"fp32_dedicated:{partition}": Fraction(2)},
            {dispatch: Fraction(1), f"fp32_integer:{partition}": Fraction(2)},
        ]
    if resource == "integer_like":
        return [{dispatch: Fraction(1),
                 f"fp32_integer:{partition}": service["initiation"] * 4}]
    if resource == "dispatch":
        return [{dispatch: Fraction(1)}]
    if resource == "constant_broadcast":
        return [{dispatch: Fraction(1),
                 f"constant_broadcast:{partition}": service["initiation"]}]
    return [{dispatch: Fraction(1), resource: service["initiation"]}]


def run_wave(ops, resident_warps, warps_per_block, include_trace,
             data_cache=None, wave_index=0, origin=Fraction(0)):
    """Run an earliest-issue greedy schedule with explicit dependencies."""
    for operation in ops:
        events = operation["events"]
        operation["reads"] = set().union(*(
            register_keys(event.get("reads", [])) for event in events
        ))
        operation["writes"] = set().union(*(
            register_keys(event.get("writes", [])) for event in events
        ))
        operation["memory"] = [
            event["memory"] for event in events if event.get("memory")
            and event["memory"]["space"] != "constant"
        ]
        operation["routes"] = [reservations(operation, partition)
                               for partition in range(PARTITIONS)]
    positions = [0] * resident_warps
    register_ready = [{} for _ in positions]
    register_consumed = [{} for _ in positions]
    warp_ready = [origin for _ in positions]
    resource_ready = {}
    resource_demand = Counter()
    memory = {}
    trace = []
    clock = finish = origin
    instructions = 0
    while any(position < len(ops) for position in positions):
        choices = []
        for warp, position in enumerate(positions):
            if position == len(ops):
                continue
            operation = ops[position]
            reads, writes = operation["reads"], operation["writes"]
            earliest = max(clock, warp_ready[warp])
            for ref in reads | writes:
                earliest = max(earliest, register_ready[warp].get(ref, 0))
            for ref in writes:
                earliest = max(earliest, register_consumed[warp].get(ref, 0))
            for detail in operation["memory"]:
                earliest = max(earliest, memory_hazard(
                    memory, detail, warp, warps_per_block
                ))
            for route, spans in enumerate(operation["routes"][warp % 4]):
                start = max(earliest, *(resource_ready.get(key, 0)
                                        for key in spans))
                choices.append((start, warp, route, spans))
        start, warp, route, spans = min(choices, key=lambda item: item[:3])
        operation = ops[positions[warp]]
        events, services = operation["events"], operation["services"]
        service = services[0]
        clock = start
        for key, duration in spans.items():
            resource_ready[key] = start + duration
            resource_demand[key] += duration
        if "pair" in operation:
            result_ready = start + operation["pair"]
            consumed = visible = result_ready
            completion = max(result_ready, start + operation["complete"])
        elif service["opcode"] in ("STS", "STL"):
            result_ready = None
            consumed = start + service["source_consumed"]
            visible = start + service["visible"]
            completion = start + service["complete"]
        elif service["opcode"] == "BRA":
            result_ready = None
            consumed = start + service.get("source_consumed", 0)
            visible = None
            completion = start + service["warp_block"]
        else:
            result_ready = start + service["result"]
            consumed = start + service.get("source_consumed", 0)
            visible = None
            completion = max(result_ready, consumed)
        cache_access = None
        if data_cache is not None:
            local = [detail for detail in operation["memory"]
                     if detail["space"] == "local"]
            if "pair" in operation and local:
                data_cache.store(local[0], warp, wave_index, start, visible)
                data_cache.counts["contracted_pair_load_sectors"] += len(
                    data_cache.sectors(local[1], warp, wave_index))
                cache_access = dict(path="contracted_same_cell_pair",
                                    ready=result_ready)
            elif local and service["opcode"] == "LDL":
                cache_access = data_cache.load(
                    local[0], warp, wave_index, start)
                result_ready = cache_access["ready"]
                completion = max(result_ready, consumed)
            elif local and service["opcode"] == "STL":
                data_cache.store(local[0], warp, wave_index, start, visible)
        for ref in operation["reads"]:
            register_consumed[warp][ref] = max(
                register_consumed[warp].get(ref, 0), consumed
            )
        for ref in operation["writes"]:
            if result_ready is None:
                raise ValueError("Resultless service cannot write registers")
            register_ready[warp][ref] = result_ready
        for detail in operation["memory"]:
            key, offset, size = memory_identity(
                detail, warp, warps_per_block
            )
            ready = visible if detail["access"] == "write" else result_ready
            if ready is None:
                raise ValueError("Memory readiness is unspecified")
            frame = memory.setdefault(key, {})
            for address in range(offset, offset + size):
                state = frame.setdefault(address, {})
                access = detail["access"]
                state[access] = max(state.get(access, 0), ready)
        warp_ready[warp] = start + spans[f"dispatch:{warp % 4}"]
        if service["opcode"] == "BRA":
            warp_ready[warp] = max(warp_ready[warp], completion)
        finish = max(finish, completion, *(resource_ready[key] for key in spans))
        positions[warp] += 1
        instructions += len(events)
        if include_trace:
            trace.append(dict(
                warp=warp, partition=warp % 4,
                event_ids=[event["id"] for event in events],
                opcodes=[event["opcode"] for event in events],
                issue=start, result_ready=result_ready,
                source_consumed=consumed, memory_visible=visible,
                complete=completion, reservations=spans, route=route,
                data_cache=cache_access,
            ))
    if data_cache is not None:
        data_cache.advance(finish)
    return dict(
        wave_cycles=finish - origin, issued_instructions=instructions,
        resource_reserved_cycles=dict(sorted(resource_demand.items())),
        trace=trace if include_trace else None,
    )


def schedule_events(events, catalog, scenario, resident_warps, work,
                    warps_per_block=None, include_trace=False):
    """Estimate a fully specified coherent event stream or request services.

    Parameters
    ----------
    events : list
        Allocated typed instructions with R/P references and byte offsets.
    catalog : dict
        Nominal physical service catalog, including transfer qualifications.
    scenario : dict
        Named, qualified overrides, memory issue assumptions and motifs.
    resident_warps : int
        Explicit occupancy capacity for one modeled SM.
    work : dict
        Common warp attempts in at least two complete synchronized waves.
    warps_per_block : int, optional
        Required when shared addresses alias across warps of one block.
    include_trace : bool
        Retain each issue and completion with resource reservations.

    Returns
    -------
    dict
        Exact rational schedule or explicit missing-service requests.
    """
    resident_warps = integer(resident_warps, "resident warps")
    if resident_warps > 48:
        raise ValueError("SM89 supports at most 48 resident warps")
    if catalog.get("kind") != "nominal_physical_service_catalog":
        raise ValueError("Nominal physical service catalog required")
    if catalog.get("solver_timings_consumed") is not False:
        raise ValueError("Solver timing inputs are not admitted")
    if not scenario.get("id") or scenario.get("full_warp_coherent") is not True:
        raise ValueError("Named coherent-full-warp scenario required")
    if warps_per_block is not None:
        integer(warps_per_block, "warps per block")
        if warps_per_block > 32 or resident_warps % warps_per_block:
            raise ValueError("Resident wave must contain complete CUDA blocks")
    events = executable_events(events)
    normalization = common_work(Fraction(0), resident_warps, work)
    missing = []
    if any(event.get("memory", {}).get("cross_warp_alias")
           for event in events if event.get("memory")):
        if warps_per_block is None:
            missing.append(dict(fields=["warps_per_block"],
                                reason="Shared aliases need block ownership"))
    services = [service_for(event, catalog, scenario, missing)
                for event in events]
    ops = operations(events, services, scenario, missing)
    paired = {event["id"] for operation in ops if "pair" in operation
              for event in operation["events"]}
    store_fields = {
        "store_source_consumed_cycles", "store_visible_cycles",
        "store_complete_cycles",
    }
    filtered = []
    for item in missing:
        if item.get("event_id") in paired:
            item = dict(item, fields=[field for field in item["fields"]
                                      if field not in store_fields])
        if item["fields"]:
            filtered.append(item)
    result = dict(
        kind="conditional_nominal_execution", scenario_id=scenario["id"],
        events_sha256=identity(events), catalog_sha256=identity(catalog),
        scenario_sha256=identity(scenario),
        resident_warps_per_sm=resident_warps,
        opcode_counts_per_warp=dict(sorted(Counter(
            event["opcode"] for event in events
        ).items())),
        common_work=normalization,
        missing_services=filtered,
        solver_timings_consumed=False,
        status="missing_services" if filtered else "finite_nominal_estimate",
        wave_cycles=None,
        scope="one declared event region per warp; synchronized SM waves",
        scheduling="earliest joint issue, warp-ID tie, dedicated-FP32 tie",
        assumptions=[
            "Live-in registers and initial memory are ready at region entry",
            "Inputs use sourced consumption delays where declared; "
            "other instruction inputs are assumed consumed at issue",
            "Load-before-store aliases wait for the prior load result",
            "Warps remain assigned to warp-ID modulo four partitions",
            "Two 16-lane FP32 routes are capacity-reservation hypotheses",
            "Aggregate SFU/vote/LSU intervals permit fractional SM times",
            "LDC broadcast hit service is separate from mutable frame traffic",
            "No instruction-fetch, outer-control or omitted caller cost",
        ],
    )
    if filtered:
        result["common_work"]["cycles"] = None
        result["common_work"]["cycles_per_warp_attempt"] = None
    else:
        cache_specification = scenario.get("data_cache")
        if cache_specification is None:
            wave = run_wave(ops, resident_warps, warps_per_block, include_trace)
            result.update(wave)
            result["common_work"] = common_work(
                wave["wave_cycles"], resident_warps, work
            )
        else:
            cache = NominalDataCache(cache_specification, resident_warps)
            origin = Fraction(0)
            waves = []
            demand = Counter()
            trace = []
            for index in range(normalization["resident_waves"]):
                wave = run_wave(
                    ops, resident_warps, warps_per_block, include_trace,
                    cache, index, origin,
                )
                wave["wave_index"] = index
                wave["origin_cycles"] = origin
                origin += wave["wave_cycles"]
                demand.update(wave["resource_reserved_cycles"])
                if include_trace:
                    trace.extend(dict(row, wave_index=index)
                                 for row in wave["trace"])
                waves.append(wave)
            total = normalization["warp_attempts_per_sm"]
            result.update(
                wave_cycles=None, wave_schedules=waves,
                issued_instructions=sum(w["issued_instructions"]
                                        for w in waves),
                resource_reserved_cycles=dict(sorted(demand.items())),
                trace=trace if include_trace else None,
                data_cache=cache.summary(),
                common_work=dict(normalization, cycles=origin,
                                 cycles_per_warp_attempt=origin / total),
            )
    return encoded(result)


def schedule_plan(plan, catalog, scenario, resident_warps, work,
                  warps_per_block=None, include_trace=False):
    """Schedule an actual conditional typed plan and bind its identity."""
    if plan.get("kind") not in (
        "conditional_typed_implicit_native_plan",
        "conditional_typed_explicit_native_plan",
    ):
        raise ValueError("Expected an actual conditional typed NativePlan")
    if (
        plan.get("native_labels_consumed") is not False
        or plan.get("measured_iteration_counts_consumed") is not False
    ):
        raise ValueError("Native/timing outcome labels are not prediction inputs")
    if scenario.get("data_cache") and scenario["data_cache"][
            "frame_bytes_per_thread"] != plan["allocation"][
                "local_frame_bytes"]:
        raise ValueError("Cache frame differs from actual typed allocation")
    result = schedule_events(
        plan["allocation"]["events"], catalog, scenario, resident_warps,
        work, warps_per_block, include_trace,
    )
    result["plan_sha256"] = identity(plan)
    result["placement_identity"] = plan.get("placement_identity")
    result["iteration_regime"] = plan["dynamic_work"]["iterations"]
    return result


def main():
    """Read a CPU-only request and write an inspectable schedule result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    call = schedule_plan if "plan" in request else schedule_events
    data = request.get("plan", request.get("events"))
    result = call(
        data, request["catalog"], request["scenario"],
        request["resident_warps"], request["common_work"],
        request.get("warps_per_block"), request.get("include_trace", False),
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
