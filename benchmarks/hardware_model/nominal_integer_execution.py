"""Exact integer-clock execution for fixed-service nominal scenarios."""

from collections import Counter
from fractions import Fraction
from math import lcm

from benchmarks.hardware_model.nominal_execution import (
    PARTITIONS,
    common_work,
    encoded,
    executable_events,
    identity,
    integer,
    memory_identity,
    operations,
    qualified,
    rational,
    register_keys,
    reservations,
    service_for,
)


TIME_FIELDS = frozenset({
    "result", "source_consumed", "visible", "complete", "warp_block",
    "initiation",
})


def prepare_ticks(ops):
    """Resolve exact interval denominators and prepare integer operations.

    Parameters
    ----------
    ops : list
        Resolved operations from the ordinary service parser.

    Returns
    -------
    tuple
        Integer ticks per cycle and fresh operations with integer services.
    """
    intervals = []
    routes = []
    for operation in ops:
        for service in operation["services"]:
            intervals.extend(value for key, value in service.items()
                             if key in TIME_FIELDS and value is not None)
        intervals.extend(operation[key] for key in ("pair", "complete")
                         if key in operation)
        choices = [reservations(operation, partition)
                   for partition in range(PARTITIONS)]
        intervals.extend(value for partition in choices
                         for route in partition for value in route.values())
        routes.append(choices)
    unit = lcm(*(rational(value, "tick interval").denominator
                 for value in intervals))

    def ticks(value):
        scaled = Fraction(value) * unit
        if scaled.denominator != 1:
            raise ValueError("Service interval is outside the exact tick grid")
        return scaled.numerator

    prepared = []
    for operation, choices in zip(ops, routes):
        events = operation["events"]
        item = dict(operation)
        item["services"] = [
            {key: ticks(value) if key in TIME_FIELDS and value is not None
             else value
             for key, value in service.items()}
            for service in operation["services"]
        ]
        for key in ("pair", "complete"):
            if key in item:
                item[key] = ticks(item[key])
        item["reads"] = set().union(*(
            register_keys(event.get("reads", [])) for event in events
        ))
        item["writes"] = set().union(*(
            register_keys(event.get("writes", [])) for event in events
        ))
        item["dependencies"] = item["reads"] | item["writes"]
        item["memory"] = [
            event["memory"] for event in events if event.get("memory")
            and event["memory"]["space"] != "constant"
        ]
        item["routes"] = [
            [{key: ticks(value) for key, value in route.items()}
             for route in partition] for partition in choices
        ]
        prepared.append(item)
    return unit, prepared


def run_wave(ops, resident_warps, warps_per_block, include_trace):
    """Schedule fixed services with integer max/add and exact original ties."""
    unit, ops = prepare_ticks(ops)
    positions = [0] * resident_warps
    register_ready = [{} for _ in positions]
    register_consumed = [{} for _ in positions]
    warp_ready = [0] * resident_warps
    resource_ready = {}
    resource_demand = Counter()
    memory = {}
    trace = []
    clock = finish = instructions = 0
    remaining = len(ops) * resident_warps
    while remaining:
        best = None
        best_spans = None
        for warp, position in enumerate(positions):
            if position == len(ops):
                continue
            operation = ops[position]
            earliest = max(clock, warp_ready[warp])
            ready = register_ready[warp]
            consumed = register_consumed[warp]
            for ref in operation["dependencies"]:
                earliest = max(earliest, ready.get(ref, 0))
            for ref in operation["writes"]:
                earliest = max(earliest, consumed.get(ref, 0))
            for detail in operation["memory"]:
                key, offset, size = memory_identity(
                    detail, warp, warps_per_block)
                frame = memory.get(key, {})
                for address in range(offset, offset + size):
                    state = frame.get(address, {})
                    earliest = max(earliest, state.get("write", 0))
                    if detail["access"] == "write":
                        earliest = max(earliest, state.get("read", 0))
            for route, spans in enumerate(operation["routes"][warp % 4]):
                start = earliest
                for key in spans:
                    start = max(start, resource_ready.get(key, 0))
                choice = (start, warp, route)
                if best is None or choice < best:
                    best, best_spans = choice, spans
        start, warp, route = best
        spans = best_spans
        operation = ops[positions[warp]]
        events = operation["events"]
        service = operation["services"][0]
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
        for ref in operation["reads"]:
            register_consumed[warp][ref] = max(
                register_consumed[warp].get(ref, 0), consumed)
        for ref in operation["writes"]:
            if result_ready is None:
                raise ValueError("Resultless service cannot write registers")
            register_ready[warp][ref] = result_ready
        for detail in operation["memory"]:
            key, offset, size = memory_identity(
                detail, warp, warps_per_block)
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
        finish = max(finish, completion,
                     *(resource_ready[key] for key in spans))
        positions[warp] += 1
        remaining -= 1
        instructions += len(events)
        if include_trace:
            trace.append(dict(
                warp=warp, partition=warp % 4,
                event_ids=[event["id"] for event in events],
                opcodes=[event["opcode"] for event in events],
                issue=Fraction(start, unit),
                result_ready=(None if result_ready is None else
                              Fraction(result_ready, unit)),
                source_consumed=Fraction(consumed, unit),
                memory_visible=(None if visible is None else
                                Fraction(visible, unit)),
                complete=Fraction(completion, unit),
                reservations={key: Fraction(value, unit)
                              for key, value in spans.items()},
                route=route, data_cache=None,
            ))
    return dict(
        wave_cycles=Fraction(finish, unit),
        issued_instructions=instructions,
        resource_reserved_cycles={key: Fraction(value, unit)
                                  for key, value in
                                  sorted(resource_demand.items())},
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
    if scenario.get("data_cache") is not None:
        raise ValueError("Integer scheduler does not support stateful data caches")
    fetch = scenario.get("instruction_fetch")
    if fetch is not None:
        qualified(fetch, "instruction_fetch")
        if fetch.get("mode") != "disabled":
            raise ValueError("Integer scheduler requires disabled instruction fetch")
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
            "Outer-control and omitted caller work are not represented",
            "Instruction fetch is omitted unless a hierarchy is supplied",
        ],
    )
    if filtered:
        result["common_work"]["cycles"] = None
        result["common_work"]["cycles_per_warp_attempt"] = None
    else:
        wave = run_wave(ops, resident_warps, warps_per_block, include_trace)
        result.update(wave)
        result["common_work"] = common_work(
            wave["wave_cycles"], resident_warps, work
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
    fetch = scenario.get("instruction_fetch")
    if fetch and fetch.get("mode") == "hierarchy" and (
            fetch["typed_plan_sha256"] != identity(plan)
            or fetch["allocation_events_sha256"] != identity(
                plan["allocation"]["events"])):
        raise ValueError("Instruction addresses differ from actual typed plan")
    result = schedule_events(
        plan["allocation"]["events"], catalog, scenario, resident_warps,
        work, warps_per_block, include_trace,
    )
    result["plan_sha256"] = identity(plan)
    result["placement_identity"] = plan.get("placement_identity")
    result["iteration_regime"] = plan["dynamic_work"]["iterations"]
    return result
