"""Evaluate an attempted-step suffix with exact carried cache state."""

from collections import Counter, OrderedDict
from copy import deepcopy
from fractions import Fraction

from benchmarks.hardware_model import nominal_execution as execution
from benchmarks.hardware_model.nominal_data_cache import NominalDataCache
from benchmarks.hardware_model.nominal_instruction_cache import (
    NominalInstructionCache,
)


def cache_state(cache):
    """Retain every mutable cache field, including pending and ready times."""
    if cache is None:
        return None
    return state_contents(deepcopy({
        key: value for key, value in vars(cache).items()
        if key not in ("specification", "levels")}))


def state_contents(value):
    """Encode tuple-keyed ordered cache maps without losing key identity."""
    if isinstance(value, dict):
        if (not isinstance(value, OrderedDict)
                and all(isinstance(key, str) for key in value)):
            return {key: state_contents(item) for key, item in value.items()}
        return dict(
            mapping_type=type(value).__name__,
            entries=[dict(key=state_contents(key), value=state_contents(item))
                     for key, item in value.items()],
        )
    if isinstance(value, tuple):
        return dict(tuple=[state_contents(item) for item in value])
    if isinstance(value, list):
        return [state_contents(item) for item in value]
    return value


def state_counts(cache):
    """Copy cumulative cache traffic counters at one exact boundary."""
    return Counter() if cache is None else Counter(cache.counts)


def schedule_preceded_step(plan, catalog, scenario, resident_warps,
                           measured_attempts, block_threads):
    """Execute one predecessor wave and compare its exact measured suffix.

    Parameters
    ----------
    plan : dict
        Actual allocated typed plan accepted by the nominal executor.
    catalog : dict
        Qualified hardware instruction services.
    scenario : dict
        Complete memory and instruction-delivery assumptions.
    resident_warps : int
        Actual candidate residency on one modeled SM.
    measured_attempts : int
        Common measured work, at least two complete resident waves.
    block_threads : int
        Candidate block geometry in threads.

    Returns
    -------
    dict
        Exact suffix costs, carried state and uninterrupted replay receipt.
    """
    resident = execution.integer(resident_warps, "resident warps")
    attempts = execution.integer(measured_attempts, "measured attempts")
    block = execution.integer(block_threads, "block threads")
    if block % 32 or attempts % resident or attempts < 2 * resident:
        raise ValueError("Measured work needs at least two complete waves")
    data_specification = scenario.get("data_cache")
    if (data_specification is not None
            and data_specification["backing"] != "reused_physical_slots"):
        raise ValueError("Repeated resident steps require reused local slots")
    measured_waves = attempts // resident
    all_work = dict(kind="synchronized_full_waves",
                    warp_attempts_per_sm=attempts + resident)
    reference = execution.schedule_plan(
        plan, catalog, scenario, resident, all_work,
        warps_per_block=block // 32,
    )
    if reference["status"] != "finite_nominal_estimate":
        return dict(status=reference["status"], admission=reference)

    events = execution.executable_events(plan["allocation"]["events"])
    missing = []
    services = [execution.service_for(event, catalog, scenario, missing)
                for event in events]
    operations = execution.operations(events, services, scenario, missing)
    cache = (NominalDataCache(data_specification, resident)
             if data_specification is not None else None)
    fetch_specification = scenario.get("instruction_fetch")
    use_fetch = (fetch_specification is not None
                 and fetch_specification["mode"] == "hierarchy")
    fetch = NominalInstructionCache(fetch_specification) if use_fetch else None
    pcs = ({int(key): value for key, value in
            fetch_specification["event_pcs"].items()} if use_fetch else None)
    initial = dict(data=cache_state(cache), instruction=cache_state(fetch))
    predecessor = execution.run_wave(
        operations, resident, block // 32, False, cache, 0, Fraction(0),
        fetch, pcs,
    )
    origin = predecessor["wave_cycles"]
    boundary = dict(data=cache_state(cache), instruction=cache_state(fetch))
    before_data, before_fetch = state_counts(cache), state_counts(fetch)
    waves, resources = [], Counter()
    for index in range(1, measured_waves + 1):
        wave = execution.run_wave(
            operations, resident, block // 32, False, cache, index, origin,
            fetch, pcs,
        )
        wave.update(wave_index=index, origin_cycles=origin)
        origin += wave["wave_cycles"]
        resources.update(wave["resource_reserved_cycles"])
        waves.append(wave)
    total = origin - predecessor["wave_cycles"]
    expected_total = execution.rational(
        reference["common_work"]["cycles"], "reference total")
    if origin != expected_total:
        raise ValueError("Carried execution differs from uninterrupted work")
    if "wave_schedules" in reference:
        expected = reference["wave_schedules"]
        if (len(expected) != measured_waves + 1
                or execution.encoded(predecessor["wave_cycles"])
                != expected[0]["wave_cycles"]):
            raise ValueError("Predecessor differs from uninterrupted prefix")
        for actual, retained in zip(waves, expected[1:], strict=True):
            if execution.encoded(actual) != retained:
                raise ValueError("Measured suffix differs from reference")
    for name, cache_object in (("data_cache", cache),
                               ("instruction_fetch", fetch)):
        if cache_object is not None and execution.encoded(
                cache_object.summary()) != reference[name]:
            raise ValueError("Final cache state differs from reference")
    return execution.encoded(dict(
        status="finite_preceded_step_estimate",
        kind="one_identical_predecessor_wave",
        plan_sha256=execution.identity(plan),
        catalog_sha256=execution.identity(catalog),
        scenario_sha256=execution.identity(scenario),
        predecessor=dict(warp_attempts_per_sm=resident,
                         cycles=predecessor["wave_cycles"]),
        common_work=dict(
            kind="synchronized_full_waves_after_one_predecessor",
            warp_attempts_per_sm=attempts, resident_waves=measured_waves,
            cycles=total, cycles_per_warp_attempt=total / attempts,
        ),
        wave_schedules=waves,
        issued_instructions=sum(wave["issued_instructions"] for wave in waves),
        resource_reserved_cycles=dict(sorted(resources.items())),
        measured_data_counts=dict(state_counts(cache) - before_data),
        measured_instruction_counts=dict(state_counts(fetch) - before_fetch),
        initial_state=initial, predecessor_boundary_state=boundary,
        final_state=dict(data=cache_state(cache), instruction=cache_state(fetch)),
        uninterrupted_reference=dict(
            sha256=execution.identity(reference),
            warp_attempts_per_sm=attempts + resident, cycles=expected_total,
            exact_suffix_verified=True,
        ),
        assumptions=[
            "Exactly one preceding attempted step per resident warp",
            "Same declared source event stream and services in every wave",
            "Synchronized drained compute boundary between waves",
            "Initial operands ready under the existing nominal contract",
            "Live cache objects retain all pending/request clocks",
            "Predecessor cost excluded; measured attempted work stays common",
            "This is carried-state sensitivity, not steady-state convergence",
            "Actual solver per-warp overlap and caller transitions omitted",
        ],
        solver_timings_consumed=False, native_labels_consumed=False,
        native_compilations=0, gpu_launches=0,
    ))
