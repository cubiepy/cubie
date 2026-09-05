"""Build source-bound finite service hypotheses for attempted implicit steps."""

from fractions import Fraction
from statistics import median

from benchmarks.hardware_model import candidate_selection as selection
from benchmarks.hardware_model import implicit_policy_graph as policy
from benchmarks.hardware_model import instruction_addresses
from benchmarks.hardware_model import nominal_execution as execution


OBJECTIVE = "attempted_algorithm_step_with_caller_visible_state"


def fraction(value):
    """Read a catalog rational without introducing floating-point fitting."""
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    if isinstance(value, list):
        return Fraction(*value)
    return Fraction(value)


def service(value, assumption, provenance, **fields):
    """Attach a physical origin and transfer hypothesis to a service."""
    return dict(provenance=provenance, assumption=assumption,
                **{key: execution.encoded(val) for key, val in fields.items()},
                **value)


def constant_footprints(lowering):
    """Report payload dedup and duplicated-source-owner layout hypotheses."""
    tables = lowering.get("immutable_tables", [])
    payload = sum(table["materialized_bytes"] for table in tables)
    owners = sum(table["materialized_bytes"]
                 * len(table["source_views"]) for table in tables)
    return dict(
        identical_payload_dedup_bytes=payload,
        duplicated_table_per_source_owner_bytes=owners,
        assumption=(
            "Payload IDs do not establish physical constant-global identity. "
            "Installed backend keys globals by Python object identity. "
            "Counting each captured owner separately is a duplication "
            "scenario; actual cross-call object identity is indeterminate."
        ),
    )


def build_scenario(graph, plan, catalog, hardware, geometry, evidence,
                   store_envelope="pair_readback", local_path="l1_hit"):
    """Bind a finite sensitivity hypothesis to an actual typed allocation.

    Parameters
    ----------
    graph, plan, catalog, hardware : dict
        Actual policy source, typed plan, nominal catalog and capacities.
    geometry : dict
        Qualified block_threads, static_shared and carveout hypothesis.
        Dynamic shared bytes derive from the actual per-run source stride.
    evidence : dict
        Qualified ballot receipt and constant-cache-hit assumption.
        Optional capacity fields are reported without forcing a hit ratio.
    store_envelope : str
        Whole-pair readback or source-input-consumption sensitivity.
    local_path : str
        One complete local load path; alternatives retain conflicting L2
        published values. No cache penalties are added to that path.

    Returns
    -------
    dict
        Scheduler scenario, occupancy, source layout and qualifications.
    """
    execution.qualified(geometry, "geometry")
    if plan.get("kind") not in (
        "conditional_implicit_policy_native_plan",
        "conditional_explicit_policy_native_plan",
    ):
        raise ValueError("Scenario binding requires a complete policy plan")
    policy.verify_policy_plan(graph, plan)
    typed = plan["typed_plan"]
    if not selection.full_warp_allocation_supported(graph):
        raise ValueError("Finite scenario needs full-warp source regimes")
    allocation = typed["allocation"]
    events = execution.executable_events(allocation["events"])
    construction = graph["candidate_construction"]
    stride = construction["shared_stride_bytes"]
    if construction["precision"] != "float32" or stride % 4:
        raise ValueError("Actual FP32 source stride must be word aligned")
    block = execution.integer(geometry["block_threads"], "block threads")
    registers = allocation["peak_resident"]["R"]
    # BatchSolverKernel.launch_kernel reserves at least one shared word.
    dynamic_shared = max(4, stride * block)
    occupancy = selection.residency(
        hardware, registers, block, geometry["static_shared"],
        dynamic_shared, geometry["carveout"],
    )
    if not occupancy["legal"]:
        return dict(status="illegal_geometry", occupancy=occupancy)
    provenance = [catalog["sources"]["icache_profiles"]]
    branch = median(fraction(v) for v in catalog["control_motif"][
        "branch_resolving_warp_cycles_per_runtime_iteration"])
    overrides = dict(BRA=service(
        {}, "Recurring branch-resolving counter motif transferred to "
        "each typed BRA; includes loop form and endpoint administration. "
        "A nominal control-form scenario, not intrinsic branch latency.",
        provenance, warp_block_cycles=branch,
    ))
    if any(e["opcode"] == "ACTIVEMASK" for e in events):
        ballot = evidence["ballot"]
        execution.qualified(ballot, "ballot")
        receipt = ballot["receipt"]
        if receipt["status"] != "INDEPENDENT_PASS_BALLOT_COMPARE_MEASUREMENT":
            raise ValueError("Verified ballot measurement is required")
        intervals = [fraction(row["exact"]) for row in receipt[
            "ordinary_exact_intervals"] if row["population"] == 1]
        mask = median(intervals)
        overrides["ACTIVEMASK"] = service(
            {}, "Dependent ballot+ISETP+loop administration median "
            "transferred to ACTIVEMASK result readiness. No comparison "
            "latency is subtracted; native opcode equivalence is unproved.",
            ballot["provenance"], result_latency_cycles=mask,
            initiation_cycles=Fraction(1, 2),
        )
    paths = catalog["memory_paths"]
    choices = dict(
        l1_hit=paths["ldl_l1_hit"]["cycles"],
        l2_273=paths["ldl_l2_hit_alternatives"][0]["cycles"],
        l2_284_8=paths["ldl_l2_hit_alternatives"][1]["cycles"],
        dram_571=paths["ldl_dram"]["cycles"],
    )
    if local_path not in choices:
        raise ValueError("Unknown complete local load path")
    overrides["LDL"] = service(
        {}, "Whole selected load path transferred from RTX4090 to local "
        "Ada memory; LSU issue remains a front-end resource ceiling. "
        "Downstream L2/DRAM bandwidth is not identified by this scenario.",
        [catalog["sources"]["ada_memory"]],
        result_latency_cycles=fraction(choices[local_path]),
    )
    for opcode, space in (("STS", "shared"), ("STL", "local")):
        pair = fraction(catalog["same_cell_store_load_motifs"][space][
            "nominal_pair_interval"])
        consumption = fraction(catalog["instructions"][opcode][
            "source_consumed"]["cycles"])
        if store_envelope == "pair_readback":
            visible = pair
        elif store_envelope == "input_consumption":
            visible = consumption
        else:
            raise ValueError("Unknown store sensitivity envelope")
        overrides[opcode] = service(
            {}, "Caller-region completion is assigned at the selected "
            "store visibility hypothesis: whole same-cell readback "
            "motif or source-input consumption. These are sensitivity "
            "envelopes, not proven runtime bounds or kernel drain times. "
            "A separately scheduled consumer still has its own load "
            "result latency; no exact pair contract is asserted here.",
            [catalog["same_cell_store_load_motifs"],
             catalog["instructions"][opcode]["source_consumed"]],
            store_source_consumed_cycles=consumption,
            store_visible_cycles=visible, store_complete_cycles=visible,
        )
    wavefronts = {}
    for event in events:
        memory = event.get("memory")
        if not memory or memory["space"] == "constant":
            continue
        count = 1
        if memory["space"] == "shared":
            if stride <= 0 or memory["bytes"] != 4:
                raise ValueError("Shared scenario needs scalar private slices")
            addresses = [lane * stride + memory["offset"]
                         for lane in range(32)]
            banks = [set() for _ in range(32)]
            for address in addresses:
                banks[(address // 4) % 32].add(address)
            count = max(map(len, banks))
        wavefronts[str(event["id"])] = count
    scenario = dict(
        id=f"nominal_e3_{store_envelope}_{local_path}",
        full_warp_coherent=True, service_overrides=overrides,
        memory_issue=dict(
            kind="scalar_full_warp_wavefronts", default_wavefronts=1,
            event_wavefronts=wavefronts,
            provenance=[geometry["provenance"], construction],
            assumption="Private per-run shared slices use actual stride; "
            "32 banks of 4 bytes serialize distinct words in a bank. "
            "Scalar same-slot local warp accesses use four 32-byte "
            "sectors and one nominal scalar LSU request.",
        ),
    )
    if any(e["opcode"] == "LDC" for e in events):
        constant = evidence["constant_cache"]
        execution.qualified(constant, "constant_cache")
        scenario["constant_cache"] = dict(
            constant, kind="immutable_broadcast_hit")
    if evidence.get("store_load_motifs"):
        scenario["store_load_motifs"] = evidence["store_load_motifs"]
    frame = allocation["local_frame_bytes"]
    warps = occupancy["resident_warps_per_sm"]
    capacity = evidence.get("capacity_hypothesis")
    if capacity:
        execution.qualified(capacity, "capacity_hypothesis")
        available = capacity["unified_l1_shared_bytes"] - geometry["carveout"]
        if available < 0:
            raise ValueError("Shared carveout exceeds unified capacity")
        footprint = frame * 32 * warps
        capacity = dict(
            capacity, l1_after_carveout_bytes=available,
            resident_local_footprint_fits_l1=footprint <= available,
            simultaneous_device_local_footprint_bytes=(
                footprint * hardware["multiprocessor_count"]),
            simultaneous_device_local_footprint_fits_l2=(
                footprint * hardware["multiprocessor_count"]
                <= capacity["l2_bytes"]),
            qualification="Capacity comparison excludes other data, tags "
            "and set conflicts. It is not a hit/miss classification.",
        )
    bound = dict(
        status="finite_conditional_scenario", objective=OBJECTIVE,
        identity=dict(graph_sha256=execution.identity(graph),
                      plan_sha256=execution.identity(plan),
                      verification="actual_policy_plan_reconstruction"),
        scenario=scenario, occupancy=occupancy, block_threads=block,
        register_count=dict(value=registers,
                            source="typed_allocation_peak_resident_R",
                            omitted_outer_ABI_registers=True),
        geometry_hypothesis=geometry,
        layout=dict(shared_stride_bytes=stride,
                    dynamic_shared_bytes=dynamic_shared,
                    local_frame_bytes_per_thread=frame,
                    resident_local_footprint_bytes=frame * 32 * warps,
                    constant=constant_footprints(typed["lowering"])),
        capacity_hypothesis=capacity,
        qualifications=[
            "Attempted-step objective excludes omitted caller work and "
            "kernel-global writeback drain.",
            "Neither store envelope bounds physical kernel runtime.",
            "Selected cache paths are explicit hypotheses. Footprint "
            "alone does not identify cache set conflicts or a miss ratio.",
            "Instruction-delivery service remains unmodeled; do not "
            "interpret this schedule as total kernel latency.",
        ],
    )
    if evidence.get("instruction_fetch"):
        bound = bind_instruction_fetch(
            bound, graph, plan, evidence["instruction_fetch"])
    return bound


def forecast(graph, plan, catalog, hardware, geometry, evidence, work,
             store_envelope="pair_readback", local_path="l1_hit"):
    """Schedule common attempted-step work under a declared finite scenario."""
    result = build_scenario(graph, plan, catalog, hardware, geometry,
                            evidence, store_envelope, local_path)
    if result["status"] != "finite_conditional_scenario":
        return result
    if evidence.get("data_cache"):
        result = bind_data_cache(result, catalog, hardware,
                                 evidence["data_cache"])
    result["schedule"] = execution.schedule_plan(
        plan.get("typed_plan", plan), catalog, result["scenario"],
        result["occupancy"]["resident_warps_per_sm"], work,
        warps_per_block=result["block_threads"] // 32,
    )
    return result


def bind_data_cache(bound, catalog, hardware, hypothesis):
    """Derive per-event cache capacity from actual occupancy and layout."""
    execution.qualified(hypothesis, "data_cache")
    l1 = hypothesis["unified_l1_shared_bytes"] - bound[
        "geometry_hypothesis"]["carveout"]
    l2 = hypothesis["l2_bytes"] // hardware["multiprocessor_count"]
    if l1 < 0:
        raise ValueError("Carveout exceeds unified L1/shared capacity")
    alternatives = {"l2_273": 0, "l2_284_8": 1}
    l2_path = hypothesis["l2_path"]
    if l2_path not in alternatives:
        raise ValueError("Choose one of the conflicting published L2 paths")
    paths = catalog["memory_paths"]
    cache = dict(
        kind="sector_lru_data_cache", l1_capacity_bytes=l1,
        l2_capacity_bytes=l2,
        frame_bytes_per_thread=bound["layout"]["local_frame_bytes_per_thread"],
        load_path_cycles=dict(
            l1=paths["ldl_l1_hit"]["cycles"],
            l2=paths["ldl_l2_hit_alternatives"][alternatives[l2_path]][
                "cycles"],
            dram=paths["ldl_dram"]["cycles"],
        ),
        provenance=hypothesis["provenance"],
        assumption=hypothesis["assumption"],
        capacity_partition="nominal_equal_per_SM_L2_share",
    )
    for key in ("initial_state", "backing", "write_policy",
                "outstanding_fills"):
        cache[key] = hypothesis[key]
    for key in ("l1_seed_sectors", "l2_seed_sectors"):
        if key in hypothesis:
            cache[key] = hypothesis[key]
    scenario = dict(bound["scenario"], data_cache=cache)
    scenario["id"] += "_capacity_paths_" + l2_path
    return dict(bound, scenario=scenario,
                cache_binding=dict(hypothesis=hypothesis,
                                   frame_source="typed_allocation",
                                   l1_after_carveout_bytes=l1,
                                   l2_nominal_share_bytes=l2))


def bind_instruction_fetch(bound, graph, plan, hypothesis, projection=None):
    """Bind a qualified hierarchy to actual allocated source-PC identities."""
    execution.qualified(hypothesis, "instruction_fetch")
    if (bound["identity"]["graph_sha256"] != execution.identity(graph)
            or bound["identity"]["plan_sha256"] != execution.identity(plan)):
        raise ValueError("Instruction scenario/source plan identities differ")
    if hypothesis["mode"] == "disabled":
        scenario = dict(bound["scenario"], instruction_fetch=hypothesis)
        scenario["id"] += "_instruction_disabled"
        return dict(bound, scenario=scenario)
    if hypothesis["mode"] != "hierarchy":
        raise ValueError("Unknown instruction hierarchy alternative")
    options = hypothesis["projection"]
    reconstructed = instruction_addresses.project_instruction_addresses(
        graph, plan, **options)
    if projection is not None and execution.identity(projection) != (
            execution.identity(reconstructed)):
        raise ValueError("Supplied instruction projection differs from source")
    projection = reconstructed
    events = plan["typed_plan"]["allocation"]["events"]
    if (projection["graph_sha256"] != execution.identity(graph)
            or projection["plan_sha256"] != execution.identity(plan)
            or projection["event_sha256"] != execution.identity(events)
            or any(projection[key] != value for key, value in options.items())
            or len(projection["event_to_pc"]) != len(events)):
        raise ValueError("Instruction projection differs from source inputs")
    pcs = {str(event["id"]): pc for event, pc in zip(
        events, projection["event_to_pc"], strict=True) if event.get("opcode")}
    fetch = dict(
        hypothesis, event_pcs=pcs,
        typed_plan_sha256=execution.identity(plan["typed_plan"]),
        allocation_events_sha256=execution.identity(events),
        projection_sha256=execution.identity(projection),
    )
    scenario = dict(bound["scenario"], instruction_fetch=fetch)
    scenario["id"] += "_instruction_" + hypothesis["id"]
    qualifications = [text for text in bound["qualifications"]
                      if not text.startswith("Instruction-delivery")]
    qualifications.append(
        "Instruction delivery follows named source-PC cache-domain and "
        "service hypotheses; physical cross-SM mapping is not inferred.")
    return dict(
        bound, scenario=scenario, qualifications=qualifications,
        instruction_binding=dict(
            projection_sha256=execution.identity(projection),
            synthetic_span_bytes=projection["span_bytes"],
            accessed_union_bytes=16 * len(projection["accessed_pc_union"]),
            hypothesis=hypothesis,
        ),
    )
