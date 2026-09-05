"""Evaluate finite source policy/placement designs under hardware scenarios."""

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import itertools
import json
import math
from pathlib import Path

from cubie import Solver
from cubie.cache_root import get_cache_root_override, set_cache_root

from benchmarks import placement_landscape as landscape
from benchmarks.hardware_model import candidate_selection as selection
from benchmarks.hardware_model import implicit_policy_graph as policy
from benchmarks.hardware_model import implicit_source_graph as source
from benchmarks.hardware_model import implicit_workload as workload
from benchmarks.hardware_model import instruction_addresses as addresses
from benchmarks.hardware_model import nominal_execution as execution
from benchmarks.hardware_model import (
    nominal_integer_execution as integer_execution,
)
from benchmarks.hardware_model import nominal_scenarios as scenarios
from benchmarks.hardware_model.nominal_data_cache import NominalDataCache


LEVELS = ("full", "count1", "count2", "count4", "false")
ALGORITHMS = ("rk23", "kvaerno3", "radau_iia_3", "rosenbrock23")


def save(path, value):
    """Retain an exact JSON artifact with a content receipt."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(execution.encoded(value), handle, separators=(",", ":"))
    else:
        path.write_text(json.dumps(execution.encoded(value), indent=2),
                        encoding="utf-8")
    return dict(path=str(path.resolve()), sha256=selection.file_digest(path))


def default_request(evidence_root, design="pilot"):
    """Build a source-only request using the retained hardware ledger."""
    root = Path(evidence_root).resolve()
    config = root / "implicit_policy_graph_cpu_e1"
    hardware = json.loads((root / "verification" / (
        "candidate_selection_cpu_e3_20260905/request.json"
    )).read_text())["hardware"]
    ballot = root / "verification" / (
        "cpu_continuation_independent_20260905/"
        "ballot_measurement_independent_e1/receipt.json"
    )
    catalog_path = Path(execution.__file__).with_name(
        "NOMINAL_SERVICE_CATALOG.json")
    catalog = json.loads(catalog_path.read_text())
    request = dict(
        schema=1, system="lorenz", design=design,
        workloads=[dict(algorithm=algorithm, inner=inner)
                   for algorithm in ALGORITHMS
                   for inner in ((None,) if algorithm == "rk23" else
                                 ("lu", "mr", "bicgstab"))],
        source_regime=dict(newton_bodies=1, krylov_bodies=1,
                           fsal_state=dict(first_step=True,
                                           all_lanes_accepted=True)),
        compiler=json.loads((config / "compiler.json").read_text()),
        architecture=json.loads((config / "architecture.json").read_text()),
        compiler_caps=("source_unpressured" if design == "pilot" else
                       "occupancy_breakpoints"),
        shared_forwarding=[False, True],
        block_threads=[128],
        carveouts=([hardware["supported_shared_carveouts"][-1]]
                   if design == "pilot" else
                   hardware["supported_shared_carveouts"]),
        store_envelopes=["pair_readback", "input_consumption"],
        local_paths=(["l1_hit"] if design == "pilot" else
                     ["l1_hit", "l2_273", "l2_284_8", "dram_571"]),
        hardware=hardware, catalog=catalog,
        evidence=dict(
            ballot=dict(receipt=json.loads(ballot.read_text()),
                        provenance=[str(ballot)],
                        assumption="Audited complete ballot motif transfer"),
            constant_cache=dict(
                provenance=[catalog["sources"]["huerta_memory"]],
                assumption="Indexed constant broadcast-hit alternative"),
        ),
        instruction_delivery=dict(
            kind="symbolic_candidate_specific_fetch",
            assumption="Finite compute/data ranking excludes fetch service",
            provenance=["INSTRUCTION_DELIVERY_RESEARCH.md"],
        ),
    )
    if design != "pilot":
        fetch_path = root / "verification" / (
            "nominal_instruction_author_20260905/actual_e3/"
            "lorenz_kvaerno3_full_default.json")
        transfer = json.loads(fetch_path.read_text())["binding"][
            "instruction_binding"]["hypothesis"]
        request["instruction_fetch_hypotheses"] = [dict(
            id="perfect_delivery", mode="disabled",
            provenance=["Explicit perfect instruction-delivery baseline"],
            assumption="No instruction-fetch delay; delivery ceiling only",
        ), dict(transfer, evidence_fixture=dict(
            path=str(fetch_path), sha256=selection.file_digest(fetch_path),
            review_status="Author fixture; independent receipt separate"))]
        request["source_regimes"] = [dict(
            id=f"N{newton}_K{krylov}", newton_bodies=newton,
            krylov_bodies=krylov,
            fsal_state=request["source_regime"]["fsal_state"],
        ) for newton, krylov in itertools.product((1, 2, 4), repeat=2)]
        request["source_regimes"].append(dict(
            id="source_caps", newton_bodies="source_cap",
            krylov_bodies="source_cap",
            fsal_state=request["source_regime"]["fsal_state"],
        ))
        request["local_paths"] = ["l2_273", "l2_284_8"]
        request["evidence"]["data_cache"] = dict(
            unified_l1_shared_bytes=131072, l2_bytes=50331648,
            l2_path="l2_273", initial_state="cold",
            backing="reused_physical_slots", write_policy="write_through_l1",
            outstanding_fills="unlimited_ceiling",
            provenance=["HARDWARE_EVIDENCE.md", "NOMINAL_DATA_CACHE.md"],
            assumption="Published Ada unified capacity and recorded host L2; "
            "fully associative sectors and equal per-SM L2 share are "
            "explicit physical hypotheses",
        )
    return request


def policy_design(groups, design):
    """Enumerate an explicit finite design without measured winner tables."""
    full = ("full",) * len(policy.GROUPS)
    choices = [full]
    if isinstance(design, list):
        choices = [tuple(item) for item in design]
    elif design == "pilot":
        choices.append(("count1",) + full[1:])
    elif design == "oat_plus_joint":
        for group in policy.GROUPS:
            if group not in groups:
                continue
            for level in LEVELS[1:]:
                candidate = list(full)
                candidate[policy.GROUPS.index(group)] = level
                choices.append(tuple(candidate))
        for level in LEVELS[1:]:
            choices.append(tuple(level if group in groups else "full"
                                 for group in policy.GROUPS))
    else:
        raise ValueError("Unknown finite policy design")
    if any(len(item) != len(full) or not set(item) <= set(LEVELS)
           for item in choices):
        raise ValueError("Policy design requires eight supported levels")
    return sorted(set(choices))


def placement_design(targets, design):
    """Cross policies with local, single-shared and joint-shared settings."""
    local = {item["setting"]: "local" for item in targets}
    choices = [local]
    if isinstance(design, list):
        choices = [dict(local, **item) for item in design]
    elif design in ("pilot", "single_plus_joint"):
        if design == "single_plus_joint":
            choices.extend(dict(local, **{item["setting"]: "shared"})
                           for item in targets)
        choices.append({key: "shared" for key in local})
    else:
        raise ValueError("Unknown finite placement design")
    for item in choices:
        if set(item) != set(local) or not set(item.values()) <= {
                "local", "shared"}:
            raise ValueError("Placement design differs from source targets")
    return [json.loads(item) for item in sorted({
        selection.canonical(choice) for choice in choices})]


def construct_source(request, item, levels, locations, folder):
    """Construct one actual host factory graph without native compilation."""
    folder.mkdir(parents=True)
    set_cache_root(folder / "codegen")
    kwargs = landscape.solver_kwargs(request["system"], item["algorithm"])
    kwargs.update(request.get("solver_settings", {}))
    if item["inner"] is None:
        kwargs.pop("linear_correction_type", None)
    else:
        kwargs["linear_correction_type"] = workload.PUBLIC_LINEAR_TYPES[
            item["inner"]]
    kwargs.update(locations)
    kwargs["unroll"] = policy.policy_flags(levels)
    solver = Solver(landscape.SYSTEMS[request["system"]]["build"](), **kwargs)
    try:
        constants = request.get("system_constants", landscape.SYSTEMS[
            request["system"]].get("constants"))
        if constants:
            solver.update(constants)
        regime = request["source_regime"]
        if item["inner"] is None:
            declared, branches = {}, {}
            fsal = regime["fsal_state"]
        else:
            descriptor = workload.describe_implicit_workload(solver)
            declared = declared_regime(descriptor, regime)
            branches = ({"generic_dirk.py:744": False}
                        if descriptor["family"] == "DIRK" else {})
            if item["inner"] == "mr":
                branches.update({
                    "linear_solver.py:294": regime.get(
                        "mr_denominator_nonzero", True),
                    "linear_solver.py:299": regime.get(
                        "mr_update_active", True),
                })
            branches.update(regime.get("branch_choices", {}))
            fsal = None
        graph = policy.describe_policy_source(
            solver, declared, policy.policy_record(levels, kwargs["unroll"]),
            branches, fsal_state=fsal,
            numerical_replay=request.get("numerical_replay"),
        )
        registered = graph["workload"]["registry"]
        available = {entry["name"] for entry in registered
                     if entry["bytes"] > 0}
        targets = [dict(name=entry["name"],
                        setting=landscape.setting_name(entry["name"]))
                   for entry in landscape.candidate_buffers(solver)
                   if entry["name"] in available]
        for setting, location in locations.items():
            matches = [entry for entry in registered
                       if landscape.setting_name(entry["name"]) == setting]
            if not matches or any(entry["declared_location"] != location
                                  for entry in matches):
                raise ValueError("Requested placement missed source registry")
        return graph, targets
    finally:
        solver.close()


def comparison_identity(graph):
    """Bind actual semantic workload and complete selected execution regime."""
    role = graph["workload"]["roles"].get("main_linear", {})
    return dict(
        workload=graph["candidate_construction"]["workload_identity"],
        family=graph["workload"]["family"],
        inner=(role.get("actual_linear_correction_type")
               or role.get("solver_type")),
        regime=graph["regime"],
        scenarios=graph["scenario_contract"],
        explicit_step=graph.get("explicit_step_contract"),
    )


def declared_regime(descriptor, regime):
    """Resolve source-cap tokens separately for each actual solver role."""
    declared = {}
    for call in descriptor["step_calls"]:
        role = descriptor["roles"][call["role"]]
        is_newton = role["solver_type"] == "newton"
        linear = descriptor["roles"]["main_linear"] if is_newton else role
        newton = regime["newton_bodies"]
        krylov = regime["krylov_bodies"]
        if newton == "source_cap":
            newton = role["cap"] if is_newton else 0
        if krylov == "source_cap":
            krylov = linear["cap"]
        declared.update(source.uniform_regime(
            dict(descriptor, step_calls=[call]), newton, krylov))
    return declared


def source_cap_coverage(graph):
    """Compare executed recurrent bodies with their actual source caps."""
    descriptor = graph["workload"]
    rows = []
    for call in descriptor.get("step_calls", []):
        role = descriptor["roles"][call["role"]]
        actual = graph["regime"]["calls"][call["id"]]
        if role["solver_type"] == "newton":
            rows.append(dict(call=call["id"], role=call["role"],
                             bodies=actual["body_iterations"],
                             cap=role["cap"]))
            linear = descriptor["roles"]["main_linear"]
            if linear["solver_type"] != "lu":
                rows.extend(dict(
                    call=call["id"], role="main_linear", newton_body=index,
                    bodies=entry["body_iterations"], cap=linear["cap"],
                ) for index, entry in enumerate(actual["linear_calls"]))
        elif role["solver_type"] != "lu":
            rows.append(dict(call=call["id"], role=call["role"],
                             bodies=actual["body_iterations"],
                             cap=role["cap"]))
    return dict(
        recurrent_calls=rows,
        all_active_caps_reached=all(row["bodies"] == row["cap"]
                                    for row in rows),
        interpretation=("Exact cap endpoints do not cover every possible "
                        "intermediate count, mask or branch regime"),
    )


def register_caps(hardware, blocks, carveouts, maximum, mode):
    """Choose source peak and upper endpoints of hardware occupancy plateaus."""
    if mode == "source_unpressured":
        return [maximum]
    if mode != "occupancy_breakpoints":
        raise ValueError("Register caps must have a hardware/source origin")
    caps = {maximum}
    for block, carveout in itertools.product(blocks, carveouts):
        previous = None
        for registers in range(1, maximum + 1):
            result = selection.residency(hardware, registers, block, 0, 4,
                                         carveout)
            current = (result["legal"], result.get("resident_warps_per_sm"))
            if previous is not None and current != previous:
                caps.add(registers - 1)
            previous = current
    return sorted(caps)


def boundary_state(cache):
    """Describe all state affecting a drained, reused-slot next wave."""
    if cache.pending:
        raise ValueError("Wave boundary has unfinished cache transactions")
    return tuple((level, tuple(cache.caches[level].items()))
                 for level in ("l1", "l2"))


def schedule_repeated(plan, catalog, scenario, resident, attempts, block):
    """Use exact ticks or compress a repeated drained cache boundary state."""
    work = dict(kind="synchronized_full_waves",
                warp_attempts_per_sm=attempts)
    instruction = scenario.get("instruction_fetch", {})
    disabled = ("instruction_fetch" not in scenario or (
        isinstance(instruction, dict)
        and instruction.get("mode") == "disabled"
        and instruction.get("provenance") and instruction.get("assumption")
    ))
    legacy = any(scenario.get(key) for key in (
        "instruction_cache", "instruction_delivery"))
    if scenario.get("data_cache") is None and disabled and not legacy:
        return integer_execution.schedule_plan(
            plan, catalog, scenario, resident, work,
            warps_per_block=block // 32,
        )
    if (not scenario.get("data_cache")
            or instruction.get("mode") == "hierarchy"
            or any(scenario.get(key) for key in (
                "instruction_cache", "instruction_delivery"))):
        return execution.schedule_plan(plan, catalog, scenario, resident,
                                       work, warps_per_block=block // 32)
    specification = scenario["data_cache"]
    if specification["backing"] != "reused_physical_slots":
        return execution.schedule_plan(plan, catalog, scenario, resident,
                                       work, warps_per_block=block // 32)
    # Admission uses the ordinary engine, including exact pair contracts.
    checked = execution.schedule_plan(
        plan, catalog, scenario, resident,
        dict(kind="synchronized_full_waves", warp_attempts_per_sm=2 * resident),
        warps_per_block=block // 32,
    )
    if checked["status"] != "finite_nominal_estimate":
        return checked
    count = execution.common_work(Fraction(0), resident, work)["resident_waves"]
    events = execution.executable_events(plan["allocation"]["events"])
    missing = []
    services = [execution.service_for(event, catalog, scenario, missing)
                for event in events]
    operations = execution.operations(events, services, scenario, missing)
    cache = NominalDataCache(specification, resident)
    seen, waves, skips = {}, [], []
    index, origin = 0, Fraction(0)
    while index < count:
        state = boundary_state(cache)
        if state in seen:
            previous_index, previous_time, previous_counts = seen[state]
            length = index - previous_index
            repetitions = (count - index) // length
            if repetitions:
                period = origin - previous_time
                delta = Counter(cache.counts)
                delta.subtract(previous_counts)
                cache.counts.update({key: value * repetitions
                                     for key, value in delta.items()})
                skipped = length * repetitions
                skips.append(dict(first_wave=index, waves=skipped,
                                  period_waves=length, period_cycles=period,
                                  repeated_state_sha256=selection.digest(state)))
                index += skipped
                origin += period * repetitions
                cache.time = origin
                if index == count:
                    break
        else:
            seen[state] = (index, origin, Counter(cache.counts))
        result = execution.run_wave(operations, resident, block // 32, False,
                                    cache, index, origin)
        waves.append(dict(index=index, cycles=result["wave_cycles"]))
        origin += result["wave_cycles"]
        index += 1
    return execution.encoded(dict(
        status="finite_nominal_estimate", common_work=dict(
            work, resident_waves=count, cycles=origin,
            cycles_per_warp_attempt=origin / attempts),
        data_cache=cache.summary(), evaluated_waves=waves, skipped_waves=skips,
        plan_sha256=execution.identity(plan),
        compression="exact drained reused-slot boundary-state recurrence",
        admission=checked["status"], solver_timings_consumed=False,
    ))


def evaluate_workload(request, item, output):
    """Produce a comparable finite cost matrix for one family/inner pair."""
    output.mkdir(parents=True)
    try:
        baseline, targets = construct_source(
            request, item, ["full"] * 8, {}, output / "discovery")
    except (ValueError, policy.native.Unresolved) as error:
        return dict(status="no_admitted_sources", rejected=[dict(
            phase="source_discovery", exception=type(error).__name__,
            reason=str(error))])
    common = comparison_identity(baseline)
    groups = {entry["group"] for entry in baseline["policy_loops"]}
    design = request["design"]
    policies = policy_design(groups, request.get(
        "policies", "pilot" if design == "pilot" else "oat_plus_joint"))
    placements = placement_design(targets, request.get(
        "placements", "pilot" if design == "pilot" else "single_plus_joint"))
    hardware, compiler = request["hardware"], request["compiler"]
    architecture = dict(request["architecture"],
                        gpr_budget=hardware["max_registers_per_thread"])
    sources, rejected, source_peaks = [], [], []
    fetch_cases = request.get("instruction_fetch_hypotheses", [None])
    fetch_ids = [case["id"] if case else "unspecified" for case in fetch_cases]
    if not fetch_cases or len(set(fetch_ids)) != len(fetch_ids):
        raise ValueError("Instruction hypotheses need distinct identifiers")
    for index, (levels, locations) in enumerate(itertools.product(
            policies, placements)):
        folder = output / f"candidate_{index:04d}"
        try:
            graph, _ = construct_source(request, item, levels, locations, folder)
            if comparison_identity(graph) != common:
                raise ValueError("Candidate workload or execution regime differs")
            record = dict(id=f"source_{index:04d}", levels=levels,
                          locations=locations,
                          placement=selection.placement_identity(graph),
                          graph=save(folder / "graph.json.gz", graph),
                          folder=folder, source=graph, unpressured={})
            for forwarding in request["shared_forwarding"]:
                plan = policy.make_policy_plan(
                    graph, architecture, compiler, "promote",
                    shared_forwarding=forwarding,
                )
                record["unpressured"][forwarding] = plan
                source_peaks.append(plan["typed_plan"]["allocation"][
                    "peak_resident"]["R"])
            sources.append(record)
        except (ValueError, policy.native.Unresolved) as error:
            rejected.append(dict(levels=levels, locations=locations,
                                 phase="source_or_unpressured_allocation",
                                 exception=type(error).__name__, reason=str(error)))
    if not sources:
        return dict(status="no_admitted_sources", rejected=rejected)
    caps = register_caps(hardware, request["block_threads"],
                         request["carveouts"], max(source_peaks),
                         request["compiler_caps"])
    candidates, jobs, missing, bindings = {}, [], [], {}
    for record, cap, forwarding in itertools.product(
            sources, caps, request["shared_forwarding"]):
        graph = record["source"]
        compiler_id = f"R{cap}_forwarding{int(forwarding)}"
        try:
            plan = policy.make_policy_plan(
                graph, dict(architecture, gpr_budget=cap), compiler, "promote",
                shared_forwarding=forwarding,
            )
            plan_path = save(record["folder"] / (compiler_id + ".json.gz"), plan)
            projection = addresses.project_instruction_addresses(graph, plan)
            projection_path = save(record["folder"] / (
                compiler_id + "_addresses.json.gz"), projection)
        except (ValueError, policy.native.Unresolved) as error:
            missing.append(dict(source_id=record["id"], compiler=compiler_id,
                                reason=str(error)))
            continue
        for block, carveout in itertools.product(request["block_threads"],
                                                 request["carveouts"]):
            identifier = f"{record['id']}_b{block}_s{carveout}"
            geometry = dict(block_threads=block, static_shared=0,
                            carveout=carveout,
                            provenance=hardware["provenance"],
                            assumption="Explicit legal CUDA geometry hypothesis")
            for store, path, fetch in itertools.product(
                    request["store_envelopes"], request["local_paths"],
                    fetch_cases):
                evidence = dict(request["evidence"])
                if fetch:
                    evidence["instruction_fetch"] = fetch
                bound = scenarios.build_scenario(
                    graph, plan, request["catalog"], hardware, geometry,
                    evidence, store, path,
                )
                scenario_id = f"{compiler_id}_{store}_{path}"
                if fetch:
                    scenario_id += "_fetch_" + fetch["id"]
                if bound["status"] != "finite_conditional_scenario":
                    missing.append(dict(candidate=identifier,
                                        scenario=scenario_id, reason=bound))
                    continue
                cache = request["evidence"].get("data_cache")
                if cache:
                    cache = dict(cache)
                    if path in ("l2_273", "l2_284_8"):
                        cache["l2_path"] = path
                    bound = scenarios.bind_data_cache(bound, request["catalog"],
                                                      hardware, cache)
                bindings.setdefault(scenario_id, {})[identifier] = save(
                    record["folder"] / (
                        identifier + "_" + scenario_id + "_binding.json.gz"),
                    bound,
                )
                candidates.setdefault(identifier, dict(
                    source_id=record["id"], levels=record["levels"],
                    locations=record["locations"], placement=record["placement"],
                    geometry=geometry, graph=record["graph"], compiler_plans={},
                ))
                candidates[identifier]["compiler_plans"][compiler_id] = dict(
                    plan=plan_path, addresses=projection_path,
                    active_pc_slots=len(projection["accessed_pc_union"]),
                    reserved_instruction_bytes=projection["span_bytes"],
                    occupancy=bound["occupancy"],
                )
                jobs.append((identifier, scenario_id, plan, bound))
    if not jobs:
        return dict(status="no_legal_scenarios", rejected=rejected,
                    missing=missing)
    attempts = 2 * math.lcm(*(job[3]["occupancy"]["resident_warps_per_sm"]
                              for job in jobs))
    costs, scheduled, cost_links = {}, {}, {}
    for identifier, scenario_id, plan, bound in jobs:
        resident = bound["occupancy"]["resident_warps_per_sm"]
        condition = dict(bound["scenario"])
        condition.pop("id", None)
        cache_key = selection.digest(dict(
            events=plan["typed_plan"]["allocation"]["events"],
            scenario=condition, resident=resident, attempts=attempts,
            block=bound["block_threads"],
        ))
        if cache_key not in scheduled:
            estimate = schedule_repeated(
                plan["typed_plan"], request["catalog"], bound["scenario"],
                resident, attempts, bound["block_threads"],
            )
            scheduled[cache_key] = estimate
            save(output / ("schedule_" + cache_key + ".json.gz"), estimate)
        estimate = scheduled[cache_key]
        if estimate["status"] != "finite_nominal_estimate":
            missing.append(dict(candidate=identifier, scenario=scenario_id,
                                reason=estimate))
            continue
        value = scenarios.fraction(estimate["common_work"]["cycles"])
        costs.setdefault(scenario_id, {})[identifier] = selection.encoded(value)
        schedule_path = output / ("schedule_" + cache_key + ".json.gz")
        cost_links.setdefault(scenario_id, {})[identifier] = dict(
            binding=bindings[scenario_id][identifier],
            schedule=dict(path=str(schedule_path.resolve()),
                          sha256=selection.file_digest(schedule_path)),
            cycles=selection.encoded(value),
            schedule_equivalence=dict(
                key=cache_key,
                candidate_typed_plan_sha256=execution.identity(
                    plan["typed_plan"]),
                representative_typed_plan_sha256=estimate["plan_sha256"],
                rule=("Exact allocation events, complete scenario excluding "
                      "display id, residency, common work and block size"),
            ),
        )
        print(f"{item['algorithm']} {item['inner']} {identifier} "
              f"{scenario_id}: {float(value / attempts):.6f} cycles/warp",
              flush=True)
    all_ids = set(candidates)
    # No missing compiler case is silently assigned another candidate's cost.
    complete = {key: row for key, row in costs.items() if set(row) == all_ids}
    if not complete:
        ranking = dict(status="no_common_complete_scenario_matrix")
    else:
        ranking = selection.minimax_regret(complete)
        ranking["status"] = "conditional_finite_scenario_default"
        regrets = ranking["maximum_relative_regret"]
        best = min(selection.rational(value) for value in regrets.values())
        ranking["minimax_ties"] = sorted(
            key for key, value in regrets.items()
            if selection.rational(value) == best)
    result = dict(
        schema=1, status="finite_design_evaluated", workload=item,
        comparison_identity=common, request_sha256=selection.digest(request),
        source_cap_regime=dict(
            role_caps={name: role.get("cap") for name, role in
                       baseline["workload"]["roles"].items()},
            coverage=source_cap_coverage(baseline),
            evaluation="Selected body counts checked against actual caps",
            cost_term="cycles_at_source_caps[policy, placement, compiler]",
        ),
        source_targets=targets, candidate_design=dict(
            policies=policies, placements=placements, exhaustive=False,
            register_caps=caps, compiler_caps_are_uncertainty=True),
        candidates=candidates, rejected=rejected, missing=missing,
        common_work=dict(kind="synchronized_full_waves",
                         warp_attempts_per_sm=attempts),
        costs=costs, cost_links=cost_links,
        common_complete_scenarios=sorted(complete), ranking=ranking,
        instruction_delivery=dict(
            request["instruction_delivery"],
            hypotheses=fetch_cases,
            candidate_terms={key: f"fetch_cycles[{key}, scenario]"
                             for key in sorted(candidates)},
            scope=("Named instruction-delivery hypotheses included"
                   if any(case and case["mode"] == "hierarchy"
                          for case in fetch_cases) else
                   "Finite ranking has no instruction-delivery service"),
            mathematical_bounds_claimed=False),
        native_labels_consumed=False, solver_timings_consumed=False,
        fitted_parameters=False, native_compilations=0, kernel_launches=0,
    )
    save(output / "result.json", result)
    return result


def run(request, output):
    """Evaluate each workload independently and retain all admitted rankings."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection.validate_hardware(request["hardware"])
    if request["instruction_delivery"]["kind"] != (
            "symbolic_candidate_specific_fetch"):
        raise ValueError("This harness requires explicit unresolved fetch terms")
    save(output / "request.json", request)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    inputs = list(Path(__file__).parent.glob("*.py")) + [
        Path(execution.__file__).with_name("NOMINAL_SERVICE_CATALOG.json")]
    hashes = {}
    for path in inputs:
        (snapshot / path.name).write_bytes(path.read_bytes())
        hashes[str(path.resolve())] = selection.file_digest(path)
    print("SOURCE_EPOCH_SNAPSHOTTED", flush=True)
    previous = get_cache_root_override()
    results, family_costs, family_candidates, family_caps = [], {}, {}, {}
    try:
        for item in request["workloads"]:
            family = item["algorithm"] + "_" + str(item["inner"])
            regimes = request.get("source_regimes", [request["source_regime"]])
            used = set()
            for index, regime in enumerate(regimes):
                effective = dict(regime)
                effective.pop("id", None)
                if item["algorithm"] in ("rk23", "rosenbrock23"):
                    effective["newton_bodies"] = 0
                if item["inner"] in (None, "lu"):
                    effective["krylov_bodies"] = 0
                regime_key = selection.canonical(effective)
                if regime_key in used:
                    continue
                used.add(regime_key)
                name = family + "_" + regime.get("id", f"regime{index}")
                changed = dict(request, source_regime=effective)
                result = evaluate_workload(changed, item, output / name)
                save(output / name / "result.json", result)
                results.append(dict(workload=item, source_regime=effective,
                                    status=result["status"],
                                    ranking=result.get("ranking")))
                if result["status"] != "finite_design_evaluated":
                    continue
                identities = {}
                for key, candidate in result["candidates"].items():
                    action = {field: candidate[field] for field in (
                        "levels", "locations", "placement", "geometry")}
                    identity = selection.digest(action)
                    identities[key] = identity
                    family_candidates.setdefault(family, {})[identity] = action
                for scenario in result["common_complete_scenarios"]:
                    row = result["costs"][scenario]
                    family_costs.setdefault(family, {})[name + ":" + scenario] = {
                        identities[key]: value for key, value in row.items()}
                    if result["source_cap_regime"]["coverage"][
                            "all_active_caps_reached"]:
                        family_caps.setdefault(family, []).append(
                            name + ":" + scenario)
    finally:
        set_cache_root(previous)
    defaults = {}
    for family, costs in family_costs.items():
        complete = {name: row for name, row in costs.items()
                    if set(row) == set(family_candidates[family])}
        defaults[family] = dict(
            status="conditional_enumerated_regime_default",
            candidates=family_candidates[family],
            finite_scenarios=sorted(complete),
            ranking=(selection.minimax_regret(complete) if complete else None),
            source_caps=dict(
                complete_cap_scenarios=[key for key in family_caps.get(
                    family, []) if key in complete],
                interpretation=("Cap endpoints and sampled regimes only; "
                                "other masks and branch regimes remain")),
            instruction_fetch=("Named hypotheses only; no target-calibrated "
                               "complete-kernel ranking"),
        )
    result = dict(status="AUTHOR_RESULT_REQUIRES_INDEPENDENT_REVIEW",
                  workloads=results, source=selection.file_digest(__file__),
                  family_defaults=defaults,
                  source_hashes=hashes,
                  changed_sources=[path for path, digest in hashes.items()
                                   if selection.file_digest(path) != digest])
    save(output / "receipt.json", result)
    return result


def main():
    """Write a reviewable request or evaluate an explicitly supplied request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-request", action="store_true")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--design", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    if args.write_request:
        save(args.output, default_request(args.evidence_root, args.design))
    else:
        run(json.loads(args.request.read_text()), args.output)


if __name__ == "__main__":
    main()
