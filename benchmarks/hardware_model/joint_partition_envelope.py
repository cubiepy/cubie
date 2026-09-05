"""Bind per-action physical partition envelopes to source-plan artifacts."""

import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.hardware_model import candidate_selection as selection
from benchmarks.hardware_model.partition_envelope import rank_partition_envelopes


KIND = "independent_legal_partition_envelope"


def validate_partition_request(request):
    """Require the full hardware partition inventory and a separate hint."""
    specification = request.get("partition_selection")
    if specification is None:
        return None
    if specification.get("kind") != KIND:
        raise ValueError("Unknown physical partition selection contract")
    supported = request["hardware"]["supported_shared_carveouts"]
    if (len(request["carveouts"]) != len(supported)
            or set(request["carveouts"]) != set(supported)):
        raise ValueError("Partition envelope requires every supported partition")
    hint = specification["requested_carveout_bytes"]
    if type(hint) is not int or hint not in supported:
        raise ValueError("Requested carveout bytes must be an explicit legal hint")
    return specification


def load_artifact(reference):
    """Load the exact source or plan bytes retained by the constructor."""
    path = Path(reference["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Partition input artifact bytes differ")
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def action_id(source_id, block, hint):
    """Name a requested action independently from achieved capacity."""
    return f"{source_id}_b{block}_h{hint}"


def bind_partition_envelope(result, request, plan_inventory, scenario_compilers):
    """Derive legal sets from allocated artifacts and rank complete costs."""
    specification = validate_partition_request(request)
    if specification is None:
        return result
    hint = specification["requested_carveout_bytes"]
    supported = request["hardware"]["supported_shared_carveouts"]
    actions, legal, source_cache, geometries = {}, {}, {}, {}
    for record in plan_inventory:
        reference = record["graph"]
        key = (reference["path"], reference["sha256"])
        if key not in source_cache:
            graph = load_artifact(reference)
            source_cache[key] = graph["candidate_construction"]
        construction = source_cache[key]
        if construction["precision"] != "float32":
            raise ValueError("Partition capacity requires actual FP32 source")
        wrapper = load_artifact(record["plan"])
        registers = wrapper["typed_plan"]["allocation"]["peak_resident"]["R"]
        stride = construction["shared_stride_bytes"]
        compiler = record["compiler"]
        for block in request["block_threads"]:
            identifier = action_id(record["source_id"], block, hint)
            dynamic_shared = max(4, stride * block)
            by_partition = {
                str(partition): selection.residency(
                    request["hardware"], registers, block, 0,
                    dynamic_shared, partition,
                ) for partition in supported
            }
            admitted = [partition for partition in supported
                        if by_partition[str(partition)]["legal"]]
            legal.setdefault(compiler, {})[identifier] = admitted
            geometries.setdefault(compiler, {})[identifier] = dict(
                registers=registers, shared_stride_bytes=stride,
                dynamic_shared_bytes=dynamic_shared,
                partitions=by_partition, graph=reference, plan=record["plan"],
            )
            action = dict(
                source_id=record["source_id"], levels=record["levels"],
                locations=record["locations"], placement=record["placement"],
                graph=reference,
                geometry=dict(
                    block_threads=block, static_shared=0,
                    requested_carveout_bytes=hint,
                    assumption="Requested preference; achieved partition unknown",
                ),
                compiler_plans={},
            )
            existing = actions.setdefault(identifier, action)
            existing["compiler_plans"][compiler] = dict(
                plan=record["plan"], addresses=record["addresses"],
                legal_partitions=admitted,
            )
    infeasible = {
        identifier: action for identifier, action in actions.items()
        if not any(legal[compiler].get(identifier) for compiler in legal)
    }
    actions = {key: action for key, action in actions.items()
               if key not in infeasible}
    envelope_costs, envelope_legal, missing, diagnostics = {}, {}, {}, {}
    complete = {}
    for scenario, compiler in scenario_compilers.items():
        rows = {identifier: {} for identifier in actions}
        declared = {identifier: legal.get(compiler, {}).get(identifier, [])
                    for identifier in actions}
        for physical_id, value in result["costs"].get(scenario, {}).items():
            physical = result["candidates"][physical_id]
            geometry = physical["geometry"]
            identifier = action_id(physical["source_id"],
                                   geometry["block_threads"], hint)
            if identifier not in actions:
                raise ValueError("A finite physical cell has no legal action")
            partition = str(geometry["carveout"])
            if partition in rows[identifier]:
                raise ValueError("Physical partition cost is duplicated")
            rows[identifier][partition] = value
        envelope_costs[scenario] = rows
        envelope_legal[scenario] = declared
        problems = {}
        for identifier in actions:
            expected = {str(value) for value in declared[identifier]}
            actual = set(rows[identifier])
            if not expected or expected != actual:
                problems[identifier] = dict(
                    legal_partitions=declared[identifier],
                    finite_partitions=sorted(actual, key=int),
                )
        if "_caller_step_only_diagnostic_" in scenario:
            diagnostics[scenario] = rows
        elif problems:
            missing[scenario] = problems
        else:
            complete[scenario] = rows
    ranking = (
        rank_partition_envelopes(
            complete, {key: envelope_legal[key] for key in complete})
        if complete and actions else
        dict(status="no_common_complete_legal_partition_matrix")
    )
    physical = dict(
        candidates=result["candidates"], costs=result["costs"],
        cost_links=result["cost_links"],
        conditional_geometry_ranking=result["ranking"],
        interpretation="Fixed achieved-capacity diagnostic cells only",
    )
    physical_map = {
        key: dict(action=action_id(value["source_id"],
                                   value["geometry"]["block_threads"], hint),
                  partition_bytes=value["geometry"]["carveout"])
        for key, value in result["candidates"].items()
    }
    return dict(
        result, schema=2, candidates=actions, costs=envelope_costs,
        legal_partitions=envelope_legal,
        cost_links=dict(kind="physical_geometry_cost_links",
                        action_partition_mapping=physical_map,
                        data=physical["cost_links"]),
        ranking=ranking, common_complete_scenarios=sorted(complete),
        diagnostic_step_only_costs=diagnostics,
        physical_geometry=physical,
        instruction_delivery=dict(
            result["instruction_delivery"], candidate_terms={
                key: f"fetch_cycles[{key}, scenario, achieved_partition]"
                for key in actions}),
        partition_selection=dict(
            specification, legal_geometry_proofs=geometries,
            infeasible_actions=infeasible, incomplete_scenarios=missing,
            source_plan_inventory=plan_inventory,
            scenario_compilers=scenario_compilers,
            common_work=result["common_work"],
            coupling="Independent per-action legal partitions; unknown driver coupling",
            empirical_partition_fit=False,
        ),
    )
