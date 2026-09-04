"""Select among a finite set of conditional hardware-model candidates.

Candidate admission reconstructs the source-only policy plan before using
its allocation.  Missing hardware services remain explicit symbols.
"""

import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from benchmarks.hardware_model import implicit_policy_graph as policy_source


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(value, name="value"):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be rational, not bool")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, (Decimal, Fraction)):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(Decimal(str(value)))
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        )
        and value[1] != 0
    ):
        return Fraction(value[0], value[1])
    raise ValueError(
        f"{name} must be an integer, decimal, or [numerator, denominator]"
    )


def encoded(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def positive_int(value, name, allow_zero=False):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is out of range")
    return value


def round_up(value, quantum):
    return ((value + quantum - 1) // quantum) * quantum


def validate_hardware(hw):
    required = {
        "warp_size",
        "subpartitions",
        "schedulers_per_sm",
        "registers_per_sm",
        "registers_per_subpartition",
        "register_allocation_unit_per_warp",
        "shared_allocation_unit",
        "max_registers_per_thread",
        "max_registers_per_block",
        "max_threads_per_sm",
        "max_threads_per_block",
        "max_blocks_per_sm",
        "max_shared_per_block_optin",
        "shared_driver_reservation",
        "supported_shared_carveouts",
        "multiprocessor_count",
    }
    if hw.get("kind") != "published_hardware_capacities" or not hw.get(
        "identity"
    ):
        raise ValueError("hardware identity/kind is missing")
    if set(required) - set(hw):
        raise ValueError(
            f"hardware fields missing: {sorted(set(required) - set(hw))}"
        )
    for key in required - {"supported_shared_carveouts"}:
        positive_int(
            hw[key], f"hardware.{key}", key == "shared_driver_reservation"
        )
    carveouts = hw["supported_shared_carveouts"]
    if (
        not isinstance(carveouts, list)
        or not carveouts
        or carveouts != sorted(set(carveouts))
    ):
        raise ValueError("shared carveouts must be a sorted unique list")
    for value in carveouts:
        positive_int(value, "shared carveout", True)
    if hw["warp_size"] != 32 or hw["subpartitions"] != hw["schedulers_per_sm"]:
        raise ValueError("unsupported warp/scheduler partition architecture")
    if (
        hw["registers_per_subpartition"] * hw["subpartitions"]
        != hw["registers_per_sm"]
    ):
        raise ValueError("register capacities are inconsistent")
    if not isinstance(hw.get("provenance"), list) or not hw["provenance"]:
        raise ValueError("hardware capacities require provenance")


def residency(
    hw, registers, block_threads, static_shared, dynamic_shared, carveout
):
    validate_hardware(hw)
    registers = positive_int(registers, "registers per thread")
    block_threads = positive_int(block_threads, "block threads")
    static_shared = positive_int(static_shared, "static shared", True)
    dynamic_shared = positive_int(dynamic_shared, "dynamic shared", True)
    if (
        block_threads % hw["warp_size"]
        or block_threads > hw["max_threads_per_block"]
    ):
        return {"legal": False, "reason": "block_threads"}
    if registers > hw["max_registers_per_thread"]:
        return {"legal": False, "reason": "registers_per_thread"}
    if carveout not in hw["supported_shared_carveouts"]:
        return {"legal": False, "reason": "unsupported_carveout"}
    user_shared = static_shared + dynamic_shared
    if user_shared > hw["max_shared_per_block_optin"]:
        return {"legal": False, "reason": "shared_per_block"}
    warps = block_threads // hw["warp_size"]
    registers_per_warp = round_up(
        hw["warp_size"] * registers, hw["register_allocation_unit_per_warp"]
    )
    registers_per_block = registers_per_warp * warps
    assumed_registers_per_cta = registers_per_warp * round_up(
        warps, hw["subpartitions"]
    )
    if assumed_registers_per_cta > hw["max_registers_per_block"]:
        return {"legal": False, "reason": "registers_per_block"}
    register_limit = (
        hw["subpartitions"]
        * (hw["registers_per_subpartition"] // registers_per_warp)
    ) // warps
    thread_limit = hw["max_threads_per_sm"] // block_threads
    allocated_shared = round_up(
        user_shared + hw["shared_driver_reservation"],
        hw["shared_allocation_unit"],
    )
    shared_limit = (
        hw["max_blocks_per_sm"]
        if allocated_shared == 0
        else carveout // allocated_shared
    )
    blocks = min(
        register_limit, thread_limit, shared_limit, hw["max_blocks_per_sm"]
    )
    return {
        "legal": blocks >= 1,
        "reason": None if blocks >= 1 else "no_resident_block",
        "resident_blocks_per_sm": blocks,
        "resident_warps_per_sm": blocks * warps,
        "registers_per_warp_allocated": registers_per_warp,
        "registers_per_block_allocated": registers_per_block,
        "registers_assumed_per_cta_for_launch_limit": assumed_registers_per_cta,
        "shared_per_block_allocated": allocated_shared,
        "limits": {
            "register": register_limit,
            "thread": thread_limit,
            "shared": shared_limit,
            "blocks": hw["max_blocks_per_sm"],
        },
    }


def load_bound(record, label):
    path = Path(record["path"]).resolve()
    if file_digest(path) != record["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return json.loads(path.read_text()), path


def validate_catalog(catalog):
    if (
        catalog.get("schema") != 1
        or catalog.get("status") != "conditional_arithmetic_only_proxy"
    ):
        raise ValueError("unsupported service catalog")
    result = {}
    for opcode, service in catalog.get("instructions", {}).items():
        if service.get("latency_cycles") is None:
            continue
        latency = rational(service["latency_cycles"], f"{opcode}.latency")
        initiation = rational(
            service["initiation_cycles"], f"{opcode}.initiation"
        )
        if latency <= 0 or initiation <= 0:
            raise ValueError("service cycles must be positive")
        if service.get("scope") not in {"sm", "subpartition"}:
            raise ValueError("service scope is unsupported")
        if not service.get("pipeline") or not service.get("provenance"):
            raise ValueError("filled service lacks qualification")
        result[opcode] = {
            "latency": latency,
            "initiation": initiation,
            "scope": service["scope"],
            "pipeline": service["pipeline"],
        }
    return result


def plan_components(plan):
    if plan.get("kind") != "conditional_typed_implicit_native_plan":
        raise ValueError("candidate is not a conditional implicit NativePlan")
    if plan.get("native_labels_consumed") is not False:
        raise ValueError("native labels are forbidden")
    if plan.get("measured_iteration_counts_consumed") is not False:
        raise ValueError("measured iteration counts are forbidden")
    events = plan["allocation"]["events"]
    opcodes = Counter()
    memory = Counter()
    executable = []
    for event in events:
        opcode = event.get("opcode")
        if opcode is None:
            continue
        opcodes[opcode] += 1
        executable.append(event)
        detail = event.get("memory")
        if detail:
            memory[(detail["space"], detail["access"], detail["bytes"])] += 1
        elif event.get("kind") in {"spill", "reload"}:
            memory[
                ("local", "write" if event["kind"] == "spill" else "read", 4)
            ] += 1
    templates = plan["static_hot_templates"]
    return {
        "typed_opcode_counts": dict(sorted(opcodes.items())),
        "memory_operation_counts": [
            {
                "space": key[0],
                "access": key[1],
                "bytes": key[2],
                "count": count,
            }
            for key, count in sorted(memory.items())
        ],
        "static_templates": {
            "count": len(templates),
            "identities": sorted(
                item["hot_template_identity"] for item in templates
            ),
            "native_copy_identity_claimed": False,
        },
        "executed_instances": plan["dynamic_work"]["iterations"],
        "trace_sha256": digest(executable),
        "event_count": len(executable),
        "events": executable,
    }


def workload_projection(graph):
    workload = graph["workload"]
    roles = workload["roles"]
    main = roles.get("main_linear")
    error = roles.get("error_linear")
    if main is None:
        raise ValueError("source workload has no main linear-solver role")
    solver = main.get("actual_linear_correction_type") or main.get(
        "solver_type"
    )
    error_solver = (
        None
        if error is None
        else (
            error.get("actual_linear_correction_type")
            or error.get("solver_type")
        )
    )
    construction = graph.get("candidate_construction", {})
    identity = construction.get("workload_identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Actual candidate workload construction is missing")
    return {
        "family": workload["family"],
        "inner_solver": solver,
        "error_inner_solver": error_solver,
        "n_states": workload["n_states"],
        "stages": workload["stage_count"],
        "actual_workload_identity": identity,
    }


def placement_identity(graph):
    """Return every actual owner-qualified registry placement."""
    result = {}
    for record in graph["registry"]:
        name = record["owner"] + ":" + record["name"]
        if name in result:
            raise ValueError("Registry buffer identity is ambiguous")
        result[name] = record["declared_location"]
    return result


def admit_candidate(bound, candidate):
    """Rebuild the bound policy/placement plan before resource estimation."""
    wrapper, path = load_bound(bound, "candidate plan")
    graph, graph_path = load_bound(bound["source_graph"], "source graph")
    if wrapper.get("kind") != "conditional_implicit_policy_native_plan":
        raise ValueError("An actual policy-specific NativePlan is required")
    policy_source.verify_policy_plan(graph, wrapper)
    verify_source_records(graph)
    if candidate["directive_identity"] != graph["policy"]:
        raise ValueError("Candidate directives differ from the actual graph")
    if candidate["placement_identity"] != placement_identity(graph):
        raise ValueError("Candidate placement differs from the actual graph")
    construction = graph["candidate_construction"]
    if construction["precision"] != "float32":
        raise ValueError("Candidate construction must use FP32")
    stride = positive_int(
        construction["shared_stride_bytes"], "actual shared stride", True
    )
    launch = candidate["launch"]
    block = positive_int(launch["block_threads"], "block threads")
    reservation = positive_int(
        launch.get("reserved_dynamic_shared_bytes", 0),
        "extra shared reservation",
        True,
    )
    if (
        launch["static_shared_bytes"] != 0
        or launch["dynamic_shared_bytes"]
        != max(4, stride * block) + reservation
    ):
        raise ValueError("Launch shared bytes differ from candidate layout")
    return wrapper["typed_plan"], graph, path, graph_path


def verify_source_records(graph):
    """Check operator spelling and mirrored loop records of extracted nodes.

    This checks the trusted extractor's records; it is not an independent
    interpretation of arbitrary Python or a source-equivalence proof.
    """
    operators = {
        "Add",
        "Sub",
        "Mult",
        "Div",
        "BitAnd",
        "BitOr",
        "BitXor",
        "LShift",
        "RShift",
        "USub",
        "UAdd",
        "Not",
        "Invert",
        "And",
        "Or",
    }
    mirrored = (
        "policy_loop_id",
        "line",
        "group",
        "part",
        "lane",
        "codegen_constant",
        "recurrent",
        "directive",
    )
    for node in graph["nodes"]:
        source = node["source"]
        fixed = source.get("loop_indices", [])
        instances = source.get("execution_loop_instances", [])
        if len(fixed) != len(instances) or any(
            any(left.get(key) != right.get(key) for key in mirrored)
            for left, right in zip(fixed, instances)
        ):
            raise ValueError("Node loop instances differ from source records")
        if node["kind"] not in operators:
            continue
        statement = ast.parse(source["syntax"]).body
        if len(statement) != 1:
            raise ValueError("Source operator has ambiguous syntax")
        expression = statement[0]
        if isinstance(expression, ast.Expr):
            expression = expression.value
        if (
            not isinstance(
                expression,
                (
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.BoolOp,
                    ast.AugAssign,
                ),
            )
            or type(expression.op).__name__ != node["kind"]
        ):
            raise ValueError("Extracted operator differs from source syntax")


def full_warp_allocation_supported(graph):
    """Check the scalar allocation model's coherent full-warp boundary."""
    if graph["regime"]["step_entry_mask"] != 0xFFFFFFFF:
        return False

    def coherent(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "entry_mask" and item != 0xFFFFFFFF:
                    return False
                if key == "active_masks" and any(
                    mask != 0xFFFFFFFF for mask in item
                ):
                    return False
                if not coherent(item):
                    return False
        elif isinstance(value, list):
            return all(coherent(item) for item in value)
        return True

    return coherent(graph["scenario_contract"])


def finite_schedule(events, services, resident_warps, schedulers):
    """Schedule one explicitly synchronized resident wave in SM cycles."""
    pcs = [0] * resident_warps
    ready = [{} for _ in range(resident_warps)]
    memory_ready = {}
    scheduler_ready = [Fraction(0)] * schedulers
    pipeline_ready = {}
    finish = Fraction(0)
    while any(pc < len(events) for pc in pcs):
        choices = []
        for warp, pc in enumerate(pcs):
            if pc == len(events):
                continue
            event = events[pc]
            service = services[event["opcode"]]
            partition = warp % schedulers
            start = scheduler_ready[partition]
            for ref in event.get("reads", []) + event.get("writes", []):
                if ref.get("register") is not None:
                    start = max(
                        start,
                        ready[warp].get(
                            (ref["bank"], ref["register"]), Fraction(0)
                        ),
                    )
            detail = event.get("memory")
            if detail:
                owner = (
                    None if detail.get("cross_warp_alias") is True else warp
                )
                key = (
                    owner,
                    detail["space"],
                    detail.get("cell"),
                    detail.get("offset"),
                )
                start = max(start, memory_ready.get(key, Fraction(0)))
            pipe = (
                service["pipeline"]
                if service["scope"] == "sm"
                else f"{partition}:{service['pipeline']}"
            )
            start = max(start, pipeline_ready.get(pipe, Fraction(0)))
            choices.append((start, warp, pipe, service, detail))
        start, warp, pipe, service, detail = min(choices)
        stop = start + service["latency"]
        event = events[pcs[warp]]
        for ref in event.get("writes", []):
            if ref.get("register") is not None:
                ready[warp][(ref["bank"], ref["register"])] = stop
        if detail:
            owner = None if detail.get("cross_warp_alias") is True else warp
            key = (
                owner,
                detail["space"],
                detail.get("cell"),
                detail.get("offset"),
            )
            memory_ready[key] = stop
        partition = warp % schedulers
        scheduler_ready[partition] = start + 1
        pipeline_ready[pipe] = start + service["initiation"]
        finish = max(finish, stop)
        pcs[warp] += 1
    return finish


def comparable_work(wave_cycles, resident_warps, work):
    """Express a resident-wave estimate in a common unit of useful work.

    Whole waves begin and drain together in this approximation.  This is
    explicit wave scheduling, not a claim about CUDA block replacement.
    """
    positive_int(resident_warps, "resident warps")
    if work.get("kind") != "synchronized_full_waves":
        raise ValueError("Work must specify synchronized_full_waves")
    warps = positive_int(
        work.get("warp_attempts_per_sm"), "common warp attempts per SM"
    )
    if warps % resident_warps:
        raise ValueError("Common work must contain complete resident waves")
    waves = warps // resident_warps
    if waves < 2:
        raise ValueError("Common work must contain at least two full waves")
    return rational(wave_cycles, "wave cycles") * waves


def exposed_delivery(delivery, scenario, trace_sha, geometry, services):
    """Admit only additional stall cycles from the same scheduled wave."""
    expected = {
        "trace_sha256": trace_sha,
        "resident_warps_per_sm": geometry["resident_warps_per_sm"],
        "service_catalog_sha256": services,
        "cache_identity_sha256": digest(scenario["cache_identity"]),
    }
    if (
        delivery.get("status") != "qualified"
        or not delivery.get("provenance")
        or delivery.get("kind") != "exposed_instruction_stall_cycles"
        or delivery.get("scope") != "one_synchronized_resident_wave"
        or delivery.get("schedule_binding") != expected
        or delivery.get("service_identity")
        != scenario.get("instruction_delivery_identity")
    ):
        raise ValueError("Delivery must bind exposed stalls to this wave")
    cycles = rational(delivery["cycles"], "exposed instruction stalls")
    if cycles < 0:
        raise ValueError("Exposed instruction stalls are negative")
    return cycles


def minimax_regret(costs):
    """Return exact finite-scenario rankings and the minimax-regret default."""
    if not costs:
        raise ValueError("finite cost matrix is empty")
    candidates = sorted(next(iter(costs.values())))
    if not candidates or any(
        sorted(row) != candidates for row in costs.values()
    ):
        raise ValueError("finite scenarios must contain the same candidates")
    fractions = {
        scenario: {
            candidate: rational(value, "cycle cost")
            for candidate, value in row.items()
        }
        for scenario, row in costs.items()
    }
    if any(value <= 0 for row in fractions.values() for value in row.values()):
        raise ValueError("cycle costs must be positive")
    regrets = {}
    for candidate in candidates:
        regrets[candidate] = max(
            fractions[scenario][candidate] / min(fractions[scenario].values())
            for scenario in fractions
        )
    winner = min(candidates, key=lambda item: (regrets[item], item))
    return {
        "default": winner,
        "maximum_relative_regret": {
            key: encoded(value) for key, value in sorted(regrets.items())
        },
        "scenario_rankings": {
            scenario: sorted(
                candidates, key=lambda item: (fractions[scenario][item], item)
            )
            for scenario in sorted(fractions)
        },
    }


def predict_candidates(request):
    if (
        request.get("schema_version") != 2
        or request.get("kind") != "finite_candidate_selection_request"
    ):
        raise ValueError("unsupported request schema")
    candidates = request.get("candidates", [])
    scenarios = request.get("scenarios", [])
    if len(candidates) < 2 or not scenarios:
        raise ValueError(
            "at least two candidates and one scenario are required"
        )
    identities = [
        (
            candidate["directive_identity"],
            candidate["placement_identity"],
            candidate["launch"],
        )
        for candidate in candidates
    ]
    if len({canonical(item) for item in identities}) != len(identities):
        raise ValueError("Joint candidate identities must be distinct")
    candidate_ids = [candidate["id"] for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    scenario_ids = [scenario.get("id") for scenario in scenarios]
    if any(
        not isinstance(item, str) or not item for item in scenario_ids
    ) or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario ids must be unique and nonempty")
    coverage = request.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("candidate_ids") != candidate_ids
        or coverage.get("scenario_ids") != scenario_ids
        or not isinstance(coverage.get("complete"), bool)
        or not coverage.get("description")
    ):
        raise ValueError("coverage does not match the finite enumeration")
    validate_hardware(request["hardware"])
    work = request.get("work", {})
    if work.get("kind") != "synchronized_full_waves":
        raise ValueError("Candidate comparison needs an explicit common work")
    positive_int(
        work.get("warp_attempts_per_sm"), "common warp attempts per SM"
    )
    catalog, catalog_path = load_bound(
        request["service_catalog"], "service catalog"
    )
    services = validate_catalog(catalog)
    expected_identity = request["comparison_identity"]
    results = []
    finite_costs = {}
    all_finite = True
    for scenario in scenarios:
        if (
            not isinstance(scenario.get("cache_identity"), dict)
            or not scenario["cache_identity"]
        ):
            raise ValueError("cache identity is missing")
        if scenario["comparison_identity"] != expected_identity:
            raise ValueError("scenario comparison identity mismatch")
        if set(scenario["plans"]) != set(candidate_ids):
            raise ValueError("scenario candidate coverage mismatch")
        regime = None
        compiler = None
        scenario_results = []
        finite_costs[scenario["id"]] = {}
        for candidate in candidates:
            bound = scenario["plans"][candidate["id"]]
            plan, graph, plan_path, graph_path = admit_candidate(
                bound, candidate
            )
            if workload_projection(graph) != expected_identity:
                raise ValueError(
                    "plan workload/family/inner-solver identity mismatch"
                )
            components = plan_components(plan)
            plan_regime = digest(plan["dynamic_work"]["iterations"])
            plan_compiler = digest(plan["compiler_alternative"])
            regime = plan_regime if regime is None else regime
            compiler = plan_compiler if compiler is None else compiler
            if plan_regime != regime or plan_compiler != compiler:
                raise ValueError("incoherent iteration/compiler scenario")
            if scenario["iteration_regime_sha256"] != plan_regime:
                raise ValueError("iteration regime identity mismatch")
            if scenario["compiler_identity_sha256"] != plan_compiler:
                raise ValueError("compiler identity mismatch")
            peak = plan["allocation"]["peak_resident"]
            launch = candidate["launch"]
            geometry = residency(
                request["hardware"],
                peak["R"],
                launch["block_threads"],
                launch["static_shared_bytes"],
                launch["dynamic_shared_bytes"],
                scenario["shared_carveout_bytes"],
            )
            missing = sorted(
                set(components["typed_opcode_counts"]) - set(services)
            )
            allocation_supported = full_warp_allocation_supported(graph)
            symbols = (
                (
                    []
                    if allocation_supported
                    else [
                        {
                            "kind": "unsupported_subgroup_allocation",
                            "reason": "per-lane spill and merge conservation unproved",
                        }
                    ]
                )
                + [
                    {"kind": "missing_service", "opcode": opcode}
                    for opcode in missing
                ]
                + [
                    {
                        "kind": "instruction_delivery",
                        "candidate": candidate["id"],
                        "scenario": scenario["id"],
                        "reason": "no qualified static native-byte delivery model",
                    }
                ]
            )
            known = None
            if geometry["legal"] and not missing and allocation_supported:
                known = finite_schedule(
                    components["events"],
                    services,
                    geometry["resident_warps_per_sm"],
                    request["hardware"]["schedulers_per_sm"],
                )
            delivery = scenario.get("instruction_delivery", {}).get(
                candidate["id"]
            )
            delivery_cycles = None
            if delivery is not None:
                if not geometry["legal"]:
                    raise ValueError(
                        "Delivery cannot describe illegal geometry"
                    )
                delivery_cycles = exposed_delivery(
                    delivery,
                    scenario,
                    components["trace_sha256"],
                    geometry,
                    request["service_catalog"]["sha256"],
                )
                symbols = [
                    item
                    for item in symbols
                    if item["kind"] != "instruction_delivery"
                ]
            wave_total = (
                None
                if known is None or delivery_cycles is None
                else known + delivery_cycles
            )
            wave_count = None
            if geometry["legal"]:
                wave_count = comparable_work(
                    1, geometry["resident_warps_per_sm"], work
                )
            total = None if wave_total is None else wave_total * wave_count
            if total is None or not geometry["legal"]:
                all_finite = False
            else:
                finite_costs[scenario["id"]][candidate["id"]] = encoded(total)
            expression = {
                "op": "sum",
                "terms": [
                    {
                        "op": "conditional_service_schedule",
                        "known_execution_cycles": None
                        if known is None
                        else encoded(known),
                        "trace_sha256": components["trace_sha256"],
                        "missing_services": missing,
                    },
                    (
                        {
                            "op": "qualified_instruction_delivery",
                            "cycles": encoded(delivery_cycles),
                        }
                        if delivery_cycles is not None
                        else symbols[-1]
                    ),
                ],
            }
            scenario_results.append(
                {
                    "candidate_id": candidate["id"],
                    "directive_identity": candidate["directive_identity"],
                    "placement_identity": candidate["placement_identity"],
                    "plan": {
                        "path": str(plan_path),
                        "sha256": bound["sha256"],
                    },
                    "source_graph": {
                        "path": str(graph_path),
                        "sha256": bound["source_graph"]["sha256"],
                    },
                    "geometry": geometry,
                    "coherent_full_warp_allocation": allocation_supported,
                    "components": {
                        key: value
                        for key, value in components.items()
                        if key != "events"
                    },
                    "unresolved": symbols,
                    "wave_cycle_expression": expression,
                    "work": work,
                    "full_waves": None
                    if wave_count is None
                    else int(wave_count),
                    "common_work_cycles": None
                    if total is None
                    else encoded(total),
                }
            )
        pairwise = []
        for left_index, left in enumerate(candidate_ids):
            for right in candidate_ids[left_index + 1 :]:
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "break_even": {
                            "op": "eq",
                            "left": {
                                "op": "candidate_cycles",
                                "candidate": left,
                                "scenario": scenario["id"],
                            },
                            "right": {
                                "op": "candidate_cycles",
                                "candidate": right,
                                "scenario": scenario["id"],
                            },
                        },
                    }
                )
        results.append(
            {
                "scenario_id": scenario["id"],
                "cache_identity": scenario["cache_identity"],
                "candidates": scenario_results,
                "pairwise": pairwise,
            }
        )
    selection = minimax_regret(finite_costs) if all_finite else None
    return {
        "schema_version": 2,
        "kind": "conditional_candidate_selection",
        "comparison_identity": expected_identity,
        "coverage": coverage,
        "work": work,
        "service_catalog": {
            "path": str(catalog_path),
            "sha256": request["service_catalog"]["sha256"],
        },
        "scenarios": results,
        "selection": selection,
        "selection_status": (
            "finite_minimax_regret"
            if selection
            else "conditional_symbolic_terms_remain"
        ),
        "native_labels_consumed": False,
        "bank_timings_consumed": False,
        "fitted_parameters_consumed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    request = json.loads(request_path.read_text(), parse_float=Decimal)
    result = predict_candidates(request)
    result["provenance"] = {
        "request_path": str(request_path),
        "request_sha256": file_digest(request_path),
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": file_digest(__file__),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
