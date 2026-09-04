"""Forecast covered implicit-body instruction bytes before compilation.

Forecasts are compiler alternatives over typed operations. They are neither
complete kernel sizes nor instruction-cache working-set measurements.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path

from benchmarks.hardware_model import implicit_native_lowering as native
from benchmarks.hardware_model import implicit_policy_graph as policy


WIDTH_BYTES = 16
WIDTH_SOURCE = (
    "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html"
    "#json-format"
)
ONE_SLOT_OPCODES = frozenset({
    "ACTIVEMASK", "BRA", "FADD", "FFMA", "FMUL", "FSETP", "IADD3",
    "IMAD", "ISETP", "LDC", "LDL", "LDS", "LOP3", "MOV", "MUFU.RCP",
    "MUFU.SQRT", "MUFU.EX2", "MUFU.LG2", "PLOP3", "SEL", "STL", "STS",
    "VOTE",
})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Footprint:
    """Count typed slots under explicit replication and sharing choices."""

    def __init__(self, graph, plan, sharing, false_lowering, project_caps):
        self.graph = graph
        self.plan = plan
        self.sharing = sharing
        self.false_lowering = false_lowering
        self.project_caps = project_caps
        self.calls = {item["context"]: item for item in graph["calls"]
                      if item.get("kind") == "source_call"}
        self.loops = {item["policy_loop_id"]: item
                      for item in graph["policy_loops"]}
        self.paths = {}

    def call_path(self, context):
        if context in self.paths:
            return self.paths[context]
        call = self.calls[context]
        site = call["call_site"]
        path = (() if site is None else self.call_path(site["context"]) + (
            (site["line"], site["syntax"]),))
        self.paths[context] = path
        return path

    def loop_slots(self, record):
        control = self.loops[record["policy_loop_id"]]
        structure = control["structure"]
        expand = (self.project_caps and record["recurrent"])
        false_full = (structure["mode"] == "backend_choice"
                      and self.false_lowering == "full")
        if expand:
            instances = structure["execution_instances"]
        else:
            instances = [record]
        if false_full:
            slots = [("backend_full", item.get(
                "execution_index", item.get("value"))) for item in instances]
        else:
            slots = [(item["part"], item["lane"]) for item in instances]
        return sorted(set(slots))

    def operation_keys(self, node):
        contexts = node["source_contexts"]
        if not contexts:
            raise ValueError("Typed operation has no source context")
        context = contexts[0]
        records = context.get("execution_loop_instances", [])
        if any(item.get("execution_loop_instances", []) != records
               for item in contexts):
            raise ValueError("Cross-instance contraction needs a region model")
        template = context["hot_template"]["function"]
        specialization = digest(template)
        inline = self.sharing == "inline"
        path = self.call_path(context["context"]) if inline else ()
        source_sites = tuple((item["line"], item["syntax"])
                             for item in contexts)
        base = (specialization, path, source_sites, node["opcode"])
        kept = [record for record in records if inline or self.loops[
            record["policy_loop_id"]]["source"]["context"]
            == context["context"]]
        dimensions = []
        for record in kept:
            control = self.loops[record["policy_loop_id"]]
            owner_path = (self.call_path(control["source"]["context"])
                          if inline else ())
            owner = (control["source"]["path"],
                     control["source"]["line"], owner_path)
            dimensions.append([(owner, slot)
                               for slot in self.loop_slots(record)])
        execution = (context["context"], tuple(
            (item["policy_loop_id"], item["execution_index"])
            for item in records))
        return base, dimensions, execution

    def count(self, nodes=None):
        # Repeated visits to one static lane do not create more instructions.
        # If visits lower differently, the maximal observed count is a named
        # envelope hypothesis, not a proof of the compiler's chosen body.
        counts = defaultdict(Counter)
        variants = defaultdict(set)
        if nodes is None:
            nodes = self.plan["lowering"]["nodes"]
        for node in nodes:
            base, dimensions, execution = self.operation_keys(node)
            for lanes in product(*dimensions):
                counts[(base, lanes)][execution] += 1
        opcodes = Counter()
        for (base, _), instances in counts.items():
            opcodes[base[-1]] += max(instances.values())
            variants[base[-1]].update(instances.values())
        known = sum(count for opcode, count in opcodes.items()
                    if opcode in ONE_SLOT_OPCODES)
        return {
            "mapped_instruction_slots": known,
            "mapped_instruction_bytes": WIDTH_BYTES * known,
            "typed_operation_counts": dict(sorted(opcodes.items())),
            "unmapped_operations": {opcode: count
                                    for opcode, count in sorted(opcodes.items())
                                    if opcode not in ONE_SLOT_OPCODES},
            "static_operation_sites": len(counts),
            "per_visit_multiplicities": {key: sorted(value)
                                         for key, value in variants.items()},
        }

    def supplementary_forms(self):
        """Count canonical loop and 32-bit address lowering alternatives."""
        controls = {name: [] for name in (
            "initialization", "induction", "header_predicate", "backedge",
            "constant_tail_control")}
        control_opcodes = {"initialization": "MOV", "induction": "IADD3",
                           "header_predicate": "ISETP", "backedge": "BRA"}
        included_controls = {
            (node["semantics"]["policy_loop_id"],
             node["semantics"].get("induction_category",
                                   node["semantics"].get("control_category")))
            for node in self.plan["lowering"]["nodes"]
            if node.get("semantics", {}).get("source_operation")
            in ("runtime_loop_induction", "runtime_fixed_loop_control")
        }
        for control in self.graph["policy_loops"]:
            structure = control["structure"]
            if structure["mode"] == "backend_choice":
                repetitions = (structure["fixed_trip_count"]
                               if self.false_lowering == "rolled" else 1)
            else:
                repetitions = max((part["dynamic_repetitions"]
                                   for part in structure["source_templates"]),
                                  default=0)
            if repetitions <= 1:
                continue
            for name, opcode in control_opcodes.items():
                if (control["policy_loop_id"], name) in included_controls:
                    continue
                controls[name].append({
                    "opcode": opcode, "source_contexts": [control["source"]]})
        address_nodes = []
        lookup_nodes = []
        unresolved_addresses = []
        included_addresses = {
            identifier for node in self.plan["lowering"]["nodes"]
            if node.get("semantics", {}).get("source_operation")
            == "dynamic_byte_address"
            for identifier in node["source_nodes"]
        }
        included_lookups = {
            identifier for node in self.plan["lowering"]["nodes"]
            if (node.get("memory") or {}).get("kind") == "immutable_constant"
            for identifier in node["source_nodes"]
        }
        for node in self.graph["nodes"]:
            addresses = node.get("address_value_ids", [])
            if addresses and node["id"] not in included_addresses:
                if len(addresses) != 1 or node.get("cell", [None])[-1] != "float32":
                    unresolved_addresses.append(node["id"])
                else:
                    address_nodes.append({
                        "opcode": "IMAD", "source_contexts": [node["source"]]})
            if node["kind"] != "CapturedIndexRead":
                continue
            if node["id"] in included_lookups:
                continue
            # Row-major byte offset: accumulate each dynamic coordinate times
            # its compile-time byte stride with one integer multiply-add.
            terms = node["index_template"]
            if (not isinstance(terms, list)
                    or any(not isinstance(term, (dict, int)) for term in terms)):
                unresolved_addresses.append(node["id"])
                continue
            if any(isinstance(term, dict)
                   and set(term) not in ({"literal"}, {"dynamic_values"})
                   for term in terms):
                unresolved_addresses.append(node["id"])
                continue
            dynamic = sum(isinstance(term, dict)
                          and "dynamic_values" in term for term in terms)
            lookup_nodes.extend({"opcode": "IMAD",
                                 "source_contexts": [node["source"]]}
                                for _ in range(dynamic))
            lookup_nodes.append({"opcode": "LDC",
                                 "source_contexts": [node["source"]]})
        control_counts = {name: self.count(nodes)
                          for name, nodes in controls.items()}
        return {
            "loop_control": control_counts,
            "loop_control_bytes": sum(item["mapped_instruction_bytes"]
                                      for item in control_counts.values()),
            "loop_forms_already_in_typed_body": [
                dict(policy_loop_id=identifier, category=category)
                for identifier, category in sorted(included_controls)
            ],
            "dynamic_cell_address_IMAD": self.count(address_nodes),
            "address_source_nodes_already_in_typed_body": sorted(
                included_addresses),
            "lookup_source_nodes_already_in_typed_body": sorted(
                included_lookups),
            "captured_constant_memory_IMAD_LDC": self.count(lookup_nodes),
            "unresolved_address_source_nodes": unresolved_addresses,
            "forms_provenance": WIDTH_SOURCE,
            "assumptions": [
                "Positive constant trip count uses one initialized do-while loop",
                "Loop increment IADD3, predicate ISETP, backedge BRA survive",
                "Constant tail requires no additional branch or loop control",
                "Dynamic FP32 cell byte address uses one 32-bit IMAD",
                "Captured row-major lookup uses one IMAD per dynamic axis and LDC",
                "Integer address forms are not a proof of scalarization or allocation",
            ],
        }


def forecast(graph, wrapper):
    """Return finite covered-body byte forecasts and explicit omissions."""
    if wrapper.get("kind") not in (
            "conditional_implicit_policy_native_plan",
            "conditional_implicit_policy_typed_body",
            "conditional_explicit_policy_native_plan",
            "conditional_explicit_policy_typed_body"):
        raise ValueError("A policy-bound typed plan is required")
    if wrapper.get("policy") != graph.get("policy"):
        raise ValueError("Graph and typed-plan policy differ")
    if (wrapper.get("native_labels_consumed") is not False
            or wrapper.get("timings_consumed") is not False
            or wrapper.get("fitted_parameters") is not False):
        raise ValueError("The forecast must precede native label inspection")
    typed = wrapper["typed_plan"]
    if (typed.get("native_labels_consumed") is not False
            or typed.get("measured_iteration_counts_consumed") is not False):
        raise ValueError("Typed plan consumed native or measured iteration labels")
    scenarios = []
    for sharing, false_lowering in product(
            ("inline", "identical_specialization_shared"), ("rolled", "full")):
        selected = Footprint(graph, typed, sharing, false_lowering, False)
        projected = Footprint(graph, typed, sharing, false_lowering, True)
        scenarios.append({
            "helper_lowering": sharing,
            "false_directive_lowering": false_lowering,
            "false_full_scope": (
                "static replication sensitivity retains typed induction, "
                "control, address and lookup forms; a distinct native "
                "full-unroll candidate requires its own fresh lowering"
                if false_lowering == "full" else None
            ),
            "covered_selected_templates": selected.count(),
            "homogeneous_recurrent_cap_projection": projected.count(),
            "selected_supplementary_forms": selected.supplementary_forms(),
            "cap_supplementary_forms": projected.supplementary_forms(),
        })
    recurrent = [item for item in graph["policy_loops"]
                 if item["kind"] == "recurrent_execution_trace"]
    return {
        "schema": 1,
        "kind": "conditional_covered_instruction_footprint",
        "policy": graph["policy"],
        "workload_identity": graph.get("candidate_construction", {}).get(
            "workload_identity"),
        "native_instruction_bytes": WIDTH_BYTES,
        "native_instruction_width_provenance": WIDTH_SOURCE,
        "opcode_lowering_hypothesis": {
            "name": "retained_typed_opcode_one_SM89_instruction",
            "one_slot_opcodes": sorted(ONE_SLOT_OPCODES),
            "calibration": "none; conditional typed-native form hypothesis",
        },
        "scenarios": scenarios,
        "coverage": {
            "source_nodes": len(graph["nodes"]),
            "typed_trace_operations": len(typed["lowering"]["nodes"]),
            "allocation_trace_instruction_slots": sum(
                event.get("opcode") is not None
                for event in typed["allocation"]["events"])
            if "allocation" in typed else None,
            "allocation_constructed": "allocation" in typed,
            "allocation_trace_is_not_static_code": True,
            "source_alias_operations": sum(
                node["kind"] in ("element_read_alias", "element_write_alias")
                for node in graph["nodes"]),
            "recurrent_loops": [{
                "group": item["group"],
                "source": {key: item["source"][key]
                           for key in ("path", "line", "context")},
                "source_cap": item["structure"]["fixed_trip_count"],
                "selected_loop_visits": len(item["executed_instances"]),
            } for item in recurrent],
            "runtime_path_choices": graph.get("branch_choices", {}),
            "complete_kernel": False,
            "temporal_working_set": False,
            "cache_threshold_decision_admitted": False,
        },
        "projection_assumptions": [
            "Repeated dynamic visits share their canonical static lane",
            "Each static site uses its maximum observed per-visit opcode count",
            "Every recurrent static lane repeats the visited typed body forms",
            "Parameter specialization and visited branch forms persist at cap",
            "Shared helpers reuse identical captured parameter specialization",
        ],
        "unresolved_additions": [
            "Unvisited branch arms and unvisited recurrent-specific code",
            "Compiler changes beyond explicitly typed loop/control forms",
            "Compiler changes beyond typed address and captured-table forms",
            "Literal, ABI, spill/reload, call/return and reconvergence code",
            "Integrator/controller/caller outside the extracted algorithm step",
            "Native instruction elimination, duplication, alignment and padding",
        ],
        "native_labels_consumed": False,
        "timings_consumed": False,
        "fitted_parameters": False,
    }


def validate_body_compiler(compiler):
    """Validate a typed compiler hypothesis without a bank budget."""
    fixed = {
        "division": "approximate_reciprocal_multiply",
        "sqrt": "approximate_native_no_refinement",
        "numeric_literals": "materialized_gpr",
        "predicate_literals": "PT_or_inverted_PT",
        "integer_dynamic_width_bits": 32,
        "predicate_spills": "canonical_uint32_local",
        "schedule": "source_order",
    }
    required = set(fixed) | {
        "name", "provenance", "fp32_flush_subnormals", "fp32_contract"}
    if (set(compiler) != required or not compiler["name"]
            or not isinstance(compiler["provenance"], list)
            or not compiler["provenance"]
            or any(compiler[key] != value for key, value in fixed.items())
            or type(compiler["fp32_flush_subnormals"]) is not bool
            or type(compiler["fp32_contract"]) is not bool):
        raise ValueError("Compiler alternative is incomplete or unsupported")
    for record in compiler["provenance"]:
        if "path" in record and file_digest(record["path"]) != record["sha256"]:
            raise ValueError("Compiler-alternative source bytes changed")


def construct_typed_body(graph, compiler, materialization="promote",
                         shared_forwarding=False):
    """Lower source operations without solving a register allocation."""
    source_check = policy.verify_policy_graph(graph)
    validate_body_compiler(compiler)
    if type(shared_forwarding) is not bool:
        raise ValueError("Shared forwarding alternative must be bool")
    dynamic = any(node["kind"] in policy.MATH_OPERATIONS
                  or node["kind"] == "CapturedIndexRead"
                  or node.get("address_value_ids") for node in graph["nodes"])
    dynamic = dynamic or any(
        value.get("source_origin") == "runtime_loop_induction"
        for value in graph["values"]
    )
    lowerer = policy.PolicyTypedLowering if dynamic else native.TypedLowering
    if shared_forwarding:
        lowerer = policy.ForwardingPolicyLowering
    lowered = lowerer(graph, compiler, materialization).build()
    if not dynamic and not shared_forwarding:
        form_check = native.verify_typed_lowering(graph, lowered, compiler)
    else:
        form_check = {"status": "dynamic_source_forms_remain_conditional"}
    for node in lowered["nodes"]:
        if (node.get("memory") or {}).get("kind") == "immutable_constant":
            policy.verify_constant_load(lowered, node)
    materializations = dict(
        local=materialization,
        shared=(policy.SHARED_FORWARDING_ALTERNATIVE if shared_forwarding
                else "retained_loads_stores"),
    )
    return {
        "schema": 1,
        "kind": ("conditional_explicit_policy_typed_body"
                 if graph["workload"]["family"] == "ERK"
                 else "conditional_implicit_policy_typed_body"),
        "policy": graph["policy"],
        "compiler_materialization": materializations,
        "typed_plan": {
            "lowering": lowered,
            "compiler_alternative": compiler,
            "compiler_materialization": materializations,
            "native_labels_consumed": False,
            "measured_iteration_counts_consumed": False,
        },
        "verification": {"source_status": source_check["status"],
                         "typed_forms": form_check},
        "allocation_constructed": False,
        "provenance": {
            "graph_sha256": digest(graph),
            "constructor": {"path": str(Path(__file__).resolve()),
                            "sha256": file_digest(__file__)},
            "lowerer": {"path": str(Path(native.__file__).resolve()),
                        "sha256": file_digest(native.__file__)},
            "policy_lowerer": {"path": str(Path(policy.__file__).resolve()),
                               "sha256": file_digest(policy.__file__)},
            "address_lowerer": policy.source_receipt(
                policy.PolicyAddressLowering) if dynamic else None,
            "induction_lowerer": policy.source_receipt(
                policy.PolicyInductionLowering) if dynamic else None,
            "loop_control_lowerer": policy.source_receipt(
                policy.PolicyLoopControlLowering) if dynamic else None,
            "captured_lookup_lowerer": policy.source_receipt(
                policy.CapturedLookupLowering) if dynamic else None,
            "shared_forwarding_lowerer": policy.source_receipt(
                policy.SharedReadForwarding) if shared_forwarding else None,
        },
        "native_labels_consumed": False,
        "timings_consumed": False,
        "fitted_parameters": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = forecast(json.loads(args.graph.read_text()),
                      json.loads(args.plan.read_text()))
    result["provenance"] = {
        key: {"path": str(path.resolve()), "sha256": file_digest(path)}
        for key, path in (("graph", args.graph), ("plan", args.plan),
                          ("forecast", Path(__file__)))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
