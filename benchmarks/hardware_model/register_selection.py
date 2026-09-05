"""Conditional source-bound register selection for whole local arrays."""

from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path

from benchmarks.hardware_model import implicit_native_lowering as native
from benchmarks.hardware_model import implicit_policy_graph as policy
from benchmarks.hardware_model import policy_integer_division as division
from benchmarks.hardware_model import register_selection_source as complete_source


RULE = "all_complete_source_proved_dynamic_local_extents"


def selection_inventory(graph):
    """Apply a complete-source compiler rule equally to every candidate."""
    storages = sorted({item["view"]["storage"] for item in graph["allocations"]})
    result = []
    for storage in storages:
        allocations = [item for item in graph["allocations"]
                       if item["view"]["storage"] == storage]
        views = [item["view"] for item in allocations]
        if any(view["bytes"] is None for view in views):
            continue
        extent = max(view["offset"] + view["bytes"] for view in views)
        if not extent:
            continue
        homes = set(range(0, extent, 4))
        covered = {offset for view in views for offset in range(
            view["offset"], view["offset"] + view["bytes"], 4)}
        proof = None
        reason = None
        if not storage.startswith("local:"):
            reason = "nonlocal placement retains its declared address space"
        elif (covered != homes or any(view["itemsize"] != 4 for view in views)
              or len({view["dtype"] for view in views}) != 1):
            reason = "whole alias extent is not uniformly covered 32-bit homes"
        else:
            roots = [item for item in allocations if item["view"]["offset"] == 0
                     and item["view"]["bytes"] == extent]
            if len(roots) != 1:
                reason = "complete source requires one whole allocation owner"
            else:
                try:
                    proof = complete_source.source_admission(graph, roots[0])
                    if not proof["source_dynamic_accesses"]:
                        reason = "complete source has no runtime-indexed access"
                except (ValueError, StopIteration) as error:
                    reason = str(error) or "whole source owner is outside the region"
        domains = {}
        accesses = [node for node in graph["nodes"]
                    if node.get("cell", [None])[0] == storage
                    and node.get("address_value_ids")]
        if reason is None:
            for node in accesses:
                domains[node["id"]] = address_domain(graph, node)
                if not set(domains[node["id"]]["offsets"]) <= homes:
                    raise ValueError("Typed index domain leaves source-proved extent")
        result.append(dict(storage=storage, eligible=reason is None,
                           reason=reason or RULE, extent_bytes=extent,
                           homes=sorted(homes), complete_source=proof,
                           domains=domains))
    return result


def address_domain(graph, node):
    """Enumerate the complete source induction domain with shared symbols."""
    roots = {}
    expressions = {}

    def expression(identifier):
        if identifier in expressions:
            return expressions[identifier]
        value = graph["values"][identifier]
        if value["kind"] == "constant":
            constant = value["constant"]
            result = ("constant", int(constant["value"]
                                      if isinstance(constant, dict)
                                      else constant))
        elif value.get("source_origin") == "runtime_loop_induction":
            proof = division.range_proof(graph["values"],
                                         graph["policy_loops"], identifier)
            # Repeated occurrences of this SSA value are correlated.
            # Distinct saved induction versions may denote different loop
            # visits, so they are separate conservative range variables.
            key = identifier
            roots[key] = list(range(proof["range_start"],
                                    proof["range_start"] + proof["range_step"]
                                    * proof["range_count"],
                                    proof["range_step"]))
            result = ("symbol", key)
        elif value["producer"] is not None:
            producer = graph["nodes"][value["producer"]]
            kind = producer["kind"]
            if kind not in ("cast", "Add", "Sub", "Mult", "FloorDiv"):
                raise ValueError("Register index lacks finite source form")
            if kind == "FloorDiv":
                division.source_form(graph, producer)
            result = (kind, *(expression(key) for key in producer["inputs"]))
        else:
            raise ValueError("Register index lacks source induction bounds")
        expressions[identifier] = result
        return result

    def evaluate(item, state):
        kind, *operands = item
        if kind == "constant":
            return operands[0]
        if kind == "symbol":
            return state[operands[0]]
        values = [evaluate(operand, state) for operand in operands]
        result = {
            "cast": lambda: values[0],
            "Add": lambda: values[0] + values[1],
            "Sub": lambda: values[0] - values[1],
            "Mult": lambda: values[0] * values[1],
            "FloorDiv": lambda: values[0] // values[1],
        }[kind]()
        if not -(2**31) <= result < 2**31:
            raise ValueError("Register index arithmetic exceeds int32")
        return result

    affine = node["address_affine"]
    terms = [(expression(term["value"]), term["stride_bytes"])
             for term in affine["terms"]]
    keys = sorted(roots)
    addresses = set()
    for values in product(*(roots[key] for key in keys)):
        state = dict(zip(keys, values))
        address = affine["constant_bytes"]
        for item, stride in terms:
            address += evaluate(item, state) * stride
            if not 0 <= address < 2**31:
                raise ValueError("Selection byte-address needs int32 proof")
        addresses.add(address)
    return dict(offsets=sorted(addresses), source_induction_domains=roots,
                address_expression=deepcopy(affine),
                guard_scope="complete source index domain before guard "
                "filtering; original guards remain ordered operations",
                execution_witness_used_as_bound=False)


class RegisterSelectionMixin:
    """Use explicit ISETP/SEL chains with actual current SSA home values."""

    def __init__(self, graph, compiler, materialization="promote",
                 selection=None):
        if not selection or selection.get("form") != "predicate_select_chain":
            raise ValueError("Explicit predicate/SEL compiler form required")
        if selection.get("domain") not in ("whole_extent", "source_domain"):
            raise ValueError("Selection domain must be an explicit choice")
        self.selection = deepcopy(selection)
        self.selection_inventory = selection_inventory(graph)
        eligible = {item["storage"] for item in self.selection_inventory
                    if item["eligible"]}
        if set(selection.get("storages", eligible)) != eligible:
            raise ValueError("Shared compiler rule must select every eligible extent")
        if selection.get("rule", RULE) != RULE:
            raise ValueError("Unknown source-only compiler selection rule")
        self.selection.update(rule=RULE, storages=sorted(eligible))
        self.selection_storages = eligible
        self.register_homes = {}
        self.selection_records = []
        self.selection_contracts = {}
        super().__init__(graph, compiler, materialization)

    def bind_layouts(self):
        """Promote exactly selected local alias extents without a size rule."""
        super().bind_layouts()
        for storage in self.selection_storages:
            layout = self.layouts[storage]
            if layout["space"] != native.base.LOCAL:
                raise ValueError("Register selection is a local-space form")
            views = [item["view"] for item in self.graph["allocations"]
                     if item["view"]["storage"] == storage]
            dtypes = {view["dtype"] for view in views}
            if len(dtypes) != 1 or next(iter(dtypes)) not in (
                    "float32", "int32", "uint32"):
                raise ValueError("Whole alias extent needs one 32-bit type")
            covered = {offset for view in views
                       for offset in range(view["offset"],
                                           view["offset"] + view["bytes"], 4)}
            homes = set(range(0, layout["bytes"], 4))
            if covered != homes or any(view["itemsize"] != 4 for view in views):
                raise ValueError("Storage homes must cover exact whole extent")
            domains = {}
            for node in self.graph["nodes"]:
                if (node.get("cell", [None])[0] != storage
                        or not node.get("address_value_ids")):
                    continue
                domain = address_domain(self.graph, node)
                if not set(domain["offsets"]) <= homes:
                    raise ValueError("Source index domain leaves register homes")
                domains[node["id"]] = domain
            self.selection_contracts[storage] = dict(
                storage=storage, bytes=layout["bytes"], dtype=next(iter(dtypes)),
                homes=sorted(homes), aliases=deepcopy(views), domains=domains)
            self.promoted.add(storage)
        cursor = 0
        for storage, layout in self.layouts.items():
            if storage in self.promoted:
                layout["frame_offset"] = None
            elif layout["space"] == native.base.LOCAL:
                layout["frame_offset"] = cursor
                cursor += layout["bytes"]
        self.named_frame_bytes = native.base.round_up(cursor, 4)
        for item in self.promotion_eligibility:
            if item["storage"] in self.selection_storages:
                item.update(promoted=True, reason="explicit_register_selection")

    def memory(self, node):
        """Retain source store/read identity through each selection chain."""
        storage, low, high, dtype = node["cell"]
        if storage not in self.selection_storages:
            return super().memory(node)
        if high - low != 4 or low % 4:
            raise ValueError("Register selection needs aligned scalar cells")
        homes = self.register_homes.setdefault(storage, {})
        before = dict(homes)
        if not node.get("address_value_ids"):
            if node["kind"] == "element_write_alias":
                homes[low] = self.mapped(node["inputs"][0], node)
            else:
                if low not in homes:
                    raise ValueError("Register home lacks source initialization")
                self.read_values[node["id"]] = homes[low]
            self.source_nodes[node["id"]] = sorted({
                parent for before in node["order_predecessors"]
                for parent in self.source_nodes.get(before, [])})
            self.selection_records.append(dict(
                source_node=node["id"], operation=node["kind"],
                storage=storage, dynamic=False, candidates=[low],
                before_homes=before, after_homes=dict(homes), typed_nodes=[],
                selected_result=(homes[low] if node["kind"] ==
                                 "element_read_alias" else None),
                stored_value=(homes[low] if node["kind"] ==
                              "element_write_alias" else None)))
            self.sync_promoted_homes(storage, dtype)
            return
        contract = self.selection_contracts[storage]
        candidates = (contract["homes"] if self.selection["domain"] == "whole_extent"
                      else contract["domains"][node["id"]]["offsets"])
        if not set(candidates) <= set(homes):
            raise ValueError("Every possible home needs actual prior initialization")
        emitted = []

        def operation(opcode, inputs, output, role, **detail):
            identifier = self.emit(
                opcode, inputs, output, [node["id"]], semantics=dict(
                    source_operation="dynamic_register_selection",
                    role=role, storage=storage, native_form_is_conditional=True,
                    compiler_form="ISETP_and_SEL; not destructive predicated MOV",
                    **detail))
            emitted.append(identifier)
            return output

        affine = node["address_affine"]
        address = self.literal(dict(dtype="int32",
                                    value=affine["constant_bytes"]))
        for term in affine["terms"]:
            index = self.mapped(term["value"], node)
            stride = self.literal(dict(dtype="int32", value=term["stride_bytes"]))
            address = operation("IMAD", [index, stride, address], self.value(
                "int32", "expression", f"mux_address:{node['id']}:{len(emitted)}"),
                "relative_byte_address", logical_operation="a*b+c", wrap_bits=32)
        read = node["kind"] == "element_read_alias"
        value = homes[candidates[0]] if read else self.mapped(node["inputs"][0], node)
        selected = value
        for offset in (candidates[1:] if read else candidates):
            constant = self.literal(dict(dtype="int32", value=offset))
            predicate = operation("ISETP", [address, constant], self.value(
                "bool", "expression", f"mux_predicate:{node['id']}:{offset}"),
                "address_equals_home", relation="Eq", operand_dtype="int32")
            output = self.value(dtype, "expression",
                                f"mux_home:{node['id']}:{offset}")
            selected = operation("SEL", [predicate,
                                         homes[offset] if read else value,
                                         selected if read else homes[offset]],
                                 output, "read_select" if read else "write_select",
                                 home_offset=offset,
                                 logical_operation="predicate ? true : false")
            if not read:
                homes[offset] = selected
        if read:
            output = self.value(dtype, "expression",
                                f"source:{node['inputs'][0]}")
            selected = operation("MOV", [selected], output,
                                 "source_read_result_binding")
            self.read_values[node["id"]] = selected
        self.source_nodes[node["id"]] = emitted
        self.selection_records.append(dict(
            source_node=node["id"], operation=node["kind"], storage=storage,
            dynamic=True,
            candidates=list(candidates), before_homes=before,
            after_homes=dict(homes), typed_nodes=emitted,
            selected_result=selected if read else None,
            stored_value=None if read else value))
        self.sync_promoted_homes(storage, dtype)

    def sync_promoted_homes(self, storage, dtype):
        """Expose the exact SSA homes through the reviewed promoted map."""
        if not hasattr(self, "promoted_cell_values"):
            self.promoted_cell_values = {}
            self.promoted_copy_bindings = []
        for offset, value in self.register_homes[storage].items():
            self.promoted_cell_values[(storage, offset, offset + 4, dtype)] = value

    def finish_source_nodes(self):
        """Expose every selected caller-visible home to fresh allocation."""
        for item in self.graph["final_cells"]:
            cell = item["cell"]
            if item["boundary"] and cell[0] in self.selection_storages:
                self.observables.add(self.register_homes[cell[0]][cell[1]])
        super().finish_source_nodes()

    def build(self):
        """Retain complete home versions, aliases and explicit form choices."""
        result = super().build()
        # The base loop retains witnessed boundary values. Replace its
        # selected-cell entries with the conditional current home versions.
        boundary = {item["value"] for item in self.graph["final_cells"]
                    if item["boundary"]}
        observables = {self.mapped(value) for value in
                       set(self.graph["observable_values"]) - boundary}
        for item in self.graph["final_cells"]:
            if not item["boundary"]:
                continue
            cell = item["cell"]
            if cell[0] in self.selection_storages:
                observables.add(self.register_homes[cell[0]][cell[1]])
            elif cell[0] in self.promoted:
                observables.add(self.mapped(item["value"]))
        result["observable_values"] = sorted(observables)
        result["register_selection"] = dict(
            selection=self.selection, contracts=self.selection_contracts,
            inventory=self.selection_inventory,
            records=self.selection_records, final_homes=self.register_homes,
            source_path=str(Path(__file__).resolve()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_admission_module=dict(
                path=str(Path(complete_source.__file__).resolve()),
                sha256=hashlib.sha256(Path(complete_source.__file__)
                                      .read_bytes()).hexdigest()),
            timing_or_native_resources_consumed=False,
            register_selection_has_no_size_threshold=True)
        return result


class RegisterSelectionLowering(RegisterSelectionMixin,
                                policy.PolicyTypedLowering):
    """Combine source register selection with existing typed producer forms."""


def make_selection_plan(graph, architecture, compiler, selection):
    """Rebuild one explicit selection alternative and allocate it afresh."""
    checked = policy.verify_policy_graph(graph)
    native.validate_source(graph)
    policy.validate_plan_inputs(architecture, compiler)
    lowered = RegisterSelectionLowering(graph, compiler, "promote", selection).build()
    allocation = native.BankAllocation(lowered, architecture["gpr_budget"],
                                        architecture["predicate_budget"]).build()
    verification = native.verify_allocation(lowered, allocation)
    result = dict(kind="conditional_register_selection_plan", graph_check=checked,
                architecture=architecture, compiler_alternative=compiler,
                lowering=lowered, allocation=allocation, verification=verification,
                selection=deepcopy(selection),
                source_graph_sha256=hashlib.sha256(json.dumps(
                    graph, sort_keys=True).encode()).hexdigest())
    # Home offsets and induction IDs are JSON object keys in saved plans.
    # Canonicalize the public artifact before admission or exact rebuild.
    return json.loads(json.dumps(result))


def verify_selection_plan(graph, plan):
    """Reconstruct the full common-rule plan from its source-only inputs."""
    expected = make_selection_plan(graph, plan["architecture"],
                                    plan["compiler_alternative"], plan["selection"])
    if plan != expected:
        raise ValueError("Register-selection plan differs from exact rebuild")
    return dict(status="REGISTER_SELECTION_PLAN_PASS",
                rule=RULE, no_native_or_timing_inputs=True)
