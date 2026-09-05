"""Model exact-cell shared forwarding with retained observable stores.

This is a separately versioned compiler alternative over a complete ERK
source graph. The original estimator and frozen forecasts are unchanged.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.hardware_model import native_plan as base


SCRIPT = Path(__file__).resolve()
BASE_SHA = "f547ee91e5f3a390d68c8113e8eb438bde03438935ca8d4b294e148fb9480471"
STORAGE = "caller:shared_scratch"


def shared_versions(graph):
    """Prove same-cell versions and retain caller-visible final writes.

    Returns
    -------
    dict
        Exact source read/write witnesses and initialization ranges.

    Notes
    -----
    Unknown calls, unresolved controls and shared live-in reads reject
    this alternative. Caller and step initialization have distinct scope.
    """
    if base.digest(base.SCRIPT) != BASE_SHA:
        raise ValueError("The frozen base estimator changed")
    base.validate_graph(graph)
    construction = base.validate_construction(graph)
    if any(
        call["kind"] not in ("allocator", "source_call")
        for call in graph["calls"]
    ):
        raise ValueError("An opaque call prevents shared forwarding")
    if any(
        control["kind"]
        not in (
            "selected_branch",
            "full_source_expansion",
        )
        for control in graph["controls"]
    ):
        raise ValueError("Unresolved control prevents shared forwarding")
    current = {}
    writes = []
    reads = []
    for node in graph["nodes"]:
        cell = node.get("cell")
        if cell is None or cell[0] != STORAGE:
            continue
        storage, low, high, dtype = cell
        if (
            dtype != "float32"
            or high - low != 4
            or low % 4
            or low < 0
            or high > construction["shared_stride_bytes"]
            or len(node["inputs"]) != 1
        ):
            raise ValueError("Shared access lacks an exact aligned cell")
        key = tuple(cell)
        (value,) = node["inputs"]
        if node["kind"] == "element_write_alias":
            record = dict(node=node["id"], cell=cell, source_value=value)
            current[key] = record
            writes.append(record)
        elif node["kind"] == "element_read_alias":
            if key not in current:
                raise ValueError("A shared live-in read needs retained load")
            previous = current[key]
            if value != previous["source_value"]:
                raise ValueError("Shared read does not match its last write")
            reads.append(
                dict(
                    node=node["id"],
                    cell=cell,
                    source_value=value,
                    reaching_write=previous["node"],
                )
            )
        else:
            raise ValueError("Unsupported shared memory event")
    final = {
        tuple(item["cell"]): item
        for item in graph["final_cells"]
        if item["cell"][0] == STORAGE
    }
    if set(final) != set(current):
        raise ValueError("Shared final-cell membership is incomplete")
    for cell, item in final.items():
        if (
            not item["boundary"]
            or item["value"] != current[cell]["source_value"]
        ):
            raise ValueError("Shared exit version differs from last write")
    initialization = []
    for call in graph["calls"]:
        view = call.get("view", {})
        if (
            call["kind"] != "allocator"
            or not call.get("boundary_binding")
            or view.get("storage") != STORAGE
            or not call.get("zero")
            or not view["bytes"]
        ):
            continue
        if (
            view["dtype"] != "float32"
            or view["itemsize"] != 4
            or view["bytes"] % 4
            or view["offset"] % 4
        ):
            raise ValueError("Unsupported caller initialization shape")
        initialization.append(
            dict(
                source=call["source"],
                allocator_source=call["function_source"],
                byte_offset=view["offset"],
                bytes=view["bytes"],
                scalar_zero_stores=view["bytes"] // 4,
                phase="caller allocation before integration loop",
                execution_multiplicity="once per entered integration call",
                native_width_and_elimination="compiler hypothesis unresolved",
            )
        )
    return dict(
        source_reads=reads,
        source_writes=writes,
        final_writes=list(current.values()),
        outer_initialization=initialization,
        region="one complete expanded step including all known helpers",
        visibility_conditions=[
            "Captured aliases and known helper bodies are exhaustive",
            "Caller supplies the captured private per-run shared slice",
            "No unresolved call, barrier or cross-thread access in region",
            "All accessed shared cells retain their final exit contents",
            "Outer loop calls and initialization are separate phases",
        ],
    )


class ForwardingLowering(base.Lowering):
    """Retain shared versions in SSA and emit one final store per cell."""

    def __init__(self, graph, materialization, contract, store_schedule):
        if store_schedule not in ("early", "late"):
            raise ValueError("Final-store schedule must be early or late")
        self.version_proof = shared_versions(graph)
        self.final_nodes = {
            item["node"] for item in self.version_proof["final_writes"]
        }
        self.final_stores = []
        self.store_schedule = store_schedule
        super().__init__(graph, materialization, contract)

    def memory(self, node):
        """Forward exact shared values without adding false memory edges."""
        if node["cell"][0] != STORAGE:
            return super().memory(node)
        inherited = sorted(
            {
                parent
                for before in node["order_predecessors"]
                for parent in self.source_nodes.get(before, [])
            }
        )
        if node["kind"] == "element_read_alias":
            value = self.mapped(node["inputs"][0])
            self.read_values[node["id"]] = value
            self.rewrites.append(
                dict(
                    rule="forward_exact_shared_version",
                    source_nodes=[node["id"]],
                    cell=node["cell"],
                    source_value=node["inputs"][0],
                    retained_value=value,
                )
            )
        elif node["id"] in self.final_nodes:
            super().memory(node)
            (final_store,) = self.source_nodes[node["id"]]
            self.final_stores.append(final_store)
            self.rewrites.append(
                dict(
                    rule="retain_observable_shared_exit_store",
                    source_nodes=[node["id"]],
                    cell=node["cell"],
                    native_node=final_store,
                    schedule=self.store_schedule,
                )
            )
        else:
            self.rewrites.append(
                dict(
                    rule="eliminate_intermediate_shared_store",
                    source_nodes=[node["id"]],
                    cell=node["cell"],
                    source_value=node["inputs"][0],
                )
            )
        # Forwarded reads depend on the retained value's producer. The
        # emitted final store is an exit effect, not its SSA producer.
        self.source_nodes[node["id"]] = inherited

    def build(self):
        """Build the source-order or region-exit final-store alternative."""
        plan = super().build()
        final_ids = set(self.final_stores)
        if self.store_schedule == "late":
            body = {node["id"] for node in plan["nodes"]} - final_ids
            for node in plan["nodes"]:
                if node["id"] in final_ids:
                    node["predecessors"] = sorted(
                        set(node["predecessors"]) | body
                    )
        plan["shared_forwarding"] = dict(
            proof=self.version_proof,
            final_native_stores=self.final_stores,
            store_schedule=self.store_schedule,
            late_scope="after all retained non-final-store region nodes",
            early_scope="original last source write, preserving its inputs",
        )
        plan["assumptions"] += [
            "Exact shared reads forward their already stored SSA version",
            "Intermediate shared writes are unobservable inside this region",
            "Final caller-visible shared stores remain observable",
            "Final-store schedule is an explicit compiler alternative",
            "No service latency or register overhead is added for forwarding",
        ]
        verify_forwarding(plan, self.graph)
        return plan


def verify_forwarding(plan, graph):
    """Bind every observable shared store to its exact source exit value."""
    proof = plan["shared_forwarding"]["proof"]
    if proof != shared_versions(graph):
        raise ValueError("Forwarding proof differs from the actual source")
    expected = {tuple(item["cell"]): item for item in proof["final_writes"]}
    # The original lowering resolves source constants and same-type cast
    # aliases. Memory materialization and FMA contraction preserve these
    # source-result identities, so the reference needs neither transform.
    reference = base.Lowering(graph, "promote", False).build()
    source_semantics = reference["final_memory"]
    observed = {}
    values = {item["id"]: item for item in plan["values"]}
    nodes = {item["id"]: item for item in plan["nodes"]}
    final_ids = set(plan["shared_forwarding"]["final_native_stores"])
    for node in plan["nodes"]:
        memory = node["memory"]
        if memory is None or memory["space"] != base.SHARED:
            continue
        if node["opcode"] != "STS" or memory["access"] != "write":
            raise ValueError("Unexpected shared access after forwarding")
        cell = tuple(memory["cell"])
        if cell in observed or cell not in expected:
            raise ValueError("Duplicate or unexpected shared exit store")
        item = expected[cell]
        if (
            node["source_nodes"] != [item["node"]]
            or node["id"] not in final_ids
        ):
            raise ValueError("Shared store lost its source-version witness")
        if len(node["inputs"]) != 2 or node["outputs"]:
            raise ValueError("Shared final store has an invalid value shape")
        source_semantic = source_semantics[json.dumps(list(cell))]
        stored = values[node["inputs"][1]]
        if stored["semantic"] != source_semantic:
            raise ValueError("Shared store differs from its source exit value")
        producer = stored.get("producer")
        if producer is not None and (
            producer not in nodes
            or stored["id"] not in nodes[producer]["outputs"]
            or producer not in node["predecessors"]
        ):
            raise ValueError("Shared store lost its value-producer dependency")
        if (
            values[node["inputs"][1]]["semantic"]
            != memory["expected_semantic"]
        ):
            raise ValueError("Shared final store changed the stored value")
        if (
            plan["final_memory"][json.dumps(list(cell))]
            != memory["expected_semantic"]
        ):
            raise ValueError("Shared final-memory contract differs")
        observed[cell] = node["id"]
    if set(observed) != set(expected) or set(observed.values()) != final_ids:
        raise ValueError("Incomplete shared exit-store membership")
    base.native_schedule(plan)
    return dict(
        status="EXACT_SHARED_VERSIONS_AND_EXIT_STORES_PASS",
        source_reads_forwarded=len(proof["source_reads"]),
        source_stores_eliminated=len(proof["source_writes"]) - len(expected),
        retained_final_stores=len(expected),
    )


def predict(
    graph,
    hardware,
    mode="promote",
    contract=False,
    block=64,
    store_schedule="early",
    register_budget=None,
    catalog=None,
):
    """Estimate a complete step under the explicit forwarding alternative."""
    sources = base.validate_graph(graph)
    construction = base.validate_construction(graph)
    flags = construction["jit_kwargs"].get("fastmath", [])
    if contract and "contract" not in flags:
        raise ValueError("Contraction requires the actual contract flag")
    if (
        any(node["kind"] == "Div" for node in graph["nodes"])
        and "arcp" not in flags
    ):
        raise ValueError("Reciprocal lowering requires the actual arcp flag")
    lowering = ForwardingLowering(
        graph, mode, contract, store_schedule
    ).build()
    order = base.native_schedule(lowering)
    unlimited = base.Allocation(
        lowering, order, max(1, len(lowering["values"]))
    ).run()
    base.verify_allocation(lowering, unlimited)
    budget = (
        min(255, max(1, unlimited["peak_words"]))
        if register_budget is None
        else base.exact_int(register_budget, "register_budget", 1)
    )
    if budget > 255:
        raise ValueError("Modeled budget exceeds hardware maximum")
    allocation = base.Allocation(lowering, order, budget).run()
    conservation = base.verify_allocation(lowering, allocation)
    dynamic = max(4, construction["shared_stride_bytes"] * block)
    geometry = base.residency(
        hardware, max(1, allocation["peak_words"]), block, dynamic
    )
    streams = []
    service = None
    if geometry["feasible"]:
        streams = [
            base.sector_stream(
                allocation["trace"],
                allocation["local_frame_bytes"],
                construction["shared_stride_bytes"],
                geometry,
                block,
                backing=backing,
            )
            for backing in ("resident_slots", "trajectory_unique")
        ]
        service = base.service_estimate(allocation["trace"], geometry, catalog)
    return dict(
        schema=1,
        kind="conditional_erk_shared_forwarding_plan",
        candidate=dict(
            materialization=mode,
            contraction=contract,
            actual_placement=construction["placement"],
            block_size=block,
            shared_final_store_schedule=store_schedule,
        ),
        provenance=dict(
            model_source_sha256=base.digest(SCRIPT),
            frozen_base_source_sha256=BASE_SHA,
            sources=sources,
            construction=construction,
        ),
        hardware=hardware,
        lowering=lowering,
        native_schedule=order,
        modeled_no_spill_words=unlimited["peak_words"],
        allocation=allocation,
        conservation=conservation,
        forwarding_conservation=verify_forwarding(lowering, graph),
        geometry=geometry,
        streams=streams,
        service=service,
        source_operation_counts=dict(
            Counter(node["kind"] for node in graph["nodes"])
        ),
        outer_initialization=lowering["shared_forwarding"]["proof"][
            "outer_initialization"
        ],
        outer_initialization_in_step_trace=False,
        status="conditional_complete_step_model"
        if geometry["feasible"]
        else "conditional_geometry_infeasible",
        claim="Compiler alternative; no native register bound or kernel time",
    )


def main():
    """Read frozen source/hardware inputs and write a new forecast artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("promote", "addressable"), default="promote"
    )
    parser.add_argument(
        "--store-schedule", choices=("early", "late"), default="early"
    )
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--register-budget", type=int)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    hardware = base.hardware_model(json.loads(args.hardware.read_text()))
    result = predict(
        graph,
        hardware,
        args.mode,
        args.contract,
        args.block,
        args.store_schedule,
        args.register_budget,
    )
    result["provenance"]["input_graph"] = dict(
        path=str(args.graph.resolve()), sha256=base.digest(args.graph)
    )
    result["provenance"]["hardware_manifest"] = dict(
        path=str(args.hardware.resolve()), sha256=base.digest(args.hardware)
    )
    base.write_json(args.output, result)
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                no_spill_words=result["modeled_no_spill_words"],
                allocated_words=result["allocation"]["peak_words"],
                spill_bytes=result["allocation"]["spill_bytes"],
                **result["forwarding_conservation"],
            )
        )
    )


if __name__ == "__main__":
    main()
