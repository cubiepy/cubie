"""Source-derived explicit Runge--Kutta workload and FSAL path adapter.

FSAL state selects an executed path. It never specializes caller inputs
or replaces a runtime vote by a compiler constant.
"""

import ast
import inspect

from cubie._utils import package_source_hash
from cubie.cubie_cache import toolchain_fingerprint

from benchmarks.hardware_model import source_value_graph as source
from benchmarks.hardware_model.buffer_descriptors import registry_layout
from benchmarks.hardware_model.expansion import (
    describe_expansion,
    CapturedGraph,
    source_receipt,
)


def describe_explicit_workload(solver):
    """Describe the actual ERK step without a fictitious solver role."""
    expansion = describe_expansion(solver)
    step = solver.kernel.single_integrator._algo_step
    if type(step).__name__ != "ERKStep":
        raise ValueError("Explicit adapter requires an actual ERKStep")
    config = step.compile_settings
    closure = inspect.getclosurevars(step.step_function.py_func).nonlocals
    root = expansion["root"]
    functions = {item["id"]: item for item in expansion["functions"]}
    captured = CapturedGraph()
    captured.add_function(step.step_function, "algorithm_step")
    rhs = source.python_function(closure["evaluate_f"])
    return dict(
        schema=1, kind="actual_explicit_workload", family="ERK",
        inner_solver=None, n_states=int(config.n),
        stage_count=int(config.tableau.stage_count), root=root,
        roles={}, step_calls=[], smoothing_enabled=False,
        actual_step_settings=dict(
            tableau=type(config.tableau).__name__,
            stage_source=functions[root]["source"],
            stage_zero_cache_enabled=bool(
                closure["first_same_as_last"] and closure["multistage"]
            ),
            rhs_function=next(
                key for key, function in captured.callables.items()
                if function is rhs
            ),
        ),
        registry=registry_layout(step), candidate=expansion["candidate"],
        functions=expansion["functions"],
        source_inventory=expansion["source_inventory_operations"],
        compilation_check=expansion["compilation_check"],
        provenance=dict(
            adapter=source_receipt(describe_explicit_workload),
            expansion=expansion["provenance"],
            package_source_hash=package_source_hash(),
            toolchain_fingerprint=toolchain_fingerprint(),
        ),
        assumptions=[
            "FSAL runtime state is an explicit step path assumption",
            "ERK contains no Newton, Krylov or direct solver role",
            "Source code replication differs from executed RHS visits",
        ],
        limitations=[
            "Native optimization and omitted paths remain conditional",
        ],
    )


def fsal_contract(descriptor, state):
    """Bind two runtime state facts to the actual tableau cache rule."""
    if (not isinstance(state, dict)
            or set(state) != {"first_step", "all_lanes_accepted"}
            or any(type(value) is not bool for value in state.values())):
        raise ValueError("ERK requires two explicit Boolean FSAL state facts")
    enabled = descriptor["actual_step_settings"]["stage_zero_cache_enabled"]
    cached = enabled and not state["first_step"] and state[
        "all_lanes_accepted"
    ]
    return dict(
        **state, cache_enabled=enabled, use_cached_rhs=cached,
        rhs_calls=descriptor["stage_count"] - int(cached),
        active_entry_mask=2**32 - 1,
        state_is_runtime_assumption=True,
    )


class ExplicitFsalValues:
    """Retain ERK step-entry branches and a separately scoped FSAL vote."""

    def configure_fsal(self, contract):
        self.fsal = contract
        self.dynamic_slice_proofs = []

    def index_affine(self, identity):
        """Prove a typed integer affine form and its full finite domain."""
        value = self.values[identity]
        if value["dtype"] not in ("int32", "literal_int"):
            raise ValueError("Slice affine form requires signed integer32")
        if value["kind"] == "constant":
            raw = value["constant"]
            number = raw["value"] if isinstance(raw, dict) else raw
            return {}, int(number), (int(number), int(number))
        if value.get("source_origin") == "runtime_loop_induction":
            control = self.policy_loop_controls[value["policy_loop_id"]]
            domain = [item["value"] for item in
                      control["structure"]["execution_instances"]]
            bounds = (min(domain), max(domain))
            return {identity: 1}, 0, bounds
        node = self.nodes[value["producer"]]
        arguments = [self.index_affine(key) for key in node["inputs"]]
        if node["kind"] == "cast" and len(arguments) == 1:
            result = arguments[0]
        elif node["kind"] in ("Add", "Sub") and len(arguments) == 2:
            left, right = arguments
            sign = 1 if node["kind"] == "Add" else -1
            terms = dict(left[0])
            for key, coefficient in right[0].items():
                terms[key] = terms.get(key, 0) + sign * coefficient
            terms = {key: val for key, val in terms.items() if val}
            bounds = (left[2][0] + right[2][0],
                      left[2][1] + right[2][1]) if sign == 1 else (
                          left[2][0] - right[2][1],
                          left[2][1] - right[2][0])
            result = terms, left[1] + sign * right[1], bounds
        elif node["kind"] == "Mult" and len(arguments) == 2:
            left, right = arguments
            if left[0] and right[0]:
                raise ValueError("Slice extent is not affine")
            fixed, variable = (left, right) if not left[0] else (right, left)
            scalar = fixed[1]
            bounds = sorted(scalar * edge for edge in variable[2])
            result = ({key: scalar * coefficient
                       for key, coefficient in variable[0].items()},
                      scalar * variable[1], tuple(bounds))
        else:
            raise ValueError("Slice extent has an unsupported integer form")
        if not -(2**31) <= result[2][0] <= result[2][1] < 2**31:
            raise ValueError("Slice affine arithmetic may overflow int32")
        return result

    def slice_view(self, reference, index, node):
        if (not isinstance(index, slice)
                or not hasattr(index.stop, "identities")):
            return super().slice_view(reference, index, node)
        if (not hasattr(index.start, "identities")
                or len(index.start.identities) != 1
                or len(index.stop.identities) != 1
                or index.step not in (None, 1)):
            self.unknown(node, "Dynamic slice lacks fixed unit-stride extent")
        lower = self.index_affine(index.start.identities[0])
        upper = self.index_affine(index.stop.identities[0])
        extent = upper[1] - lower[1]
        shape = reference.get("shape")
        if (lower[0] != upper[0] or extent <= 0 or not shape
                or lower[2][0] < 0 or upper[2][1] > shape[0]):
            self.unknown(node, "Dynamic slice is not wholly bounded affine")
        result = super().slice_view(reference, slice(
            index.start, int(index.stop), index.step
        ), node)
        result["fixed_extent_proof"] = dict(
            lower_value=index.start.identities[0],
            upper_value=index.stop.identities[0], extent=extent,
            lower_bounds=list(lower[2]), upper_bounds=list(upper[2]),
            source_shape=shape, signed_integer_bits=32,
            proof="equal affine terms over complete declared loop domain",
        )
        self.dynamic_slice_proofs.append(dict(
            source=self.location(node), **result["fixed_extent_proof"]
        ))
        if result["shape"][0] != extent:
            self.unknown(node, "Dynamic slice shape differs from proof")
        return result

    def condition(self, node, environment):
        spelling = ast.unparse(node)
        if (self.function_ids[self.stack[-1]] == self.descriptor["root"]
                and spelling in (
                    "not first_step_flag",
                    "not multistage or not use_cached_rhs",
                ) and self.condition_constant(node, environment)
                is source.UNKNOWN):
            choice = (not self.fsal["first_step"]
                      if spelling == "not first_step_flag"
                      else not self.fsal["use_cached_rhs"])
            predicate = self.scalar(self.expression(node, environment), node)
            self.decisions[predicate["identity"]] = choice
            self.controls.append(dict(
                kind="runtime_branch_choice", source=self.location(node),
                predicate=predicate["identity"], choice=choice,
                reason="declared ERK FSAL runtime state",
                is_codegen_constant=False,
            ))
            _, self.last_branch_event = self.operation(
                "BranchDecision", [predicate], node, selected_path=choice,
                decision_reason="declared ERK FSAL runtime state",
                is_codegen_constant=False,
            )
            return choice
        return super().condition(node, environment)

    def call(self, node, environment):
        target = self.expression(node.func, environment)["raw"]
        function = source.python_function(target)
        if (self.primitives.get(id(function)) != "AllSync" or self.frames
                or self.function_ids[self.stack[-1]]
                != self.descriptor["root"]):
            return super().call(node, environment)
        if (node.keywords or len(node.args) != 2
                or ast.unparse(node.args[1]) != "accepted_flag != int32(0)"
                or self.fsal["first_step"]
                or not self.fsal["cache_enabled"]):
            self.unknown(node, "Unexpected ERK FSAL collective source")
        arguments = [self.scalar(self.expression(arg, environment), node)
                     for arg in node.args]
        entry = self.regime["step_entry_mask"]
        if ([self.values[arg["identity"]]["dtype"] for arg in arguments]
                != ["uint32", "bool"]
                or self.values[arguments[0]["identity"]].get(
                    "declared_lane_mask") != entry):
            self.unknown(node, "ERK FSAL vote mask or predicate type differs")
        result = self.primitive(
            "AllSync", arguments, node, "bool",
            participating_mask=entry, active_entry_mask=entry,
            mask_equality="explicit uniform participating-path contract",
            primitive_source=source_receipt(function),
            collective_scope="ERK_step_FSAL",
            declared_result=self.fsal["all_lanes_accepted"],
        )
        self.decisions[result["identity"]] = self.fsal["all_lanes_accepted"]
        return result


def verify_explicit_path(graph):
    """Check actual RHS visits and the independent FSAL vote contract."""
    descriptor = graph["workload"]
    if (descriptor.get("kind") != "actual_explicit_workload"
            or descriptor.get("family") != "ERK"
            or descriptor.get("inner_solver") is not None
            or descriptor.get("roles") != {}
            or descriptor.get("step_calls") != []
            or graph["scenario_contract"] != {}):
        raise ValueError("Explicit workload contains an implicit role axis")
    state = graph["explicit_step_contract"]
    expected = fsal_contract(descriptor, {
        key: state[key] for key in ("first_step", "all_lanes_accepted")
    })
    if state != expected:
        raise ValueError("ERK FSAL contract differs from actual cache rule")
    rhs = descriptor["actual_step_settings"]["rhs_function"]
    visits = [call for call in graph["calls"]
              if call.get("kind") == "source_call"
              and call["function"] == rhs]
    if len(visits) != expected["rhs_calls"]:
        raise ValueError("ERK RHS invocation coverage differs from FSAL path")
    votes = [node for node in graph["nodes"]
             if node["kind"] in ("AllSync", "AnySync")]
    vote_count = int(state["cache_enabled"] and not state["first_step"])
    if len(votes) != vote_count or any(
        vote["kind"] != "AllSync"
        or vote.get("collective_scope") != "ERK_step_FSAL"
        or vote.get("declared_result") != state["all_lanes_accepted"]
        or vote["source"].get("runtime_region") is not None
        for vote in votes
    ):
        raise ValueError("ERK FSAL step collective differs")
    return expected
