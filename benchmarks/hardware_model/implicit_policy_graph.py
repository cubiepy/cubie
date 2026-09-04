"""Build source-only implicit graphs for exact eight-group policies.

The interpreter retains a separate execution witness for dynamic loop
indices.  That witness may resolve a source cell, but it is never exposed
to the source constant folder.  Static loop templates and executed loop
instances therefore remain separate records.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from cubie import Solver
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_simsafe import UnrollFlags, unroll_if

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model import implicit_native_lowering as native
from benchmarks.hardware_model import implicit_source_graph as implicit
from benchmarks.hardware_model import implicit_workload as workload
from benchmarks.hardware_model import source_value_graph as source
from benchmarks.hardware_model.expansion import CapturedGraph, source_receipt
from benchmarks.hardware_model.buffer_descriptors import registry_layout
from benchmarks.hardware_model.workload_identity import workload_identity


SCRIPT = Path(__file__).resolve()
GROUPS = (
    "unroll_stage",
    "unroll_step_element",
    "unroll_accumulator",
    "unroll_solver_element",
    "unroll_norms",
    "unroll_other_small",
    "unroll_newton_exits",
    "unroll_krylov_exits",
)
LEVELS = {
    "full": (True, None),
    "count1": (True, 1),
    "count2": (True, 2),
    "count4": (True, 4),
    "false": (False, None),
}


class TracedIndex(int):
    """An exact execution index carrying a nonconstant value identity."""

    def __new__(cls, value, identities=()):
        result = int.__new__(cls, int(value))
        result.identities = tuple(sorted(set(identities)))
        return result


def parse_policy(text):
    """Parse exactly eight comma-separated policy levels."""
    levels = tuple(part.strip().lower() for part in text.split(","))
    if len(levels) != len(GROUPS):
        raise ValueError("Policy must contain exactly eight levels")
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown policy levels: {unknown}")
    return levels


def policy_flags(levels):
    """Create the production flag object for one validated policy."""
    if len(levels) != len(GROUPS) or any(x not in LEVELS for x in levels):
        raise ValueError("Policy levels are incomplete or unsupported")
    return UnrollFlags(**dict(zip(GROUPS, (LEVELS[x] for x in levels))))


def policy_record(levels, flags):
    """Retain user spelling and normalized closure values."""
    records = []
    for group, level in zip(GROUPS, levels):
        value = getattr(flags, group)
        if tuple(value) != LEVELS[level]:
            raise ValueError("Production policy conversion changed a level")
        records.append(dict(group=group, level=level, flag=list(value)))
    return records


def loop_structure(values, flag):
    """Return canonical source templates and exact fixed-loop instances."""
    values = list(values)
    flag = tuple(flag)
    if flag == (True, None):
        mode = "full"
        parts = [dict(
            kind="fully_expanded",
            static_body_copies=len(values),
            dynamic_repetitions=1 if values else 0,
            codegen_constant_indices=True,
            lanes=list(range(len(values))),
            indices=values,
        )] if values else []
    elif flag[0] is True and flag[1] in (1, 2, 4):
        mode = "counted"
        count = int(flag[1])
        quotient, remainder = divmod(len(values), count)
        parts = []
        if quotient:
            known = quotient == 1
            parts.append(dict(
                kind="counted_main",
                static_body_copies=count,
                dynamic_repetitions=quotient,
                codegen_constant_indices=known,
                lanes=list(range(count)),
                indices=values[:count] if known else None,
            ))
        if remainder:
            tail = values[quotient * count:]
            parts.append(dict(
                kind="constant_tail",
                static_body_copies=remainder,
                dynamic_repetitions=1,
                codegen_constant_indices=True,
                lanes=list(range(remainder)),
                indices=tail,
            ))
    elif flag == (False, None):
        mode = "backend_choice"
        parts = [dict(
            kind="backend_choice_template",
            static_body_copies=1,
            dynamic_repetitions=len(values),
            codegen_constant_indices=False,
            lanes=[0],
            indices=None,
        )] if values else []
    else:
        raise ValueError(f"Unsupported normalized unroll directive {flag}")
    instances = []
    main = 0
    count = flag[1] if mode == "counted" else None
    if count is not None:
        main = (len(values) // count) * count
    for position, value in enumerate(values):
        if mode == "full":
            part, lane, chunk, known = "fully_expanded", position, 0, True
        elif mode == "counted" and position < main:
            part = "counted_main"
            lane, chunk = position % count, position // count
            known = main == count
        elif mode == "counted":
            part = "constant_tail"
            lane, chunk, known = position - main, 0, True
        else:
            part, lane, chunk, known = (
                "backend_choice_template", 0, position, False
            )
        instances.append(dict(
            position=position,
            value=value,
            part=part,
            lane=lane,
            chunk=chunk,
            codegen_constant=known,
        ))
    return dict(
        mode=mode,
        directive=list(flag),
        fixed_trip_count=len(values),
        source_templates=parts,
        execution_instances=instances,
        source_templates_are_not_native_copies=True,
        execution_instances_are_not_static_copies=True,
        native_replication_known=False,
    )


class PolicyRegionValues(implicit.RegionValues):
    """Interpret fixed policy loops with nonconstant induction witnesses."""

    def __init__(self, graph, descriptor, scenarios, branch_choices=None):
        super().__init__(graph, descriptor, scenarios, branch_choices)
        self.trace_values = {}
        self.trace_decisions = set()
        self.policy_loop_controls = []
        self.address_edges = []
        self.loop_groups = {}
        for function in descriptor["functions"]:
            path = Path(function["source"]["source_path"]).resolve()
            for region in function["replicated_regions"]:
                group = region.get("unroll_group")
                if group is None:
                    continue
                key = (str(path).lower(), int(region["line"]))
                self.loop_groups.setdefault(key, set()).add(group)
        if any(len(groups) != 1 for groups in self.loop_groups.values()):
            raise ValueError("A source loop has conflicting group bindings")

    def loop_group(self, node):
        """Resolve one loop to its captured production closure binding."""
        key = (str(Path(self.path).resolve()).lower(), int(node.lineno))
        groups = self.loop_groups.get(key)
        if groups is None:
            self.unknown(node, "Loop lacks a workload group binding")
        return next(iter(groups))

    def trace_raw(self, value):
        """Return an execution witness without changing source constancy."""
        if value["raw"] is not source.UNKNOWN:
            return value["raw"]
        identity = value.get("identity")
        return self.trace_values.get(identity, source.UNKNOWN)

    def trace_environment(self, environment):
        """Map names to selected execution values for semantic replay only."""
        return {name: self.trace_raw(value)
                for name, value in environment.items()}

    def record_trace(self, result, value):
        """Attach a typed execution witness to a nonconstant graph value."""
        identity = result.get("identity")
        if identity is None or value is source.UNKNOWN:
            return result
        dtype = self.values[identity]["dtype"]
        if dtype == "int32":
            value = np.int32(value)
        elif dtype == "uint32":
            value = np.uint32(value)
        elif dtype == "float32":
            value = np.float32(value)
        elif dtype == "bool":
            value = bool(value)
        self.trace_values[identity] = value
        self.values[identity]["declared_trace_value"] = source.snapshot(value)
        self.values[identity]["trace_value_is_not_codegen_constant"] = True
        return result

    def dynamic_index(self, node, instance, group, loop_id, recurrent):
        """Create one runtime induction SSA value for the selected trace."""
        label = (
            f"loop:{loop_id}:{Path(self.path).name}:{node.lineno}:{group}:"
            f"{instance['part']}:{instance['chunk']}:{instance['lane']}"
        )
        result = self.live_in("int32", label)
        value = self.values[result["identity"]]
        value.update(
            source_origin="runtime_loop_induction",
            external_kernel_input=False,
            loop_group=group,
            loop_line=node.lineno,
            loop_part=instance["part"],
            policy_loop_id=loop_id,
            template_lane=instance["lane"],
            execution_position=instance["position"],
            recurrent=bool(recurrent),
        )
        if self.frames:
            frame = self.frames[-1]
            value["runtime_region"] = dict(
                role=frame["role"],
                call=frame["call"],
                entry_mask=frame["scenario"]["entry_mask"],
            )
        return self.record_trace(result, instance["value"])

    def binary(self, node, left, right):
        """Propagate trace integers without admitting them as constants."""
        operands = [self.scalar(value, node) for value in (left, right)]
        types = [self.values[value["identity"]]["dtype"]
                 for value in operands]
        if set(types) == {"literal_int", "int32"} and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr)
        ):
            position = types.index("literal_int")
            literal = operands[position]["raw"]
            if type(literal) is not int or not -(2**31) <= literal < 2**31:
                self.unknown(node, "Integer literal exceeds int32")
            operands[position] = self.operation(
                "cast", [operands[position]], node, dtype="int32",
                raw=np.int32(literal),
                conversion="exact_int32_literal_operand",
            )[0]
            types[position] = "int32"
        result = super().binary(node, operands[0], operands[1])
        traces = [self.trace_raw(value) for value in operands]
        if any(value is source.UNKNOWN for value in traces):
            return result
        operation = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.BitAnd: lambda a, b: a & b,
            ast.BitOr: lambda a, b: a | b,
        }.get(type(node.op))
        if operation is None:
            return result
        with np.errstate(all="ignore"):
            trace = operation(*traces)
        return self.record_trace(result, trace)

    def expression(self, node, environment):
        """Retain trace results for casts and runtime comparisons."""
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value, environment)
            index = self.index(node.slice, environment)
            if base["reference"] is not None:
                reference = self.slice_view(base["reference"], index, node)
                if reference["shape"]:
                    return source.item(reference=reference)
                return self.read(reference, node)
            captured = base.get("captured_reference")
            if captured is not None or isinstance(
                base["raw"], (tuple, np.ndarray)
            ):
                root = (captured["raw"] if captured is not None else
                        base["raw"])
                indices = (captured["indices"] if captured is not None else
                           []) + [index]
                identities = sorted({value for part in indices
                                     for value in self.index_identities(part)})
                selected = root
                for part in indices:
                    selected = selected[self.plain_index(part)]
                if isinstance(selected, np.ndarray):
                    if not identities:
                        return source.item(selected)
                    result = source.item()
                    result["captured_reference"] = dict(
                        raw=root, indices=indices
                    )
                    return result
                if identities:
                    dtype = source.scalar_type(selected)
                    if dtype not in ("float32", "int32", "uint32", "bool"):
                        self.unknown(
                            node, "Dynamic captured read is not a typed scalar"
                        )
                    inputs = [source.item(identity=value)
                              for value in identities]
                    result = self.operation(
                        "CapturedIndexRead", inputs, node, dtype=dtype,
                        captured=source.snapshot(root),
                        index_template=[self.index_template(part)
                                        for part in indices],
                        selected_execution_index=[self.plain_index(part)
                                                  for part in indices],
                        declared_trace_result=source.snapshot(selected),
                        is_codegen_constant=False,
                    )[0]
                    return self.record_trace(result, selected)
                return source.item(selected)
            self.unknown(node, "Unknown indexed object")
        result = super().expression(node, environment)
        identity = result.get("identity")
        if identity is None:
            return result
        producer = self.values[identity].get("producer")
        if producer is not None and self.nodes[producer]["kind"] == "cast":
            input_value = source.item(
                identity=self.nodes[producer]["inputs"][0]
            )
            trace = self.trace_raw(input_value)
            if trace is not source.UNKNOWN:
                dtype = self.values[identity]["dtype"]
                trace = {"int32": np.int32, "uint32": np.uint32,
                         "float32": np.float32, "bool": bool}[dtype](trace)
                self.record_trace(result, trace)
        if isinstance(node, ast.Compare):
            trace = source.constant(node, self.trace_environment(environment))
            if trace is not source.UNKNOWN:
                if type(trace) not in (bool, np.bool_):
                    self.unknown(node, "Comparison trace is not Boolean")
                self.decisions[identity] = bool(trace)
                self.trace_decisions.add(identity)
                self.record_trace(result, bool(trace))
                self.nodes[producer]["declared_trace_result"] = bool(trace)
        return result

    @staticmethod
    def index_identities(index):
        """Return dynamic value identities embedded in one exact index."""
        if isinstance(index, TracedIndex):
            return list(index.identities)
        if isinstance(index, tuple):
            return sorted({
                value for part in index
                for value in PolicyRegionValues.index_identities(part)
            })
        if isinstance(index, slice):
            return sorted({
                value for part in (index.start, index.stop, index.step)
                for value in PolicyRegionValues.index_identities(part)
            })
        return []

    @staticmethod
    def plain_index(index):
        """Strip execution-witness wrappers for exact semantic lookup."""
        if isinstance(index, TracedIndex):
            return int(index)
        if isinstance(index, tuple):
            return tuple(
                PolicyRegionValues.plain_index(part) for part in index
            )
        if isinstance(index, slice):
            return slice(*(PolicyRegionValues.plain_index(part) for part in (
                index.start, index.stop, index.step
            )))
        return index

    @staticmethod
    def index_template(index):
        """Describe literals and dynamic operands without their trace value."""
        if isinstance(index, TracedIndex):
            return dict(dynamic_values=list(index.identities))
        if isinstance(index, tuple):
            return [PolicyRegionValues.index_template(part) for part in index]
        if isinstance(index, slice):
            return dict(slice=[
                PolicyRegionValues.index_template(part)
                for part in (index.start, index.stop, index.step)
            ])
        return dict(literal=index)

    def condition(self, node, environment):
        """Select a runtime path while naming trace-index decisions exactly."""
        self.last_branch_event = None
        folded = self.condition_constant(node, environment)
        if folded is not source.UNKNOWN:
            return folded
        predicate = self.scalar(self.expression(node, environment), node)
        if self.values[predicate["identity"]]["dtype"] != "bool":
            self.unknown(node, "Runtime branch does not have bool predicate")
        identity = predicate["identity"]
        if identity in self.decisions:
            choice = self.decisions[identity]
            reason = (
                "exact dynamic-induction execution witness"
                if identity in self.trace_decisions else
                "declared uniform convergence-mask regime"
            )
        else:
            key = f"{Path(self.path).name}:{node.lineno}"
            if key not in self.branch_choices:
                self.unknown(
                    node, "Runtime branch needs explicit choice " + key
                )
            self.used_branch_choices.add(key)
            choice = self.branch_choices[key]
            if type(choice) is not bool:
                self.unknown(node, "Branch choice must be bool")
            reason = "explicit runtime-path assumption"
        self.controls.append(dict(
            kind="runtime_branch_choice", source=self.location(node),
            predicate=identity, choice=choice, reason=reason,
            is_codegen_constant=False,
        ))
        _, self.last_branch_event = self.operation(
            "BranchDecision", [predicate], node, selected_path=choice,
            decision_reason=reason, is_codegen_constant=False,
        )
        return choice

    def index(self, node, environment):
        """Resolve a cell from an execution witness, never from a fake fold."""
        if isinstance(node, ast.Slice):
            return slice(*(
                None if part is None else self.index(part, environment)
                for part in (node.lower, node.upper, node.step)
            ))
        if isinstance(node, ast.Tuple):
            return tuple(self.index(part, environment) for part in node.elts)
        folded = source.constant(node, self.raw_environment(environment))
        if isinstance(folded, (bool, np.bool_)):
            self.unknown(node, "Boolean array indices are unsupported")
        if isinstance(folded, (int, np.integer)):
            return int(folded)
        value = self.expression(node, environment)
        trace = self.trace_raw(value)
        if isinstance(trace, (bool, np.bool_)):
            self.unknown(node, "Boolean trace index is unsupported")
        if not isinstance(trace, (int, np.integer)):
            self.unknown(node, "Index lacks an exact execution witness")
        return TracedIndex(trace, [value["identity"]])

    def slice_view(self, reference, index, node):
        """Carry dynamic address values beside the exact selected cell."""
        identities = []

        def plain(value):
            if isinstance(value, TracedIndex):
                identities.extend(value.identities)
                return int(value)
            if isinstance(value, tuple):
                return tuple(plain(part) for part in value)
            if isinstance(value, slice):
                return slice(plain(value.start), plain(value.stop),
                             plain(value.step))
            return value

        result = super().slice_view(reference, plain(index), node)
        inherited = reference.get("address_value_ids", [])
        if identities or inherited:
            result["address_value_ids"] = sorted(
                set(identities) | set(inherited)
            )
        return result

    def address(self, reference, node_id):
        """Record address dependencies without changing cell-value inputs."""
        values = sorted(set(reference.get("address_value_ids", [])))
        if not values:
            return
        record = self.nodes[node_id]
        record["address_value_ids"] = values
        producers = [self.values[value]["producer"] for value in values
                     if self.values[value]["producer"] is not None]
        record["order_predecessors"] = sorted(
            set(record["order_predecessors"]) | set(producers)
        )
        self.address_edges.extend(
            dict(value=value, access_node=node_id) for value in values
        )

    def read(self, reference, node):
        """Read one exact trace cell and retain its dynamic address edge."""
        result = super().read(reference, node)
        node_id = next(iter(result["ordering"]))
        self.address(reference, node_id)
        return result

    def write(self, reference, result, node):
        """Write one exact trace cell and retain its dynamic address edge."""
        before = len(self.nodes)
        super().write(reference, result, node)
        if len(self.nodes) != before + 1:
            self.unknown(node, "A scalar write did not emit one alias event")
        self.address(reference, before)

    def location(self, node):
        """Hash static templates independently of execution positions."""
        result = source.SourceValues.location(self, node)
        if self.frames:
            frame = self.frames[-1]
            result["runtime_region"] = dict(
                role=frame["role"], call=frame["call"],
                body_index=frame.get("body_index"),
                entry_mask=frame["scenario"]["entry_mask"],
                phase=frame.get("phase", "entry"),
            )
        if self.templates:
            loops = []
            for loop in self.loop_indices:
                loops.append(dict(
                    policy_loop_id=loop["policy_loop_id"],
                    line=loop["line"], group=loop["group"],
                    part=loop["part"], lane=loop["lane"],
                    codegen_constant=loop["codegen_constant"],
                    static_index=(loop["execution_index"]
                                  if loop["codegen_constant"] else None),
                    recurrent=loop["recurrent"],
                    directive=loop["directive"],
                ))
            result["execution_loop_instances"] = [
                dict(loop) for loop in self.loop_indices
            ]
            result["loop_indices"] = loops
            fixed = [loop for loop in loops if loop["recurrent"] is False]
            template = dict(
                function=self.templates[-1], line=result["line"],
                syntax=result["syntax"], fixed_indices=fixed,
                policy_loop_templates=loops,
            )
            result["hot_template_identity"] = hashlib.sha256(
                json.dumps(template, sort_keys=True).encode()
            ).hexdigest()
            result["hot_template"] = template
            result["template_is_not_native_copy_identity"] = True
        return result

    def fixed_loop(self, node, environment):
        """Execute one fixed or recurrent loop under its exact directive."""
        iterator = node.iter
        if (not isinstance(iterator, ast.Call)
                or self.expression(iterator.func, environment)["raw"]
                is not unroll_if or len(iterator.args) != 2):
            self.unknown(node, "Loop lacks actual unroll directive")
        bounds = iterator.args[0]
        if (not isinstance(bounds, ast.Call)
                or self.expression(
                    bounds.func, environment
                )["raw"] is not range
                or bounds.keywords):
            self.unknown(node, "Unsupported loop range")
        values = list(range(*(source.SourceValues.index(self, argument,
                                                        environment)
                              for argument in bounds.args)))
        flag = source.constant(iterator.args[1],
                               self.raw_environment(environment))
        if not isinstance(flag, tuple):
            self.unknown(node, "Loop directive is not a closure tuple")
        structure = loop_structure(values, flag)
        group = self.loop_group(node)
        frame = self.frames[-1] if self.frames else None
        recurrence = None
        if frame is not None:
            role = self.descriptor["roles"][frame["role"]]
            recurrence = role["iteration"].get("source_region")
        recurrent = (
            recurrence is not None
            and node.lineno == recurrence["line"]
            and self.stack[-1] == id(self.role_functions[frame["role"]])
        )
        instances = structure["execution_instances"]
        if recurrent:
            count = len(frame["scenario"]["active_masks"])
            if len(values) != role["cap"]:
                self.unknown(node, "Recurrent source cap differs")
            instances = instances[:min(role["cap"], count + 1)]
        control = dict(
            kind=("recurrent_execution_trace" if recurrent else
                  "policy_fixed_execution_trace"),
            source=self.location(node),
            group=group,
            indices=[item["value"] for item in instances],
            directive=list(flag),
            structure=structure,
            executed_instances=instances,
            code_copies_are_execution_count=False,
            policy_loop_id=len(self.policy_loop_controls),
        )
        self.controls.append(control)
        self.policy_loop_controls.append(control)
        if node.orelse or len(instances) > 10000:
            self.unknown(node, "Unsupported loop execution trace")
        for instance in instances:
            if recurrent:
                frame["body_index"] = instance["value"]
                frame["phase"] = (
                    "body" if instance["value"] < count else "exit_vote"
                )
            if instance["codegen_constant"]:
                index = source.item(instance["value"])
            else:
                index = self.dynamic_index(
                    node, instance, group, control["policy_loop_id"], recurrent
                )
            self.assign(node.target, index, environment)
            loop = dict(
                policy_loop_id=control["policy_loop_id"],
                line=node.lineno,
                group=group,
                part=instance["part"],
                lane=instance["lane"],
                chunk=instance["chunk"],
                execution_index=instance["value"],
                codegen_constant=instance["codegen_constant"],
                recurrent=recurrent,
                directive=list(flag),
            )
            self.loop_indices.append(loop)
            returned, value = self.block(node.body, environment)
            self.loop_indices.pop()
            if returned == "break":
                break
            if returned:
                return returned, value
        if recurrent:
            frame.pop("body_index", None)
            frame["phase"] = "exit"
        return False, source.item(None)

    def block(self, statements, environment):
        """Intercept policy loops and reuse the verified statement engine."""
        for node in statements:
            if isinstance(node, ast.For):
                returned, value = self.fixed_loop(node, environment)
            else:
                returned, value = super().block([node], environment)
            if returned:
                return returned, value
        return False, source.item(None)


class PolicyTypedLowering(native.TypedLowering):
    """Extend the verified typed lowerer for dynamic captured-table reads."""

    def typed_operation(self, kind, inputs, dtype, node):
        if kind != "CapturedIndexRead":
            return super().typed_operation(kind, inputs, dtype, node)
        types = [self.values[value]["dtype"] for value in inputs]
        if not inputs or any(value != "int32" for value in types):
            raise native.Unresolved(
                "Captured-table lookup needs dynamic int32 indices"
            )
        if dtype not in ("float32", "int32", "uint32", "bool"):
            raise native.Unresolved("Captured-table lookup result is untyped")
        return "CAPTURED_LOOKUP", dict(
            source_operation=kind,
            table_sha256=hashlib.sha256(
                payload(node["captured"]).encode()
            ).hexdigest(),
            index_template=node["index_template"],
            result_dtype=dtype,
            lowering_alternatives=[
                "constant_or_parameter_memory_lookup",
                "comparison_select_tree",
            ],
            native_form_unresolved=True,
        )


def validate_plan_inputs(architecture, compiler):
    """Apply the reused typed lowerer's exact alternative contracts."""
    required_architecture = {
        "name", "provenance", "gpr_budget", "predicate_budget",
        "gpr_scope", "predicate_scope",
    }
    if (
        set(architecture) != required_architecture
        or not architecture["provenance"]
        or architecture["gpr_scope"] != "per_thread_scenario"
        or architecture["predicate_scope"] != "per_thread_scenario"
    ):
        raise ValueError("Architecture needs explicit bank-budget provenance")
    required_compiler = {
        "name", "provenance", "fp32_flush_subnormals", "fp32_contract",
        "division", "sqrt", "numeric_literals", "predicate_literals",
        "integer_dynamic_width_bits", "predicate_spills", "schedule",
    }
    if (
        set(compiler) != required_compiler
        or not compiler["name"]
        or not isinstance(compiler["provenance"], list)
        or not compiler["provenance"]
        or type(compiler["fp32_flush_subnormals"]) is not bool
        or type(compiler["fp32_contract"]) is not bool
        or compiler["division"] != "approximate_reciprocal_multiply"
        or compiler["sqrt"] != "approximate_native_no_refinement"
        or compiler["numeric_literals"] != "materialized_gpr"
        or compiler["predicate_literals"] != "PT_or_inverted_PT"
        or compiler["integer_dynamic_width_bits"] != 32
        or compiler["predicate_spills"] != "canonical_uint32_local"
        or compiler["schedule"] != "source_order"
    ):
        raise ValueError("Compiler alternative is incomplete or unsupported")
    for record in compiler["provenance"]:
        if (
            "path" in record
            and hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise ValueError("Compiler-alternative source bytes changed")


def special_typed_plan(graph, architecture, compiler, materialization):
    """Build the typed alternative when a captured lookup remains dynamic."""
    checked, regime = native.validate_source(graph)
    validate_plan_inputs(architecture, compiler)
    lowered = PolicyTypedLowering(graph, compiler, materialization).build()
    if lowered != PolicyTypedLowering(
        graph, compiler, materialization
    ).build():
        raise ValueError("Policy typed lowering is not deterministic")
    special = []
    for source_node in graph["nodes"]:
        if source_node["kind"] != "CapturedIndexRead":
            continue
        mapped = lowered["source_node_mapping"][source_node["id"]]
        if len(mapped) != 1:
            raise ValueError("Captured lookup has no unique typed form")
        node = lowered["nodes"][mapped[0]]
        if (
            node["opcode"] != "CAPTURED_LOOKUP"
            or len(node["inputs"]) != len(source_node["inputs"])
            or node["semantics"].get("native_form_unresolved") is not True
            or node["semantics"].get("source_operation")
            != "CapturedIndexRead"
        ):
            raise ValueError("Captured lookup typed form differs")
        special.append(mapped[0])
    allocation = native.BankAllocation(
        lowered,
        architecture["gpr_budget"],
        architecture["predicate_budget"],
    ).build()
    allocation_check = native.verify_allocation(lowered, allocation)
    templates = {}
    for node in lowered["nodes"]:
        for context in node["source_contexts"]:
            identity = context["hot_template_identity"]
            item = templates.setdefault(identity, dict(
                hot_template_identity=identity,
                source=context["hot_template"],
                selected_trace_instances=0,
                modeled_opcodes=Counter(),
            ))
            item["selected_trace_instances"] += 1
            item["modeled_opcodes"][node["opcode"]] += 1
    for item in templates.values():
        item["modeled_opcodes"] = dict(item["modeled_opcodes"])
    return dict(
        schema=1,
        kind="conditional_typed_implicit_native_plan",
        provenance=dict(
            lowerer=source_receipt(special_typed_plan),
            reused_typed_lowerer=source_receipt(native.TypedLowering),
            source_files=checked,
        ),
        architecture=architecture,
        compiler_alternative=compiler,
        lowering=lowered,
        static_hot_templates=list(templates.values()),
        allocation=allocation,
        verification=dict(
            **allocation_check,
            typed_forms=dict(
                status="PASS",
                captured_lookup_forms=len(special),
                exact_rebuild=True,
                inherited_forms="reused verified TypedLowering methods",
            ),
        ),
        assumptions=dict(
            integer_arithmetic="32-bit wrapping; signed comparisons typed",
            floating_mode=("FTZ" if compiler["fp32_flush_subnormals"]
                           else "non_FTZ"),
            captured_lookup=(
                "typed dynamic table operation; native memory/select form "
                "is unresolved"
            ),
            native_control="BRA family; reconvergence work unresolved",
            register_policy=(
                "fixed order with explicit predicate word spills"
            ),
        ),
        dynamic_work=dict(
            trace_operations=len(lowered["nodes"]),
            trace_opcodes=dict(Counter(
                node["opcode"] for node in lowered["nodes"]
            )),
            iterations=regime,
            iteration_values_are_explicit_scenarios=True,
            static_hot_instruction_bytes=None,
            hot_templates_are_not_native_copy_counts=True,
        ),
        native_labels_consumed=False,
        measured_iteration_counts_consumed=False,
        complete_kernel_prediction=False,
        unresolved=[
            "Captured lookup native memory or select lowering",
            "Actual native scheduling, loop replication, and ABI temporaries",
            "Memory warp layout, cache reuse, and service",
        ],
    )


def finish_graph(engine, graph, descriptor, caller, step, policy):
    """Build the stable graph schema from one completed policy trace."""
    outputs = {value["identity"] for cell, value in engine.cells.items()
               if cell[0] in engine.boundary}
    returned = engine.returned
    if returned["raw"] is not None or returned["identity"] is not None:
        outputs.add(engine.scalar(returned, source.source_function(step))[
            "identity"
        ])
    consumers = {}
    value_edges = []
    for node in engine.nodes:
        for value in node["inputs"]:
            consumers.setdefault(value, set()).add(node["id"])
            producer = engine.values[value]["producer"]
            if producer is not None:
                value_edges.append(dict(
                    producer=producer, consumer=node["id"], value=value
                ))
    certificates = []
    for call in engine.calls:
        if call["kind"] != "source_call":
            continue
        nodes = engine.nodes[call["first_node"]:call["end_node"]]
        membership = {node["id"] for node in nodes}
        used = {value for node in nodes
                for value in node["inputs"] + node["outputs"]}
        incoming = {value for value in used
                    if engine.values[value]["producer"] not in membership
                    and engine.values[value]["kind"] != "constant"}
        outgoing = set()
        for value in engine.values:
            identifier = value["id"]
            producer = value["producer"]
            if value["kind"] == "constant":
                continue
            existed = producer is None or producer < call["end_node"]
            later = identifier in outputs or any(
                node >= call["end_node"]
                for node in consumers.get(identifier, ())
            )
            if existed and later:
                outgoing.add(identifier)
                if producer is None or producer < call["first_node"]:
                    incoming.add(identifier)
        certificates.append(dict(
            context=call["context"], function=call["function"],
            node_ids=sorted(membership), live_ins=sorted(incoming),
            observable_outputs=sorted(outgoing),
            scope="lexical interval with conservative caller live-through",
        ))
    result = dict(
        schema=2,
        kind="typed_implicit_execution_region",
        provenance=dict(
            extractor=source_receipt(describe_policy_source),
            dependencies=[
                source_receipt(PolicyRegionValues),
                source_receipt(implicit.RegionValues),
                source_receipt(source.SourceValues),
                source_receipt(workload.evaluate_regime),
                source_receipt(workload_identity),
            ],
            functions=graph.functions,
        ),
        policy=policy,
        workload=descriptor,
        caller=caller,
        registry=registry_layout(
            engine.solver.kernel.single_integrator._algo_step
        ),
        allocations=list(engine.allocations.values()),
        calls=engine.calls,
        values=engine.values,
        nodes=engine.nodes,
        controls=engine.controls,
        policy_loops=engine.policy_loop_controls,
        value_edges=value_edges,
        address_edges=engine.address_edges,
        aliases=engine.aliases,
        live_ins=engine.input_ids,
        certificates=certificates,
        required_control_nodes=[node["id"] for node in engine.nodes
                                if node["kind"] == "BranchDecision"],
        observable_values=sorted(outputs),
        final_cells=[dict(
            cell=list(cell), value=value["identity"],
            boundary=cell[0] in engine.boundary,
        ) for cell, value in sorted(engine.cells.items())],
        scenario_contract=engine.scenarios,
        regime=engine.regime,
        branch_choices=engine.branch_choices,
        operation_inventory=[dict(
            kind=kind, inputs=list(inputs), outputs=list(outputs), count=count,
        ) for (kind, inputs, outputs), count in Counter(
            (node["kind"],
             tuple(engine.values[key]["dtype"] for key in node["inputs"]),
             tuple(engine.values[key]["dtype"] for key in node["outputs"]))
            for node in engine.nodes
        ).items()],
        compilation_check=dict(
            native_overloads=0,
            batch_kernel_requested=False,
            device_function_executed=False,
        ),
        semantic_contract=dict(
            path="one declared uniform successful path",
            floating_point=(
                "FP32 source operations; selected-path replay assumes finite "
                "inputs and intermediates"
            ),
            nan_sensitive_primitives=(
                "source callable identity is retained; Python min/max is not "
                "identified with fmin/fmax"
            ),
            integer_values=(
                "int32 storage-compatible low-32-bit abstraction; dynamic "
                "induction witnesses are not source constants"
            ),
            native_integer_width="unresolved before specialization",
            vote_mask="participating mask equals declared active-entry mask",
            dynamic_indices=(
                "declared trace values select exact cells for semantic "
                "replay; raw values remain UNKNOWN to constant folding"
            ),
        ),
        limitations=[
            "Runtime branch choices are workload hypotheses",
            "Executed recurrent and fixed bodies are not native code copies",
            "Canonical counted main/tail structure is a source alternative",
            "False leaves native expansion to the compiler",
            "Address edges are not native address-instruction predictions",
            "No native specialization, label, timing, or fitted term is used",
        ],
    )
    return result


def describe_policy_source(solver, scenarios, policy, branch_choices=None):
    """Capture one actual implicit step under an explicit production policy."""
    descriptor = workload.describe_implicit_workload(solver)
    step_owner = solver.kernel.single_integrator._algo_step
    actual = step_owner.compile_settings.unroll
    if [list(getattr(actual, group)) for group in GROUPS] != [
        item["flag"] for item in policy
    ]:
        raise ValueError("Solver closure policy differs from requested policy")
    graph = CapturedGraph()
    graph.add_function(step_owner.step_function, "algorithm_step")
    engine = PolicyRegionValues(graph, descriptor, scenarios, branch_choices)
    engine.solver = solver
    step, bound, caller = source.caller_bindings(engine, solver)
    engine.returned = engine.invoke(step, bound)
    if set(engine.entered_calls) != set(scenarios):
        raise ValueError("Source path did not enter every supplied role call")
    unused = set(engine.branch_choices) - engine.used_branch_choices
    if unused:
        raise ValueError(f"Unused runtime branch choices: {sorted(unused)}")
    if any(dispatcher.overloads for dispatcher in graph.dispatchers):
        raise ValueError("Native specialization appeared during extraction")
    result = finish_graph(
        engine, graph, descriptor, caller, step, policy
    )
    result["candidate_construction"] = dict(
        workload_identity=workload_identity(solver.system, solver),
        shared_stride_bytes=4 * (
            int(solver.kernel.shared_memory_elements)
            + int(solver.kernel.shared_memory_needs_padding)
        ),
        precision="float32",
    )
    verify_policy_graph(result)
    return result


def payload(value):
    """Return a stable JSON payload for one source value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def semantic_hashes(graph):
    """Hash the selected source semantics, using trace indices as witnesses."""
    hashes = {}
    for value in graph["values"]:
        if value["kind"] == "constant":
            record = ["constant", value["dtype"], value["constant"]]
        elif value.get("trace_value_is_not_codegen_constant") is True:
            record = ["trace_witness", value["dtype"],
                      value["declared_trace_value"]]
        elif value["kind"] == "live_in":
            record = ["live_in", value["dtype"], value.get("label")]
        else:
            node = graph["nodes"][value["producer"]]
            record = [node["kind"], value["dtype"],
                      [hashes[item] for item in node["inputs"]]]
        hashes[value["id"]] = hashlib.sha256(
            payload(record).encode()
        ).hexdigest()
    return hashes


def semantic_certificate(graph):
    """Return final-cell and observable symbolic hashes for one trace."""
    hashes = semantic_hashes(graph)
    return dict(
        final_cells={payload(item["cell"]): hashes[item["value"]]
                     for item in graph["final_cells"]},
        observables=sorted(hashes[value]
                           for value in graph["observable_values"]),
    )


def restored_scalar(record):
    """Restore one exact scalar snapshot without host type promotion."""
    if isinstance(record, dict) and set(record) == {"dtype", "value"}:
        dtype = record["dtype"]
        value = restored_scalar(record["value"])
        if dtype == "float32":
            return np.float32(value)
        if dtype == "int32":
            return np.int32(value)
        if dtype == "uint32":
            return np.uint32(value)
        if dtype == "bool":
            return np.bool_(value)
        raise ValueError(f"Unsupported scalar snapshot dtype {dtype}")
    if isinstance(record, (bool, int, float)):
        return record
    raise ValueError("Snapshot is not a supported scalar")


def typed_scalar(dtype, value):
    """Cast one value to its recorded source scalar type."""
    if dtype == "float32":
        return np.float32(value)
    if dtype == "int32":
        return np.int32(value)
    if dtype == "uint32":
        return np.uint32(value)
    if dtype == "bool":
        return np.bool_(value)
    if dtype == "literal_int":
        return int(value)
    if dtype == "literal_float":
        return float(value)
    raise ValueError(f"Unsupported replay dtype {dtype}")


def deterministic_live_in(value, seed):
    """Choose a finite reproducible value for one ordinary live-in."""
    dtype = value["dtype"]
    label = payload(value.get("label"))
    number = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    if dtype == "float32":
        magnitude = np.float32((number % 11 + 2 + seed) / 32.0)
        return -magnitude if (number // 11) % 2 else magnitude
    if dtype in ("int32", "uint32", "literal_int"):
        return typed_scalar(dtype, (number + seed) % 3)
    if dtype == "bool":
        return np.bool_((number + seed) % 2)
    raise ValueError(f"Unsupported live-in dtype {dtype}")


def numeric_payload(dtype, value):
    """Serialize a replayed scalar with its exact typed bit pattern."""
    value = typed_scalar(dtype, value)
    if dtype == "float32":
        if not np.isfinite(value):
            raise ValueError("Replay produced a nonfinite FP32 value")
        bits = struct.pack("<f", float(value)).hex()
    elif dtype == "int32":
        bits = struct.pack("<i", int(value)).hex()
    elif dtype == "uint32":
        bits = struct.pack("<I", int(value)).hex()
    elif dtype == "bool":
        bits = "01" if bool(value) else "00"
    elif dtype == "literal_int":
        bits = str(int(value))
    elif dtype == "literal_float":
        if not np.isfinite(value):
            raise ValueError("Replay produced a nonfinite literal float")
        bits = float(value).hex()
    else:
        raise ValueError(f"Unsupported replay result dtype {dtype}")
    return dict(dtype=dtype, bits=bits)


def resolve_index_template(template, values):
    """Resolve one dynamic captured-array index from replay values."""
    if isinstance(template, dict) and "dynamic_values" in template:
        identities = template["dynamic_values"]
        if len(identities) != 1:
            raise ValueError("Dynamic index expression is not scalar")
        return int(values[identities[0]])
    if isinstance(template, dict) and "slice" in template:
        raise ValueError("Dynamic captured slices are not supported")
    if isinstance(template, dict) and set(template) == {"literal"}:
        literal = template["literal"]
        if isinstance(literal, bool) or not isinstance(literal, int):
            raise ValueError("Captured-array literal index is not integer")
        return literal
    if isinstance(template, list):
        return [resolve_index_template(part, values) for part in template]
    if isinstance(template, bool) or not isinstance(template, int):
        raise ValueError("Captured-array index is not an integer")
    return template


def replay_node(node, inputs, output_dtype, replay_values):
    """Evaluate one admitted source node under exact typed operations."""
    kind = node["kind"]
    if kind in ("element_read_alias", "element_write_alias"):
        if node["outputs"]:
            raise ValueError(f"Control/alias node {kind} has an output")
        return None
    if kind == "BranchDecision":
        if node["outputs"]:
            raise ValueError("BranchDecision has an output")
        if (
            node["decision_reason"]
            == "exact dynamic-induction execution witness"
            and bool(inputs[0]) != node["selected_path"]
        ):
            raise ValueError("Dynamic trace selected a different branch")
        return None
    if kind == "CapturedIndexRead":
        indices = resolve_index_template(node["index_template"],
                                         replay_values)
        selected = captured_selection(node["captured"], indices)
        return typed_scalar(output_dtype, restored_scalar(selected))
    if kind == "ActiveMask":
        return typed_scalar(output_dtype,
                            node["declared_active_entry_mask"])
    if kind in ("AllSync", "AnySync"):
        return typed_scalar(output_dtype, node["declared_result"])
    if kind == "cast":
        return typed_scalar(output_dtype, inputs[0])
    if kind == "Select":
        return typed_scalar(output_dtype,
                            inputs[1] if bool(inputs[0]) else inputs[2])
    binary = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mult": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
        "BitAnd": lambda a, b: a & b,
        "BitOr": lambda a, b: a | b,
        "And": lambda a, b: bool(a) and bool(b),
        "Or": lambda a, b: bool(a) or bool(b),
        "Minimum": min,
        "Maximum": max,
        "CompareEq": lambda a, b: a == b,
        "CompareNotEq": lambda a, b: a != b,
        "CompareLt": lambda a, b: a < b,
        "CompareLtE": lambda a, b: a <= b,
        "CompareGt": lambda a, b: a > b,
        "CompareGtE": lambda a, b: a >= b,
    }
    unary = {
        "UAdd": lambda value: +value,
        "USub": lambda value: -value,
        "Not": lambda value: not value,
        "Abs": abs,
        "Sqrt": np.sqrt,
    }
    if kind in binary and len(inputs) == 2:
        with np.errstate(all="raise"):
            result = binary[kind](*inputs)
        return typed_scalar(output_dtype, result)
    if kind in unary and len(inputs) == 1:
        with np.errstate(all="raise"):
            result = unary[kind](inputs[0])
        return typed_scalar(output_dtype, result)
    raise ValueError(f"Unsupported replay operation {kind}")


def numeric_semantic_certificate(graph, seed):
    """Replay one selected path and certify exact boundary result bits."""
    replay = {}
    for value in graph["values"]:
        if value["kind"] == "constant":
            replay[value["id"]] = typed_scalar(
                value["dtype"], restored_scalar(value["constant"])
            )
        elif value.get("source_origin") == "runtime_loop_induction":
            replay[value["id"]] = typed_scalar(
                value["dtype"],
                restored_scalar(value["declared_trace_value"]),
            )
        elif value["kind"] == "live_in":
            replay[value["id"]] = deterministic_live_in(value, seed)
    for node in graph["nodes"]:
        inputs = [replay[value] for value in node["inputs"]]
        if not node["outputs"]:
            replay_node(node, inputs, None, replay)
            continue
        if len(node["outputs"]) != 1:
            raise ValueError("Replay supports scalar single-output nodes")
        output = graph["values"][node["outputs"][0]]
        replay[output["id"]] = replay_node(
            node, inputs, output["dtype"], replay
        )
        if "declared_trace_value" in output:
            declared = restored_scalar(output["declared_trace_value"])
            if numeric_payload(output["dtype"], replay[output["id"]]) != (
                numeric_payload(output["dtype"], declared)
            ):
                raise ValueError("Declared trace result differs from replay")
    boundary = {
        payload(item["cell"]): numeric_payload(
            graph["values"][item["value"]]["dtype"],
            replay[item["value"]],
        )
        for item in graph["final_cells"] if item["boundary"]
    }
    final_values = {item["value"] for item in graph["final_cells"]
                    if item["boundary"]}
    remaining = [value for value in graph["observable_values"]
                 if value not in final_values]
    observables = {
        str(position): numeric_payload(
            graph["values"][value]["dtype"], replay[value]
        )
        for position, value in enumerate(remaining)
    }
    return dict(seed=seed, boundary_cells=boundary,
                observables=observables)


def numeric_semantic_certificates(graph, seeds=(0, 1, 2)):
    """Return several deterministic selected-path replay certificates."""
    return [numeric_semantic_certificate(graph, seed) for seed in seeds]


def template_values(template):
    """Return dynamic value IDs contained in an index template."""
    if isinstance(template, dict) and "dynamic_values" in template:
        return list(template["dynamic_values"])
    if isinstance(template, dict) and "slice" in template:
        return [value for part in template["slice"]
                for value in template_values(part)]
    if isinstance(template, list):
        return [value for part in template for value in template_values(part)]
    return []


def captured_selection(captured, indices):
    """Select an exact scalar from a serialized captured array."""
    value = captured["values"] if isinstance(captured, dict) else captured
    for index in indices:
        if isinstance(index, list):
            for part in index:
                value = value[int(part)]
        else:
            value = value[int(index)]
    if isinstance(captured, dict) and captured.get("kind") == "array":
        return source.snapshot(np.dtype(captured["dtype"]).type(value))
    return value


def verify_function_sources(functions, policy_by_group):
    """Bind captured source bytes and closure flags to one policy."""
    for function in functions:
        source_record = function["source"]
        path = Path(source_record["source_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != (
            source_record["source_sha256"]
        ):
            raise ValueError("Captured function source bytes changed")
        closure = function.get("closure_constants", {})
        for loop in function.get("loops", []):
            group = loop.get("unroll_group")
            if group is None:
                continue
            expected = policy_by_group[group]
            if loop.get("flag") != expected:
                raise ValueError("Function loop flag differs from policy")
            binding = loop.get("flag_binding")
            if binding not in closure or closure[binding] != expected:
                raise ValueError("Function closure flag differs from policy")


def verify_workload_policy(workload_record, policy_by_group):
    """Bind workload recurrent regions and construction state to policy."""
    check = workload_record.get("compilation_check", {})
    if (
        set(check) != {
            "batch_kernel_requested", "inspected_dispatchers",
            "native_overloads",
        }
        or
        check.get("native_overloads") != 0
        or check.get("batch_kernel_requested") is not False
        or not isinstance(check.get("inspected_dispatchers"), int)
        or check["inspected_dispatchers"] <= 0
    ):
        raise ValueError("Workload native-construction boundary differs")
    verify_function_sources(workload_record["functions"], policy_by_group)
    modes = {
        (True, None): "full",
        (True, 1): "counted",
        (True, 2): "counted",
        (True, 4): "counted",
        (False, None): "backend_choice",
    }
    for role in workload_record["roles"].values():
        region = role["iteration"].get("source_region")
        if region is None:
            continue
        group = region["unroll_group"]
        expected = policy_by_group[group]
        if (
            region.get("actual_closure_flag") != expected
            or region.get("candidate_flag") != expected
            or region.get("mode") != modes[tuple(expected)]
        ):
            raise ValueError("Workload recurrent flag differs from policy")


def source_loop_values(graph, control, trees):
    """Reparse one captured source range under its closure scalars."""
    calls = {
        call["context"]: call["function"] for call in graph["calls"]
        if call.get("kind") == "source_call"
    }
    context = control["source"]["context"]
    if context not in calls:
        raise ValueError("Policy loop has no source-call owner")
    functions = {
        function["id"]: function
        for function in graph["provenance"]["functions"]
    }
    function = functions[calls[context]]
    path = Path(function["source"]["source_path"])
    if str(path).lower() != control["source"]["path"].lower():
        raise ValueError("Policy loop source owner differs")
    if path not in trees:
        trees[path] = ast.parse(path.read_text())
    matches = [node for node in ast.walk(trees[path])
               if isinstance(node, ast.For)
               and node.lineno == control["source"]["line"]]
    if len(matches) != 1:
        raise ValueError("Policy loop source range is ambiguous")
    iterator = matches[0].iter
    described = [loop for loop in function.get("loops", [])
                 if loop["source"]["line"] == control["source"]["line"]]
    if (
        len(described) != 1
        or described[0].get("unroll_group") != control["group"]
        or described[0].get("iterator") != ast.unparse(iterator)
    ):
        raise ValueError("Policy loop descriptor differs from source")
    if (
        not isinstance(iterator, ast.Call)
        or not isinstance(iterator.func, ast.Name)
        or iterator.func.id != "unroll_if"
        or len(iterator.args) != 2
    ):
        raise ValueError("Policy source loop does not call unroll_if")
    range_call = iterator.args[0]
    if (
        not isinstance(range_call, ast.Call)
        or not isinstance(range_call.func, ast.Name)
        or range_call.func.id != "range"
        or range_call.keywords
        or not 1 <= len(range_call.args) <= 3
    ):
        raise ValueError("Policy source loop range is unsupported")
    environment = {}
    for name, value in function.get("closure_constants", {}).items():
        try:
            environment[name] = restored_scalar(value)
        except ValueError:
            pass
    bounds = [source.constant(argument, environment)
              for argument in range_call.args]
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in bounds
    ):
        raise ValueError("Policy source range bound is not a captured integer")
    values = list(range(*(int(value) for value in bounds)))
    if described[0].get("trip_count") != len(values):
        raise ValueError("Policy loop descriptor trip count differs")
    return values


def source_call_stage(graph, call, calls):
    """Derive the source stage from the enclosing loop induction."""
    site = call["call_site"]
    parent = calls[site["context"]]
    functions = {item["id"]: item
                 for item in graph["provenance"]["functions"]}
    function = functions[parent["function"]]
    path = Path(site["path"])
    tree = ast.parse(path.read_text())
    enclosing = [node for node in ast.walk(tree)
                 if isinstance(node, ast.For)
                 and node.lineno < site["line"] <= node.end_lineno]
    instances = site.get("execution_loop_instances", [])
    stage_loops = [(node, record) for node in enclosing
                   for record in instances
                   if record["group"] == "unroll_stage"
                   and record["line"] == node.lineno]
    if not stage_loops:
        return 0
    if len(stage_loops) != 1:
        raise ValueError("Source role call has ambiguous stage nesting")
    loop, instance = stage_loops[0]
    control = graph["policy_loops"][instance["policy_loop_id"]]
    if (control["source"]["context"] != parent["context"]
            or not any(item["value"] == instance["execution_index"]
                       for item in control["executed_instances"])):
        raise ValueError("Source stage induction differs from its owner")
    if not isinstance(loop.target, ast.Name):
        raise ValueError("Source stage induction target is unsupported")
    environment = {"int32": np.int32, "int": int}
    for name, value in function.get("closure_constants", {}).items():
        try:
            environment[name] = restored_scalar(value)
        except ValueError:
            pass
    environment[loop.target.id] = instance["execution_index"]
    for statement in loop.body:
        if statement.lineno >= site["line"]:
            break
        if (isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            value = source.constant(statement.value, environment)
            if value is not source.UNKNOWN:
                environment[statement.targets[0].id] = value
    stage = environment.get("stage_idx")
    if (isinstance(stage, (bool, np.bool_))
            or not isinstance(stage, (int, np.integer))):
        raise ValueError("Source stage index is not a proved integer")
    return int(stage)


def verify_role_invocations(graph):
    """Bind every role invocation to its source caller and full regime."""
    ordered = [item for item in graph["calls"]
               if item.get("kind") == "source_call"]
    calls = {item["context"]: item for item in ordered}
    if len(calls) != len(ordered):
        raise ValueError("Source-call contexts are not unique")
    roles = graph["workload"]["roles"]
    pending = iter(graph["workload"]["step_calls"])
    owners = {}
    actual = Counter()
    expected = Counter()
    for call in graph["workload"]["step_calls"]:
        scenario = graph["scenario_contract"][call["id"]]
        expected[(call["role"], call["id"])] += call["multiplicity"]
        for index in range(len(scenario.get("linear_calls", []))):
            expected[("main_linear", f"{call['id']}.linear{index}")] += 1
    for call in ordered:
        site = call["call_site"]
        parent = None if site is None else owners.get(site["context"])
        matching = [name for name, role in roles.items()
                    if role["function"] == call["function"]]
        if not matching:
            owners[call["context"]] = parent
            continue
        if (parent is not None and parent["role"] == "main_newton"
                and "main_linear" in matching):
            parent_call = calls[site["context"]]
            controls = [item for item in graph["policy_loops"]
                        if item["source"]["context"]
                        == parent_call["context"]
                        and item["kind"] == "recurrent_execution_trace"]
            if len(controls) != 1:
                raise ValueError("Newton call has no exact recurrent owner")
            induction = [item for item in site.get(
                "execution_loop_instances", [])
                if item["policy_loop_id"] == controls[0]["policy_loop_id"]]
            if len(induction) != 1:
                raise ValueError("Linear call lacks its Newton body index")
            index = induction[0]["execution_index"]
            nested = parent["scenario"].get("linear_calls", [])
            if type(index) is not int or not 0 <= index < len(nested):
                raise ValueError("Linear call exceeds the Newton body regime")
            owner = dict(role="main_linear",
                         call=f"{parent['call']}.linear{index}",
                         scenario=nested[index])
        else:
            declared = next(pending, None)
            if declared is None or declared["role"] not in matching:
                raise ValueError("Source role order differs from workload")
            if "stage" in declared and source_call_stage(
                    graph, call, calls) != declared["stage"]:
                raise ValueError("Source stage differs from workload call")
            owner = dict(role=declared["role"], call=declared["id"],
                         scenario=graph["scenario_contract"][declared["id"]])
        owners[call["context"]] = owner
        actual[(owner["role"], owner["call"])] += 1
    if actual != expected or next(pending, None) is not None:
        raise ValueError("Source role invocation coverage differs")

    def check_location(location):
        owner = owners.get(location.get("context"))
        region = location.get("runtime_region")
        if owner is None:
            if region is not None:
                raise ValueError("Runtime region has no source role owner")
            return
        if region is None or any(region.get(key) != expected_value
                                for key, expected_value in (
            ("role", owner["role"]), ("call", owner["call"]),
            ("entry_mask", owner["scenario"]["entry_mask"]),
        )):
            raise ValueError("Runtime region differs from source invocation")

    for call in ordered:
        if call["call_site"] is not None:
            check_location(call["call_site"])
    for item in graph["nodes"] + graph["controls"]:
        check_location(item["source"])
    return owners


def verify_loop_contexts(graph, owners):
    """Bind loop IDs in nodes and induction values to their controls."""
    controls = graph["policy_loops"]
    if [item["policy_loop_id"] for item in controls] != list(
        range(len(controls))
    ):
        raise ValueError("Policy loop IDs differ")
    by_id = {item["policy_loop_id"]: item for item in controls}
    source_calls = {
        item["context"]: item["function"] for item in graph["calls"]
        if item.get("kind") == "source_call"
    }
    for control in controls:
        if control["kind"] != "recurrent_execution_trace":
            continue
        region = control["source"].get("runtime_region", {})
        call = owners.get(control["source"]["context"])
        scenario = None if call is None else call["scenario"]
        role = graph["workload"]["roles"].get(region.get("role"))
        if (
            call is None
            or scenario is None
            or role is None
            or call["role"] != region["role"]
            or scenario["entry_mask"] != region.get("entry_mask")
            or source_calls.get(control["source"]["context"])
            != role["iteration"]["source_region"]["function"]
        ):
            raise ValueError("Recurrent control runtime owner differs")

    def check_instance(record):
        identifier = record.get("policy_loop_id")
        control = by_id.get(identifier)
        if control is None:
            raise ValueError("Execution instance has no policy control")
        recurrent = control["kind"] == "recurrent_execution_trace"
        expected = {
            "policy_loop_id": identifier,
            "line": control["source"]["line"],
            "group": control["group"],
            "part": record.get("part"),
            "lane": record.get("lane"),
            "chunk": record.get("chunk"),
            "execution_index": record.get("execution_index"),
            "codegen_constant": record.get("codegen_constant"),
            "recurrent": recurrent,
            "directive": control["directive"],
        }
        if record != expected:
            raise ValueError("Execution instance differs from policy control")
        matching = [item for item in control["executed_instances"] if (
            item["part"] == record["part"]
            and item["lane"] == record["lane"]
            and item["chunk"] == record["chunk"]
            and item["value"] == record["execution_index"]
            and item["codegen_constant"] == record["codegen_constant"]
        )]
        if len(matching) != 1:
            raise ValueError("Execution instance is outside its control")
        return control

    for node in graph["nodes"]:
        context = node["source"]
        for record in context.get("execution_loop_instances", []):
            control = check_instance(record)
            if (
                control["kind"] == "recurrent_execution_trace"
                and context["context"] == control["source"]["context"]
            ):
                region = context.get("runtime_region", {})
                owner = control["source"].get("runtime_region", {})
                for key in ("role", "call", "entry_mask"):
                    if region.get(key) != owner.get(key):
                        raise ValueError(
                            "Recurrent node runtime owner differs"
                        )
    for value in graph["values"]:
        if value.get("source_origin") != "runtime_loop_induction":
            continue
        control = by_id.get(value.get("policy_loop_id"))
        if control is None:
            raise ValueError("Dynamic induction has no policy control")
        if (
            value["loop_group"] != control["group"]
            or value["loop_line"] != control["source"]["line"]
            or value["recurrent"]
            != (control["kind"] == "recurrent_execution_trace")
        ):
            raise ValueError("Dynamic induction owner differs")
        if value["recurrent"]:
            owner = control["source"]["runtime_region"]
            region = value.get("runtime_region", {})
            for key in ("role", "call", "entry_mask"):
                if region.get(key) != owner.get(key):
                    raise ValueError("Recurrent induction owner differs")


def verify_policy_graph(graph):
    """Rebuild policy, loop, address and selected semantic invariants."""
    if graph.get("kind") != "typed_implicit_execution_region":
        raise ValueError("Policy graph kind differs")
    policy = graph.get("policy")
    if not isinstance(policy, list) or len(policy) != len(GROUPS):
        raise ValueError("Policy record is incomplete")
    for record, group in zip(policy, GROUPS):
        if record.get("group") != group or record.get("level") not in LEVELS:
            raise ValueError("Policy group order or level differs")
        if record.get("flag") != list(LEVELS[record["level"]]):
            raise ValueError("Policy directive identity differs")
    policy_by_group = {item["group"]: item["flag"] for item in policy}
    compilation = graph.get("compilation_check")
    if compilation != {
        "native_overloads": 0,
        "batch_kernel_requested": False,
        "device_function_executed": False,
    }:
        raise ValueError("Policy graph native-construction boundary differs")
    verify_function_sources(
        graph["provenance"]["functions"], policy_by_group
    )
    verify_workload_policy(graph["workload"], policy_by_group)
    scenarios = graph.get("scenario_contract")
    if not isinstance(scenarios, dict):
        raise ValueError("Policy graph lacks an exact scenario contract")
    expected_regime = workload.evaluate_regime(
        graph["workload"], scenarios,
        step_entry_mask=graph["regime"]["step_entry_mask"],
    )
    if graph["regime"] != expected_regime:
        raise ValueError("Policy graph regime differs from scenarios")
    retained_controls = [
        control for control in graph["controls"]
        if control.get("kind") in {
            "policy_fixed_execution_trace", "recurrent_execution_trace"
        }
    ]
    if retained_controls != graph.get("policy_loops"):
        raise ValueError("Policy loop controls differ from control stream")
    role_owners = verify_role_invocations(graph)
    verify_loop_contexts(graph, role_owners)
    explicit_choices = {}
    for control in graph["controls"]:
        if (
            control.get("kind") != "runtime_branch_choice"
            or control.get("reason") != "explicit runtime-path assumption"
        ):
            continue
        key = (
            f"{Path(control['source']['path']).name}:"
            f"{control['source']['line']}"
        )
        choice = control["choice"]
        if key in explicit_choices and explicit_choices[key] != choice:
            raise ValueError("Explicit branch choices conflict")
        explicit_choices[key] = choice
    if graph.get("branch_choices") != explicit_choices:
        raise ValueError("Explicit branch choices differ from controls")
    values, nodes = graph["values"], graph["nodes"]
    if [value["id"] for value in values] != list(range(len(values))):
        raise ValueError("Source value IDs differ")
    if [node["id"] for node in nodes] != list(range(len(nodes))):
        raise ValueError("Source node IDs differ")
    dynamic = {
        value["id"]: value for value in values
        if value.get("source_origin") == "runtime_loop_induction"
    }
    for value in dynamic.values():
        if (
            value["kind"] != "live_in"
            or "constant" in value
            or value.get("trace_value_is_not_codegen_constant") is not True
            or value.get("external_kernel_input") is not False
            or "declared_trace_value" not in value
        ):
            raise ValueError("Dynamic induction became a source constant")
    expected_dynamic = []
    for control in graph.get("policy_loops", []):
        for instance in control["executed_instances"]:
            if instance["codegen_constant"]:
                continue
            expected_dynamic.append(dict(
                policy_loop_id=control["policy_loop_id"],
                loop_group=control["group"],
                loop_line=control["source"]["line"],
                loop_part=instance["part"],
                template_lane=instance["lane"],
                execution_position=instance["position"],
                recurrent=control["kind"] == "recurrent_execution_trace",
                declared_trace_value=source.snapshot(
                    np.int32(instance["value"])
                ),
            ))
    dynamic_fields = (
        "policy_loop_id", "loop_group", "loop_line", "loop_part",
        "template_lane", "execution_position", "recurrent",
        "declared_trace_value",
    )
    actual_dynamic = [
        {key: value[key] for key in dynamic_fields}
        for value in dynamic.values()
    ]
    if Counter(payload(item) for item in actual_dynamic) != Counter(
        payload(item) for item in expected_dynamic
    ):
        raise ValueError(
            "Dynamic induction witnesses differ from loop instances"
        )
    expected_address = []
    for node in nodes:
        address = node.get("address_value_ids", [])
        if any(
            values[value].get(
                "trace_value_is_not_codegen_constant"
            ) is not True
            or values[value]["dtype"] != "int32"
            or "constant" in values[value]
            for value in address
        ):
            raise ValueError("Address dependency is not a dynamic trace index")
        expected_address.extend(
            dict(value=value, access_node=node["id"]) for value in address
        )
    if graph.get("address_edges") != expected_address:
        raise ValueError("Dynamic address edges differ")
    for node in nodes:
        if node["kind"] != "CapturedIndexRead":
            continue
        if (
            len(node["outputs"]) != 1
            or node.get("is_codegen_constant") is not False
            or node["inputs"] != template_values(node["index_template"])
            or any(values[value]["dtype"] != "int32"
                   or "constant" in values[value]
                   or values[value].get(
                       "trace_value_is_not_codegen_constant"
                   ) is not True for value in node["inputs"])
        ):
            raise ValueError("Dynamic captured-index operands differ")
        captured = node["captured"]
        if captured.get("kind") != "array":
            raise ValueError("Dynamic captured source is not an array")
        array = np.asarray(captured["values"], dtype=captured["dtype"])
        if (
            list(array.shape) != captured["shape"]
            or hashlib.sha256(array.tobytes(order="C")).hexdigest()
            != captured["sha256"]
        ):
            raise ValueError("Dynamic captured array snapshot differs")
        declared = {
            value: restored_scalar(values[value]["declared_trace_value"])
            for value in node["inputs"]
        }
        selected_indices = resolve_index_template(
            node["index_template"], declared
        )
        if selected_indices != node["selected_execution_index"]:
            raise ValueError("Captured-index execution witness differs")
        selected = captured_selection(
            node["captured"], selected_indices
        )
        output = values[node["outputs"][0]]
        if (
            selected != node["declared_trace_result"]
            or output.get("declared_trace_value") != selected
            or output.get("trace_value_is_not_codegen_constant") is not True
            or "constant" in output
        ):
            raise ValueError("Dynamic captured-index trace result differs")
    trees = {}
    for control in graph.get("policy_loops", []):
        if control["directive"] != policy_by_group[control["group"]]:
            raise ValueError("Loop closure directive differs from policy")
        structure = control["structure"]
        values_ = source_loop_values(graph, control, trees)
        expected = loop_structure(values_, control["directive"])
        if structure != expected:
            raise ValueError("Static main/tail loop structure differs")
        if control.get("code_copies_are_execution_count") is not False:
            raise ValueError("Execution trace was labeled as code copies")
        instances = expected["execution_instances"]
        if control["kind"] == "recurrent_execution_trace":
            owner = role_owners[control["source"]["context"]]
            if owner["role"] == "main_newton":
                call = graph["regime"]["calls"].get(owner["call"])
            else:
                call = workload.linear_regime(
                    graph["workload"]["roles"][owner["role"]],
                    owner["scenario"],
                )
            if call is None or not isinstance(
                call.get("loop_top_votes"), int
            ):
                raise ValueError(
                    "Recurrent loop lacks an exact admitted vote count"
                )
            instances = instances[:call["loop_top_votes"]]
        elif control["kind"] != "policy_fixed_execution_trace":
            raise ValueError("Policy loop control kind differs")
        if control["executed_instances"] != instances:
            raise ValueError("Executed loop instances differ")
        if control["indices"] != [item["value"] for item in instances]:
            raise ValueError("Executed loop indices differ")
    functions = {
        function["id"]: function
        for function in graph["provenance"]["functions"]
    }
    for node in nodes:
        context = node["source"]
        if "hot_template" not in context:
            continue
        if context.get("template_is_not_native_copy_identity") is not True:
            raise ValueError("Source template was labeled as a native copy")
        template_function = context["hot_template"]["function"]
        function = functions.get(template_function["function"])
        expected_source = None if function is None else {
            "path": function["source"]["source_path"],
            "sha256": function["source"]["source_sha256"],
        }
        if template_function["source"] != expected_source:
            raise ValueError("Hot template function provenance differs")
        identity = hashlib.sha256(json.dumps(
            context["hot_template"], sort_keys=True
        ).encode()).hexdigest()
        if identity != context["hot_template_identity"]:
            raise ValueError("Static hot-template identity differs")
    edges = []
    for node in nodes:
        for value in node["inputs"]:
            producer = values[value]["producer"]
            if producer is not None:
                edges.append(dict(
                    producer=producer, consumer=node["id"], value=value
                ))
    if graph["value_edges"] != edges:
        raise ValueError("Source value edges differ")
    for receipt in [graph["provenance"]["extractor"]] + graph[
        "provenance"
    ]["dependencies"]:
        if hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest() != (
            receipt["sha256"]
        ):
            raise ValueError("Policy graph dependency bytes changed")
    symbolic = semantic_certificate(graph)
    numerical = numeric_semantic_certificates(graph)
    return dict(
        status="POLICY_SOURCE_PASS",
        loops=len(graph["policy_loops"]),
        dynamic_inductions=len(dynamic),
        address_edges=len(expected_address),
        symbolic_semantic_certificate=symbolic,
        numeric_semantic_certificates=numerical,
    )


def verify_policy_cohort(graphs):
    """Require comparable semantics and distinct directive identities."""
    if len(graphs) < 2:
        raise ValueError("Policy cohort needs at least two candidates")
    checks = [verify_policy_graph(graph) for graph in graphs]
    semantics = [check["numeric_semantic_certificates"] for check in checks]
    if any(item != semantics[0] for item in semantics[1:]):
        raise ValueError("Policy candidates changed selected source semantics")
    identities = [payload(graph["policy"]) for graph in graphs]
    if len(set(identities)) != len(identities):
        raise ValueError("Policy candidates do not have distinct directives")
    return dict(
        status="POLICY_COHORT_PASS",
        candidates=len(graphs),
        numeric_semantic_certificates=semantics[0],
        symbolic_shapes_may_differ=True,
    )


def make_policy_plan(graph, architecture, compiler, materialization):
    """Connect an exact policy graph to the verified typed lowerer."""
    checked = verify_policy_graph(graph)
    if any(node["kind"] == "CapturedIndexRead" for node in graph["nodes"]):
        plan = special_typed_plan(
            graph, architecture, compiler, materialization
        )
    else:
        plan = native.make_plan(
            graph, architecture, compiler, materialization
        )
    return dict(
        schema=1,
        kind="conditional_implicit_policy_native_plan",
        provenance=dict(
            adapter=source_receipt(make_policy_plan),
            typed_lowerer=source_receipt(native.make_plan),
        ),
        policy=graph["policy"],
        policy_verification=checked,
        loop_model=[control["structure"] | {
            "group": control["group"],
            "source": control["source"],
        } for control in graph["policy_loops"]],
        dynamic_address_edges=graph["address_edges"],
        typed_plan=plan,
        interpretation=dict(
            static_templates="canonical source templates, not native copies",
            execution="exact selected source execution instances",
            address=(
                "trace-cell dependency; native address arithmetic remains "
                "unmodeled by the reused typed lowerer"
            ),
        ),
        native_labels_consumed=False,
        timings_consumed=False,
        fitted_parameters=False,
    )


def verify_policy_plan(graph, plan):
    """Rebuild an entire policy plan from its admitted public inputs."""
    typed = plan.get("typed_plan", {})
    lowering = typed.get("lowering", {})
    expected = make_policy_plan(
        graph,
        typed.get("architecture"),
        typed.get("compiler_alternative"),
        lowering.get("materialization"),
    )
    if plan != expected:
        raise ValueError("Policy native plan differs from exact rebuild")
    return dict(
        status="POLICY_PLAN_PASS",
        graph_nodes=len(graph["nodes"]),
        typed_nodes=len(lowering["nodes"]),
        exact_rebuild=True,
    )


def main():
    """Construct a source-only policy graph and optional typed plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="lorenz")
    parser.add_argument("--algo", default="kvaerno3")
    parser.add_argument("--linear-solver", default="lu")
    parser.add_argument("--newton-bodies", type=int, default=1)
    parser.add_argument("--krylov-bodies", type=int, default=1)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--branch-choices", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--materialization", default="promote",
                        choices=("promote", "addressable"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.architecture is None) != (args.compiler is None):
        raise ValueError("Architecture and compiler must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / "codegen"
    cache.mkdir()
    previous = get_cache_root_override()
    solver = None
    try:
        set_cache_root(cache.resolve())
        levels = parse_policy(args.policy)
        flags = policy_flags(levels)
        policy = policy_record(levels, flags)
        system = placement.SYSTEMS[args.system]["build"]()
        kwargs = placement.solver_kwargs(args.system, args.algo)
        kwargs["linear_correction_type"] = workload.PUBLIC_LINEAR_TYPES[
            args.linear_solver
        ]
        kwargs["unroll"] = flags
        solver = Solver(system, **kwargs)
        descriptor = workload.describe_implicit_workload(solver)
        scenarios = implicit.uniform_regime(
            descriptor, args.newton_bodies, args.krylov_bodies
        )
        choices = {} if args.branch_choices is None else json.loads(
            args.branch_choices.read_text()
        )
        result = describe_policy_source(solver, scenarios, policy, choices)
        workload.write_json(args.output / "graph.json", result)
        plan = None
        if args.architecture is not None:
            architecture = json.loads(args.architecture.read_text())
            compiler = json.loads(args.compiler.read_text())
            plan = make_policy_plan(
                result, architecture, compiler, args.materialization
            )
            verify_policy_plan(result, plan)
            workload.write_json(args.output / "plan.json", plan)
        receipt = dict(
            status="IMPLICIT_POLICY_GRAPH_PASS",
            graph_sha256=hashlib.sha256(
                (args.output / "graph.json").read_bytes()
            ).hexdigest(),
            plan_sha256=(None if plan is None else hashlib.sha256(
                (args.output / "plan.json").read_bytes()
            ).hexdigest()),
            policy=policy,
            nodes=len(result["nodes"]),
            dynamic_inductions=sum(
                value.get("source_origin") == "runtime_loop_induction"
                for value in result["values"]
            ),
            native_overloads=0,
            native_compilations=0,
            kernel_launches=0,
        )
        workload.write_json(args.output / "receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
    finally:
        if solver is not None:
            solver.close()
        set_cache_root(previous)


if __name__ == "__main__":
    main()
