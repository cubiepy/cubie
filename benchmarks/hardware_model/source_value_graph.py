"""Certify typed source-value frontiers for captured, fully expanded ERK.

This is a source semantics certificate under explicit scalar replacement
and no-rematerialization assumptions. It does not predict registers.
Device functions are inspected, never executed or specialized.
"""

import argparse
import ast
from collections import Counter
import heapq
import inspect
import json
from pathlib import Path

import numpy as np

from cubie.buffer_registry import buffer_registry
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_simsafe import float32, int32, unroll_if

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model.buffer_descriptors import (
    BufferEffects, interval, overlaps, registry_layout, view,
)
from benchmarks.hardware_model.expansion import (
    CapturedGraph, constant, snapshot, source_receipt,
)
from benchmarks.hardware_model.workload import (
    UNKNOWN, describe_workload, python_function, source_function,
)


SCRIPT = Path(__file__).resolve()


class Unsupported(ValueError):
    """The selected source region has no complete certificate."""


def item(raw=UNKNOWN, reference=None, identity=None, ordering=()):
    """Keep raw constants, byte aliases and scalar values distinct."""
    return dict(raw=raw, reference=reference, identity=identity,
                ordering=set(ordering))


def scalar_type(raw):
    """Name a source type without host numeric promotion."""
    if isinstance(raw, np.generic):
        return raw.dtype.name
    return {bool: "bool", int: "literal_int", float: "literal_float",
            type(None): "none"}.get(type(raw))


class SourceValues:
    """Interpret supported original ASTs and retain source value edges."""

    slice_view = BufferEffects.slice_view

    def __init__(self, graph):
        self.graph = graph
        self.nodes = []
        self.values = []
        self.cells = {}
        self.accesses = {}
        self.allocations = {}
        self.calls = []
        self.controls = []
        self.aliases = []
        self.context = "caller"
        self.path = None
        self.loop_indices = []
        self.stack = []
        self.boundary = set()
        self.input_ids = []
        self.captured_roots = []
        self.function_ids = {id(v): k for k, v in graph.callables.items()}

    def captured_view(self, raw):
        if not isinstance(raw, np.ndarray):
            return None
        root = raw
        while isinstance(root.base, np.ndarray):
            root = root.base
        index = next((i for i, existing in enumerate(self.captured_roots)
                      if existing is root), None)
        if index is None:
            index = len(self.captured_roots)
            self.captured_roots.append(root)
        return dict(root=index, dtype=raw.dtype.name, shape=list(raw.shape),
                    strides=list(raw.strides),
                    byte_offset=int(raw.ctypes.data) - int(root.ctypes.data),
                    root_snapshot=snapshot(root))

    def unknown(self, node, reason, refs=()):
        raise Unsupported(f"{self.path}:{node.lineno}: {reason}: "
                          f"{ast.unparse(node)}")

    def location(self, node):
        return dict(path=self.path, line=getattr(node, "lineno", None),
                    context=self.context,
                    loop_indices=[dict(x) for x in self.loop_indices],
                    syntax=ast.unparse(node) if node is not None else None)

    def make_value(self, dtype, producer, kind, raw=UNKNOWN, **details):
        identifier = len(self.values)
        record = dict(id=identifier, dtype=dtype, producer=producer,
                      kind=kind, **details)
        if raw is not UNKNOWN:
            record["constant"] = snapshot(raw)
        self.values.append(record)
        return item(raw, identity=identifier)

    def scalar(self, value, node):
        if value["reference"] is not None:
            self.unknown(node, "Array object is not a scalar result")
        if value["identity"] is None:
            dtype = scalar_type(value["raw"])
            if dtype is None:
                self.unknown(node, "Scalar type is unresolved")
            return self.make_value(dtype, None, "constant", value["raw"],
                                   source=self.location(node))
        return value

    def operation(self, kind, operands, node, dtype=None, raw=UNKNOWN,
                  ordering=(), **details):
        operands = [self.scalar(x, node) for x in operands]
        identifier = len(self.nodes)
        order = set(ordering)
        for operand in operands:
            order.update(operand["ordering"])
        result = item(raw)
        if dtype is not None:
            result = self.make_value(dtype, identifier, "expression", raw,
                                     source=self.location(node))
        self.nodes.append(dict(
            id=identifier, kind=kind,
            inputs=[x["identity"] for x in operands],
            outputs=[] if dtype is None else [result["identity"]],
            order_predecessors=sorted(order), source=self.location(node),
            **details,
        ))
        return result, identifier

    def live_in(self, dtype, label):
        result = self.make_value(dtype, None, "live_in", label=label)
        self.input_ids.append(result["identity"])
        return result

    def raw_environment(self, environment):
        return {key: value["raw"] for key, value in environment.items()}

    def condition(self, node, environment):
        raw = self.raw_environment(environment)
        result = constant(node, raw)
        if (result is UNKNOWN and isinstance(node, ast.Compare) and
                len(node.ops) == 1 and
                isinstance(node.ops[0], (ast.Is, ast.IsNot)) and
                isinstance(node.comparators[0], ast.Constant) and
                node.comparators[0].value is None):
            left = constant(node.left, raw)
            if left is not UNKNOWN:
                return ((left is None) if isinstance(node.ops[0], ast.Is)
                        else (left is not None))
        return result

    def index(self, node, environment):
        if isinstance(node, ast.Slice):
            parts = [None if x is None else self.index(x, environment)
                     for x in (node.lower, node.upper, node.step)]
            return slice(*parts)
        if isinstance(node, ast.Tuple):
            return tuple(self.index(x, environment) for x in node.elts)
        result = constant(node, self.raw_environment(environment))
        if isinstance(result, (bool, np.bool_)):
            self.unknown(node, "Boolean array indices are unsupported")
        if not isinstance(result, (int, np.integer)):
            self.unknown(node, "Index is not a proved integer constant")
        return int(result)

    def cell(self, reference, node):
        extent = interval(reference)
        if (extent is None or reference["shape"] != [] or
                reference["bytes"] != reference["itemsize"]):
            self.unknown(node, "Scalar byte extent is unresolved")
        key = (reference["storage"], *extent, reference["dtype"])
        for other in self.cells:
            if (other[0] == key[0] and other != key and
                    overlaps(other[1:3], key[1:3])):
                self.unknown(node, "Overlapping differently typed cells")
        return key

    def read(self, reference, node):
        key = self.cell(reference, node)
        if key not in self.cells:
            if key[0] not in self.boundary:
                self.unknown(node, "Read of uninitialized private element")
            self.cells[key] = self.live_in(reference["dtype"], list(key))
        result = self.cells[key]
        previous = self.accesses.get(key, dict(write=None, reads=[]))
        order = [] if previous["write"] is None else [previous["write"]]
        _, identifier = self.operation(
            "element_read_alias", [result], node, ordering=order,
            cell=list(key),
        )
        previous["reads"].append(identifier)
        self.accesses[key] = previous
        return dict(result, ordering={identifier})

    def write(self, reference, result, node):
        key = self.cell(reference, node)
        result = self.scalar(result, node)
        actual_type = self.values[result["identity"]]["dtype"]
        if actual_type != reference["dtype"]:
            self.unknown(node, f"Unproved store conversion {actual_type} to "
                         f"{reference['dtype']}")
        previous = self.accesses.get(key, dict(write=None, reads=[]))
        order = previous["reads"] + (
            [] if previous["write"] is None else [previous["write"]]
        )
        _, identifier = self.operation(
            "element_write_alias", [result], node, ordering=order,
            cell=list(key),
        )
        self.cells[key] = dict(result, ordering=set())
        self.accesses[key] = dict(write=identifier, reads=[])

    def binary(self, node, left, right):
        folded = constant(ast.BinOp(
            left=ast.Name(id="left", ctx=ast.Load()), op=node.op,
            right=ast.Name(id="right", ctx=ast.Load()),
        ), {"left": left["raw"], "right": right["raw"]})
        if isinstance(folded, (int, np.integer)):
            return item(folded)
        operands = [self.scalar(x, node) for x in (left, right)]
        types = [self.values[x["identity"]]["dtype"] for x in operands]
        if types[0] != types[1] or types[0] not in ("float32", "int32"):
            self.unknown(node, f"Unproved numeric promotion {types}")
        allowed = (ast.Add, ast.Sub, ast.Mult, ast.Div)
        if types[0] == "int32":
            allowed = (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr)
        if not isinstance(node.op, allowed):
            self.unknown(node, "Unproved operator result type")
        return self.operation(type(node.op).__name__, operands, node,
                              dtype=types[0])[0]

    def expression(self, node, environment):
        if node is None:
            return item(None)
        if isinstance(node, ast.Name):
            if node.id not in environment:
                self.unknown(node, "Unbound source name")
            return environment[node.id]
        if isinstance(node, ast.Constant):
            return item(node.value)
        if isinstance(node, ast.Attribute):
            parent = self.expression(node.value, environment)
            if parent["raw"] is UNKNOWN or parent["reference"]:
                self.unknown(node, "Unknown attribute receiver")
            return item(getattr(parent["raw"], node.attr))
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value, environment)
            index = self.index(node.slice, environment)
            if base["reference"] is not None:
                ref = self.slice_view(base["reference"], index, node)
                if ref["shape"]:
                    return item(reference=ref)
                return self.read(ref, node)
            if isinstance(base["raw"], (tuple, np.ndarray)):
                return item(base["raw"][index])
            self.unknown(node, "Unknown indexed object")
        if isinstance(node, ast.BinOp):
            folded = constant(node, self.raw_environment(environment))
            if isinstance(folded, (int, np.integer)):
                return item(folded)
            return self.binary(node, self.expression(node.left, environment),
                               self.expression(node.right, environment))
        if isinstance(node, ast.UnaryOp):
            operand = self.scalar(self.expression(node.operand, environment),
                                  node)
            dtype = self.values[operand["identity"]]["dtype"]
            if isinstance(node.op, ast.Not):
                if dtype not in ("bool", "float32", "int32"):
                    self.unknown(node, "Unproved truth-value conversion")
                dtype = "bool"
            elif dtype not in ("float32", "int32"):
                folded = constant(node, self.raw_environment(environment))
                if folded is not UNKNOWN:
                    return item(folded)
                self.unknown(node, "Unproved unary numeric type")
            elif not (isinstance(node.op, (ast.UAdd, ast.USub)) or
                      dtype == "int32" and isinstance(node.op, ast.Invert)):
                self.unknown(node, "Unsupported unary operator/type pair")
            return self.operation(type(node.op).__name__, [operand], node,
                                  dtype=dtype)[0]
        if isinstance(node, (ast.Compare, ast.BoolOp, ast.IfExp)):
            folded = constant(node, self.raw_environment(environment))
            if folded is UNKNOWN:
                self.unknown(node, "Runtime predicate needs a region graph")
            return item(folded)
        if isinstance(node, ast.Call):
            return self.call(node, environment)
        self.unknown(node, "Unsupported scalar expression")

    def allocate(self, function, bound, node, boundary=False):
        closure = inspect.getclosurevars(function).nonlocals
        dtype = np.dtype(closure["_dtype"])
        if closure["_use_shared"] or closure["_use_persistent"]:
            parameter = "shared" if closure["_use_shared"] else "persistent"
            parent = bound.arguments[parameter]["reference"]
            if parent is None:
                self.unknown(node, "Allocator parent is not a byte view")
            ref = self.slice_view(parent, closure[f"_{parameter}_slice"],
                                  node)
            ref.update(dtype=dtype.name, itemsize=dtype.itemsize,
                       shape=[ref["bytes"] // dtype.itemsize])
        else:
            count = int(closure["_local_size"])
            storage = f"local:{self.context}:{node.lineno}:{len(self.calls)}"
            ref = view(storage, 0, count * dtype.itemsize, dtype.itemsize,
                       dtype.name)
        storage = ref["storage"]
        self.allocations.setdefault(storage, dict(
            storage=storage, source=self.location(node), view=ref,
            boundary=boundary, elements=int(closure["elements"]),
            local_elements=int(closure["_local_size"]),
        ))
        if boundary:
            self.boundary.add(storage)
        self.calls.append(dict(
            kind="allocator", source=self.location(node),
            function_source=source_receipt(function), view=ref,
            zero=bool(closure["_zero"]), boundary_binding=boundary,
        ))
        if closure["_zero"] and not boundary:
            for index in range(ref["shape"][0]):
                self.write(self.slice_view(ref, index, node),
                           item(dtype.type(0)), node)
        return item(reference=ref)

    def call(self, node, environment):
        target = self.expression(node.func, environment)["raw"]
        args = [self.expression(x, environment) for x in node.args]
        kwargs = {x.arg: self.expression(x.value, environment)
                  for x in node.keywords}
        if any(x.arg is None for x in node.keywords):
            self.unknown(node, "Expanded keyword arguments")
        dtype = None
        if target is np.float32 or target is float32:
            dtype = "float32"
        elif target is np.int32 or target is int32:
            dtype = "int32"
        if dtype is not None and len(args) == 1 and not kwargs:
            raw = constant(node, self.raw_environment(environment))
            return self.operation("cast", args, node, dtype=dtype,
                                  raw=raw)[0]
        function = python_function(target)
        if function is None:
            self.unknown(node, "Unknown call effect or numeric type")
        bound = inspect.signature(function).bind(*args, **kwargs)
        if (function.__name__ == "allocate_buffer" and
                Path(inspect.getsourcefile(function)).name ==
                "buffer_registry.py"):
            return self.allocate(function, bound, node)
        if id(function) not in self.function_ids:
            self.unknown(node, "Call is outside captured helper closure")
        return self.invoke(function, bound.arguments, self.location(node))

    def assign(self, target, result, environment):
        if isinstance(target, ast.Name):
            environment[target.id] = result
            self.aliases.append(dict(source=self.location(target),
                                     name=target.id,
                                     value=result["identity"],
                                     view=result["reference"],
                                     captured_view=self.captured_view(
                                         result["raw"])))
        elif isinstance(target, ast.Subscript):
            base = self.expression(target.value, environment)
            if base["reference"] is None:
                self.unknown(target, "Write to captured or unknown array")
            ref = self.slice_view(base["reference"],
                                  self.index(target.slice, environment),
                                  target)
            self.write(ref, result, target)
        else:
            self.unknown(target, "Unsupported assignment target")

    def block(self, statements, environment):
        for node in statements:
            if isinstance(node, ast.Assign):
                result = self.expression(node.value, environment)
                for target in node.targets:
                    self.assign(target, result, environment)
            elif isinstance(node, ast.AugAssign):
                result = self.binary(
                    node, self.expression(node.target, environment),
                    self.expression(node.value, environment),
                )
                self.assign(node.target, result, environment)
            elif isinstance(node, ast.If):
                condition = self.condition(node.test, environment)
                if condition is UNKNOWN:
                    self.unknown(node.test, "Unresolved branch condition")
                self.controls.append(dict(kind="selected_branch",
                                          source=self.location(node),
                                          condition=bool(condition)))
                returned, result = self.block(
                    node.body if condition else node.orelse, environment,
                )
                if returned:
                    return returned, result
            elif isinstance(node, ast.For):
                iterable = node.iter
                if (not isinstance(iterable, ast.Call) or
                        self.expression(iterable.func, environment)["raw"]
                        is not unroll_if or len(iterable.args) != 2):
                    self.unknown(node, "Loop lacks a captured full directive")
                flag = constant(iterable.args[1],
                                self.raw_environment(environment))
                if flag != (True, None):
                    self.unknown(node, "Loop directive is not full expansion")
                bounds = iterable.args[0]
                if (not isinstance(bounds, ast.Call) or
                        self.expression(bounds.func, environment)["raw"]
                        is not range or bounds.keywords):
                    self.unknown(node, "Loop is not a literal bound range")
                indices = list(range(*(self.index(x, environment)
                                       for x in bounds.args)))
                if len(indices) > 10000 or node.orelse:
                    self.unknown(node, "Loop exceeds supported expansion")
                self.controls.append(dict(kind="full_source_expansion",
                                          source=self.location(node),
                                          indices=indices, flag=list(flag)))
                for index in indices:
                    self.assign(node.target, item(index), environment)
                    self.loop_indices.append(dict(line=node.lineno,
                                                  index=index))
                    returned, result = self.block(node.body, environment)
                    self.loop_indices.pop()
                    if returned:
                        return returned, result
            elif isinstance(node, ast.Expr):
                if not isinstance(node.value, ast.Constant):
                    self.expression(node.value, environment)
            elif isinstance(node, ast.Return):
                result = self.expression(node.value, environment)
                if result["raw"] is not None:
                    result = self.scalar(result, node)
                return True, result
            elif not isinstance(node, ast.Pass):
                self.unknown(node, "Unsupported statement or residual loop")
        return False, item(None)

    def invoke(self, function, bound, call_site=None):
        if id(function) in self.stack:
            raise Unsupported("Recursive helper call")
        closure = inspect.getclosurevars(function)
        environment = {k: item(v) for k, v in dict(
            closure.builtins, **closure.globals, **closure.nonlocals,
        ).items()}
        environment.update(bound)
        previous = self.context, self.path
        self.context = f"call{len(self.calls)}:{function.__qualname__}"
        self.path = inspect.getsourcefile(function)
        record = dict(kind="source_call", context=self.context,
                      function=self.function_ids[id(function)],
                      source=source_receipt(function), call_site=call_site,
                      first_node=len(self.nodes),
                      bindings={k: dict(value=v["identity"],
                                        view=v["reference"])
                                for k, v in bound.items()})
        record["closure_constants"] = {
            key: snapshot(value) for key, value in closure.nonlocals.items()
            if snapshot(value) is not UNKNOWN
        }
        record["closure_array_views"] = {
            key: self.captured_view(value)
            for key, value in closure.nonlocals.items()
            if isinstance(value, np.ndarray)
        }
        self.calls.append(record)
        self.stack.append(id(function))
        _, result = self.block(source_function(function).body, environment)
        self.stack.pop()
        record["end_node"] = len(self.nodes)
        self.context, self.path = previous
        return result


def caller_bindings(engine, solver):
    """Bind the actual ODE-loop step call and captured buffer allocators."""
    integrator = solver.kernel.single_integrator
    function = python_function(integrator._loop.device_function)
    node = source_function(function)
    closure = inspect.getclosurevars(function)
    raw = dict(closure.builtins, **closure.globals, **closure.nonlocals)
    environment = {k: item(v) for k, v in raw.items()}
    engine.path = inspect.getsourcefile(function)
    group = buffer_registry._groups[integrator._loop]
    dtype = np.dtype(group.parent_dtype)
    if dtype.name != "float32" or raw["precision"] is not float32:
        raise Unsupported("Caller precision is not the FP32 contract")
    for name, count in (
        ("shared_scratch", group.shared_buffer_size()),
        ("persistent_local", group.persistent_local_buffer_size()),
    ):
        environment[name] = item(reference=view(
            "caller:" + name, 0, int(count) * dtype.itemsize,
            dtype.itemsize, dtype.name,
        ))
        engine.boundary.add("caller:" + name)
    allocation_assignments = []
    for statement in node.body:
        if (isinstance(statement, ast.Assign) and
                isinstance(statement.value, ast.Call) and
                isinstance(statement.value.func, ast.Name) and
                statement.value.func.id.startswith("alloc_")):
            target = python_function(raw[statement.value.func.id])
            args = [engine.expression(x, environment)
                    for x in statement.value.args]
            bound = inspect.signature(target).bind(*args)
            result = engine.allocate(target, bound, statement.value,
                                     boundary=True)
            for assignment in statement.targets:
                engine.assign(assignment, result, environment)
            allocation_assignments.append(statement.lineno)
    call_nodes = [x for x in ast.walk(node)
                  if isinstance(x, ast.Call) and
                  isinstance(x.func, ast.Name) and
                  x.func.id == "step_function"]
    if len(call_nodes) != 1:
        raise Unsupported("Caller does not contain exactly one step call")
    call = call_nodes[0]
    # These are the actual caller's narrowed precision and boolean flags;
    # arbitrary step-entry contents are not loop-entry initialization.
    for name in ("dt_eff", "t_prec"):
        environment[name] = engine.live_in("float32", "caller:" + name)
    for name in ("first_step_flag", "prev_step_accepted_flag"):
        environment[name] = engine.live_in("bool", "caller:" + name)
    environment["driver_coefficients"] = item(reference=view(
        "caller:driver_coefficients", 0, None, 4, "float32", shape=None,
    ))
    engine.boundary.add("caller:driver_coefficients")
    step = python_function(integrator._algo_step.step_function)
    args = [engine.expression(x, environment) for x in call.args]
    bound = inspect.signature(step).bind(*args)
    return step, bound.arguments, dict(
        source=source_receipt(function), step_call_line=call.lineno,
        step_call=ast.unparse(call), allocator_lines=allocation_assignments,
        boundary="arbitrary invocation; contents are symbolic live-ins",
        scalar_type_contract=dict(
            dt_eff="float32 narrowed effective timestep",
            t_prec="float32 narrowed time", first_step_flag="bool",
            prev_step_accepted_flag="bool",
        ),
    )


def graph_certificate(nodes, values, live_ins, outputs, max_states=50000):
    """Return complete legal witnesses and a bounded exact minimax search.

    The objective counts distinct retained FP32 semantic values after
    operations. Other types have separate counts. Constants are explicit
    values but are not retained mutable values. No hardware units enter.
    """
    if type(max_states) is not int or max_states < 1:
        raise ValueError("The exact-search state budget must be positive")
    identifiers = {node["id"]: index for index, node in enumerate(nodes)}
    value_map = {value["id"]: value for value in values}
    producers = {key: identifiers.get(value["producer"])
                 for key, value in value_map.items()}
    consumers = {key: 0 for key in value_map}
    dependencies = []
    for index, node in enumerate(nodes):
        previous = {identifiers[x] for x in node["order_predecessors"]
                    if x in identifiers}
        for key in node["inputs"]:
            consumers[key] |= 1 << index
            if producers[key] is not None:
                previous.add(producers[key])
        dependencies.append(sum(1 << x for x in previous))
    complete = (1 << len(nodes)) - 1
    boundary_outputs = set(outputs)
    live_ins = set(live_ins)

    def frontier(mask):
        retained = []
        for key, value in value_map.items():
            if value["kind"] == "constant":
                continue
            producer = producers[key]
            available = key in live_ins or (
                producer is not None and mask & (1 << producer)
            )
            if available and (consumers[key] & ~mask or
                              key in boundary_outputs):
                retained.append(key)
        counts = Counter(value_map[key]["dtype"] for key in retained)
        return retained, dict(counts)

    def witness(schedule):
        mask = 0
        rows = []
        peak = Counter()
        remaining = {key: users.bit_count()
                     for key, users in consumers.items()}
        retained = {key for key in live_ins
                    if value_map[key]["kind"] != "constant" and
                    (remaining[key] or key in boundary_outputs)}
        for index in [None] + list(schedule):
            if index is not None:
                bit = 1 << index
                if mask & bit or dependencies[index] & ~mask:
                    raise ValueError("Invalid schedule witness")
                mask |= bit
                for key in set(nodes[index]["inputs"]):
                    remaining[key] -= 1
                    if not remaining[key] and key not in boundary_outputs:
                        retained.discard(key)
                for key in nodes[index]["outputs"]:
                    if remaining[key] or key in boundary_outputs:
                        retained.add(key)
            counts = dict(Counter(value_map[key]["dtype"]
                                  for key in retained))
            peak |= Counter(counts)
            rows.append(dict(after=None if index is None else
                             nodes[index]["id"], values=sorted(retained),
                             typed_counts=counts))
        if mask != complete:
            raise ValueError("Incomplete schedule witness")
        return dict(schedule=[nodes[i]["id"] for i in schedule],
                    frontiers=rows, peak_by_type=dict(peak),
                    objective_peak=peak["float32"])

    source = witness(range(len(nodes)))
    initial = frontier(0)[1].get("float32", 0)
    best = {0: initial}
    parent = {}
    queue = [(initial, 0, 0)]
    expanded = 0
    solution = None
    while queue and expanded < max_states:
        cost, _, mask = heapq.heappop(queue)
        if best[mask] != cost:
            continue
        expanded += 1
        if mask == complete:
            solution = []
            while mask:
                mask, index = parent[mask]
                solution.append(index)
            solution.reverse()
            break
        if expanded >= max_states:
            break
        for index, dependency in enumerate(dependencies):
            bit = 1 << index
            if mask & bit or dependency & ~mask:
                continue
            after = mask | bit
            candidate = max(cost, frontier(after)[1].get("float32", 0))
            if candidate < best.get(after, float("inf")):
                best[after] = candidate
                parent[after] = mask, index
                heapq.heappush(queue, (candidate, -after.bit_count(), after))
    return dict(
        source_schedule=source,
        exact_search=dict(
            objective="minimize maximum boundary FP32 semantic-value count",
            status="optimum_proved" if solution is not None else
            "state_limit_no_optimum_claim",
            algorithm="Dijkstra bottleneck search over dependency ideals",
            max_expanded_states=max_states, expanded_states=expanded,
            discovered_states=len(best),
            witness=None if solution is None else witness(solution),
        ),
    )


def describe_source_values(solver, max_states=50000):
    """Describe an actual captured ERK step and its bound helper regions."""
    workload = describe_workload(solver)
    step_owner = solver.kernel.single_integrator._algo_step
    if type(step_owner).__name__ != "ERKStep":
        raise Unsupported("This complete-region adapter requires ERKStep")
    graph = CapturedGraph()
    graph.add_function(step_owner.step_function, "algorithm_step")
    engine = SourceValues(graph)
    step, bound, caller = caller_bindings(engine, solver)
    returned = engine.invoke(step, bound)
    output_ids = {value["identity"] for key, value in engine.cells.items()
                  if key[0] in engine.boundary}
    if returned["raw"] is not None or returned["identity"] is not None:
        returned = engine.scalar(returned, source_function(step))
        output_ids.add(returned["identity"])
    consumers = {}
    for node in engine.nodes:
        for key in node["inputs"]:
            consumers.setdefault(key, set()).add(node["id"])
    certificates = []
    for call in engine.calls:
        if call["kind"] != "source_call":
            continue
        nodes = engine.nodes[call["first_node"]:call["end_node"]]
        member_ids = {node["id"] for node in nodes}
        referenced = {key for node in nodes
                      for key in node["inputs"] + node["outputs"]}
        inputs = {key for key in referenced
                  if engine.values[key]["producer"] not in member_ids and
                  engine.values[key]["kind"] != "constant"}
        # Actual lexical call cuts include surrounding live-through values.
        # Values used only before the call are not helper outputs.
        outputs = set()
        for value in engine.values:
            key = value["id"]
            producer = value["producer"]
            if value["kind"] == "constant":
                continue
            before_exit = producer is None or producer < call["end_node"]
            used_after = key in output_ids or any(
                consumer >= call["end_node"]
                for consumer in consumers.get(key, set())
            )
            if before_exit and used_after:
                outputs.add(key)
                if producer is None or producer < call["first_node"]:
                    inputs.add(key)
        referenced |= inputs | outputs
        if call["function"] == "f0":
            inputs |= set(engine.input_ids)
            outputs |= output_ids
            referenced |= inputs | outputs
        certificate = graph_certificate(
            nodes, [engine.values[key] for key in sorted(referenced)],
            inputs, outputs, max_states=max_states,
        )
        certificates.append(dict(
            context=call["context"], function=call["function"],
            scope="actual lexical call interval with caller live-through",
            node_ids=sorted(member_ids), live_ins=sorted(inputs),
            observable_outputs=sorted(outputs), **certificate,
        ))
    edges = [dict(producer=engine.values[key]["producer"],
                  consumer=node["id"], value=key)
             for node in engine.nodes for key in node["inputs"]
             if engine.values[key]["producer"] is not None]
    if any(dispatcher.overloads for dispatcher in graph.dispatchers):
        raise RuntimeError("Native overload appeared during extraction")
    return dict(
        schema_version=1, kind="source_value_frontier_certificate",
        provenance=dict(
            extractor=source_receipt(describe_source_values),
            dependencies=[source_receipt(fn) for fn in
                          (constant, BufferEffects.slice_view,
                           describe_workload, source_function)],
            functions=graph.functions,
        ),
        caller=caller, workload=workload["workload"],
        registry=registry_layout(step_owner),
        allocations=list(engine.allocations.values()), calls=engine.calls,
        final_cells=[dict(cell=list(key), value=value["identity"],
                          boundary=key[0] in engine.boundary)
                     for key, value in sorted(engine.cells.items())],
        controls=engine.controls, aliases=engine.aliases,
        values=engine.values, nodes=engine.nodes, value_edges=edges,
        order_edges=[dict(before=before, after=node["id"])
                     for node in engine.nodes
                     for before in node["order_predecessors"]],
        certificates=certificates,
        compilation_check=dict(native_overloads=0,
                               batch_kernel_requested=False,
                               device_function_executed=False),
        contract=dict(
            model="typed source semantics, no CSE/floating folding/remat",
            scalar_replacement="exact same-type byte cells and copy aliases",
            boundary="all accessed caller cells retain exit contents",
            frontier="after each operation; live-ins available at entry",
            ordering="same-cell RAW/WAR/WAW; value edges separately typed",
            native_register_prediction=False,
            unsupported="unknown control, partial loops, overlapping types, "
                        "unknown indices/calls/promotions fail explicitly",
        ),
    )


def main():
    """Construct one host-only ERK solver and save the complete receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="lorenz")
    parser.add_argument("--algo", default="rk4")
    parser.add_argument("--max-states", type=int, default=50000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.max_states < 1:
        parser.error("--max-states must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cache_root.mkdir(parents=True, exist_ok=False)
    previous = get_cache_root_override()
    set_cache_root(args.cache_root.resolve())
    solver = None
    try:
        system = placement.SYSTEMS[args.system]["build"]()
        solver = placement.make_solver(system, args.system, args.algo)
        result = describe_source_values(solver, args.max_states)
        result["provenance"]["isolated_cache_root"] = str(
            args.cache_root.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
        print(json.dumps(dict(
            output=str(args.output.resolve()), nodes=len(result["nodes"]),
            values=len(result["values"]),
            certificates=[dict(context=x["context"],
                               source_peak=x["source_schedule"][
                                   "objective_peak"],
                               exact_status=x["exact_search"]["status"])
                          for x in result["certificates"]],
            **result["compilation_check"],
        )))
    finally:
        if solver is not None:
            solver.close()
        set_cache_root(previous)


if __name__ == "__main__":
    main()
