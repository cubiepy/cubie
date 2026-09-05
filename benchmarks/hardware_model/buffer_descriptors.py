"""Observe buffer identities and element effects before native compilation.

Known calls are interpreted as source regions, with byte-based aliases.
Residual loops and branches are conservative control-flow summaries.
The resulting scalar-replacement opportunities are conditional source
facts, never register-allocation, native-traffic or timing predictions.
"""

import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from cubie.buffer_registry import buffer_registry
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cubie_cache import toolchain_fingerprint

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model.expansion import (
    CapturedGraph,
    Expansion,
    constant,
    snapshot,
    source_receipt,
)
from benchmarks.hardware_model.workload import (
    UNKNOWN,
    describe_workload,
    python_function,
    source_function,
)


SCRIPT = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def value(raw=UNKNOWN, refs=(), dependencies=()):
    """Represent a source scalar, callable or set of possible byte views."""
    return dict(raw=raw, refs=list(refs), dependencies=set(dependencies))


def view(storage, offset=0, size=None, itemsize=None, dtype=None,
         shape=UNKNOWN):
    """Describe a contiguous view; unknown offsets alias the whole object."""
    if shape is UNKNOWN:
        shape = [size // itemsize] if size is not None and itemsize else None
    return dict(
        storage=storage, offset=offset, bytes=size,
        itemsize=itemsize, dtype=dtype, shape=shape,
    )


def unique_views(views):
    """Deduplicate identical views; storage identities retain overlap."""
    return list({json.dumps(v, sort_keys=True): v for v in views}.values())


def merge_values(values):
    """Conservatively join scalar dependencies and possible buffer aliases."""
    values = list(values)
    first = values[0]["raw"] if values else UNKNOWN
    raw = first if all(item["raw"] is first for item in values) else UNKNOWN
    return value(
        raw=raw,
        refs=unique_views(v for item in values for v in item["refs"]),
        dependencies=set().union(*(v["dependencies"] for v in values)),
    )


def interval(reference):
    """Return a known half-open byte interval, or an unknown extent."""
    if reference["offset"] is None or reference["bytes"] is None:
        return None
    return reference["offset"], reference["offset"] + reference["bytes"]


def overlaps(left, right):
    """Conservatively compare intervals from the same allocation."""
    if left is None or right is None:
        return True
    return left[0] < right[1] and right[0] < left[1]


def registry_layout(step):
    """Resolve the actual registry tree to step-relative byte windows."""
    records = []
    seen = set()

    def visit(owner, path, shared_base, persistent_base):
        if id(owner) in seen:
            return
        seen.add(id(owner))
        group = buffer_registry._groups.get(owner)
        if group is None:
            return
        parent_size = np.dtype(group.parent_dtype).itemsize
        entries = {}
        for name, entry in group.entries.items():
            shared = group.shared_layout.get(name)
            persistent = group.persistent_layout.get(name)
            selected = shared if shared is not None else persistent
            base = shared_base if shared is not None else persistent_base
            space = (
                "shared" if shared is not None else
                "persistent_local" if persistent is not None else "local"
            )
            byte_window = None
            if selected is not None and base is not None:
                byte_window = [
                    base + selected.start * parent_size,
                    base + selected.stop * parent_size,
                ]
            record = dict(
                owner=path, owner_type=type(owner).__name__, name=name,
                declared_location=entry.location,
                resolved_location=space, declared_alias=entry.aliases,
                elements=int(entry.size), dtype=np.dtype(entry.dtype).name,
                itemsize=int(entry.itemsize), parent_itemsize=parent_size,
                bytes=int(entry.size) * entry.itemsize,
                step_relative_byte_window=byte_window,
                group_slice=None if selected is None else [
                    selected.start, selected.stop, selected.step,
                ],
                local_elements=group.local_sizes.get(name),
                persistent=bool(entry.persistent),
                protected_ranges=[list(pair) for pair in entry.protected],
            )
            entries[name] = record
            records.append(record)
        for name, child in group.children.items():
            shared = entries[f"{name}_shared"]["step_relative_byte_window"]
            persistent = entries[f"{name}_persistent"][
                "step_relative_byte_window"
            ]
            visit(
                child, f"{path}.{name}",
                shared[0] if shared else None,
                persistent[0] if persistent else None,
            )

    visit(step, "step", 0, 0)
    return records


class EffectExpansion(Expansion):
    """Keep recurrence boundaries around the canonical source inventory."""

    def loop(self, node, environment):
        first = len(self.regions)
        body, after = super().loop(node, environment)
        region = self.regions[first]
        fixed_full = region["mode"] == "full" and all(
            part["kind"] == "fully_expanded" for part in region["parts"]
        )
        if not body or (fixed_full and not region.get("early_exit_or_skip")):
            return body, after
        wrapper = copy.copy(node)
        wrapper.body = body
        wrapper.orelse = []
        wrapper._buffer_region = region
        # This boundary encloses the main/tail source inventory. It does
        # not assert that a backend executes that inventory as one loop.
        return [wrapper], after


class BufferEffects:
    """Interpret captured source with conservative memory-SSA frontiers."""

    def __init__(self, graph, workload):
        self.graph = graph
        self.records = {r["id"]: r for r in workload["functions"]}
        for record, actual in zip(workload["functions"], graph.functions):
            flags = {x["source"]["line"]: x["flag"]
                     for x in actual["loops"]}
            for loop in record["loops"]:
                loop["actual_closure_flag"] = flags.get(
                    loop["source"]["line"],
                )
        self.function_ids = {
            id(function): key for key, function in graph.callables.items()
        }
        self.events = []
        self.calls = []
        self.controls = []
        self.unknowns = []
        self.allocations = {}
        self.memory = {}
        self.stack = []
        self.context = None
        self.counter = 0
        self.expansions = []
        self.captured_arrays = []
        self.return_states = []
        self.loop_exits = []

    def unknown(self, node, reason, refs=()):
        self.unknowns.append(dict(
            context=self.context, line=getattr(node, "lineno", None),
            syntax=ast.unparse(node), reason=reason,
            storages=sorted({r["storage"] for r in refs}),
        ))

    def wrap(self, raw, label):
        if isinstance(raw, np.ndarray):
            root = raw
            while isinstance(root.base, np.ndarray):
                root = root.base
            matching = next((item for item in self.captured_arrays
                             if np.may_share_memory(root, item[0])), None)
            key = matching[1] if matching else f"captured:{label}"
            if matching is None:
                self.captured_arrays.append((root, key))
            else:
                root = matching[0]
            offset = (
                int(raw.ctypes.data) - int(root.ctypes.data)
                if raw.flags.c_contiguous and root.flags.c_contiguous
                else None
            )
            self.allocations.setdefault(key, dict(
                id=key, kind="captured_array", bytes=int(root.nbytes),
                dtype=root.dtype.name, shape=list(root.shape),
                source_snapshot=snapshot(root),
                source_mutability="readonly" if not raw.flags.writeable
                else "mutable_capture",
            ))
            return value(refs=[view(
                key, offset, int(raw.nbytes), raw.dtype.itemsize,
                raw.dtype.name, shape=list(raw.shape),
            )])
        return value(raw)

    def memory_event(self, kind, node, reference, dependencies=(), weak=False):
        storage = reference["storage"]
        bounds = interval(reference)
        frontier = self.memory.setdefault(storage, {})
        reaching = set().union(*(
            versions for span, versions in frontier.items()
            if overlaps(span, bounds)
        ))
        identifier = len(self.events)
        event = dict(
            id=identifier, kind=kind, context=self.context,
            line=getattr(node, "lineno", None), syntax=ast.unparse(node),
            view=dict(reference),
            reaching_writes=sorted(reaching),
            value_dependencies=sorted(set(dependencies)),
            dependencies=sorted(reaching | set(dependencies)),
            update="read" if kind == "read" else
            "may_write_union" if weak else "definite_write",
        )
        self.events.append(event)
        if kind != "read":
            # A known complete store kills only fully covered intervals.
            if bounds is not None and not weak:
                frontier = {
                    span: versions for span, versions in frontier.items()
                    if span is None or not (
                        bounds[0] <= span[0] and span[1] <= bounds[1]
                    )
                }
            # Unknown stores remain alongside every possibly live producer.
            frontier.setdefault(bounds, set()).add(identifier)
            self.memory[storage] = frontier
        return identifier

    def opaque(self, node, arguments, reason):
        combined = merge_values(arguments)
        self.unknown(node, reason, combined["refs"])
        for reference in combined["refs"]:
            combined["dependencies"].add(self.memory_event(
                "read", node, reference, combined["dependencies"],
            ))
            self.memory_event(
                "opaque_write", node, reference, combined["dependencies"],
                weak=True,
            )
        # The result may alias an input; an unknown fresh result is also
        # possible. Subsequent accesses retain unresolved identity.
        return combined

    def slice_view(self, reference, index, node):
        width = reference["itemsize"]
        base = reference["offset"]
        shape = reference.get("shape")
        if isinstance(index, tuple):
            if any(isinstance(part, slice) for part in index):
                self.unknown(
                    node, "Mixed tuple slices require per-axis strides.",
                    [reference],
                )
                return view(reference["storage"], None, None, width,
                            reference["dtype"], shape=None)
            result = reference
            for part in index:
                result = self.slice_view(result, part, node)
            return result
        tail = shape[1:] if shape else []
        stride = width
        if stride is not None:
            for dimension in tail:
                stride *= dimension
        if isinstance(index, slice) and width and base is not None:
            size = reference["bytes"]
            if size is not None:
                start, stop, step = index.indices(size // stride)
                if step == 1:
                    return view(
                        reference["storage"], base + start * stride,
                        max(0, stop - start) * stride, width,
                        reference["dtype"],
                        shape=[max(0, stop - start)] + tail,
                    )
            self.unknown(node, "Slice extent or stride is unresolved.", [
                reference,
            ])
        elif isinstance(index, (int, np.integer)) and width:
            index = int(index)
            if index < 0 and reference["bytes"] is not None:
                index += reference["bytes"] // stride
            if reference["bytes"] is not None and not (
                0 <= index < reference["bytes"] // stride
            ):
                self.unknown(node, "Constant index exceeds the known view.",
                             [reference])
                return view(reference["storage"], None, width, width,
                            reference["dtype"], shape=tail)
            if index >= 0 and base is not None:
                return view(
                    reference["storage"], base + index * stride,
                    stride, width, reference["dtype"], shape=tail,
                )
        return view(
            reference["storage"], None, stride, width, reference["dtype"],
            shape=tail,
        )

    def expression(self, node, environment):
        if node is None:
            return value(None)
        if isinstance(node, ast.Name):
            return environment.get(node.id, value())
        if isinstance(node, ast.Constant):
            return value(node.value)
        if isinstance(node, ast.IfExp):
            test = self.expression(node.test, environment)
            if isinstance(test["raw"], (bool, np.bool_)):
                return self.expression(
                    node.body if test["raw"] else node.orelse, environment,
                )
            entry = copy.deepcopy(self.memory)
            branches = []
            for part in (node.body, node.orelse):
                self.memory = copy.deepcopy(entry)
                local = dict(environment)
                result = self.expression(part, local)
                branches.append((result, self.memory, local))
            self.merge_memory([item[1] for item in branches])
            self.merge_environment(environment, [item[2] for item in branches])
            result = merge_values([item[0] for item in branches])
            result["dependencies"].update(test["dependencies"])
            self.controls.append(dict(
                kind="conditional_expression_join", context=self.context,
                line=node.lineno, syntax=ast.unparse(node),
            ))
            return result
        if isinstance(node, ast.BoolOp):
            result = self.expression(node.values[0], environment)
            for part in node.values[1:]:
                if isinstance(result["raw"], (bool, np.bool_)):
                    stop = (not result["raw"] if isinstance(node.op, ast.And)
                            else bool(result["raw"]))
                    if stop:
                        break
                    result = self.expression(part, environment)
                    continue
                before = copy.deepcopy(self.memory)
                local = dict(environment)
                other = self.expression(part, local)
                self.merge_memory([before, self.memory])
                self.merge_environment(environment, [dict(environment), local])
                result = merge_values([result, other])
                self.controls.append(dict(
                    kind="short_circuit_join", context=self.context,
                    line=node.lineno, syntax=ast.unparse(node),
                ))
            return result
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            values = [self.expression(node.left, environment)]
            for index, part in enumerate(node.comparators):
                if index == 0:
                    values.append(self.expression(part, environment))
                    continue
                before = copy.deepcopy(self.memory)
                local = dict(environment)
                values.append(self.expression(part, local))
                self.merge_memory([before, self.memory])
                self.merge_environment(environment, [dict(environment), local])
            self.controls.append(dict(
                kind="chained_comparison_join", context=self.context,
                line=node.lineno, syntax=ast.unparse(node),
                interpretation="Later comparators may be skipped",
            ))
            result = merge_values(values)
            result["raw"] = UNKNOWN
            return result
        if isinstance(node, ast.Slice):
            parts = [self.expression(x, environment) for x in (
                node.lower, node.upper, node.step,
            )]
            if all(p["raw"] is None or isinstance(
                p["raw"], (int, np.integer),
            ) for p in parts):
                return value(slice(*(p["raw"] for p in parts)))
            return merge_values(parts)
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value, environment)
            index = self.expression(node.slice, environment)
            references = [
                self.slice_view(ref, index["raw"], node)
                for ref in base["refs"]
            ]
            has_slice = isinstance(node.slice, ast.Slice) or (
                isinstance(node.slice, ast.Tuple)
                and any(isinstance(part, ast.Slice)
                        for part in node.slice.elts)
            )
            if has_slice or any(
                reference.get("shape") for reference in references
            ):
                return value(refs=references,
                             dependencies=index["dependencies"])
            dependencies = base["dependencies"] | index["dependencies"]
            for reference in references:
                dependencies.add(self.memory_event(
                    "read", node, reference, index["dependencies"],
                ))
            if not references:
                self.unknown(node, "Subscript base identity is unresolved.")
            return value(dependencies=dependencies)
        if isinstance(node, ast.Call):
            return self.call(node, environment)
        if isinstance(node, ast.NamedExpr):
            result = self.expression(node.value, environment)
            self.assign(node.target, result, environment)
            return result
        if isinstance(node, (ast.Tuple, ast.List)):
            items = [self.expression(x, environment) for x in node.elts]
            result = merge_values(items)
            result["items"] = items
            result["raw"] = tuple(item["raw"] for item in items)
            return result
        children = [
            self.expression(child, environment)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        ]
        result = merge_values(children)
        constants = {
            name: item["raw"] for name, item in environment.items()
            if item["raw"] is not UNKNOWN and not item["refs"]
        }
        result["raw"] = constant(node, constants)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = constant(node.left, constants)
            right = constant(node.comparators[0], constants)
            if left is not UNKNOWN and right is not UNKNOWN and (
                left is None or right is None
            ):
                if isinstance(node.ops[0], ast.Is):
                    result["raw"] = left is right
                elif isinstance(node.ops[0], ast.IsNot):
                    result["raw"] = left is not right
        # Scalar arithmetic on unknown array objects is not scalarized.
        if result["refs"]:
            self.unknown(node, "Array-valued expression is unresolved.",
                         result["refs"])
        return result

    def allocator(self, function, bound, node):
        closure = inspect.getclosurevars(function).nonlocals
        dtype = np.dtype(closure["_dtype"])
        count = int(closure["elements"])
        key = f"local:{self.context}:{node.lineno}:{self.counter}"
        self.counter += 1
        space = (
            "shared" if closure["_use_shared"] else
            "persistent_local" if closure["_use_persistent"] else "local"
        )
        refs = []
        if space == "local":
            size = int(closure["_local_size"])
            self.allocations[key] = dict(
                id=key, kind="local_allocation_call_instance",
                bytes=size * dtype.itemsize, dtype=dtype.name,
                shape=[size], source_line=node.lineno,
                context=self.context,
                lifetime="one static call instance; recurrence reuse unknown",
            )
            refs = [view(key, 0, size * dtype.itemsize,
                         dtype.itemsize, dtype.name)]
        else:
            parameter = "shared" if space == "shared" else "persistent"
            selection = closure[f"_{parameter}_slice"]
            for ref in bound.arguments[parameter]["refs"]:
                item = self.slice_view(ref, selection, node)
                item.update(
                    itemsize=dtype.itemsize, dtype=dtype.name,
                    shape=[item["bytes"] // dtype.itemsize]
                    if item["bytes"] is not None else None,
                )
                refs.append(item)
        receipt = dict(
            kind="captured_allocator", line=node.lineno,
            binding=ast.unparse(node.func),
            context=self.context, location=space, elements=count,
            local_elements=int(closure["_local_size"]),
            dtype=dtype.name, zero=bool(closure["_zero"]),
            actual_zero_unroll=snapshot(closure["_unroll"]),
            returned_views=refs,
        )
        self.calls.append(receipt)
        if not refs:
            self.unknown(node, "Allocator parent view is unresolved.")
        if closure["_zero"]:
            for ref in refs:
                # Complete array initialization, not a native store count.
                self.memory_event("initialization", node, ref,
                                  weak=len(refs) > 1)
        return value(refs=refs)

    def call(self, node, environment):
        arguments = [self.expression(x, environment) for x in node.args]
        keywords = {
            item.arg: self.expression(item.value, environment)
            for item in node.keywords
        }
        if isinstance(node.func, ast.Attribute) and node.func.attr == "view":
            base = self.expression(node.func.value, environment)
            if len(arguments) == 1 and not keywords:
                try:
                    dtype = np.dtype(arguments[0]["raw"])
                except (TypeError, ValueError):
                    dtype = None
                if dtype is not None:
                    refs = []
                    for ref in base["refs"]:
                        shape = ref.get("shape")
                        if shape and ref["itemsize"] and (
                            shape[-1] * ref["itemsize"] % dtype.itemsize == 0
                        ):
                            shape = shape[:-1] + [
                                shape[-1] * ref["itemsize"] // dtype.itemsize,
                            ]
                        else:
                            shape = None
                        refs.append(dict(ref, itemsize=dtype.itemsize,
                                         dtype=dtype.name, shape=shape))
                    return value(refs=refs)
            return self.opaque(node, [base] + arguments,
                               "Reinterpretation dtype is unresolved.")
        target = self.expression(node.func, environment)
        function = python_function(target["raw"])
        all_arguments = arguments + list(keywords.values())
        if any(isinstance(x, ast.Starred) for x in node.args) or (
            None in keywords
        ):
            return self.opaque(node, all_arguments + [target],
                               "Expanded call arguments are unresolved.")
        if function is not None:
            try:
                bound = inspect.signature(function).bind(
                    *arguments, **keywords,
                )
                for name, parameter in inspect.signature(
                    function,
                ).parameters.items():
                    if name not in bound.arguments:
                        bound.arguments[name] = self.wrap(
                            parameter.default, f"default:{name}",
                        )
            except (TypeError, ValueError) as error:
                return self.opaque(node, all_arguments,
                                   f"Signature binding failed: {error}")
            if (
                function.__name__ == "allocate_buffer"
                and Path(inspect.getsourcefile(function)).resolve()
                == Path(inspect.getsourcefile(type(buffer_registry))).resolve()
            ):
                return self.allocator(function, bound, node)
            identifier = self.function_ids.get(id(function))
            if identifier is not None:
                return self.invoke(identifier, bound.arguments, node)
        receiver = []
        if isinstance(node.func, ast.Attribute):
            receiver = [self.expression(node.func.value, environment)]
        if any(item["refs"] for item in all_arguments + receiver):
            return self.opaque(node, all_arguments + receiver,
                               "Opaque call may read, write or escape views.")
        result = merge_values(all_arguments)
        constants = {
            name: item["raw"] for name, item in environment.items()
            if item["raw"] is not UNKNOWN
        }
        result["raw"] = constant(node, constants)
        return result

    def assign(self, target, result, environment):
        if isinstance(target, ast.Name):
            environment[target.id] = result
        elif isinstance(target, (ast.Tuple, ast.List)):
            items = result.get("items")
            if items is not None and len(items) == len(target.elts):
                for child, item in zip(target.elts, items):
                    self.assign(child, item, environment)
            else:
                self.unknown(target, "Destructuring aliases are unresolved.",
                             result["refs"])
                for child in target.elts:
                    self.assign(child, merge_values([result]), environment)
        elif isinstance(target, ast.Subscript):
            base = self.expression(target.value, environment)
            index = self.expression(target.slice, environment)
            for reference in base["refs"]:
                self.memory_event(
                    "write", target,
                    self.slice_view(reference, index["raw"], target),
                    result["dependencies"] | index["dependencies"],
                    weak=len(base["refs"]) > 1,
                )
            if not base["refs"]:
                self.unknown(target, "Store base identity is unresolved.")
        else:
            self.opaque(target, [result], "Unsupported assignment target.")

    def merge_memory(self, states):
        result = {}
        for state in states:
            for storage, spans in state.items():
                target = result.setdefault(storage, {})
                for span, versions in spans.items():
                    target.setdefault(span, set()).update(versions)
        self.memory = result

    def merge_environment(self, target, states):
        """Join aliases from all feasible paths without reusing one path."""
        names = set().union(*(set(state) for state in states))
        joined = {name: merge_values([state.get(name, value())
                                      for state in states]) for name in names}
        target.clear()
        target.update(joined)

    def block(self, statements, environment):
        returns = []
        for node in statements:
            if isinstance(node, ast.Assign):
                result = self.expression(node.value, environment)
                for target in node.targets:
                    self.assign(target, result, environment)
            elif isinstance(node, ast.AnnAssign):
                self.assign(node.target, self.expression(
                    node.value, environment,
                ), environment)
            elif isinstance(node, ast.AugAssign):
                left = self.expression(node.target, environment)
                right = self.expression(node.value, environment)
                result = merge_values([left, right])
                result["raw"] = constant(ast.BinOp(
                    left=ast.Name(id="left"), op=node.op,
                    right=ast.Name(id="right"),
                ), dict(left=left["raw"], right=right["raw"]))
                self.assign(node.target, result, environment)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test, environment)["raw"]
                if isinstance(condition, (bool, np.bool_)):
                    outputs, stopped = self.block(
                        node.body if condition else node.orelse, environment,
                    )
                    returns.extend(outputs)
                    if stopped:
                        return returns, True
                    continue
                entry = copy.deepcopy(self.memory)
                before = dict(environment)
                branches = []
                for body in (node.body, node.orelse):
                    self.memory = copy.deepcopy(entry)
                    branch = dict(before)
                    outputs, stopped = self.block(body, branch)
                    returns.extend(outputs)
                    branches.append((branch, self.memory, stopped))
                active = [branch for branch in branches if not branch[2]]
                self.merge_memory([
                    branch[1] for branch in active or branches
                ])
                environment.clear()
                names = set().union(*(set(x[0]) for x in active or branches))
                for name in names:
                    environment[name] = merge_values([
                        branch[0].get(name, value())
                        for branch in active or branches
                    ])
                self.controls.append(dict(
                    kind="branch_join", context=self.context,
                    line=node.lineno, guard=ast.unparse(node.test),
                    interpretation="Union of feasible paths; no path counts",
                ))
                if not active:
                    return returns, True
            elif isinstance(node, (ast.For, ast.While)):
                self.expression(node.iter if isinstance(node, ast.For)
                                else node.test, environment)
                before = dict(environment)
                entry = copy.deepcopy(self.memory)
                references = unique_views(
                    ref for item in environment.values()
                    for ref in item["refs"]
                )
                phis = [self.memory_event("loop_phi", node, ref, weak=True)
                        for ref in references]
                if isinstance(node, ast.For):
                    self.assign(node.target, value(), environment)
                start = len(self.events)
                self.loop_exits.append([])
                outputs, _ = self.block(node.body, environment)
                returns.extend(outputs)
                exits = self.loop_exits.pop()
                end = len(self.events)
                self.merge_memory([entry, self.memory] + [x[0] for x in exits])
                environments = [before, environment] + [x[1] for x in exits]
                for name in set().union(*(set(x) for x in environments)):
                    environment[name] = merge_values([
                        item.get(name, value()) for item in environments
                    ])
                for identifier in phis:
                    event = self.events[identifier]
                    storage = event["view"]["storage"]
                    event["backedge_write_candidates"] = [
                        item["id"] for item in self.events[start:end]
                        if item["kind"] != "read"
                        and item["view"]["storage"] == storage
                        and overlaps(interval(event["view"]),
                                     interval(item["view"]))
                    ]
                outputs, _ = self.block(node.orelse, environment)
                returns.extend(outputs)
                self.controls.append(dict(
                    kind="recurrent_source_region", context=self.context,
                    line=node.lineno, syntax=ast.unparse(node).splitlines()[0],
                    event_range=[start, end], loop_phi_events=phis,
                    dynamic_invocations="symbolic; body represented once",
                    alias_state="Conservative loop-carried union",
                    source_region=getattr(node, "_buffer_region", None),
                ))
            elif isinstance(node, ast.Return):
                returns.append(self.expression(node.value, environment))
                if self.return_states:
                    self.return_states[-1].append(copy.deepcopy(self.memory))
                return returns, True
            elif isinstance(node, (ast.Break, ast.Continue)):
                self.controls.append(dict(kind=type(node).__name__,
                                          context=self.context,
                                          line=node.lineno))
                if self.loop_exits:
                    self.loop_exits[-1].append((
                        copy.deepcopy(self.memory), dict(environment),
                    ))
                return returns, True
            elif isinstance(node, ast.Expr):
                self.expression(node.value, environment)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        environment.pop(target.id, None)
                    else:
                        self.opaque(target, list(environment.values()),
                                    "Non-name deletion is unresolved.")
            elif isinstance(node, ast.Pass):
                continue
            else:
                self.opaque(node, list(environment.values()),
                            "Unsupported statement/control effects.")
                environment.update({name: merge_values([item])
                                    for name, item in environment.items()})
        return returns, False

    def invoke(self, identifier, bound, call_node=None):
        function = self.graph.callables[identifier]
        node = source_function(function)
        if identifier in self.stack:
            return self.opaque(call_node or node, list(bound.values()),
                               "Recursive helper effects are unresolved.")
        parent = self.context
        context = f"c{len(self.calls)}:{identifier}"
        record = dict(
            kind="known_helper", id=context, function=identifier,
            parent=parent, line=getattr(call_node, "lineno", None),
            parameter_views={name: item["refs"]
                             for name, item in bound.items()},
            first_event=len(self.events),
        )
        self.calls.append(record)
        self.context = context
        self.stack.append(identifier)
        closure = inspect.getclosurevars(function)
        raw = dict(closure.builtins, **closure.globals, **closure.nonlocals)
        constants = dict(raw)
        constants.update({name: item["raw"] for name, item in bound.items()
                          if item["raw"] is not UNKNOWN})
        expansion = EffectExpansion(
            copy.deepcopy(self.records[identifier]), True,
        )
        body, _ = expansion.block(node.body, constants)
        self.expansions.append(dict(
            context=context, function=identifier,
            replicated_regions=expansion.regions,
            residual_unknowns=list(expansion.unknowns.values()),
            folds=list(expansion.facts.values()),
        ))
        environment = {name: self.wrap(item, f"{identifier}:{name}")
                       for name, item in raw.items()}
        environment.update(bound)
        self.return_states.append([])
        outputs, _ = self.block(body, environment)
        result = merge_values(outputs)
        self.merge_memory(self.return_states.pop() + [self.memory])
        record.update(last_event=len(self.events),
                      returned_views=result["refs"],
                      returned_value_dependencies=sorted(
                          result["dependencies"],
                      ))
        self.stack.pop()
        self.context = parent
        return result

    def summaries(self):
        buffers = defaultdict(lambda: dict(
            reads=0, writes=0, dynamic_accesses=0, opaque_effects=0,
            loop_phi_events=0, constant_intervals=set(),
        ))
        last_use = {}
        for event in self.events:
            row = buffers[event["view"]["storage"]]
            kind = event["kind"]
            if kind in ("read", "write", "initialization"):
                row["reads" if kind == "read" else "writes"] += 1
                bounds = interval(event["view"])
                if bounds is None:
                    row["dynamic_accesses"] += 1
                else:
                    row["constant_intervals"].add(bounds)
            elif kind == "opaque_write":
                row["opaque_effects"] += 1
            elif kind == "loop_phi":
                row["loop_phi_events"] += 1
            for dependency in event["dependencies"]:
                last_use[dependency] = event["id"]
        for storage, row in buffers.items():
            row["constant_intervals"] = [list(span) for span in sorted(
                row["constant_intervals"],
            )]
            row["proof_state"] = (
                "unresolved_effect_or_index" if row["opaque_effects"]
                or row["dynamic_accesses"] else
                "conditional_loop_carried_scalar_values"
                if row["loop_phi_events"] else
                "literal_element_scalar_replacement_opportunity"
            )
            row["native_scalarization"] = "unknown"
        for call in self.calls:
            if call["kind"] != "known_helper":
                continue
            start, end = call["first_event"], call["last_event"]
            body = self.events[start:end]
            live_in = {
                dependency for event in body
                for dependency in event["dependencies"]
                if dependency < start
            }
            live_out = {
                identifier for identifier in range(start, end)
                if last_use.get(identifier, -1) >= end
            }
            call.update(
                live_in_version_ids=sorted(live_in),
                live_out_version_ids=sorted(live_out),
                unversioned_input_read_events=[
                    event["id"] for event in body
                    if event["kind"] == "read"
                    and not event["reaching_writes"]
                ],
                transitive_effects=dict(Counter(
                    event["kind"] for event in body
                )),
                boundary_interpretation=(
                    "Source memory SSA under conservative control unions; "
                    "ordering dependencies included, no native liveness"
                ),
            )
        endpoints = Counter()
        unknown_width = 0
        for identifier, end in last_use.items():
            event = self.events[identifier]
            if event["kind"] == "read":
                continue
            width = event["view"]["bytes"]
            if width is None:
                unknown_width += 1
                continue
            endpoints[identifier] += width
            endpoints[end + 1] -= width
        current = peak = area = 0
        for index in range(len(self.events)):
            current += endpoints[index]
            peak = max(peak, current)
            area += current
        return dict(buffers), dict(
            known_payload_peak_bytes=peak,
            known_payload_area_byte_events=area,
            live_versions_with_unknown_width=unknown_width,
            interpretation=(
                "Lexical memory-version retention hypothesis, including "
                "control unions and ordering dependencies; not native "
                "registers or a required live-value lower bound"
            ),
        )


def describe_buffers(solver, unroll=None):
    """Return interprocedural precompile buffer effects for one step.

    Parameters
    ----------
    solver
        Constructed Solver with zero native overloads.
    unroll : dict, optional
        Requested source expansion directives; does not mutate Solver.

    Returns
    -------
    dict
        Actual registry layouts, captured allocation instances, byte-view
        call bindings, memory versions and explicit proof conditions.
    """
    workload = describe_workload(solver, unroll)
    if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != SCRIPT_SHA256:
        raise RuntimeError("Extractor changed after import; restart")
    step = solver.kernel.single_integrator._algo_step
    graph = CapturedGraph()
    root = graph.add_function(step.step_function, "algorithm_step")
    effects = BufferEffects(graph, workload)
    function = graph.callables[root]
    group = buffer_registry._groups[step]
    dtype = np.dtype(step.precision)
    arrays = {
        rec["id"]: set(rec["buffer_access_spans"]) & set(rec["arguments"])
        for rec in workload["functions"]
    }
    bindings = {}
    for rec in workload["functions"]:
        callee_bindings = []
        actual = graph.callables[rec["id"]]
        closure = inspect.getclosurevars(actual)
        captures = dict(closure.builtins, **closure.globals,
                        **closure.nonlocals)
        for node in ast.walk(source_function(actual)):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Name,
            ):
                continue
            target = python_function(captures.get(node.func.id))
            callee = effects.function_ids.get(id(target))
            if callee is None or any(kw.arg is None for kw in node.keywords):
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                continue
            try:
                bound_syntax = inspect.signature(target).bind(
                    *node.args, **{kw.arg: kw.value for kw in node.keywords},
                )
            except TypeError:
                continue
            callee_bindings.append((callee, bound_syntax.arguments))
        bindings[rec["id"]] = callee_bindings
    changed = True
    while changed:
        changed = False
        for rec in workload["functions"]:
            for callee, arguments in bindings[rec["id"]]:
                for parameter, argument in arguments.items():
                    if parameter not in arrays.get(callee, set()):
                        continue
                    if not isinstance(argument, ast.AST):
                        continue
                    for child in ast.walk(argument):
                        if isinstance(child, ast.Name) and (
                            child.id in rec["arguments"]
                            and child.id not in arrays[rec["id"]]
                        ):
                            arrays[rec["id"]].add(child.id)
                            changed = True
    bound = {}
    for name in inspect.signature(function).parameters:
        if name not in arrays[root] and name not in (
            "shared", "persistent_local",
        ):
            bound[name] = value()
            continue
        key = (f"argument:{name}" if name in ("shared", "persistent_local")
               else "external_arguments_may_alias")
        size = (
            group.shared_buffer_size() * dtype.itemsize if name == "shared"
            else group.persistent_local_buffer_size() * dtype.itemsize
            if name == "persistent_local" else None
        )
        # Formal arguments are potential arrays until their use proves it.
        exact = name in ("shared", "persistent_local")
        bound[name] = value(refs=[view(
            key, 0 if exact else None, size,
            dtype.itemsize if exact else None,
            dtype.name if exact else None,
        )])
        effects.allocations[key] = dict(
            id=key, kind="external_formal_argument", bytes=size,
            dtype="untyped formal; precision used only for array elements",
        )
    effects.invoke(root, bound)
    if any(dispatcher.overloads for dispatcher in graph.dispatchers):
        raise RuntimeError("Source observation created native overloads")
    buffers, liveness = effects.summaries()
    layout = registry_layout(step)
    for call in effects.calls:
        if call["kind"] != "captured_allocator":
            continue
        matches = []
        for entry in layout:
            if (
                entry["resolved_location"] != call["location"]
                or entry["dtype"] != call["dtype"]
                or entry["elements"] != call["elements"]
            ):
                continue
            windows = [interval(ref) for ref in call["returned_views"]]
            expected = entry["step_relative_byte_window"]
            if call["location"] == "local" or (
                expected is not None and tuple(expected) in windows
            ):
                matches.append(dict(owner=entry["owner"], name=entry["name"]))
        call["compatible_registry_entries"] = matches
        call["registry_label_resolution"] = (
            "Unique compatible metadata; byte identity comes from closure"
            if len(matches) == 1 else
            "Ambiguous labels retained; no name-based identity assertion"
        )
    if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != SCRIPT_SHA256:
        raise RuntimeError("Extractor changed during observation; restart")
    return dict(
        schema_version=1, kind="source_buffer_identity_and_effects",
        provenance=dict(
            extractor=source_receipt(describe_buffers),
            expansion=source_receipt(Expansion),
            workload=source_receipt(describe_workload),
            registry=source_receipt(type(buffer_registry)),
            toolchain_fingerprint=toolchain_fingerprint(),
        ),
        candidate=unroll or {}, workload=workload["workload"],
        effective_step_jit_kwargs={
            key: sorted(item) if isinstance(item, set) else item
            for key, item in step.jit_kwargs.items()
        },
        functions=workload["functions"],
        registry=layout, allocations=effects.allocations,
        calls=effects.calls, source_memory_ssa=effects.events,
        controls=effects.controls, expansion=effects.expansions,
        buffers=buffers, source_retention_hypothesis=liveness,
        residual_unknowns=effects.unknowns,
        compilation_check=dict(native_overloads=0,
                               dispatchers=len(graph.dispatchers)),
        limitations=[
            "Scope is one step source region; time-loop lifetime unknown.",
            "Requested expansion is conditional on backend lowering.",
            "Residual loops have one body and explicit backedge summaries.",
            "Branch unions do not imply simultaneous execution.",
            "External formals conservatively share one unknown alias set.",
            "Opaque callbacks may mutate or escape passed byte views.",
            "Noncontiguous captures use unknown byte offsets.",
            "No native registers, spills, traffic, cache cost or timing fit.",
        ],
    )


def main():
    """Construct one host-only harness solver and save descriptor JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--unroll", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cache_root.mkdir(parents=True, exist_ok=False)
    previous = get_cache_root_override()
    set_cache_root(args.cache_root.resolve())
    solver = None
    try:
        system = placement.SYSTEMS[args.system]["build"]()
        solver = placement.make_solver(system, args.system, args.algo)
        result = describe_buffers(solver, json.loads(args.unroll))
        result["provenance"]["isolated_cache_root"] = str(
            args.cache_root.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
        print(json.dumps(dict(
            output=str(args.output.resolve()),
            events=len(result["source_memory_ssa"]),
            calls=len(result["calls"]), buffers=len(result["buffers"]),
            **result["compilation_check"],
        )))
    finally:
        if solver is not None:
            solver.close()
        set_cache_root(previous)


if __name__ == "__main__":
    main()
