"""Analyze actual caller source use/kill without native specialization."""

import ast
import copy
import inspect
import hashlib

from attrs import fields, has
from cubie._serialize import canonical_digest
from benchmarks.hardware_model.workload_identity import workload_identity

import numpy as np

from benchmarks.hardware_model.expansion import (
    UNKNOWN,
    constant,
    snapshot,
    source_receipt,
)
from benchmarks.hardware_model.workload import python_function, source_function
from benchmarks.hardware_model import implicit_policy_graph as policy
from benchmarks.hardware_model import implicit_native_lowering as native
from benchmarks.hardware_model.caller_lowering_mixin import CallerLiveThrough


def names(node, ctx):
    return {
        x.id
        for x in ast.walk(node)
        if isinstance(x, ast.Name) and isinstance(x.ctx, ctx)
    }


def proved(node, env):
    v = constant(node, env)
    if v is not UNKNOWN:
        return v
    if isinstance(node, ast.BoolOp):
        vs = [proved(x, env) for x in node.values]
        if isinstance(node.op, ast.And) and any(x is False for x in vs):
            return False
        if isinstance(node.op, ast.Or) and any(x is True for x in vs):
            return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        if proved(node.left, env) is False or proved(node.right, env) is False:
            return False
    return UNKNOWN


def fold(body, env):
    result = []
    env = dict(env)
    for s in body:
        s = copy.deepcopy(s)
        s._caller_constants = {
            k
            for k, v in env.items()
            if isinstance(v, (bool, int, float, np.generic)) or v is None
        }
        if isinstance(s, ast.If):
            v = proved(s.test, env)
            if v is not UNKNOWN and type(v).__name__ in ("bool", "bool_"):
                folded, env = fold(s.body if v else s.orelse, env)
                result.extend(folded)
                continue
            s.body, left = fold(s.body, env)
            s.orelse, right = fold(s.orelse, env)
            env = {
                k: v
                for k, v in left.items()
                if k in right
                and snapshot(v) is not UNKNOWN
                and snapshot(v) == snapshot(right[k])
            }
        elif isinstance(s, (ast.For, ast.While)):
            loop_env = {
                k: v for k, v in env.items() if k not in names(s, ast.Store)
            }
            if isinstance(s, ast.For):
                iterator = s.iter
                if (
                    isinstance(iterator, ast.Call)
                    and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == "unroll_if"
                ):
                    iterator = iterator.args[0]
                if (
                    isinstance(iterator, ast.Call)
                    and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == "range"
                ):
                    vals = [proved(x, loop_env) for x in iterator.args]
                    if all(
                        isinstance(x, int) or type(x).__name__ == "int32"
                        for x in vals
                    ) and not len(range(*(int(x) for x in vals))):
                        folded, env = fold(s.orelse, env)
                        result.extend(folded)
                        continue
            s.body, _ = fold(s.body, loop_env)
            s.orelse, _ = fold(s.orelse, loop_env)
            env = loop_env
        elif isinstance(s, ast.Assign):
            v = proved(s.value, env)
            for target in s.targets:
                if isinstance(target, ast.Name):
                    env.pop(target.id, None)
                    if v is not UNKNOWN:
                        env[target.id] = v
        elif isinstance(s, ast.AugAssign):
            name = getattr(s.target, "id", "")
            v = proved(
                ast.BinOp(
                    left=ast.Name(id=name, ctx=ast.Load()),
                    op=s.op,
                    right=s.value,
                ),
                env,
            )
            env.pop(name, None)
            if v is not UNKNOWN:
                env[name] = v
        result.append(s)
    return result, env


class FunctionDemand:
    """Derive helper formal demand from retained source effects and returns."""

    def __init__(self):
        self.cache = {}
        self.active = set()
        self.records = []
        self.errors = []

    def expression_uses(self, node, environment):
        if node is None:
            return set()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            raw_target = environment.get(node.func.id)
            target = python_function(raw_target)
            if target is not None and hasattr(raw_target, "overloads"):
                try:
                    formal = self.describe(target)
                    arguments = inspect.signature(target).bind(
                        *node.args, **{x.arg: x.value for x in node.keywords}
                    )
                except (OSError, TypeError, ValueError) as error:
                    self.errors.append(
                        dict(
                            name=getattr(target, "__qualname__", repr(target)),
                            error=str(error),
                        )
                    )
                    formal = None
                if formal is not None:
                    result = set()
                    for name, argument in arguments.arguments.items():
                        if name in formal["needed_formals"]:
                            result |= self.expression_uses(
                                argument, environment
                            )
                        elif isinstance(argument, ast.AST) and any(
                            isinstance(x, ast.Call) for x in ast.walk(argument)
                        ):
                            # Unused formal does not erase argument effects.
                            result |= self.expression_uses(
                                argument, environment
                            )
                    return result
        if isinstance(node, ast.Name):
            return {node.id} if isinstance(node.ctx, ast.Load) else set()
        return set().union(
            *(
                self.expression_uses(x, environment)
                for x in ast.iter_child_nodes(node)
            )
        )

    def describe(self, function):
        key = id(function)
        if key in self.cache:
            return self.cache[key]
        if key in self.active:
            raise ValueError("Recursive helper formal demand")
        source = source_function(function)
        closure = inspect.getclosurevars(function)
        environment = dict(
            closure.builtins, **closure.globals, **closure.nonlocals
        )
        self.active.add(key)
        body, _ = fold(source.body, environment)
        graph = ControlFlow(body, environment, self)
        graph.solve()
        needed = graph.live_in[graph.entry] & set(
            inspect.signature(function).parameters
        )
        result = dict(
            source=source_receipt(function),
            line=source.lineno,
            name=function.__qualname__,
            needed_formals=sorted(needed),
            fixed_point_rounds=graph.rounds,
        )
        self.active.remove(key)
        self.cache[key] = result
        self.records.append(result)
        return result


class ControlFlow:
    """Solve backward use/kill over retained branch and loop control."""

    def __init__(self, body, environment, demand):
        self.nodes = []
        self.environment = environment
        self.demand = demand
        self.entry = self.block(body, None)
        self.live_in = [set() for _ in self.nodes]
        self.live_out = [set() for _ in self.nodes]
        self.rounds = 0

    def uses(self, expression):
        return self.demand.expression_uses(expression, self.environment)

    def block(self, body, following, break_to=None, continue_to=None):
        successor = following
        for statement in reversed(body):
            index = len(self.nodes)
            record = dict(
                statement=statement,
                successors=[],
                uses=set(),
                definitions=set(),
            )
            self.nodes.append(record)
            if isinstance(statement, ast.If):
                record["uses"] = self.uses(statement.test)
                record["successors"] = [
                    self.block(
                        statement.body, successor, break_to, continue_to
                    ),
                    self.block(
                        statement.orelse, successor, break_to, continue_to
                    ),
                ]
            elif isinstance(statement, (ast.For, ast.While)):
                expression = (
                    statement.iter
                    if isinstance(statement, ast.For)
                    else statement.test
                )
                record["uses"] = self.uses(expression)
                if isinstance(statement, ast.For):
                    record["definitions"] = names(statement.target, ast.Store)
                record["successors"] = [
                    self.block(statement.body, index, successor, index),
                    self.block(
                        statement.orelse, successor, break_to, continue_to
                    ),
                ]
            else:
                record["uses"] = self.uses(statement)
                record["definitions"] = names(statement, ast.Store)
                if isinstance(statement, ast.AugAssign) and isinstance(
                    statement.target, ast.Name
                ):
                    record["uses"].add(statement.target.id)
                record["successors"] = [
                    break_to
                    if isinstance(statement, ast.Break)
                    else continue_to
                    if isinstance(statement, ast.Continue)
                    else None
                    if isinstance(statement, ast.Return)
                    else successor
                ]
            record["uses"] -= getattr(statement, "_caller_constants", set())
            successor = index
        return successor

    def solve(self):
        while True:
            changed = False
            self.rounds += 1
            for index, record in enumerate(self.nodes):
                after = set().union(
                    *(
                        self.live_in[x]
                        for x in record["successors"]
                        if x is not None
                    )
                )
                before = record["uses"] | (after - record["definitions"])
                if (
                    before != self.live_in[index]
                    or after != self.live_out[index]
                ):
                    changed = True
                self.live_in[index], self.live_out[index] = before, after
            if not changed:
                for record in self.nodes:
                    statement = record["statement"]
                    if isinstance(statement, ast.For):
                        exit_node = record["successors"][1]
                        exiting = (
                            self.live_in[exit_node]
                            if exit_node is not None
                            else set()
                        )
                        if names(statement.target, ast.Store) & exiting:
                            raise ValueError(
                                "Caller loop target needs edge-specific "
                                "zero-trip definition handling"
                            )
                return


def source_types(body, closure, aliases):
    """Resolve scalar widths from actual casts and typed array element views."""
    types = {
        name: {np.asarray(value).dtype.name}
        for name, value in closure.items()
        if isinstance(value, (np.generic, bool))
    }
    views = {
        entry["name"]: entry["view"]
        for entry in aliases
        if entry.get("view") is not None
        and entry.get("source", {}).get("context") == "caller"
    }
    unresolved = []

    def expression(node):
        if isinstance(node, ast.Name):
            return types.get(node.id, set())
        if isinstance(node, ast.Constant):
            return {
                {bool: "bool", int: "literal_int", float: "literal_float"}.get(
                    type(node.value), "none"
                )
            }
        if isinstance(node, ast.Compare):
            return {"bool"}
        if isinstance(node, ast.UnaryOp):
            return (
                {"bool"}
                if isinstance(node.op, ast.Not)
                else expression(node.operand)
            )
        if isinstance(node, ast.BoolOp):
            return {"bool"}
        if isinstance(node, ast.BinOp):
            left, right = expression(node.left), expression(node.right)
            joined = left | right
            concrete = joined - {"literal_int", "literal_float"}
            if "float64" in concrete:
                return {"float64"}
            if "float32" in concrete:
                return {"float32"}
            return concrete or joined
        if isinstance(node, ast.Subscript) and isinstance(
            node.value, ast.Name
        ):
            reference = views.get(node.value.id)
            return {reference["dtype"]} if reference else set()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            casts = {
                "float64": "float64",
                "float32": "float32",
                "precision": "float32",
                "narrow_time": "float32",
                "int32": "int32",
                "uint32": "uint32",
                "bool_": "bool",
                "activemask": "uint32",
                "all_sync": "bool",
                "any_sync": "bool",
            }
            if name in casts:
                return {casts[name]}
            if name == "selp":
                return expression(node.args[1]) | expression(node.args[2])
            if name in ("fmin", "fmax", "min", "max"):
                return set().union(*(expression(x) for x in node.args))
        return set()

    statements = list(ast.walk(ast.Module(body=body, type_ignores=[])))
    while True:
        changed = False
        for node in statements:
            if isinstance(node, ast.Assign):
                inferred = expression(node.value)
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                inferred = expression(
                    ast.BinOp(left=node.target, op=node.op, right=node.value)
                )
                targets = [node.target]
            elif isinstance(node, ast.For):
                inferred = {"int32"}
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and inferred:
                    prior = types.get(target.id, set())
                    combined = prior | inferred
                    if combined != prior:
                        types[target.id] = combined
                        changed = True
        if not changed:
            break
    return types, views, unresolved


def describe_caller(solver, graph):
    """Capture source caller liveness and exact existing step aliases."""
    integrator = solver.kernel.single_integrator
    dispatcher = integrator.device_function
    if dispatcher.overloads:
        raise ValueError("Caller observation requires uncompiled dispatchers")
    function = python_function(integrator._loop.device_function)
    source = source_function(function)
    closure = inspect.getclosurevars(function)
    environment = dict(
        closure.builtins, **closure.globals, **closure.nonlocals
    )
    if (
        source_receipt(function)["sha256"]
        != graph["caller"]["source"]["sha256"]
    ):
        raise ValueError("Caller source differs from bound step graph")
    actual_workload = workload_identity(solver.system, solver)
    if actual_workload != graph["candidate_construction"]["workload_identity"]:
        raise ValueError(
            "Caller and step graph have different actual workload identities"
        )
    body, _ = fold(source.body, environment)
    demand = FunctionDemand()
    flow = ControlFlow(body, environment, demand)
    flow.live_in = [set() for _ in flow.nodes]
    flow.live_out = [set() for _ in flow.nodes]
    flow.solve()
    calls = [
        index
        for index, record in enumerate(flow.nodes)
        if isinstance(record["statement"], ast.Assign)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "step_function"
            for node in ast.walk(record["statement"].value)
        )
    ]
    if len(calls) != 1:
        raise ValueError("Actual caller must have one step call")
    call_index = calls[0]
    live = (flow.live_in[call_index] & flow.live_out[call_index]) - set(
        environment
    )
    types, views, unresolved = source_types(
        body, environment, graph["aliases"]
    )
    for argument in source.args.args:
        subscripts = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == argument.arg
        ]
        if (
            argument.arg in live
            and argument.arg not in types
            and argument.arg not in views
            and subscripts
        ):
            rank = max(
                len(node.slice.elts)
                if isinstance(node.slice, ast.Tuple)
                else 1
                for node in subscripts
            )
            views[argument.arg] = dict(
                storage="caller_argument:" + argument.arg,
                offset=0,
                bytes=None,
                dtype=None,
                itemsize=None,
                address_space="global",
                source_index_rank=rank,
                source_index_expressions=[
                    ast.unparse(node) for node in subscripts
                ],
            )
    scalars, pointers = [], []
    for name in sorted(live):
        if name in views:
            pointers.append(dict(name=name, reference=views[name]))
            continue
        inferred = sorted(
            types.get(name, set()) - {"literal_int", "literal_float"}
        )
        aliases = [
            value["id"]
            for value in graph["values"]
            if value.get("label") == "caller:" + name
        ]
        scalars.append(
            dict(name=name, dtypes=inferred, step_source_values=aliases)
        )
    return dict(
        kind="source_caller_liveness_inventory",
        source=source_receipt(function),
        call_line=graph["caller"]["step_call_line"],
        step_graph_sha256=hashlib.sha256(
            policy.payload(graph).encode()
        ).hexdigest(),
        caller_configuration=caller_configuration(solver),
        scalar_live_through=scalars,
        pointer_live_through=pointers,
        helper_formal_demand=demand.records,
        helper_errors=demand.errors,
        fixed_point_rounds=flow.rounds,
        unresolved=unresolved,
        folded_source=ast.unparse(ast.Module(body=body, type_ignores=[])),
        scope="Typed inventory pending memory-cell and allocation join",
    )


class CellEffects:
    """Derive read-before-write and must-write sets on exact caller views."""

    def __init__(self):
        self.records = []
        self.active = set()
        self.unresolved = []
        self.sources = []
        self.skip_functions = set()
        self.dynamic_accesses = []
        self.dead_scalar_reads = []

    @staticmethod
    def cell(reference, index):
        if not isinstance(index, (int, np.integer)):
            return None
        itemsize = reference.get("itemsize")
        offset = reference.get("offset")
        size = reference.get("bytes")
        if itemsize is None or offset is None or size is None:
            return None
        start = int(index) * itemsize
        if start < 0 or start + itemsize > size:
            raise ValueError("Caller memory index leaves its exact view")
        return (
            reference["storage"],
            offset + start,
            offset + start + itemsize,
            reference["dtype"],
        )

    def resolve(self, expression, views, constants):
        if isinstance(expression, ast.Name):
            return views.get(expression.id)
        if isinstance(expression, ast.Subscript):
            parent = self.resolve(expression.value, views, constants)
            if parent is not None and isinstance(expression.slice, ast.Slice):
                lower = (
                    proved(expression.slice.lower, constants)
                    if expression.slice.lower is not None
                    else 0
                )
                upper = (
                    proved(expression.slice.upper, constants)
                    if expression.slice.upper is not None
                    else parent["bytes"] // parent["itemsize"]
                )
                if isinstance(lower, (int, np.integer)) and isinstance(
                    upper, (int, np.integer)
                ):
                    result = dict(parent)
                    result.update(
                        offset=parent["offset"]
                        + int(lower) * parent["itemsize"],
                        bytes=(int(upper) - int(lower)) * parent["itemsize"],
                    )
                    return result
        return None

    def access(self, expression, views, constants, write=False):
        reference = self.resolve(expression.value, views, constants)
        if reference is None:
            return set(), set()
        if reference.get("address_space") == "global":
            return set(), set()
        tainted = names(expression.slice, ast.Load) & constants.get(
            "_caller_dynamic_symbols", set()
        )
        if tainted:
            self.dynamic_accesses.append(
                dict(
                    storage=reference["storage"],
                    syntax=ast.unparse(expression),
                    line=expression.lineno,
                    runtime_symbols=sorted(tainted),
                )
            )
        index = proved(expression.slice, constants)
        cell = self.cell(reference, index)
        if cell is not None:
            return (set(), {cell}) if write else ({cell}, set())
        if reference.get("bytes") is None:
            self.unresolved.append(
                dict(
                    reason="Unbounded caller cell view",
                    syntax=ast.unparse(expression),
                )
            )
            return set(), set()
        cells = {
            self.cell(reference, position)
            for position in range(reference["bytes"] // reference["itemsize"])
        }
        self.unresolved.append(
            dict(
                reason="Runtime-indexed caller storage",
                storage=reference["storage"],
                syntax=ast.unparse(expression),
            )
        )
        # A runtime write may hit every cell but kills no specific old cell.
        return (set(), set()) if write else (cells, set())

    def expression(self, expression, views, constants):
        if expression is None:
            return set(), set()
        if isinstance(expression, ast.Subscript):
            reads, writes = self.access(expression, views, constants)
            index_reads, index_writes = self.expression(
                expression.slice, views, constants
            )
            return reads | index_reads, writes | index_writes
        if isinstance(expression, ast.Call) and isinstance(
            expression.func, ast.Name
        ):
            raw = constants.get(expression.func.id)
            target = python_function(raw)
            if target is not None and hasattr(raw, "overloads"):
                if id(target) in self.skip_functions:
                    return set(), set()
                closure = inspect.getclosurevars(target)
                if "_local_size" in closure.nonlocals:
                    return set(), set()
                arguments = inspect.signature(target).bind(
                    *expression.args,
                    **{x.arg: x.value for x in expression.keywords},
                )
                bound_views = {
                    name: self.resolve(argument, views, constants)
                    for name, argument in arguments.arguments.items()
                }
                bound_views = {
                    k: v for k, v in bound_views.items() if v is not None
                }
                summary = self.describe(target, bound_views)
                return set(map(tuple, summary["read_before_write"])), set(
                    map(tuple, summary["must_write"])
                )
        reads, writes = set(), set()
        for child in ast.iter_child_nodes(expression):
            part_reads, part_writes = self.expression(child, views, constants)
            reads |= part_reads - writes
            writes |= part_writes
        return reads, writes

    def prepare(self, body, views, constants):
        result = []
        views, constants = dict(views), dict(constants)
        for original in body:
            statement = copy.deepcopy(original)
            if isinstance(statement, ast.If):
                value = proved(statement.test, constants)
                if value is not UNKNOWN and isinstance(
                    value, (bool, np.bool_)
                ):
                    selected = statement.body if value else statement.orelse
                    part, views, constants = self.prepare(
                        selected, views, constants
                    )
                    result.extend(part)
                    continue
                statement.body, _, _ = self.prepare(
                    statement.body, views, constants
                )
                statement.orelse, _, _ = self.prepare(
                    statement.orelse, views, constants
                )
                constants = {
                    k: v
                    for k, v in constants.items()
                    if k not in names(statement, ast.Store)
                }
            elif isinstance(statement, ast.For):
                iterator = statement.iter
                directive = None
                if (
                    isinstance(iterator, ast.Call)
                    and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == "unroll_if"
                ):
                    directive = (
                        proved(iterator.args[1], constants)
                        if len(iterator.args) > 1
                        else None
                    )
                    iterator = iterator.args[0]
                if (
                    isinstance(iterator, ast.Call)
                    and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == "range"
                ):
                    arguments = [proved(x, constants) for x in iterator.args]
                    if all(
                        isinstance(x, (int, np.integer)) for x in arguments
                    ) and isinstance(statement.target, ast.Name):
                        if any(
                            isinstance(node, (ast.Break, ast.Continue))
                            for node in ast.walk(statement)
                        ):
                            raise ValueError(
                                "Fixed-loop cell expansion needs explicit exit handling"
                            )
                        positions = range(*(int(x) for x in arguments))
                        full = (
                            isinstance(directive, (tuple, list))
                            and directive[0] is True
                            and (
                                directive[1] is None
                                or len(positions) <= directive[1]
                            )
                        )
                        dynamic_symbols = set(
                            constants.get("_caller_dynamic_symbols", set())
                        )
                        if not full:
                            dynamic_symbols.add(statement.target.id)
                        constants["_caller_dynamic_symbols"] = dynamic_symbols
                        for index in positions:
                            constants[statement.target.id] = index
                            part, views, constants = self.prepare(
                                statement.body, views, constants
                            )
                            result.extend(part)
                        constants.pop(statement.target.id, None)
                        constants["_caller_dynamic_symbols"] = set(
                            constants.get("_caller_dynamic_symbols", set())
                        ) - {statement.target.id}
                        part, views, constants = self.prepare(
                            statement.orelse, views, constants
                        )
                        result.extend(part)
                        continue
                statement.body, _, _ = self.prepare(
                    statement.body, views, constants
                )
                statement.orelse, _, _ = self.prepare(
                    statement.orelse, views, constants
                )
            elif isinstance(statement, ast.While):
                invariant = {
                    k: v
                    for k, v in constants.items()
                    if k not in names(statement, ast.Store)
                }
                statement.body, _, _ = self.prepare(
                    statement.body, views, invariant
                )
                statement.orelse, _, _ = self.prepare(
                    statement.orelse, views, invariant
                )
            expression = (
                statement.test
                if isinstance(statement, (ast.If, ast.While))
                else statement.iter
                if isinstance(statement, ast.For)
                else statement.value
                if isinstance(
                    statement,
                    (ast.Assign, ast.AugAssign, ast.Return, ast.Expr),
                )
                else None
            )
            reads, writes = self.expression(expression, views, constants)
            constant_writes = {}
            if isinstance(statement, (ast.Assign, ast.AugAssign)):
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for assignment in targets:
                    if isinstance(assignment, ast.Subscript):
                        more_reads, more_writes = self.access(
                            assignment, views, constants, write=True
                        )
                        if isinstance(statement, ast.AugAssign):
                            prior_reads, _ = self.access(
                                assignment, views, constants
                            )
                            reads |= prior_reads
                        reads |= more_reads
                        writes |= more_writes
                        value = proved(statement.value, constants)
                        if isinstance(statement, ast.Assign) and (
                            value is not UNKNOWN and np.isscalar(value)
                        ):
                            for cell in more_writes:
                                typed = np.dtype(cell[3]).type(value)
                                if not np.isfinite(typed):
                                    raise ValueError(
                                        "Caller constant store must be finite"
                                    )
                                constant_writes[cell] = snapshot(typed)
                    elif isinstance(assignment, ast.Name):
                        reference = self.resolve(
                            statement.value, views, constants
                        )
                        if isinstance(
                            statement.value, ast.Call
                        ) and isinstance(statement.value.func, ast.Name):
                            target = python_function(
                                constants.get(statement.value.func.id)
                            )
                            if target is not None:
                                captured = inspect.getclosurevars(
                                    target
                                ).nonlocals
                                if "_local_size" in captured:
                                    if (
                                        captured["_use_shared"]
                                        or captured["_use_persistent"]
                                    ):
                                        parent_name = (
                                            "shared"
                                            if captured["_use_shared"]
                                            else "persistent"
                                        )
                                        bindings = inspect.signature(
                                            target
                                        ).bind(*statement.value.args)
                                        parent = self.resolve(
                                            bindings.arguments[parent_name],
                                            views,
                                            constants,
                                        )
                                        selected = captured[
                                            "_" + parent_name + "_slice"
                                        ]
                                        dtype = np.dtype(captured["_dtype"])
                                        reference = dict(
                                            parent,
                                            offset=parent["offset"]
                                            + selected.start
                                            * parent["itemsize"],
                                            bytes=(
                                                selected.stop - selected.start
                                            )
                                            * parent["itemsize"],
                                            dtype=dtype.name,
                                            itemsize=dtype.itemsize,
                                        )
                                    else:
                                        # Helper-local allocation has no incoming caller value.
                                        reference = None
                        if reference is not None:
                            views[assignment.id] = reference
                        dynamic_symbols = set(
                            constants.get("_caller_dynamic_symbols", set())
                        )
                        if names(statement.value, ast.Load) & dynamic_symbols:
                            dynamic_symbols.add(assignment.id)
                        else:
                            dynamic_symbols.discard(assignment.id)
                        constants["_caller_dynamic_symbols"] = dynamic_symbols
                        value = proved(statement.value, constants)
                        constants.pop(assignment.id, None)
                        if value is not UNKNOWN:
                            constants[assignment.id] = value
            statement._cell_reads = reads
            statement._cell_writes = writes
            statement._cell_constant_writes = constant_writes
            result.append(statement)
        return result, views, constants

    def remove_dead_scalar_reads(self, flow):
        """Use scalar demand before admitting effect-free memory reads."""
        while True:
            flow.solve()
            removed = False
            for index, record in enumerate(flow.nodes):
                statement = record["statement"]
                if not (
                    isinstance(statement, ast.Assign)
                    and all(
                        isinstance(target, ast.Name)
                        for target in statement.targets
                    )
                    and record["definitions"]
                    and not record["definitions"] & flow.live_out[index]
                    and not statement._cell_writes
                    and not any(
                        isinstance(node, (ast.Call, ast.NamedExpr))
                        for node in ast.walk(statement.value)
                    )
                ):
                    continue
                if statement._cell_reads:
                    self.dead_scalar_reads.append(
                        dict(
                            line=statement.lineno,
                            syntax=ast.unparse(statement),
                            cells=sorted(statement._cell_reads),
                            scalar_definitions=sorted(record["definitions"]),
                            scalar_live_out=sorted(flow.live_out[index]),
                            proof="Unused scalar assignment with no calls or writes",
                        )
                    )
                    statement._cell_reads = set()
                if record["uses"]:
                    record["uses"] = set()
                    removed = True
            if not removed:
                return

    def describe(self, function, bound_views):
        key = (id(function), repr(sorted(bound_views.items())))
        if key in self.active:
            raise ValueError("Recursive caller memory effect summary")
        self.active.add(key)
        source = source_function(function)
        closure = inspect.getclosurevars(function)
        constants = dict(
            closure.builtins, **closure.globals, **closure.nonlocals
        )
        body, _, _ = self.prepare(source.body, bound_views, constants)
        flow = ControlFlow(body, constants, FunctionDemand())
        self.remove_dead_scalar_reads(flow)
        for record in flow.nodes:
            record["uses"] = record["statement"]._cell_reads
            record["definitions"] = record["statement"]._cell_writes
        flow.live_in = [set() for _ in flow.nodes]
        flow.live_out = [set() for _ in flow.nodes]
        flow.solve()
        # Definite writes solve a forward intersection across every return.
        universe = set().union(*(x["definitions"] for x in flow.nodes))
        incoming = [set(universe) for _ in flow.nodes]
        outgoing = [set(universe) for _ in flow.nodes]
        parents = [[] for _ in flow.nodes]
        for index, record in enumerate(flow.nodes):
            for after in record["successors"]:
                if after is not None:
                    parents[after].append(index)
        while True:
            changed = False
            for index, record in enumerate(flow.nodes):
                before = (
                    set()
                    if index == flow.entry
                    else set.intersection(
                        *(outgoing[p] for p in parents[index])
                    )
                    if parents[index]
                    else set()
                )
                after = before | record["definitions"]
                if before != incoming[index] or after != outgoing[index]:
                    incoming[index], outgoing[index] = before, after
                    changed = True
            if not changed:
                break
        terminal = [
            index
            for index, record in enumerate(flow.nodes)
            if None in record["successors"]
        ]
        must = (
            set.intersection(*(outgoing[x] for x in terminal))
            if terminal
            else set()
        )
        result = dict(
            source=source_receipt(function),
            name=function.__qualname__,
            read_before_write=sorted(flow.live_in[flow.entry]),
            must_write=sorted(must),
        )
        self.active.remove(key)
        self.records.append(result)
        return result


def preceding_constant_cells(flow, call, graph):
    """Prove constants through the unique straight-line pre-call segment."""
    wanted = set(flow.live_out[call])
    wanted = {
        cell
        for cell in wanted
        if not any(
            node["cell"][0] == cell[0]
            and (
                node.get("address_value_ids")
                or node["cell"][1] < cell[2]
                and cell[1] < node["cell"][2]
            )
            for node in graph["nodes"]
            if "cell" in node
        )
    }
    parents = [[] for _ in flow.nodes]
    for index, record in enumerate(flow.nodes):
        for after in record["successors"]:
            if after is not None:
                parents[after].append(index)
    found, visited = {}, set()
    current = call
    while wanted - found.keys() and len(parents[current]) == 1:
        previous = parents[current][0]
        if previous in visited:
            break
        visited.add(previous)
        record = flow.nodes[previous]
        statement = record["statement"]
        if isinstance(
            statement,
            (ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Return),
        ):
            break
        writes = statement._cell_writes
        constants = statement._cell_constant_writes
        memory_target = any(
            isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
            for node in ast.walk(statement)
        )
        if memory_target:
            if not writes or set(constants) != writes:
                break
        elif any(isinstance(node, ast.Call) for node in ast.walk(statement)):
            break
        for cell in wanted - found.keys():
            overlapping = {
                other
                for other in writes
                if other[0] == cell[0]
                and other[1] < cell[2]
                and cell[1] < other[2]
            }
            if overlapping and overlapping != {cell}:
                wanted.discard(cell)
            elif cell in constants:
                found[cell] = dict(
                    cell=list(cell),
                    constant=constants[cell],
                    source_store_line=statement.lineno,
                    source_store=ast.unparse(statement),
                    source_path="Unique straight-line predecessor segment",
                    step_alias_proof="No overlapping exact-cell or runtime-alias source access in step",
                    materialization="Recreate source constant after step",
                )
        current = previous
    return found


def caller_cell_inventory(solver, graph):
    """Bind extra caller cell liveness after exact step-boundary cells join."""
    integrator = solver.kernel.single_integrator
    _ = integrator.device_function
    function = python_function(integrator._loop.device_function)
    source = source_function(function)
    closure = inspect.getclosurevars(function)
    constants = dict(closure.builtins, **closure.globals, **closure.nonlocals)
    loops = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.While)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "step_function"
            for call in ast.walk(node)
        )
    ]
    if len(loops) != 1:
        raise ValueError("Caller must have one loop containing its step call")
    views = {
        entry["name"]: entry["view"]
        for entry in graph["aliases"]
        if entry.get("view") is not None
        and entry.get("source", {}).get("context") == "caller"
    }
    effects = CellEffects()
    effects.skip_functions.add(id(python_function(constants["step_function"])))
    body, _, _ = effects.prepare(loops, views, constants)
    flow = ControlFlow(body, constants, FunctionDemand())
    effects.remove_dead_scalar_reads(flow)
    for record in flow.nodes:
        record["uses"] = record["statement"]._cell_reads
        record["definitions"] = record["statement"]._cell_writes
    flow.live_in = [set() for _ in flow.nodes]
    flow.live_out = [set() for _ in flow.nodes]
    flow.solve()
    calls = [
        index
        for index, record in enumerate(flow.nodes)
        if isinstance(record["statement"], ast.Assign)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "step_function"
            for node in ast.walk(record["statement"].value)
        )
    ]
    if len(calls) != 1:
        raise ValueError("Cell CFG changed step call identity")
    existing = {
        tuple(entry["cell"]): entry["value"]
        for entry in graph["final_cells"]
        if entry["boundary"]
    }
    constants = preceding_constant_cells(flow, calls[0], graph)
    rows = []
    for cell in sorted(flow.live_out[calls[0]]):
        if cell in constants:
            continue
        rows.append(
            dict(
                cell=list(cell),
                step_final_source_value=existing.get(cell),
                additional_caller_value=cell not in existing,
                address_space=(
                    "shared" if cell[0] == "caller:shared_scratch" else "local"
                ),
            )
        )
    return dict(
        kind="caller_cell_read_before_write_liveness",
        step_graph_sha256=hashlib.sha256(
            policy.payload(graph).encode()
        ).hexdigest(),
        caller_configuration=caller_configuration(solver),
        source=source_receipt(function),
        cells=rows,
        rematerialized_constant_cells=list(constants.values()),
        dead_scalar_read_proofs=effects.dead_scalar_reads,
        helper_effects=effects.records,
        unresolved=effects.unresolved,
        dynamic_accesses=effects.dynamic_accesses,
        step_join=dict(
            rule="Every accessed boundary cell has a step exit observable; reused by exact storage/byte/type identity",
            existing_cells=len(existing),
        ),
        scope="Existing step cell versions reused; step summary omitted only for determining additional cells outside that set",
    )


class CallerTypedLowering(CallerLiveThrough, policy.PolicyTypedLowering):
    """Compose the source caller retention with the selected step lowerer."""


def make_caller_plan(
    graph,
    compiler,
    inventory,
    cells,
    gpr_budget,
    predicate_budget,
    cell_form="promoted_constant_cells",
    pointer_form="parameter_rematerialization",
):
    """Freshly lower and allocate the declared complete caller alternative."""
    if (
        inventory["step_graph_sha256"]
        != hashlib.sha256(policy.payload(graph).encode()).hexdigest()
    ):
        raise ValueError(
            "Caller inventory belongs to a different exact step graph"
        )
    if (
        cells["step_graph_sha256"] != inventory["step_graph_sha256"]
        or cells["caller_configuration"] != inventory["caller_configuration"]
    ):
        raise ValueError("Caller scalar and cell constructions differ")
    lowered = CallerTypedLowering(
        graph, compiler, inventory, cells, cell_form, pointer_form
    ).build()
    allocation = native.BankAllocation(
        lowered, gpr_budget, predicate_budget
    ).build()
    check = native.verify_allocation(lowered, allocation)
    return dict(
        kind="conditional_caller_live_through_native_plan",
        lowering=lowered,
        allocation=allocation,
        allocation_verification=check,
        caller_alternative=dict(
            cell_form=cell_form, pointer_form=pointer_form
        ),
        source=source_receipt(CallerTypedLowering),
        measured_timings_consumed=False,
        measured_register_counts_consumed=False,
    )


def caller_configuration(solver):
    """Bind actual caller/child settings including unroll and placement."""

    def actual(value):
        if has(type(value)):
            return (
                type(value).__module__,
                type(value).__qualname__,
                tuple(
                    (field.name, actual(getattr(value, field.name)))
                    for field in fields(type(value))
                    if field.eq
                ),
            )
        if isinstance(value, (tuple, list)):
            return tuple(actual(x) for x in value)
        return value

    records = {}
    seen = set()

    def visit(factory, path):
        if id(factory) in seen:
            return
        seen.add(id(factory))
        records[path] = canonical_digest(actual(factory.compile_settings))
        for name, child in vars(factory).items():
            if hasattr(child, "compile_settings"):
                visit(child, path + "." + name)

    visit(solver.kernel.single_integrator, "integrator")
    return records
