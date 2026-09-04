"""Count canonical source expansion using captured, uncompiled closures.

These are AST transformation facts, not native instruction predictions.
Counted loops use an explicitly described main-chunk/tail representation.
Iterative loops retain one symbolic body even when full unroll is requested.
"""

import argparse
import ast
from collections import Counter
import copy
import hashlib
import inspect
import json
import math
import numbers
import operator
from pathlib import Path

import numpy as np

from cubie.cuda_simsafe import (
    float32 as device_float32,
    float64 as device_float64,
    int32 as device_int32,
    unroll_if as device_unroll_if,
)

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model.static_descriptors import syntax_counts
from benchmarks.hardware_model.workload import (
    UNKNOWN,
    WorkloadGraph,
    describe_workload,
    python_function,
    source_function,
)


SCHEMA_VERSION = 1
EXTRACTOR_PATH = Path(__file__).resolve()
EXTRACTOR_SHA256 = hashlib.sha256(EXTRACTOR_PATH.read_bytes()).hexdigest()
ITERATIVE_GROUPS = {
    "unroll_newton_exits": "N[call,warp]",
    "unroll_krylov_exits": "K[call,warp]",
}
INTEGER_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def snapshot(value):
    """Return exact supported closure values with type/byte provenance."""
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        return dict(
            kind="array",
            dtype=str(value.dtype),
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            values=snapshot(value.tolist()),
        )
    if isinstance(value, (list, tuple)):
        items = [snapshot(item) for item in value]
        return items if all(item is not UNKNOWN for item in items) else UNKNOWN
    if isinstance(value, np.generic):
        item = snapshot(value.item())
        return dict(dtype=str(value.dtype), value=item)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else dict(float_hex=value.hex())
    return UNKNOWN


def digest(value):
    """Hash a JSON-serializable source descriptor."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_receipt(function):
    """Record a source dependency's current path and byte hash."""
    path = Path(inspect.getsourcefile(function)).resolve()
    return dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def scalar_value(value):
    """Identify scalars accepted by the constant expression interpreter."""
    return value is None or isinstance(value, (numbers.Real, np.bool_, bool))


def comparable(left, right):
    """Accept integer predicates and proven typed-float/zero comparisons."""
    floating = (float, np.floating)
    if not any(isinstance(v, floating) for v in (left, right)):
        return True
    for value in (left, right):
        if not isinstance(value, numbers.Real) or not math.isfinite(value):
            return False
        if (
            isinstance(value, np.floating)
            and value != 0
            and (abs(value) < np.finfo(value.dtype).tiny)
        ):
            return False
    if left == 0 or right == 0:
        # Zero is exact in every involved integer/floating representation.
        return True
    return isinstance(left, np.floating) and type(left) is type(right)


def constant(node, environment):
    """Interpret literals, indexing and bounded integer/control expressions.

    Floating arithmetic and arbitrary calls are deliberately unevaluated.
    No user function, dispatcher, generated function or module is executed.
    """
    try:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return environment.get(node.id, UNKNOWN)
        if isinstance(node, (ast.Tuple, ast.List)):
            values = tuple(constant(item, environment) for item in node.elts)
            return values if all(v is not UNKNOWN for v in values) else UNKNOWN
        if isinstance(node, ast.Subscript):
            value = constant(node.value, environment)
            index = constant(node.slice, environment)
            if not isinstance(value, (tuple, list, np.ndarray)):
                return UNKNOWN
            indices = index if isinstance(index, tuple) else (index,)
            if all(isinstance(i, numbers.Integral) for i in indices):
                return value[index]
        if isinstance(node, ast.UnaryOp):
            value = constant(node.operand, environment)
            if value is UNKNOWN:
                return UNKNOWN
            if isinstance(node.op, ast.Not) and scalar_value(value):
                return not value
            if isinstance(value, numbers.Integral):
                operation = {
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                    ast.Invert: operator.invert,
                }.get(type(node.op))
                if operation:
                    result = operation(int(value))
                    return result if -(2**31) <= result < 2**31 else UNKNOWN
        if isinstance(node, ast.BinOp):
            left = constant(node.left, environment)
            right = constant(node.right, environment)
            operation = INTEGER_BINARY.get(type(node.op))
            if operation and all(
                isinstance(item, numbers.Integral) for item in (left, right)
            ):
                if isinstance(node.op, (ast.LShift, ast.RShift)) and not (
                    0 <= int(right) < 32
                ):
                    return UNKNOWN
                result = operation(int(left), int(right))
                return result if -(2**31) <= result < 2**31 else UNKNOWN
        if isinstance(node, ast.Compare):
            values = [constant(node.left, environment)] + [
                constant(item, environment) for item in node.comparators
            ]
            if all(
                value is not UNKNOWN and scalar_value(value)
                for value in values
            ):
                results = []
                for op, left, right in zip(node.ops, values, values[1:]):
                    operation = COMPARISONS.get(type(op))
                    if operation is None or not comparable(left, right):
                        return UNKNOWN
                    results.append(bool(operation(left, right)))
                return all(results)
        if isinstance(node, ast.BoolOp):
            # Do not erase evaluation of an unknown, possibly effectful term.
            result = UNKNOWN
            for item in node.values:
                result = constant(item, environment)
                if result is UNKNOWN or not scalar_value(result):
                    return UNKNOWN
                if isinstance(node.op, ast.And) and not result:
                    return result
                if isinstance(node.op, ast.Or) and result:
                    return result
            return result
        if isinstance(node, ast.IfExp):
            condition = constant(node.test, environment)
            if condition is not UNKNOWN and scalar_value(condition):
                return constant(
                    node.body if condition else node.orelse, environment
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords or len(node.args) != 1:
                return UNKNOWN
            value = constant(node.args[0], environment)
            target = environment.get(node.func.id)
            if target is len and isinstance(value, (tuple, list, np.ndarray)):
                return len(value)
            if any(
                target is item
                for item in (int, np.int32, np.int64, device_int32)
            ) and isinstance(value, numbers.Integral):
                result = int(value)
                return result if -(2**31) <= result < 2**31 else UNKNOWN
            if isinstance(value, numbers.Real):
                if target is np.float32 or target is device_float32:
                    return np.float32(value)
                if target is np.float64 or target is device_float64:
                    return np.float64(value)
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        ZeroDivisionError,
    ):
        return UNKNOWN
    return UNKNOWN


def counts(statements):
    """Use the static descriptor's unweighted source-operation vocabulary."""
    node = ast.FunctionDef(
        name="region",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=statements,
        decorator_list=[],
    )
    return Counter(syntax_counts(node))


def assigned_names(statements):
    """Find scalar definitions invalidated at a recurrent region entry."""
    return {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def common_environment(left, right):
    """Keep only identical supported values at a control-flow merge."""
    result = {}
    for name in left.keys() & right.keys():
        if left[name] is right[name]:
            result[name] = left[name]
            continue
        a, b = snapshot(left[name]), snapshot(right[name])
        if a is not UNKNOWN and b is not UNKNOWN and digest(a) == digest(b):
            result[name] = left[name]
    return result


def mutable_capture(value):
    """Identify mutable captures, including arrays inside immutable tuples."""
    return isinstance(value, (np.ndarray, list)) or (
        isinstance(value, tuple) and any(mutable_capture(v) for v in value)
    )


def invalidate_arrays(environment):
    """Forget array/sequence constants after a possible mutating alias use."""
    for name in list(environment):
        if mutable_capture(environment[name]):
            environment.pop(name)


def invalidate_call_arguments(node, environment):
    """Forget sequence bindings before a possibly mutating expression."""
    for expression in ast.walk(node):
        if isinstance(expression, ast.NamedExpr):
            for name in assigned_names([expression.target]):
                environment.pop(name, None)
    pure = (
        len,
        range,
        int,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        device_int32,
        device_float32,
        device_float64,
        device_unroll_if,
    )
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = constant(call.func, environment)
        if any(target is function for function in pure):
            continue
        arguments = list(call.args) + [kw.value for kw in call.keywords]
        if isinstance(call.func, ast.Attribute):
            arguments.append(call.func.value)
        for argument in arguments:
            value = constant(argument, environment)
            if value is not UNKNOWN and scalar_value(value):
                continue
            # Unknown indexing/slices can still pass a view into a capture.
            names = [
                item.id
                for item in ast.walk(argument)
                if isinstance(item, ast.Name)
            ]
            if mutable_capture(value) or any(
                mutable_capture(environment.get(name)) for name in names
            ):
                invalidate_arrays(environment)
                return


class CapturedGraph(WorkloadGraph):
    """Keep actual callable identities alongside workload graph records."""

    def __init__(self):
        super().__init__({})
        self.callables = {}

    def add_function(self, dispatcher, role):
        identifier = super().add_function(dispatcher, role)
        if identifier is not None:
            self.callables[identifier] = python_function(dispatcher)
        return identifier


class Expressions(ast.NodeTransformer):
    """Fold safe expressions, recording the source facts that justify it."""

    def __init__(self, engine, environment):
        self.engine = engine
        self.environment = environment

    def visit(self, node):
        if not isinstance(node, ast.expr):
            return super().visit(node)
        if isinstance(
            node,
            (
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.NamedExpr,
            ),
        ):
            self.engine.unknown(
                node,
                "Expression has a nested scope or "
                "assignment; authored syntax is retained.",
            )
            return node
        if isinstance(node, (ast.Name, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            return self.generic_visit(node)
        value = constant(node, self.environment)
        if isinstance(node, ast.Compare):
            values = [constant(node.left, self.environment)] + [
                constant(item, self.environment) for item in node.comparators
            ]
            if all(v is not UNKNOWN and scalar_value(v) for v in values):
                if not all(
                    comparable(a, b) for a, b in zip(values, values[1:])
                ):
                    self.engine.unknown(
                        node,
                        "Floating comparison promotion/precision "
                        "is unproved; predicate remains unresolved.",
                    )
        if value is not UNKNOWN and scalar_value(value):
            if not isinstance(node, (ast.Constant, ast.Name)):
                self.engine.fact("constant_expression", node, value)
            item = value.item() if isinstance(value, np.generic) else value
            if isinstance(item, float) and not math.isfinite(item):
                return self.generic_visit(node)
            return ast.copy_location(ast.Constant(value=item), node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for operand in (node.left, node.right):
                item = constant(operand, self.environment)
                if isinstance(item, numbers.Real) and item == 0:
                    self.engine.fact(
                        "retained_zero_coefficient_product", node, item
                    )
                    break
        return self.generic_visit(node)


class Expansion:
    """Canonical fixed-loop expansion and conservative closure folding."""

    def __init__(self, record, fold):
        self.record = record
        self.fold = fold
        self.loop_by_line = {
            loop["source"]["line"]: loop for loop in record["loops"]
        }
        self.regions = []
        self.facts = {}
        self.unknowns = {}
        self.context = []

    def fact(self, kind, node, value=UNKNOWN):
        if not self.fold:
            return
        key = (kind, node.lineno, ast.unparse(node))
        entry = self.facts.setdefault(
            key,
            dict(
                kind=kind,
                line=node.lineno,
                expression=key[2],
                occurrences=0,
                values=[],
            ),
        )
        entry["occurrences"] += 1
        encoded = snapshot(value)
        if encoded is not UNKNOWN and encoded not in entry["values"]:
            entry["values"].append(encoded)

    def unknown(self, node, reason):
        key = (node.lineno, reason)
        self.unknowns[key] = dict(line=node.lineno, reason=reason)

    def expression(self, node, environment):
        result = copy.deepcopy(node)
        if self.fold:
            return Expressions(self, environment).visit(result)
        return result

    def loop(self, node, environment):
        loop = self.loop_by_line.get(node.lineno, {})
        group = loop.get("unroll_group")
        iterator = node.iter if isinstance(node, ast.For) else None
        invalidate_call_arguments(
            iterator if iterator is not None else node.test, environment
        )
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and environment.get(iterator.func.id) is device_unroll_if
        ):
            iterator = iterator.args[0]
        values = None
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and environment.get(iterator.func.id) is range
            and not iterator.keywords
        ):
            args = [constant(arg, environment) for arg in iterator.args]
            if all(isinstance(arg, numbers.Integral) for arg in args):
                try:
                    values = range(*(int(arg) for arg in args))
                except (ValueError, TypeError, OverflowError):
                    pass
        trip = len(values) if values is not None else None
        request = loop.get("requested_replication", {})
        mode = request.get("mode", "ordinary_loop")
        count = request.get("requested_count")
        iterative = group in ITERATIVE_GROUPS
        region = dict(
            function=self.record["id"],
            line=node.lineno,
            unroll_group=group,
            context=list(self.context),
            mode=mode,
            actual_closure_flag=loop.get("actual_closure_flag"),
            candidate_flag=loop.get("flag"),
            requested_count=count,
            fixed_trip_count=trip,
            dynamic_work=ITERATIVE_GROUPS[group] if iterative else trip,
            dynamic_work_is_not_code_replication=True,
            native_replication_known=False,
            parts=[],
        )
        self.regions.append(region)
        assigned = assigned_names(node.body)
        entry = {
            name: value
            for name, value in environment.items()
            if name not in assigned
        }
        target = (
            node.target.id
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
            else None
        )
        if target:
            entry.pop(target, None)
        results = []
        sequences = {
            name for name, value in entry.items() if mutable_capture(value)
        }
        surviving_sequences = set(sequences)

        def part(kind, indices, repetitions, indices_known):
            body = []
            local = {
                name: value
                for name, value in entry.items()
                if name not in sequences or name in surviving_sequences
            }
            self.context.append(dict(line=node.lineno, part=kind))
            for index in indices:
                if target and indices_known:
                    local[target] = index
                elif target:
                    local.pop(target, None)
                expanded, local = self.block(node.body, local)
                body.extend(expanded)
            surviving_sequences.intersection_update(local)
            self.context.pop()
            part_record = dict(
                kind=kind,
                source_body_copies=len(indices),
                dynamic_repetitions=repetitions,
                loop_indices_known=indices_known,
                indices=list(indices) if indices_known else None,
                source_operations=dict(counts(body)),
            )
            region["parts"].append(part_record)
            results.extend(body)

        if iterative:
            # State from a preceding iteration must not become a constant.
            region["iteration_upper_bound"] = trip
            region["requested_replication"] = request
            part(
                "symbolic_iteration_template",
                [None],
                ITERATIVE_GROUPS[group],
                False,
            )
        elif trip == 0:
            self.fact("empty_fixed_loop", node, 0)
        elif mode == "full" and values is not None and target:
            part("fully_expanded", values, 1, True)
        elif mode == "counted" and values is not None and target:
            # Canonical strip-mining is a source model, not backend output.
            quotient, remainder = divmod(trip, count)
            region.update(
                main_chunk_count=quotient, remainder_iterations=remainder
            )
            if quotient:
                indices = (
                    list(values[:count])
                    if quotient == 1
                    else [None for _ in range(count)]
                )
                part("counted_main", indices, quotient, quotient == 1)
            if remainder:
                part("constant_tail", values[quotient * count :], 1, True)
        else:
            part(
                "backend_choice_template"
                if mode == "backend_choice"
                else "retained_loop_template",
                [None],
                trip,
                False,
            )
            if mode == "backend_choice":
                self.unknown(
                    node,
                    "False leaves expansion to the backend; "
                    "the reported single template is not a size.",
                )
            elif trip is None:
                self.unknown(node, "Range or while trip count is dynamic.")
            elif not target:
                self.unknown(node, "Loop target is not a scalar name.")
        if any(
            isinstance(child, (ast.Break, ast.Continue, ast.Return))
            for child in ast.walk(node)
        ):
            region["early_exit_or_skip"] = True
            self.unknown(
                node,
                "Expanded body counts retain possible control "
                "paths; execution depends on early exits/skips.",
            )
        # Post-loop scalar state requires recurrence/exit analysis.
        after = {
            name: value
            for name, value in environment.items()
            if name not in assigned
        }
        for name in sequences - surviving_sequences:
            after.pop(name, None)
        if target:
            after.pop(target, None)
        if node.orelse:
            extra, _ = self.block(node.orelse, after)
            results.extend(extra)
            self.unknown(node, "Loop else is retained as a possible path.")
        return results, after

    def block(self, statements, environment):
        environment = dict(environment)
        result = []
        for statement in statements:
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                self.unknown(
                    statement,
                    "Nested definition has its own "
                    "callable descriptor; body is not inlined.",
                )
                environment.pop(statement.name, None)
                continue
            if isinstance(statement, (ast.For, ast.While)):
                expanded, environment = self.loop(statement, environment)
                result.extend(expanded)
                continue
            if isinstance(statement, ast.If):
                invalidate_call_arguments(statement.test, environment)
                condition = constant(statement.test, environment)
                known = condition is not UNKNOWN and scalar_value(condition)
                if self.fold and known:
                    self.fact(
                        "constant_guard", statement.test, bool(condition)
                    )
                    selected = (
                        statement.body if condition else statement.orelse
                    )
                    expanded, environment = self.block(selected, environment)
                    result.extend(expanded)
                else:
                    body, left = self.block(statement.body, environment)
                    other, right = self.block(statement.orelse, environment)
                    transformed = copy.copy(statement)
                    transformed.test = self.expression(
                        statement.test, environment
                    )
                    transformed.body, transformed.orelse = body, other
                    result.append(transformed)
                    environment = (
                        (left if condition else right)
                        if known
                        else (common_environment(left, right))
                    )
                continue
            if isinstance(
                statement,
                (ast.With, ast.AsyncWith, ast.Try, ast.Match, ast.AsyncFor),
            ):
                self.unknown(
                    statement,
                    "Unsupported structured control; "
                    "authored syntax retained without folding.",
                )
                result.append(copy.deepcopy(statement))
                environment.clear()
                continue
            invalidate_call_arguments(statement, environment)
            transformed = self.expression(statement, environment)
            result.append(transformed)
            if isinstance(statement, ast.Assign):
                value = constant(statement.value, environment)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        environment.pop(target.id, None)
                        if value is not UNKNOWN:
                            environment[target.id] = value
                    elif isinstance(target, ast.Subscript):
                        target_base = constant(target.value, environment)
                        if mutable_capture(target_base):
                            invalidate_arrays(environment)
                    else:
                        for name in assigned_names([target]):
                            environment.pop(name, None)
                        invalidate_arrays(environment)
                        self.unknown(
                            target,
                            "Destructuring target bindings "
                            "are invalidated without evaluation.",
                        )
            elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
                if isinstance(statement.target, ast.Subscript):
                    value = constant(statement.target.value, environment)
                    if mutable_capture(value):
                        invalidate_arrays(environment)
                for name in assigned_names([statement]):
                    environment.pop(name, None)
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        environment.pop(target.id, None)
                invalidate_arrays(environment)
                self.unknown(statement, "Deleted bindings are invalidated.")
            else:
                for name in assigned_names([statement]):
                    environment.pop(name, None)
            if isinstance(statement, (ast.Return, ast.Break, ast.Continue)):
                break
        return result, environment


def describe_expansion(solver, unroll=None):
    """Describe a candidate from actual closures without native compilation.

    Parameters
    ----------
    solver
        Constructed Solver whose device dispatchers have no overloads.
    unroll : dict, optional
        Valid group directives; the solver is not modified.

    Returns
    -------
    dict
        Per-function source operations, fixed-loop regions, proven folds,
        exact captured constants, symbolic iterative work and limitations.
    """
    if unroll is not None and not isinstance(unroll, dict):
        raise ValueError("unroll must be a dictionary of group directives")
    if (
        hashlib.sha256(EXTRACTOR_PATH.read_bytes()).hexdigest()
        != EXTRACTOR_SHA256
    ):
        raise RuntimeError("Extractor source changed after import; restart")
    workload = describe_workload(solver, unroll)
    graph = CapturedGraph()
    graph.add_function(
        solver.kernel.single_integrator._algo_step.step_function,
        "algorithm_step",
    )
    records = []
    totals = {
        key: Counter()
        for key in (
            "authored",
            "expanded_before_folds",
            "expanded_after_folds",
        )
    }
    for record in workload["functions"]:
        record = copy.deepcopy(record)
        actual = graph.functions[int(record["id"][1:])]
        actual_flags = {
            loop["source"]["line"]: loop["flag"] for loop in actual["loops"]
        }
        for loop in record["loops"]:
            loop["actual_closure_flag"] = actual_flags.get(
                loop["source"]["line"]
            )
        function = graph.callables[record["id"]]
        node = source_function(function)
        closure = inspect.getclosurevars(function)
        environment = dict(
            closure.builtins, **closure.globals, **closure.nonlocals
        )
        constants = {
            name: snapshot(value) for name, value in closure.nonlocals.items()
        }
        constants = {
            name: value
            for name, value in constants.items()
            if value is not UNKNOWN
        }
        globals_snapshot = {
            name: snapshot(value) for name, value in closure.globals.items()
        }
        globals_snapshot = {
            name: value
            for name, value in globals_snapshot.items()
            if value is not UNKNOWN
        }
        fingerprints = digest(
            dict(nonlocals=constants, globals=globals_snapshot)
        )
        before = Expansion(record, False)
        before_body, _ = before.block(node.body, environment)
        after = Expansion(record, True)
        after_body, _ = after.block(node.body, environment)
        operations = dict(
            authored=dict(counts(node.body)),
            expanded_before_folds=dict(counts(before_body)),
            expanded_after_folds=dict(counts(after_body)),
        )
        for key, value in operations.items():
            totals[key].update(value)
        records.append(
            dict(
                id=record["id"],
                name=record["name"],
                roles=record["roles"],
                source=record["source"],
                static_descriptor_join=record["static_descriptor_join"],
                arguments=record["arguments"],
                closure_constants=constants,
                closure_sha256=fingerprints,
                captured_global_constants=globals_snapshot,
                operations=operations,
                replicated_regions=after.regions,
                folds=list(after.facts.values()),
                residual_unknowns=list(after.unknowns.values()),
                calls=record["calls"],
                callable_bodies_are_not_multiplied_by_call_frequency=True,
            )
        )
        current = inspect.getclosurevars(function)
        reread = {}
        for scope in ("nonlocals", "globals"):
            values = {
                name: snapshot(value)
                for name, value in getattr(current, scope).items()
            }
            reread[scope] = {
                name: value
                for name, value in values.items()
                if value is not UNKNOWN
            }
        if digest(reread) != fingerprints:
            raise RuntimeError("Captured closure changed during extraction")
    if any(dispatcher.overloads for dispatcher in graph.dispatchers):
        raise RuntimeError("Source extraction produced a native overload")
    if (
        hashlib.sha256(EXTRACTOR_PATH.read_bytes()).hexdigest()
        != EXTRACTOR_SHA256
    ):
        raise RuntimeError("Extractor source changed during extraction")
    return dict(
        schema_version=SCHEMA_VERSION,
        kind="canonical_source_expansion",
        provenance=dict(
            extractor_path=str(EXTRACTOR_PATH),
            extractor_sha256=EXTRACTOR_SHA256,
            workload_extractor=source_receipt(describe_workload),
            operation_vocabulary=source_receipt(syntax_counts),
        ),
        candidate=unroll or {},
        workload=workload["workload"],
        root=workload["root"],
        functions=records,
        source_inventory_operations={
            key: dict(value) for key, value in totals.items()
        },
        compilation_check=workload["compilation_check"],
        interpretation=dict(
            counts="Unweighted source syntax, not executed/native operations.",
            counted="Canonical strip-mined main plus constant tail; backend "
            "lowering and folding can differ.",
            iterative="One recurrent body template; requested replication "
            "and symbolic executed work are separate metadata.",
            false="One unresolved body template, not an expansion estimate.",
            call_instances="IDs distinguish actual closures, including n and "
            "s*n solve widths; helpers are not inlined.",
        ),
        limitations=[
            "No native specialization, SASS, register "
            "or performance estimate.",
            "No floating arithmetic reassociation or zero*unknown deletion; "
            "NaN/infinity and signed-zero semantics are not assumed away.",
            "Counted main indices stay unknown when there is more than one "
            "chunk; the canonical constant tail can expose coefficients.",
            "Loop overhead and branch lowering are not included in expanded "
            "body counts. Dynamic branches include both written paths.",
            "Scalar recurrences are invalidated at recurrent template entry; "
            "no interprocedural propagation, array alias or exit-path proof.",
            "Source inventory sums each captured callable once. It is not "
            "the hot inlined instruction working set or per-step work.",
        ],
    )


def main():
    """Construct a harness solver and emit CPU-only expansion JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--unroll", default="{}", help="JSON group directives")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    system = placement.SYSTEMS[args.system]["build"]()
    solver = placement.make_solver(system, args.system, args.algo)
    result = describe_expansion(solver, json.loads(args.unroll))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                functions=len(result["functions"]),
                **result["compilation_check"],
            )
        )
    )


if __name__ == "__main__":
    main()
