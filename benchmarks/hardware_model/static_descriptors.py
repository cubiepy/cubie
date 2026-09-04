"""Extract source descriptors without importing or executing generated code.

Counts describe Python source syntax. They are neither SASS counts nor
execution counts. Straight-line regions keep separate dependency graphs;
control-flow joins and loop-carried dependencies require a workload model.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
CASTS = {
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "float",
    "int",
}
MATH_CALLS = {
    "abs",
    "fabs",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "pow",
    "fma",
    "floor",
    "ceil",
    "trunc",
    "isfinite",
    "isinf",
    "isnan",
    "copysign",
    "fmod",
    "remainder",
}
SELECT_CALLS = {"selp", "fmax", "fmin", "min", "max"}
SOURCE_KINDS = {"literal", "scalar_input", "scalar_definition"}


def expression_text(node):
    """Return a stable readable expression for a source node."""
    return ast.unparse(node) if node is not None else None


def owned_nodes(node):
    """Walk a function's syntax, excluding nested function bodies."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            continue
        yield from owned_nodes(child)


def functions_in(statements, prefix="", guards=()):
    """Yield each lexical function, including conditional alternatives."""
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{statement.name}" if prefix else statement.name
            yield name, guards, statement
            yield from functions_in(statement.body, name, guards)
        else:
            for field, value in ast.iter_fields(statement):
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, ast.stmt) for item in value)
                ):
                    guard = {
                        "kind": type(statement).__name__,
                        "line": statement.lineno,
                        "branch": field,
                        "condition": expression_text(
                            getattr(statement, "test", None)
                        ),
                    }
                    yield from functions_in(value, prefix, guards + (guard,))


def literal_index(node):
    """Return a literal integer/tuple index, or an explicit unknown."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    if type(value) is int:
        return value
    if isinstance(value, tuple) and all(type(v) is int for v in value):
        return list(value)
    return None


def static_range_count(node):
    """Count literal range bounds without evaluating source code."""
    if (
        isinstance(node, ast.Call)
        and expression_text(node.func) == "unroll_if"
    ):
        node = node.args[0] if node.args else node
    if not isinstance(node, ast.Call) or expression_text(node.func) != "range":
        return None
    if node.keywords or not 1 <= len(node.args) <= 3:
        return None
    values = [literal_index(value) for value in node.args]
    if not all(type(value) is int for value in values):
        return None
    try:
        return len(range(*values))
    except (ValueError, OverflowError):
        return None


def syntax_counts(function):
    """Count written operators, not expanded or executed operations."""
    counts = Counter()
    nodes = (
        node
        for statement in function.body
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in owned_nodes(statement)
    )
    for node in nodes:
        if isinstance(node, ast.BinOp):
            counts[f"binary.{type(node.op).__name__}"] += 1
        elif isinstance(node, ast.UnaryOp):
            counts[f"unary.{type(node.op).__name__}"] += 1
        elif isinstance(node, ast.Compare):
            for operator in node.ops:
                counts[f"compare.{type(operator).__name__}"] += 1
        elif isinstance(node, ast.BoolOp):
            counts[f"boolean.{type(node.op).__name__}"] += len(node.values) - 1
        elif isinstance(node, ast.AugAssign):
            counts[f"binary.{type(node.op).__name__}"] += 1
        elif isinstance(node, ast.Call):
            counts[f"call.{expression_text(node.func)}"] += 1
        elif isinstance(node, ast.Subscript):
            counts[f"subscript.{type(node.ctx).__name__}"] += 1
    return dict(sorted(counts.items()))


class Region:
    """One lexical straight-line region with versioned scalar values."""

    def __init__(self, region_id, context, types, precision):
        self.region_id = region_id
        self.context = context
        self.types = types
        self.precision = precision
        self.nodes = []
        self.scalars = {}
        self.versions = Counter()
        self.buffer_stores = {}
        self.buffer_events = {}
        self.dynamic_stores = {}
        self.buffer_reads = {}
        self.accesses = []
        self.limitations = []

    def limitation(self, node, reason):
        entry = {
            "line": node.lineno,
            "kind": type(node).__name__,
            "reason": reason,
        }
        if entry not in self.limitations:
            self.limitations.append(entry)

    def add(self, node, kind, dependencies=(), dtype="unknown", **fields):
        dependencies = list(dict.fromkeys(dependencies))
        depth = max((self.nodes[i]["depth"] for i in dependencies), default=0)
        if kind not in SOURCE_KINDS:
            depth += 1
        result = dict(
            id=len(self.nodes),
            kind=kind,
            line=getattr(node, "lineno", None),
            dependencies=dependencies,
            dtype=dtype,
            depth=depth,
            **fields,
        )
        self.nodes.append(result)
        return result["id"]

    def value_type(self, dependencies):
        types = {self.nodes[i]["dtype"] for i in dependencies}
        return next(iter(types)) if len(types) == 1 else "unknown"

    def name(self, node):
        if node.id not in self.scalars:
            self.scalars[node.id] = self.add(
                node,
                "scalar_input",
                dtype=self.types.get(node.id, "unknown"),
                name=node.id,
                version=0,
            )
        return self.scalars[node.id]

    def access(self, node, write=False, value=None):
        buffer = expression_text(node.value)
        index = literal_index(node.slice)
        index_text = expression_text(node.slice)
        dependencies = [] if value is None else [value]
        simple_base = isinstance(node.value, ast.Name)
        if not simple_base:
            self.limitation(
                node, "Compound array base; alias identity unknown."
            )
            dependencies.append(self.expression(node.value))
        if index is None:
            dependencies.append(self.expression(node.slice))
            self.limitation(
                node, "Dynamic/slice index; exact element unknown."
            )
        key = (buffer, json.dumps(index)) if index is not None else None
        if key in self.buffer_stores:
            dependencies.append(self.buffer_stores[key])
        if index is None:
            dependencies.extend(
                value
                for (name, _), value in self.buffer_stores.items()
                if name == buffer
            )
        if buffer in self.dynamic_stores:
            dependencies.append(self.dynamic_stores[buffer])
        if write:
            dependencies.extend(
                event
                for (name, element), events in self.buffer_reads.items()
                if name == buffer
                and (index is None or element is None or element == key[1])
                for event in events
            )
        if buffer in self.buffer_events:
            event = self.buffer_events[buffer]
            if index is None or event["dynamic"]:
                dependencies.append(event["id"])
        dtype = self.types.get(buffer, "unknown")
        dtype = dtype.removesuffix("[]")
        result = self.add(
            node,
            "array_store" if write else "array_load",
            dependencies,
            dtype=dtype,
            buffer=buffer,
            index=index,
            index_source=index_text,
            constant_index=index is not None,
        )
        self.accesses.append(result)
        if write:
            if key is None:
                self.dynamic_stores[buffer] = result
                self.buffer_stores = {
                    k: v
                    for k, v in self.buffer_stores.items()
                    if k[0] != buffer
                }
            else:
                self.buffer_stores[key] = result
            self.buffer_reads = {
                read_key: events
                for read_key, events in self.buffer_reads.items()
                if read_key[0] != buffer
                or (index is not None and read_key[1] != key[1])
            }
        else:
            self.buffer_reads.setdefault(
                (buffer, key[1] if key else None), []
            ).append(result)
        self.buffer_events[buffer] = dict(id=result, dynamic=index is None)
        return result

    def expression(self, node):
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.Constant):
            return self.add(
                node,
                "literal",
                dtype=type(node.value).__name__,
                value=repr(node.value),
            )
        if isinstance(node, ast.Subscript):
            return self.access(node)
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            children = (
                [node.left, node.right]
                if isinstance(node, ast.BinOp)
                else [node.operand]
            )
            dependencies = [self.expression(child) for child in children]
            return self.add(
                node,
                "binary" if len(children) == 2 else "unary",
                dependencies,
                dtype=self.value_type(dependencies),
                operator=type(node.op).__name__,
            )
        if isinstance(node, ast.Call):
            name = expression_text(node.func)
            short = name.rsplit(".", 1)[-1]
            dependencies = [self.expression(arg) for arg in node.args]
            dependencies += [self.expression(kw.value) for kw in node.keywords]
            category, dtype = "opaque", "unknown"
            if short == "precision":
                category, dtype = "cast", self.precision or "precision"
            elif short in CASTS:
                category, dtype = "cast", short
            elif short in MATH_CALLS and (
                name.startswith("math.") or name in MATH_CALLS
            ):
                category = "math"
                dtype = (
                    "bool"
                    if short in {"isfinite", "isinf", "isnan"}
                    else self.value_type(dependencies)
                )
            elif short in SELECT_CALLS:
                category = "select"
                dtype = self.value_type(
                    dependencies[1:] if short == "selp" else dependencies
                )
            else:
                self.limitation(
                    node, "Opaque call; effects and latency unknown."
                )
                self.buffer_stores.clear()
            return self.add(
                node,
                "call",
                dependencies,
                dtype=dtype,
                callee=name,
                category=category,
            )
        if isinstance(node, ast.Compare):
            if len(node.ops) > 1:
                self.limitation(node, "Chained comparison short-circuits.")
            dependencies = [self.expression(node.left)]
            dependencies += [self.expression(v) for v in node.comparators]
            return self.add(
                node,
                "comparison",
                dependencies,
                dtype="bool",
                operators=[type(op).__name__ for op in node.ops],
            )
        if isinstance(node, (ast.IfExp, ast.BoolOp)):
            self.limitation(
                node,
                "Conditional expression; branches counted "
                "syntactically, not as an executed sequence.",
            )
        children = [
            child
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        ]
        dependencies = [self.expression(child) for child in children]
        self.limitation(
            node, "Expression requires external semantic handling."
        )
        return self.add(
            node,
            "unresolved_expression",
            dependencies,
            expression_kind=type(node).__name__,
            source=expression_text(node),
        )

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.versions[target.id] += 1
            self.scalars[target.id] = self.add(
                target,
                "scalar_definition",
                [value],
                dtype=self.nodes[value]["dtype"],
                name=target.id,
                version=self.versions[target.id],
            )
        elif isinstance(target, ast.Subscript):
            self.access(target, write=True, value=value)
        else:
            self.limitation(
                target, "Destructuring/attribute assignment unresolved."
            )
            self.add(
                target,
                "unresolved_assignment",
                [value],
                target=expression_text(target),
            )

    def statement(self, statement):
        if isinstance(statement, ast.Assign):
            value = self.expression(statement.value)
            for target in statement.targets:
                self.assign(target, value)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self.assign(statement.target, self.expression(statement.value))
        elif isinstance(statement, ast.AugAssign):
            before = self.expression(statement.target)
            value = self.expression(statement.value)
            result = self.add(
                statement,
                "binary",
                [before, value],
                dtype=self.value_type([before, value]),
                operator=type(statement.op).__name__,
            )
            self.assign(statement.target, result)
        elif isinstance(statement, (ast.Return, ast.Expr)):
            if statement.value is not None and not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                value = self.expression(statement.value)
                if isinstance(statement, ast.Return):
                    self.add(statement, "return", [value])
        elif not isinstance(statement, ast.Pass):
            self.limitation(statement, "Statement semantics are not modelled.")

    def finish(self):
        consumers = {node["id"]: [] for node in self.nodes}
        for node in self.nodes:
            for dependency in node["dependencies"]:
                consumers[dependency].append(node["id"])
        scalars = [
            node
            for node in self.nodes
            if node["kind"] in {"scalar_input", "scalar_definition"}
        ]
        intervals = []
        for node in scalars:
            uses = consumers[node["id"]]
            if uses:
                intervals.append(
                    dict(
                        id=node["id"],
                        name=node["name"],
                        version=node["version"],
                        start=0
                        if node["kind"] == "scalar_input"
                        else node["id"],
                        last_use=max(uses),
                        uses=uses,
                        live_in=node["kind"] == "scalar_input",
                    )
                )
        starts, ends = {}, {}
        for interval in intervals:
            starts.setdefault(interval["start"], []).append(interval["id"])
            ends.setdefault(interval["last_use"] + 1, []).append(
                interval["id"]
            )
        active = set()
        profile = []
        for node in self.nodes:
            position = node["id"]
            active.difference_update(ends.get(position, ()))
            active.update(starts.get(position, ()))
            profile.append(len(active))
            if node["kind"] in {"call", "array_load", "array_store"}:
                node["live_scalar_ids"] = sorted(active)
        arrays = {}
        for event_id in self.accesses:
            event = self.nodes[event_id]
            row = arrays.setdefault(
                event["buffer"],
                {
                    "reads": 0,
                    "writes": 0,
                    "constant_reads": 0,
                    "constant_writes": 0,
                    "dynamic_reads": 0,
                    "dynamic_writes": 0,
                    "elements": {},
                },
            )
            action = "reads" if event["kind"] == "array_load" else "writes"
            row[action] += 1
            prefix = "constant" if event["constant_index"] else "dynamic"
            row[f"{prefix}_{action}"] += 1
            element = row["elements"].setdefault(
                event["index_source"],
                {
                    "index": event["index"],
                    "constant": event["constant_index"],
                    "reads": 0,
                    "writes": 0,
                },
            )
            element[action] += 1
        typed = Counter()
        for node in self.nodes:
            if node["kind"] in SOURCE_KINDS:
                continue
            key = (
                node["kind"],
                node.get("operator", node.get("callee", "")),
                tuple(self.nodes[i]["dtype"] for i in node["dependencies"]),
                node["dtype"],
            )
            typed[key] += 1
        return {
            "id": self.region_id,
            "context": self.context,
            "nodes": self.nodes,
            "scalar_intervals": intervals,
            "source_order_scalar_liveness": {
                "peak": max(profile, default=0),
                "area": sum(profile),
                "positions": len(profile),
                "profile": profile,
                "unit": "named scalar versions at syntactic operation boundaries",
                "includes_scalar_live_ins": True,
                "is_register_estimate": False,
            },
            "critical_path_depth": max(
                (n["depth"] for n in self.nodes), default=0
            ),
            "typed_operations": [
                dict(
                    kind=k[0],
                    operator=k[1],
                    operand_types=list(k[2]),
                    result_type=k[3],
                    count=count,
                )
                for k, count in sorted(typed.items())
            ],
            "buffers": arrays,
            "limitations": self.limitations,
            "dependency_scope": "within region; same named buffer only",
        }


def analyse_function(
    function, qualified_name, guards=(), precision=None, argument_types=None
):
    """Describe a function's syntax and separately scoped dependency DAGs.

    Parameters
    ----------
    function : ast.FunctionDef
        Function in an already parsed source module.
    qualified_name : str
        Lexical name; source line disambiguates conditional alternatives.
    guards : tuple
        Enclosing source branches, retained without evaluation.
    precision : str, optional
        Explicit meaning of the generated ``precision(...)`` cast.
    argument_types : dict, optional
        Caller-supplied scalar/array types. Missing types stay unknown.

    Returns
    -------
    dict
        JSON-serializable descriptors without runtime or SASS estimates.
    """
    regions, controls = [], []
    types = argument_types or {}

    def visit(statements, context):
        region = None
        for statement in statements:
            simple = isinstance(
                statement,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                    ast.Expr,
                    ast.Return,
                    ast.Pass,
                ),
            )
            if simple:
                if region is None:
                    region = Region(len(regions), context, types, precision)
                    regions.append(region)
                region.statement(statement)
                if isinstance(statement, ast.Return):
                    region = None
                continue
            region = None
            control = {
                "kind": type(statement).__name__,
                "line": statement.lineno,
                "condition": expression_text(getattr(statement, "test", None)),
                "execution_multiplicity": None,
                "source": expression_text(statement),
            }
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                control.update(
                    nested_function=statement.name,
                    handling="analysed as separate function",
                )
            else:
                control["handling"] = "separate lexical blocks; no join graph"
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                control.update(
                    target=expression_text(statement.target),
                    iterable=expression_text(statement.iter),
                    literal_range_trip_count=static_range_count(
                        statement.iter
                    ),
                )
            controls.append(control)
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            for field, value in ast.iter_fields(statement):
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, ast.stmt) for item in value)
                ):
                    visit(
                        value,
                        context
                        + [
                            {
                                "line": statement.lineno,
                                "kind": type(statement).__name__,
                                "branch": field,
                            }
                        ],
                    )
                elif isinstance(value, ast.expr) and field != "target":
                    header = Region(
                        len(regions),
                        context
                        + [
                            {
                                "line": statement.lineno,
                                "kind": "control_header",
                                "field": field,
                            }
                        ],
                        types,
                        precision,
                    )
                    regions.append(header)
                    header.expression(value)
                elif isinstance(value, list):
                    for child in value:
                        if isinstance(
                            child, (ast.ExceptHandler, ast.match_case)
                        ):
                            visit(
                                child.body,
                                context
                                + [
                                    {
                                        "line": getattr(
                                            child, "lineno", statement.lineno
                                        ),
                                        "kind": type(child).__name__,
                                        "branch": field,
                                    }
                                ],
                            )

    visit(function.body, list(guards))
    finished = [region.finish() for region in regions]
    decorators = [expression_text(item) for item in function.decorator_list]
    return {
        "id": f"{qualified_name}@{function.lineno}",
        "qualified_name": qualified_name,
        "line": function.lineno,
        "end_line": function.end_lineno,
        "enclosing_guards": list(guards),
        "decorators": decorators,
        "device_decorator": any("cuda.jit" in item for item in decorators),
        "arguments": [
            arg.arg
            for arg in function.args.posonlyargs
            + function.args.args
            + function.args.kwonlyargs
        ],
        "declared_argument_types": types,
        "syntax_counts": syntax_counts(function),
        "controls": controls,
        "regions": finished,
        "dynamic_execution_model_complete": not controls
        and not any(region["limitations"] for region in finished),
        "cross_region_dependencies_modelled": False,
        "peak_region_scalar_liveness": max(
            (r["source_order_scalar_liveness"]["peak"] for r in finished),
            default=0,
        ),
        "maximum_region_dependency_depth": max(
            (r["critical_path_depth"] for r in finished), default=0
        ),
    }


def describe_source(
    path, precision=None, argument_types=None, function_filter=None
):
    """Read generated Python and return static descriptors and provenance.

    Parameters
    ----------
    path : str or pathlib.Path
        Source file, read without imports or evaluation.
    precision : str, optional
        Explicit dtype for ``precision`` casts, not inferred from timings.
    argument_types : dict, optional
        Explicit name-to-type map; array entries may end in ``[]``.
    function_filter : str, optional
        Substring selecting qualified function names.

    Returns
    -------
    dict
        Source identities, analysis limitations and per-function graphs.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    source = raw.decode("utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    functions = [
        analyse_function(node, name, guards, precision, argument_types)
        for name, guards, node in functions_in(tree.body)
        if function_filter is None or function_filter in name
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "static_source_descriptors",
        "provenance": {
            "source_path": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_bytes": len(raw),
            "precision_cast_dtype": precision,
            "argument_types": argument_types or {},
            "function_filter": function_filter,
        },
        "semantics": {
            "counts": "written syntax once; no loop or call expansion",
            "dependencies": "source expression order, scalar versions and "
            "same-buffer literal-index read-after-write edges",
            "depth": "unit operation-node dependency depth; literals, inputs "
            "and scalar copies have zero depth cost; not latency",
            "liveness": "within-region named scalar versions, not compiler "
            "allocation or intermediate expression registers",
            "dtype": "explicit casts/type inputs only; unknown propagation "
            "and compiler promotion require compiler/type evidence",
        },
        "limitations": [
            "No constant folding, FMA contraction, compiler reordering, CSE, "
            "inlining, register allocation or SASS translation is inferred.",
            "No dynamic execution counts, branch probabilities, loop-carried "
            "dependencies, cross-region phi values or runtime trace is inferred.",
            "Different buffer names may alias; external alias information is "
            "required before using memory edges as a complete dependence DAG.",
            "Factory closure values and conditional function alternatives are "
            "recorded but not resolved. Select the actual generated helper.",
        ],
        "functions": functions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--precision")
    parser.add_argument(
        "--argument-types",
        type=Path,
        help="JSON name-to-dtype map, supplied as evidence",
    )
    parser.add_argument("--function", help="Qualified-name substring")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    types = (
        json.loads(args.argument_types.read_text(encoding="utf-8"))
        if args.argument_types
        else None
    )
    result = describe_source(args.source, args.precision, types, args.function)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
