"""Complete-source local-array admission for conditional scalarization."""

import ast
import hashlib
from pathlib import Path


UNKNOWN = object()


def runtime_index_markers(count, directive):
    """Mirror source loop_structure's main/tail constant-index algebra."""
    if directive is True:
        directive = [True, None]
    elif directive is False or directive is None:
        directive = [False, None]
    directive = tuple(directive)
    if directive == (True, None):
        return [False] * count
    if directive == (False, None):
        return [True] * count
    if (
        directive[0] is not True
        or isinstance(directive[1], bool)
        or directive[1] not in (1, 2, 4)
    ):
        raise ValueError("Unsupported source unroll directive")
    copies = directive[1]
    quotient, remainder = divmod(count, copies)
    return [quotient != 1] * (quotient * copies) + [False] * remainder


class Index(int):
    """Exact source index with a compiler-runtime dependence marker."""

    def __new__(cls, value, dynamic=False):
        if not -(2**31) <= value < 2**31:
            raise ValueError("Source scalarization index exceeds int32")
        result = int.__new__(cls, value)
        result.dynamic = dynamic
        return result

    def combine(self, other, operation):
        if not isinstance(other, int):
            return UNKNOWN
        return Index(
            operation(int(self), int(other)),
            self.dynamic or getattr(other, "dynamic", False),
        )

    def __add__(self, other):
        return self.combine(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self.combine(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self.combine(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self.combine(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return self.combine(other, lambda a, b: a // b)

    def __neg__(self):
        return Index(-int(self), self.dynamic)


class View:
    """Finite source array view expressed as whole-allocation cell indices."""

    def __init__(self, cells, dynamic=False):
        self.cells = tuple(cells)
        self.dynamic = dynamic


def contains_view(value):
    """Find selected aliases even inside unsupported compound arguments."""
    return isinstance(value, View) or (
        isinstance(value, (list, tuple))
        and any(contains_view(item) for item in value)
    )


def constant(value):
    """Decode only bound compile-time values, never replay scalar inputs."""
    if isinstance(value, dict):
        if "value" in value:
            return value["value"]
        if value.get("kind") == "array":
            return value["values"]
        return UNKNOWN
    if isinstance(value, (bool, int, float, type(None))):
        return value
    return UNKNOWN


class SourceAdmission:
    """Check source control, alias accesses, bounds, and initialization."""

    def __init__(self, graph, allocation):
        self.graph = graph
        self.allocation = allocation
        self.storage = allocation["view"]["storage"]
        self.cells = allocation["view"]["bytes"] // 4
        self.accesses = []
        self.conditions = []
        self.loop_forms = []
        self.sources = {}
        self.stack = []
        self.calls = [
            call for call in graph["calls"] if call["kind"] == "source_call"
        ]
        self.functions = {
            item["id"]: item for item in graph["provenance"]["functions"]
        }

    def function(self, call):
        """Read the complete bound function AST from its source file."""
        source = call["source"]
        path = Path(source["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError("Scalarization source file identity differs")
        self.sources[str(path)] = source["sha256"]
        name = call["context"].split(":", 1)[1].split(".")[-1]
        choices = [
            node
            for node in ast.walk(ast.parse(data))
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(choices) != 1:
            raise ValueError("Scalarization function AST is not unique")
        return choices[0]

    def access(self, view, index, initialized, mode, node, call):
        """Prove and record an actual source access to a selected home."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Complete-source index is not a bounded integer")
        if not 0 <= index < len(view.cells):
            raise ValueError("Complete-source index leaves its alias extent")
        cell = view.cells[index]
        dynamic = view.dynamic or getattr(index, "dynamic", False)
        if dynamic and len(initialized) != self.cells:
            raise ValueError(
                "Dynamic selection precedes complete source initialization"
            )
        if mode == "read" and cell not in initialized:
            raise ValueError("Source path reads a home before initialization")
        if mode == "write":
            initialized.add(cell)
        self.accesses.append(
            {
                "function": call["function"],
                "line": node.lineno,
                "mode": mode,
                "cell": cell,
                "dynamic_index": dynamic,
                "syntax": ast.unparse(node),
            }
        )
        return UNKNOWN

    def expression(self, node, env, initialized, call):
        """Evaluate source constants while checking every selected access."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, UNKNOWN)
        if isinstance(node, ast.Subscript):
            view = self.expression(node.value, env, initialized, call)
            if isinstance(node.slice, ast.Slice):
                parts = [
                    self.expression(part, env, initialized, call)
                    if part
                    else None
                    for part in (
                        node.slice.lower,
                        node.slice.upper,
                        node.slice.step,
                    )
                ]
                if isinstance(view, View):
                    if any(part is UNKNOWN for part in parts):
                        raise ValueError("Selected slice lacks source bounds")
                    start, stop, step = parts
                    start = 0 if start is None else start
                    stop = len(view.cells) if stop is None else stop
                    step = 1 if step is None else step
                    if not (
                        step == 1 and 0 <= start <= stop <= len(view.cells)
                    ):
                        raise ValueError("Selected slice leaves whole extent")
                    return View(
                        view.cells[start:stop],
                        view.dynamic
                        or getattr(start, "dynamic", False)
                        or getattr(stop, "dynamic", False),
                    )
                return UNKNOWN
            index = self.expression(node.slice, env, initialized, call)
            if isinstance(view, View):
                return self.access(
                    view, index, initialized, "read", node, call
                )
            if isinstance(view, (list, tuple)) and isinstance(index, int):
                result = view[index]
                if isinstance(result, int):
                    return Index(result, getattr(index, "dynamic", False))
                return result
            return UNKNOWN
        if isinstance(node, (ast.Tuple, ast.List)):
            return [
                self.expression(item, env, initialized, call)
                for item in node.elts
            ]
        if isinstance(node, ast.Call):
            callable_value = self.expression(node.func, env, initialized, call)
            if contains_view(callable_value):
                raise ValueError(
                    "Selected array is used as a callable receiver"
                )
            arguments = [
                self.expression(arg, env, initialized, call)
                for arg in node.args
            ]
            keywords = {
                item.arg: self.expression(item.value, env, initialized, call)
                for item in node.keywords
            }
            name = ast.unparse(node.func)
            if any(
                contains_view(value)
                for value in arguments + list(keywords.values())
            ):
                if any(
                    contains_view(value) and not isinstance(value, View)
                    for value in arguments + list(keywords.values())
                ):
                    raise ValueError(
                        "Compound selected-array escape is unsupported"
                    )
                enclosing = self.function(call)
                if any(
                    isinstance(item, ast.Name)
                    and isinstance(item.ctx, ast.Store)
                    and item.id == name
                    for item in ast.walk(enclosing)
                ):
                    raise ValueError(
                        "Selected helper name is rebound in source"
                    )
                candidates = [
                    item
                    for item in self.functions[call["function"]]["calls"]
                    if item["binding"] == name
                    and item["source"]["line"] == node.lineno
                ]
                if not candidates:
                    raise ValueError(
                        "Selected array escapes to unbound helper"
                    )
                identities = {item["callee"] for item in candidates}
                if len(identities) != 1:
                    raise ValueError("Selected helper source binding differs")
                descriptor = self.functions[next(iter(identities))]
                target = dict(
                    function=descriptor["id"],
                    context="source:" + descriptor["qualified_name"],
                    source=dict(
                        path=descriptor["source"]["source_path"],
                        sha256=descriptor["source"]["source_sha256"],
                    ),
                    closure_constants=descriptor["closure_constants"],
                )
                function = self.function(target)
                child = {
                    key: constant(value)
                    for key, value in target["closure_constants"].items()
                }
                child.update(
                    dict(
                        zip([arg.arg for arg in function.args.args], arguments)
                    )
                )
                child.update(keywords)
                self.execute_function(target, function, child, initialized)
                return UNKNOWN
            if (
                name in ("int32", "uint32", "int", "bool")
                and len(arguments) == 1
            ):
                value = arguments[0]
                if value is not UNKNOWN:
                    return (
                        bool(value)
                        if name == "bool"
                        else Index(
                            int(value), getattr(value, "dynamic", False)
                        )
                    )
            if name == "range" and all(
                isinstance(arg, int) for arg in arguments
            ):
                return range(*arguments)
            if name == "unroll_if" and arguments:
                return arguments[0]
            return UNKNOWN
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand, env, initialized, call)
            if value is UNKNOWN:
                return UNKNOWN
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left = self.expression(node.left, env, initialized, call)
            right = self.expression(node.right, env, initialized, call)
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.FloorDiv: lambda: left // right,
            }
            if type(node.op) in operations:
                result = operations[type(node.op)]()
                if isinstance(result, int) and not -(2**31) <= result < 2**31:
                    raise ValueError("Source integer arithmetic exceeds int32")
                return result
            return UNKNOWN
        conditional_expression = (
            isinstance(node, (ast.BoolOp, ast.IfExp))
            or isinstance(node, ast.Compare)
            and len(node.ops) > 1
        )
        if conditional_expression and any(
            isinstance(part, ast.Call) for part in ast.walk(node)
        ):
            raise ValueError(
                "Call-bearing short-circuit expression is unsupported"
            )
        if isinstance(node, ast.Compare):
            values = [
                self.expression(part, env, initialized, call)
                for part in [node.left] + node.comparators
            ]
            if any(value is UNKNOWN for value in values):
                return UNKNOWN
            for left, right, operation in zip(values, values[1:], node.ops):
                operations = {
                    ast.Eq: lambda: left == right,
                    ast.NotEq: lambda: left != right,
                    ast.Lt: lambda: left < right,
                    ast.LtE: lambda: left <= right,
                    ast.Gt: lambda: left > right,
                    ast.GtE: lambda: left >= right,
                    ast.Is: lambda: left is right,
                    ast.IsNot: lambda: left is not right,
                }
                if type(operation) not in operations:
                    return UNKNOWN
                if not operations[type(operation)]():
                    return False
            return True
        if isinstance(node, ast.BoolOp):
            values = [
                self.expression(part, env, initialized, call)
                for part in node.values
            ]
            if isinstance(node.op, ast.And) and any(
                value is False for value in values
            ):
                return False
            if isinstance(node.op, ast.Or) and any(
                value is True for value in values
            ):
                return True
            if any(value is UNKNOWN for value in values):
                return UNKNOWN
            return all(values) if isinstance(node.op, ast.And) else any(values)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                value = self.expression(child, env, initialized, call)
                if isinstance(value, View):
                    raise ValueError(
                        "Selected array has unsupported source use"
                    )
        return UNKNOWN

    def assign(self, target, value, env, initialized, call):
        """Track whole alias assignment and exact indexed memory writes."""
        if isinstance(value, (list, tuple)) and contains_view(value):
            raise ValueError(
                "Selected alias containers are outside supported source forms"
            )
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, ast.Subscript):
            if contains_view(value):
                raise ValueError("Array-valued scalar store is unsupported")
            view = self.expression(target.value, env, initialized, call)
            index = self.expression(target.slice, env, initialized, call)
            if isinstance(view, View):
                self.access(view, index, initialized, "write", target, call)
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = (
                value
                if isinstance(value, (list, tuple))
                and len(value) == len(target.elts)
                else [UNKNOWN] * len(target.elts)
            )
            if contains_view(value) and not isinstance(value, (list, tuple)):
                raise ValueError("Array unpacking is unsupported")
            for part, item in zip(target.elts, values):
                self.assign(part, item, env, initialized, call)
        else:
            raise ValueError("Unsupported assignment in complete-source scan")

    def statements(self, statements, env, initialized, call):
        """Visit both runtime arms and every source-counted loop iteration."""
        for node in statements:
            if isinstance(node, ast.Assign):
                if node.lineno == self.allocation["source"]["line"] and (
                    call["context"] == self.allocation["source"]["context"]
                ):
                    value = View(range(self.cells))
                else:
                    value = self.expression(node.value, env, initialized, call)
                for target in node.targets:
                    self.assign(target, value, env, initialized, call)
            elif isinstance(node, ast.AugAssign):
                value = self.expression(
                    ast.BinOp(left=node.target, op=node.op, right=node.value),
                    env,
                    initialized,
                    call,
                )
                self.assign(node.target, value, env, initialized, call)
            elif isinstance(node, ast.Expr):
                self.expression(node.value, env, initialized, call)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test, env, initialized, call)
                self.conditions.append(
                    {
                        "function": call["function"],
                        "line": node.lineno,
                        "condition": ast.unparse(node.test),
                        "value": None
                        if condition is UNKNOWN
                        else bool(condition),
                        "both_arms": condition is UNKNOWN,
                    }
                )
                if condition is not UNKNOWN:
                    self.statements(
                        node.body if condition else node.orelse,
                        env,
                        initialized,
                        call,
                    )
                else:
                    left, right = dict(env), dict(env)
                    left_init, right_init = set(initialized), set(initialized)
                    self.statements(node.body, left, left_init, call)
                    self.statements(node.orelse, right, right_init, call)
                    initialized.intersection_update(left_init & right_init)
                    initialized.update(left_init & right_init)
                    for key in set(left) | set(right):
                        a, b = left.get(key, UNKNOWN), right.get(key, UNKNOWN)
                        equal = a is b or (
                            isinstance(a, View)
                            and isinstance(b, View)
                            and a.cells == b.cells
                            and a.dynamic == b.dynamic
                        )
                        if not isinstance(a, View) and not isinstance(b, View):
                            equal = a is b or (
                                a is not UNKNOWN
                                and b is not UNKNOWN
                                and a == b
                            )
                        if (
                            contains_view(a) or contains_view(b)
                        ) and not equal:
                            raise ValueError(
                                "Runtime branch changes selected alias binding"
                            )
                        env[key] = a if equal else UNKNOWN
            elif isinstance(node, ast.For):
                iterations = self.expression(node.iter, env, initialized, call)
                if not isinstance(iterations, range):
                    raise ValueError("Complete-source loop lacks fixed range")
                directive = None
                if (
                    isinstance(node.iter, ast.Call)
                    and ast.unparse(node.iter.func) == "unroll_if"
                ):
                    flag = node.iter.args[1]
                    if isinstance(flag, ast.Name):
                        directive = call["closure_constants"].get(flag.id)
                    else:
                        try:
                            directive = ast.literal_eval(flag)
                        except (ValueError, TypeError):
                            raise ValueError(
                                "Unroll directive is not source-constant"
                            ) from None
                markers = runtime_index_markers(len(iterations), directive)
                self.loop_forms.append(
                    dict(
                        function=call["function"],
                        line=node.lineno,
                        source_range=[
                            iterations.start,
                            iterations.stop,
                            iterations.step,
                        ],
                        directive=directive,
                        runtime_index_markers=markers,
                        rule="policy.loop_structure counted main/tail algebra",
                    )
                )
                for position, index in enumerate(iterations):
                    self.assign(
                        node.target,
                        Index(index, markers[position]),
                        env,
                        initialized,
                        call,
                    )
                    self.statements(node.body, env, initialized, call)
                self.statements(node.orelse, env, initialized, call)
            elif isinstance(node, ast.Return):
                raise ValueError(
                    "Early return needs explicit source-flow proof"
                )
            elif isinstance(node, ast.FunctionDef):
                names = {
                    key for key, value in env.items() if contains_view(value)
                }
                if any(
                    isinstance(item, ast.Name) and item.id in names
                    for item in ast.walk(node)
                ):
                    raise ValueError("Nested function captures selected alias")
            elif isinstance(node, ast.Pass):
                continue
            else:
                raise ValueError(
                    "Unsupported complete-source control: "
                    + type(node).__name__
                )

    def execute_function(self, call, function, env, initialized):
        """Scan a bound helper without using numerical replay inputs."""
        if call["function"] in self.stack:
            raise ValueError("Recursive selected-array helper is unsupported")
        self.stack.append(call["function"])
        try:
            body = function.body
            last = body[-1] if body else None
            self.statements(
                body[:-1] if isinstance(last, ast.Return) else body,
                env,
                initialized,
                call,
            )
            if isinstance(last, ast.Return) and last.value:
                value = self.expression(last.value, env, initialized, call)
                if contains_view(value):
                    raise ValueError("Selected array escapes through return")
        finally:
            self.stack.pop()

    def run(self):
        """Return an independently reproducible complete-source receipt."""
        call = next(
            item
            for item in self.calls
            if item["context"] == self.allocation["source"]["context"]
        )
        function = self.function(call)
        env = {
            key: constant(value)
            for key, value in call["closure_constants"].items()
        }
        env.update({argument.arg: UNKNOWN for argument in function.args.args})
        initialized = set()
        self.execute_function(call, function, env, initialized)
        return {
            "status": "COMPLETE_SOURCE_ARRAY_ADMISSION_PASS",
            "sources": self.sources,
            "storage": self.storage,
            "homes": self.cells,
            "accesses": self.accesses,
            "source_dynamic_accesses": sum(
                item["dynamic_index"] for item in self.accesses
            ),
            "conditions": self.conditions,
            "source_loop_forms": self.loop_forms,
            "entry_scalars": "unknown, not numerical replay values",
            "zero_trip": "exact source range; no initialization credited",
        }


def source_admission(graph, allocation):
    """Prove complete source array eligibility for a conditional form."""
    return SourceAdmission(graph, allocation).run()
