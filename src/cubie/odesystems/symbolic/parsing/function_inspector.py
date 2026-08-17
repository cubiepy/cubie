"""Inspect Python callables to extract ODE structure via AST analysis.

Walks the abstract syntax tree of a user-provided function to identify
state accesses, constant/parameter accesses, local assignments, return
expressions, and function calls. Converts AST nodes to SymPy expressions.

Published Functions
-------------------
:func:`inspect_ode_function`
    Analyse a callable and return a :class:`FunctionInspection` dataclass.

Published Classes
-----------------
:class:`FunctionInspection`
    Result container holding parsed AST metadata.

:class:`AstToSympyConverter`
    Recursive converter from :mod:`ast` nodes to :mod:`sympy` expressions.
"""

import ast
import copy
import inspect
import textwrap
import warnings
from typing import Any, Callable, Dict, List, Optional, Set

import sympy as sp

from .parse_primitives import KNOWN_FUNCTIONS

# Map of module-qualified names to their bare equivalents
_MODULE_PREFIXES = {"math", "np", "numpy", "cmath"}

# Augmented-assignment operators → BinOp operator constructors
_AUGOP_TO_BINOP = {
    ast.Add: ast.Add,
    ast.Sub: ast.Sub,
    ast.Mult: ast.Mult,
    ast.Div: ast.Div,
    ast.FloorDiv: ast.FloorDiv,
    ast.Pow: ast.Pow,
    ast.Mod: ast.Mod,
}


class FunctionInspection:
    """Result of inspecting an ODE function's AST.

    Parameters
    ----------
    param_names
        All parameter names from the function signature.
    state_param
        Name of the state vector parameter (second positional arg).
    constant_params
        Names of constant/parameter arguments (third+ positional args).
    scalar_params
        Subset of ``constant_params`` never accessed as containers
        (no subscript or attribute access); each is treated as a
        scalar bound to the like-named declared symbol, matching
        SciPy's ``args=`` convention.
    state_accesses
        List of dicts with keys ``base``, ``key``, ``pattern_type``.
    constant_accesses
        List of dicts with keys ``base``, ``key``, ``pattern_type``.
    assignments
        Mapping of local variable names to their AST expression nodes.
    return_node
        The AST Return node found in the function body.
    function_calls
        Set of function names invoked in the body.
    func_def
        The parsed :class:`ast.FunctionDef` node.
    """

    def __init__(
        self,
        param_names: List[str],
        state_param: str,
        constant_params: List[str],
        state_accesses: List[Dict[str, Any]],
        constant_accesses: List[Dict[str, Any]],
        assignments: Dict[str, ast.expr],
        return_node: ast.Return,
        function_calls: Set[str],
        func_def: ast.FunctionDef,
        scalar_params: Optional[List[str]] = None,
    ) -> None:
        self.param_names = param_names
        self.state_param = state_param
        self.constant_params = constant_params
        self.scalar_params = scalar_params or []
        self.state_accesses = state_accesses
        self.constant_accesses = constant_accesses
        self.assignments = assignments
        self.return_node = return_node
        self.function_calls = function_calls
        self.func_def = func_def


class _OdeAstVisitor(ast.NodeVisitor):
    """Walk an ODE function body collecting access patterns."""

    def __init__(
        self, state_param: str, constant_params: List[str]
    ) -> None:
        self.state_param = state_param
        self.constant_params = constant_params
        self.state_accesses: List[Dict[str, Any]] = []
        self.constant_accesses: List[Dict[str, Any]] = []
        self.assignments: Dict[str, ast.expr] = {}
        self.return_nodes: List[ast.Return] = []
        self.function_calls: Set[str] = set()
        self._entered_function = False

    def _record_access(
        self, base: str, key: Any, ptype: str
    ) -> None:
        entry = {"base": base, "key": key, "pattern_type": ptype}
        if base == self.state_param:
            self.state_accesses.append(entry)
        elif base in self.constant_params:
            self.constant_accesses.append(entry)

    def _record_subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.value, ast.Name):
            return
        slc = node.slice
        if isinstance(slc, ast.Constant):
            key = slc.value
            ptype = "int" if isinstance(key, int) else "string"
        elif isinstance(slc, ast.Name):
            key = slc.id
            ptype = "name"
        else:
            key = ast.dump(slc)
            ptype = "expr"
        self._record_access(node.value.id, key, ptype)

    def _record_attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            self._record_access(node.value.id, node.attr, "attribute")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._record_subscript(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._record_attribute(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id] = node.value
            elif isinstance(target, (ast.Tuple, ast.List)):
                if isinstance(node.value, (ast.Tuple, ast.List)):
                    # Positional unpacking: a, b = expr1, expr2
                    if len(target.elts) == len(node.value.elts):
                        for tgt, val in zip(
                            target.elts, node.value.elts
                        ):
                            if isinstance(tgt, ast.Name):
                                self.assignments[tgt.id] = val
                    else:
                        raise ValueError(
                            f"Tuple unpacking length mismatch: "
                            f"{len(target.elts)} targets, "
                            f"{len(node.value.elts)} values"
                        )
                else:
                    # RHS is not a tuple/list (e.g. function call)
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.assignments[elt.id] = node.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(
            node.target, ast.Name
        ):
            self.assignments[node.target.id] = node.value
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            name = node.target.id
            binop_cls = _AUGOP_TO_BINOP.get(type(node.op))
            if binop_cls is None:
                raise NotImplementedError(
                    f"Unsupported augmented assignment operator: "
                    f"{type(node.op).__name__}"
                )
            prior = self.assignments.get(name)
            if prior is None:
                raise ValueError(
                    f"Augmented assignment to '{name}' with no "
                    f"prior assignment"
                )
            combined = ast.BinOp(
                left=prior, op=binop_cls(), right=node.value
            )
            ast.copy_location(combined, node)
            self.assignments[name] = combined
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_nodes.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name:
            self.function_calls.add(name)
        self.generic_visit(node)

    # -- If/elif/else → IfExp synthesis --------------------------------

    def visit_If(self, node: ast.If) -> None:
        """Convert if/elif/else assignment blocks to ``ast.IfExp`` nodes.

        Intercepts the ``If`` node so that ``generic_visit`` does not
        recurse into both branches and silently overwrite assignments.
        Instead, assignments from each branch are collected and merged
        into ``IfExp`` (ternary) nodes that the downstream converter
        maps to ``sp.Piecewise``.
        """
        if_assigns = self._collect_branch_assignments(node.body)
        else_assigns = self._collect_branch_assignments(node.orelse)

        all_names = set(if_assigns.keys()) | set(else_assigns.keys())

        for name in all_names:
            if_val = if_assigns.get(name)
            else_val = else_assigns.get(name)

            if if_val is not None and else_val is not None:
                ifexp = ast.IfExp(
                    test=node.test, body=if_val, orelse=else_val
                )
            elif if_val is not None:
                fallback = self.assignments.get(name)
                if fallback is None:
                    raise ValueError(
                        f"Variable '{name}' assigned in if-branch "
                        f"but has no prior value and no else-branch. "
                        f"Add an else clause or a default assignment "
                        f"before the if statement."
                    )
                ifexp = ast.IfExp(
                    test=node.test, body=if_val, orelse=fallback
                )
            else:
                fallback = self.assignments.get(name)
                if fallback is None:
                    raise ValueError(
                        f"Variable '{name}' assigned in else-branch "
                        f"but has no prior value and no if-branch."
                    )
                ifexp = ast.IfExp(
                    test=node.test, body=fallback, orelse=else_val
                )

            ast.copy_location(ifexp, node)
            self.assignments[name] = ifexp

        # Visit expressions for state/constant accesses and calls.
        self._visit_exprs_in_stmts([node])

    def _visit_exprs_in_stmts(self, stmts: list) -> None:
        """Visit expression sub-trees for accesses and calls only.

        For-loops are unrolled first so that recorded accesses carry
        the substituted constant subscripts (``y[0]``, not ``y[i]``),
        matching what :meth:`_collect_branch_assignments` converts.
        """
        for stmt in self._flatten_for_loops(stmts):
            if isinstance(stmt, ast.If):
                self._visit_expr(stmt.test)
                self._visit_exprs_in_stmts(stmt.body + stmt.orelse)
            elif isinstance(stmt, ast.Assign):
                self._visit_expr(stmt.value)
            elif isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None:
                    self._visit_expr(stmt.value)
            elif isinstance(stmt, ast.AugAssign):
                self._visit_expr(stmt.value)
            elif isinstance(stmt, ast.Expr):
                self._visit_expr(stmt.value)

    def _visit_expr(self, node: ast.expr) -> None:
        """Visit an expression sub-tree to capture accesses and calls."""
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                # walk already recurses, so record without
                # generic_visit.
                self._record_subscript(child)
            elif isinstance(child, ast.Attribute):
                self._record_attribute(child)
            elif isinstance(child, ast.Call):
                cname = _call_name(child)
                if cname:
                    self.function_calls.add(cname)
            elif isinstance(child, ast.NamedExpr):
                if isinstance(child.target, ast.Name):
                    self.assignments[child.target.id] = child.value

    def _collect_branch_assignments(
        self, stmts: List[ast.stmt]
    ) -> Dict[str, ast.expr]:
        """Extract assignments from a branch body without side effects.

        Handles ``Assign``, ``AnnAssign``, ``AugAssign``, nested ``If``
        (for elif chains), and ``For`` (unrolled before processing).
        Statements with no branch-expression equivalent (``return``,
        ``while``, ``with``, ...) raise.  Returns
        ``{name: ast_expression_node}``.
        """
        branch: Dict[str, ast.expr] = {}
        for stmt in self._flatten_for_loops(stmts):
            if isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None and isinstance(
                    stmt.target, ast.Name
                ):
                    branch[stmt.target.id] = stmt.value
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        branch[target.id] = stmt.value
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        if isinstance(
                            stmt.value, (ast.Tuple, ast.List)
                        ) and len(target.elts) == len(
                            stmt.value.elts
                        ):
                            for tgt, val in zip(
                                target.elts, stmt.value.elts
                            ):
                                if isinstance(tgt, ast.Name):
                                    branch[tgt.id] = val
                        else:
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    branch[elt.id] = stmt.value

            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    binop_cls = _AUGOP_TO_BINOP.get(type(stmt.op))
                    if binop_cls is None:
                        raise NotImplementedError(
                            f"Unsupported augmented assignment "
                            f"operator: {type(stmt.op).__name__}"
                        )
                    prior = branch.get(
                        name, self.assignments.get(name)
                    )
                    if prior is None:
                        raise ValueError(
                            f"Augmented assignment to '{name}' "
                            f"with no prior assignment"
                        )
                    combined = ast.BinOp(
                        left=prior,
                        op=binop_cls(),
                        right=stmt.value,
                    )
                    ast.copy_location(combined, stmt)
                    branch[name] = combined

            elif isinstance(stmt, ast.If):
                # Nested if (elif chain) — recurse
                nested_if = self._collect_branch_assignments(
                    stmt.body
                )
                nested_else = self._collect_branch_assignments(
                    stmt.orelse
                )
                nested_names = (
                    set(nested_if.keys()) | set(nested_else.keys())
                )
                for name in nested_names:
                    nif = nested_if.get(name)
                    nelse = nested_else.get(name)
                    if nif is not None and nelse is not None:
                        ifexp = ast.IfExp(
                            test=stmt.test, body=nif, orelse=nelse
                        )
                    elif nif is not None:
                        fallback = branch.get(
                            name, self.assignments.get(name)
                        )
                        if fallback is None:
                            raise ValueError(
                                f"Variable '{name}' in nested if "
                                f"has no fallback value"
                            )
                        ifexp = ast.IfExp(
                            test=stmt.test,
                            body=nif,
                            orelse=fallback,
                        )
                    else:
                        fallback = branch.get(
                            name, self.assignments.get(name)
                        )
                        if fallback is None:
                            raise ValueError(
                                f"Variable '{name}' in nested "
                                f"else has no fallback value"
                            )
                        ifexp = ast.IfExp(
                            test=stmt.test,
                            body=fallback,
                            orelse=nelse,
                        )
                    ast.copy_location(ifexp, stmt)
                    branch[name] = ifexp

            elif isinstance(stmt, ast.Return):
                raise NotImplementedError(
                    "'return' inside an if/elif/else branch is not "
                    "supported in ODE functions. Assign the branch "
                    "result to a variable and return once at the end "
                    "of the function."
                )

            elif isinstance(stmt, (ast.Expr, ast.Pass)):
                # Bare expressions have no assignment to collect;
                # their accesses and calls are recorded by
                # _visit_exprs_in_stmts.
                pass

            else:
                raise NotImplementedError(
                    f"'{type(stmt).__name__}' statements are not "
                    f"supported inside if/elif/else branches in ODE "
                    f"functions."
                )

        return branch

    # -- For-loop unrolling --------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        """Unroll for-loops with constant iterables.

        Substitutes the loop variable with each concrete value and
        visits the body repeatedly, so that ``y[i]`` becomes ``y[0]``,
        ``y[1]``, etc.
        """
        for stmt in self._unroll_for(node):
            self.visit(stmt)

    def _unroll_for(self, node: ast.For) -> List[ast.stmt]:
        """Return the loop body once per value, loop variable bound.

        Each returned statement is a deep copy of a body statement
        with the loop variable substituted by a concrete constant.
        """
        if node.orelse:
            raise NotImplementedError(
                "for/else is not supported in ODE functions."
            )
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                "Only simple loop variables are supported "
                "(e.g. 'for i in ...'). Tuple unpacking in "
                "for-loops is not supported."
            )
        loop_var = node.target.id
        values = _extract_for_iterable(node.iter)
        unrolled: List[ast.stmt] = []
        for val in values:
            for stmt in node.body:
                unrolled.append(
                    _substitute_name(stmt, loop_var, val)
                )
        return unrolled

    def _flatten_for_loops(
        self, stmts: List[ast.stmt]
    ) -> List[ast.stmt]:
        """Replace each for-loop in *stmts* with its unrolled body."""
        flat: List[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                flat.extend(
                    self._flatten_for_loops(self._unroll_for(stmt))
                )
            else:
                flat.append(stmt)
        return flat

    # -- NamedExpr (:=) support ----------------------------------------

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Treat walrus operator as a regular assignment."""
        if isinstance(node.target, ast.Name):
            self.assignments[node.target.id] = node.value
        self.generic_visit(node)

    # -- Explicit rejections -------------------------------------------

    def visit_While(self, node: ast.While) -> None:
        raise NotImplementedError(
            "While-loops are not supported in ODE functions. "
            "Use a for-loop with a constant iterable instead."
        )

    def visit_Try(self, node: ast.Try) -> None:
        raise NotImplementedError(
            "try/except is not supported in ODE functions."
        )

    def visit_TryStar(self, node: ast.stmt) -> None:
        raise NotImplementedError(
            "try/except* is not supported in ODE functions."
        )

    def visit_Match(self, node: ast.stmt) -> None:
        raise NotImplementedError(
            "match statements are not supported in ODE functions. "
            "Use if/elif/else instead."
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._entered_function:
            raise NotImplementedError(
                "Nested function definitions are not supported in "
                "ODE functions. Pass helper callables via the "
                "user_functions argument instead."
            )
        self._entered_function = True
        self.generic_visit(node)

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> None:
        raise NotImplementedError(
            "Async functions are not supported in ODE functions."
        )

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise NotImplementedError(
            "List comprehensions are not supported in ODE "
            "functions. Use a for-loop with a constant iterable "
            "instead."
        )

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise NotImplementedError(
            "Set comprehensions are not supported in ODE functions."
        )

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise NotImplementedError(
            "Dict comprehensions are not supported in ODE "
            "functions."
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise NotImplementedError(
            "Generator expressions are not supported in ODE "
            "functions. Use a for-loop with a constant iterable "
            "instead."
        )

    def visit_With(self, node: ast.With) -> None:
        raise NotImplementedError(
            "The 'with' statement is not supported in ODE "
            "functions."
        )

    def visit_Delete(self, node: ast.Delete) -> None:
        raise NotImplementedError(
            "The 'del' statement is not supported in ODE "
            "functions."
        )

    def visit_Assert(self, node: ast.Assert) -> None:
        raise NotImplementedError(
            "The 'assert' statement is not supported in ODE "
            "functions."
        )

    def visit_Raise(self, node: ast.Raise) -> None:
        raise NotImplementedError(
            "The 'raise' statement is not supported in ODE "
            "functions."
        )

    def visit_Global(self, node: ast.Global) -> None:
        raise NotImplementedError(
            "The 'global' statement is not supported in ODE "
            "functions. Pass values as constants via the third "
            "argument instead."
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise NotImplementedError(
            "The 'nonlocal' statement is not supported in ODE "
            "functions."
        )

    def visit_Import(self, node: ast.Import) -> None:
        pass  # Allow — used for math.sin etc.

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        pass  # Allow — e.g. from math import sin


def _call_name(node: ast.Call) -> Optional[str]:
    """Extract the callable name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            return f"{node.func.value.id}.{node.func.attr}"
    return None


def _resolve_func_name(name: str) -> Optional[str]:
    """Strip module prefix to get the bare function name."""
    if "." in name:
        parts = name.split(".")
        if parts[0] in _MODULE_PREFIXES:
            return parts[1]
    return name


def _substitute_name(node: ast.AST, name: str, value: Any) -> ast.AST:
    """Deep-copy *node*, replacing ``Name(id=name, ctx=Load)`` with
    ``Constant(value=value)``."""

    class _Substitutor(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if n.id == name and isinstance(n.ctx, ast.Load):
                replacement = ast.Constant(value=value)
                ast.copy_location(replacement, n)
                return replacement
            return n

    return _Substitutor().visit(copy.deepcopy(node))


def _extract_for_iterable(node: ast.expr) -> list:
    """Extract concrete values from a for-loop iterable.

    Supports ``range(stop)``, ``range(start, stop)``,
    ``range(start, stop, step)``, literal lists, and literal tuples.
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            args: List[int] = []
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(
                    a.value, int
                ):
                    args.append(a.value)
                elif (
                    isinstance(a, ast.UnaryOp)
                    and isinstance(a.op, ast.USub)
                    and isinstance(a.operand, ast.Constant)
                ):
                    args.append(-a.operand.value)
                else:
                    raise NotImplementedError(
                        "for-loop range() arguments must be integer "
                        "literals. Use a literal list or tuple, or "
                        "pass iteration bounds as constants."
                    )
            return list(range(*args))
        raise NotImplementedError(
            f"for-loop iterable must be range(), a literal list, "
            f"or a literal tuple — got function call "
            f"'{_call_name(node) or '?'}'"
        )
    elif isinstance(node, (ast.List, ast.Tuple)):
        values: list = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                values.append(elt.value)
            elif (
                isinstance(elt, ast.UnaryOp)
                and isinstance(elt.op, ast.USub)
                and isinstance(elt.operand, ast.Constant)
            ):
                values.append(-elt.operand.value)
            else:
                raise NotImplementedError(
                    "for-loop literal list/tuple elements must be "
                    "constants (int, float, string)"
                )
        return values
    elif isinstance(node, ast.Name):
        raise NotImplementedError(
            f"for-loop over variable '{node.id}' is not supported. "
            f"Use a literal iterable: range(), list, or tuple."
        )
    else:
        raise NotImplementedError(
            "for-loop iterable must be range(), a literal list, "
            "or a literal tuple."
        )


class AstToSympyConverter:
    """Convert AST expression nodes to SymPy expressions.

    Parameters
    ----------
    symbol_map
        Mapping of variable names to SymPy symbols/expressions.
    user_callables
        Mapping of user-defined function names to their Python
        implementations. Calls to these names take priority over
        :data:`KNOWN_FUNCTIONS`.
    user_function_classes
        Mapping of user-defined function names to SymPy ``Function``
        placeholders (or subclasses) applied when a call stays
        symbolic.
    symbolic_user_names
        User-function names that must never be inlined (device
        functions and functions with derivative helpers).
    inline_assignments
        Local assignment names whose AST expression is converted in
        place of the bare name (derivative-output names such as
        ``dx``).
    strict_names
        When ``True`` an unknown bare name raises ``ValueError``
        instead of creating a dangling symbol.
    name_hints
        Mapping of known-but-unreachable names to the access spelling
        the error message should suggest (e.g. ``{"mu": "p.mu"}``).
    """

    def __init__(
        self,
        symbol_map: Dict[str, sp.Basic],
        user_callables: Optional[Dict[str, Callable]] = None,
        user_function_classes: Optional[Dict[str, Any]] = None,
        symbolic_user_names: Optional[Set[str]] = None,
        inline_assignments: Optional[Dict[str, ast.expr]] = None,
        strict_names: bool = False,
        name_hints: Optional[Dict[str, str]] = None,
    ) -> None:
        self.symbol_map = symbol_map
        self.user_callables = user_callables or {}
        self.user_function_classes = user_function_classes or {}
        self.symbolic_user_names = symbolic_user_names or set()
        self.inline_assignments = inline_assignments or {}
        self.strict_names = strict_names
        self.name_hints = name_hints or {}

    def convert(self, node: ast.expr) -> sp.Expr:
        """Recursively convert an AST node to a SymPy expression.

        Parameters
        ----------
        node
            AST expression node to convert.

        Returns
        -------
        sp.Expr
            Equivalent SymPy expression.

        Raises
        ------
        NotImplementedError
            If the AST node type is not supported.
        """
        if isinstance(node, ast.Constant):
            return self._convert_constant(node)
        elif isinstance(node, ast.Name):
            return self._convert_name(node)
        elif isinstance(node, ast.BinOp):
            return self._convert_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._convert_unaryop(node)
        elif isinstance(node, ast.Call):
            return self._convert_call(node)
        elif isinstance(node, ast.Subscript):
            return self._convert_subscript(node)
        elif isinstance(node, ast.Attribute):
            return self._convert_attribute(node)
        elif isinstance(node, ast.Compare):
            return self._convert_compare(node)
        elif isinstance(node, ast.IfExp):
            return self._convert_ifexp(node)
        elif isinstance(node, ast.BoolOp):
            return self._convert_boolop(node)
        elif isinstance(node, ast.Tuple):
            # Should not appear at expression level; handled by
            # callers that unpack return tuples/lists
            raise NotImplementedError(
                "Tuple expressions should be unpacked by the caller"
            )
        elif isinstance(node, ast.List):
            raise NotImplementedError(
                "List expressions should be unpacked by the caller"
            )
        else:
            raise NotImplementedError(
                f"Unsupported AST node type: {type(node).__name__}. "
                f"Only arithmetic, function calls, comparisons, and "
                f"ternary if/else are supported."
            )

    def _convert_constant(self, node: ast.Constant) -> sp.Expr:
        val = node.value
        if isinstance(val, int):
            return sp.Integer(val)
        elif isinstance(val, float):
            return sp.Float(val)
        else:
            raise NotImplementedError(
                f"Unsupported constant type: {type(val).__name__}"
            )

    def _convert_name(self, node: ast.Name) -> sp.Expr:
        name = node.id
        if name in self.symbol_map:
            return self.symbol_map[name]
        if name in self.inline_assignments:
            return self.convert(self.inline_assignments[name])
        if self.strict_names:
            if name in self.name_hints:
                raise ValueError(
                    f"Symbol '{name}' is declared but cannot be "
                    f"referenced by bare name inside an ODE function. "
                    f"Access it through its container argument: "
                    f"'{self.name_hints[name]}'."
                )
            raise ValueError(
                f"Unknown symbol '{name}' in ODE function body. Names "
                f"must resolve to the time argument, a container "
                f"access on the state or parameter arguments, a local "
                f"assignment, or a declared observable. Declare "
                f"'{name}' as a parameter, constant, or driver and "
                f"access it through a container argument."
            )
        # Create a real symbol for unknown names
        sym = sp.Symbol(name, real=True)
        self.symbol_map[name] = sym
        return sym

    def _convert_binop(self, node: ast.BinOp) -> sp.Expr:
        left = self.convert(node.left)
        right = self.convert(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            return left / right
        elif isinstance(op, ast.FloorDiv):
            return sp.floor(left / right)
        elif isinstance(op, ast.Pow):
            return left ** right
        elif isinstance(op, ast.Mod):
            return sp.Mod(left, right)
        else:
            raise NotImplementedError(
                f"Unsupported binary op: {type(op).__name__}"
            )

    def _convert_unaryop(self, node: ast.UnaryOp) -> sp.Expr:
        operand = self.convert(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.Not):
            return sp.Not(operand)
        else:
            raise NotImplementedError(
                f"Unsupported unary op: {type(node.op).__name__}"
            )

    def _convert_call(self, node: ast.Call) -> sp.Expr:
        raw_name = _call_name(node)
        if raw_name is None:
            raise NotImplementedError(
                "Only named function calls are supported"
            )
        name = _resolve_func_name(raw_name)
        user_name = None
        if raw_name in self.user_callables:
            user_name = raw_name
        elif name in self.user_callables:
            user_name = name
        if user_name is not None:
            args = [self.convert(a) for a in node.args]
            if user_name not in self.symbolic_user_names:
                fn = self.user_callables[user_name]
                try:
                    val = fn(*args)
                except (TypeError, AttributeError, ValueError):
                    # Symbolic arguments rejected by the function body
                    # (e.g. NumPy ufuncs, float() coercion): fall back
                    # to a symbolic call. Other errors are genuine bugs
                    # in the user function and propagate.
                    val = None
                if isinstance(val, sp.Expr):
                    return val
            return self.user_function_classes[user_name](*args)
        if name not in KNOWN_FUNCTIONS:
            raise NotImplementedError(
                f"Unknown function '{raw_name}'. Supported: "
                f"{sorted(KNOWN_FUNCTIONS.keys())}, plus any names "
                f"supplied via the user_functions argument."
            )
        sp_func = KNOWN_FUNCTIONS[name]
        args = [self.convert(a) for a in node.args]
        return sp_func(*args)

    def _convert_subscript(self, node: ast.Subscript) -> sp.Expr:
        if isinstance(node.value, ast.Name):
            base_name = node.value.id
            slc = node.slice
            if isinstance(slc, ast.Constant):
                key = slc.value
            else:
                raise NotImplementedError(
                    "Only constant subscripts are supported"
                )
            lookup = f"{base_name}[{key!r}]" if isinstance(
                key, str
            ) else f"{base_name}[{key}]"
            if lookup in self.symbol_map:
                return self.symbol_map[lookup]
            raise NotImplementedError(
                f"Subscript '{lookup}' not found in symbol map"
            )
        raise NotImplementedError("Complex subscript targets not supported")

    def _convert_attribute(self, node: ast.Attribute) -> sp.Expr:
        if isinstance(node.value, ast.Name):
            lookup = f"{node.value.id}.{node.attr}"
            if lookup in self.symbol_map:
                return self.symbol_map[lookup]
            raise NotImplementedError(
                f"Attribute '{lookup}' not found in symbol map"
            )
        raise NotImplementedError("Complex attribute targets not supported")

    def _convert_compare(self, node: ast.Compare) -> sp.Expr:
        left = self.convert(node.left)
        result = None
        for op, comparator_node in zip(node.ops, node.comparators):
            right = self.convert(comparator_node)
            rel = self._comparison_op(op, left, right)
            result = rel if result is None else sp.And(result, rel)
            left = right
        return result

    @staticmethod
    def _comparison_op(
        op: ast.cmpop, left: sp.Expr, right: sp.Expr
    ) -> sp.Expr:
        if isinstance(op, ast.Gt):
            return sp.Gt(left, right)
        elif isinstance(op, ast.GtE):
            return sp.Ge(left, right)
        elif isinstance(op, ast.Lt):
            return sp.Lt(left, right)
        elif isinstance(op, ast.LtE):
            return sp.Le(left, right)
        elif isinstance(op, ast.Eq):
            return sp.Eq(left, right)
        elif isinstance(op, ast.NotEq):
            return sp.Ne(left, right)
        else:
            raise NotImplementedError(
                f"Unsupported comparison: {type(op).__name__}"
            )

    def _convert_ifexp(self, node: ast.IfExp) -> sp.Expr:
        body = self.convert(node.body)
        test = self.convert(node.test)
        orelse = self.convert(node.orelse)
        return sp.Piecewise((body, test), (orelse, True))

    def _convert_boolop(self, node: ast.BoolOp) -> sp.Expr:
        values = [self.convert(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return sp.And(*values)
        elif isinstance(node.op, ast.Or):
            return sp.Or(*values)
        else:
            raise NotImplementedError(
                f"Unsupported bool op: {type(node.op).__name__}"
            )


def inspect_ode_function(func: Callable) -> FunctionInspection:
    """Analyse a callable to extract ODE structure from its AST.

    Parameters
    ----------
    func
        A Python function defining an ODE right-hand side. Must accept
        at least two positional arguments (time, state).

    Returns
    -------
    FunctionInspection
        Parsed metadata about the function's structure.

    Raises
    ------
    TypeError
        If ``func`` is a lambda, builtin, or not callable.
    ValueError
        If ``func`` has fewer than 2 parameters or no return statement.
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func).__name__}")

    # Reject lambdas
    if getattr(func, "__name__", "") == "<lambda>":
        raise TypeError(
            "Lambda functions are not supported. Use a named function "
            "with a def statement."
        )

    # Reject builtins without source
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        raise TypeError(
            "Cannot inspect source of builtin or C-extension functions. "
            "Use a pure Python function."
        )

    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise ValueError("Could not find function definition in source")

    params = [arg.arg for arg in func_def.args.args]
    if len(params) < 2:
        raise ValueError(
            f"ODE function must accept at least 2 parameters "
            f"(time, state), got {len(params)}: {params}"
        )

    time_param = params[0]
    state_param = params[1]
    constant_params = params[2:]

    # Warn on unconventional names
    if time_param != "t":
        warnings.warn(
            f"First parameter '{time_param}' is conventionally named 't'",
            stacklevel=2,
        )
    if state_param not in ("y", "state", "x", "Y", "X"):
        warnings.warn(
            f"Second parameter '{state_param}' is conventionally named "
            f"'y' or 'state'",
            stacklevel=2,
        )

    visitor = _OdeAstVisitor(state_param, constant_params)
    visitor.visit(func_def)

    if not visitor.return_nodes:
        raise ValueError(
            "ODE function must contain a return statement"
        )
    if len(visitor.return_nodes) > 1:
        raise ValueError(
            "ODE function must contain exactly one return statement, "
            f"found {len(visitor.return_nodes)}"
        )

    # Validate consistent access patterns per base
    _validate_access_consistency(
        visitor.state_accesses, state_param
    )
    for cp in constant_params:
        cp_accesses = [
            a for a in visitor.constant_accesses if a["base"] == cp
        ]
        _validate_access_consistency(cp_accesses, cp)

    # Extra args never accessed as containers but referenced by bare
    # name are scalars bound to the like-named declared symbol
    # (SciPy's args= convention). Unreferenced args are ignored.
    container_bases = {a["base"] for a in visitor.constant_accesses}
    used_names = {
        node.id
        for node in ast.walk(func_def)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    scalar_params = [
        cp
        for cp in constant_params
        if cp not in container_bases and cp in used_names
    ]

    return FunctionInspection(
        param_names=params,
        state_param=state_param,
        constant_params=constant_params,
        state_accesses=visitor.state_accesses,
        constant_accesses=visitor.constant_accesses,
        assignments=visitor.assignments,
        return_node=visitor.return_nodes[0],
        function_calls=visitor.function_calls,
        func_def=func_def,
        scalar_params=scalar_params,
    )


def _validate_access_consistency(
    accesses: List[Dict[str, Any]], param_name: str
) -> None:
    """Reject mixed access patterns on the same base variable.

    Parameters
    ----------
    accesses
        Access records for a single base variable.
    param_name
        Name of the parameter for error messages.

    Raises
    ------
    ValueError
        If both integer and string subscript patterns are used on the
        same base.
    """
    patterns = {a["pattern_type"] for a in accesses}
    patterns.discard("expr")
    patterns.discard("name")
    if len(patterns) > 1:
        raise ValueError(
            f"Mixed access patterns on '{param_name}': {patterns}. "
            f"Use a single pattern (int subscript, string subscript, "
            f"or attribute access)."
        )
