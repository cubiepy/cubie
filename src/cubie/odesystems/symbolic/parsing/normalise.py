"""Normalise user equation input into structural IR equations.

The single symbolic front end: string and SymPy input converge on
one representation — a list of
:class:`~cubie.odesystems.symbolic.structural.system_structure.Equation`
objects holding engine IR expressions, with derivatives replaced by
:class:`~cubie.odesystems.symbolic.structural.symbolics.DerivativeRegistry`
symbols — plus the resolved declarations.

SymPy is the parsing layer only: strings parse through
``sympy.parse_expr`` and SymPy input is accepted directly, but every
expression converts to engine IR here, at the parse boundary, before
any downstream pass runs. Pre-converted IR ``(lhs, rhs)`` pairs (the
CellML loader's output) pass through without touching SymPy.

Left-hand sides are state-aware: ``dX`` names the derivative of
``X`` only when ``X`` is a declared unknown (otherwise ``dX`` is an
auxiliary, or an inferred state when no states were declared). A
bare, unassigned ``dX`` token anywhere else — inside an expression
left-hand side (``c*dx = f(...)``, an implicit equation) or on a
right-hand side — binds to the derivative of unknown ``X``.
Derivatives inside expressions may also use the explicit
``d(x, t)`` call (strings) or :class:`sympy.Derivative` (SymPy
input).

Published Functions
-------------------
:func:`normalise_input`
    Parse string, SymPy, or IR equations into a
    :class:`NormalisedSystem`.
"""

import re
from typing import Callable, Dict, Iterable, List, Optional

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import from_sympy
from cubie.odesystems.symbolic.parsing.parser import (
    KNOWN_FUNCTIONS,
    PARSE_TRANSFORMS,
    TIME_SYMBOL,
    _build_sympy_user_functions,
    _func_call_re,
    _inline_nondevice_calls,
    _normalise_indexed_tokens,
    _rename_user_calls,
    _sanitise_input_math,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
_NUMERIC_LITERAL_PATTERN = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
)
_D_CALL_LHS_PATTERN = re.compile(
    r"^d\s*\(\s*([A-Za-z_]\w*)\s*,\s*t\s*\)$"
)


class NormalisedSystem:
    """Equations and declarations in the shared front-end form.

    Parameters
    ----------
    equations
        Structural IR equations with registry derivative symbols.
    registry
        The derivative registry used by ``equations``.
    funcs
        Callables referenced by the equations, resolved against
        ``user_functions`` and :data:`KNOWN_FUNCTIONS`.
    unknown_names
        All unknown names (declared states, observables, and
        inferred auxiliaries).
    aux_names
        Names of auxiliaries inferred from assignments.
    new_params
        Names inferred as parameters from equation usage.
    inferred_states
        State names inferred from derivative assignments when no
        states were declared (non-strict input only).
    rename
        User-function rename map from the string path (empty for
        SymPy input).
    derivative_names
        Renamed user-function name to derivative-placeholder print
        name, for functions with user-supplied derivative helpers.
    """

    def __init__(
        self,
        equations: List[Equation],
        registry: DerivativeRegistry,
        funcs: Dict[str, Callable],
        unknown_names: set,
        aux_names: List[str],
        new_params: List[str],
        inferred_states: List[str],
        rename: Dict[str, str],
        derivative_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self.equations = equations
        self.registry = registry
        self.funcs = funcs
        self.unknown_names = unknown_names
        self.aux_names = aux_names
        self.new_params = new_params
        self.inferred_states = inferred_states
        self.rename = rename
        self.derivative_names = dict(derivative_names or {})


def _process_calls(
    lines: Iterable[str],
    user_functions: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Callable]:
    """Resolve callables referenced in equations, allowing ``d()``."""

    calls = set()
    user_functions = user_functions or {}
    for line in lines:
        calls |= set(_func_call_re.findall(line))
    calls.discard("d")
    funcs = {}
    for name in calls:
        if name in user_functions:
            funcs[name] = user_functions[name]
        elif name in KNOWN_FUNCTIONS:
            funcs[name] = KNOWN_FUNCTIONS[name]
        else:
            raise ValueError(
                f"Your equations contain a call to a function "
                f"{name}() that isn't part of Sympy and wasn't "
                f"provided in the user_functions dict."
            )
    return funcs


def _derivative_print_names(
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    rename: Dict[str, str],
) -> Dict[str, str]:
    """Map renamed function names to derivative placeholder names.

    Only functions with a user-supplied derivative helper appear;
    every other function differentiates to the default
    ``d_<name>`` placeholder.
    """

    names: Dict[str, str] = {}
    for orig, deriv in (user_function_derivatives or {}).items():
        if user_functions is not None and orig not in user_functions:
            continue
        printed = getattr(deriv, "__name__", None)
        if printed:
            names[rename.get(orig, orig)] = printed
    return names


def _replace_derivative_calls(
    expr: sp.Expr,
    registry: DerivativeRegistry,
    unknown_names: set,
) -> sp.Expr:
    """Replace ``d(x, t)`` calls with registry derivative symbols.

    Nested calls resolve innermost-first, producing higher-order
    derivative symbols. Raises when the differentiated quantity is
    not an unknown symbol. The registry works in IR symbols; the
    replacement embeds a same-named SymPy symbol, which converts back
    to the identical interned IR node at the conversion step.
    """

    def repl(call: sp.Expr) -> sp.Expr:
        args = call.args
        if len(args) != 2 or args[1] != TIME_SYMBOL:
            raise ValueError(
                f"Derivative notation must be d(<symbol>, t); got "
                f"{call}"
            )
        inner = args[0]
        if not isinstance(inner, sp.Symbol):
            raise ValueError(
                f"Cannot differentiate non-symbol expression {inner} "
                "in d() notation."
            )
        inner_ir = ir.sym(inner.name)
        base, _ = registry.base_and_order(inner_ir)
        if base.name not in unknown_names:
            raise ValueError(
                f"d({inner}, t) differentiates {inner}, but "
                f"{base.name} is not a declared state or unknown."
            )
        return sp.Symbol(registry.derivative(inner_ir).name, real=True)

    def is_d_call(node: sp.Basic) -> bool:
        return (
            isinstance(node, AppliedUndef)
            and node.func.__name__ == "d"
        )

    # replace() rebuilds post-order, so inner calls resolve before
    # outer ones and nesting yields higher-order symbols.
    return expr.replace(is_d_call, repl)


def _replace_sympy_derivatives(
    expr: sp.Expr,
    registry: DerivativeRegistry,
    unknown_names: set,
) -> sp.Expr:
    """Replace :class:`sympy.Derivative` nodes with registry symbols."""

    def repl(node: sp.Derivative) -> sp.Expr:
        inner = node.expr
        if not isinstance(inner, sp.Symbol):
            raise ValueError(
                f"Cannot differentiate non-symbol expression "
                f"{inner}."
            )
        if inner.name not in unknown_names:
            raise ValueError(
                f"Derivative of {inner} found, but {inner} is not a "
                "declared state or unknown."
            )
        order = 0
        for var, count in node.variable_count:
            if var != TIME_SYMBOL and str(var) != "t":
                raise ValueError(
                    f"Only time derivatives are supported; got "
                    f"{node}."
                )
            order += int(count)
        sym = ir.sym(inner.name)
        for _ in range(order):
            sym = registry.derivative(sym)
        return sp.Symbol(sym.name, real=True)

    return expr.replace(
        lambda node: isinstance(node, sp.Derivative), repl
    )


def _bind_unassigned_derivative_tokens(
    equations: List[Equation],
    registry: DerivativeRegistry,
    unknown_names: set,
) -> List[Equation]:
    """Bind unassigned ``dX`` atoms to registry derivative symbols.

    Applies to expression left-hand sides and to every right-hand
    side: a bare ``dX`` token whose name is not itself assigned and
    whose ``X`` is an unknown names the derivative of ``X``.
    """

    assigned_names = {
        eq.lhs.name
        for eq in equations
        if isinstance(eq.lhs, ir.Sym)
        and not registry.is_derivative(eq.lhs)
    }

    def derivative_rules(expression):
        rules = {}
        for atom in ir.free_atoms(expression):
            if not isinstance(atom, ir.Sym):
                continue
            if registry.is_derivative(atom):
                continue
            name = atom.name
            if (
                name.startswith("d")
                and len(name) > 1
                and name not in assigned_names
                and name[1:] in unknown_names
            ):
                rules[atom] = registry.derivative(ir.sym(name[1:]))
        return rules

    bound = []
    for eq in equations:
        lhs = eq.lhs
        if not isinstance(lhs, (ir.Sym, ir.Num)):
            lhs_rules = derivative_rules(lhs)
            if lhs_rules:
                lhs = ir.xreplace(lhs, lhs_rules)
        rhs = eq.rhs
        rhs_rules = derivative_rules(rhs)
        if rhs_rules:
            rhs = ir.xreplace(rhs, rhs_rules)
        if lhs is eq.lhs and rhs is eq.rhs:
            bound.append(eq)
        else:
            bound.append(Equation(lhs, rhs))
    return bound


def _infer_parameters(
    equations: List[Equation],
    registry: DerivativeRegistry,
    declared_names: set,
    strict: bool,
) -> List[str]:
    """Collect undeclared free symbols as parameter names.

    Iterates atoms in structural sort-key order so the inferred
    parameter order is deterministic across processes.
    """

    new_params: List[str] = []
    for eq in equations:
        atoms = sorted(
            eq.free_symbols(), key=lambda atom: atom.sort_key
        )
        for atom in atoms:
            if not isinstance(atom, ir.Sym):
                continue
            if registry.is_derivative(atom):
                continue
            if atom.name in declared_names:
                continue
            if strict:
                raise ValueError(
                    f"Equations reference undefined symbol "
                    f"{atom.name}."
                )
            new_params.append(atom.name)
            declared_names.add(atom.name)
    return new_params


def _parse_string_equations(
    lines,
    raw_lines,
    registry,
    unknown_names,
    known_symbol_map,
    user_functions,
    user_function_derivatives,
    strict,
    state_names,
):
    """Parse ``lhs = rhs`` lines into structural IR equations.

    Returns ``(equations, funcs, new_params, aux_names,
    inferred_states, rename)``.
    """

    funcs = _process_calls(lines, user_functions)
    sanitized_lines, rename = _rename_user_calls(
        lines, user_functions or {}
    )
    parse_locals, _alias_map, _dev_map = _build_sympy_user_functions(
        user_functions or {}, rename, user_function_derivatives
    )

    local_dict = dict(known_symbol_map)
    local_dict.update(parse_locals)
    local_dict.setdefault("t", TIME_SYMBOL)
    # The d(x, t) notation reserves the name ``d`` as a function
    # only when it is used; otherwise ``d`` stays available as an
    # ordinary symbol.
    if any(re.search(r"\bd\s*\(", line) for line in sanitized_lines):
        local_dict["d"] = sp.Function("d")
    for name in unknown_names:
        local_dict.setdefault(name, sp.Symbol(name, real=True))
    # RHS references to dX bind to the derivative assignment of
    # state X (assignment-reference semantics, not a derivative
    # term). Declared names win over the generated symbol.
    derivative_names = set()
    for name in state_names:
        dname = f"d{name}"
        if dname not in local_dict:
            local_dict[dname] = sp.Symbol(dname, real=True)
            derivative_names.add(dname)

    # Pre-seed every assignment-target name so forward references
    # bind to the assignment rather than to a same-named SymPy
    # global (``zoo``, ``pi``, ``E``, ...).
    for line in sanitized_lines:
        target = line.split("=", 1)[0].strip()
        if _NUMERIC_LITERAL_PATTERN.match(target):
            continue
        if not _IDENTIFIER_PATTERN.match(target):
            continue
        local_dict.setdefault(target, sp.Symbol(target, real=True))

    sym_pairs = []
    aux_names = []
    inferred_states = []

    def infer_state(name):
        unknown_names.add(name)
        inferred_states.append(name)
        local_dict.setdefault(name, sp.Symbol(name, real=True))
        dname = f"d{name}"
        if dname not in known_symbol_map:
            local_dict.setdefault(
                dname, sp.Symbol(dname, real=True)
            )
            derivative_names.add(dname)

    for raw_line, line in zip(raw_lines, sanitized_lines):
        lhs_str, rhs_str = [p.strip() for p in line.split("=", 1)]

        # The left-hand side is analysed before the right-hand side
        # so state inference and LHS-specific errors take priority
        # over undefined-symbol errors from the RHS parse.
        if _NUMERIC_LITERAL_PATTERN.match(lhs_str):
            lhs_expr = sp.Number(lhs_str)
        elif lhs_str.startswith("d") and "(" in lhs_str:
            # d(x, t) notation: explicit derivative intent, so an
            # undeclared x is inferred as a state in non-strict
            # mode.
            inner_match = _D_CALL_LHS_PATTERN.match(lhs_str)
            if (
                inner_match
                and inner_match.group(1) not in unknown_names
            ):
                name = inner_match.group(1)
                if strict:
                    raise ValueError(
                        f"Unknown state in derivative notation: "
                        f"d({name}, t). No state called {name} "
                        "found."
                    )
                infer_state(name)
            lhs_expr = parse_expr(
                lhs_str,
                transformations=PARSE_TRANSFORMS,
                local_dict=local_dict,
            )
            lhs_expr = _replace_derivative_calls(
                lhs_expr, registry, unknown_names
            )
        elif _IDENTIFIER_PATTERN.match(lhs_str):
            if (
                lhs_str.startswith("d")
                and len(lhs_str) > 1
                and lhs_str[1:] in unknown_names
            ):
                lhs_expr = sp.Symbol(
                    registry.derivative(ir.sym(lhs_str[1:])).name,
                    real=True,
                )
            elif (
                lhs_str.startswith("d")
                and len(lhs_str) > 1
                and not state_names
                and not strict
                and _IDENTIFIER_PATTERN.match(lhs_str[1:])
            ):
                # No states declared: a dX assignment infers state X
                # (quickstart convenience). With declared states,
                # unknown d-prefixed names are auxiliaries instead.
                name = lhs_str[1:]
                infer_state(name)
                lhs_expr = sp.Symbol(
                    registry.derivative(ir.sym(name)).name,
                    real=True,
                )
            elif lhs_str in known_symbol_map:
                raise ValueError(
                    f"{lhs_str} is an immutable input (constant, "
                    "parameter, or driver) but is being assigned. It "
                    "must be a state, observable, or auxiliary."
                )
            else:
                candidate = local_dict.get(lhs_str)
                if isinstance(candidate, sp.Symbol):
                    lhs_expr = candidate
                else:
                    lhs_expr = sp.Symbol(lhs_str, real=True)
                if lhs_str not in unknown_names:
                    # An assignment defines the symbol, so even
                    # strict mode admits anonymous auxiliaries.
                    aux_names.append(lhs_str)
                    unknown_names.add(lhs_str)
                    local_dict.setdefault(lhs_str, lhs_expr)
        else:
            # Expression LHS: an implicit equation.
            lhs_text = _sanitise_input_math(lhs_str)
            try:
                if strict:
                    lhs_expr = parse_expr(
                        lhs_text,
                        transformations=PARSE_TRANSFORMS,
                        local_dict=local_dict,
                    )
                else:
                    lhs_expr = parse_expr(
                        lhs_text, local_dict=local_dict
                    )
            except (SyntaxError, NameError, TypeError) as exc:
                raise ValueError(
                    f"Unsupported left-hand side '{lhs_str}' in "
                    f"equation '{raw_line}'. Expected a symbol, dX, "
                    "d(x, t), a number, or an expression (implicit "
                    "equation)."
                ) from exc
            lhs_expr = _replace_derivative_calls(
                lhs_expr, registry, unknown_names
            )

        rhs_text = _sanitise_input_math(rhs_str)
        if strict:
            try:
                rhs_expr = parse_expr(
                    rhs_text,
                    transformations=PARSE_TRANSFORMS,
                    local_dict=local_dict,
                )
            except (NameError, TypeError) as exc:
                raise ValueError(
                    f"Undefined symbols in equation '{raw_line}'"
                ) from exc
        else:
            # Without transformations parse_expr auto-creates
            # symbols for undeclared names (inferred as parameters
            # below).
            rhs_expr = parse_expr(rhs_text, local_dict=local_dict)
        rhs_expr = _inline_nondevice_calls(
            rhs_expr, user_functions or {}, rename
        )
        rhs_expr = _replace_derivative_calls(
            rhs_expr, registry, unknown_names
        )

        sym_pairs.append((lhs_expr, rhs_expr))

    # Convert to IR once, sharing conversion work across equations.
    memo = {}
    equations = [
        Equation(
            from_sympy(
                lhs,
                memo,
                allowed_functions=parse_locals,
            ),
            from_sympy(
                rhs,
                memo,
                allowed_functions=parse_locals,
            ),
        )
        for lhs, rhs in sym_pairs
    ]

    equations = _bind_unassigned_derivative_tokens(
        equations, registry, unknown_names
    )

    # Infer undeclared RHS symbols as parameters (non-strict), after
    # derivative replacement so derivative symbols don't count.
    # Exclusion is by name so a forward reference to a later ``dX``
    # assignment binds to it instead of becoming a parameter.
    declared_names = (
        set(known_symbol_map)
        | unknown_names
        | derivative_names
        | {"t"}
    )
    new_params = _infer_parameters(
        equations, registry, declared_names, strict
    )

    return (
        equations,
        funcs,
        new_params,
        aux_names,
        inferred_states,
        rename,
    )


def _parse_sympy_equations(
    dxdt,
    registry,
    unknown_names,
    known_symbol_map,
    user_functions,
    user_function_derivatives,
    strict,
    state_names,
):
    """Normalise SymPy or IR equation input into structural equations.

    Returns ``(equations, funcs, new_params, aux_names,
    inferred_states, rename)``. SymPy sides convert to engine IR
    after derivative replacement and user-function resolution;
    ``(lhs, rhs)`` pairs whose sides are already IR expressions pass
    through without touching SymPy. User-function calls appearing as
    :class:`~sympy.core.function.AppliedUndef` nodes are resolved
    against ``user_functions``: non-device callables are inlined and
    device callables are kept as symbolic calls.
    """

    if isinstance(dxdt, (list, tuple)):
        raw_equations = list(dxdt)
    else:
        raw_equations = [dxdt]

    user_functions = user_functions or {}
    funcs = {}

    def resolve_calls(expr: sp.Expr) -> sp.Expr:
        called = {
            node.func.__name__
            for node in expr.atoms(AppliedUndef)
        }
        called.discard("d")
        for name in called:
            if name in user_functions:
                funcs[name] = user_functions[name]
            elif name not in KNOWN_FUNCTIONS:
                raise ValueError(
                    f"Your equations contain a call to a function "
                    f"{name}() that isn't part of Sympy and wasn't "
                    f"provided in the user_functions dict."
                )
        if user_functions:
            expr = _inline_nondevice_calls(
                expr,
                user_functions,
                {name: name for name in user_functions},
            )
        return expr

    equations = []
    aux_names = []
    inferred_states = []

    def bind_lhs(lhs: ir.Expr) -> ir.Expr:
        """Apply the state-aware ``dX`` rule to a symbol LHS."""

        if not isinstance(lhs, ir.Sym):
            return lhs
        if registry.is_derivative(lhs):
            return lhs
        name = lhs.name
        if not (name.startswith("d") and len(name) > 1):
            return lhs
        base = name[1:]
        if base in unknown_names:
            return registry.derivative(ir.sym(base))
        if (
            not state_names
            and not strict
            and _IDENTIFIER_PATTERN.match(base)
        ):
            unknown_names.add(base)
            inferred_states.append(base)
            return registry.derivative(ir.sym(base))
        return lhs

    memo = {}
    for i, eq in enumerate(raw_equations):
        if isinstance(eq, sp.Equality):
            lhs, rhs = eq.lhs, eq.rhs
        elif isinstance(eq, tuple) and len(eq) == 2:
            lhs, rhs = eq
        else:
            raise TypeError(
                f"Equation {i}: expected sp.Eq or a (lhs, rhs) "
                f"tuple, got {type(eq).__name__}."
            )
        def to_ir(side):
            if isinstance(side, ir.Expr):
                return side
            try:
                side = sp.sympify(side)
            except (sp.SympifyError, TypeError) as exc:
                raise TypeError(
                    f"Equation {i}: could not convert "
                    f"({lhs!r}, {rhs!r}) to SymPy expressions; each "
                    f"side must be a SymPy expression, string, or "
                    f"number."
                ) from exc
            side = _replace_sympy_derivatives(
                side, registry, unknown_names
            )
            return from_sympy(
                resolve_calls(side),
                memo,
                allowed_functions=user_functions,
            )

        lhs_ir = to_ir(lhs)
        rhs_ir = to_ir(rhs)
        lhs_ir = bind_lhs(lhs_ir)
        if (
            isinstance(lhs_ir, ir.Sym)
            and lhs_ir.name in known_symbol_map
        ):
            raise ValueError(
                f"{lhs_ir.name} is an immutable input (constant, "
                "parameter, or driver) but is being assigned. It "
                "must be a state, observable, or auxiliary."
            )
        equations.append(Equation(lhs_ir, rhs_ir))

    equations = _bind_unassigned_derivative_tokens(
        equations, registry, unknown_names
    )

    derivative_names = set()
    for name in state_names:
        dname = f"d{name}"
        if dname not in known_symbol_map:
            derivative_names.add(dname)
    declared_names = (
        set(known_symbol_map)
        | unknown_names
        | derivative_names
        | {"t"}
    )
    for eq in equations:
        lhs = eq.lhs
        if (
            isinstance(lhs, ir.Sym)
            and lhs.name not in declared_names
            and not registry.is_derivative(lhs)
        ):
            # An assignment defines the symbol, so even strict mode
            # admits anonymous auxiliaries.
            aux_names.append(lhs.name)
            unknown_names.add(lhs.name)
            declared_names.add(lhs.name)
    new_params = _infer_parameters(
        equations, registry, declared_names, strict
    )
    return equations, funcs, new_params, aux_names, inferred_states, {}


def normalise_input(
    dxdt,
    unknown_names: set,
    known_symbol_map: Dict[str, sp.Symbol],
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    strict: bool,
    state_names: Iterable[str],
) -> NormalisedSystem:
    """Normalise string, SymPy, or IR equations into structural form.

    Parameters
    ----------
    dxdt
        Equation strings (newline-joined or a sequence), SymPy
        equations (:class:`~sympy.Equality` or ``(lhs, rhs)``
        tuples), or pre-converted IR ``(lhs, rhs)`` pairs.
    unknown_names
        Names of the declared unknowns (states and observables).
        Mutated: inferred auxiliaries and states are added.
    known_symbol_map
        Immutable inputs (parameters, constants, drivers) by name.
    user_functions, user_function_derivatives
        Callables referenced in the equations and their analytic
        derivative helpers.
    strict
        Reject undeclared symbols instead of inferring them.
    state_names
        Names of the declared states; controls derivative-name
        binding on right-hand sides and ``dX`` state inference.
    """

    state_names = set(state_names)
    reserved = (
        set(known_symbol_map)
        | set(unknown_names)
        | {"t"}
        | set(KNOWN_FUNCTIONS)
    )
    registry = DerivativeRegistry(reserved)

    is_string_input = isinstance(dxdt, str) or (
        isinstance(dxdt, (list, tuple))
        and bool(dxdt)
        and isinstance(dxdt[0], str)
    )
    if is_string_input:
        if isinstance(dxdt, str):
            lines = [
                ln.strip()
                for ln in dxdt.strip().splitlines()
                if ln.strip()
            ]
        else:
            lines = [ln.strip() for ln in dxdt if ln.strip()]
        raw_lines = list(lines)
        lines = _normalise_indexed_tokens(lines)
        (
            equations,
            funcs,
            new_params,
            aux_names,
            inferred_states,
            rename,
        ) = _parse_string_equations(
            lines,
            raw_lines,
            registry,
            unknown_names,
            known_symbol_map,
            user_functions,
            user_function_derivatives,
            strict,
            state_names,
        )
    else:
        (
            equations,
            funcs,
            new_params,
            aux_names,
            inferred_states,
            rename,
        ) = _parse_sympy_equations(
            dxdt,
            registry,
            unknown_names,
            known_symbol_map,
            user_functions,
            user_function_derivatives,
            strict,
            state_names,
        )

    return NormalisedSystem(
        equations=equations,
        registry=registry,
        funcs=funcs,
        unknown_names=unknown_names,
        aux_names=aux_names,
        new_params=new_params,
        inferred_states=inferred_states,
        rename=rename,
        derivative_names=_derivative_print_names(
            user_functions, user_function_derivatives, rename
        ),
    )
