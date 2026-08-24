"""Shared parse-layer primitives.

Leaf module below :mod:`normalise`, :mod:`assemble`, and
:mod:`function_parser`: the ``ParsedEquations`` container, the SymPy
parse constants, and the shared user-function helpers.
"""

import re
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import attrs
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import T

from ..engine import expr as ir_expr
from ..indexedbasemaps import IndexedBases
from cubie._utils import is_devfunc

__all__ = [
    "EquationWarning",
    "KNOWN_FUNCTIONS",
    "PARSE_TRANSFORMS",
    "ParsedEquations",
    "TIME_SYMBOL",
]

# Lambda notation, Auto-number, factorial notation, implicit multiplication
PARSE_TRANSFORMS = (T[0][0], T[3][0], T[4][0], T[8][0])

_INDEXED_NAME_PATTERN = re.compile(r"(?P<name>[A-Za-z_]\w*)\[(?P<index>\d+)\]")

TIME_SYMBOL = sp.Symbol("t", real=True)


KNOWN_FUNCTIONS = {
    # Basic mathematical functions
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pow": sp.Pow,
    # Trigonometric functions
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,
    # Hyperbolic functions
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    # Special functions
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "lgamma": sp.loggamma,
    # Rounding and absolute
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    # Min/Max
    "Min": sp.Min,
    "Max": sp.Max,
    "min": sp.Min,
    "max": sp.Max,
    "Piecewise": sp.Piecewise,
    "sign": sp.sign,
}


@attrs.define(frozen=True)
class ParsedEquations:
    """Container separating state, observable, and auxiliary assignments.

    Parameters
    ----------
    ordered
        Equations in evaluation order exactly as supplied by the
        parser, as engine IR ``(lhs, rhs)`` pairs.
    state_derivatives
        Equations whose left-hand side corresponds to ``dx/dt`` outputs.
    observables
        Equations assigning user-requested observable symbols.
    auxiliaries
        Anonymous helper assignments required by either ``dx/dt`` or the
        observables.
    state_symbols
        Symbols that identify the derivative outputs.
    observable_symbols
        Symbols designating observables.
    auxiliary_symbols
        Symbols introduced for intermediate calculations.
    derivative_names
        Renamed user-function name to derivative-placeholder print
        name, for user functions with supplied derivative helpers.
    function_aliases
        Accepted IR call names and their generated-source names.
    nonfloat_functions
        IR call names printed without a precision cast.
    mass_matrix
        Solver mass matrix derived by structural simplification as
        nested row tuples; ``None`` for solved (identity) systems.
    """

    ordered: Tuple[Tuple[ir_expr.Expr, ir_expr.Expr], ...]
    state_derivatives: Tuple[Tuple[ir_expr.Expr, ir_expr.Expr], ...]
    observables: Tuple[Tuple[ir_expr.Expr, ir_expr.Expr], ...]
    auxiliaries: Tuple[Tuple[ir_expr.Expr, ir_expr.Expr], ...]
    _state_symbols: frozenset = attrs.field(repr=False)
    _observable_symbols: frozenset = attrs.field(repr=False)
    _auxiliary_symbols: frozenset = attrs.field(repr=False)
    derivative_names: Dict[str, str] = attrs.field(
        factory=dict, repr=False
    )
    function_aliases: Dict[str, str] = attrs.field(
        factory=dict, repr=False
    )
    nonfloat_functions: frozenset = attrs.field(
        factory=frozenset, repr=False
    )
    mass_matrix: Optional[Tuple[Tuple[float, ...], ...]] = attrs.field(
        default=None, repr=False
    )

    def __iter__(self) -> Iterable[Tuple[ir_expr.Expr, ir_expr.Expr]]:
        """Iterate over all equations in the original evaluation order."""

        return iter(self.ordered)

    def __len__(self) -> int:
        """Return the number of stored equations."""

        return len(self.ordered)

    def __getitem__(
        self, index: int
    ) -> Tuple[ir_expr.Expr, ir_expr.Expr]:
        """Return the equation at ``index`` from the original ordering."""

        return self.ordered[index]

    def copy(self) -> Dict[ir_expr.Expr, ir_expr.Expr]:
        """Return a mapping copy compatible with ``topological_sort``."""

        return {lhs: rhs for lhs, rhs in self.ordered}

    def to_equation_list(
        self,
    ) -> list[Tuple[ir_expr.Expr, ir_expr.Expr]]:
        """Return the stored equations as a mutable list."""

        return list(self.ordered)

    @property
    def state_symbols(self) -> frozenset:
        """Symbols representing derivative outputs."""

        return self._state_symbols

    @property
    def observable_symbols(self) -> frozenset:
        """Symbols representing observable outputs."""

        return self._observable_symbols

    @property
    def auxiliary_symbols(self) -> frozenset:
        """Symbols representing auxiliary assignments."""

        return self._auxiliary_symbols

    def non_observable_equations(
        self,
    ) -> list[Tuple[ir_expr.Expr, ir_expr.Expr]]:
        """Return equations whose outputs are not observables."""

        observable_syms = self.observable_symbols
        return [eq for eq in self.ordered if eq[0] not in observable_syms]

    @property
    def dxdt_equations(
        self,
    ) -> Tuple[Tuple[ir_expr.Expr, ir_expr.Expr], ...]:
        """Return equations required to evaluate ``dx/dt`` outputs."""

        return tuple(self.non_observable_equations())

    @property
    def observable_system(
        self,
    ) -> Tuple[Tuple[ir_expr.Expr, ir_expr.Expr], ...]:
        """Return equations contributing to observable evaluation."""

        return self.ordered

    @classmethod
    def from_equations(
        cls,
        equations: Iterable[Tuple[ir_expr.Expr, ir_expr.Expr]],
        index_map: "IndexedBases",
        derivative_names: Optional[Dict[str, str]] = None,
        function_aliases: Optional[Dict[str, str]] = None,
        nonfloat_functions: Optional[Iterable[str]] = None,
        mass_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None,
    ) -> "ParsedEquations":
        """Partition equations according to their assigned symbols.

        Membership is resolved by symbol name against the index
        map's dxdt and observable collections, so the SymPy-facing
        ``IndexedBases`` and the IR equation pairs interoperate.
        """

        if isinstance(equations, dict):
            items = list(equations.items())
        else:
            items = list(equations)
        ordered = tuple((lhs, rhs) for lhs, rhs in items)
        state_symbols = frozenset(
            ir_expr.sym(str(key))
            for key in index_map.dxdt.ref_map.keys()
        )
        observable_symbols = frozenset(
            ir_expr.sym(str(key))
            for key in index_map.observables.ref_map.keys()
        )
        state_eqs = tuple(eq for eq in ordered if eq[0] in state_symbols)
        observable_eqs = tuple(
            eq for eq in ordered if eq[0] in observable_symbols
        )
        auxiliary_eqs = tuple(
            eq
            for eq in ordered
            if eq[0] not in state_symbols and eq[0] not in observable_symbols
        )
        auxiliary_symbols = frozenset(eq[0] for eq in auxiliary_eqs)
        return cls(
            ordered=ordered,
            state_derivatives=state_eqs,
            observables=observable_eqs,
            auxiliaries=auxiliary_eqs,
            state_symbols=state_symbols,
            observable_symbols=observable_symbols,
            auxiliary_symbols=auxiliary_symbols,
            derivative_names=dict(derivative_names or {}),
            function_aliases=dict(function_aliases or {}),
            nonfloat_functions=frozenset(nonfloat_functions or ()),
            mass_matrix=mass_matrix,
        )


class EquationWarning(Warning):
    """Warning raised for recoverable issues in equation definitions."""


_func_call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


# ---------------------------- Input cleaning ------------------------------- #
def _sanitise_input_math(expr_str: str) -> str:
    """Convert Python conditional syntax into SymPy-compatible constructs.

    Parameters
    ----------
    expr_str
        Expression string to sanitise before parsing.

    Returns
    -------
    str
        SymPy-compatible expression string.
    """
    expr_str = _replace_if(expr_str)
    return expr_str


def _replace_if(expr_str: str) -> str:
    """Recursively replace ternary conditionals with ``Piecewise`` blocks.

    Parameters
    ----------
    expr_str
        Expression string that may contain inline conditional expressions.

    Returns
    -------
    str
        Expression with ternary conditionals rewritten for SymPy parsing.
    """
    match = re.search(r"(.+?) if (.+?) else (.+)", expr_str)
    if match:
        true_str = _replace_if(match.group(1).strip())
        cond_str = _replace_if(match.group(2).strip())
        false_str = _replace_if(match.group(3).strip())
        return f"Piecewise(({true_str}, {cond_str}), ({false_str}, True))"
    return expr_str


def _normalise_indexed_tokens(lines: Iterable[str]) -> list[str]:
    """Collapse numeric index access into scalar-style symbol names.

    Parameters
    ----------
    lines
        Raw equation strings supplied by the user.

    Returns
    -------
    list[str]
        Lines with occurrences of ``name[index]`` rewritten as ``nameindex``
        whenever ``index`` is an integer literal.
    """

    def _replace(match: re.Match[str]) -> str:
        base = match.group("name")
        index = match.group("index")
        return f"{base}{index}"

    return [_INDEXED_NAME_PATTERN.sub(_replace, line) for line in lines]


# ---------------------------- Function handling --------------------------- #


def _rename_user_calls(
    lines: Iterable[str],
    user_functions: Optional[Dict[str, Callable]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Rename user-defined callables to avoid collisions with SymPy names.

    Parameters
    ----------
    lines
        Raw equation strings to inspect for function calls.
    user_functions
        Mapping of user-defined names to callables referenced in the
        equations.

    Returns
    -------
    tuple
        Sanitised lines and a mapping from original names to suffixed names.
    """
    if not user_functions:
        return list(lines), {}
    rename = {name: f"{name}_" for name in user_functions.keys()}
    renamed_lines = []
    # Replace only function-call tokens: name( -> name_(
    for line in lines:
        new_line = line
        for name, underscored in rename.items():
            new_line = re.sub(rf"\b{name}\s*\(", f"{underscored}(", new_line)
        renamed_lines.append(new_line)
    return renamed_lines, rename


def _build_sympy_user_functions(
    user_functions: Optional[Dict[str, Callable]],
    rename: Dict[str, str],
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, bool]]:
    """Create SymPy ``Function`` placeholders for user-defined callables.

    Parameters
    ----------
    user_functions
        Mapping of user-provided callable names to their implementations.
    rename
        Mapping from original user function names to temporary suffixed names
        used during parsing.
    user_function_derivatives
        Mapping from user function names to callables that evaluate analytic
        derivatives.

    Returns
    -------
    tuple
        Parsing locals, pretty-name aliases, and device-function flags.

    Notes
    -----
    Device functions or user functions with derivative helpers are wrapped in
    dynamic ``Function`` subclasses whose ``fdiff`` method yields symbolic
    derivative placeholders so that downstream printers can emit gradient
    kernels.
    """
    parse_locals = {}
    alias_map = {}
    is_device_map = {}

    for orig_name, func in (user_functions or {}).items():
        sym_name = rename.get(orig_name, orig_name)
        alias_map[sym_name] = orig_name
        dev = is_devfunc(func)
        is_device_map[sym_name] = dev
        # Resolve derivative print name (if provided)
        deriv_callable = None
        if (
            user_function_derivatives
            and orig_name in user_function_derivatives
        ):
            deriv_callable = user_function_derivatives[orig_name]
        deriv_print_name = None
        if deriv_callable is not None:
            try:
                deriv_print_name = deriv_callable.__name__
            except Exception:
                deriv_print_name = None
        should_wrap = dev or deriv_callable is not None
        if should_wrap:
            # Build a dynamic Function subclass with name sym_name and fdiff
            # that generates <deriv_print_name or d_orig>(args..., argindex-1)
            def _make_class(
                sym_name=sym_name,
                orig_name=orig_name,
                deriv_print_name=deriv_print_name,
            ):
                class _UserDevFunc(sp.Function):
                    nargs = None

                    @classmethod
                    def eval(cls, *args):
                        return None

                    def fdiff(self, argindex=1):
                        target_name = deriv_print_name or f"d_{orig_name}"
                        deriv_func = sp.Function(target_name)
                        return deriv_func(*self.args, sp.Integer(argindex - 1))

                _UserDevFunc.__name__ = sym_name
                return _UserDevFunc

            parse_locals[sym_name] = _make_class()
        else:
            parse_locals[sym_name] = sp.Function(sym_name)
    return parse_locals, alias_map, is_device_map


def _inline_nondevice_calls(
    expr: sp.Expr,
    user_functions: Dict[str, Callable],
    rename: Dict[str, str],
) -> sp.Expr:
    """Inline callable results for non-device user functions when possible.

    Parameters
    ----------
    expr
        Expression potentially containing calls to user-defined functions.
    user_functions
        Mapping from user-provided function names to their implementations.
    rename
        Mapping from original user function names to suffixed parser names.

    Returns
    -------
    sympy.Expr
        Expression with inlineable calls replaced by their evaluated result.
    """
    if not user_functions:
        return expr

    def _try_inline(applied):
        # applied is an AppliedUndef or similar; get its name
        name = applied.func.__name__
        # reverse-map if this is an underscored user function
        orig_name = None
        for k, v in rename.items():
            if v == name:
                orig_name = k
                break
        fn = user_functions.get(orig_name)
        if fn is None or is_devfunc(fn):
            return applied
        try:
            # Try evaluate on SymPy args
            val = fn(*applied.args)
            # Ensure it's a SymPy expression
            if isinstance(val, (sp.Expr, sp.Symbol)):
                return val
            # Fall back to keeping symbolic call
            return applied
        except Exception:
            return applied

    # Replace any AppliedUndef whose name matches an underscored function
    for _, sym_name in rename.items():
        f = sp.Function(sym_name)
        expr = expr.replace(
            lambda e: isinstance(e, AppliedUndef) and e.func == f, _try_inline
        )
    return expr
