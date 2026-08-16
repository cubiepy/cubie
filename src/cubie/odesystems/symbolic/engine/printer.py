"""Render engine IR as Numba-CUDA source."""

from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple

from cubie.odesystems.symbolic.engine.expr import (
    Add,
    Arr,
    BoolConst,
    BoolOp,
    Call,
    Expr,
    Local,
    Mul,
    Num,
    Piecewise,
    Pow,
    Rel,
    Sym,
)
__all__ = ["CUDA_FUNCTIONS", "IRPrinter", "print_cuda",
           "print_cuda_multiple"]

# Map IR call names to CUDA/Python math equivalents. Keys match the
# names produced by the SymPy conversion (SymPy class names) and the
# parser's KNOWN_FUNCTIONS.
CUDA_FUNCTIONS: Dict[str, str] = {
    "sin": "math.sin",
    "cos": "math.cos",
    "tan": "math.tan",
    "asin": "math.asin",
    "acos": "math.acos",
    "atan": "math.atan",
    "atan2": "math.atan2",
    "sinh": "math.sinh",
    "cosh": "math.cosh",
    "tanh": "math.tanh",
    "asinh": "math.asinh",
    "acosh": "math.acosh",
    "atanh": "math.atanh",
    "exp": "math.exp",
    "expm1": "math.expm1",
    "log": "math.log",
    "log2": "math.log2",
    "log10": "math.log10",
    "log1p": "math.log1p",
    "erf": "math.erf",
    "erfc": "math.erfc",
    "gamma": "math.gamma",
    "loggamma": "math.lgamma",
    "hypot": "math.hypot",
    "Abs": "math.fabs",
    "floor": "math.floor",
    "ceiling": "math.ceil",
    "sqrt": "math.sqrt",
    "pow": "math.pow",
    "Min": "min",
    "Max": "max",
    "copysign": "math.copysign",
    "fmod": "math.fmod",
    "modf": "math.modf",
    "frexp": "math.frexp",
    "ldexp": "math.ldexp",
    "remainder": "math.remainder",
    "isnan": "math.isnan",
    "isinf": "math.isinf",
    "isfinite": "math.isfinite",
}

# Precedence levels (Python expression grammar, higher binds tighter)
_PREC_TERNARY = 1
_PREC_OR = 2
_PREC_AND = 3
_PREC_NOT = 4
_PREC_REL = 5
_PREC_BITOR = 6
_PREC_BITAND = 7
_PREC_ADD = 8
_PREC_MUL = 9
_PREC_UNARY = 10
_PREC_POW = 11
_PREC_ATOM = 12


def _format_number(value) -> str:
    """Format a numeric payload as source text."""
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    return repr(value)


class IRPrinter:
    """Stateless-per-expression printer for engine IR nodes.

    Parameters
    ----------
    symbol_map
        Mapping from symbol *name* to replacement expression (an
        :class:`Arr` or :class:`Sym`) for scalar-to-array remapping.
    function_aliases
        Mapping from renamed user-function name to its original
        (printable) name.

    Notes
    -----
    Constants never reach the printer as symbols: their values are
    substituted into the IR as numeric literals before code
    generation, so they print as ``precision(...)`` literals.
    """

    def __init__(
        self,
        symbol_map: Optional[Dict[str, Expr]] = None,
        function_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.symbol_map = symbol_map or {}
        self.function_aliases = function_aliases or {}

    # -- public API -------------------------------------------------

    def assignment(self, lhs: Expr, rhs: Expr) -> str:
        """Return ``lhs = rhs`` as one source line."""
        return f"{self._atom_target(lhs)} = {self._print(rhs, _PREC_TERNARY)}"

    def expression(self, node: Expr) -> str:
        """Return the source text for ``node``."""
        return self._print(node, _PREC_TERNARY)

    # -- internals ----------------------------------------------------

    def _atom_target(self, lhs: Expr) -> str:
        if isinstance(lhs, Local):
            return lhs.name
        if isinstance(lhs, Sym):
            mapped = self.symbol_map.get(lhs.name)
            if mapped is not None:
                return self._print(mapped, _PREC_ATOM)
            return lhs.name
        if isinstance(lhs, Arr):
            return f"{lhs.name}[{lhs.index}]"
        raise TypeError(f"invalid assignment target: {lhs!r}")

    def _paren(self, text: str, prec: int, context: int) -> str:
        if prec < context:
            return f"({text})"
        return text

    def _print(self, node: Expr, context: int) -> str:
        text, prec = self._render(node)
        return self._paren(text, prec, context)

    def _render(self, node: Expr) -> Tuple[str, int]:
        if isinstance(node, Num):
            text = _format_number(node.value)
            return f"precision({text})", _PREC_ATOM
        if isinstance(node, Local):
            return node.name, _PREC_ATOM
        if isinstance(node, Sym):
            mapped = self.symbol_map.get(node.name)
            if mapped is not None and mapped is not node:
                return self._render(mapped)
            return node.name, _PREC_ATOM
        if isinstance(node, Arr):
            return f"{node.name}[{node.index}]", _PREC_ATOM
        if isinstance(node, Add):
            return self._render_add(node)
        if isinstance(node, Mul):
            return self._render_mul(node)
        if isinstance(node, Pow):
            return self._render_pow(node)
        if isinstance(node, Call):
            return self._render_call(node)
        if isinstance(node, Piecewise):
            return self._render_piecewise(node)
        if isinstance(node, Rel):
            lhs = self._print(node.lhs, _PREC_ADD)
            rhs = self._print(node.rhs, _PREC_ADD)
            return f"{lhs} {node.op} {rhs}", _PREC_REL
        if isinstance(node, BoolOp):
            return self._render_bool(node)
        if isinstance(node, BoolConst):
            return ("True" if node.value else "False"), _PREC_ATOM
        raise TypeError(f"unknown node type: {type(node)!r}")

    def _render_add(self, node: Add) -> Tuple[str, int]:
        parts: List[str] = []
        for term in node.args:
            coeff_negative, positive = _split_negation(term)
            if coeff_negative:
                # A subtracted sum must keep its parentheses:
                # ``a - (b + c)``, never ``a - b + c``.
                rendered = self._print(positive, _PREC_ADD + 1)
                if not parts:
                    parts.append(f"-{rendered}")
                else:
                    parts.append(f"- {rendered}")
            else:
                rendered = self._print(positive, _PREC_ADD)
                if not parts:
                    parts.append(rendered)
                else:
                    parts.append(f"+ {rendered}")
        return " ".join(parts), _PREC_ADD

    def _render_mul(self, node: Mul) -> Tuple[str, int]:
        negated, positive = _split_negation(node)
        if negated:
            inner = self._print(positive, _PREC_UNARY)
            return f"-{inner}", _PREC_UNARY
        numerator: List[Expr] = []
        denominator: List[Expr] = []
        for factor in node.args:
            if isinstance(factor, Pow) and isinstance(factor.exp, Num):
                exp_value = factor.exp.value
                if (
                    isinstance(exp_value, (int, Fraction))
                    and exp_value < 0
                ):
                    denominator.append(
                        _pow_or_base(factor.base, -exp_value)
                    )
                    continue
            numerator.append(factor)

        if not numerator:
            numer_text = "precision(1)"
        else:
            # Factors after the first print at one level tighter:
            # ``%`` shares multiplication precedence, so a bare
            # ``z*x % y`` would parse as ``(z*x) % y``.
            numer_parts = [
                self._print(
                    f, _PREC_MUL if i == 0 else _PREC_MUL + 1
                )
                for i, f in enumerate(numerator)
            ]
            numer_text = "*".join(numer_parts)
        if not denominator:
            return numer_text, _PREC_MUL
        if len(numerator) > 1:
            numer_text = f"({numer_text})"
        denom_parts = [
            self._print(f, _PREC_UNARY) for f in denominator
        ]
        denom_text = "*".join(denom_parts)
        if len(denominator) > 1:
            denom_text = f"({denom_text})"
        return f"{numer_text}/{denom_text}", _PREC_MUL

    def _render_pow(self, node: Pow) -> Tuple[str, int]:
        base_text = self._print(node.base, _PREC_POW + 1)
        exp = node.exp

        if isinstance(exp, Num):
            value = exp.value
            as_float = float(value)
            if as_float == 0.5:
                inner = self._print(node.base, _PREC_TERNARY)
                return f"math.sqrt({inner})", _PREC_ATOM
            if as_float == -0.5:
                inner = self._print(node.base, _PREC_TERNARY)
                return (
                    f"(precision(1)/math.sqrt({inner}))",
                    _PREC_ATOM,
                )
            if isinstance(value, int):
                if value < 0:
                    positive = _pow_or_base(node.base, -value)
                    denom = self._print(positive, _PREC_UNARY)
                    return (
                        f"(precision(1)/{denom})",
                        _PREC_ATOM,
                    )
                if value in (2, 3):
                    return self._render_mult_chain(node.base, value)
                return (
                    f"{base_text}**precision({value})",
                    _PREC_POW,
                )
            if isinstance(value, float) and as_float in (2.0, 3.0):
                return self._render_mult_chain(
                    node.base, int(as_float)
                )
            exp_text = _format_number(value)
            return f"{base_text}**precision({exp_text})", _PREC_POW

        exp_text = self._print(exp, _PREC_POW)
        return f"{base_text}**{exp_text}", _PREC_POW

    def _render_mult_chain(
        self, base: Expr, power: int
    ) -> Tuple[str, int]:
        base_text = self._print(base, _PREC_UNARY)
        chain = "*".join([base_text] * power)
        return f"({chain})", _PREC_ATOM

    def _render_call(self, node: Call) -> Tuple[str, int]:
        name = node.name
        if name == "sign":
            inner = self._print(node.args[0], _PREC_REL)
            return (
                f"selp({inner} == precision(0), precision(0), "
                f"math.copysign(precision(1), {inner}))"
            ), _PREC_ATOM
        if name == "Mod":
            lhs = self._print(node.args[0], _PREC_MUL)
            rhs = self._print(node.args[1], _PREC_UNARY)
            return f"{lhs} % {rhs}", _PREC_MUL
        target = CUDA_FUNCTIONS.get(name)
        if target is None:
            target = self.function_aliases.get(name)
        if target is None:
            raise ValueError(f"unsupported function in IR: {name}")
        args = ", ".join(
            self._print(a, _PREC_TERNARY) for a in node.args
        )
        return f"{target}({args})", _PREC_ATOM

    def _render_piecewise(self, node: Piecewise) -> Tuple[str, int]:
        # selp lowers to a hardware select; a ternary would branch.
        pairs = list(node.pairs)
        last_value, _ = pairs[-1]
        rendered = self._print(last_value, _PREC_TERNARY)
        for value, cond in reversed(pairs[:-1]):
            value_text = self._print(value, _PREC_TERNARY)
            cond_text = self._print(cond, _PREC_TERNARY)
            rendered = f"selp({cond_text}, {value_text}, {rendered})"
        return rendered, _PREC_ATOM

    def _render_bool(self, node: BoolOp) -> Tuple[str, int]:
        # Bitwise & / | avoid short-circuit branches.
        if node.kind == "not":
            inner = self._print(node.args[0], _PREC_NOT)
            return f"not {inner}", _PREC_NOT
        joiner = " & " if node.kind == "and" else " | "
        level = _PREC_BITAND if node.kind == "and" else _PREC_BITOR
        parts = [self._print(a, level) for a in node.args]
        return joiner.join(parts), level


def _split_negation(term: Expr) -> Tuple[bool, Expr]:
    """Split a leading numeric minus sign off ``term`` for printing."""
    if isinstance(term, Num):
        value = term.value
        if (isinstance(value, float) and value < 0.0) or (
            not isinstance(value, float) and value < 0
        ):
            from cubie.odesystems.symbolic.engine.expr import num

            return True, num(-value)
        return False, term
    if isinstance(term, Mul) and isinstance(term.args[0], Num):
        coeff = term.args[0].value
        negative = (
            coeff < 0.0 if isinstance(coeff, float) else coeff < 0
        )
        if negative:
            from cubie.odesystems.symbolic.engine.expr import (
                mul,
                num,
            )

            flipped = mul(num(-coeff), *term.args[1:])
            return True, flipped
    return False, term


def _pow_or_base(base: Expr, exponent) -> Expr:
    """Return ``base ** exponent`` folding the exponent-one case."""
    from cubie.odesystems.symbolic.engine.expr import num, pow_

    if not isinstance(exponent, (int, Fraction)):
        raise TypeError("positive numeric exponent expected")
    if exponent == 1:
        return base
    return pow_(base, num(exponent))


def _coerce_node(node) -> Expr:
    """Accept SymPy input at the printer boundary."""
    if isinstance(node, Expr):
        return node
    from cubie.odesystems.symbolic.engine.from_sympy import (
        from_sympy,
    )

    return from_sympy(node)


def _coerce_symbol_map(
    symbol_map,
) -> Optional[Dict[str, Expr]]:
    """Accept SymPy-keyed symbol maps at the printer boundary."""
    if not symbol_map:
        return symbol_map
    coerced: Dict[str, Expr] = {}
    for key, value in symbol_map.items():
        if key == "__function_aliases__":
            continue
        if not isinstance(value, Expr) and not hasattr(
            value, "free_symbols"
        ):
            # Symbol tables can carry device callables and function
            # classes; only expression-valued entries remap symbols.
            continue
        coerced[str(key)] = _coerce_node(value)
    return coerced


def print_cuda(
    node: Expr,
    symbol_map: Optional[Dict[str, Expr]] = None,
    function_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Render one IR (or SymPy) expression as CUDA source text."""
    printer = IRPrinter(
        symbol_map=_coerce_symbol_map(symbol_map),
        function_aliases=function_aliases,
    )
    return printer.expression(_coerce_node(node))


def print_cuda_multiple(
    assignments: Iterable[Tuple[Expr, Expr]],
    symbol_map: Optional[Dict[str, Expr]] = None,
    function_aliases: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Render assignment pairs as CUDA-compatible source lines.

    Parameters
    ----------
    assignments
        ``(lhs, rhs)`` IR pairs in emission order.
    symbol_map
        Mapping from symbol name to replacement node (array refs).
    function_aliases
        Renamed-user-function to printable-name mapping.

    Returns
    -------
    list of str
        One source line per assignment.
    """
    if function_aliases is None and isinstance(symbol_map, dict):
        aliases = symbol_map.get("__function_aliases__")
        if isinstance(aliases, dict):
            function_aliases = aliases
    printer = IRPrinter(
        symbol_map=_coerce_symbol_map(symbol_map),
        function_aliases=function_aliases,
    )
    return [
        printer.assignment(_coerce_node(lhs), _coerce_node(rhs))
        for lhs, rhs in assignments
    ]
