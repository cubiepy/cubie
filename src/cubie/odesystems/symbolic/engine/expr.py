"""Define the interned expression tree used by code generation."""

import math
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Union
from weakref import WeakValueDictionary

__all__ = [
    "Expr",
    "Num",
    "Sym",
    "Local",
    "Arr",
    "Add",
    "Mul",
    "Pow",
    "Call",
    "Piecewise",
    "Rel",
    "BoolOp",
    "BoolConst",
    "num",
    "sym",
    "local",
    "arr",
    "add",
    "mul",
    "sub",
    "div",
    "neg",
    "pow_",
    "call",
    "piecewise",
    "rel",
    "bool_op",
    "TRUE",
    "FALSE",
    "ZERO",
    "ONE",
    "NEG_ONE",
    "xreplace",
    "diff",
    "free_atoms",
    "count_ops",
    "DifferentiationError",
]

NumberLike = Union[int, float, Fraction]

_INTERN = WeakValueDictionary()


def _intern(key: tuple, factory: Callable[[], "Expr"]) -> "Expr":
    """Return the interned node for ``key``, creating it when absent."""
    node = _INTERN.get(key)
    if node is None:
        node = factory()
        _INTERN[key] = node
    return node


class Expr:
    """Abstract immutable expression node.

    Attributes
    ----------
    sort_key
        Structural total-order key. Used to order commutative
        arguments deterministically across processes.
    """

    __slots__ = ("sort_key", "_free", "__weakref__")

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    # Convenience operators used by generator code -----------------
    def __add__(self, other: "ExprLike") -> "Expr":
        return add(self, other)

    def __radd__(self, other: "ExprLike") -> "Expr":
        return add(other, self)

    def __sub__(self, other: "ExprLike") -> "Expr":
        return sub(self, other)

    def __rsub__(self, other: "ExprLike") -> "Expr":
        return sub(other, self)

    def __mul__(self, other: "ExprLike") -> "Expr":
        return mul(self, other)

    def __rmul__(self, other: "ExprLike") -> "Expr":
        return mul(other, self)

    def __truediv__(self, other: "ExprLike") -> "Expr":
        return div(self, other)

    def __rtruediv__(self, other: "ExprLike") -> "Expr":
        return div(other, self)

    def __pow__(self, other: "ExprLike") -> "Expr":
        return pow_(self, other)

    def __neg__(self) -> "Expr":
        return neg(self)


ExprLike = Union[Expr, int, float, Fraction]


class Num(Expr):
    """Numeric literal. ``value`` is ``int``, ``float`` or ``Fraction``."""

    __slots__ = ("value",)

    def __init__(self, value: NumberLike) -> None:
        self.value = value
        self.sort_key = (
            0,
            _num_sort_value(value),
            _num_type_rank(value),
            str(value),
        )

    def __repr__(self) -> str:
        return f"Num({self.value!r})"

    def __reduce__(self):
        return (num, (self.value,))


class Sym(Expr):
    """Named scalar symbol."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name
        self.sort_key = (1, name)

    def __repr__(self) -> str:
        return f"Sym({self.name})"

    def __reduce__(self):
        return (sym, (self.name,))


class Local(Expr):
    """Generated scalar local."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name
        self.sort_key = (2, name)

    def __repr__(self) -> str:
        return f"Local({self.name})"

    def __reduce__(self):
        return (local, (self.name,))


class Arr(Expr):
    """Array element reference ``name[index]`` with a fixed int index."""

    __slots__ = ("name", "index")

    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index
        self.sort_key = (2, name, index)

    def __repr__(self) -> str:
        return f"Arr({self.name}[{self.index}])"

    def __reduce__(self):
        return (arr, (self.name, self.index))


class Add(Expr):
    """N-ary sum. Arguments are stored in sort-key order."""

    __slots__ = ("args",)

    def __init__(self, args: Tuple[Expr, ...]) -> None:
        self.args = args
        self.sort_key = (5, tuple(a.sort_key for a in args))

    def __repr__(self) -> str:
        return f"Add({', '.join(map(repr, self.args))})"

    def __reduce__(self):
        return (add, tuple(self.args))


class Mul(Expr):
    """N-ary product. Arguments are stored in sort-key order."""

    __slots__ = ("args",)

    def __init__(self, args: Tuple[Expr, ...]) -> None:
        self.args = args
        self.sort_key = (6, tuple(a.sort_key for a in args))

    def __repr__(self) -> str:
        return f"Mul({', '.join(map(repr, self.args))})"

    def __reduce__(self):
        return (mul, tuple(self.args))


class Pow(Expr):
    """Power ``base ** exp``."""

    __slots__ = ("base", "exp")

    def __init__(self, base: Expr, exp: Expr) -> None:
        self.base = base
        self.exp = exp
        self.sort_key = (7, base.sort_key, exp.sort_key)

    def __repr__(self) -> str:
        return f"Pow({self.base!r}, {self.exp!r})"

    def __reduce__(self):
        return (pow_, (self.base, self.exp))


class Call(Expr):
    """Applied function ``name(args...)``.

    ``name`` is the printed lookup key: a known math function name
    (``exp``, ``sin`` …), a renamed user function (``foo_``), or a
    derivative placeholder (``d_foo``).
    """

    __slots__ = ("name", "args")

    def __init__(self, name: str, args: Tuple[Expr, ...]) -> None:
        self.name = name
        self.args = args
        self.sort_key = (8, name, tuple(a.sort_key for a in args))

    def __repr__(self) -> str:
        return f"Call({self.name}, {', '.join(map(repr, self.args))})"

    def __reduce__(self):
        return (call, (self.name,) + tuple(self.args))


class Piecewise(Expr):
    """Ordered ``(value, condition)`` selection.

    The final condition is :data:`TRUE`.
    """

    __slots__ = ("pairs",)

    def __init__(self, pairs: Tuple[Tuple[Expr, Expr], ...]) -> None:
        self.pairs = pairs
        self.sort_key = (
            9,
            tuple((v.sort_key, c.sort_key) for v, c in pairs),
        )

    def __repr__(self) -> str:
        return f"Piecewise({self.pairs!r})"

    def __reduce__(self):
        return (piecewise, tuple(self.pairs))


class Rel(Expr):
    """Binary relation ``lhs <op> rhs`` with op in <, <=, >, >=, ==, !=."""

    __slots__ = ("op", "lhs", "rhs")

    def __init__(self, op: str, lhs: Expr, rhs: Expr) -> None:
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.sort_key = (10, op, lhs.sort_key, rhs.sort_key)

    def __repr__(self) -> str:
        return f"Rel({self.lhs!r} {self.op} {self.rhs!r})"

    def __reduce__(self):
        return (rel, (self.op, self.lhs, self.rhs))


class BoolOp(Expr):
    """Boolean combination: kind in {'and', 'or', 'not'}."""

    __slots__ = ("kind", "args")

    def __init__(self, kind: str, args: Tuple[Expr, ...]) -> None:
        self.kind = kind
        self.args = args
        self.sort_key = (11, kind, tuple(a.sort_key for a in args))

    def __repr__(self) -> str:
        return f"BoolOp({self.kind}, {self.args!r})"

    def __reduce__(self):
        return (bool_op, (self.kind,) + tuple(self.args))


class BoolConst(Expr):
    """Boolean literal (interned singletons :data:`TRUE`/:data:`FALSE`)."""

    __slots__ = ("value",)

    def __init__(self, value: bool) -> None:
        self.value = value
        self.sort_key = (12, value)

    def __repr__(self) -> str:
        return f"BoolConst({self.value})"

    def __reduce__(self):
        return (_bool_const, (self.value,))


def _num_sort_value(value: NumberLike) -> float:
    """Return ``float(value)`` for ordering, saturating on overflow.

    The trailing ``str(value)`` sort-key element keeps the order
    total when two distinct payloads saturate to the same float.
    """
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _num_type_rank(value: NumberLike) -> int:
    """Rank numeric payload types so equal-valued literals stay distinct."""
    if isinstance(value, bool):
        raise TypeError("bool is not a numeric literal")
    if isinstance(value, int):
        return 0
    if isinstance(value, Fraction):
        return 1
    return 2


def _norm_number(value: NumberLike) -> NumberLike:
    """Normalise a numeric payload to builtin int / Fraction / float.

    Subclass instances (``numpy.float64`` passes ``isinstance(x,
    float)``) are coerced to the builtin type so ``repr`` in the
    printer emits plain literals.
    """
    if isinstance(value, bool):
        raise TypeError("bool is not a numeric literal")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return int(value)
        return value
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)
    if item is not None:
        # NumPy scalar outside the float/int subclass hierarchy
        # (e.g. float32); .item() yields the closest builtin.
        return _norm_number(item())
    raise TypeError(f"unsupported numeric type: {type(value)!r}")


def num(value: NumberLike) -> Num:
    """Return the interned numeric literal for ``value``."""
    value = _norm_number(value)
    key = ("num", type(value).__name__, value)
    return _intern(key, lambda: Num(value))


ZERO = num(0)
ONE = num(1)
NEG_ONE = num(-1)


def sym(name: str) -> Sym:
    """Return the interned scalar symbol named ``name``."""
    key = ("sym", name)
    return _intern(key, lambda: Sym(name))


def local(name: str) -> Local:
    """Return the generated local named ``name``."""
    key = ("local", name)
    return _intern(key, lambda: Local(name))


def arr(name: str, index: int) -> Arr:
    """Return the interned array reference ``name[index]``."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(
            f"array index must be a plain int, got {index!r}"
        )
    key = ("arr", name, index)
    return _intern(key, lambda: Arr(name, index))


TRUE = _intern(("bool", True), lambda: BoolConst(True))
FALSE = _intern(("bool", False), lambda: BoolConst(False))


def _bool_const(value: bool) -> "BoolConst":
    """Return the interned boolean literal (pickle restore hook)."""
    return TRUE if value else FALSE


def _as_expr(value: ExprLike) -> Expr:
    """Coerce a Python number to a :class:`Num` node."""
    if isinstance(value, Expr):
        return value
    return num(value)


def is_zero(node: Expr) -> bool:
    """Return whether ``node`` is a numeric zero of any payload type."""
    return isinstance(node, Num) and node.value == 0


def is_one(node: Expr) -> bool:
    """Return whether ``node`` is a numeric one of any payload type."""
    return isinstance(node, Num) and node.value == 1


def _num_add(a: NumberLike, b: NumberLike) -> NumberLike:
    if isinstance(a, float) or isinstance(b, float):
        return float(a) + float(b)
    return _norm_number(Fraction(a) + Fraction(b))


def _num_mul(a: NumberLike, b: NumberLike) -> NumberLike:
    if isinstance(a, float) or isinstance(b, float):
        return float(a) * float(b)
    return _norm_number(Fraction(a) * Fraction(b))


def add(*terms: ExprLike) -> Expr:
    """Return the folded, interned sum of ``terms``.

    Flattens nested sums, folds numeric terms, drops zeros, and
    collects like terms by their non-numeric factor.
    """
    const: NumberLike = 0
    # coeff accumulation keyed by the non-numeric part of each term
    coeffs: Dict[Expr, NumberLike] = {}
    order: List[Expr] = []

    def absorb(node: Expr) -> None:
        nonlocal const
        if isinstance(node, Num):
            const = _num_add(const, node.value)
            return
        if isinstance(node, Add):
            for sub_arg in node.args:
                absorb(sub_arg)
            return
        coeff: NumberLike = 1
        rest = node
        if isinstance(node, Mul) and isinstance(node.args[0], Num):
            coeff = node.args[0].value
            remaining = node.args[1:]
            if len(remaining) == 1:
                rest = remaining[0]
            else:
                rest = _raw_mul(remaining)
        if rest in coeffs:
            coeffs[rest] = _num_add(coeffs[rest], coeff)
        else:
            coeffs[rest] = coeff
            order.append(rest)

    for term in terms:
        absorb(_as_expr(term))

    args: List[Expr] = []
    for rest in order:
        coeff = coeffs[rest]
        if coeff == 0:
            continue
        if coeff == 1:
            args.append(rest)
        else:
            args.append(_raw_mul_with_coeff(coeff, rest))
    args.sort(key=lambda node: node.sort_key)
    if const != 0:
        # Numeric term goes last so sums print naturally
        # (``x + precision(5)``); the position is canonical because
        # folding leaves at most one numeric term.
        args.append(num(const))
    if not args:
        return ZERO
    if len(args) == 1:
        return args[0]
    key = ("add", tuple(id(a) for a in args))
    args_tuple = tuple(args)
    return _intern(key, lambda: Add(args_tuple))


def _raw_mul(factors: Tuple[Expr, ...]) -> Expr:
    """Build an interned Mul from pre-folded factors (len >= 2)."""
    ordered = sorted(factors, key=lambda node: node.sort_key)
    key = ("mul", tuple(id(a) for a in ordered))
    ordered_tuple = tuple(ordered)
    return _intern(key, lambda: Mul(ordered_tuple))


def _raw_mul_with_coeff(coeff: NumberLike, rest: Expr) -> Expr:
    """Build ``coeff * rest`` without re-running full mul folding."""
    if isinstance(rest, Mul):
        return _raw_mul((num(coeff),) + rest.args)
    return _raw_mul((num(coeff), rest))


def mul(*factors: ExprLike) -> Expr:
    """Return the folded, interned product of ``factors``.

    Flattens nested products, folds numeric factors, short-circuits
    zero, drops ones, and combines powers of identical bases.
    """
    const: NumberLike = 1
    exps: Dict[Expr, List[Expr]] = {}
    order: List[Expr] = []

    def absorb(node: Expr) -> None:
        nonlocal const
        if isinstance(node, Num):
            const = _num_mul(const, node.value)
            return
        if isinstance(node, Mul):
            for sub_arg in node.args:
                absorb(sub_arg)
            return
        base = node
        exponent: Expr = ONE
        if isinstance(node, Pow):
            base = node.base
            exponent = node.exp
        if base in exps:
            exps[base].append(exponent)
        else:
            exps[base] = [exponent]
            order.append(base)

    for factor in factors:
        absorb(_as_expr(factor))

    if const == 0:
        return ZERO

    args: List[Expr] = []
    for base in order:
        exponent = add(*exps[base]) if len(exps[base]) > 1 else exps[base][0]
        factor = pow_(base, exponent)
        if factor is ONE:
            continue
        if isinstance(factor, Num):
            const = _num_mul(const, factor.value)
            continue
        args.append(factor)

    if const == 0:
        # Power combination can fold a factor to a numeric zero
        # (e.g. ``0**x * 0**(2 - x)``), so re-check after folding.
        return ZERO
    const_is_one = const == 1
    if not args:
        return num(const)
    if not const_is_one:
        args.append(num(const))
    if len(args) == 1:
        return args[0]
    args.sort(key=lambda node: node.sort_key)
    key = ("mul", tuple(id(a) for a in args))
    args_tuple = tuple(args)
    return _intern(key, lambda: Mul(args_tuple))


def neg(value: ExprLike) -> Expr:
    """Return ``-value``."""
    return mul(NEG_ONE, value)


def sub(a: ExprLike, b: ExprLike) -> Expr:
    """Return ``a - b``."""
    return add(a, mul(NEG_ONE, b))


def div(a: ExprLike, b: ExprLike) -> Expr:
    """Return ``a / b`` as ``a * b**-1``."""
    return mul(a, pow_(b, NEG_ONE))


def pow_(base: ExprLike, exp: ExprLike) -> Expr:
    """Return the folded, interned power ``base ** exp``."""
    base = _as_expr(base)
    exp = _as_expr(exp)

    if isinstance(exp, Num):
        if exp.value == 0:
            return ONE
        if exp.value == 1:
            return base
        if isinstance(base, Num):
            folded = _fold_numeric_pow(base.value, exp.value)
            if folded is not None:
                return num(folded)
        if isinstance(base, Pow) and isinstance(exp.value, int):
            # (b**e1)**n -> b**(e1*n) is exact for integer n
            return pow_(base.base, mul(base.exp, exp))
    if isinstance(base, Num) and base.value == 1:
        return ONE
    key = ("pow", id(base), id(exp))
    return _intern(key, lambda: Pow(base, exp))


def _fold_numeric_pow(
    base: NumberLike, exp: NumberLike
) -> Optional[NumberLike]:
    """Fold a numeric power when the result is exactly representable."""
    if isinstance(exp, int):
        if isinstance(base, int):
            if exp >= 0:
                return base**exp
            if base != 0:
                return _norm_number(Fraction(base) ** exp)
            return None
        if isinstance(base, Fraction):
            if base != 0 or exp >= 0:
                return _norm_number(base**exp)
            return None
        # Zero base with negative exponent stays a Pow node.
        try:
            return float(base) ** exp
        except (OverflowError, ZeroDivisionError):
            return None
    if isinstance(base, float) or isinstance(exp, float):
        try:
            result = float(base) ** float(exp)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        if isinstance(result, complex):
            return None
        return result
    return None


def call(name: str, *args: ExprLike) -> Expr:
    """Return the interned function application ``name(args...)``."""
    arg_nodes = tuple(_as_expr(a) for a in args)
    key = ("call", name, tuple(id(a) for a in arg_nodes))
    return _intern(key, lambda: Call(name, arg_nodes))


def piecewise(*pairs: Tuple[ExprLike, Expr]) -> Expr:
    """Return the interned piecewise selection over ``(value, cond)``.

    Branches after the first :data:`TRUE` condition are dropped; a
    single-branch piecewise with a true condition collapses to its
    value.
    """
    norm: List[Tuple[Expr, Expr]] = []
    for value, cond in pairs:
        value = _as_expr(value)
        if cond is FALSE:
            continue
        norm.append((value, cond))
        if cond is TRUE:
            break
    if not norm:
        raise ValueError("piecewise requires at least one live branch")
    if norm[-1][1] is not TRUE:
        raise ValueError("piecewise requires a final true branch")
    if len(norm) == 1 and norm[0][1] is TRUE:
        return norm[0][0]
    pairs_tuple = tuple(norm)
    key = (
        "piecewise",
        tuple((id(v), id(c)) for v, c in pairs_tuple),
    )
    return _intern(key, lambda: Piecewise(pairs_tuple))


_REL_FOLDS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def rel(op: str, lhs: ExprLike, rhs: ExprLike) -> Expr:
    """Return ``lhs <op> rhs``; two numeric operands fold to a bool."""
    if op not in ("<", "<=", ">", ">=", "==", "!="):
        raise ValueError(f"unsupported relational operator: {op}")
    lhs = _as_expr(lhs)
    rhs = _as_expr(rhs)
    if isinstance(lhs, Num) and isinstance(rhs, Num):
        return _bool_const(_REL_FOLDS[op](lhs.value, rhs.value))
    key = ("rel", op, id(lhs), id(rhs))
    return _intern(key, lambda: Rel(op, lhs, rhs))


def bool_op(kind: str, *args: Expr) -> Expr:
    """Return the boolean combination; literal arguments fold."""
    if kind not in ("and", "or", "not"):
        raise ValueError(f"unsupported boolean operator: {kind}")
    if kind == "not" and len(args) != 1:
        raise ValueError("'not' takes exactly one argument")
    if kind == "not":
        if args[0] is TRUE:
            return FALSE
        if args[0] is FALSE:
            return TRUE
    else:
        absorbing = FALSE if kind == "and" else TRUE
        neutral = TRUE if kind == "and" else FALSE
        live = []
        for arg in args:
            if arg is absorbing:
                return absorbing
            if arg is neutral:
                continue
            live.append(arg)
        if not live:
            return neutral
        if len(live) == 1:
            return live[0]
        args = tuple(live)
    key = ("boolop", kind, tuple(id(a) for a in args))
    args_tuple = tuple(args)
    return _intern(key, lambda: BoolOp(kind, args_tuple))


# ----------------------------------------------------------------
# Traversals
# ----------------------------------------------------------------


def _children(node: Expr) -> Tuple[Expr, ...]:
    """Return the direct children of ``node``."""
    if isinstance(node, (Add, Mul, Call, BoolOp)):
        return node.args
    if isinstance(node, Pow):
        return (node.base, node.exp)
    if isinstance(node, Piecewise):
        flat: List[Expr] = []
        for value, cond in node.pairs:
            flat.append(value)
            flat.append(cond)
        return tuple(flat)
    if isinstance(node, Rel):
        return (node.lhs, node.rhs)
    return ()


def _rebuild(node: Expr, children: Tuple[Expr, ...]) -> Expr:
    """Rebuild ``node`` with replacement ``children``."""
    if isinstance(node, Add):
        return add(*children)
    if isinstance(node, Mul):
        return mul(*children)
    if isinstance(node, Pow):
        return pow_(children[0], children[1])
    if isinstance(node, Call):
        return call(node.name, *children)
    if isinstance(node, Piecewise):
        pairs = tuple(
            (children[i], children[i + 1])
            for i in range(0, len(children), 2)
        )
        return piecewise(*pairs)
    if isinstance(node, Rel):
        return rel(node.op, children[0], children[1])
    if isinstance(node, BoolOp):
        return bool_op(node.kind, *children)
    return node


def xreplace(
    node: Expr,
    mapping: Dict[Expr, Expr],
    memo: Optional[Dict[Expr, Expr]] = None,
) -> Expr:
    """Replace nodes appearing in ``mapping`` in one simultaneous pass.

    Parameters
    ----------
    node
        Expression to rewrite.
    mapping
        Node-for-node replacements. Matching is by interned identity,
        applied top-down without revisiting replacement images.
    memo
        Optional shared memo dictionary; pass one dictionary across
        several calls that use the same ``mapping`` to share work.

    Returns
    -------
    Expr
        Rewritten expression (``node`` itself when nothing matched).
    """
    if not mapping:
        return node
    if memo is None:
        memo = {}

    def walk(current: Expr) -> Expr:
        hit = mapping.get(current)
        if hit is not None:
            return hit
        cached = memo.get(current)
        if cached is not None:
            return cached
        children = _children(current)
        if not children:
            memo[current] = current
            return current
        new_children = tuple(walk(child) for child in children)
        if new_children == children:
            result = current
        else:
            result = _rebuild(current, new_children)
        memo[current] = result
        return result

    return walk(node)


def free_atoms(node: Expr) -> frozenset:
    """Return the :class:`Sym` and :class:`Arr` leaves ``node`` reads.

    Results are cached on the node, and caches are shared across the
    DAG, so repeated queries over large assignment lists stay cheap.
    """
    cached = getattr(node, "_free", None)
    if cached is not None:
        return cached
    if isinstance(node, (Sym, Local, Arr)):
        result = frozenset((node,))
    else:
        result = frozenset().union(
            *(free_atoms(child) for child in _children(node))
        )
    node._free = result
    return result


def count_ops(node: Expr, memo: Optional[Dict[Expr, int]] = None) -> int:
    """Return an arithmetic operation count for ``node``.

    Counts one op per binary combination in sums/products, one per
    power, function call, relation, and piecewise branch test. The
    count deliberately follows tree size, matching how
    ``sympy.count_ops`` is used by the auxiliary-cache planner: as a
    relative cost heuristic, not an exact FLOP model.
    """
    if memo is None:
        memo = {}
    cached = memo.get(node)
    if cached is not None:
        return cached
    if isinstance(node, (Num, Sym, Local, Arr, BoolConst)):
        result = 0
    elif isinstance(node, (Add, Mul)):
        result = len(node.args) - 1 + sum(
            count_ops(a, memo) for a in node.args
        )
    elif isinstance(node, Pow):
        result = (
            1
            + count_ops(node.base, memo)
            + count_ops(node.exp, memo)
        )
    elif isinstance(node, Call):
        result = 1 + sum(count_ops(a, memo) for a in node.args)
    elif isinstance(node, Piecewise):
        result = 0
        for value, cond in node.pairs:
            result += 1 + count_ops(value, memo) + count_ops(cond, memo)
    elif isinstance(node, Rel):
        result = (
            1
            + count_ops(node.lhs, memo)
            + count_ops(node.rhs, memo)
        )
    elif isinstance(node, BoolOp):
        result = len(node.args) + sum(
            count_ops(a, memo) for a in node.args
        )
    else:
        raise TypeError(f"unknown node type: {type(node)!r}")
    memo[node] = result
    return result


# Relative device throughput costs; an add or multiply is 1.
DEVICE_WEIGHT_DIVIDE = 4
DEVICE_WEIGHT_SQRT = 8
DEVICE_WEIGHT_TRANSCENDENTAL = 16
DEVICE_CALL_WEIGHTS: Dict[str, int] = {
    name: 1
    for name in (
        "abs",
        "fabs",
        "min",
        "max",
        "fmin",
        "fmax",
        "floor",
        "ceil",
        "trunc",
        "round",
        "copysign",
        "sign",
    )
}
DEVICE_CALL_WEIGHTS["sqrt"] = DEVICE_WEIGHT_SQRT

# Largest integer exponent printed as a multiplication chain.
_POW_CHAIN_LIMIT = 4


def _pow_device_weight(exp: Expr) -> int:
    """Return the device cost of raising a value to ``exp``."""
    if isinstance(exp, Num):
        value = exp.value
        if value == 0.5:
            return DEVICE_WEIGHT_SQRT
        if value == -0.5:
            return DEVICE_WEIGHT_SQRT + DEVICE_WEIGHT_DIVIDE
        if isinstance(value, float) and not value.is_integer():
            return DEVICE_WEIGHT_TRANSCENDENTAL
        if isinstance(value, (int, float)):
            magnitude = abs(int(value))
            if magnitude <= _POW_CHAIN_LIMIT:
                cost = max(magnitude - 1, 0)
                if value < 0:
                    cost += DEVICE_WEIGHT_DIVIDE
                return cost
    return DEVICE_WEIGHT_TRANSCENDENTAL


def count_device_ops(
    node: Expr, memo: Optional[Dict[Expr, int]] = None
) -> int:
    """Return a device-weighted operation count for ``node``.

    Sums and products count 1 per combination, small integer powers
    count as multiplication chains, reciprocals and square roots
    carry mid-range weights, and transcendental calls carry
    :data:`DEVICE_WEIGHT_TRANSCENDENTAL`.
    """
    if memo is None:
        memo = {}
    cached = memo.get(node)
    if cached is not None:
        return cached
    if isinstance(node, (Num, Sym, Local, Arr, BoolConst)):
        result = 0
    elif isinstance(node, (Add, Mul)):
        result = len(node.args) - 1 + sum(
            count_device_ops(a, memo) for a in node.args
        )
    elif isinstance(node, Pow):
        result = (
            _pow_device_weight(node.exp)
            + count_device_ops(node.base, memo)
            + count_device_ops(node.exp, memo)
        )
    elif isinstance(node, Call):
        result = DEVICE_CALL_WEIGHTS.get(
            node.name, DEVICE_WEIGHT_TRANSCENDENTAL
        ) + sum(count_device_ops(a, memo) for a in node.args)
    elif isinstance(node, Piecewise):
        result = 0
        for value, cond in node.pairs:
            result += (
                1
                + count_device_ops(value, memo)
                + count_device_ops(cond, memo)
            )
    elif isinstance(node, Rel):
        result = (
            1
            + count_device_ops(node.lhs, memo)
            + count_device_ops(node.rhs, memo)
        )
    elif isinstance(node, BoolOp):
        result = len(node.args) + sum(
            count_device_ops(a, memo) for a in node.args
        )
    else:
        raise TypeError(f"unknown node type: {type(node)!r}")
    memo[node] = result
    return result


# ----------------------------------------------------------------
# Differentiation
# ----------------------------------------------------------------


class DifferentiationError(NotImplementedError):
    """Raised when no analytic derivative rule exists for a function."""


_SQRT_PI = 1.7724538509055159  # float(sqrt(pi)), used by erf/erfc
_LN2 = 0.6931471805599453
_LN10 = 2.302585092994046


def _d_exp(args, dargs):
    return mul(call("exp", args[0]), dargs[0])


def _d_expm1(args, dargs):
    return mul(call("exp", args[0]), dargs[0])


def _d_log(args, dargs):
    return div(dargs[0], args[0])


def _d_log2(args, dargs):
    return div(dargs[0], mul(num(_LN2), args[0]))


def _d_log10(args, dargs):
    return div(dargs[0], mul(num(_LN10), args[0]))


def _d_log1p(args, dargs):
    return div(dargs[0], add(ONE, args[0]))


def _d_sqrt(args, dargs):
    return div(dargs[0], mul(num(2), call("sqrt", args[0])))


def _d_sin(args, dargs):
    return mul(call("cos", args[0]), dargs[0])


def _d_cos(args, dargs):
    return mul(NEG_ONE, call("sin", args[0]), dargs[0])


def _d_tan(args, dargs):
    return mul(
        add(ONE, pow_(call("tan", args[0]), num(2))), dargs[0]
    )


def _d_asin(args, dargs):
    return div(
        dargs[0],
        call("sqrt", sub(ONE, pow_(args[0], num(2)))),
    )


def _d_acos(args, dargs):
    return neg(
        div(
            dargs[0],
            call("sqrt", sub(ONE, pow_(args[0], num(2)))),
        )
    )


def _d_atan(args, dargs):
    return div(dargs[0], add(ONE, pow_(args[0], num(2))))


def _d_atan2(args, dargs):
    y, x = args
    dy, dx = dargs
    denom = add(pow_(x, num(2)), pow_(y, num(2)))
    return div(sub(mul(x, dy), mul(y, dx)), denom)


def _d_sinh(args, dargs):
    return mul(call("cosh", args[0]), dargs[0])


def _d_cosh(args, dargs):
    return mul(call("sinh", args[0]), dargs[0])


def _d_tanh(args, dargs):
    return mul(
        sub(ONE, pow_(call("tanh", args[0]), num(2))), dargs[0]
    )


def _d_asinh(args, dargs):
    return div(
        dargs[0],
        call("sqrt", add(pow_(args[0], num(2)), ONE)),
    )


def _d_acosh(args, dargs):
    return div(
        dargs[0],
        call("sqrt", sub(pow_(args[0], num(2)), ONE)),
    )


def _d_atanh(args, dargs):
    return div(dargs[0], sub(ONE, pow_(args[0], num(2))))


def _d_erf(args, dargs):
    gauss = call("exp", neg(pow_(args[0], num(2))))
    return mul(num(2.0 / _SQRT_PI), gauss, dargs[0])


def _d_erfc(args, dargs):
    gauss = call("exp", neg(pow_(args[0], num(2))))
    return mul(num(-2.0 / _SQRT_PI), gauss, dargs[0])


def _d_abs(args, dargs):
    return mul(call("sign", args[0]), dargs[0])


def _d_zero(args, dargs):
    return ZERO


def _d_mod(args, dargs):
    # Mod(a, b) = a - b*floor(a/b); floor differentiates to zero
    # almost everywhere.
    a, b = args
    da, db = dargs
    return sub(da, mul(call("floor", div(a, b)), db))


def _d_min(args, dargs):
    return _minmax_derivative(args, dargs, "<=")


def _d_max(args, dargs):
    return _minmax_derivative(args, dargs, ">=")


def _minmax_derivative(args, dargs, op):
    """Piecewise derivative for n-ary Min/Max selections."""
    value, dvalue = args[0], dargs[0]
    for other, dother in zip(args[1:], dargs[1:]):
        cond = rel(op, value, other)
        dvalue = piecewise((dvalue, cond), (dother, TRUE))
        value = piecewise((value, cond), (other, TRUE))
    return dvalue


_DERIVATIVES: Dict[str, Callable] = {
    "exp": _d_exp,
    "expm1": _d_expm1,
    "log": _d_log,
    "log2": _d_log2,
    "log10": _d_log10,
    "log1p": _d_log1p,
    "sqrt": _d_sqrt,
    "sin": _d_sin,
    "cos": _d_cos,
    "tan": _d_tan,
    "asin": _d_asin,
    "acos": _d_acos,
    "atan": _d_atan,
    "atan2": _d_atan2,
    "sinh": _d_sinh,
    "cosh": _d_cosh,
    "tanh": _d_tanh,
    "asinh": _d_asinh,
    "acosh": _d_acosh,
    "atanh": _d_atanh,
    "erf": _d_erf,
    "erfc": _d_erfc,
    "Abs": _d_abs,
    "sign": _d_zero,
    "floor": _d_zero,
    "ceiling": _d_zero,
    "Min": _d_min,
    "Max": _d_max,
    "Mod": _d_mod,
}

_NO_DERIVATIVE = frozenset(
    ("gamma", "loggamma", "hypot", "fmod", "remainder", "copysign")
)


def diff(
    node: Expr,
    var: Union[Sym, Arr],
    memo: Optional[Dict[Expr, Expr]] = None,
    derivative_names: Optional[Dict[str, str]] = None,
) -> Expr:
    """Differentiate ``node`` with respect to ``var``.

    Parameters
    ----------
    node
        Expression to differentiate.
    var
        Scalar symbol or array reference treated as the variable; all
        other leaves are constants.
    memo
        Optional memo dictionary keyed by node; pass one dictionary
        across calls with the same ``var`` to share work over a DAG.
    derivative_names
        Mapping from user-function name to its derivative-placeholder
        print name. Functions not in the mapping (and not in the
        analytic rule table) differentiate to
        ``d_<name>(args..., arg_index)``.

    Returns
    -------
    Expr
        Analytic derivative, folded through the interning
        constructors.

    Raises
    ------
    DifferentiationError
        When a function has no analytic derivative rule and is a
        known math function (rather than a user placeholder).
    """
    if memo is None:
        memo = {}
    names = derivative_names or {}

    def walk(current: Expr) -> Expr:
        if current is var:
            return ONE
        if isinstance(current, (Num, Sym, Local, Arr, BoolConst)):
            return ZERO
        cached = memo.get(current)
        if cached is not None:
            return cached
        if isinstance(current, Add):
            result = add(*(walk(a) for a in current.args))
        elif isinstance(current, Mul):
            terms = []
            args = current.args
            for i, factor in enumerate(args):
                dfactor = walk(factor)
                if dfactor is ZERO:
                    continue
                rest = args[:i] + args[i + 1:]
                terms.append(mul(dfactor, *rest))
            result = add(*terms) if terms else ZERO
        elif isinstance(current, Pow):
            base, exponent = current.base, current.exp
            dbase = walk(base)
            dexp = walk(exponent)
            terms = []
            if dbase is not ZERO:
                terms.append(
                    mul(
                        exponent,
                        pow_(base, sub(exponent, ONE)),
                        dbase,
                    )
                )
            if dexp is not ZERO:
                terms.append(
                    mul(current, call("log", base), dexp)
                )
            result = add(*terms) if terms else ZERO
        elif isinstance(current, Call):
            dargs = [walk(a) for a in current.args]
            if all(d is ZERO for d in dargs):
                result = ZERO
            else:
                rule = _DERIVATIVES.get(current.name)
                if rule is not None:
                    result = rule(current.args, dargs)
                elif current.name in _NO_DERIVATIVE:
                    raise DifferentiationError(
                        f"no analytic derivative rule for "
                        f"'{current.name}'"
                    )
                else:
                    # User function: chain rule through derivative
                    # placeholders d_<name>(args..., arg_index).
                    target = names.get(
                        current.name, f"d_{current.name.rstrip('_')}"
                    )
                    terms = []
                    for i, dArg in enumerate(dargs):
                        if dArg is ZERO:
                            continue
                        partial = call(
                            target, *current.args, num(i)
                        )
                        terms.append(mul(partial, dArg))
                    result = add(*terms) if terms else ZERO
        elif isinstance(current, Piecewise):
            result = piecewise(
                *((walk(v), c) for v, c in current.pairs)
            )
        elif isinstance(current, (Rel, BoolOp)):
            raise DifferentiationError(
                "cannot differentiate a boolean expression"
            )
        else:
            raise TypeError(f"unknown node type: {type(current)!r}")
        memo[current] = result
        return result

    return walk(node)
