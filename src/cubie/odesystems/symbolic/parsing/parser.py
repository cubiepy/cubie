"""Parse ODE and DAE definitions into engine IR equations.

String, SymPy, and callable inputs produce :class:`ParsedEquations`.
DAEs pass through structural simplification before assembly.
"""

import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import sympy as sp
from sympy.parsing.sympy_parser import T
from sympy.core.function import AppliedUndef
import attrs

from ..engine import expr as ir_expr
from ..engine.from_sympy import (
    convert_assignments,
    derivative_name_map,
)
from ..indexedbasemaps import IndexedBases
from ..sym_utils import hash_system_definition
from cubie._utils import is_devfunc

# Lambda notation, Auto-number, factorial notation, implicit multiplication
PARSE_TRANSFORMS = (T[0][0], T[3][0], T[4][0], T[8][0])

_INDEXED_NAME_PATTERN = re.compile(r"(?P<name>[A-Za-z_]\w*)\[(?P<index>\d+)\]")

TIME_SYMBOL = sp.Symbol("t", real=True)
DRIVER_SETTING_KEYS = {"time", "driver_sample_period", "wrap", "order"}


def _detect_input_type(dxdt: Union[str, Iterable, Callable]) -> str:
    """Detect whether dxdt contains strings, SymPy expressions, or a callable.

    Determines input format by inspecting the type of dxdt itself and,
    for iterables, examining the first element to categorize as either
    string-based or SymPy-based input.

    Parameters
    ----------
    dxdt
        System equations as string, iterable, or callable.

    Returns
    -------
    str
        Either 'string', 'sympy', or 'function' indicating input format.

    Raises
    ------
    TypeError
        If input type cannot be determined or is invalid.
    ValueError
        If empty iterable is provided.
    """
    if dxdt is None:
        raise TypeError("dxdt cannot be None")

    if callable(dxdt) and not isinstance(dxdt, (str, list, tuple)):
        return "function"

    if isinstance(dxdt, str):
        return "string"

    if isinstance(dxdt, (sp.Equality, sp.Expr)):
        return "sympy"

    try:
        items = list(dxdt)
    except TypeError:
        raise TypeError(
            f"dxdt must be string or iterable, got {type(dxdt).__name__}"
        )

    if len(items) == 0:
        raise ValueError("dxdt iterable cannot be empty")

    first_elem = items[0]

    if isinstance(first_elem, str):
        return "string"
    elif isinstance(first_elem, (sp.Expr, sp.Equality)):
        return "sympy"
    elif isinstance(first_elem, tuple) and len(first_elem) == 2:
        # A (lhs, rhs) pair; both members must be engine IR nodes or
        # convertible to SymPy expressions, or the pair is rejected
        # here rather than failing deep inside the normaliser.
        for side, member in zip(("lhs", "rhs"), first_elem):
            if not isinstance(
                member,
                (ir_expr.Expr, sp.Basic, str, int, float, complex),
            ):
                raise TypeError(
                    f"dxdt element 0 is a (lhs, rhs) tuple whose "
                    f"{side} is {member!r} "
                    f"({type(member).__name__}); each member must "
                    f"be an IR or SymPy expression, string, or "
                    f"number."
                )
        return "sympy"

    raise TypeError(
        f"dxdt elements must be strings or symbolic expressions, "
        f"got {type(first_elem).__name__}. "
        f"Valid symbolic formats: sp.Equality or a (lhs, rhs) tuple."
    )


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


def _process_parameters(
    states: Union[Dict[str, float], Iterable[str]],
    parameters: Union[Dict[str, float], Iterable[str]],
    constants: Union[Dict[str, float], Iterable[str]],
    observables: Iterable[str],
    drivers: Iterable[str],
    state_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    parameter_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    constant_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    observable_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    driver_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
) -> IndexedBases:
    """Convert user-specified symbols into ``IndexedBases`` structures.

    Parameters
    ----------
    states
        State symbols or mapping to initial values.
    parameters
        Parameter symbols or mapping to default values.
    constants
        Constant symbols or mapping to default values.
    observables
        Observable symbol names supplied by the user.
    drivers
        External driver symbol names.
    state_units
        Optional units for states. Defaults to "dimensionless".
    parameter_units
        Optional units for parameters. Defaults to "dimensionless".
    constant_units
        Optional units for constants. Defaults to "dimensionless".
    observable_units
        Optional units for observables. Defaults to "dimensionless".
    driver_units
        Optional units for drivers. Defaults to "dimensionless".

    Returns
    -------
    IndexedBases
        Structured representation of all indexed symbol collections.
    """
    indexed_bases = IndexedBases.from_user_inputs(
        states,
        parameters,
        constants,
        observables,
        drivers,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    return indexed_bases


def parse_input(
    dxdt: Union[str, Iterable, Callable],
    states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], Dict[str, Any]]] = None,
    user_functions: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = False,
    state_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    parameter_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    constant_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    observable_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    driver_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    simplify: bool = False,
    state_priority: Optional[Dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    simplify_options: Optional[Dict[str, Any]] = None,
):
    """Process user equations and symbol metadata into structured components.

    Parameters
    ----------
    dxdt
        System equations as a newline-delimited string, an iterable
        of strings, SymPy equations, or a callable. In addition to
        explicit forms, implicit equations (``0 = g(...)``),
        higher-order/nested derivatives, derivative terms inside
        expressions, and algebraic unknowns are accepted (symbolic
        input only); such systems route through structural
        simplification automatically.
    states
        All unknowns of the system (differential or algebraic) as
        names or a mapping to initial values.
    observables
        Observable variable names whose trajectories should be saved.
    parameters
        Parameter names or mapping to default values.
    constants
        Constant names or mapping to default values that remain fixed across
        runs.
    drivers
        Driver variable names supplied at runtime. Accepts either an iterable
        of driver labels or a dictionary mapping driver labels to default
        values and, when using driver arrays, configuration entries such as
        ``time``, ``dt``, ``wrap``, and ``order``.
    user_functions
        Mapping of callable names used in equations to their implementations.
    user_function_derivatives
        Mapping of callable names to derivative helper functions.
    strict
        When ``False``, infer missing symbol declarations from equation usage.
    state_units
        Optional units for states. Defaults to "dimensionless".
    parameter_units
        Optional units for parameters. Defaults to "dimensionless".
    constant_units
        Optional units for constants. Defaults to "dimensionless".
    observable_units
        Optional units for observables. Defaults to "dimensionless".
    driver_units
        Optional units for drivers. Defaults to "dimensionless".
    simplify
        Force MTK-style structural simplification (alias
        elimination, index reduction, tearing) even for systems that
        are already in explicit form. DAE-shaped input enables it
        automatically.
    state_priority
        Per-unknown state-selection priorities (higher values are
        preferred as solver states). Structural path only.
    irreducible
        Unknowns that must not be eliminated. Structural path only.
    simplify_options
        Extra keyword arguments forwarded to
        :func:`~cubie.odesystems.symbolic.structural.simplify.structural_simplify`.

    Returns
    -------
    tuple
        ``(index_map, all_symbols, funcs, parsed_equations, fn_hash,
        simplified)``. ``simplified`` is the
        :class:`~cubie.odesystems.symbolic.structural.simplify.SimplifiedSystem`
        when structural simplification ran (it carries the mass
        matrix for torn systems) and ``None`` otherwise.

    Notes
    -----
    When ``strict`` is ``False``, undeclared variables inferred from equation
    usage are added automatically, except for anonymous auxiliaries that are
    retained for intermediate computation but not persisted as observables.
    """
    from .assemble import assemble_explicit, assemble_simplified
    from .normalise import classify_system, normalise_input

    input_type = _detect_input_type(dxdt)

    if input_type == "function":
        if simplify:
            raise TypeError(
                "Callable dxdt input is explicit-ODE only and cannot be "
                "combined with simplify=True."
            )
        return _parse_function_path(
            dxdt,
            states=states,
            observables=observables,
            parameters=parameters,
            constants=constants,
            drivers=drivers,
            user_functions=user_functions,
            user_function_derivatives=user_function_derivatives,
            strict=strict,
            state_units=state_units,
            parameter_units=parameter_units,
            constant_units=constant_units,
            observable_units=observable_units,
            driver_units=driver_units,
        )

    if states is None and strict:
        raise ValueError(
            "No state symbols were provided - if you want to build a model "
            "from a set of equations alone, set strict=False"
        )

    states_dict = dict(states) if isinstance(states, dict) else {
        str(name): 0.0 for name in (states or [])
    }
    observables = list(observables or [])
    parameters = parameters if parameters is not None else {}
    constants = constants if constants is not None else {}

    driver_dict = None
    if drivers is None:
        driver_names = []
    elif isinstance(drivers, dict):
        driver_dict = drivers
        driver_names = [
            key for key in drivers.keys() if key not in DRIVER_SETTING_KEYS
        ]
        if not driver_names:
            raise ValueError(
                "Driver dictionary must include at least one driver symbol."
            )
    else:
        driver_names = list(drivers)

    known_symbol_map = {}
    for name in list(parameters) + list(constants) + driver_names:
        known_symbol_map[str(name)] = sp.Symbol(str(name), real=True)

    unknown_names = {str(name) for name in states_dict}
    unknown_names |= {str(name) for name in observables}

    normalised = normalise_input(
        dxdt,
        unknown_names,
        known_symbol_map,
        user_functions,
        user_function_derivatives,
        strict,
        set(states_dict),
    )
    for name in normalised.inferred_states:
        states_dict[name] = 0.0

    shape = classify_system(
        normalised, states_dict.keys(), observables
    )
    use_structural = simplify or shape == "dae"
    if use_structural and not simplify:
        warn(
            "DAE constructs detected (implicit equations, higher-order "
            "or in-expression derivatives, or unknowns without "
            "derivative equations); structural simplification enabled.",
            EquationWarning,
        )

    if not use_structural:
        return assemble_explicit(
            normalised,
            states_dict,
            observables,
            parameters,
            constants,
            driver_names,
            driver_dict,
            user_functions,
            user_function_derivatives,
            state_units=state_units,
            parameter_units=parameter_units,
            constant_units=constant_units,
            observable_units=observable_units,
            driver_units=driver_units,
        )

    if isinstance(parameters, dict):
        parameters_dict = dict(parameters)
    else:
        parameters_dict = {str(name): 0.0 for name in parameters}
    if isinstance(constants, dict):
        constants_dict = dict(constants)
    else:
        constants_dict = {str(name): 0.0 for name in constants}

    return assemble_simplified(
        normalised,
        states_dict,
        observables,
        parameters_dict,
        constants_dict,
        driver_names,
        driver_dict,
        known_symbol_map,
        user_functions,
        user_function_derivatives,
        state_priority=state_priority,
        irreducible=irreducible,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
        simplify_options=simplify_options,
    )


def _parse_function_path(
    dxdt: Callable,
    states,
    observables,
    parameters,
    constants,
    drivers,
    user_functions,
    user_function_derivatives,
    strict,
    state_units,
    parameter_units,
    constant_units,
    observable_units,
    driver_units,
):
    """Parse callable ``dxdt`` input (explicit-ODE only)."""

    from .function_parser import (
        infer_function_states,
        parse_function_input,
    )

    if states is None:
        if strict:
            raise ValueError(
                "No state symbols were provided - if you want to build a "
                "model from a set of equations alone, set strict=False"
            )
        states = infer_function_states(dxdt)
    if observables is None:
        observables = []
    if parameters is None:
        parameters = {}
    if constants is None:
        constants = {}
    driver_dict = None
    if drivers is None:
        drivers = []
    elif isinstance(drivers, dict):
        driver_dict = drivers
        drivers = [
            key for key in drivers.keys() if key not in DRIVER_SETTING_KEYS
        ]
        if len(drivers) == 0:
            raise ValueError(
                "Driver dictionary must include at least one driver symbol."
            )

    index_map = _process_parameters(
        states=states,
        parameters=parameters,
        constants=constants,
        observables=observables,
        drivers=drivers,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )

    equation_map, funcs, new_params = parse_function_input(
        func=dxdt,
        index_map=index_map,
        observables=list(observables),
        user_functions=user_functions,
        user_function_derivatives=user_function_derivatives,
        strict=strict,
        declared_states=list(states),
    )
    # Derivative placeholder names must be recovered from the SymPy
    # function objects before the equations convert to IR.
    function_derivative_names = derivative_name_map(equation_map)
    equation_map = convert_assignments(
        equation_map,
        allowed_functions=user_functions,
    )
    all_symbols = index_map.all_symbols.copy()
    all_symbols.setdefault("t", TIME_SYMBOL)

    for param in new_params:
        index_map.parameters.push(param)
        all_symbols[str(param)] = param

    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)

    if user_functions:
        all_symbols.update({name: fn for name, fn in user_functions.items()})
        if user_function_derivatives:
            all_symbols.update(
                {
                    fn.__name__: fn
                    for fn in user_function_derivatives.values()
                    if callable(fn)
                }
            )

    parsed_equations = ParsedEquations.from_equations(
        equation_map,
        index_map,
        derivative_names=function_derivative_names,
        function_aliases={name: name for name in funcs},
    )

    fn_hash = hash_system_definition(
        parsed_equations,
        index_map.constants.default_values,
        state_labels=index_map.state_names,
        dxdt_labels=index_map.dxdt_names,
        parameter_labels=index_map.parameter_names,
        driver_labels=index_map.driver_names,
        observable_labels=index_map.observables.ref_map.keys(),
        derivative_names=parsed_equations.derivative_names,
        function_aliases=parsed_equations.function_aliases,
    )

    return index_map, all_symbols, funcs, parsed_equations, fn_hash, None
