"""Emit CUDA factory code for direct sparse LU linear solves."""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from cubie.odesystems.solver_helpers import HelperVariant
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.adapter import SystemIR, system_ir
from cubie.odesystems.symbolic.engine.assignments import (
    cse_and_stack,
    prune_unused,
    topological_sort,
)
from cubie.odesystems.symbolic.engine.printer import (
    print_cuda_multiple,
)
from cubie.odesystems.symbolic.codegen.jacobian import (
    generate_jacobian,
)
from cubie.odesystems.symbolic.codegen.linear_operators import (
    _resolve_jvp,
    _state_increment_subs,
)
from cubie.odesystems.symbolic.codegen.preconditioners import (
    DIAG_DIVISION_FLOOR,
)
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.codegen._matrix_utils import (
    mass_diagonal_flags,
)
from cubie._env import operation_ordering_default
from cubie.time_logger import default_timelogger

for _variant in HelperVariant:
    default_timelogger.register_event(
        f"codegen_lu_solve_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} direct LU solve",
    )

LU_SCALAR_NNZ_LIMIT = 150
"""Largest ``nnz(L+U)`` emitted as named scalars.

At or below this, ``lu_nnz`` reports 0 and the ``factor`` argument
is unused; above it, entries index the ``factor`` array.
"""

LU_MAX_NNZ = 8192
"""Ceiling on ``nnz(L+U)`` above which generation refuses."""

LU_MAX_FACTOR_FLOPS = 131072
"""Ceiling on predicted factorisation flops above which generation
refuses."""


LU_SOLVE_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DIRECT LU SOLVE FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0,"
    " lineinfo=None):\n"
    '    """Auto-generated direct sparse LU solve.\n'
    "    Solves (beta * M - gamma * a_ij * h * J) @ x = rhs with a\n"
    "    static symbolic factorisation; J is evaluated at\n"
    "    {eval_point}.\n"
    "    Returns device function:\n"
    "      lu_solve(state, parameters, drivers, {cached_arg}base_state,\n"
    "               t, h, a_ij, rhs, x, factor) -> int32\n"
    "    The return value counts magnitude-floored pivots; zero\n"
    "    marks a clean factorisation. rhs is read-only; x is\n"
    "    written unconditionally.\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def lu_solve(\n"
    "        state, parameters, drivers, {cached_arg}base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, rhs, x, factor\n"
    "    ):\n"
    "{body}\n"
    "        return int32(_cubie_codegen_lu_singular)\n"
    "    return lu_solve\n"
    "# Store lu_nnz for retrieval when loading from file cache\n"
    "{func_name}.lu_nnz = {lu_nnz}\n"
)


def _markowitz_symbolic_lu(
    pattern: Set[Tuple[int, int]],
    n: int,
) -> Tuple[List[int], Set[Tuple[int, int]], int]:
    """Order and symbolically factorise a structural pattern.

    The symmetric permutation follows the Markowitz rule: the
    remaining diagonal minimising ``(row_deg - 1) * (col_deg - 1)``
    pivots next, ties broken by index.

    Parameters
    ----------
    pattern
        Structural nonzeros of the matrix as ``(row, col)`` pairs,
        including every diagonal entry.
    n
        Matrix width.

    Returns
    -------
    tuple
        Elimination order (original index of pivot ``k``), the
        ``L + U`` pattern in permuted coordinates, and the
        predicted factorisation flop count.
    """
    rows: Dict[int, Set[int]] = {i: set() for i in range(n)}
    cols: Dict[int, Set[int]] = {j: set() for j in range(n)}
    for i, j in pattern:
        rows[i].add(j)
        cols[j].add(i)

    active = set(range(n))
    perm: List[int] = []
    filled = set(pattern)
    flops = 0
    for _ in range(n):
        best = min(
            active,
            key=lambda r: (
                (len(rows[r] & active) - 1)
                * (len(cols[r] & active) - 1),
                r,
            ),
        )
        perm.append(best)
        active.discard(best)
        row_targets = sorted(j for j in rows[best] if j in active)
        col_sources = sorted(i for i in cols[best] if i in active)
        flops += len(col_sources)
        flops += 2 * len(col_sources) * len(row_targets)
        for i in col_sources:
            for j in row_targets:
                if (i, j) not in filled:
                    filled.add((i, j))
                    rows[i].add(j)
                    cols[j].add(i)

    position = {orig: k for k, orig in enumerate(perm)}
    lu_pattern = {
        (position[i], position[j]) for (i, j) in filled
    }
    return perm, lu_pattern, flops


def _lu_body_from_entries(
    sysir: SystemIR,
    entry_exprs: Dict[Tuple[int, int], ir.Expr],
    mass_diag: Tuple[bool, ...],
    prefix_assigns: List[Tuple[ir.Expr, ir.Expr]],
    operation_ordering: str,
    cse: bool = True,
) -> Tuple[str, int]:
    """Build the solve body from per-entry Jacobian expressions.

    ``entry_exprs`` holds the structurally nonzero Jacobian entries
    in original coordinates, already substituted to their evaluation
    point; ``prefix_assigns`` supplies the auxiliary assignments the
    entries reference.

    Returns
    -------
    tuple of str and int
        The printed body and the ``factor`` buffer length (zero when
        the factor is emitted as named scalars).
    """
    n = len(sysir.state_symbols)
    pattern = set(entry_exprs) | {(i, i) for i in range(n)}
    perm, lu_pattern, factor_flops = _markowitz_symbolic_lu(
        pattern, n
    )
    nnz = len(lu_pattern)
    if nnz > LU_MAX_NNZ or factor_flops > LU_MAX_FACTOR_FLOPS:
        raise ValueError(
            "Direct LU generation refused: predicted factor size "
            f"nnz(L+U)={nnz} (limit {LU_MAX_NNZ}) at "
            f"{factor_flops} flops (limit {LU_MAX_FACTOR_FLOPS}). "
            "Use an iterative linear_correction_type for this "
            "system."
        )
    use_factor_array = nnz > LU_SCALAR_NNZ_LIMIT
    slots = {
        entry: idx for idx, entry in enumerate(sorted(lu_pattern))
    }

    def factor_ref(a: int, b: int) -> ir.Expr:
        if use_factor_array:
            return ir.arr("factor", slots[(a, b)])
        return ir.sym(f"_cubie_codegen_lu_f_{a}_{b}")

    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    h_sym = ir.sym("_cubie_codegen_h")
    floor_num = ir.num(DIAG_DIVISION_FLOOR)

    # Block A: shifted-matrix entries; CSE runs over this block only.
    w_syms: Dict[Tuple[int, int], ir.Expr] = {}
    w_assigns: List[Tuple[ir.Expr, ir.Expr]] = []
    position = {orig: k for k, orig in enumerate(perm)}
    for (i, j) in sorted(pattern):
        a, b = position[i], position[j]
        mass_term = (
            beta_sym if (i == j and mass_diag[i]) else ir.ZERO
        )
        jac_term = entry_exprs.get((i, j), ir.ZERO)
        value = ir.sub(
            mass_term,
            ir.mul(gamma_sym, a_ij_sym, h_sym, jac_term),
        )
        w_sym = ir.sym(f"_cubie_codegen_lu_w_{a}_{b}")
        w_syms[(a, b)] = w_sym
        w_assigns.append((w_sym, value))

    block_a = list(prefix_assigns) + w_assigns
    if cse:
        block_a = cse_and_stack(
            block_a, operation_ordering=operation_ordering
        )
    else:
        block_a = topological_sort(
            block_a, operation_ordering=operation_ordering
        )

    # Block B: elimination, substitution, and the pivot guard.
    # Left-looking form: one assignment per L/U entry.
    block_b: List[Tuple[ir.Expr, ir.Expr]] = []
    for (a, b) in sorted(lu_pattern):
        base = w_syms.get((a, b), ir.ZERO)
        terms = [base]
        for m in range(min(a, b)):
            if (a, m) in lu_pattern and (m, b) in lu_pattern:
                terms.append(
                    ir.mul(
                        ir.num(-1),
                        factor_ref(a, m),
                        factor_ref(m, b),
                    )
                )
        total = ir.add(*terms)
        if a > b:
            block_b.append(
                (factor_ref(a, b), ir.div(total, factor_ref(b, b)))
            )
        elif a == b:
            raw = ir.sym(f"_cubie_codegen_lu_d_{a}")
            block_b.append((raw, total))
            guarded = ir.piecewise(
                (
                    raw,
                    ir.rel(">=", ir.call("Abs", raw), floor_num),
                ),
                (floor_num, ir.TRUE),
            )
            block_b.append((factor_ref(a, a), guarded))
        else:
            block_b.append((factor_ref(a, b), total))

    # Forward substitution on the permuted rhs; L has a unit diagonal.
    y_syms = [
        ir.sym(f"_cubie_codegen_lu_y_{a}") for a in range(n)
    ]
    for a in range(n):
        terms = [ir.arr("rhs", perm[a])]
        for b in range(a):
            if (a, b) in lu_pattern:
                terms.append(
                    ir.mul(ir.num(-1), factor_ref(a, b), y_syms[b])
                )
        block_b.append((y_syms[a], ir.add(*terms)))

    # Back substitution, then scatter through the permutation.
    x_syms = [
        ir.sym(f"_cubie_codegen_lu_xs_{a}") for a in range(n)
    ]
    for a in range(n - 1, -1, -1):
        terms = [y_syms[a]]
        for b in range(a + 1, n):
            if (a, b) in lu_pattern:
                terms.append(
                    ir.mul(ir.num(-1), factor_ref(a, b), x_syms[b])
                )
        block_b.append(
            (x_syms[a], ir.div(ir.add(*terms), factor_ref(a, a)))
        )
    for a in range(n):
        block_b.append((ir.arr("x", perm[a]), x_syms[a]))

    # Pivot guard flag: counts magnitude-floored pivots.
    indicators = [
        ir.piecewise(
            (
                ir.ONE,
                ir.rel(
                    "<",
                    ir.call(
                        "Abs", ir.sym(f"_cubie_codegen_lu_d_{a}")
                    ),
                    floor_num,
                ),
            ),
            (ir.ZERO, ir.TRUE),
        )
        for a in range(n)
    ]
    singular_sym = ir.sym("_cubie_codegen_lu_singular")
    block_b.append((singular_sym, ir.add(*indicators)))

    block_b = topological_sort(
        block_b, operation_ordering=operation_ordering
    )

    outputs = [ir.arr("x", i) for i in range(n)]
    outputs.append(singular_sym)
    exprs = prune_unused(
        block_a + block_b, output_symbols=outputs
    )

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    body = "\n".join("        " + ln for ln in lines)
    return body, (nnz if use_factor_array else 0)


def _inline_entry_exprs(
    sysir: SystemIR,
    jac: List[List[ir.Expr]],
    state_is_increment: bool,
) -> Tuple[
    Dict[Tuple[int, int], ir.Expr],
    List[Tuple[ir.Expr, ir.Expr]],
]:
    """Return substituted Jacobian entries plus their auxiliaries.

    Renames dx/observable outputs to codegen locals and, when
    ``state_is_increment``, evaluates every state symbol at
    ``base_state + a_ij * state``.
    """
    n = len(sysir.state_symbols)
    subs_map: Dict[ir.Expr, ir.Expr] = {}
    for idx, dx_sym in enumerate(sysir.dxdt_symbols):
        subs_map[dx_sym] = ir.sym(f"_cubie_codegen_dx_{idx}")
    for idx, obs_sym in enumerate(sysir.observable_symbols):
        subs_map[obs_sym] = ir.sym(
            f"_cubie_codegen_aux_{idx + 1}"
        )
    if state_is_increment:
        subs_map.update(_state_increment_subs(sysir))

    memo: dict = {}
    prefix_assigns = [
        (
            ir.xreplace(lhs, subs_map, memo),
            ir.xreplace(rhs, subs_map, memo),
        )
        for lhs, rhs in sysir.equations
    ]

    entry_exprs: Dict[Tuple[int, int], ir.Expr] = {}
    for i in range(n):
        for j in range(n):
            entry = jac[i][j]
            if ir.is_zero(entry):
                continue
            entry_exprs[(i, j)] = ir.xreplace(
                entry, subs_map, memo
            )
    return entry_exprs, prefix_assigns


def _cached_entry_exprs(
    sysir: SystemIR,
    jac: List[List[ir.Expr]],
    jvp_equations: JVPEquations,
) -> Tuple[
    Dict[Tuple[int, int], ir.Expr],
    List[Tuple[ir.Expr, ir.Expr]],
]:
    """Return Jacobian entries bound to the auxiliary cache.

    Cached and runtime auxiliaries stay by reference to the prefix
    assignments; prepare-only nodes inline their JVP-graph
    expressions; auxiliary names absent from the JVP graph inline
    their original right-hand sides.
    """
    n = len(sysir.state_symbols)
    cached_aux, runtime_aux, _ = jvp_equations.cached_partition()

    prefix_assigns: List[Tuple[ir.Expr, ir.Expr]] = []
    prefix_assigns.extend(
        (lhs, ir.arr("cached_aux", idx))
        for idx, (lhs, _) in enumerate(cached_aux)
    )
    prefix_assigns.extend(runtime_aux)

    # Substitute every symbol not assigned in the prefix.
    assigned = {lhs for lhs, _ in prefix_assigns}
    subs_memo: dict = {}
    aux_subs: Dict[ir.Expr, ir.Expr] = {}
    for lhs in jvp_equations.non_jvp_order:
        if lhs in assigned:
            continue
        aux_subs[lhs] = ir.xreplace(
            jvp_equations.non_jvp_exprs[lhs],
            aux_subs,
            subs_memo,
        )

    # Map original observable names to the JVP pipeline's aux_<n>.
    obs_renames = {
        obs_sym: ir.sym(f"_cubie_codegen_aux_{idx + 1}")
        for idx, obs_sym in enumerate(sysir.observable_symbols)
    }

    # Names absent from the JVP graph inline their original rhs,
    # dependencies first.
    memo: dict = {}
    dx_symbols = set(sysir.dxdt_symbols)
    original_aux = [
        (
            ir.xreplace(lhs, obs_renames, memo),
            ir.xreplace(rhs, obs_renames, memo),
        )
        for lhs, rhs in sysir.equations
        if lhs not in dx_symbols
    ]
    for lhs, rhs in original_aux:
        if lhs in assigned or lhs in aux_subs:
            continue
        aux_subs[lhs] = ir.xreplace(rhs, aux_subs, subs_memo)

    entry_exprs: Dict[Tuple[int, int], ir.Expr] = {}
    for i in range(n):
        for j in range(n):
            entry = jac[i][j]
            if ir.is_zero(entry):
                continue
            entry = ir.xreplace(entry, obs_renames, memo)
            entry_exprs[(i, j)] = ir.xreplace(
                entry, aux_subs, subs_memo
            )
    return entry_exprs, prefix_assigns


def generate_lu_solve_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PLAIN,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "lu_solve_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = operation_ordering_default(),
) -> Tuple[str, int]:
    """Generate the direct LU solve factory for one variant.

    ``PLAIN`` evaluates ``J`` at ``base_state + a_ij * state``;
    ``AT_STATE`` at ``state`` with ``a_ij`` scaling the matrix only;
    ``CACHED`` at ``state`` with auxiliaries from ``cached_aux``.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    variant
        Helper variant selecting the evaluation-point and auxiliary
        conventions.
    M
        0/1 diagonal mass matrix; identity when omitted.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination to the
        matrix-entry block.
    jvp_equations
        Prebuilt JVP equations for the cached variant; generated
        when absent.

    Returns
    -------
    tuple of str and int
        Generated factory source and the ``factor`` buffer length
        (zero when the factor is emitted as named scalars).
    """
    event = f"codegen_lu_solve_{variant.value}"
    default_timelogger.start_event(event)

    sysir = system_ir(equations, index_map)
    n = len(sysir.state_symbols)
    mass_diag = mass_diagonal_flags(M, n)
    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        operation_ordering=operation_ordering,
    )
    if variant.cached:
        jvp_equations = _resolve_jvp(
            equations,
            index_map,
            cse,
            jvp_equations,
            operation_ordering,
        )
        entry_exprs, prefix_assigns = _cached_entry_exprs(
            sysir, jac, jvp_equations
        )
        eval_point = "state, auxiliaries from cached_aux"
    else:
        entry_exprs, prefix_assigns = _inline_entry_exprs(
            sysir, jac, variant is HelperVariant.PLAIN
        )
        eval_point = (
            "base_state + a_ij * state"
            if variant is HelperVariant.PLAIN
            else "state"
        )
    body, lu_nnz = _lu_body_from_entries(
        sysir=sysir,
        entry_exprs=entry_exprs,
        mass_diag=mass_diag,
        prefix_assigns=prefix_assigns,
        operation_ordering=operation_ordering,
        cse=cse,
    )
    code = LU_SOLVE_TEMPLATE.format(
        func_name=func_name,
        cached_arg="cached_aux, " if variant.cached else "",
        body=body,
        lu_nnz=lu_nnz,
        eval_point=eval_point,
    )
    default_timelogger.stop_event(event)
    return code, lu_nnz


__all__ = [
    "LU_SCALAR_NNZ_LIMIT",
    "LU_MAX_NNZ",
    "LU_MAX_FACTOR_FLOPS",
    "generate_lu_solve_code",
]
