"""Emit CUDA factory code for direct sparse LU linear solves.

Published Functions
-------------------
:func:`generate_lu_solve_code`
    Emit the direct solve factory for one helper variant.

:func:`generate_lu_prepare_blocks_code`
    Emit the step-start block factorisation filling ``cached_aux``.

:func:`generate_lu_smoothing_solve_code`
    Emit the smoothed-error solve on the real eigenvalue block.
"""

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import sympy as sp

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
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    build_stage_substitutions,
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
    block_eigenstructure,
    mass_diagonal_flags,
)
from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)
from cubie._env import operation_ordering_default
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.time_logger import default_timelogger

_SINGULAR_PIVOT = int(CUBIE_RESULT_CODES.SINGULAR_PIVOT)

for _variant in HelperVariant:
    default_timelogger.register_event(
        f"codegen_lu_solve_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} direct LU solve",
    )
    default_timelogger.register_event(
        f"codegen_lu_prepare_blocks_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} LU block preparation",
    )
default_timelogger.register_event(
    "codegen_lu_smoothing_solve",
    "codegen",
    "Codegen time for the LU smoothing solve",
)

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
    "      lu_solve(state, parameters, drivers, cached_aux, base_state,\n"
    "               t, h, a_ij, rhs, x, factor) -> int32\n"
    "    Returns int32({singular_code}) (SINGULAR_PIVOT) when any\n"
    "    pivot was magnitude-floored, else int32(0). rhs is\n"
    "    read-only; x is written unconditionally.\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def lu_solve(\n"
    "        state, parameters, drivers, cached_aux, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, rhs, x, factor\n"
    "    ):\n"
    "{body}\n"
    "        return selp(\n"
    "            _cubie_codegen_lu_singular != int32(0),\n"
    "            int32({singular_code}),\n"
    "            int32(0),\n"
    "        )\n"
    "    return lu_solve\n"
    "# Store lu_nnz for retrieval when loading from file cache\n"
    "{func_name}.lu_nnz = {lu_nnz}\n"
)


LU_SUBSTITUTE_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED PREFACTORED LU SUBSTITUTION FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0,"
    " lineinfo=None):\n"
    '    """Auto-generated prefactored LU substitution.\n'
    "    {description}\n"
    "    Factors are read from cached_aux, filled by the companion\n"
    "    lu_prepare_blocks helper at step start.\n"
    "    Returns device function:\n"
    "      lu_solve(state, parameters, drivers, cached_aux,\n"
    "               base_state, t, h, a_ij, rhs, x, factor) -> int32\n"
    "    rhs is read-only; x is written unconditionally.\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "{preamble}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def lu_solve(\n"
    "        state, parameters, drivers, cached_aux, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, rhs, x, factor\n"
    "    ):\n"
    "{body}\n"
    "        return int32(0)\n"
    "    return lu_solve\n"
    "# Store lu_nnz for retrieval when loading from file cache\n"
    "{func_name}.lu_nnz = {lu_nnz}\n"
)


LU_PREPARE_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LU BLOCK PREPARATION FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0,"
    " lineinfo=None):\n"
    '    """Auto-generated step-start LU block factorisation.\n'
    "    {description}\n"
    "    Evaluates J once at the given state and stores every block's\n"
    "    L/U entries into cached_aux at literal offsets.\n"
    "    Returns device function:\n"
    "      prepare_lu(state, parameters, drivers, t, h, cached_aux)\n"
    "          -> int32\n"
    "    Returns int32({singular_code}) (SINGULAR_PIVOT) when any\n"
    "    pivot was magnitude-floored, else int32(0).\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def prepare_lu(\n"
    "        state, parameters, drivers, t, _cubie_codegen_h,"
    " cached_aux\n"
    "    ):\n"
    "{body}\n"
    "        return selp(\n"
    "            _cubie_codegen_lu_singular != int32(0),\n"
    "            int32({singular_code}),\n"
    "            int32(0),\n"
    "        )\n"
    "    return prepare_lu\n"
    "# Store aux_count for retrieval when loading from file cache\n"
    "{func_name}.aux_count = {aux_count}\n"
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


_FLOOR_NUM = ir.num(DIAG_DIVISION_FLOOR)
_FLOOR_SQUARED_NUM = ir.num(DIAG_DIVISION_FLOOR * DIAG_DIVISION_FLOOR)


def _real_factor_exprs(
    lu_pattern: Set[Tuple[int, int]],
    w_lookup: Callable[[int, int], ir.Expr],
    write_ref: Callable[[int, int], ir.Expr],
    read_ref: Callable[[int, int], ir.Expr],
    name_prefix: str,
) -> Tuple[
    List[Tuple[ir.Expr, ir.Expr]],
    List[ir.Expr],
]:
    """Return one real block's elimination and pivot indicators."""
    exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    indicators: List[ir.Expr] = []
    for (a, b) in sorted(lu_pattern):
        terms = [w_lookup(a, b)]
        for m in range(min(a, b)):
            if (a, m) in lu_pattern and (m, b) in lu_pattern:
                terms.append(
                    ir.mul(
                        ir.num(-1),
                        read_ref(a, m),
                        read_ref(m, b),
                    )
                )
        total = ir.add(*terms)
        if a > b:
            exprs.append(
                (write_ref(a, b), ir.div(total, read_ref(b, b)))
            )
        elif a == b:
            raw = ir.sym(f"{name_prefix}_d_{a}")
            exprs.append((raw, total))
            guarded = ir.piecewise(
                (
                    raw,
                    ir.rel(">=", ir.call("Abs", raw), _FLOOR_NUM),
                ),
                (_FLOOR_NUM, ir.TRUE),
            )
            exprs.append((write_ref(a, a), guarded))
            indicators.append(
                ir.piecewise(
                    (
                        ir.ONE,
                        ir.rel(
                            "<", ir.call("Abs", raw), _FLOOR_NUM
                        ),
                    ),
                    (ir.ZERO, ir.TRUE),
                )
            )
        else:
            exprs.append((write_ref(a, b), total))
    return exprs, indicators


def _real_substitution_exprs(
    lu_pattern: Set[Tuple[int, int]],
    perm: List[int],
    read_ref: Callable[[int, int], ir.Expr],
    rhs_read: Callable[[int], ir.Expr],
    out_write: Callable[[int, ir.Expr], Tuple[ir.Expr, ir.Expr]],
    name_prefix: str,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Return forward/back substitution for one real block."""
    n = len(perm)
    exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    y_syms = [ir.sym(f"{name_prefix}_y_{a}") for a in range(n)]
    for a in range(n):
        terms = [rhs_read(a)]
        for b in range(a):
            if (a, b) in lu_pattern:
                terms.append(
                    ir.mul(ir.num(-1), read_ref(a, b), y_syms[b])
                )
        exprs.append((y_syms[a], ir.add(*terms)))
    x_syms = [ir.sym(f"{name_prefix}_xs_{a}") for a in range(n)]
    for a in range(n - 1, -1, -1):
        terms = [y_syms[a]]
        for b in range(a + 1, n):
            if (a, b) in lu_pattern:
                terms.append(
                    ir.mul(ir.num(-1), read_ref(a, b), x_syms[b])
                )
        exprs.append(
            (x_syms[a], ir.div(ir.add(*terms), read_ref(a, a)))
        )
    for a in range(n):
        exprs.append(out_write(perm[a], x_syms[a]))
    return exprs


def _complex_factor_exprs(
    lu_pattern: Set[Tuple[int, int]],
    w_lookup: Callable[[int, int], Tuple[ir.Expr, ir.Expr]],
    write_pair: Callable[[int, int], Tuple[ir.Expr, ir.Expr]],
    read_pair: Callable[[int, int], Tuple[ir.Expr, ir.Expr]],
    name_prefix: str,
) -> Tuple[
    List[Tuple[ir.Expr, ir.Expr]],
    List[ir.Expr],
]:
    """Return one complex block's elimination as (re, im) pairs."""
    exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    indicators: List[ir.Expr] = []
    pivot_inverse: Dict[int, ir.Expr] = {}

    def _pivot_inverse_sym(column: int) -> ir.Expr:
        cached = pivot_inverse.get(column)
        if cached is not None:
            return cached
        piv_re, piv_im = read_pair(column, column)
        inv_sym = ir.sym(f"{name_prefix}_pinv_{column}")
        exprs.append(
            (
                inv_sym,
                ir.div(
                    ir.ONE,
                    ir.add(
                        ir.mul(piv_re, piv_re),
                        ir.mul(piv_im, piv_im),
                    ),
                ),
            )
        )
        pivot_inverse[column] = inv_sym
        return inv_sym

    for (a, b) in sorted(lu_pattern):
        base_re, base_im = w_lookup(a, b)
        terms_re = [base_re]
        terms_im = [base_im]
        for m in range(min(a, b)):
            if (a, m) in lu_pattern and (m, b) in lu_pattern:
                l_re, l_im = read_pair(a, m)
                u_re, u_im = read_pair(m, b)
                terms_re.append(
                    ir.mul(ir.num(-1), l_re, u_re)
                )
                terms_re.append(ir.mul(l_im, u_im))
                terms_im.append(
                    ir.mul(ir.num(-1), l_re, u_im)
                )
                terms_im.append(
                    ir.mul(ir.num(-1), l_im, u_re)
                )
        total_re = ir.sym(f"{name_prefix}_t_{a}_{b}_re")
        total_im = ir.sym(f"{name_prefix}_t_{a}_{b}_im")
        exprs.append((total_re, ir.add(*terms_re)))
        exprs.append((total_im, ir.add(*terms_im)))
        dest_re, dest_im = write_pair(a, b)
        if a > b:
            piv_re, piv_im = read_pair(b, b)
            inv_sym = _pivot_inverse_sym(b)
            exprs.append(
                (
                    dest_re,
                    ir.mul(
                        ir.add(
                            ir.mul(total_re, piv_re),
                            ir.mul(total_im, piv_im),
                        ),
                        inv_sym,
                    ),
                )
            )
            exprs.append(
                (
                    dest_im,
                    ir.mul(
                        ir.sub(
                            ir.mul(total_im, piv_re),
                            ir.mul(total_re, piv_im),
                        ),
                        inv_sym,
                    ),
                )
            )
        elif a == b:
            magnitude = ir.sym(f"{name_prefix}_dmag_{a}")
            exprs.append(
                (
                    magnitude,
                    ir.add(
                        ir.mul(total_re, total_re),
                        ir.mul(total_im, total_im),
                    ),
                )
            )
            keep = ir.rel(">=", magnitude, _FLOOR_SQUARED_NUM)
            exprs.append(
                (
                    dest_re,
                    ir.piecewise(
                        (total_re, keep), (_FLOOR_NUM, ir.TRUE)
                    ),
                )
            )
            exprs.append(
                (
                    dest_im,
                    ir.piecewise(
                        (total_im, keep), (ir.ZERO, ir.TRUE)
                    ),
                )
            )
            indicators.append(
                ir.piecewise(
                    (
                        ir.ONE,
                        ir.rel(
                            "<", magnitude, _FLOOR_SQUARED_NUM
                        ),
                    ),
                    (ir.ZERO, ir.TRUE),
                )
            )
        else:
            exprs.append((dest_re, total_re))
            exprs.append((dest_im, total_im))
    return exprs, indicators


def _complex_substitution_exprs(
    lu_pattern: Set[Tuple[int, int]],
    perm: List[int],
    read_pair: Callable[[int, int], Tuple[ir.Expr, ir.Expr]],
    rhs_read: Callable[[int], Tuple[ir.Expr, ir.Expr]],
    out_write: Callable[
        [int, ir.Expr, ir.Expr],
        List[Tuple[ir.Expr, ir.Expr]],
    ],
    name_prefix: str,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Return forward/back substitution for one complex block."""
    n = len(perm)
    exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    y_re = [ir.sym(f"{name_prefix}_y_{a}_re") for a in range(n)]
    y_im = [ir.sym(f"{name_prefix}_y_{a}_im") for a in range(n)]
    for a in range(n):
        rhs_re_val, rhs_im_val = rhs_read(a)
        terms_re = [rhs_re_val]
        terms_im = [rhs_im_val]
        for b in range(a):
            if (a, b) in lu_pattern:
                l_re, l_im = read_pair(a, b)
                terms_re.append(
                    ir.mul(ir.num(-1), l_re, y_re[b])
                )
                terms_re.append(ir.mul(l_im, y_im[b]))
                terms_im.append(
                    ir.mul(ir.num(-1), l_re, y_im[b])
                )
                terms_im.append(
                    ir.mul(ir.num(-1), l_im, y_re[b])
                )
        exprs.append((y_re[a], ir.add(*terms_re)))
        exprs.append((y_im[a], ir.add(*terms_im)))
    x_re = [ir.sym(f"{name_prefix}_xs_{a}_re") for a in range(n)]
    x_im = [ir.sym(f"{name_prefix}_xs_{a}_im") for a in range(n)]
    for a in range(n - 1, -1, -1):
        terms_re = [y_re[a]]
        terms_im = [y_im[a]]
        for b in range(a + 1, n):
            if (a, b) in lu_pattern:
                u_re, u_im = read_pair(a, b)
                terms_re.append(
                    ir.mul(ir.num(-1), u_re, x_re[b])
                )
                terms_re.append(ir.mul(u_im, x_im[b]))
                terms_im.append(
                    ir.mul(ir.num(-1), u_re, x_im[b])
                )
                terms_im.append(
                    ir.mul(ir.num(-1), u_im, x_re[b])
                )
        num_re = ir.sym(f"{name_prefix}_n_{a}_re")
        num_im = ir.sym(f"{name_prefix}_n_{a}_im")
        exprs.append((num_re, ir.add(*terms_re)))
        exprs.append((num_im, ir.add(*terms_im)))
        piv_re, piv_im = read_pair(a, a)
        inv_sym = ir.sym(f"{name_prefix}_binv_{a}")
        exprs.append(
            (
                inv_sym,
                ir.div(
                    ir.ONE,
                    ir.add(
                        ir.mul(piv_re, piv_re),
                        ir.mul(piv_im, piv_im),
                    ),
                ),
            )
        )
        exprs.append(
            (
                x_re[a],
                ir.mul(
                    ir.add(
                        ir.mul(num_re, piv_re),
                        ir.mul(num_im, piv_im),
                    ),
                    inv_sym,
                ),
            )
        )
        exprs.append(
            (
                x_im[a],
                ir.mul(
                    ir.sub(
                        ir.mul(num_im, piv_re),
                        ir.mul(num_re, piv_im),
                    ),
                    inv_sym,
                ),
            )
        )
    for a in range(n):
        exprs.extend(out_write(perm[a], x_re[a], x_im[a]))
    return exprs


def _lu_body_from_entries(
    sysir: SystemIR,
    entry_exprs: Dict[Tuple[int, int], ir.Expr],
    mass_diag: Tuple[bool, ...],
    prefix_assigns: List[Tuple[ir.Expr, ir.Expr]],
    operation_ordering: str,
    cse: bool = True,
    width: Optional[int] = None,
    jac_scale_syms: Optional[Tuple[ir.Expr, ...]] = None,
) -> Tuple[str, int]:
    """Build the solve body from per-entry Jacobian expressions.

    ``entry_exprs`` holds the structurally nonzero Jacobian entries
    at their evaluation point; ``prefix_assigns`` the auxiliary
    assignments they reference; ``width`` the matrix width;
    ``jac_scale_syms`` the Jacobian-term scaling (default
    ``gamma * a_ij * h``).

    Returns ``(printed body, factor buffer length)``.
    """
    n = width if width is not None else len(sysir.state_symbols)
    pattern = set(entry_exprs) | {(i, i) for i in range(n)}
    perm, lu_pattern, _ = _markowitz_symbolic_lu(
        pattern, n
    )
    nnz = len(lu_pattern)
    slots = {
        entry: idx for idx, entry in enumerate(sorted(lu_pattern))
    }

    def factor_ref(a: int, b: int) -> ir.Expr:
        return ir.arr("factor", slots[(a, b)])

    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    h_sym = ir.sym("_cubie_codegen_h")
    if jac_scale_syms is None:
        jac_scale_syms = (gamma_sym, a_ij_sym, h_sym)

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
            ir.mul(*jac_scale_syms, jac_term),
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
    factor_exprs, indicators = _real_factor_exprs(
        lu_pattern,
        w_lookup=lambda a, b: w_syms.get((a, b), ir.ZERO),
        write_ref=factor_ref,
        read_ref=factor_ref,
        name_prefix="_cubie_codegen_lu",
    )
    block_b = list(factor_exprs)
    block_b.extend(
        _real_substitution_exprs(
            lu_pattern,
            perm,
            read_ref=factor_ref,
            rhs_read=lambda a: ir.arr("rhs", perm[a]),
            out_write=lambda orig, value: (
                ir.arr("x", orig),
                value,
            ),
            name_prefix="_cubie_codegen_lu",
        )
    )
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
    return body, nnz


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
    jvp_equations: JVPEquations,
) -> Tuple[
    Dict[Tuple[int, int], ir.Expr],
    List[Tuple[ir.Expr, ir.Expr]],
]:
    """Return the graph's entry symbols over the cached-slot chain."""
    n = len(sysir.state_symbols)
    prefix_assigns = jvp_equations.cached_runtime_assignments()
    entry_exprs: Dict[Tuple[int, int], ir.Expr] = {}
    for i in range(n):
        for j in range(n):
            entry = jvp_equations.jacobian_entry(i, j)
            if ir.is_zero(entry):
                continue
            entry_exprs[(i, j)] = entry
    return entry_exprs, prefix_assigns


def _stacked_entry_exprs(
    sysir: SystemIR,
    jac: List[List[ir.Expr]],
    coeff_matrix: List[List[ir.Expr]],
    node_values: Tuple[ir.Expr, ...],
) -> Tuple[
    Dict[Tuple[int, int], ir.Expr],
    List[Tuple[ir.Expr, ir.Expr]],
]:
    """Return coupled-matrix entries ``a[bi][bj] * J(Y_bi)``."""
    n = len(sysir.state_symbols)
    stage_count = len(coeff_matrix)
    metadata_exprs, coeff_symbols, node_symbols = (
        build_stage_metadata(coeff_matrix, node_values)
    )
    prefix_assigns: List[Tuple[ir.Expr, ir.Expr]] = list(
        metadata_exprs
    )
    entry_exprs: Dict[Tuple[int, int], ir.Expr] = {}
    for bi in range(stage_count):
        subs_map = build_stage_substitutions(
            sysir,
            bi,
            coeff_symbols,
            node_symbols,
            coeff_matrix,
            state_vector_name="state",
        )
        memo: dict = {}
        prefix_assigns.extend(
            (
                ir.xreplace(lhs, subs_map, memo),
                ir.xreplace(rhs, subs_map, memo),
            )
            for lhs, rhs in sysir.equations
        )
        stage_entries: Dict[Tuple[int, int], ir.Expr] = {}
        for i in range(n):
            for j in range(n):
                entry = jac[i][j]
                if ir.is_zero(entry):
                    continue
                stage_entries[(i, j)] = ir.xreplace(
                    entry, subs_map, memo
                )
        for bj in range(stage_count):
            if ir.is_zero(coeff_matrix[bi][bj]):
                continue
            coeff_sym = coeff_symbols[bi][bj]
            for (i, j), value in stage_entries.items():
                entry_exprs[(bi * n + i, bj * n + j)] = ir.mul(
                    coeff_sym, value
                )
    return entry_exprs, prefix_assigns


def _coefficient_floats(
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
) -> Tuple[Tuple[float, ...], ...]:
    """Return the tableau ``A`` matrix as float rows."""
    return tuple(
        tuple(float(sp.sympify(entry)) for entry in row)
        for row in stage_coefficients
    )


def _distinct_diagonals(
    coeff_floats: Tuple[Tuple[float, ...], ...],
) -> List[float]:
    """Return the distinct nonzero tableau diagonals, ascending."""
    values: List[float] = []
    for idx, row in enumerate(coeff_floats):
        entry = row[idx] if idx < len(row) else 0.0
        if entry == 0.0:
            continue
        if entry not in values:
            values.append(entry)
    if not values:
        raise ValueError(
            "Prefactored LU requires at least one nonzero tableau "
            "diagonal."
        )
    return sorted(values)


def _system_pattern_structure(
    sysir: SystemIR,
    jac: List[List[ir.Expr]],
) -> Tuple[
    List[int],
    Set[Tuple[int, int]],
    Dict[Tuple[int, int], int],
    int,
    int,
]:
    """Return the Markowitz structure of the system's own pattern."""
    n = len(sysir.state_symbols)
    pattern = {
        (i, j)
        for i in range(n)
        for j in range(n)
        if not ir.is_zero(jac[i][j])
    } | {(i, i) for i in range(n)}
    perm, lu_pattern, flops = _markowitz_symbolic_lu(pattern, n)
    slots = {
        entry: idx for idx, entry in enumerate(sorted(lu_pattern))
    }
    return perm, lu_pattern, slots, len(lu_pattern), flops


def _finalise_prepare_body(
    block_a: List[Tuple[ir.Expr, ir.Expr]],
    block_b: List[Tuple[ir.Expr, ir.Expr]],
    indicators: List[ir.Expr],
    total_reals: int,
    sysir: SystemIR,
    operation_ordering: str,
    cse: bool,
) -> str:
    """Sort, prune, and print a block-preparation body."""
    singular_sym = ir.sym("_cubie_codegen_lu_singular")
    if cse:
        block_a = cse_and_stack(
            block_a, operation_ordering=operation_ordering
        )
    else:
        block_a = topological_sort(
            block_a, operation_ordering=operation_ordering
        )
    block_b = list(block_b)
    block_b.append((singular_sym, ir.add(*indicators)))
    block_b = topological_sort(
        block_b, operation_ordering=operation_ordering
    )
    outputs = [
        ir.arr("cached_aux", idx) for idx in range(total_reals)
    ]
    outputs.append(singular_sym)
    exprs = prune_unused(
        block_a + block_b, output_symbols=outputs
    )
    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def _finalise_solve_lines(
    exprs: List[Tuple[ir.Expr, ir.Expr]],
    width: int,
    sysir: SystemIR,
    operation_ordering: str,
) -> List[str]:
    """Sort, prune to ``x``, and print a substitution-solve body."""
    exprs = topological_sort(
        exprs, operation_ordering=operation_ordering
    )
    outputs = [ir.arr("x", i) for i in range(width)]
    exprs = prune_unused(exprs, output_symbols=outputs)
    return print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )


def _prefactored_solve_source(
    sysir: SystemIR,
    jac: List[List[ir.Expr]],
    coeff_floats: Tuple[Tuple[float, ...], ...],
    func_name: str,
    operation_ordering: str,
) -> Tuple[str, int]:
    """Emit the per-diagonal prefactored substitution factory."""
    perm, lu_pattern, slots, nnz, _ = _system_pattern_structure(
        sysir, jac
    )
    diagonals = _distinct_diagonals(coeff_floats)

    branch_bodies: List[List[str]] = []
    for block, _diag in enumerate(diagonals):
        offset = block * nnz

        def read_ref(a: int, b: int, _off=offset) -> ir.Expr:
            return ir.arr("cached_aux", _off + slots[(a, b)])

        exprs = _real_substitution_exprs(
            lu_pattern,
            perm,
            read_ref=read_ref,
            rhs_read=lambda a: ir.arr("rhs", perm[a]),
            out_write=lambda orig, value: (
                ir.arr("x", orig),
                value,
            ),
            name_prefix="_cubie_codegen_lu",
        )
        lines = _finalise_solve_lines(
            exprs, len(perm), sysir, operation_ordering
        )
        branch_bodies.append(lines)

    preamble_lines = [
        f"    _cubie_codegen_lu_diag_{k} = precision({diag!r})\n"
        for k, diag in enumerate(diagonals)
    ]
    preamble = "".join(preamble_lines)

    if len(diagonals) == 1:
        body = "\n".join(
            "        " + ln for ln in branch_bodies[0]
        )
    else:
        chunks: List[str] = []
        for k in range(len(diagonals) - 1):
            keyword = "if" if k == 0 else "elif"
            chunks.append(
                f"        {keyword} _cubie_codegen_a_ij == "
                f"_cubie_codegen_lu_diag_{k}:"
            )
            chunks.extend(
                "            " + ln for ln in branch_bodies[k]
            )
        chunks.append("        else:")
        chunks.extend(
            "            " + ln for ln in branch_bodies[-1]
        )
        body = "\n".join(chunks)

    description = (
        "Substitutes rhs against the step-start factor of\n"
        "    beta*M - gamma*h*d_k*J(y_n) for the tableau diagonal\n"
        "    d_k selected by a_ij."
    )
    code = LU_SUBSTITUTE_TEMPLATE.format(
        func_name=func_name,
        description=description,
        preamble=preamble,
        body=body,
        lu_nnz=0,
    )
    return code, 0


def _transformed_solve_source(
    sysir: SystemIR,
    jac: List[List[ir.Expr]],
    coeff_floats: Tuple[Tuple[float, ...], ...],
    func_name: str,
    operation_ordering: str,
) -> Tuple[str, int]:
    """Emit the eigenvalue block-transform substitution factory."""
    n = len(sysir.state_symbols)
    stage_count = len(coeff_floats)
    perm, lu_pattern, slots, nnz, _ = _system_pattern_structure(
        sysir, jac
    )
    real_values, pair_values, transform, _ = block_eigenstructure(
        coeff_floats
    )
    n_real = len(real_values)
    lam = np.zeros((stage_count, stage_count))
    for k, value in enumerate(real_values):
        lam[k, k] = value
    for p, (alpha, beta_im) in enumerate(pair_values):
        row = n_real + 2 * p
        lam[row, row] = alpha
        lam[row, row + 1] = beta_im
        lam[row + 1, row] = -beta_im
        lam[row + 1, row + 1] = alpha
    inverse_transform = np.linalg.inv(np.asarray(transform))
    lam_inv_t = lam @ inverse_transform

    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_sym = ir.sym("_cubie_codegen_h")
    ghinv = ir.sym("_cubie_codegen_lu_ghinv")
    exprs: List[Tuple[ir.Expr, ir.Expr]] = [
        (ghinv, ir.div(ir.ONE, ir.mul(gamma_sym, h_sym)))
    ]

    # Transformed right-hand side: b' = (L T^-1 (x) I) b / (gamma*h).
    def bp_sym(row: int, comp: int) -> ir.Expr:
        return ir.sym(f"_cubie_codegen_lu_bp_{row}_{comp}")

    for row in range(stage_count):
        for comp in range(n):
            terms = [
                ir.mul(
                    ir.num(float(lam_inv_t[row, j])),
                    ir.arr("rhs", j * n + comp),
                )
                for j in range(stage_count)
                if lam_inv_t[row, j] != 0.0
            ]
            combo = ir.add(*terms) if terms else ir.ZERO
            exprs.append((bp_sym(row, comp), ir.mul(ghinv, combo)))

    def xp_sym(row: int, comp: int) -> ir.Expr:
        return ir.sym(f"_cubie_codegen_lu_xp_{row}_{comp}")

    # Real blocks.
    for k in range(n_real):
        offset = k * nnz

        def read_ref(a: int, b: int, _off=offset) -> ir.Expr:
            return ir.arr("cached_aux", _off + slots[(a, b)])

        exprs.extend(
            _real_substitution_exprs(
                lu_pattern,
                perm,
                read_ref=read_ref,
                rhs_read=lambda a, _row=k: bp_sym(_row, perm[a]),
                out_write=lambda orig, value, _row=k: (
                    xp_sym(_row, orig),
                    value,
                ),
                name_prefix=f"_cubie_codegen_lu_r{k}",
            )
        )

    # Pair solve: (mu*M - J0) z = b'1 - i*b'2; x' rows are Re z, -Im z.
    for p in range(len(pair_values)):
        offset = n_real * nnz + p * 2 * nnz
        row_a = n_real + 2 * p
        row_b = row_a + 1

        def read_pair(
            a: int, b: int, _off=offset
        ) -> Tuple[ir.Expr, ir.Expr]:
            base = _off + 2 * slots[(a, b)]
            return (
                ir.arr("cached_aux", base),
                ir.arr("cached_aux", base + 1),
            )

        def rhs_read(
            a: int, _ra=row_a, _rb=row_b
        ) -> Tuple[ir.Expr, ir.Expr]:
            return (
                bp_sym(_ra, perm[a]),
                ir.mul(ir.num(-1), bp_sym(_rb, perm[a])),
            )

        def out_write(
            orig: int,
            re_val: ir.Expr,
            im_val: ir.Expr,
            _ra=row_a,
            _rb=row_b,
        ) -> List[Tuple[ir.Expr, ir.Expr]]:
            return [
                (xp_sym(_ra, orig), re_val),
                (
                    xp_sym(_rb, orig),
                    ir.mul(ir.num(-1), im_val),
                ),
            ]

        exprs.extend(
            _complex_substitution_exprs(
                lu_pattern,
                perm,
                read_pair=read_pair,
                rhs_read=rhs_read,
                out_write=out_write,
                name_prefix=f"_cubie_codegen_lu_c{p}",
            )
        )

    # Inverse transform: x = (T (x) I) x'.
    transform_rows = transform
    for j in range(stage_count):
        for comp in range(n):
            terms = [
                ir.mul(
                    ir.num(float(transform_rows[j][k])),
                    xp_sym(k, comp),
                )
                for k in range(stage_count)
                if transform_rows[j][k] != 0.0
            ]
            combo = ir.add(*terms) if terms else ir.ZERO
            exprs.append((ir.arr("x", j * n + comp), combo))

    lines = _finalise_solve_lines(
        exprs, stage_count * n, sysir, operation_ordering
    )
    body = "\n".join("        " + ln for ln in lines)
    description = (
        "Solves the coupled frozen-Jacobian FIRK system through the\n"
        "    real block-diagonalising transform of inv(A): transform\n"
        "    the rhs, substitute against each eigenvalue block's\n"
        "    step-start factor, transform back."
    )
    code = LU_SUBSTITUTE_TEMPLATE.format(
        func_name=func_name,
        description=description,
        preamble="",
        body=body,
        lu_nnz=0,
    )
    return code, 0


def generate_lu_solve_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PLAIN,
    M: Optional[Union[Iterable, object]] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "lu_solve_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = operation_ordering_default(),
) -> Tuple[str, int]:
    """Generate the direct LU solve factory for one variant.

    ``PLAIN`` evaluates ``J`` at ``base_state + a_ij * state``;
    ``AT_STATE`` at ``state`` with ``a_ij`` scaling the matrix only;
    ``CACHED`` at ``state`` with auxiliaries from ``cached_aux``;
    ``STACKED_STAGES`` factorises the coupled ``s*n`` FIRK matrix
    with the tableau baked in; ``PREFACTORED`` substitutes against
    step-start per-diagonal factors; ``PREFACTORED_STACKED`` is the
    eigenvalue block-transform solve against step-start block
    factors.

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
    stage_coefficients
        Butcher tableau A matrix; stage-data variants only.
    stage_nodes
        Butcher tableau c vector; stage-data variants only.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination to the
        matrix-entry block.
    jvp_equations
        Prebuilt JVP equations for the cached variant; generated
        when absent.

    Returns ``(factory source, factor length)``; substitution-only
    variants report length zero.
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
    if variant.takes_stage_data:
        coeff_matrix, node_values, stage_count = prepare_stage_data(
            stage_coefficients, stage_nodes
        )
    if variant is HelperVariant.PREFACTORED:
        code, lu_nnz = _prefactored_solve_source(
            sysir,
            jac,
            _coefficient_floats(stage_coefficients),
            func_name,
            operation_ordering,
        )
        default_timelogger.stop_event(event)
        return code, lu_nnz
    if variant is HelperVariant.PREFACTORED_STACKED:
        code, lu_nnz = _transformed_solve_source(
            sysir,
            jac,
            _coefficient_floats(stage_coefficients),
            func_name,
            operation_ordering,
        )
        default_timelogger.stop_event(event)
        return code, lu_nnz

    width = n
    jac_scale_syms = None
    if variant.stacked_stages:
        entry_exprs, prefix_assigns = _stacked_entry_exprs(
            sysir, jac, coeff_matrix, node_values
        )
        width = stage_count * n
        mass_diag = mass_diag * stage_count
        jac_scale_syms = (
            ir.sym("_cubie_codegen_gamma"),
            ir.sym("_cubie_codegen_h"),
        )
        eval_point = (
            "each stage's base_state + sum(a_ik * state_k) with the "
            "tableau baked in (a_ij unused)"
        )
    elif variant.cached:
        jvp_equations = _resolve_jvp(
            equations,
            index_map,
            cse,
            jvp_equations,
            operation_ordering,
        )
        entry_exprs, prefix_assigns = _cached_entry_exprs(
            sysir, jvp_equations
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
        width=width,
        jac_scale_syms=jac_scale_syms,
    )
    code = LU_SOLVE_TEMPLATE.format(
        func_name=func_name,
        body=body,
        lu_nnz=lu_nnz,
        eval_point=eval_point,
        singular_code=_SINGULAR_PIVOT,
    )
    default_timelogger.stop_event(event)
    return code, lu_nnz


def generate_lu_prepare_blocks_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PREFACTORED,
    M: Optional[Union[Iterable, object]] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "lu_prepare_blocks_factory",
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> Tuple[str, int]:
    """Generate the step-start LU block factorisation factory.

    ``PREFACTORED`` factorises ``beta*M - gamma*h*d_k*J(state)`` for
    each distinct nonzero tableau diagonal ``d_k``, stored
    consecutively in ``cached_aux``. ``PREFACTORED_STACKED``
    factorises the block transform's eigenvalue blocks
    ``(beta*lambda/(gamma*h))*M - J(state)``: real blocks first,
    then one complex block per conjugate pair with interleaved
    re/im entries.

    Returns
    -------
    tuple of str and int
        Generated factory source and the flat ``cached_aux`` factor
        length in reals.
    """
    event = f"codegen_lu_prepare_blocks_{variant.value}"
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
    perm, lu_pattern, slots, nnz, _ = (
        _system_pattern_structure(sysir, jac)
    )
    coeff_floats = _coefficient_floats(stage_coefficients)
    entry_exprs, prefix_assigns = _inline_entry_exprs(
        sysir, jac, state_is_increment=False
    )
    pattern_positions = sorted(
        {
            (a, b)
            for (a, b) in lu_pattern
            if (perm[a], perm[b]) in entry_exprs
            or perm[a] == perm[b]
        }
    )

    def permuted_entry(a: int, b: int) -> ir.Expr:
        return entry_exprs.get((perm[a], perm[b]), ir.ZERO)

    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_sym = ir.sym("_cubie_codegen_h")

    block_a: List[Tuple[ir.Expr, ir.Expr]] = list(prefix_assigns)
    block_b: List[Tuple[ir.Expr, ir.Expr]] = []
    indicators: List[ir.Expr] = []

    if variant is HelperVariant.PREFACTORED:
        diagonals = _distinct_diagonals(coeff_floats)
        total_reals = len(diagonals) * nnz
        for block, diag in enumerate(diagonals):
            offset = block * nnz
            scale = ir.sym(f"_cubie_codegen_lu_b{block}_scale")
            block_a.append(
                (
                    scale,
                    ir.mul(gamma_sym, h_sym, ir.num(diag)),
                )
            )
            w_syms: Dict[Tuple[int, int], ir.Expr] = {}
            for (a, b) in pattern_positions:
                i, j = perm[a], perm[b]
                mass_term = (
                    beta_sym
                    if (i == j and mass_diag[i])
                    else ir.ZERO
                )
                value = ir.sub(
                    mass_term,
                    ir.mul(scale, permuted_entry(a, b)),
                )
                w_sym = ir.sym(
                    f"_cubie_codegen_lu_b{block}_w_{a}_{b}"
                )
                w_syms[(a, b)] = w_sym
                block_a.append((w_sym, value))

            def factor_ref(
                a: int, b: int, _off=offset
            ) -> ir.Expr:
                return ir.arr(
                    "cached_aux", _off + slots[(a, b)]
                )

            block_exprs, block_indicators = _real_factor_exprs(
                lu_pattern,
                w_lookup=lambda a, b, _w=w_syms: _w.get(
                    (a, b), ir.ZERO
                ),
                write_ref=factor_ref,
                read_ref=factor_ref,
                name_prefix=f"_cubie_codegen_lu_b{block}",
            )
            block_b.extend(block_exprs)
            indicators.extend(block_indicators)
        description = (
            "Factorises beta*M - gamma*h*d_k*J(state) for each\n"
            "    distinct nonzero tableau diagonal d_k."
        )
    else:
        real_values, pair_values, _, _ = block_eigenstructure(
            coeff_floats
        )
        n_real = len(real_values)
        total_reals = (n_real + 2 * len(pair_values)) * nnz
        ghinv = ir.sym("_cubie_codegen_lu_ghinv")
        block_a.append(
            (ghinv, ir.div(ir.ONE, ir.mul(gamma_sym, h_sym)))
        )
        for k, value in enumerate(real_values):
            offset = k * nnz
            mu = ir.sym(f"_cubie_codegen_lu_b{k}_mu")
            block_a.append(
                (
                    mu,
                    ir.mul(beta_sym, ir.num(value), ghinv),
                )
            )
            w_syms = {}
            for (a, b) in pattern_positions:
                i, j = perm[a], perm[b]
                mass_term = (
                    mu if (i == j and mass_diag[i]) else ir.ZERO
                )
                value_expr = ir.sub(
                    mass_term, permuted_entry(a, b)
                )
                w_sym = ir.sym(
                    f"_cubie_codegen_lu_b{k}_w_{a}_{b}"
                )
                w_syms[(a, b)] = w_sym
                block_a.append((w_sym, value_expr))

            def factor_ref(
                a: int, b: int, _off=offset
            ) -> ir.Expr:
                return ir.arr(
                    "cached_aux", _off + slots[(a, b)]
                )

            block_exprs, block_indicators = _real_factor_exprs(
                lu_pattern,
                w_lookup=lambda a, b, _w=w_syms: _w.get(
                    (a, b), ir.ZERO
                ),
                write_ref=factor_ref,
                read_ref=factor_ref,
                name_prefix=f"_cubie_codegen_lu_b{k}",
            )
            block_b.extend(block_exprs)
            indicators.extend(block_indicators)
        for p, (alpha, beta_im) in enumerate(pair_values):
            offset = n_real * nnz + p * 2 * nnz
            mu_re = ir.sym(f"_cubie_codegen_lu_p{p}_mu_re")
            mu_im = ir.sym(f"_cubie_codegen_lu_p{p}_mu_im")
            block_a.append(
                (
                    mu_re,
                    ir.mul(beta_sym, ir.num(alpha), ghinv),
                )
            )
            block_a.append(
                (
                    mu_im,
                    ir.mul(beta_sym, ir.num(beta_im), ghinv),
                )
            )
            w_pairs: Dict[
                Tuple[int, int], Tuple[ir.Expr, ir.Expr]
            ] = {}
            for (a, b) in pattern_positions:
                i, j = perm[a], perm[b]
                if i == j and mass_diag[i]:
                    value_re = ir.sub(mu_re, permuted_entry(a, b))
                    value_im = mu_im
                else:
                    value_re = ir.sub(
                        ir.ZERO, permuted_entry(a, b)
                    )
                    value_im = ir.ZERO
                sym_re = ir.sym(
                    f"_cubie_codegen_lu_p{p}_w_{a}_{b}_re"
                )
                sym_im = ir.sym(
                    f"_cubie_codegen_lu_p{p}_w_{a}_{b}_im"
                )
                w_pairs[(a, b)] = (sym_re, sym_im)
                block_a.append((sym_re, value_re))
                block_a.append((sym_im, value_im))

            def write_pair(
                a: int, b: int, _off=offset
            ) -> Tuple[ir.Expr, ir.Expr]:
                base = _off + 2 * slots[(a, b)]
                return (
                    ir.arr("cached_aux", base),
                    ir.arr("cached_aux", base + 1),
                )

            block_exprs, block_indicators = _complex_factor_exprs(
                lu_pattern,
                w_lookup=lambda a, b, _w=w_pairs: _w.get(
                    (a, b), (ir.ZERO, ir.ZERO)
                ),
                write_pair=write_pair,
                read_pair=write_pair,
                name_prefix=f"_cubie_codegen_lu_p{p}",
            )
            block_b.extend(block_exprs)
            indicators.extend(block_indicators)
        description = (
            "Factorises the block transform's eigenvalue blocks\n"
            "    (beta*lambda/(gamma*h))*M - J(state): real blocks\n"
            "    first, then interleaved re/im complex pair blocks."
        )

    body = _finalise_prepare_body(
        block_a,
        block_b,
        indicators,
        total_reals,
        sysir,
        operation_ordering,
        cse,
    )
    code = LU_PREPARE_TEMPLATE.format(
        func_name=func_name,
        description=description,
        body=body,
        aux_count=total_reals,
        singular_code=_SINGULAR_PIVOT,
    )
    default_timelogger.stop_event(event)
    return code, total_reals


def generate_lu_smoothing_solve_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "lu_smoothing_solve_factory",
    operation_ordering: str = operation_ordering_default(),
) -> Tuple[str, int]:
    """Generate the smoothing solve sharing the real eigen block.

    Solves ``(beta*M - gamma*g*h*J(y_n)) x = rhs`` with
    ``g = 1/lambda_real`` by scaling the rhs with
    ``lambda_real/(gamma*h)`` and substituting against the real
    block factor ``lu_prepare_blocks`` stored first in
    ``cached_aux``.
    """
    default_timelogger.start_event("codegen_lu_smoothing_solve")

    sysir = system_ir(equations, index_map)
    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        operation_ordering=operation_ordering,
    )
    perm, lu_pattern, slots, nnz, _ = _system_pattern_structure(
        sysir, jac
    )
    coeff_floats = _coefficient_floats(stage_coefficients)
    real_values, _, _, _ = block_eigenstructure(coeff_floats)
    if len(real_values) != 1:
        raise ValueError(
            "The smoothing solve requires exactly one real "
            f"eigenvalue of inv(A); found {len(real_values)}."
        )
    lam_real = real_values[0]

    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_sym = ir.sym("_cubie_codegen_h")
    scale = ir.sym("_cubie_codegen_lu_scale")
    exprs: List[Tuple[ir.Expr, ir.Expr]] = [
        (
            scale,
            ir.div(ir.num(lam_real), ir.mul(gamma_sym, h_sym)),
        )
    ]

    def read_ref(a: int, b: int) -> ir.Expr:
        return ir.arr("cached_aux", slots[(a, b)])

    exprs.extend(
        _real_substitution_exprs(
            lu_pattern,
            perm,
            read_ref=read_ref,
            rhs_read=lambda a: ir.mul(
                scale, ir.arr("rhs", perm[a])
            ),
            out_write=lambda orig, value: (
                ir.arr("x", orig),
                value,
            ),
            name_prefix="_cubie_codegen_lu",
        )
    )
    lines = _finalise_solve_lines(
        exprs, len(perm), sysir, operation_ordering
    )
    body = "\n".join("        " + ln for ln in lines)
    description = (
        "Solves (beta*M - gamma*g*h*J(y_n)) x = rhs with\n"
        "    g = 1/lambda_real against the real eigenvalue block's\n"
        "    step-start factor."
    )
    code = LU_SUBSTITUTE_TEMPLATE.format(
        func_name=func_name,
        description=description,
        preamble="",
        body=body,
        lu_nnz=0,
    )
    default_timelogger.stop_event("codegen_lu_smoothing_solve")
    return code, 0


__all__ = [
    "generate_lu_solve_code",
    "generate_lu_prepare_blocks_code",
    "generate_lu_smoothing_solve_code",
]
