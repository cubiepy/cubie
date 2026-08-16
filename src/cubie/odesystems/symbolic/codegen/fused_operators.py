"""Emit CUDA factory code for fused operator-preconditioner helpers.

Every Krylov iteration of the matrix-free linear solvers applies the
preconditioner and then the linear operator to the same vector. The
generators here fuse both applications into one device function
computing ``z = P(v)`` and ``out = A(z)`` from a single math graph:
the preconditioner and operator expressions are merged *before*
common-subexpression elimination, so shared Jacobian entries are
computed once, the intermediate ``z`` lives in scalar locals instead
of a scratch array, and the emission-ordering policy sees the whole
per-iteration math DAG at once.

Published Functions
-------------------
:func:`generate_fused_operator_code`
    Newton--Krylov variant; the Jacobian evaluates at
    ``base_state + a_ij * state``.

:func:`generate_fused_operator_at_state_code`
    Error-smoothing variant; the Jacobian evaluates at ``state`` and
    ``a_ij`` scales the matrix only.

:func:`generate_fused_operator_cached_code`
    Rosenbrock-W variant reading precomputed auxiliaries from the
    shared ``cached_aux`` buffer (slot layout owned by
    ``prepare_jac``).

:func:`generate_n_stage_fused_operator_code`
    Flattened all-stages FIRK variant over ``s * n`` unknowns.

Notes
-----
Neumann members are unrolled to their truncation order, so the order
becomes part of the emitted source for fused helpers that contain
one. Jacobi members bake the guarded-diagonal division inline. The
device contract appends two output vectors: ``z_out`` receives the
preconditioned vector and ``out`` receives the operator applied to
it.

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Unfused operator generators (still used at solve entry).
:mod:`cubie.odesystems.symbolic.codegen.preconditioners`
    Unfused preconditioner generators.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    _guarded_diag_division,
    _mass_diag_term,
)
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    build_stage_substitutions,
)
from cubie.odesystems.symbolic.codegen._matrix_utils import (
    mass_matrix_ir,
)
from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.parsing.parser import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger

default_timelogger.register_event(
    "codegen_generate_fused_operator_code", "codegen",
    "Codegen time for generate_fused_operator_code")
default_timelogger.register_event(
    "codegen_generate_fused_operator_at_state_code", "codegen",
    "Codegen time for generate_fused_operator_at_state_code")
default_timelogger.register_event(
    "codegen_generate_fused_operator_cached_code", "codegen",
    "Codegen time for generate_fused_operator_cached_code")
default_timelogger.register_event(
    "codegen_generate_n_stage_fused_operator_code", "codegen",
    "Codegen time for generate_n_stage_fused_operator_code")

FUSED_OPERATOR_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED FUSED OPERATOR-PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated fused preconditioner-operator application.\n'
    "    Computes z_out = P(v) and out = beta * (M @ z_out)\n"
    "    - gamma * a_ij * h * (J @ z_out) in one math graph.\n"
    "    Returns device function:\n"
    "      fused_operator(\n"
    "          state, parameters, drivers, base_state, t, h, a_ij,\n"
    "          v, z_out, out\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta_inv = precision(\n"
    "        1.0 / _cubie_codegen_beta\n"
    "    )\n"
    "    _cubie_codegen_h_eff_factor = precision(\n"
    "        _cubie_codegen_gamma * _cubie_codegen_beta_inv\n"
    "    )\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def fused_operator(\n"
    "        state, parameters, drivers, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, z_out, out,\n"
    "    ):\n"
    "{body}\n"
    "    return fused_operator\n"
)


FUSED_OPERATOR_CACHED_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED FUSED OPERATOR-PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated cached fused preconditioner-operator.\n'
    "    Computes z_out = P(v) and out = beta * (M @ z_out)\n"
    "    - gamma * a_ij * h * (J @ z_out) using cached auxiliaries.\n"
    "    Returns device function:\n"
    "      fused_operator(\n"
    "          state, parameters, drivers, cached_aux, base_state, t,\n"
    "          h, a_ij, v, z_out, out\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta_inv = precision(\n"
    "        1.0 / _cubie_codegen_beta\n"
    "    )\n"
    "    _cubie_codegen_h_eff_factor = precision(\n"
    "        _cubie_codegen_gamma * _cubie_codegen_beta_inv\n"
    "    )\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def fused_operator(\n"
    "        state, parameters, drivers, cached_aux, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, z_out, out,\n"
    "    ):\n"
    "{body}\n"
    "    return fused_operator\n"
)


N_STAGE_FUSED_OPERATOR_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE FUSED OPERATOR-PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated FIRK fused preconditioner-operator.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns:\n"
    "    z_out = P(v) and out = beta * (M @ z_out)\n"
    "    - gamma * h * ((A x J) @ z_out) in one math graph.\n"
    "    Returns device function:\n"
    "      fused_operator(\n"
    "          state, parameters, drivers, base_state, t, h, a_ij,\n"
    "          v, z_out, out\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta_inv = precision(\n"
    "        1.0 / _cubie_codegen_beta\n"
    "    )\n"
    "    _cubie_codegen_h_eff_factor = precision(\n"
    "        _cubie_codegen_gamma * _cubie_codegen_beta_inv\n"
    "    )\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def fused_operator(\n"
    "        state, parameters, drivers, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, z_out, out,\n"
    "    ):\n"
    "{body}\n"
    "    return fused_operator\n"
)


FUSED_CSE_SCOPE = "split"
"""CSE scoping strategy for the non-cached fused builders.

``"split"`` runs CSE separately over the preconditioner and operator
halves so only the scalar ``z`` interface crosses between them —
shared Jacobian entries are recomputed per half with tight liveness,
exactly as the unfused pair computes them, and the fusion win is the
eliminated ``preconditioned_vec`` array round-trip. ``"whole"`` runs
one CSE over the merged graph, computing shared entries once at the
price of holding them live across the ``z`` barrier; measured on the
Fabbri kernel this loses to ``"split"`` on local-memory pressure.
The final emission ordering always runs over the whole fused graph.
"""


def _member_type(member) -> str:
    """Return ``"jacobi"`` or ``"neumann"`` for one member kind.

    Parameters
    ----------
    member
        A concrete preconditioner
        :class:`~cubie.odesystems.solver_helpers.SolverHelperKind`
        or its string value.

    Raises
    ------
    ValueError
        If the member is not a Jacobi or Neumann preconditioner kind.
    """
    value = getattr(member, "value", member)
    if "jacobi" in value:
        return "jacobi"
    if "neumann" in value:
        return "neumann"
    raise ValueError(
        f"Unsupported fused preconditioner member {value!r}."
    )


def _named_vector(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
    exprs: Dict[int, ir.Expr],
    length: int,
    prefix: str,
) -> Dict[int, ir.Expr]:
    """Bind vector expressions to named locals and return the symbols.

    Expressions that are already atoms pass through unnamed so trivial
    copies never materialise.
    """
    named: Dict[int, ir.Expr] = {}
    for idx in range(length):
        expression = exprs.get(idx, ir.ZERO)
        if isinstance(
            expression, (ir.Sym, ir.Local, ir.Arr, ir.Num)
        ):
            named[idx] = expression
            continue
        symbol = ir.sym(f"{prefix}_{idx}")
        assignments.append((symbol, expression))
        named[idx] = symbol
    return named


def _single_stage_jacobi(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
    vec: Dict[int, ir.Expr],
    j_syms: Dict[Tuple[int, int], ir.Expr],
    mass: List[List[ir.Expr]],
    n: int,
    member_idx: int,
    matrix_scale: Tuple[ir.Expr, ...],
) -> Dict[int, ir.Expr]:
    """Apply a Jacobi member: ``z_i = v_i / safe(diag_i)``."""
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    out: Dict[int, ir.Expr] = {}
    for i in range(n):
        j_ii = j_syms.get((i, i), ir.ZERO)
        diag_sym = ir.sym(f"_cubie_codegen_fdiag{member_idx}_{i}")
        diag_val = ir.sub(
            _mass_diag_term(mass, i, beta_sym),
            ir.mul(gamma_sym, *matrix_scale, j_ii),
        )
        assignments.append((diag_sym, diag_val))
        safe_sym, guarded = _guarded_diag_division(
            diag_sym, i, stage_idx=f"fused{member_idx}"
        )
        assignments.append((safe_sym, guarded))
        out[i] = ir.div(vec[i], safe_sym)
    return _named_vector(
        assignments, out, n, f"_cubie_codegen_fz{member_idx}"
    )


def _single_stage_neumann(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
    vec: Dict[int, ir.Expr],
    j_syms: Dict[Tuple[int, int], ir.Expr],
    n: int,
    member_idx: int,
    order: int,
    h_eff_sym: ir.Expr,
) -> Dict[int, ir.Expr]:
    """Apply an unrolled Neumann member in Horner form.

    ``acc_{k+1} = v + h_eff * (J @ acc_k)`` starting from
    ``acc_0 = v``; the member output is ``beta_inv * acc_order``.
    """
    beta_inv_sym = ir.sym("_cubie_codegen_beta_inv")
    acc = dict(vec)
    for k in range(order):
        jv: Dict[int, ir.Expr] = {}
        for i in range(n):
            terms = [
                ir.mul(j_syms[(i, j)], acc[j])
                for j in range(n)
                if (i, j) in j_syms
            ]
            jv[i] = ir.add(*terms) if terms else ir.ZERO
        jv = _named_vector(
            assignments,
            jv,
            n,
            f"_cubie_codegen_fjv{member_idx}_{k}",
        )
        nxt = {
            i: ir.add(vec[i], ir.mul(h_eff_sym, jv[i]))
            for i in range(n)
        }
        acc = _named_vector(
            assignments,
            nxt,
            n,
            f"_cubie_codegen_facc{member_idx}_{k}",
        )
    scaled = {
        i: ir.mul(beta_inv_sym, acc[i]) for i in range(n)
    }
    return _named_vector(
        assignments, scaled, n, f"_cubie_codegen_fz{member_idx}"
    )


def _scope_tagged(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Rename every scalar assignment target into the ``_p_`` scope.

    Applied to the preconditioner half under split-scope CSE so its
    intermediates never collide with — or share with — the operator
    half's same-named intermediates.
    """
    tag_map: Dict[ir.Expr, ir.Expr] = {}
    for lhs, _ in assignments:
        if isinstance(lhs, ir.Sym):
            tag_map[lhs] = ir.sym(f"_cubie_codegen_p_{lhs.name}")
    memo: dict = {}
    return [
        (
            ir.xreplace(lhs, tag_map, memo),
            ir.xreplace(rhs, tag_map, memo),
        )
        for lhs, rhs in assignments
    ]


def _build_single_stage_fused_body(
    equations: ParsedEquations,
    index_map: IndexedBases,
    sysir: SystemIR,
    member_types: Sequence[str],
    mass: List[List[ir.Expr]],
    order: int,
    operation_ordering: str,
    state_is_increment: bool,
    cse: bool,
) -> str:
    """Build the fused body for the single-stage helper variants."""
    n = len(sysir.state_symbols)
    split = FUSED_CSE_SCOPE == "split"

    subs_map: Dict[ir.Expr, ir.Expr] = {}
    for idx, dx_sym in enumerate(sysir.dxdt_symbols):
        subs_map[dx_sym] = ir.sym(f"_cubie_codegen_dx_{idx}")
    obs_renames: Dict[ir.Expr, ir.Expr] = {}
    for idx, obs_sym in enumerate(sysir.observable_symbols):
        obs_renames[obs_sym] = ir.sym(
            f"_cubie_codegen_aux_{idx + 1}"
        )
    subs_map.update(obs_renames)
    if state_is_increment:
        subs_map.update(_state_increment_subs(sysir))

    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        operation_ordering=operation_ordering,
    )

    h_sym = ir.sym("_cubie_codegen_h")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_eff_factor_sym = ir.sym("_cubie_codegen_h_eff_factor")

    memo: dict = {}
    aux_assignments: List[Tuple[ir.Expr, ir.Expr]] = [
        (
            ir.xreplace(lhs, subs_map, memo),
            ir.xreplace(rhs, subs_map, memo),
        )
        for lhs, rhs in sysir.equations
    ]

    # The preconditioner half needs the diagonal only unless a
    # Neumann member applies the full matrix.
    needs_full_matrix = "neumann" in member_types

    def entry_assignments(diagonal_only):
        entries: List[Tuple[ir.Expr, ir.Expr]] = []
        symbols: Dict[Tuple[int, int], ir.Expr] = {}
        for i in range(n):
            for j in range(n):
                if diagonal_only and i != j:
                    continue
                entry = jac[i][j]
                if ir.is_zero(entry):
                    continue
                j_sym = ir.sym(f"_cubie_codegen_j_{i}_{j}")
                entries.append(
                    (j_sym, ir.xreplace(entry, subs_map, memo))
                )
                symbols[(i, j)] = j_sym
        return entries, symbols

    op_entries, op_j_syms = entry_assignments(False)

    pre_assignments: List[Tuple[ir.Expr, ir.Expr]]
    if split:
        pre_entries, pre_j_syms = entry_assignments(
            not needs_full_matrix
        )
        pre_assignments = _scope_tagged(
            aux_assignments + pre_entries
        )
        tag = {
            key: ir.sym(f"_cubie_codegen_p_{sym.name}")
            for key, sym in pre_j_syms.items()
        }
        pre_j_syms = tag
        op_assignments = list(aux_assignments) + op_entries
    else:
        pre_assignments = list(aux_assignments) + op_entries
        pre_j_syms = op_j_syms
        op_assignments = pre_assignments

    h_eff_sym = ir.sym("_cubie_codegen_f_h_eff")
    if needs_full_matrix:
        pre_assignments.append(
            (
                h_eff_sym,
                ir.mul(h_sym, h_eff_factor_sym, a_ij_sym),
            )
        )

    vec: Dict[int, ir.Expr] = {
        i: ir.arr("v", i) for i in range(n)
    }
    matrix_scale = (h_sym, a_ij_sym)
    for member_idx, member in enumerate(member_types):
        if member == "jacobi":
            vec = _single_stage_jacobi(
                pre_assignments,
                vec,
                pre_j_syms,
                mass,
                n,
                member_idx,
                matrix_scale,
            )
        else:
            vec = _single_stage_neumann(
                pre_assignments,
                vec,
                pre_j_syms,
                n,
                member_idx,
                order,
                h_eff_sym,
            )

    tail: List[Tuple[ir.Expr, ir.Expr]] = []
    for i in range(n):
        tail.append((ir.arr("z_out", i), vec[i]))
        mv_terms = [
            ir.mul(mass[i][j], vec[j])
            for j in range(n)
            if not ir.is_zero(mass[i][j])
        ]
        mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
        jv_terms = [
            ir.mul(op_j_syms[(i, j)], vec[j])
            for j in range(n)
            if (i, j) in op_j_syms
        ]
        jv = ir.add(*jv_terms) if jv_terms else ir.ZERO
        tail.append(
            (
                ir.arr("out", i),
                ir.sub(
                    ir.mul(beta_sym, mv),
                    ir.mul(gamma_sym, a_ij_sym, h_sym, jv),
                ),
            )
        )

    if split:
        return _finalise_split_body(
            pre_assignments,
            op_assignments + tail,
            sysir,
            operation_ordering,
            cse,
        )
    return _finalise_body(
        pre_assignments + tail, sysir, operation_ordering, cse
    )


def _prune_and_print(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
    sysir: SystemIR,
    indent: str,
) -> str:
    """Prune to both output arrays and print the fused body."""
    outputs = [
        lhs
        for lhs, _ in assignments
        if isinstance(lhs, ir.Arr) and lhs.name in ("z_out", "out")
    ]
    assignments = prune_unused(
        assignments, output_symbols=outputs
    )
    lines = print_cuda_multiple(
        assignments,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    if not lines:
        lines = ["pass"]
    return "\n".join(indent + ln for ln in lines)


def _finalise_body(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
    sysir: SystemIR,
    operation_ordering: str,
    cse: bool,
    indent: str = "        ",
) -> str:
    """CSE, order, prune to both outputs, and print the fused body."""
    if cse:
        assignments = cse_and_stack(
            assignments,
            operation_ordering=operation_ordering,
        )
    else:
        assignments = topological_sort(
            assignments,
            operation_ordering=operation_ordering,
        )
    return _prune_and_print(assignments, sysir, indent)


def _finalise_split_body(
    pre_assignments: List[Tuple[ir.Expr, ir.Expr]],
    op_assignments: List[Tuple[ir.Expr, ir.Expr]],
    sysir: SystemIR,
    operation_ordering: str,
    cse: bool,
    indent: str = "        ",
) -> str:
    """CSE each half separately, order the whole graph, and print.

    The halves share only the scalar ``z`` interface (and read-only
    inputs); the final emission ordering runs over the concatenated
    graph so the policy still sees the whole fused DAG.
    """
    if cse:
        pre_assignments = cse_and_stack(
            pre_assignments,
            symbol="_cubie_codegen_pcse",
            operation_ordering=operation_ordering,
        )
        op_assignments = cse_and_stack(
            op_assignments,
            operation_ordering=operation_ordering,
        )
    assignments = topological_sort(
        list(pre_assignments) + list(op_assignments),
        operation_ordering=operation_ordering,
    )
    return _prune_and_print(assignments, sysir, indent)


def _instantiate_jvp_direction(
    jvp_equations: JVPEquations,
    direction: Dict[int, ir.Expr],
    n: int,
    tag: str,
    base_assignments: List[Tuple[ir.Expr, ir.Expr]],
) -> Dict[int, ir.Expr]:
    """Instantiate the cached JVP graph for one direction vector.

    v-independent assignments are shared (emitted once into
    ``base_assignments`` by the caller); v-dependent assignments are
    renamed with ``tag`` and rewritten to read the supplied direction
    expressions instead of ``v``.

    Returns
    -------
    dict
        Output-index to JVP-value expression map.
    """
    v_dependent = jvp_equations.v_dependent_nodes
    subs: Dict[ir.Expr, ir.Expr] = {
        ir.arr("v", j): direction.get(j, ir.ZERO)
        for j in range(n)
    }
    for lhs in jvp_equations.non_jvp_order:
        if lhs in v_dependent:
            subs[lhs] = ir.sym(
                f"_cubie_codegen_{tag}_{lhs.name}"
            )
    memo: dict = {}
    for lhs in jvp_equations.non_jvp_order:
        if lhs not in v_dependent:
            continue
        rhs = jvp_equations.non_jvp_exprs[lhs]
        base_assignments.append(
            (subs[lhs], ir.xreplace(rhs, subs, memo))
        )
    return {
        idx: ir.xreplace(term, subs, memo)
        for idx, term in jvp_equations.jvp_terms.items()
    }


def _build_cached_fused_body(
    equations: ParsedEquations,
    index_map: IndexedBases,
    sysir: SystemIR,
    member_types: Sequence[str],
    mass: List[List[ir.Expr]],
    order: int,
    jvp_equations: JVPEquations,
    operation_ordering: str,
    cse: bool,
) -> str:
    """Build the cached (Rosenbrock-W) fused body.

    Auxiliaries come from the shared ``cached_aux`` buffer; the slot
    layout is the one ``prepare_jac`` fills, so the fused helper stays
    positionally compatible with the unfused cached family.
    """
    n = len(sysir.state_symbols)
    cached_aux, runtime_aux, _ = jvp_equations.cached_partition()

    assignments: List[Tuple[ir.Expr, ir.Expr]] = [
        (lhs, ir.arr("cached_aux", idx))
        for idx, (lhs, _) in enumerate(cached_aux)
    ]
    v_dependent = jvp_equations.v_dependent_nodes
    assignments.extend(
        (lhs, rhs)
        for lhs, rhs in runtime_aux
        if lhs not in v_dependent
    )

    h_sym = ir.sym("_cubie_codegen_h")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_eff_factor_sym = ir.sym("_cubie_codegen_h_eff_factor")
    h_eff_sym = ir.sym("_cubie_codegen_f_h_eff")
    if "neumann" in member_types:
        assignments.append(
            (
                h_eff_sym,
                ir.mul(h_sym, h_eff_factor_sym, a_ij_sym),
            )
        )

    diag_exprs: Optional[Dict[int, ir.Expr]] = None
    if "jacobi" in member_types:
        # Bind every auxiliary the diagonal can reference, exactly as
        # the unfused cached Jacobi body does.
        jac = generate_jacobian(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            operation_ordering=operation_ordering,
        )
        cached_slots = {
            lhs: idx for idx, (lhs, _) in enumerate(cached_aux)
        }
        runtime_symbols = {lhs for lhs, _ in runtime_aux}
        subs_memo: dict = {}
        aux_subs: Dict[ir.Expr, ir.Expr] = {}
        for lhs in jvp_equations.non_jvp_order:
            slot = cached_slots.get(lhs)
            if slot is not None:
                aux_subs[lhs] = ir.arr("cached_aux", slot)
            elif lhs in runtime_symbols:
                aux_subs[lhs] = jvp_equations.non_jvp_exprs[lhs]
            else:
                aux_subs[lhs] = ir.xreplace(
                    jvp_equations.non_jvp_exprs[lhs],
                    aux_subs,
                    subs_memo,
                )
        obs_renames = {
            obs_sym: ir.sym(f"_cubie_codegen_aux_{idx + 1}")
            for idx, obs_sym in enumerate(
                sysir.observable_symbols
            )
        }
        memo: dict = {}
        diag_exprs = {}
        for i in range(n):
            j_ii = ir.xreplace(jac[i][i], obs_renames, memo)
            diag_exprs[i] = ir.xreplace(j_ii, aux_subs)

    vec: Dict[int, ir.Expr] = {
        i: ir.arr("v", i) for i in range(n)
    }
    application = 0
    for member_idx, member in enumerate(member_types):
        if member == "jacobi":
            out: Dict[int, ir.Expr] = {}
            for i in range(n):
                diag_sym = ir.sym(
                    f"_cubie_codegen_fdiag{member_idx}_{i}"
                )
                diag_val = ir.sub(
                    _mass_diag_term(mass, i, beta_sym),
                    ir.mul(
                        gamma_sym,
                        h_sym,
                        a_ij_sym,
                        diag_exprs[i],
                    ),
                )
                assignments.append((diag_sym, diag_val))
                safe_sym, guarded = _guarded_diag_division(
                    diag_sym, i, stage_idx=f"fused{member_idx}"
                )
                assignments.append((safe_sym, guarded))
                out[i] = ir.div(vec[i], safe_sym)
            vec = _named_vector(
                assignments,
                out,
                n,
                f"_cubie_codegen_fz{member_idx}",
            )
        else:
            beta_inv_sym = ir.sym("_cubie_codegen_beta_inv")
            acc = dict(vec)
            for k in range(order):
                jv = _instantiate_jvp_direction(
                    jvp_equations,
                    acc,
                    n,
                    f"fa{application}",
                    assignments,
                )
                application += 1
                jv = _named_vector(
                    assignments,
                    jv,
                    n,
                    f"_cubie_codegen_fjv{member_idx}_{k}",
                )
                nxt = {
                    i: ir.add(
                        vec[i], ir.mul(h_eff_sym, jv[i])
                    )
                    for i in range(n)
                }
                acc = _named_vector(
                    assignments,
                    nxt,
                    n,
                    f"_cubie_codegen_facc{member_idx}_{k}",
                )
            scaled = {
                i: ir.mul(beta_inv_sym, acc[i])
                for i in range(n)
            }
            vec = _named_vector(
                assignments,
                scaled,
                n,
                f"_cubie_codegen_fz{member_idx}",
            )

    operator_jv = _instantiate_jvp_direction(
        jvp_equations, vec, n, f"fa{application}", assignments
    )
    for i in range(n):
        assignments.append((ir.arr("z_out", i), vec[i]))
        mv_terms = [
            ir.mul(mass[i][j], vec[j])
            for j in range(n)
            if not ir.is_zero(mass[i][j])
        ]
        mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
        assignments.append(
            (
                ir.arr("out", i),
                ir.sub(
                    ir.mul(beta_sym, mv),
                    ir.mul(
                        gamma_sym,
                        a_ij_sym,
                        h_sym,
                        operator_jv.get(i, ir.ZERO),
                    ),
                ),
            )
        )

    return _finalise_body(
        assignments, sysir, operation_ordering, cse
    )


def _build_n_stage_fused_body(
    equations: ParsedEquations,
    index_map: IndexedBases,
    sysir: SystemIR,
    member_types: Sequence[str],
    mass: List[List[ir.Expr]],
    order: int,
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    operation_ordering: str,
    cse: bool,
) -> str:
    """Build the flattened all-stages FIRK fused body.

    Stage block ``s`` of both applications is
    ``J(Y_s) @ (sum_j a[s][j] * x[j*n:i])`` — the same Kronecker
    convention as the unfused n-stage operator and preconditioner.
    """
    n = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)
    metadata_exprs, coeff_symbols, node_symbols = (
        build_stage_metadata(stage_coefficients, stage_nodes)
    )

    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        operation_ordering=operation_ordering,
    )

    h_sym = ir.sym("_cubie_codegen_h")
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_eff_factor_sym = ir.sym("_cubie_codegen_h_eff_factor")
    split = FUSED_CSE_SCOPE == "split"
    needs_full_matrix = "neumann" in member_types

    def stage_material(diagonal_only):
        """Stage equations plus the requested Jacobian entries."""
        material: List[Tuple[ir.Expr, ir.Expr]] = []
        symbols: Dict[Tuple[int, int, int], ir.Expr] = {}
        for stage_idx in range(stage_count):
            subs_map = build_stage_substitutions(
                sysir,
                stage_idx,
                coeff_symbols,
                node_symbols,
                stage_coefficients,
                state_vector_name="state",
            )
            memo: dict = {}
            material.extend(
                (
                    ir.xreplace(lhs, subs_map, memo),
                    ir.xreplace(rhs, subs_map, memo),
                )
                for lhs, rhs in sysir.equations
            )
            for i in range(n):
                for j in range(n):
                    if diagonal_only and i != j:
                        continue
                    entry = jac[i][j]
                    if ir.is_zero(entry):
                        continue
                    j_sym = ir.sym(
                        f"_cubie_codegen_j_{stage_idx}_{i}_{j}"
                    )
                    material.append(
                        (j_sym, ir.xreplace(entry, subs_map, memo))
                    )
                    symbols[(stage_idx, i, j)] = j_sym
        return material, symbols

    op_material, op_j_syms = stage_material(False)

    if split:
        pre_material, pre_j_syms = stage_material(
            not needs_full_matrix
        )
        # Tableau metadata assignments live in the operator half
        # only; the preconditioner half reads them as free symbols.
        pre_assignments = _scope_tagged(pre_material)
        pre_j_syms = {
            key: ir.sym(f"_cubie_codegen_p_{sym.name}")
            for key, sym in pre_j_syms.items()
        }
        op_assignments = list(metadata_exprs) + op_material
    else:
        pre_assignments = list(metadata_exprs) + op_material
        pre_j_syms = op_j_syms
        op_assignments = pre_assignments

    h_eff_sym = ir.sym("_cubie_codegen_f_h_eff")
    if needs_full_matrix:
        pre_assignments.append(
            (h_eff_sym, ir.mul(h_sym, h_eff_factor_sym))
        )

    def stage_coupled_jv(
        source: Dict[int, ir.Expr],
        j_symbols: Dict[Tuple[int, int, int], ir.Expr],
    ) -> Dict[int, ir.Expr]:
        """Return ``(A x J) @ source`` over the flat index space."""
        result: Dict[int, ir.Expr] = {}
        for stage_idx in range(stage_count):
            combo: Dict[int, ir.Expr] = {}
            for comp in range(n):
                terms = [
                    ir.mul(
                        coeff_symbols[stage_idx][contrib],
                        source[contrib * n + comp],
                    )
                    for contrib in range(stage_count)
                    if not ir.is_zero(
                        stage_coefficients[stage_idx][contrib]
                    )
                ]
                combo[comp] = (
                    ir.add(*terms) if terms else ir.ZERO
                )
            for i in range(n):
                terms = [
                    ir.mul(
                        j_symbols[(stage_idx, i, j)], combo[j]
                    )
                    for j in range(n)
                    if (stage_idx, i, j) in j_symbols
                ]
                result[stage_idx * n + i] = (
                    ir.add(*terms) if terms else ir.ZERO
                )
        return result

    total = stage_count * n
    vec: Dict[int, ir.Expr] = {
        k: ir.arr("v", k) for k in range(total)
    }
    for member_idx, member in enumerate(member_types):
        if member == "jacobi":
            out: Dict[int, ir.Expr] = {}
            for stage_idx in range(stage_count):
                diag_coeff = coeff_symbols[stage_idx][stage_idx]
                for comp in range(n):
                    flat = stage_idx * n + comp
                    j_ii = pre_j_syms.get(
                        (stage_idx, comp, comp), ir.ZERO
                    )
                    diag_sym = ir.sym(
                        "_cubie_codegen_fdiag"
                        f"{member_idx}_{stage_idx}_{comp}"
                    )
                    diag_val = ir.sub(
                        _mass_diag_term(mass, comp, beta_sym),
                        ir.mul(
                            gamma_sym, h_sym, diag_coeff, j_ii
                        ),
                    )
                    pre_assignments.append((diag_sym, diag_val))
                    safe_sym, guarded = _guarded_diag_division(
                        diag_sym,
                        comp,
                        stage_idx=(
                            f"fused{member_idx}_{stage_idx}"
                        ),
                    )
                    pre_assignments.append((safe_sym, guarded))
                    out[flat] = ir.div(vec[flat], safe_sym)
            vec = _named_vector(
                pre_assignments,
                out,
                total,
                f"_cubie_codegen_fz{member_idx}",
            )
        else:
            beta_inv_sym = ir.sym("_cubie_codegen_beta_inv")
            acc = dict(vec)
            for k in range(order):
                jv = stage_coupled_jv(acc, pre_j_syms)
                jv = _named_vector(
                    pre_assignments,
                    jv,
                    total,
                    f"_cubie_codegen_fjv{member_idx}_{k}",
                )
                nxt = {
                    idx: ir.add(
                        vec[idx], ir.mul(h_eff_sym, jv[idx])
                    )
                    for idx in range(total)
                }
                acc = _named_vector(
                    pre_assignments,
                    nxt,
                    total,
                    f"_cubie_codegen_facc{member_idx}_{k}",
                )
            scaled = {
                idx: ir.mul(beta_inv_sym, acc[idx])
                for idx in range(total)
            }
            vec = _named_vector(
                pre_assignments,
                scaled,
                total,
                f"_cubie_codegen_fz{member_idx}",
            )

    operator_jv = stage_coupled_jv(vec, op_j_syms)
    tail: List[Tuple[ir.Expr, ir.Expr]] = []
    for stage_idx in range(stage_count):
        for comp in range(n):
            flat = stage_idx * n + comp
            tail.append(
                (ir.arr("z_out", flat), vec[flat])
            )
            mv_terms = [
                ir.mul(mass[comp][col], vec[stage_idx * n + col])
                for col in range(n)
                if not ir.is_zero(mass[comp][col])
            ]
            mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
            tail.append(
                (
                    ir.arr("out", flat),
                    ir.sub(
                        ir.mul(beta_sym, mv),
                        ir.mul(
                            gamma_sym,
                            h_sym,
                            operator_jv[flat],
                        ),
                    ),
                )
            )

    if split:
        return _finalise_split_body(
            pre_assignments,
            op_assignments + tail,
            sysir,
            operation_ordering,
            cse,
        )
    return _finalise_body(
        pre_assignments + tail, sysir, operation_ordering, cse
    )


def generate_fused_operator_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    preconditioner_members: Sequence,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "fused_operator_factory",
    cse: bool = True,
    order: int = 1,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the Newton--Krylov fused operator-preconditioner.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    preconditioner_members
        Concrete preconditioner kinds applied in order before the
        operator.
    M
        Mass matrix; identity when omitted.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    order
        Neumann truncation order baked into the unrolled source.
    jvp_equations
        Unused; accepted for signature parity with the operator
        generators.
    operation_ordering
        Dependency ordering policy for the fused graph.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    default_timelogger.start_event(
        "codegen_generate_fused_operator_code"
    )
    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    member_types = [
        _member_type(member) for member in preconditioner_members
    ]
    body = _build_single_stage_fused_body(
        equations,
        index_map,
        sysir,
        member_types,
        mass,
        order,
        operation_ordering,
        state_is_increment=True,
        cse=cse,
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    result = FUSED_OPERATOR_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    default_timelogger.stop_event(
        "codegen_generate_fused_operator_code"
    )
    return result


def generate_fused_operator_at_state_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    preconditioner_members: Sequence,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "fused_operator_at_state_factory",
    cse: bool = True,
    order: int = 1,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the at-state fused operator-preconditioner.

    The Jacobian evaluates at the ``state`` argument; ``a_ij`` scales
    the matrix only. Parameters match
    :func:`generate_fused_operator_code`.
    """
    default_timelogger.start_event(
        "codegen_generate_fused_operator_at_state_code"
    )
    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    member_types = [
        _member_type(member) for member in preconditioner_members
    ]
    body = _build_single_stage_fused_body(
        equations,
        index_map,
        sysir,
        member_types,
        mass,
        order,
        operation_ordering,
        state_is_increment=False,
        cse=cse,
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    result = FUSED_OPERATOR_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    default_timelogger.stop_event(
        "codegen_generate_fused_operator_at_state_code"
    )
    return result


def generate_fused_operator_cached_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    preconditioner_members: Sequence,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "fused_operator_cached_factory",
    cse: bool = True,
    order: int = 1,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the cached (Rosenbrock-W) fused helper.

    Reads the ``cached_aux`` buffer with the slot layout
    ``prepare_jac`` fills. Parameters match
    :func:`generate_fused_operator_code`; ``jvp_equations`` supplies
    the shared cache selection when prebuilt.
    """
    default_timelogger.start_event(
        "codegen_generate_fused_operator_cached_code"
    )
    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    member_types = [
        _member_type(member) for member in preconditioner_members
    ]
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    body = _build_cached_fused_body(
        equations,
        index_map,
        sysir,
        member_types,
        mass,
        order,
        jvp_equations,
        operation_ordering,
        cse=cse,
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    result = FUSED_OPERATOR_CACHED_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    default_timelogger.stop_event(
        "codegen_generate_fused_operator_cached_code"
    )
    return result


def generate_n_stage_fused_operator_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    preconditioner_members: Sequence,
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
    stage_nodes: Sequence[Union[float, object]],
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "n_stage_fused_operator_factory",
    cse: bool = True,
    order: int = 1,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the flattened n-stage FIRK fused helper.

    Parameters match :func:`generate_fused_operator_code` plus the
    Butcher ``stage_coefficients`` / ``stage_nodes``.
    """
    default_timelogger.start_event(
        "codegen_generate_n_stage_fused_operator_code"
    )
    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    member_types = [
        _member_type(member) for member in preconditioner_members
    ]
    body = _build_n_stage_fused_body(
        equations,
        index_map,
        sysir,
        member_types,
        mass,
        order,
        coeff_matrix,
        node_values,
        operation_ordering,
        cse=cse,
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    result = N_STAGE_FUSED_OPERATOR_TEMPLATE.format(
        func_name=func_name,
        body=body,
        const_lines=const_block,
        stage_count=stage_count,
    )
    default_timelogger.stop_event(
        "codegen_generate_n_stage_fused_operator_code"
    )
    return result


__all__ = [
    "generate_fused_operator_code",
    "generate_fused_operator_at_state_code",
    "generate_fused_operator_cached_code",
    "generate_n_stage_fused_operator_code",
]
