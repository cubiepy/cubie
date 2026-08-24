from hashlib import sha256

import numpy as np
import pytest
import sympy as sp

from cubie.cuda_simsafe import cuda
from cubie.cuda_simsafe import numba_from_dtype as from_dtype
from cubie.memory import default_memmgr
from cubie.odesystems.solver_helpers import (
    HelperVariant,
    SolverHelperRequest,
)
from cubie.odesystems.symbolic.codegen import (
    generate_jacobi_preconditioner_code,
    generate_linear_operator_code,
    generate_neumann_preconditioner_code,
    generate_prepare_jac_code,
    generate_residual_code,
)
from cubie.odesystems.symbolic.engine import convert_assignments
from cubie.odesystems.symbolic.engine import expr as ir_expr
from cubie.odesystems.symbolic.helper_registry import (
    ApplyMass,
    InitLuSolve,
    InitResidual,
    LinearOperator,
    LuPrepareBlocks,
    LuSmoothingSolve,
    LuSolve,
    PrepareJac,
    Residual,
    helper_source_hash,
)
from cubie.odesystems.symbolic.parsing import (
    JVPEquations as _JVPEquations,
)
from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
    plan_auxiliary_cache,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from tests.system_fixtures import (
    build_diode_line_system,
    build_transistor_amplifier_system,
)
from tests._utils import (
    COLLIDING_CONSTANTS_F32,
    COLLIDING_CONSTANTS_F64,
    FLOAT64_PRECISION,
    HODGKIN_HUXLEY_SYSTEM,
    LINEAR_SYSTEM,
)


def JVPEquations(exprs, **kwargs):
    """Build JVPEquations from SymPy pairs via IR conversion."""
    return _JVPEquations(convert_assignments(exprs), **kwargs)


def _ir(symbol):
    """Return the IR symbol matching a SymPy symbol's name."""
    return ir_expr.sym(str(symbol))


def _stable_factory_tag(*values):
    """Return a stable short tag for generated factory names."""
    digest = sha256()
    for value in values:
        encoded = value if isinstance(value, bytes) else repr(value).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


@pytest.fixture(scope="session")
def operator_system(precision):
    """Build a linear system with a constant Jacobian."""

    dxdt = [
        "dx0 = a*x0 + b*x1",
        "dx1 = c*x0 + d*x1",
    ]
    constants = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    system = create_ODE_system(
        dxdt, states=["x0", "x1"], constants=constants, precision=precision
    )
    return system


def _build_operator_factory(system, precision):
    def factory(beta, gamma, M):
        fname = (
            "operator_apply_factory_"
            f"{_stable_factory_tag(beta, gamma, M.tobytes())}"
        )
        code = generate_linear_operator_code(
            system.equations,
            system.indices,
            M=M,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        op_fac, was_cached = system.gen_file.import_function(fname, code)
        return op_fac(from_dtype(system.precision))

    return factory


@pytest.fixture(scope="session")
def operator_factory(operator_system, precision):
    """Return a factory producing operator_apply device functions."""

    return _build_operator_factory(operator_system, precision)


@pytest.fixture(scope="session")
def operator_kernel(precision):
    """Kernel applying operator_apply to a vector."""

    n = 2

    def make_kernel(op):
        @cuda.jit
        def kernel(t, h, a_ij, vec, base_state, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            cached_aux = cuda.local.array(1, precision)
            # base_state is provided by caller (can be empty placeholder)
            op(
                state, parameters, drivers, cached_aux, base_state,
                t, h, a_ij, vec, out,
            )

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def cached_system():
    """Build a nonlinear system with state-dependent Jacobian."""

    dxdt = [
        "dx0 = a*x0*x1 + b*sin(x0)",
        "dx1 = c*x0*x1 + d*cos(x1)",
    ]
    constants = {"a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9}
    system = create_ODE_system(dxdt, states=["x0", "x1"], constants=constants)
    return system


@pytest.fixture(scope="session")
def prepare_jac_factory(cached_system, precision):
    """Return a factory producing prepare_jac device functions."""

    def factory():
        fname = "prepare_jac_factory"
        code, aux_count = generate_prepare_jac_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        prep_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        prepare = prep_fac(from_dtype(cached_system.precision))
        return prepare, aux_count

    return factory


@pytest.fixture(scope="session")
def cached_operator_factory(cached_system, precision):
    """Return a factory producing cached operator device functions."""

    def factory(beta, gamma, M):
        fname = (
            "cached_operator_factory_"
            f"{_stable_factory_tag(beta, gamma, M.tobytes())}"
        )
        code = generate_linear_operator_code(
            cached_system.equations,
            cached_system.indices,
            variant=HelperVariant.CACHED,
            M=M,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        op_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return op_fac(from_dtype(cached_system.precision))

    return factory


@pytest.fixture(scope="session")
def cached_operator_kernel(cached_system, precision):
    """Kernel applying cached operator to a vector."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, op, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            h,
            a_ij,
            vec,
            base_state,
            out,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, h, cached_aux)
            op(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
            )

        return kernel

    return make_kernel


def test_split_jvp_expressions_caches_high_cost_terms():
    """Cache the expression removing the largest runtime operation count."""

    x0, x1 = sp.symbols("x0 x1")
    dep0 = sp.Symbol("dep0")
    heavy = sp.Symbol("aux_heavy")
    simple = sp.Symbol("simple")
    j_00 = sp.Symbol("j_00")

    exprs = [
        (
            dep0,
            sp.sin(x0) + sp.cos(x1),
        ),
        (
            heavy,
            dep0**3
            + sp.exp(dep0)
            + sp.tan(dep0)
            + sp.log(dep0 + 2)
            + dep0 * sp.sinh(dep0),
        ),
        (simple, x0 + x1),
        (j_00, heavy + simple),
        (j_01 := sp.Symbol("j_01"), simple),
        (
            sp.Symbol("jvp[0]"),
            j_00 * sp.Symbol("v[0]") + j_01 * sp.Symbol("v[1]"),
        ),
    ]

    equations = JVPEquations(exprs)
    dep0, heavy, j_00, j_01 = (
        _ir(dep0), _ir(heavy), _ir(j_00), _ir(j_01),
    )
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = [lhs for lhs, _ in prepare_assigns]

    assert cached_symbols == [j_00]
    assert list(selection.cached_leaf_order) == [j_00]
    assert heavy not in runtime_symbols
    assert heavy in prepare_symbols
    assert dep0 not in runtime_symbols
    assert dep0 in prepare_symbols
    assert equations.jvp_terms[0] is ir_expr.add(
        ir_expr.mul(j_00, ir_expr.arr("v", 0)),
        ir_expr.mul(j_01, ir_expr.arr("v", 1)),
    )


def test_canonical_cached_views_cover_the_graph():
    """Slot bindings plus graph expressions cover every non-JVP name."""

    x0, x1 = sp.symbols("x0 x1")
    dep0 = sp.Symbol("dep0")
    heavy = sp.Symbol("aux_heavy")
    simple = sp.Symbol("simple")
    j_00 = sp.Symbol("j_00")
    j_01 = sp.Symbol("j_01")

    exprs = [
        (dep0, sp.sin(x0) + sp.cos(x1)),
        (
            heavy,
            dep0**3
            + sp.exp(dep0)
            + sp.tan(dep0)
            + sp.log(dep0 + 2)
            + dep0 * sp.sinh(dep0),
        ),
        (simple, x0 + x1),
        (j_00, heavy + simple),
        (j_01, simple),
        (
            sp.Symbol("jvp[0]"),
            j_00 * sp.Symbol("v[0]") + j_01 * sp.Symbol("v[1]"),
        ),
    ]

    equations = JVPEquations(exprs)
    slots = equations.cached_slot_order
    assert list(slots) == [_ir(j_00)]

    runtime_set = equations.cached_runtime_assignments()
    assert [lhs for lhs, _ in runtime_set] == list(
        equations.non_jvp_order
    )
    bound = dict(runtime_set)
    for idx, symbol in enumerate(slots):
        assert bound[symbol] is ir_expr.arr("cached_aux", idx)
    for lhs in equations.non_jvp_order:
        if lhs in set(slots):
            continue
        assert bound[lhs] is equations.non_jvp_exprs[lhs]

    fill = equations.prepare_fill_assignments()
    stores = {
        lhs.index: rhs
        for lhs, rhs in fill
        if isinstance(lhs, ir_expr.Arr) and lhs.name == "cached_aux"
    }
    assert [stores[idx] for idx in range(len(slots))] == list(slots)
    prepare_symbols = {
        lhs for lhs, _ in equations.cached_partition()[2]
    }
    for lhs, rhs in fill:
        if isinstance(lhs, ir_expr.Arr):
            continue
        assert lhs in prepare_symbols
        assert rhs is equations.non_jvp_exprs[lhs]


def test_split_jvp_expressions_limits_cache_size():
    """Limit cached expressions to twice the output dimension."""

    x = sp.symbols("x")
    heavy_symbols = [sp.Symbol(f"aux_heavy{i}") for i in range(3)]
    heavy_exprs = [
        (
            sp.sin(x)
            + sp.cos(x)
            + sp.exp(x)
            + sp.log(x + 2)
            + sp.tan(x)
            + sp.sinh(x)
        ),
        (
            sp.sin(2 * x)
            + sp.cos(2 * x)
            + sp.exp(2 * x)
            + sp.log(x + 3)
            + sp.tan(2 * x)
            + sp.sinh(2 * x)
            + x**2
        ),
        (
            sp.sin(3 * x)
            + sp.cos(3 * x)
            + sp.exp(3 * x)
            + sp.log(x + 4)
            + sp.tan(3 * x)
            + sp.sinh(3 * x)
            + x**3
            + sp.sqrt(x + 1)
        ),
    ]

    exprs = list(zip(heavy_symbols, heavy_exprs))
    j_00 = sp.Symbol("j_00")
    exprs.append((j_00, sum(heavy_symbols)))
    exprs.append((sp.Symbol("jvp[0]"), j_00 * sp.Symbol("v[0]")))

    equations = JVPEquations(exprs)
    cached_aux, runtime_aux, _ = equations.cached_partition()
    selection = equations.cache_selection

    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]

    assert cached_symbols == [_ir(j_00)]
    assert list(selection.cached_leaf_order) == [_ir(j_00)]
    assert all(
        _ir(sym) not in runtime_symbols for sym in heavy_symbols
    )


def test_split_jvp_expressions_groups_cse_dependents():
    """Cache dependents sharing a CSE prerequisite as a single group."""

    x0, x1 = sp.symbols("x0 x1")
    cse_sym = sp.Symbol("_cse0")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    jac = sp.Symbol("j_00")

    exprs = [
        (
            cse_sym,
            sp.sin(x0) + sp.cos(x1) + sp.exp(x0 + x1) + sp.log(x0 + 3),
        ),
        (
            aux_a,
            cse_sym**2
            + sp.exp(cse_sym)
            + sp.sin(cse_sym)
            + sp.tan(cse_sym)
            + sp.log(cse_sym + 2),
        ),
        (
            aux_b,
            cse_sym**3
            + sp.cos(cse_sym)
            + sp.sinh(cse_sym)
            + sp.acos(sp.tanh(x0))
            + sp.atan(cse_sym + 1),
        ),
        (jac, aux_a + aux_b),
        (sp.Symbol("jvp[0]"), jac * sp.Symbol("v[0]")),
    ]

    equations = JVPEquations(exprs, read_price=5)
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    cse_sym, aux_a, aux_b, jac = (
        _ir(cse_sym), _ir(aux_a), _ir(aux_b), _ir(jac),
    )
    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = [lhs for lhs, _ in prepare_assigns]

    assert cached_symbols == [jac]
    assert list(selection.cached_leaf_order) == [jac]
    # jac's whole support moves into the prepare fill.
    assert cse_sym in prepare_symbols
    assert aux_a in prepare_symbols
    assert aux_b in prepare_symbols
    assert runtime_symbols == []
    assert equations.jvp_terms[0] is ir_expr.mul(
        jac, ir_expr.arr("v", 0)
    )


def test_split_jvp_expressions_limits_cse_depth_for_slots():
    """A one-slot budget still absorbs the whole dead support chain."""

    x0, x1 = sp.symbols("x0 x1")
    cse_root = sp.Symbol("_cse0")
    cse_mid = sp.Symbol("_cse1")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    aux_c = sp.Symbol("aux_c")
    jac = sp.Symbol("j_00")

    exprs = [
        (
            cse_root,
            sp.sin(x0) + sp.cos(x1) + sp.exp(x0 + x1),
        ),
        (
            cse_mid,
            cse_root**2 + sp.exp(cse_root) + sp.tan(cse_root),
        ),
        (
            aux_a,
            cse_mid**2 + sp.sin(cse_mid) + sp.log(cse_mid + 2),
        ),
        (
            aux_b,
            cse_mid**3
            + sp.exp(cse_mid)
            + sp.sinh(cse_mid)
            + sp.atan(cse_mid + 1)
            + sp.sqrt(cse_mid + 3),
        ),
        (aux_c, x0 + x1),
        (jac, aux_a + aux_b + aux_c),
        (sp.Symbol("jvp[0]"), jac * sp.Symbol("v[0]")),
    ]

    equations = JVPEquations(
        exprs,
        max_cached_terms=1,
        read_price=1,
    )
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    cse_root, cse_mid, aux_a, aux_b, aux_c, jac = (
        _ir(cse_root), _ir(cse_mid), _ir(aux_a),
        _ir(aux_b), _ir(aux_c), _ir(jac),
    )
    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = {lhs for lhs, _ in prepare_assigns}

    assert cached_symbols == [jac]
    assert list(selection.cached_leaf_order) == [jac]
    assert aux_a not in runtime_symbols
    assert aux_b not in runtime_symbols
    assert aux_c not in runtime_symbols
    assert cse_mid in prepare_symbols
    # The _cse chain feeds only the cached leaf and leaves with it.
    assert cse_root in prepare_symbols
    assert cse_root not in runtime_symbols


def test_cache_plan_shared_cse_with_slot_limit():
    """Ensure shared CSE branches remain available with cache limits."""

    x0, x1 = sp.symbols("x0 x1")
    cse_sym = sp.Symbol("_cse_shared")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    jac_a = sp.Symbol("j_00")
    jac_b = sp.Symbol("j_01")

    exprs = [
        (
            cse_sym,
            sp.sin(x0) + sp.cos(x1) + sp.exp(x0 + x1) + sp.log(x0 + 2),
        ),
        (
            aux_a,
            cse_sym**2
            + sp.sin(cse_sym)
            + sp.tan(cse_sym)
            + sp.log(cse_sym + 3),
        ),
        (
            aux_b,
            cse_sym**3
            + sp.cos(cse_sym)
            + sp.sinh(cse_sym)
            + sp.log(cse_sym + 4),
        ),
        (
            jac_a,
            aux_a + sp.exp(cse_sym) + sp.sin(aux_a),
        ),
        (
            jac_b,
            aux_b + sp.tanh(cse_sym) + sp.cos(aux_b),
        ),
        (sp.Symbol("jvp[0]"), jac_a * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[1]"), jac_b * sp.Symbol("v[1]")),
    ]

    equations = JVPEquations(
        exprs,
        max_cached_terms=1,
        read_price=1,
    )
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    assert len(selection.cached_leaf_order) == 1
    cached_leaf = selection.cached_leaf_order[0]
    cse_sym, aux_a, aux_b, jac_a, jac_b = (
        _ir(cse_sym), _ir(aux_a), _ir(aux_b),
        _ir(jac_a), _ir(jac_b),
    )
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = [lhs for lhs, _ in prepare_assigns]

    assert cse_sym in runtime_symbols
    assert cse_sym in prepare_symbols

    if cached_leaf == jac_a:
        assert jac_b in runtime_symbols
        assert aux_b in runtime_symbols
        assert aux_a in prepare_symbols
    else:
        assert jac_a in runtime_symbols
        assert aux_a in runtime_symbols
        assert aux_b in prepare_symbols

    remaining_leaf = jac_b if cached_leaf == jac_a else jac_a
    assert remaining_leaf in runtime_symbols


def test_build_expression_costs_tracks_jvp_dependencies():
    """Propagate JVP usage counts through dependency closures."""

    x0, x1 = sp.symbols("x0 x1")
    dep0 = sp.Symbol("dep0")
    heavy = sp.Symbol("aux_heavy")
    simple = sp.Symbol("simple")
    j_00 = sp.Symbol("j_00")

    non_jvp_order = [dep0, heavy, simple, j_00]
    non_jvp_exprs = {
        dep0: sp.sin(x0) + sp.cos(x1),
        heavy: dep0**2 + sp.exp(dep0),
        simple: x0 + x1,
        j_00: heavy + simple,
    }
    jvp_terms = {0: j_00 * sp.Symbol("v[0]")}

    exprs = [(sym, non_jvp_exprs[sym]) for sym in non_jvp_order]
    exprs.append((sp.Symbol("jvp[0]"), jvp_terms[0]))
    equations = JVPEquations(exprs)

    dep0, heavy, simple, j_00 = (
        _ir(dep0), _ir(heavy), _ir(simple), _ir(j_00),
    )
    assert equations.jvp_usage == {j_00: 1}
    assert equations.jvp_closure_usage[j_00] == 1
    assert equations.jvp_closure_usage[heavy] == 1
    assert equations.jvp_closure_usage[dep0] == 1
    assert equations.jvp_closure_usage[simple] == 1


def test_equations_track_costs_and_v_dependence():
    """Track cumulative costs and direction-vector dependence."""

    x0, x1 = sp.symbols("x0 x1")
    seed = sp.Symbol("cse1")
    branch_a = sp.Symbol("cse7")
    branch_b = sp.Symbol("cse10")
    j_00 = sp.Symbol("j_00")
    j_20 = sp.Symbol("j_20")
    j_22 = sp.Symbol("j_22")
    j_02 = sp.Symbol("j_02")
    assignments = [
        (seed, x0 + x1),
        (branch_a, seed + x0),
        (branch_b, seed * x1),
        (j_00, branch_a + x0),
        (j_20, branch_a + x1),
        (j_22, branch_b + x0),
        (j_02, branch_b + x1),
        (sp.Symbol("jvp[0]"), j_00 * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[1]"), j_20 * sp.Symbol("v[1]")),
        (sp.Symbol("jvp[2]"), j_22 * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[3]"), j_02 * sp.Symbol("v[1]")),
    ]

    equations = JVPEquations(assignments)

    seed, branch_a, branch_b = _ir(seed), _ir(branch_a), _ir(branch_b)
    j_00, j_20, j_22, j_02 = (
        _ir(j_00), _ir(j_20), _ir(j_22), _ir(j_02),
    )
    assert equations.order_index[seed] == 0
    assert equations.total_ops_cost[branch_a] == 2
    assert equations.total_ops_cost[j_00] == 3
    assert equations.total_ops_cost[ir_expr.arr("jvp", 0)] == 4
    # v appears only in the jvp dot products.
    assert equations.v_dependent_nodes == frozenset()


@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, np.diag([1.0, 0.0])),
        (0.5, 2.0, 1.0, np.diag([0.0, 1.0])),
    ],
)
def test_operator_apply_dense(
    beta,
    gamma,
    h,
    M,
    operator_factory,
    operator_kernel,
    precision,
    tolerance,
):
    """Evaluate operator_apply for scalings and 0/1 mass diagonals."""

    op = operator_factory(beta, gamma, M)
    kernel = operator_kernel(op)
    v = np.array([1.0, -1.0], dtype=precision)
    out = np.zeros(2, dtype=precision)
    empty_base = np.empty(0, dtype=precision)
    kernel[1, 1](
        precision(0.0), precision(h), precision(1.0), v, empty_base, out
    )
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    expected = beta * M @ v - gamma * h * J @ v
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_operator_apply_constants_folded(operator_system):
    """Constants are literals in emitted source, never bindings."""
    code = generate_linear_operator_code(
        operator_system.equations, operator_system.indices
    )
    assert "_cubie_codegen_const_" not in code
    assert "constants[" not in code


@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 0.25, np.eye(2)),
        (1.0, 1.0, 0.25, np.diag([1.0, 0.0])),
        (0.5, 1.7, 0.15, np.diag([0.0, 1.0])),
    ],
)
def test_cached_operator_apply_dense(
    beta,
    gamma,
    h,
    M,
    cached_operator_factory,
    cached_operator_kernel,
    cached_system,
    prepare_jac_factory,
    precision,
    tolerance,
):
    """Evaluate cached operator using precomputed auxiliaries."""

    prepare, aux_count = prepare_jac_factory()
    op = cached_operator_factory(beta, gamma, M)
    kernel = cached_operator_kernel(prepare, op, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)

    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.8, -1.1], dtype=precision)
    vec = vec[:state_len]
    out = np.zeros(state_len, dtype=precision)

    empty_base = np.empty(0, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        precision(h),
        precision(1.0),
        vec,
        empty_base,
        out,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    beta_val = precision(beta)
    gamma_val = precision(gamma)
    h_val = precision(h)
    mass = np.array(M, dtype=precision)
    expected = beta_val * mass @ vec - gamma_val * h_val * jacobian @ vec

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


# ---------------------------------------------------------------------------
# Neumann preconditioner expression tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neumann_factory(operator_system, precision):
    """Return a factory producing Neumann preconditioner device functions."""

    def factory(beta, gamma, order):
        fname = (
            f"neumann_preconditioner_factory_{int(beta)}_{int(gamma)}_{order}"
        )
        code = generate_neumann_preconditioner_code(
            operator_system.equations,
            operator_system.indices,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        pre_fac, was_cached = operator_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            from_dtype(operator_system.precision),
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def neumann_kernel(precision):
    """Apply the Neumann preconditioner to a vector."""

    n = 2

    def make_kernel(pre):
        @cuda.jit
        def kernel(t, h, a_ij, vec, base_state, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            cached_aux = cuda.local.array(1, precision)
            jvp = cuda.local.array(n, precision)
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def neumann_cached_factory(cached_system, precision):
    """Return a factory producing cached Neumann preconditioners."""

    def factory(beta, gamma, order):
        fname = (
            "neumann_cached_factory_"
            f"{int(beta * 10)}_{int(gamma * 10)}_{order}"
        )
        code = generate_neumann_preconditioner_code(
            cached_system.equations,
            cached_system.indices,
            variant=HelperVariant.CACHED,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            from_dtype(cached_system.precision),
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def neumann_cached_kernel(cached_system, precision):
    """Apply cached Neumann preconditioner to a vector."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, pre, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            h,
            a_ij,
            vec,
            base_state,
            out,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)
            jvp = cuda.local.array(n_state, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, h, cached_aux)
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel


@pytest.mark.parametrize(
    "solver_settings_override",
    [FLOAT64_PRECISION],
    ids=[""],
    indirect=True,
)
@pytest.mark.parametrize(
    "beta,gamma,h,order",
    [
        (1.0, 1.0, 0.25, 0),
        (1.0, 1.0, 0.25, 1),
        (1.0, 1.0, 0.25, 2),
        (0.5, 2.0, 0.1, 3),
    ],
)
def test_neumann_preconditioner_expression(
    beta,
    gamma,
    h,
    order,
    neumann_factory,
    neumann_kernel,
    precision,
    tolerance,
):
    """Validate Neumann preconditioner against a truncated series.

    System: dx/dt = J x with J = [[a, b], [c, d]] = [[1, 2], [3, 4]].
    Preconditioner approximates (beta*I - gamma*h*J)^{-1} via truncated series.
    """
    pre = neumann_factory(beta, gamma, order)
    kernel = neumann_kernel(pre)

    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)
    empty_base = np.empty(0, dtype=precision)

    kernel[1, 1](
        precision(0.0), precision(h), precision(1.0), v, empty_base, out
    )

    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    beta_inv = 1.0 / beta
    T = (gamma * beta_inv) * h * J

    # Truncated Neumann series: beta^{-1} sum_{k=0}^{order} (T^k) v
    expected = np.zeros_like(v)
    Tk_v = v.copy()
    expected += Tk_v
    for _ in range(order):
        Tk_v = T @ Tk_v
        expected += Tk_v
    expected = beta_inv * expected

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.mark.parametrize(
    "beta,gamma,h,order",
    [
        (1.0, 1.0, 0.25, 0),
        (1.0, 1.0, 0.25, 1),
        (1.0, 1.0, 0.25, 2),
        (0.5, 1.5, 0.1, 3),
    ],
)
def test_neumann_preconditioner_cached_expression(
    beta,
    gamma,
    h,
    order,
    neumann_cached_factory,
    neumann_cached_kernel,
    cached_system,
    prepare_jac_factory,
    precision,
    tolerance,
):
    """Validate cached Neumann preconditioner with stored auxiliaries."""

    prepare, aux_count = prepare_jac_factory()
    pre = neumann_cached_factory(beta, gamma, order)
    kernel = neumann_cached_kernel(prepare, pre, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)

    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.7, -1.3], dtype=precision)
    vec = vec[:state_len]
    out = np.zeros(state_len, dtype=precision)

    empty_base = np.empty(0, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        precision(h),
        precision(1.0),
        vec,
        empty_base,
        out,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    beta_val = precision(beta)
    gamma_val = precision(gamma)
    beta_inv = precision(1.0) / beta_val
    h_val = precision(h)
    T = (gamma_val * beta_inv) * h_val * jacobian

    expected = np.zeros(state_len, dtype=precision)
    Tk_v = vec.copy()
    expected += Tk_v
    for _ in range(order):
        Tk_v = T @ Tk_v
        expected += Tk_v
    expected = beta_inv * expected

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.fixture(scope="session")
def stage_residual_factory(operator_system, precision):
    def factory(beta, gamma, a_ii, M):
        fname = (
            "stage_residual_factory_"
            f"{_stable_factory_tag(beta, gamma, M.tobytes())}"
        )
        code = generate_residual_code(
            operator_system.equations,
            operator_system.indices,
            M=M,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        res_fac, was_cached = operator_system.gen_file.import_function(
            fname, code
        )
        return res_fac(from_dtype(operator_system.precision))

    return factory


@pytest.fixture(scope="session")
def residual_kernel(precision):
    def make_kernel(residual):
        @cuda.jit
        def kernel(t, h, aij, vec, base_state, out):
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            residual(vec, parameters, drivers, t, h, aij, base_state, out)

        return kernel

    return make_kernel


@pytest.mark.parametrize(
    "beta,gamma,h,a_ii,M",
    [
        (1.0, 1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, 0.5, np.diag([1.0, 0.0])),
        (0.5, 2.0, 1.0, 0.25, np.diag([0.0, 1.0])),
    ],
)
def test_stage_residual(
    beta,
    gamma,
    h,
    a_ii,
    M,
    stage_residual_factory,
    residual_kernel,
    precision,
    tolerance,
):
    residual = stage_residual_factory(beta, gamma, a_ii, M)
    kernel = residual_kernel(residual)
    stage = np.array([0.5, -0.3], dtype=precision)
    base = np.array([0.25, -0.25], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](
        precision(0.0), precision(h), precision(a_ii), stage, base, out
    )
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    eval_point = base + a_ii * stage
    expected = beta * (M @ stage) - gamma * h * (J @ eval_point)
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def _colliding_system_f(point):
    """Return the colliding-constants system derivative."""
    x0, x1 = point
    return np.array(
        [-2.5 * x0 + 0.75 * x1, -0.75 * x1],
        dtype=point.dtype,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        COLLIDING_CONSTANTS_F32,
        COLLIDING_CONSTANTS_F64,
    ],
    indirect=True,
)
def test_solver_helper_preserves_colliding_constants(
    system, residual_kernel, precision, tolerance
):
    """Helper generation leaves beta/gamma constants untouched."""

    residual = system.get_solver_helper(
        role="residual", beta=1.0, gamma=1.0
    ).device_function
    assert system.constants.values_array.dtype == np.dtype(precision)
    assert system.constants.values_dict["beta"] == precision(2.5)
    assert system.constants.values_dict["gamma"] == precision(0.75)

    kernel = residual_kernel(residual)
    stage = np.zeros(2, dtype=precision)
    base = np.array([1.0, 2.0], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](
        precision(0.0), precision(1.0), precision(1.0), stage, base,
        out
    )
    # residual(u=0) = -h * f(base_state) with the system's own
    # constants; the corrupted form would use beta = gamma = 1.
    expected = -_colliding_system_f(base)
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        COLLIDING_CONSTANTS_F32,
        COLLIDING_CONSTANTS_F64,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "first_scalings,second_scalings",
    [
        ((1.0, 1.0), (2.0, 1.0)),
        ((1.0, 1.0), (1.0, 0.5)),
    ],
    ids=["beta-only", "gamma-only"],
)
def test_solver_helper_rebuilds_on_scaling_change(
    system,
    residual_kernel,
    precision,
    tolerance,
    first_scalings,
    second_scalings,
):
    """Each changed solver scaling regenerates the cached helper."""

    stage = np.array([0.5, -0.3], dtype=precision)
    base = np.array([1.0, 2.0], dtype=precision)
    a_ii = precision(0.5)
    eval_point = base + a_ii * stage

    results = []
    helpers = []
    for beta, gamma in (first_scalings, second_scalings):
        residual = system.get_solver_helper(
            role="residual",
            beta=beta,
            gamma=gamma,
        ).device_function
        helpers.append(residual)
        kernel = residual_kernel(residual)
        out = np.zeros(2, dtype=precision)
        kernel[1, 1](
            precision(0.0), precision(1.0), a_ii, stage, base, out
        )
        results.append(out)
        expected = (
            beta * stage - gamma * _colliding_system_f(eval_point)
        )
        assert np.allclose(
            out,
            expected,
            atol=tolerance.abs_tight,
            rtol=tolerance.rel_tight,
        )
    assert helpers[0] is not helpers[1]
    assert not np.allclose(results[0], results[1])


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_neumann_helper_rebuilds_on_order_change(system):
    """A changed Neumann order produces a distinct bound member.

    The order is a factory-binding argument, not a source input, so
    both orders share one generated factory while binding distinct
    members.
    """
    first_kwargs = dict(
        role="neumann_preconditioner",
        beta=1.0,
        gamma=1.0,
        preconditioner_order=1,
    )
    second_kwargs = dict(first_kwargs, preconditioner_order=2)
    first = system.get_solver_helper(**first_kwargs)
    second = system.get_solver_helper(**second_kwargs)

    assert first is not second
    assert first.device_function is not second.device_function
    assert helper_source_hash(
        system, SolverHelperRequest(**first_kwargs)
    ) == helper_source_hash(
        system, SolverHelperRequest(**second_kwargs)
    )


@pytest.mark.parametrize(
    "role,stacked,expected",
    [
        ("neumann_preconditioner", False, 2),
        ("jacobi_preconditioner", False, 0),
        ("jacobi_preconditioner", True, 0),
    ],
    ids=["neumann", "jacobi", "stacked-jacobi"],
)
def test_request_order_defaults_to_the_role(role, stacked, expected):
    """An unset request order follows the requested role."""
    stage_kwargs = {}
    if stacked:
        stage_kwargs = {
            "stage_coefficients": ((0.25,),),
            "stage_nodes": (0.25,),
        }
    request = SolverHelperRequest(
        role=role, stacked=stacked, **stage_kwargs
    )
    assert request.preconditioner_order == expected


def test_request_order_rejects_values_above_two():
    """SolverHelperRequest rejects unsupported series orders."""
    with pytest.raises(ValueError):
        SolverHelperRequest(
            role="neumann_preconditioner", preconditioner_order=3
        )


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_stacked_jacobi_series_order_rejected(system):
    """Stacked multi-stage jacobi requests refuse series orders."""
    with pytest.raises(ValueError, match="preconditioner_order=0"):
        system.get_solver_helper(
            role="jacobi_preconditioner",
            jacobian_at="stage",
            stacked=True,
            preconditioner_order=1,
            stage_coefficients=((0.25, 0.0), (0.5, 0.25)),
            stage_nodes=(0.25, 0.75),
        )


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_helper_requests_reuse_members_without_touching_settings(system):
    """Repeated requests reuse members; settings stay untouched."""

    settings_before = system.compile_settings
    hash_before = system.config_hash

    scaled_kwargs = dict(role="linear_operator", beta=2.5, gamma=0.5)
    scaled = system.get_solver_helper(**scaled_kwargs)
    first = system.get_solver_helper(role="prepare_jac", jacobian_at="step")
    second = system.get_solver_helper(role="prepare_jac", jacobian_at="step")
    repeat_scaled = system.get_solver_helper(**scaled_kwargs)

    assert first is second
    assert first.cached_auxiliary_count is not None
    assert repeat_scaled is scaled
    # Helper requests never mutate the system's compile settings or
    # its configuration identity.
    assert system.compile_settings is settings_before
    assert system.config_hash == hash_before


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_unknown_helper_fails_at_request_construction(system):
    """Unknown helper kinds fail when the request is constructed."""

    cached = system.get_solver_helper(role="prepare_jac", jacobian_at="step")

    with pytest.raises(ValueError):
        SolverHelperRequest(role="not_a_helper")

    assert (
        system.get_solver_helper(role="prepare_jac", jacobian_at="step")
        is cached
    )


@pytest.fixture(scope="session")
def jacobi_factory(cached_system, precision):
    """Return a factory producing Jacobi preconditioner device functions."""

    def factory(beta, gamma, M=None, order=0):
        mass_tag = (
            "eye"
            if M is None
            else _stable_factory_tag(np.asarray(M).tobytes())
        )
        fname = (
            "jacobi_preconditioner_factory_"
            f"{int(beta * 10)}_{int(gamma * 10)}_{mass_tag}"
        )
        code = generate_jacobi_preconditioner_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
            M=M,
            beta=beta,
            gamma=gamma,
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            from_dtype(cached_system.precision),
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def jacobi_kernel(precision):
    """Apply the Jacobi preconditioner to a vector."""

    n = 2

    def make_kernel(pre):
        @cuda.jit
        def kernel(t, h, a_ij, state_values, base_state, vec, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            cached_aux = cuda.local.array(1, precision)
            jvp = cuda.local.array(n, precision)
            for idx in range(n):
                state[idx] = state_values[idx]
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel


def _cached_system_jacobian_diagonal(eval_point):
    """Jacobian diagonal of the cached_system fixture equations.

    dx0 = a*x0*x1 + b*sin(x0) -> J00 = a*x1 + b*cos(x0)
    dx1 = c*x0*x1 + d*cos(x1) -> J11 = c*x0 - d*sin(x1)
    with constants a=0.5, b=1.3, c=-0.7, d=0.9.
    """
    x0, x1 = eval_point
    j00 = 0.5 * x1 + 1.3 * np.cos(x0)
    j11 = -0.7 * x0 - 0.9 * np.sin(x1)
    return np.array([j00, j11])


@pytest.mark.parametrize(
    "beta,gamma,h,a_ij",
    [
        (1.0, 1.0, 0.2, 0.5),
        (0.5, 2.0, 0.1, 1.0),
    ],
)
def test_jacobi_preconditioner_diagonal(
    beta,
    gamma,
    h,
    a_ij,
    jacobi_factory,
    jacobi_kernel,
    precision,
    tolerance,
):
    """Validate Jacobi output against the analytic Jacobian diagonal.

    The preconditioner divides v elementwise by
    ``beta - gamma*h*a_ij*J_ii`` with J evaluated at
    ``base_state + a_ij*state``.
    """
    pre = jacobi_factory(beta, gamma)
    kernel = jacobi_kernel(pre)

    state = np.array([0.3, -0.6], dtype=precision)
    base = np.array([0.1, 0.2], dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](
        precision(0.0), precision(h), precision(a_ij), state, base, v, out
    )

    diag_j = _cached_system_jacobian_diagonal(base + a_ij * state)
    expected = v / (beta - gamma * h * a_ij * diag_j)

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.fixture(scope="session")
def jacobi_zero_diag_factory(operator_system, precision):
    """Jacobi preconditioner for the constant-Jacobian system."""

    fname = "jacobi_preconditioner_zero_diag"
    code = generate_jacobi_preconditioner_code(
        operator_system.equations,
        operator_system.indices,
        func_name=fname,
    )
    pre_fac, was_cached = operator_system.gen_file.import_function(
        fname, code
    )
    return pre_fac(from_dtype(operator_system.precision))


def test_jacobi_preconditioner_zero_diagonal_guard(
    jacobi_zero_diag_factory,
    jacobi_kernel,
    precision,
    tolerance,
):
    """A vanishing diagonal yields finite output, not inf/NaN.

    operator_system has J = [[1, 2], [3, 4]]; with
    beta = gamma = h = a_ij = 1 the first diagonal entry
    ``1 - 1*1*1*J00 = 0`` exactly, so the division guard floors it.
    The second entry ``1 - 4 = -3`` is untouched.
    """
    kernel = jacobi_kernel(jacobi_zero_diag_factory)

    state = np.zeros(2, dtype=precision)
    base = np.zeros(2, dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](
        precision(0.0),
        precision(1.0),
        precision(1.0),
        state,
        base,
        v,
        out,
    )

    assert np.all(np.isfinite(out))
    assert np.isclose(
        out[1],
        v[1] / precision(-3.0),
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def _cached_system_jacobian(eval_point):
    """Full Jacobian of the cached_system fixture equations."""
    x0, x1 = eval_point
    return np.array(
        [
            [0.5 * x1 + 1.3 * np.cos(x0), 0.5 * x0],
            [-0.7 * x1, -0.7 * x0 - 0.9 * np.sin(x1)],
        ],
        dtype=np.float64,
    )


def _polynomial_jacobi(operator, v, order):
    """Return ``sum_[k=0..order] (D^-1 N)^k D^-1 v``, ``D`` its diagonal."""
    diagonal = np.diag(np.diag(operator))
    off_diagonal = diagonal - operator
    iteration = np.linalg.solve(diagonal, off_diagonal)
    scaled = np.linalg.solve(diagonal, v)
    total = scaled.copy()
    term = scaled.copy()
    for _ in range(order):
        term = iteration @ term
        total = total + term
    return total


@pytest.fixture(scope="session")
def n_stage_jacobi_factory(cached_system, precision):
    """Return a factory producing FIRK Jacobi preconditioners."""

    def factory(beta, gamma, stage_coefficients, stage_nodes, order=0):
        tag = _stable_factory_tag(
            beta, gamma, stage_coefficients, stage_nodes
        )
        fname = f"n_stage_jacobi_factory_{tag}"
        code = generate_jacobi_preconditioner_code(
            cached_system.equations,
            cached_system.indices,
            variant=HelperVariant.STACKED_STAGES,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            from_dtype(cached_system.precision),
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def n_stage_jacobi_kernel(precision):
    """Apply a two-stage FIRK preconditioner to a flattened vector."""

    width = 4

    def make_kernel(pre):
        @cuda.jit
        def kernel(t, h, stage_values, base_state, vec, out):
            state = cuda.local.array(width, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            cached_aux = cuda.local.array(1, precision)
            jvp = cuda.local.array(width, precision)
            for idx in range(width):
                state[idx] = stage_values[idx]
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                precision(1.0),
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "mass",
    [None, np.diag([1.0, 0.0])],
    ids=["identity-mass", "torn-mass"],
)
def test_jacobi_preconditioner_series(
    order,
    mass,
    jacobi_factory,
    jacobi_kernel,
    precision,
    tolerance,
):
    """Order ``p`` expands the series about the operator diagonal."""
    beta, gamma, h, a_ij = 0.7, 1.3, 0.2, 0.5
    pre = jacobi_factory(beta, gamma, M=mass, order=order)
    kernel = jacobi_kernel(pre)

    state = np.array([0.3, -0.6], dtype=precision)
    base = np.array([0.1, 0.2], dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](
        precision(0.0), precision(h), precision(a_ij), state, base, v, out
    )

    mass_matrix = np.eye(2) if mass is None else np.asarray(mass)
    jacobian = _cached_system_jacobian(base + a_ij * state)
    operator = beta * mass_matrix - gamma * h * a_ij * jacobian
    expected = _polynomial_jacobi(
        operator, v.astype(np.float64), order
    )

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.mark.parametrize("order", [1, 2])
def test_jacobi_preconditioner_cached_series(
    order,
    prepare_jac_factory,
    jacobi_cached_factory,
    neumann_cached_kernel,
    precision,
    tolerance,
):
    """The cached Jacobi series reads its Jacobian from the cache."""
    beta, gamma, h, a_ij = 1.0, 1.0, 0.2, 0.5
    prepare, aux_count = prepare_jac_factory()
    pre = jacobi_cached_factory(beta, gamma, order=order)
    kernel = neumann_cached_kernel(prepare, pre, aux_count)

    state = np.array([0.3, -0.6], dtype=precision)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    base = np.zeros(2, dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](
        state,
        params,
        drivers,
        precision(0.0),
        precision(h),
        precision(a_ij),
        v,
        base,
        out,
    )

    # The cached variant evaluates J at ``state`` directly.
    jacobian = _cached_system_jacobian(state)
    operator = beta * np.eye(2) - gamma * h * a_ij * jacobian
    expected = _polynomial_jacobi(
        operator, v.astype(np.float64), order
    )

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_n_stage_jacobi_preconditioner_series(
    order,
    n_stage_jacobi_factory,
    n_stage_jacobi_kernel,
    precision,
    tolerance,
):
    """Each FIRK series term applies the whole ``A (x) J`` operator."""
    beta, gamma, h = 0.9, 1.1, 0.15
    stage_coefficients = ((0.25, 0.0), (0.5, 0.25))
    stage_nodes = (0.25, 0.75)
    pre = n_stage_jacobi_factory(
        beta, gamma, stage_coefficients, stage_nodes, order=order
    )
    kernel = n_stage_jacobi_kernel(pre)

    stage_values = np.array([0.3, -0.6, 0.15, 0.4], dtype=precision)
    base = np.array([0.1, 0.2], dtype=precision)
    v = np.array([0.7, -1.3, 0.4, 0.9], dtype=precision)
    out = np.zeros(4, dtype=precision)

    kernel[1, 1](
        precision(0.0), precision(h), stage_values, base, v, out
    )

    operator = np.zeros((4, 4))
    for stage in range(2):
        point = base.astype(np.float64)
        for contrib in range(2):
            point = point + stage_coefficients[stage][contrib] * (
                stage_values[2 * contrib: 2 * contrib + 2]
            )
        jacobian = _cached_system_jacobian(point)
        for contrib in range(2):
            block = -gamma * h * stage_coefficients[stage][contrib]
            operator[
                2 * stage: 2 * stage + 2,
                2 * contrib: 2 * contrib + 2,
            ] = block * jacobian
        operator[
            2 * stage: 2 * stage + 2, 2 * stage: 2 * stage + 2
        ] += beta * np.eye(2)
    expected = _polynomial_jacobi(
        operator, v.astype(np.float64), order
    )

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.fixture(scope="session")
def jacobi_cached_factory(cached_system, precision):
    """Return a factory producing cached Jacobi preconditioners."""

    def factory(beta, gamma, order=0):
        fname = (
            "jacobi_cached_factory_"
            f"{int(beta * 10)}_{int(gamma * 10)}"
        )
        code = generate_jacobi_preconditioner_code(
            cached_system.equations,
            cached_system.indices,
            variant=HelperVariant.CACHED,
            func_name=fname,
            beta=beta,
            gamma=gamma,
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            from_dtype(cached_system.precision),
            order=order,
        )

    return factory


@pytest.mark.parametrize(
    "beta,gamma,h,a_ij",
    [
        (1.0, 1.0, 0.2, 0.5),
        (0.5, 2.0, 0.1, 1.0),
    ],
)
def test_jacobi_preconditioner_cached_diagonal(
    beta,
    gamma,
    h,
    a_ij,
    prepare_jac_factory,
    jacobi_cached_factory,
    neumann_cached_kernel,
    precision,
    tolerance,
):
    """Validate cached Jacobi output against the Jacobian diagonal.

    The cached variant evaluates J at ``state`` directly (Rosenbrock
    convention) rather than at ``base_state + a_ij*state``.
    """
    prepare, aux_count = prepare_jac_factory()
    pre = jacobi_cached_factory(beta, gamma)
    kernel = neumann_cached_kernel(prepare, pre, aux_count)

    state = np.array([0.3, -0.6], dtype=precision)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    base = np.zeros(2, dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](
        state,
        params,
        drivers,
        precision(0.0),
        precision(h),
        precision(a_ij),
        v,
        base,
        out,
    )

    diag_j = _cached_system_jacobian_diagonal(state)
    expected = v / (beta - gamma * h * a_ij * diag_j)

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_jacobi_preconditioner_mass_matrix(
    jacobi_factory,
    jacobi_kernel,
    precision,
    tolerance,
):
    """Jacobi divides by ``beta*M_ii - gamma*h*a_ij*J_ii``; a zero
    mass row drops the beta term."""
    beta, gamma, h, a_ij = 1.0, 1.0, 0.2, 0.5
    mass = np.diag([1.0, 0.0])
    pre = jacobi_factory(beta, gamma, M=mass)
    kernel = jacobi_kernel(pre)

    state = np.array([0.3, -0.6], dtype=precision)
    base = np.array([0.1, 0.2], dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](
        precision(0.0), precision(h), precision(a_ij), state, base, v, out
    )

    diag_j = _cached_system_jacobian_diagonal(base + a_ij * state)
    expected = v / (
        beta * np.diag(mass) - gamma * h * a_ij * diag_j
    )

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_torn_structure_selects_distinct_cached_helpers(
    jacobi_kernel,
    precision,
    tolerance,
):
    """A torn system and its explicit twin share no helper source.

    Same-named systems with different mass structure carry different
    ``fn_hash`` values and must not reuse each other's cached device
    functions, in memory or on disk.
    """
    explicit = create_ODE_system(
        [
            "dx0 = -k0*x0 + x0*x1",
            "dx1 = -k1*x1 + x0*x0",
        ],
        states=["x0", "x1"],
        constants={"k0": 1.0, "k1": 2.0},
        precision=precision,
        name="mass_cache_key_sys",
    )
    torn = create_ODE_system(
        [
            "dx0 = -k0*x0 + x0*x1",
            "0 = -k1*x1 + x0*x0 + x1**5",
        ],
        states=["x0", "x1"],
        constants={"k0": 1.0, "k1": 2.0},
        precision=precision,
        name="mass_cache_key_sys",
    )
    assert explicit.mass is None
    assert torn.mass is not None
    assert torn.fn_hash != explicit.fn_hash

    jacobi_kwargs = dict(
        role="jacobi_preconditioner", beta=1.0, gamma=1.0
    )
    jacobi_request = SolverHelperRequest(**jacobi_kwargs)
    assert helper_source_hash(
        torn, jacobi_request
    ) != helper_source_hash(explicit, jacobi_request)

    h, a_ij = 0.2, 0.5
    state = np.array([0.3, -0.6], dtype=precision)
    base = np.array([0.1, 0.2], dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    eval_point = base + a_ij * state

    pre_eye = explicit.get_solver_helper(**jacobi_kwargs).device_function
    out_eye = np.zeros(2, dtype=precision)
    jacobi_kernel(pre_eye)[1, 1](
        precision(0.0),
        precision(h),
        precision(a_ij),
        state,
        base,
        v,
        out_eye,
    )

    pre_torn = torn.get_solver_helper(**jacobi_kwargs).device_function
    out_torn = np.zeros(2, dtype=precision)
    jacobi_kernel(pre_torn)[1, 1](
        precision(0.0),
        precision(h),
        precision(a_ij),
        state,
        base,
        v,
        out_torn,
    )

    # Explicit twin: J00 = -k0 + x1, J11 = -k1.
    diag_j_explicit = np.array([-1.0 + eval_point[1], -2.0])
    expected_eye = v / (1.0 - h * a_ij * diag_j_explicit)
    # Torn twin: J11 = -k1 + 5*x1**4; the zero mass row drops beta.
    diag_j_torn = np.array(
        [-1.0 + eval_point[1], -2.0 + 5.0 * eval_point[1] ** 4]
    )
    expected_torn = v / (
        np.array([1.0, 0.0]) - h * a_ij * diag_j_torn
    )

    assert np.allclose(
        out_eye,
        expected_eye,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )
    assert np.allclose(
        out_torn,
        expected_torn,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


# Hodgkin-Huxley cache-planner tests, driven through the system spine.


@pytest.fixture(scope="session")
def system_operator_pair_kernel(system, precision):
    """Kernel comparing cached and at-state operators on ``system``."""

    n_state = len(system.indices.states.index_map)
    n_params = len(system.indices.parameters.index_map)
    n_drivers = len(system.indices.drivers.index_map)

    def make_kernel(prepare, cached_op, inline_op, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values, t, h, a_ij, vec, out_cached, out_inline
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)
            for idx in range(n_state):
                state[idx] = state_values[idx]
            prepare(state, parameters, drivers, t, h, cached_aux)
            cached_op(
                state,
                parameters,
                drivers,
                cached_aux,
                state,
                t,
                h,
                a_ij,
                vec,
                out_cached,
            )
            inline_op(
                state,
                parameters,
                drivers,
                cached_aux,
                state,
                t,
                h,
                a_ij,
                vec,
                out_inline,
            )

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def system_cached_precond_kernel(system, precision):
    """Kernel applying prepare plus a cached preconditioner on ``system``."""

    n_state = len(system.indices.states.index_map)
    n_params = len(system.indices.parameters.index_map)
    n_drivers = len(system.indices.drivers.index_map)

    def make_kernel(prepare, pre, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(state_values, t, h, a_ij, vec, out):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)
            jvp = cuda.local.array(n_state, precision)
            for idx in range(n_state):
                state[idx] = state_values[idx]
            prepare(state, parameters, drivers, t, h, cached_aux)
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel


@pytest.mark.parametrize(
    "solver_settings_override",
    [HODGKIN_HUXLEY_SYSTEM],
    indirect=True,
)
def test_hh_planner_selects_cached_slots(system):
    """The planner activates on a transcendental-heavy real system."""
    equations = system._get_jvp_exprs()
    selection = equations.cache_selection

    assert len(selection.cached_leaf_order) > 0
    assert len(selection.cached_leaf_order) <= equations.cache_slot_limit
    assert selection.saved >= (
        selection.read_price * len(selection.cached_leaf_order)
    )

    all_nodes = set(equations.non_jvp_order)
    cached = set(selection.cached_leaf_order)
    removed = set(selection.removal_nodes)
    runtime = set(selection.runtime_nodes)
    prepare = set(selection.prepare_nodes)
    assert cached <= removed
    assert cached <= prepare
    assert removed | runtime == all_nodes
    assert removed & runtime == set()
    assert not (cached & equations.v_dependent_nodes)
    for lhs in runtime:
        for dep in equations.dependencies.get(lhs, set()):
            assert dep in runtime or dep in cached
    for lhs in prepare:
        for dep in equations.dependencies.get(lhs, set()):
            assert dep in prepare
    for lhs in removed - cached:
        for consumer in equations.dependents.get(lhs, set()):
            assert consumer in removed
        assert equations.jvp_usage.get(lhs, 0) == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [HODGKIN_HUXLEY_SYSTEM],
    indirect=True,
)
def test_cache_selection_changes_cached_source_hash(system):
    """A changed cache selection renames cached-family sources only."""
    cached_request = SolverHelperRequest(
        role="prepare_jac", jacobian_at="step"
    )
    plain_request = SolverHelperRequest(
        role="linear_operator", beta=1.0, gamma=1.0
    )
    equations = system._get_jvp_exprs()
    original = equations.cache_selection
    cached_before = helper_source_hash(system, cached_request)
    plain_before = helper_source_hash(system, plain_request)
    replanned = _JVPEquations(
        list(equations.ordered_assignments), max_cached_terms=1
    )
    try:
        equations.update_cache_selection(
            plan_auxiliary_cache(replanned)
        )
        assert helper_source_hash(
            system, cached_request
        ) != cached_before
        assert helper_source_hash(
            system, plain_request
        ) == plain_before
    finally:
        equations.update_cache_selection(original)


@pytest.mark.parametrize(
    "solver_settings_override",
    [HODGKIN_HUXLEY_SYSTEM],
    indirect=True,
)
def test_hh_cached_operator_matches_inline(
    system, system_operator_pair_kernel, precision, tolerance
):
    """Cached prepare+operator equals the at-state operator on HH."""
    n = len(system.indices.states.index_map)

    prepare_helper = system.get_solver_helper(
        role="prepare_jac",
        jacobian_at="step",
    )
    prepare = prepare_helper.device_function
    aux_count = prepare_helper.cached_auxiliary_count
    assert aux_count is not None
    assert aux_count > 0

    cached_op = system.get_solver_helper(
        role="linear_operator",
        jacobian_at="step",
        beta=1.0,
        gamma=1.0,
    ).device_function
    inline_op = system.get_solver_helper(
        role="linear_operator",
        jacobian_at="state",
        beta=1.0,
        gamma=1.0,
    ).device_function

    kernel = system_operator_pair_kernel(
        prepare, cached_op, inline_op, aux_count
    )

    state_values = np.array([-62.0, 0.07, 0.55, 0.34], dtype=precision)
    vec = np.array([0.8, -1.1, 0.4, -0.3], dtype=precision)
    out_cached = np.zeros(n, dtype=precision)
    out_inline = np.zeros(n, dtype=precision)

    kernel[1, 1](
        state_values,
        precision(0.0),
        precision(0.25),
        precision(1.0),
        vec,
        out_cached,
        out_inline,
    )

    assert np.allclose(
        out_cached,
        out_inline,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [HODGKIN_HUXLEY_SYSTEM],
    indirect=True,
)
def test_hh_cached_jacobi_reads_prepare_only_auxiliaries(
    system, system_cached_precond_kernel, precision, tolerance
):
    """Cached Jacobi diagonals reading prepare-only nodes match HH."""
    n = len(system.indices.states.index_map)

    prepare_helper = system.get_solver_helper(
        role="prepare_jac",
        jacobian_at="step",
    )
    prepare = prepare_helper.device_function
    aux_count = prepare_helper.cached_auxiliary_count

    jacobi = system.get_solver_helper(
        role="jacobi_preconditioner",
        jacobian_at="step",
        beta=1.0,
        gamma=1.0,
    ).device_function

    kernel = system_cached_precond_kernel(prepare, jacobi, aux_count)

    h = precision(0.25)
    a_ij = precision(1.0)
    # State order is hg, m, n, vm.
    state_values = np.array([0.55, 0.07, 0.34, -62.0], dtype=precision)
    vec = np.array([0.4, -1.1, -0.3, 0.8], dtype=precision)
    out = np.zeros(n, dtype=precision)
    kernel[1, 1](state_values, precision(0.0), h, a_ij, vec, out)

    hg, m, nn, vm = (float(value) for value in state_values)
    constants = system.constants.values_dict
    alpha_m = 0.1 * (vm + 40.0) / (1.0 - np.exp(-(vm + 40.0) / 10.0))
    beta_m = 4.0 * np.exp(-(vm + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(vm + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(vm + 35.0) / 10.0))
    alpha_n = 0.01 * (vm + 55.0) / (1.0 - np.exp(-(vm + 55.0) / 10.0))
    beta_n = 0.125 * np.exp(-(vm + 65.0) / 80.0)
    diag_j = np.array(
        [
            -(alpha_h + beta_h),
            -(alpha_m + beta_m),
            -(alpha_n + beta_n),
            -(
                constants["g_na"] * m**3 * hg
                + constants["g_k"] * nn**4
                + constants["g_l"]
            )
            / constants["c_m"],
        ]
    )
    expected = vec / (1.0 - float(h) * float(a_ij) * diag_j)

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


def test_legal_variants_derive_from_capabilities():
    """Variant legality follows the declared role capabilities."""
    assert LinearOperator.legal_variants() == frozenset(
        {
            HelperVariant.PLAIN,
            HelperVariant.CACHED,
            HelperVariant.AT_STATE,
            HelperVariant.STACKED_STAGES,
            HelperVariant.CACHED_STACKED,
        }
    )
    assert Residual.legal_variants() == frozenset(
        {
            HelperVariant.PLAIN,
            HelperVariant.CACHED,
            HelperVariant.STACKED_STAGES,
        }
    )
    assert ApplyMass.legal_variants() == frozenset(
        {HelperVariant.PLAIN, HelperVariant.CACHED}
    )
    assert PrepareJac.legal_variants() == frozenset(
        {HelperVariant.CACHED}
    )
    assert LuSolve.legal_variants() == frozenset(
        {
            HelperVariant.PLAIN,
            HelperVariant.CACHED,
            HelperVariant.AT_STATE,
            HelperVariant.STACKED_STAGES,
            HelperVariant.PREFACTORED,
            HelperVariant.PREFACTORED_STACKED,
        }
    )
    assert LuPrepareBlocks.legal_variants() == frozenset(
        {
            HelperVariant.PREFACTORED,
            HelperVariant.PREFACTORED_STACKED,
        }
    )
    assert LuSmoothingSolve.legal_variants() == frozenset(
        {HelperVariant.PREFACTORED_STACKED}
    )
    assert InitResidual.legal_variants() == frozenset(
        {HelperVariant.PLAIN}
    )
    assert InitLuSolve.legal_variants() == frozenset(
        {HelperVariant.PLAIN}
    )


@pytest.mark.parametrize(
    "role,axis_kwargs",
    [
        ("apply_mass", {"stacked": True}),
        ("residual", {"jacobian_at": "state"}),
        ("evaluate_inv_mass_f", {"jacobian_at": "state"}),
        ("prepare_jac", {}),
        ("linear_operator", {"jacobian_at": "step", "prefactored": True}),
        ("init_residual", {"stacked": True}),
        ("init_lu_solve", {"jacobian_at": "step", "prefactored": True}),
    ],
)
def test_illegal_role_variant_pairs_fail_at_construction(
    role, axis_kwargs
):
    """Combinations outside the declared grid raise on construction."""
    with pytest.raises(ValueError):
        SolverHelperRequest(role=role, **axis_kwargs)


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_cached_variant_on_cache_invariant_role_serves_plain(system):
    """CACHED on a role without a Jacobian returns the PLAIN member."""
    plain = system.get_solver_helper(role="residual", beta=1.0, gamma=1.0)
    cached = system.get_solver_helper(
        role="residual",
        jacobian_at="step",
        beta=1.0,
        gamma=1.0,
    )
    assert cached is plain
    assert cached.prepare_jac is None


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_cached_member_carries_prepare_companion(system):
    """A cached Jacobian-carrying member serves its prepare_jac."""
    operator = system.get_solver_helper(
        role="linear_operator",
        jacobian_at="step",
        beta=1.0,
        gamma=1.0,
    )
    direct = system.get_solver_helper(role="prepare_jac", jacobian_at="step")
    assert operator.prepare_jac is direct.device_function
    assert (
        operator.cached_auxiliary_count
        == direct.cached_auxiliary_count
    )


def test_lu_solve_helper_metadata_and_member_reuse(operator_system):
    """lu_solve requests carry lu_nnz and reuse bound members."""
    first = operator_system.get_solver_helper("lu_solve")
    second = operator_system.get_solver_helper("lu_solve")
    assert first is second
    assert isinstance(first.lu_nnz, int)
    assert first.lu_nnz >= 0
    at_state = operator_system.get_solver_helper(
        "lu_solve", jacobian_at="state"
    )
    assert isinstance(at_state.lu_nnz, int)
    cached = operator_system.get_solver_helper(
        "lu_solve", jacobian_at="step"
    )
    assert isinstance(cached.lu_nnz, int)
    assert cached.prepare_jac is not None
    assert cached.cached_auxiliary_count is not None


def test_lu_solve_lu_nnz_survives_source_cache(precision):
    """A reimported cached factory reports the same lu_nnz."""
    dxdt = [
        "dx0 = a*x0 + b*x1",
        "dx1 = c*x0 + d*x1",
    ]
    constants = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    first_system = create_ODE_system(
        dxdt,
        states=["x0", "x1"],
        constants=constants,
        precision=precision,
        name="lu_cache_roundtrip",
    )
    first = first_system.get_solver_helper("lu_solve")
    second_system = create_ODE_system(
        dxdt,
        states=["x0", "x1"],
        constants=constants,
        precision=precision,
        name="lu_cache_roundtrip",
    )
    second = second_system.get_solver_helper("lu_solve")
    assert second.lu_nnz == first.lu_nnz


def test_lu_solve_scaled_binding_matches_dense(
    operator_system, precision, tolerance
):
    """A beta/gamma-bound lu_solve matches the dense shifted solve."""
    beta = 0.8
    gamma = 0.6
    member = operator_system.get_solver_helper(
        "lu_solve", beta=beta, gamma=gamma
    )
    lu_solve = member.device_function
    factor_len = max(member.lu_nnz, 1)

    n = 2
    h = precision(0.05)
    a_ij = precision(0.5)
    jac = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    shifted = (
        precision(beta) * np.eye(n, dtype=precision)
        - precision(gamma) * a_ij * h * jac
    ).astype(precision)
    rhs = np.array([1.5, -0.25], dtype=precision)
    expected = np.linalg.solve(shifted, rhs)

    @cuda.jit
    def kernel(rhs_vec, x, status):
        state = cuda.local.array(n, precision)
        base_state = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        factor = cuda.local.array(factor_len, precision)
        cached_aux = cuda.local.array(1, precision)
        for i in range(n):
            state[i] = precision(0.0)
            base_state[i] = precision(0.0)
        status[0] = lu_solve(
            state,
            parameters,
            drivers,
            cached_aux,
            base_state,
            precision(0.0),
            h,
            a_ij,
            rhs_vec,
            x,
            factor,
        )

    x = np.zeros(n, dtype=precision)
    status = np.zeros(1, dtype=np.int32)
    kernel[1, 1](rhs, x, status)
    assert status[0] == 0
    assert np.allclose(
        x,
        expected,
        rtol=tolerance.rel_tight * 10,
        atol=tolerance.abs_tight * 10,
    )


def test_none_preconditioner_is_identity(operator_system, precision):
    """The 'none' role serves an identity preconditioner."""
    plain = operator_system.get_solver_helper("none")
    at_state = operator_system.get_solver_helper(
        "none", jacobian_at="state"
    )
    frozen = operator_system.get_solver_helper("none", jacobian_at="step")
    stacked = operator_system.get_solver_helper(
        "none",
        stacked=True,
        stage_coefficients=((0.5,),),
        stage_nodes=(0.5,),
    )
    for member in (plain, at_state, frozen, stacked):
        assert member.device_function is not None

    fn = plain.device_function

    @cuda.jit
    def kernel(v, out):
        state = cuda.local.array(2, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        cached_aux = cuda.local.array(1, precision)
        base_state = cuda.local.array(2, precision)
        jvp = cuda.local.array(2, precision)
        fn(
            state,
            parameters,
            drivers,
            cached_aux,
            base_state,
            precision(0.0),
            precision(0.5),
            precision(0.5),
            v,
            out,
            jvp,
        )

    stream = default_memmgr.get_group_stream()
    v = np.asarray([3.0, -7.0], dtype=precision)
    v_dev = cuda.to_device(v, stream=stream)
    out_dev = cuda.to_device(
        np.zeros(2, dtype=precision), stream=stream
    )
    kernel[1, 1, stream](v_dev, out_dev)
    out = out_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert np.all(out == v)


def _dae_lu_backward_error(system, precision, h_value, a_ij_value, seed):
    """Solve W x = rhs with the plain LU; return the backward
    error against the operator-assembled W."""
    lu_member = system.get_solver_helper(
        "lu_solve", jacobian_at="stage"
    )
    op_member = system.get_solver_helper(
        "linear_operator", jacobian_at="stage"
    )
    lu_solve = lu_member.device_function
    operator = op_member.device_function

    n = len(system.states.names)
    base = system.states.values_array.astype(precision)
    rng = np.random.default_rng(seed)
    rhs = rng.normal(size=n).astype(precision)
    params = system.parameters.values_array.astype(precision)
    h_typed = precision(h_value)
    a_typed = precision(a_ij_value)
    t_typed = precision(0.0)
    factor_len = max(lu_member.lu_nnz, 1)

    @cuda.jit
    def kernel(base, params, rhs, x, factor, wmat, unit, wcol):
        state = cuda.local.array(n, precision)
        drivers = cuda.local.array(1, precision)
        cached = cuda.local.array(1, precision)
        for i in range(n):
            state[i] = precision(0.0)
        lu_solve(
            state, params, drivers, cached, base, t_typed,
            h_typed, a_typed, rhs, x, factor,
        )
        for j in range(n):
            for i in range(n):
                unit[i] = precision(0.0)
            unit[j] = precision(1.0)
            operator(
                state, params, drivers, cached, base, t_typed,
                h_typed, a_typed, unit, wcol,
            )
            for i in range(n):
                wmat[i, j] = wcol[i]

    stream = default_memmgr.get_group_stream()
    base_dev = cuda.to_device(base, stream=stream)
    params_dev = cuda.to_device(params, stream=stream)
    rhs_dev = cuda.to_device(rhs, stream=stream)
    x_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    factor_dev = cuda.to_device(
        np.zeros(factor_len, dtype=precision), stream=stream
    )
    wmat_dev = cuda.to_device(
        np.zeros((n, n), dtype=precision), stream=stream
    )
    unit_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    wcol_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    kernel[1, 1, stream](
        base_dev, params_dev, rhs_dev, x_dev, factor_dev,
        wmat_dev, unit_dev, wcol_dev,
    )
    x_out = x_dev.copy_to_host(stream=stream).astype(np.float64)
    wmat = wmat_dev.copy_to_host(stream=stream).astype(np.float64)
    stream.synchronize()
    assert np.all(np.isfinite(x_out))
    residual = wmat @ x_out - rhs.astype(np.float64)
    scale = (
        np.linalg.norm(wmat, np.inf)
        * np.linalg.norm(x_out, np.inf)
        + np.linalg.norm(rhs.astype(np.float64), np.inf)
    )
    return np.linalg.norm(residual, np.inf) / scale


def test_lu_solve_offslot_chain_backward_stable(
    precision, tolerance
):
    """The zero-diagonal chain-boundary slot solves backward-stably
    at both step sizes."""
    system = build_diode_line_system(precision)
    for h_value, seed in ((1e-3, 21), (1e-5, 22)):
        eta = _dae_lu_backward_error(
            system, precision, h_value, 0.4358665, seed
        )
        assert eta < tolerance.rel_loose


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"precision": np.float64}],
    indirect=True,
)
def test_lu_solve_introduced_state_slots_backward_stable(
    precision, tolerance
):
    """Introduced derivative-state slots solve backward-stably."""
    system = build_transistor_amplifier_system(precision)
    eta = _dae_lu_backward_error(
        system, precision, 1e-4, 0.4358665, 23
    )
    assert eta < tolerance.rel_loose
