from hashlib import sha256

import numpy as np
import pytest
import sympy as sp
from cubie.cuda_simsafe import cuda, numba_from_dtype as from_dtype

from cubie.odesystems.symbolic.codegen import (
    generate_cached_jvp_code,
    generate_cached_operator_apply_code,
    generate_jacobi_preconditioner_cached_code,
    generate_jacobi_preconditioner_code,
    generate_neumann_preconditioner_cached_code,
    generate_neumann_preconditioner_code,
    generate_operator_apply_code,
    generate_prepare_jac_code,
    generate_stage_residual_code,
)
from cubie.odesystems.solver_helpers import SolverHelperRequest
from cubie.odesystems.symbolic.engine import convert_assignments
from cubie.odesystems.symbolic.engine import expr as ir_expr
from cubie.odesystems.symbolic.helper_registry import helper_source_hash
from cubie.odesystems.symbolic.parsing import (
    JVPEquations as _JVPEquations,
)
from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
    plan_auxiliary_cache,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from tests._utils import FLOAT64_PRECISION
from tests._utils import (
    COLLIDING_CONSTANTS_F32,
    COLLIDING_CONSTANTS_F64,
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
        code = generate_operator_apply_code(
            system.equations,
            system.indices,
            M=M,
            func_name=fname,
        )
        op_fac, was_cached = system.gen_file.import_function(fname, code)
        return op_fac(
            system.constants.values_dict,
            from_dtype(system.precision),
            beta=beta,
            gamma=gamma,
        )

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
            # base_state is provided by caller (can be empty placeholder)
            op(state, parameters, drivers, base_state, t, h, a_ij, vec, out)

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
        prepare = prep_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
        )
        return prepare, aux_count

    return factory


@pytest.fixture(scope="session")
def cached_jvp_factory(cached_system, precision):
    """Return a factory producing calculate_cached_jvp device functions."""

    def factory():
        fname = "cached_jvp_factory"
        code = generate_cached_jvp_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        jvp_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return jvp_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
        )

    return factory


@pytest.fixture(scope="session")
def cached_jvp_kernel(cached_system, precision):
    """Apply cached JVP outputs for comparison with analytic Jacobian."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, cached_jvp, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            vec,
            out_cached,
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

            prepare(state, parameters, drivers, t, cached_aux)
            cached_jvp(
                state,
                parameters,
                drivers,
                cached_aux,
                t,
                vec,
                out_cached,
            )

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def cached_operator_factory(cached_system, precision):
    """Return a factory producing cached operator device functions."""

    def factory(beta, gamma, M):
        fname = (
            "cached_operator_factory_"
            f"{_stable_factory_tag(beta, gamma, M.tobytes())}"
        )
        code = generate_cached_operator_apply_code(
            cached_system.equations,
            cached_system.indices,
            M=M,
            func_name=fname,
        )
        op_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return op_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
        )

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

            prepare(state, parameters, drivers, t, cached_aux)
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
        (1.0, 1.0, 1.0, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, np.array([[1.0, 0.5], [0.5, 2.0]])),
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
    """Evaluate operator_apply for specific scalings and mass matrices."""

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
    code = generate_operator_apply_code(
        operator_system.equations, operator_system.indices
    )
    assert "_cubie_codegen_const_" not in code
    assert "constants[" not in code


def test_cached_jvp_matches_jacobian(
    cached_system,
    prepare_jac_factory,
    cached_jvp_factory,
    cached_jvp_kernel,
    precision,
    tolerance,
):
    """Ensure cached JVP equals the analytic Jacobian-vector product."""

    prepare, aux_count = prepare_jac_factory()
    cached_jvp = cached_jvp_factory()
    kernel = cached_jvp_kernel(prepare, cached_jvp, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.8, -1.1], dtype=precision)
    vec = vec[:state_len]
    out_cached = np.zeros(state_len, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        vec,
        out_cached,
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
    expected = jacobian @ vec

    assert np.allclose(
        out_cached,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 0.25, np.eye(2)),
        (1.0, 1.0, 0.25, np.diag([1.2, 0.8])),
        (0.5, 1.7, 0.15, np.array([[1.0, 0.3], [0.4, 1.5]])),
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
        )
        pre_fac, was_cached = operator_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            operator_system.constants.values_dict,
            from_dtype(operator_system.precision),
            beta=beta,
            gamma=gamma,
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def neumann_kernel(precision):
    """Apply the Neumann preconditioner to a vector, passing scratch."""

    n = 2

    def make_kernel(pre):
        @cuda.jit
        def kernel(t, h, a_ij, vec, base_state, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            jvp = cuda.local.array(n, precision)
            scratch = cuda.local.array(n, precision)
            chain_scratch = cuda.local.array(n, precision)
            pre(
                state,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
                scratch,
                chain_scratch,
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
        code = generate_neumann_preconditioner_cached_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
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
            scratch = cuda.local.array(n_state, precision)
            chain_scratch = cuda.local.array(n_state, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
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
                scratch,
                chain_scratch,
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
            f"{_stable_factory_tag(M.tobytes())}"
        )
        code = generate_stage_residual_code(
            operator_system.equations,
            operator_system.indices,
            M=M,
            func_name=fname,
        )
        res_fac, was_cached = operator_system.gen_file.import_function(
            fname, code
        )
        return res_fac(
            operator_system.constants.values_dict,
            from_dtype(operator_system.precision),
            beta=beta,
            gamma=gamma,
        )

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
        (1.0, 1.0, 1.0, 0.5, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, 0.25, np.array([[1.0, 0.5], [0.5, 2.0]])),
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
        SolverHelperRequest(kind="stage_residual", beta=1.0, gamma=1.0)
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
            SolverHelperRequest(
                kind="stage_residual", beta=beta, gamma=gamma
            )
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
    first_request = SolverHelperRequest(
        kind="neumann_preconditioner",
        beta=1.0,
        gamma=1.0,
        preconditioner_order=1,
    )
    second_request = SolverHelperRequest(
        kind="neumann_preconditioner",
        beta=1.0,
        gamma=1.0,
        preconditioner_order=2,
    )
    first = system.get_solver_helper(first_request)
    second = system.get_solver_helper(second_request)

    assert first is not second
    assert first.device_function is not second.device_function
    assert helper_source_hash(
        system, first_request
    ) == helper_source_hash(system, second_request)


@pytest.mark.parametrize(
    "solver_settings_override",
    [LINEAR_SYSTEM],
    indirect=True,
)
def test_helper_requests_reuse_members_without_touching_settings(system):
    """Repeated requests reuse members; settings stay untouched."""

    settings_before = system.compile_settings
    hash_before = system.config_hash

    scaled_request = SolverHelperRequest(
        kind="linear_operator", beta=2.5, gamma=0.5
    )
    scaled = system.get_solver_helper(scaled_request)
    first = system.get_solver_helper(
        SolverHelperRequest(kind="prepare_jac")
    )
    second = system.get_solver_helper(
        SolverHelperRequest(kind="prepare_jac")
    )
    repeat_scaled = system.get_solver_helper(scaled_request)

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

    cached = system.get_solver_helper(
        SolverHelperRequest(kind="prepare_jac")
    )

    with pytest.raises(ValueError):
        SolverHelperRequest(kind="not_a_helper")

    assert (
        system.get_solver_helper(
            SolverHelperRequest(kind="prepare_jac")
        )
        is cached
    )


@pytest.fixture(scope="session")
def jacobi_factory(cached_system, precision):
    """Return a factory producing Jacobi preconditioner device functions."""

    def factory(beta, gamma, M=None):
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
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
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
            jvp = cuda.local.array(n, precision)
            scratch = cuda.local.array(n, precision)
            chain_scratch = cuda.local.array(n, precision)
            for idx in range(n):
                state[idx] = state_values[idx]
            pre(
                state,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
                scratch,
                chain_scratch,
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
    return pre_fac(
        operator_system.constants.values_dict,
        from_dtype(operator_system.precision),
        beta=1.0,
        gamma=1.0,
    )


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


def test_chained_preconditioner_composition(
    operator_system,
    precision,
    tolerance,
):
    """Chained ["neumann", "jacobi"] equals jacobi(neumann(v)).

    The composed generated helper feeds P0's output into P1, so it
    must reproduce sequential application of the individually
    generated preconditioners.
    """
    kwargs = {
        "beta": 1.0,
        "gamma": 1.0,
        "preconditioner_order": 1,
    }
    neumann = operator_system.get_solver_helper(
        SolverHelperRequest(kind="neumann_preconditioner", **kwargs)
    ).device_function
    jacobi = operator_system.get_solver_helper(
        SolverHelperRequest(kind="jacobi_preconditioner", **kwargs)
    ).device_function
    chained = operator_system.get_solver_helper(
        SolverHelperRequest(
            kind="chained_preconditioner",
            chained_kinds=(
                "neumann_preconditioner",
                "jacobi_preconditioner",
            ),
            **kwargs,
        )
    ).device_function

    n = 2

    @cuda.jit
    def kernel(t, h, a_ij, vec, base_state, out_chained, out_seq):
        state = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        jvp = cuda.local.array(n, precision)
        scratch = cuda.local.array(n, precision)
        chain_scratch = cuda.local.array(n, precision)
        intermediate = cuda.local.array(n, precision)
        for idx in range(n):
            state[idx] = precision(0.0)
        chained(
            state, parameters, drivers, base_state,
            t, h, a_ij, vec, out_chained, jvp, scratch,
            chain_scratch,
        )
        neumann(
            state, parameters, drivers, base_state,
            t, h, a_ij, vec, intermediate, jvp, scratch,
            chain_scratch,
        )
        jacobi(
            state, parameters, drivers, base_state,
            t, h, a_ij, intermediate, out_seq, jvp, scratch,
            chain_scratch,
        )

    v = np.array([0.7, -1.3], dtype=precision)
    base = np.zeros(2, dtype=precision)
    out_chained = np.zeros(2, dtype=precision)
    out_seq = np.zeros(2, dtype=precision)

    kernel[1, 1](
        precision(0.0),
        precision(0.25),
        precision(0.5),
        v,
        base,
        out_chained,
        out_seq,
    )

    assert np.allclose(
        out_chained,
        out_seq,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_three_stage_chained_preconditioner_composition(
    operator_system,
    precision,
    tolerance,
):
    """A three-type chain equals sequential stage application."""
    kwargs = {
        "beta": 1.0,
        "gamma": 1.0,
        "preconditioner_order": 1,
    }
    neumann = operator_system.get_solver_helper(
        SolverHelperRequest(kind="neumann_preconditioner", **kwargs)
    ).device_function
    jacobi = operator_system.get_solver_helper(
        SolverHelperRequest(kind="jacobi_preconditioner", **kwargs)
    ).device_function
    chained = operator_system.get_solver_helper(
        SolverHelperRequest(
            kind="chained_preconditioner",
            chained_kinds=(
                "neumann_preconditioner",
                "jacobi_preconditioner",
                "neumann_preconditioner",
            ),
            **kwargs,
        )
    ).device_function

    n = 2

    @cuda.jit
    def kernel(t, h, a_ij, vec, base_state, out_chained, out_seq):
        state = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        jvp = cuda.local.array(n, precision)
        scratch = cuda.local.array(n, precision)
        chain_scratch = cuda.local.array(n, precision)
        step_one = cuda.local.array(n, precision)
        step_two = cuda.local.array(n, precision)
        for idx in range(n):
            state[idx] = precision(0.0)
        chained(
            state, parameters, drivers, base_state,
            t, h, a_ij, vec, out_chained, jvp, scratch,
            chain_scratch,
        )
        neumann(
            state, parameters, drivers, base_state,
            t, h, a_ij, vec, step_one, jvp, scratch,
            chain_scratch,
        )
        jacobi(
            state, parameters, drivers, base_state,
            t, h, a_ij, step_one, step_two, jvp, scratch,
            chain_scratch,
        )
        neumann(
            state, parameters, drivers, base_state,
            t, h, a_ij, step_two, out_seq, jvp, scratch,
            chain_scratch,
        )

    v = np.array([0.7, -1.3], dtype=precision)
    base = np.zeros(2, dtype=precision)
    out_chained = np.zeros(2, dtype=precision)
    out_seq = np.zeros(2, dtype=precision)

    kernel[1, 1](
        precision(0.0),
        precision(0.25),
        precision(0.5),
        v,
        base,
        out_chained,
        out_seq,
    )

    assert np.allclose(
        out_chained,
        out_seq,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.fixture(scope="session")
def jacobi_cached_factory(cached_system, precision):
    """Return a factory producing cached Jacobi preconditioners."""

    def factory(beta, gamma):
        fname = (
            "jacobi_cached_factory_"
            f"{int(beta * 10)}_{int(gamma * 10)}"
        )
        code = generate_jacobi_preconditioner_cached_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        pre_fac, was_cached = cached_system.gen_file.import_function(
            fname, code
        )
        return pre_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
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
    """Jacobi divides by ``beta*M_ii - gamma*h*a_ij*J_ii``.

    A diagonal mass matrix scales the beta term per component;
    off-diagonal mass entries are ignored by the diagonal
    preconditioner.
    """
    beta, gamma, h, a_ij = 1.0, 1.0, 0.2, 0.5
    mass = np.diag([2.0, 3.0])
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


def test_mass_matrix_selects_distinct_cached_helpers(
    jacobi_kernel,
    precision,
    tolerance,
):
    """Systems differing only in mass generate distinct helpers.

    The generated source bakes mass entries in, and the mass matrix
    is part of the system definition (folded into ``fn_hash``), so a
    same-named system with a different mass matrix must not reuse the
    other system's cached device function (in memory or from the
    generated-code file on disk).
    """
    equations = [
        "dx0 = -k0*x0 + x0*x1",
        "dx1 = -k1*x1 + x0*x0",
    ]
    mass = np.diag([2.0, 3.0]).astype(precision)
    system = create_ODE_system(
        equations,
        states=["x0", "x1"],
        constants={"k0": 1.0, "k1": 2.0},
        precision=precision,
        name="mass_cache_key_sys",
    )

    h, a_ij = 0.2, 0.5
    state = np.array([0.3, -0.6], dtype=precision)
    base = np.array([0.1, 0.2], dtype=precision)
    v = np.array([0.7, -1.3], dtype=precision)
    eval_point = base + a_ij * state
    # J00 = -k0 + x1, J11 = -k1 at the evaluation point
    diag_j = np.array([-1.0 + eval_point[1], -2.0])

    jacobi_request = SolverHelperRequest(
        kind="jacobi_preconditioner", beta=1.0, gamma=1.0
    )
    pre_eye = system.get_solver_helper(jacobi_request).device_function
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

    # A same-named system with a different mass shares the base
    # equation identity but re-keys mass-consuming helper source, so
    # its Jacobi helper never aliases the identity-mass helper cached
    # above.
    system_mass = create_ODE_system(
        equations,
        states=["x0", "x1"],
        constants={"k0": 1.0, "k1": 2.0},
        precision=precision,
        name="mass_cache_key_sys",
    )
    system_mass.update_compile_settings(mass=mass)
    assert system_mass.fn_hash == system.fn_hash
    assert helper_source_hash(
        system_mass, jacobi_request
    ) != helper_source_hash(system, jacobi_request)
    pre_mass = system_mass.get_solver_helper(
        jacobi_request
    ).device_function
    out_mass = np.zeros(2, dtype=precision)
    jacobi_kernel(pre_mass)[1, 1](
        precision(0.0),
        precision(h),
        precision(a_ij),
        state,
        base,
        v,
        out_mass,
    )

    expected_eye = v / (1.0 - h * a_ij * diag_j)
    expected_mass = v / (np.diag(mass) - h * a_ij * diag_j)

    assert np.allclose(
        out_eye,
        expected_eye,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )
    assert np.allclose(
        out_mass,
        expected_mass,
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
            prepare(state, parameters, drivers, t, cached_aux)
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
            scratch = cuda.local.array(n_state, precision)
            chain_scratch = cuda.local.array(n_state, precision)
            for idx in range(n_state):
                state[idx] = state_values[idx]
            prepare(state, parameters, drivers, t, cached_aux)
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
                scratch,
                chain_scratch,
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
    cached_request = SolverHelperRequest(kind="prepare_jac")
    plain_request = SolverHelperRequest(
        kind="linear_operator", beta=1.0, gamma=1.0
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
        SolverHelperRequest(kind="prepare_jac")
    )
    prepare = prepare_helper.device_function
    aux_count = prepare_helper.cached_auxiliary_count
    assert aux_count is not None
    assert aux_count > 0

    cached_op = system.get_solver_helper(
        SolverHelperRequest(
            kind="linear_operator_cached", beta=1.0, gamma=1.0
        )
    ).device_function
    inline_op = system.get_solver_helper(
        SolverHelperRequest(
            kind="linear_operator_at_state", beta=1.0, gamma=1.0
        )
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
        SolverHelperRequest(kind="prepare_jac")
    )
    prepare = prepare_helper.device_function
    aux_count = prepare_helper.cached_auxiliary_count

    jacobi = system.get_solver_helper(
        SolverHelperRequest(
            kind="jacobi_preconditioner_cached", beta=1.0, gamma=1.0
        )
    ).device_function

    kernel = system_cached_precond_kernel(prepare, jacobi, aux_count)

    h = precision(0.25)
    a_ij = precision(1.0)
    state_values = np.array([-62.0, 0.07, 0.55, 0.34], dtype=precision)
    vec = np.array([0.8, -1.1, 0.4, -0.3], dtype=precision)
    out = np.zeros(n, dtype=precision)
    kernel[1, 1](state_values, precision(0.0), h, a_ij, vec, out)

    vm, m, hg, nn = (float(value) for value in state_values)
    constants = system.constants.values_dict
    alpha_m = 0.1 * (vm + 40.0) / (1.0 - np.exp(-(vm + 40.0) / 10.0))
    beta_m = 4.0 * np.exp(-(vm + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(vm + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(vm + 35.0) / 10.0))
    alpha_n = 0.01 * (vm + 55.0) / (1.0 - np.exp(-(vm + 55.0) / 10.0))
    beta_n = 0.125 * np.exp(-(vm + 65.0) / 80.0)
    diag_j = np.array(
        [
            -(
                constants["g_na"] * m**3 * hg
                + constants["g_k"] * nn**4
                + constants["g_l"]
            )
            / constants["c_m"],
            -(alpha_m + beta_m),
            -(alpha_h + beta_h),
            -(alpha_n + beta_n),
        ]
    )
    expected = vec / (1.0 - float(h) * float(a_ij) * diag_j)

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )

