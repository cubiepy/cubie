"""Fused operator-preconditioner helpers match the sequential pair.

Every fused variant must produce the same ``z = P(v)`` and
``out = A(z)`` as the corresponding unfused preconditioner followed
by the unfused operator, for every supported preconditioner member
set. The fused body is a differently ordered compilation of the same
algebra, so agreement is asserted to floating tolerances scaled to
the configured precision rather than bitwise.
"""

import numpy as np
import pytest

from cubie.cuda_simsafe import cuda
from cubie.odesystems.solver_helpers import (
    SolverHelperKind,
    SolverHelperRequest,
    resolve_fused_kind,
)
from cubie.odesystems.symbolic.helper_registry import (
    helper_source_hash,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

STAGE_A = ((5.0 / 12.0, -1.0 / 12.0), (3.0 / 4.0, 1.0 / 4.0))
STAGE_C = (1.0 / 3.0, 1.0)

BETA = 0.9
GAMMA = 0.8
ORDER = 2
T_VALUE = 0.2
H_VALUE = 0.01
A_IJ_VALUE = 0.435


@pytest.fixture(scope="session")
def fused_system(precision):
    """Nonlinear system with an observable and a shared auxiliary."""

    dxdt = [
        "aux = x0 * x1 + p0",
        "obs1 = aux * x2",
        "dx0 = -p0 * x0 + x1 * x2 + obs1",
        "dx1 = p1 * (x0 - x1) - aux",
        "dx2 = x0 * x1 - c0 * x2 + aux * aux",
    ]
    system = create_ODE_system(
        dxdt,
        states={"x0": 1.0, "x1": 0.5, "x2": 2.0},
        parameters={"p0": 0.3, "p1": 1.7},
        constants={"c0": 2.66},
        observables=["obs1"],
        name="fused_operator_test_system",
        precision=precision,
    )
    return system


def _tolerances(precision):
    if precision == np.float64:
        return {"rtol": 1e-12, "atol": 1e-13}
    return {"rtol": 2e-5, "atol": 2e-6}


def _member_kinds(names, cached=False, n_stage=False, at_state=False):
    prefix = "n_stage_" if n_stage else ""
    suffix = "_cached" if cached else "_at_state" if at_state else ""
    return tuple(
        SolverHelperKind(f"{prefix}{name}_preconditioner{suffix}")
        for name in names
    )


def _request_kwargs(n_stage=False):
    kwargs = {
        "beta": BETA,
        "gamma": GAMMA,
        "preconditioner_order": ORDER,
    }
    if n_stage:
        kwargs["stage_coefficients"] = STAGE_A
        kwargs["stage_nodes"] = STAGE_C
    return kwargs


def _reference_preconditioner(system, members, kwargs):
    if len(members) == 1:
        request = SolverHelperRequest(kind=members[0], **kwargs)
    else:
        chained_value = members[0].value
        for token in ("jacobi", "neumann"):
            chained_value = chained_value.replace(token, "chained")
        request = SolverHelperRequest(
            kind=SolverHelperKind(chained_value),
            chained_kinds=members,
            **kwargs,
        )
    return system.get_solver_helper(request).device_function


@pytest.fixture(scope="session")
def fused_comparison_kernel(fused_system, precision):
    """Kernel running fused and sequential paths on one input."""

    n_state = len(fused_system.indices.states.index_map)
    n_params = len(fused_system.indices.parameters.index_map)
    param_len = max(n_params, 1)

    def make_kernel(fused, pc, op, prepare, aux_count, width):
        aux_len = max(aux_count, 1)
        driver_len = max(width // n_state, 1)
        cached = prepare is not None

        @cuda.jit
        def kernel(
            state_values,
            base_values,
            parameter_values,
            vec,
            z_fused,
            az_fused,
            z_ref,
            az_ref,
        ):
            state = cuda.local.array(width, precision)
            base_state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)
            jvp = cuda.local.array(width, precision)
            scratch = cuda.local.array(width, precision)
            chain_scratch = cuda.local.array(width, precision)

            for idx in range(width):
                state[idx] = state_values[idx]
            for idx in range(n_state):
                base_state[idx] = base_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(driver_len):
                drivers[idx] = precision(0.0)

            t = precision(T_VALUE)
            h = precision(H_VALUE)
            a_ij = precision(A_IJ_VALUE)

            if cached:
                prepare(state, parameters, drivers, t, cached_aux)
                pc(
                    state, parameters, drivers, cached_aux,
                    base_state, t, h, a_ij, vec, z_ref, jvp,
                    scratch, chain_scratch,
                )
                op(
                    state, parameters, drivers, cached_aux,
                    base_state, t, h, a_ij, z_ref, az_ref,
                )
                fused(
                    state, parameters, drivers, cached_aux,
                    base_state, t, h, a_ij, vec, z_fused, az_fused,
                )
            else:
                pc(
                    state, parameters, drivers, base_state, t, h,
                    a_ij, vec, z_ref, jvp, scratch, chain_scratch,
                )
                op(
                    state, parameters, drivers, base_state, t, h,
                    a_ij, z_ref, az_ref,
                )
                fused(
                    state, parameters, drivers, base_state, t, h,
                    a_ij, vec, z_fused, az_fused,
                )

        return kernel

    return make_kernel


def _run_case(
    fused_system,
    fused_comparison_kernel,
    precision,
    names,
    cached=False,
    n_stage=False,
    at_state=False,
):
    n_state = len(fused_system.indices.states.index_map)
    width = 2 * n_state if n_stage else n_state
    members = _member_kinds(
        names, cached=cached, n_stage=n_stage, at_state=at_state
    )
    kwargs = _request_kwargs(n_stage=n_stage)

    fused = fused_system.get_solver_helper(
        SolverHelperRequest(
            kind=resolve_fused_kind(
                cached=cached, n_stage=n_stage, at_state=at_state
            ),
            chained_kinds=members,
            **kwargs,
        )
    ).device_function
    pc = _reference_preconditioner(fused_system, members, kwargs)
    operator_value = "linear_operator"
    if n_stage:
        operator_value = "n_stage_linear_operator"
    elif cached:
        operator_value = "linear_operator_cached"
    elif at_state:
        operator_value = "linear_operator_at_state"
    op = fused_system.get_solver_helper(
        SolverHelperRequest(
            kind=SolverHelperKind(operator_value), **kwargs
        )
    ).device_function

    prepare = None
    aux_count = 0
    if cached:
        prepare_result = fused_system.get_solver_helper(
            SolverHelperRequest(kind=SolverHelperKind.PREPARE_JAC)
        )
        prepare = prepare_result.device_function
        aux_count = prepare_result.cached_auxiliary_count

    kernel = fused_comparison_kernel(
        fused, pc, op, prepare, aux_count, width
    )

    rng = np.random.default_rng(42)
    state = rng.normal(size=width).astype(precision)
    base = rng.normal(size=n_state).astype(precision)
    parameters = np.array([0.3, 1.7], dtype=precision)
    vec = rng.normal(size=width).astype(precision)
    z_fused = np.zeros(width, dtype=precision)
    az_fused = np.zeros(width, dtype=precision)
    z_ref = np.zeros(width, dtype=precision)
    az_ref = np.zeros(width, dtype=precision)

    kernel[1, 1](
        state, base, parameters, vec,
        z_fused, az_fused, z_ref, az_ref,
    )

    tolerances = _tolerances(precision)
    assert np.all(np.isfinite(z_ref)) and np.all(np.isfinite(az_ref))
    np.testing.assert_allclose(z_fused, z_ref, **tolerances)
    np.testing.assert_allclose(az_fused, az_ref, **tolerances)


@pytest.mark.parametrize(
    "names",
    [("jacobi",), ("neumann",), ("jacobi", "neumann")],
    ids=["jacobi", "neumann", "chain"],
)
def test_fused_matches_sequential(
    fused_system, fused_comparison_kernel, precision, names
):
    """Newton-Krylov fused variant equals preconditioner then operator."""
    _run_case(
        fused_system, fused_comparison_kernel, precision, names
    )


@pytest.mark.parametrize(
    "names", [("jacobi",), ("neumann",)], ids=["jacobi", "neumann"]
)
def test_fused_at_state_matches_sequential(
    fused_system, fused_comparison_kernel, precision, names
):
    """At-state fused variant equals the at-state sequential pair."""
    _run_case(
        fused_system,
        fused_comparison_kernel,
        precision,
        names,
        at_state=True,
    )


@pytest.mark.parametrize(
    "names", [("jacobi",), ("neumann",)], ids=["jacobi", "neumann"]
)
def test_fused_cached_matches_sequential(
    fused_system, fused_comparison_kernel, precision, names
):
    """Cached fused variant reads prepare_jac slots like the pair."""
    _run_case(
        fused_system,
        fused_comparison_kernel,
        precision,
        names,
        cached=True,
    )


@pytest.mark.parametrize(
    "names", [("jacobi",), ("neumann",)], ids=["jacobi", "neumann"]
)
def test_n_stage_fused_matches_sequential(
    fused_system, fused_comparison_kernel, precision, names
):
    """Flattened FIRK fused variant equals the n-stage pair."""
    _run_case(
        fused_system,
        fused_comparison_kernel,
        precision,
        names,
        n_stage=True,
    )


def test_fused_request_requires_members():
    """Fused kinds reject empty or foreign member sets."""
    with pytest.raises(ValueError):
        SolverHelperRequest(
            kind=SolverHelperKind.FUSED_OPERATOR_PRECONDITIONER,
        )
    with pytest.raises(ValueError):
        SolverHelperRequest(
            kind=SolverHelperKind.FUSED_OPERATOR_PRECONDITIONER,
            chained_kinds=(
                SolverHelperKind.JACOBI_PRECONDITIONER_CACHED,
            ),
        )


def test_fused_single_member_allowed():
    """One concrete member is a valid fused composition."""
    request = SolverHelperRequest(
        kind=SolverHelperKind.FUSED_OPERATOR_PRECONDITIONER,
        chained_kinds=(SolverHelperKind.JACOBI_PRECONDITIONER,),
    )
    assert request.chain_identity == ("jacobi_preconditioner",)


def test_fused_source_hash_order_sensitivity(fused_system):
    """Neumann members bake the unrolled order into source identity."""
    def request(names, order):
        return SolverHelperRequest(
            kind=SolverHelperKind.FUSED_OPERATOR_PRECONDITIONER,
            chained_kinds=_member_kinds(names),
            preconditioner_order=order,
        )

    neumann_low = helper_source_hash(
        fused_system, request(("neumann",), 1)
    )
    neumann_high = helper_source_hash(
        fused_system, request(("neumann",), 3)
    )
    assert neumann_low != neumann_high

    jacobi_low = helper_source_hash(
        fused_system, request(("jacobi",), 1)
    )
    jacobi_high = helper_source_hash(
        fused_system, request(("jacobi",), 3)
    )
    assert jacobi_low == jacobi_high
