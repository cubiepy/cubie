"""Tests for the Neumann preconditioner convergence diagnostic.

Covers the pure-numeric :func:`neumann_spectral_radius`,
:func:`check_neumann_convergence`, the wiring into
:meth:`SymbolicODE.get_solver_helper`, and the per-consumer cache
policies selecting independent evaluators on a shared system.
"""

import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import pytest

from cubie.odesystems.solver_helpers import SolverHelperRequest
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.cubie_cache import CachePolicy, CUBIECache
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    check_neumann_convergence,
    neumann_spectral_radius,
)
from tests._utils import (
    DIAGONALLY_DOMINANT,
    GATING_SINGULARITY,
    OFF_DIAGONAL_HEAVY,
    SINGULAR_INITIAL_STATE,
)


# Keys every check_neumann_convergence return must expose, regardless of
# which path produced it.
_RESULT_KEYS = {
    "rho_per_unit_step_factor",
    "critical_step_factor",
    "step_factor",
    "rho_series",
    "series_converges",
}


@pytest.mark.parametrize(
    "solver_settings_override", [DIAGONALLY_DOMINANT], indirect=True
)
def test_small_step_converges_for_diagonally_dominant_system(system):
    """A sufficiently small supplied step reports convergence."""
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(CachePolicy()),
        step_size=1e-4,
        stage_coefficients=1.0,
    )
    assert result["series_converges"] is True
    assert result["rho_series"] < 1.0
    assert set(result) == _RESULT_KEYS


@pytest.mark.parametrize(
    "solver_settings_override", [OFF_DIAGONAL_HEAVY], indirect=True
)
def test_large_step_diverges_for_off_diagonal_heavy_system(system):
    """A supplied step beyond the critical magnitude warns."""
    with pytest.warns(UserWarning):
        result = check_neumann_convergence(
            system.indices,
            system._get_neumann_evaluator(CachePolicy()),
            step_size=1.0,
            stage_coefficients=1.0,
        )
    assert result["series_converges"] is False
    assert result["rho_series"] >= 1.0
    assert result["critical_step_factor"] <= 1.0


@pytest.mark.parametrize(
    "solver_settings_override", [GATING_SINGULARITY], indirect=True
)
def test_gating_singularity_converges_without_false_divergence(system):
    """Guarded gating terms do not trigger a false divergence report."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = check_neumann_convergence(
            system.indices,
            system._get_neumann_evaluator(CachePolicy()),
            step_size=1e-4,
            stage_coefficients=1.0,
        )
    assert result["series_converges"] is True
    assert result["rho_series"] < 1.0


@pytest.mark.parametrize(
    "solver_settings_override", [SINGULAR_INITIAL_STATE], indirect=True
)
def test_non_finite_jacobian_reports_not_verified(system):
    """A non-finite Jacobian yields a nan radius and no verdict."""
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(CachePolicy()),
    )
    assert result["series_converges"] is None
    assert np.isnan(result["rho_per_unit_step_factor"])
    assert set(result) == _RESULT_KEYS


@pytest.mark.parametrize(
    "solver_settings_override", [OFF_DIAGONAL_HEAVY], indirect=True
)
def test_get_solver_helper_runs_diagnostic_for_neumann_type(system):
    """Static helper check reports a step limit, not divergence."""
    with pytest.warns(UserWarning, match="not a divergence verdict"):
        system.get_solver_helper(
            SolverHelperRequest(
                kind="neumann_preconditioner", beta=1.0, gamma=1.0
            )
        )


def _seed_kernel_cache(*target_dirs):
    """Copy the shared kernel-cache artifact into per-test dirs.

    A per-test cache directory starts empty, so a diagnostic launch
    in the CI consumer leg would cold-compile there and trip the
    zero-compile gate. Each directory receives a private copy of the
    environment kernel cache: isolation between the directories is
    preserved while their kernels load instead of compiling. Without
    the environment cache the directories stay empty and launches
    compile as usual.
    """
    shared = os.environ.get("CUBIE_KERNEL_CACHE_DIR", "").strip()
    if not shared or not Path(shared).is_dir():
        return
    # Lock files are live inter-process state, not cache content:
    # another worker may hold a byte-range lock that makes the copy
    # raise on Windows.
    skip_locks = shutil.ignore_patterns("*.lock")
    for target in target_dirs:
        shutil.copytree(
            shared, target, dirs_exist_ok=True, ignore=skip_locks
        )


@pytest.mark.parametrize(
    "solver_settings_override", [DIAGONALLY_DOMINANT], indirect=True
)
def test_kernel_cache_policies_stay_isolated(system, tmp_path):
    """Two live kernels with distinct policies never interfere.

    Each kernel's helper getter routes its own policy into helper
    validation, so the shared system keys one evaluator per policy
    and a runtime cache update on one kernel leaves the other
    kernel's diagnostic service untouched.
    """
    dir_a = tmp_path / "kernel_a"
    dir_b = tmp_path / "kernel_b"
    _seed_kernel_cache(dir_a, dir_b, tmp_path / "kernel_a_moved")
    kernel_a = BatchSolverKernel(
        system,
        algorithm_settings={"algorithm": "euler"},
        cache=dir_a,
    )
    kernel_b = BatchSolverKernel(
        system,
        algorithm_settings={"algorithm": "euler"},
        cache=dir_b,
    )
    try:
        request = SolverHelperRequest(
            kind="neumann_preconditioner", beta=1.0, gamma=1.0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel_a._solver_helper_fn(request)
            kernel_b._solver_helper_fn(request)

        policy_a = kernel_a.cache_handler.policy
        policy_b = kernel_b.cache_handler.policy
        evaluator_a = system._neumann_diagnostics[policy_a]
        evaluator_b = system._neumann_diagnostics[policy_b]
        assert evaluator_a is not evaluator_b
        assert evaluator_a.cache_policy.cache_dir == dir_a
        assert evaluator_b.cache_policy.cache_dir == dir_b

        # A runtime cache update on A rebinds only A's getter; B's
        # evaluator keeps its policy and its built kernel.
        built_b = evaluator_b.get_cached_output("evaluation_kernel")
        kernel_a.update(cache_dir=tmp_path / "kernel_a_moved")
        assert evaluator_b.cache_policy.cache_dir == dir_b
        kept_b = evaluator_b.get_cached_output("evaluation_kernel")
        assert kept_b is built_b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel_a._solver_helper_fn(request)
        moved = kernel_a.cache_handler.policy
        assert (
            system._neumann_diagnostics[moved].cache_policy.cache_dir
            == tmp_path / "kernel_a_moved"
        )
        assert evaluator_b.cache_policy.cache_dir == dir_b
    finally:
        kernel_a.close()
        kernel_b.close()


@pytest.mark.parametrize(
    "solver_settings_override", [DIAGONALLY_DOMINANT], indirect=True
)
def test_policy_keyed_evaluators_are_stable_per_policy(system, tmp_path):
    """Equal policies share one evaluator; distinct policies do not."""
    default_first = system._get_neumann_evaluator(CachePolicy())
    first = default_first.get_cached_output("evaluation_kernel")
    again = system._get_neumann_evaluator(CachePolicy()).get_cached_output(
        "evaluation_kernel"
    )
    assert again is first

    policy = CachePolicy(cache_enabled=True, cache_dir=tmp_path)
    other = system._get_neumann_evaluator(policy)
    assert other is not default_first
    # An equal policy resolves to the same evaluator and keeps its
    # built kernel; the default-policy evaluator is untouched.
    built = other.get_cached_output("evaluation_kernel")
    same_policy = CachePolicy(cache_enabled=True, cache_dir=tmp_path)
    kept = system._get_neumann_evaluator(same_policy).get_cached_output(
        "evaluation_kernel"
    )
    assert kept is built
    assert (
        system._get_neumann_evaluator(CachePolicy()).get_cached_output(
            "evaluation_kernel"
        )
        is first
    )


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override", [DIAGONALLY_DOMINANT], indirect=True
)
def test_evaluator_attaches_configured_disk_cache(system, tmp_path):
    """With caching enabled the built kernel carries a CUBIECache."""
    evaluator = system._get_neumann_evaluator(
        CachePolicy(cache_enabled=True, cache_dir=tmp_path)
    )
    kernel = evaluator.get_cached_output("evaluation_kernel")
    assert isinstance(kernel._cache, CUBIECache)
    assert Path(kernel._cache.cache_path).parent == tmp_path


def test_spectral_radius_tracks_beta():
    """beta scales the implemented Neumann series matrix."""
    jacobian = np.array([[-1.0, 100.0], [100.0, -1.0]])
    diverging = neumann_spectral_radius(
        jacobian,
        beta=1.0,
        gamma=1.0,
        stage_coefficients=1.0,
        step_factor_value=1.0,
    )
    converging = neumann_spectral_radius(
        jacobian,
        beta=1000.0,
        gamma=1.0,
        stage_coefficients=1.0,
        step_factor_value=1.0,
    )
    assert diverging["series_converges"] is False
    assert diverging["rho_series"] >= 1.0
    assert converging["series_converges"] is True
    assert converging["rho_series"] < 1.0


def test_spectral_radius_reports_exact_critical_step():
    """The static limit is the reciprocal unit-step radius."""
    jacobian = np.array([[0.0, 2.0], [-3.0, 0.0]])
    static = neumann_spectral_radius(
        jacobian, beta=2.0, gamma=3.0, stage_coefficients=0.25
    )
    critical = static["critical_step_factor"]

    below = neumann_spectral_radius(
        jacobian,
        beta=2.0,
        gamma=3.0,
        stage_coefficients=0.25,
        step_factor_value=0.5 * critical,
    )
    above = neumann_spectral_radius(
        jacobian,
        beta=2.0,
        gamma=3.0,
        stage_coefficients=0.25,
        step_factor_value=2.0 * critical,
    )

    assert static["step_factor"] == "h"
    assert np.isclose(
        critical,
        1.0 / static["rho_per_unit_step_factor"],
    )
    assert static["rho_series"] is None
    assert static["series_converges"] is None
    assert np.isclose(below["rho_series"], 0.5)
    assert below["series_converges"] is True
    assert np.isclose(above["rho_series"], 2.0)
    assert above["series_converges"] is False


def test_single_stage_without_coefficient_reports_effective_step():
    """Unknown runtime a_ij yields a bound on abs(a_ij*h)."""
    result = neumann_spectral_radius(np.array([[4.0]]))
    assert result["step_factor"] == "a_ij * h"
    assert np.isclose(result["rho_per_unit_step_factor"], 4.0)
    assert np.isclose(result["critical_step_factor"], 0.25)


def test_zero_jacobian_reports_unbounded_critical_step():
    """A zero Jacobian (constant RHS) converges for every factor."""
    result = neumann_spectral_radius(
        np.zeros((2, 2)), step_factor_value=1e6
    )
    assert result["rho_per_unit_step_factor"] == 0.0
    assert result["critical_step_factor"] == float("inf")
    assert result["series_converges"] is True


def test_nontrivial_firk_tableau_uses_kronecker_coupling():
    """FIRK radius matches the full A-tensor-J spectrum."""
    jacobian = np.array([[2.0, 1.0], [-1.0, 3.0]])
    coefficients = np.array([[0.2, -0.1], [0.4, 0.3]])
    expected = max(abs(np.linalg.eigvals(np.kron(coefficients, jacobian))))
    result = neumann_spectral_radius(
        jacobian,
        stage_coefficients=coefficients,
    )
    assert result["step_factor"] == "h"
    assert np.isclose(result["rho_per_unit_step_factor"], expected)


def test_spectral_radius_single_stage_matches_unit_tableau():
    """A one-entry unit tableau reproduces the single-stage result."""
    jacobian = np.array([[-2.0, 0.5], [0.5, -2.0]])
    single = neumann_spectral_radius(jacobian, beta=1.0, gamma=1.0)
    staged = neumann_spectral_radius(
        jacobian, beta=1.0, gamma=1.0, stage_coefficients=[[1.0]]
    )
    assert np.isclose(
        single["rho_per_unit_step_factor"],
        staged["rho_per_unit_step_factor"],
    )
