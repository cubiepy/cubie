import numpy as np
import pytest

from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
    MatrixFreeSolver,
)
from cubie.integrators.norms import ScaledNorm


def test_matrix_free_solver_config_precision_property(precision):
    """Verify numba_precision and simsafe_precision properties work correctly.
    """
    config = MatrixFreeSolverConfig(precision=precision, solver_width=3)

    # Verify numba_precision returns correct type
    numba_prec = config.numba_precision
    assert numba_prec is not None

    # Verify simsafe_precision returns correct type
    simsafe_prec = config.simsafe_precision
    assert simsafe_prec is not None

    # Verify precision attribute is stored correctly
    assert config.precision == precision


def test_matrix_free_solver_config_validation():
    """Verify precision and n validators work correctly."""
    # Test valid configuration
    config = MatrixFreeSolverConfig(precision=np.float64, solver_width=5)
    assert config.precision == np.float64
    assert config.solver_width == 5

    # Test n must be >= 1
    with pytest.raises(ValueError):
        MatrixFreeSolverConfig(precision=np.float32, solver_width=0)

    # Test invalid precision type raises error
    with pytest.raises(ValueError):
        MatrixFreeSolverConfig(precision=np.int32, solver_width=3)


def test_matrix_free_solver_config_max_iters_default():
    """Verify max_iters defaults to 100."""
    config = MatrixFreeSolverConfig(precision=np.float64, solver_width=3)
    assert config.max_iters == 100


def test_matrix_free_solver_config_max_iters_validation():
    """Verify max_iters rejects values < 1 and > 32767."""
    # Test valid boundary values
    config_min = MatrixFreeSolverConfig(
        precision=np.float64,
        solver_width=3,
        max_iters=1,
    )
    assert config_min.max_iters == 1

    config_max = MatrixFreeSolverConfig(
        precision=np.float64, solver_width=3, max_iters=32767
    )
    assert config_max.max_iters == 32767

    # Test invalid: below minimum
    with pytest.raises((ValueError, TypeError)):
        MatrixFreeSolverConfig(
            precision=np.float64, solver_width=3, max_iters=0
        )

    # Test invalid: above maximum
    with pytest.raises((ValueError, TypeError)):
        MatrixFreeSolverConfig(
            precision=np.float64, solver_width=3, max_iters=32768
        )

    # Test invalid: wrong type
    with pytest.raises((ValueError, TypeError)):
        MatrixFreeSolverConfig(
            precision=np.float64, solver_width=3, max_iters=50.5
        )


def test_matrix_free_solver_config_norm_device_function_field():
    """Verify norm_device_function field exists and accepts None or Callable.
    """
    # Test default is None
    config = MatrixFreeSolverConfig(precision=np.float64, solver_width=3)
    assert config.norm_device_function is None

    # Test accepts a callable
    def dummy_norm():
        pass

    config_with_fn = MatrixFreeSolverConfig(
        precision=np.float64, solver_width=3, norm_device_function=dummy_norm
    )
    assert config_with_fn.norm_device_function is dummy_norm

    # Verify eq=False behavior: configs with different functions are still
    # equal (since norm_device_function is excluded from equality)
    def another_norm():
        pass

    config_other = MatrixFreeSolverConfig(
        precision=np.float64, solver_width=3, norm_device_function=another_norm
    )
    assert config_with_fn == config_other


def test_matrix_free_solver_creates_norm():
    """Verify MatrixFreeSolver creates ScaledNorm in constructor."""

    # Create a concrete subclass for testing (MatrixFreeSolver is abstract)
    class _StubSolver(MatrixFreeSolver):
        def build(self):
            pass

    solver = _StubSolver(
        precision=np.float64,
        solver_width=3,
        solver_type="newton",
    )

    # Verify norm attribute exists and is a ScaledNorm instance
    assert hasattr(solver, "norm")
    assert isinstance(solver.norm, ScaledNorm)

    # Verify norm has correct precision and n
    assert solver.norm.precision == np.float64
    assert solver.norm.solver_width == 3


def test_matrix_free_solver_forwards_kwargs_to_norm(precision):
    """Verify kwargs passed to MatrixFreeSolver reach ScaledNorm.

    Tests that prefixed tolerance parameters (e.g., krylov_atol) are
    correctly forwarded to the nested ScaledNorm factory through the
    MatrixFreeSolver constructor.
    """

    class TestSolver(MatrixFreeSolver):
        def build(self):
            pass

    n = 3
    krylov_atol = np.array([1e-10, 1e-9, 1e-8], dtype=precision)
    krylov_rtol = np.array([1e-5, 1e-4, 1e-3], dtype=precision)

    solver = TestSolver(
        precision=precision,
        solver_type="krylov",
        solver_width=n,
        krylov_atol=krylov_atol,
        krylov_rtol=krylov_rtol,
    )

    # Verify kwargs reached the nested ScaledNorm
    assert np.allclose(solver.norm.atol, krylov_atol)
    assert np.allclose(solver.norm.rtol, krylov_rtol)


def test_matrix_free_solver_update_with_no_changes_returns_empty_set():
    """update() with no arguments returns an empty set without error."""

    class _StubSolver(MatrixFreeSolver):
        def build(self):
            pass

    solver = _StubSolver(
        precision=np.float64,
        solver_width=3,
        solver_type="newton",
    )
    assert solver.update() == set()
    assert solver.update(updates_dict={}) == set()


def test_matrix_free_solver_n_and_max_iters_properties():
    """n and max_iters forward to compile_settings on a concrete solver."""
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        MRLinearSolver,
    )

    solver = MRLinearSolver(
        precision=np.float64,
        solver_width=7,
        krylov_max_iters=11,
    )
    assert solver.solver_width == 7
    assert solver.max_iters == 11
