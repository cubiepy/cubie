"""Characterise when the Neumann series preconditioner converges.

The Neumann series approximates ``(betaI - gamma h (A (x) J))^-1`` via
the truncated expansion ``beta^-1 (I + T + T^2 + ...)`` where
``T = (gamma h / beta)(A (x) J)``. This converges if and only if the
spectral radius ``rho(T) < 1``.

Since ``T`` scales linearly with the step factor, the static diagnostic
reports its spectral radius per unit ``h`` for FIRK, or per unit
``a_ij * h`` for a single-stage helper whose runtime ``a_ij`` is not
known. An exact convergence verdict is available only when the full
step factor is supplied.

The Jacobian is evaluated at the initial state by central finite
differences of the system's **compiled** ``dxdt`` device function,
launched on the device at the system's compiled precision, so the
diagnostic reflects the behaviour of the production device code rather
than a host-side reconstruction. Finite-differencing the guarded
right-hand side also takes removable singularities numerically: the
analytic derivative of a CellML gating term such as
``(V - E) / (exp((V - E) / k) - 1)`` is ``0 / 0`` at the resting
potential even though the term itself is finite there, and the guarded
device code evaluates it cleanly.

Published Objects
-----------------
:class:`NeumannRHSEvaluator`
    CUDAFactory building the finite-difference Jacobian kernel that
    runs the compiled ``dxdt``.
:func:`neumann_spectral_radius`
    Pure-numeric Neumann-series spectral-radius diagnostic as a function
    of the Jacobian and the ``beta``/``gamma``/tableau parameters.
:func:`check_neumann_convergence`
    Evaluate convergence diagnostics for the Neumann preconditioner and
    log them when divergence is likely.

Findings go to this module's logger at debug level. Enable them with
``logging.getLogger("cubie").setLevel(logging.DEBUG)``.
"""

import logging
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import sympy as sp
from attrs import define, field, validators as val, frozen

from cubie.CUDAFactory import (
    CUDADispatcherCache,
    CUDAFactory,
    CUDAFactoryConfig,
)
from cubie._utils import PrecisionDType
from cubie.cubie_cache import CachePolicy, CubieCacheHandler
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases

logger = logging.getLogger(__name__)


@frozen
class NeumannEvaluatorConfig(CUDAFactoryConfig):
    """Compile settings for the Jacobian evaluation kernel.

    Parameters
    ----------
    dxdt_function
        Compiled ``dxdt`` device function the kernel wraps. Excluded
        from hashing; a change still invalidates the build.
    dxdt_settings_hash
        Hash of the owning system's own settings and constants, so the
        disk-cache key changes whenever ``dxdt`` is regenerated with
        different semantics under the same equations.
    system_hash
        The owning system's equation/layout identity, used as the
        disk-cache freshness stamp.
    """

    dxdt_function: Optional[Callable] = field(default=None, eq=False)
    dxdt_settings_hash: str = field(
        default="",
        validator=val.instance_of(str),
    )
    system_hash: str = field(
        default="",
        validator=val.instance_of(str),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


@define
class NeumannEvaluatorCache(CUDADispatcherCache):
    """Container for the compiled Jacobian evaluation kernel."""

    evaluation_kernel: Optional[Callable] = field(default=None)


class NeumannRHSEvaluator(CUDAFactory):
    """Finite-difference Jacobian evaluator for a compiled system.

    Launches the system's compiled ``dxdt`` device function at
    perturbed initial states and forms the Jacobian by central finite
    differences, so the diagnostic sees exactly the device code the
    solver runs, at the compiled precision. Cache policy is fixed at
    construction; evaluator ownership and per-policy keying live with
    :class:`SymbolicODE`.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        cache_policy: CachePolicy,
        system_name: str = "",
    ) -> None:
        super().__init__()
        # setup_compile_settings reaches the handler via
        # _invalidate_cache, so the handler must exist first.
        self._cache_handler = CubieCacheHandler(
            cache_policy, system_name=system_name
        )
        self.setup_compile_settings(
            NeumannEvaluatorConfig(precision=precision)
        )

    @property
    def cache_policy(self) -> CachePolicy:
        """Return the cache policy this evaluator's kernel follows."""
        return self._cache_handler.policy

    def build(self) -> NeumannEvaluatorCache:
        """Compile the evaluation kernel for the configured ``dxdt``."""
        config = self.compile_settings
        dxdt_function = config.dxdt_function
        if dxdt_function is None:
            return NeumannEvaluatorCache()

        # no cover: start
        @cuda.jit(**self.jit_kwargs)
        def evaluate_rhs(
            states, parameters, drivers, observables, out, t
        ):
            i = cuda.grid(1)
            if i < states.shape[0]:
                dxdt_function(
                    states[i],
                    parameters,
                    drivers,
                    observables[i],
                    out[i],
                    t,
                )
        # no cover: end
        if not CUDA_SIMULATION:
            disk_cache = self._cache_handler.configured_cache(
                config.system_hash,
                config.values_hash,
            )
            if disk_cache is not None:
                evaluate_rhs._cache = disk_cache
        return NeumannEvaluatorCache(evaluation_kernel=evaluate_rhs)

    def _invalidate_cache(self) -> None:
        """Invalidate the built kernel, flushing in flush-on-change mode."""
        super()._invalidate_cache()
        self._cache_handler.invalidate()

    def jacobian(
        self,
        index_map: IndexedBases,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Return the state Jacobian at the initial state.

        Parameters
        ----------
        index_map
            Index maps supplying current state/parameter values.
        t0
            Time at which to evaluate the right-hand side.

        Returns
        -------
        numpy.ndarray
            The ``state_count x state_count`` Jacobian in float64.
            Entries that cannot be evaluated are returned as ``nan``.
        """
        precision = np.dtype(self.precision)
        state_map = index_map.states.index_map
        n = len(state_map)
        base = np.zeros(n, dtype=np.float64)
        state_defaults = index_map.states.default_values
        for sym, idx in state_map.items():
            base[idx] = float(state_defaults[sym])

        parameters = np.zeros(1, dtype=precision)
        if hasattr(index_map, "parameters"):
            parameter_map = index_map.parameters.index_map
            if parameter_map:
                parameters = np.zeros(
                    len(parameter_map), dtype=precision
                )
                defaults = index_map.parameters.default_values
                for sym, idx in parameter_map.items():
                    parameters[idx] = float(defaults[sym])

        n_drivers = 1
        if hasattr(index_map, "drivers"):
            n_drivers = max(1, len(index_map.drivers.index_map))
        drivers = np.zeros(n_drivers, dtype=precision)

        n_observables = max(1, len(index_map.observables.index_map))
        observables = np.zeros((2 * n, n_observables), dtype=precision)
        out = np.zeros((2 * n, n), dtype=precision)

        # Central-difference step: cube-root of the compiled
        # precision's epsilon balances round-off against truncation
        # error for a second-order stencil at that precision.
        fd_step = float(np.finfo(precision).eps) ** (1.0 / 3.0)
        states = np.empty((2 * n, n), dtype=precision)
        for col in range(n):
            step = fd_step * max(abs(base[col]), 1.0)
            forward = base.copy()
            backward = base.copy()
            forward[col] += step
            backward[col] -= step
            states[2 * col] = forward
            states[2 * col + 1] = backward

        kernel = self.get_cached_output("evaluation_kernel")
        threads = 32
        blocks = (2 * n + threads - 1) // threads

        def launch():
            device_states = cuda.to_device(states)
            device_parameters = cuda.to_device(parameters)
            device_drivers = cuda.to_device(drivers)
            device_observables = cuda.to_device(observables)
            device_out = cuda.to_device(out)
            kernel[blocks, threads](
                device_states,
                device_parameters,
                device_drivers,
                device_observables,
                device_out,
                precision.type(t0),
            )
            return device_out.copy_to_host()

        if CUDA_SIMULATION:
            # The simulator executes device code as host Python,
            # where domain errors raise instead of producing the
            # non-finite values device math returns; translate them
            # into the caller's could-not-evaluate path.
            try:
                evaluated = launch()
            except (
                ZeroDivisionError,
                FloatingPointError,
                OverflowError,
                ValueError,
            ):
                return np.full((n, n), np.nan)
        else:
            evaluated = launch()

        evaluated = evaluated.astype(np.float64)
        jacobian = np.empty((n, n), dtype=np.float64)
        for col in range(n):
            # Effective step from the precision-cast states so the
            # divisor matches the perturbation the device code saw.
            denominator = float(states[2 * col, col]) - float(
                states[2 * col + 1, col]
            )
            jacobian[:, col] = (
                evaluated[2 * col] - evaluated[2 * col + 1]
            ) / denominator
        return jacobian


def neumann_spectral_radius(
    jacobian: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
    stage_coefficients: Optional[
        Union[float, Sequence[Sequence[Union[float, sp.Expr]]]]
    ] = None,
    step_factor_value: Optional[float] = None,
) -> Dict[str, object]:
    """Return the Neumann-series spectral radius and step limit.

    The series matrix is ``T = (gamma / beta) h (A (x) J)``. This
    routine evaluates its spectral radius per unit step factor and,
    when ``step_factor_value`` is supplied, the radius at that value.

    Parameters
    ----------
    jacobian
        The ``n x n`` state Jacobian.
    beta, gamma
        Transformation parameters from the FIRK formulation.
    stage_coefficients
        Butcher-tableau ``A`` matrix, or the single known ``a_ij``
        value as a float. When omitted, the runtime coefficient is
        unknown and the reported step factor is ``a_ij * h`` rather
        than ``h``.
    step_factor_value
        Value of the reported step factor at which to evaluate the
        infinite series. When omitted, no convergence verdict is made.

    Returns
    -------
    dict
        ``rho_per_unit_step_factor`` -- radius per unit step factor;
        ``nan`` when the Jacobian has non-finite entries.
        ``critical_step_factor`` -- largest factor magnitude.
        ``step_factor`` -- ``"h"`` or ``"a_ij * h"``.
        ``rho_series`` -- radius at ``step_factor_value``, or ``None``.
        ``series_converges`` -- exact infinite-series verdict; ``None``
        without a factor value or a finite radius.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if stage_coefficients is None:
        coupling = jacobian
        step_factor = "a_ij * h"
    elif isinstance(stage_coefficients, float):
        coupling = stage_coefficients * jacobian
        step_factor = "h"
    else:
        coefficients = np.asarray(
            stage_coefficients, dtype=np.float64
        )
        coupling = np.kron(coefficients, jacobian)
        step_factor = "h"

    if np.isfinite(coupling).all():
        eigenvalues = np.linalg.eigvals((gamma / beta) * coupling)
        rho_per_unit_step_factor = float(np.max(np.abs(eigenvalues)))
    else:
        rho_per_unit_step_factor = float("nan")

    # A zero Jacobian (constant right-hand side) has radius zero, so
    # every step factor converges.
    critical_step_factor = (
        float("inf")
        if rho_per_unit_step_factor == 0.0
        else 1.0 / rho_per_unit_step_factor
    )
    rho_series = (
        None
        if step_factor_value is None
        else abs(step_factor_value) * rho_per_unit_step_factor
    )
    series_converges = None
    if rho_series is not None and np.isfinite(rho_series):
        series_converges = rho_series < 1.0

    return {
        "rho_per_unit_step_factor": rho_per_unit_step_factor,
        "critical_step_factor": critical_step_factor,
        "step_factor": step_factor,
        "rho_series": rho_series,
        "series_converges": series_converges,
    }


def check_neumann_convergence(
    index_map: IndexedBases,
    evaluator: NeumannRHSEvaluator,
    stage_coefficients: Optional[
        Union[float, Sequence[Sequence[Union[float, sp.Expr]]]]
    ] = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    t0: float = 0.0,
    step_size: Optional[float] = None,
) -> Dict[str, object]:
    """Characterise Neumann convergence at the initial state.

    Evaluates the state Jacobian at the initial state by finite
    differences of the compiled ``dxdt`` device function and computes
    the spectral radius of the implemented Neumann series matrix.

    Parameters
    ----------
    index_map
        Index maps (states, parameters, etc.) with default values.
    evaluator
        Finite-difference Jacobian evaluator wrapping the system's
        compiled ``dxdt`` device function.
    stage_coefficients
        Butcher-tableau ``A`` matrix for the FIRK method, or the
        runtime single-stage ``a_ij`` value as a float.
    beta, gamma
        Transformation parameters from the FIRK formulation.
    t0
        Time at which to evaluate the Jacobian.
    step_size
        Optional step size for an exact local convergence verdict.

    Returns
    -------
    dict
        The :func:`neumann_spectral_radius` result.
        ``series_converges`` is ``None`` when the full step factor is
        unavailable or the Jacobian cannot be evaluated.

    Notes
    -----
    The tableau ``c`` nodes influence the Jacobian only through the
    ``O(h)`` per-stage time/state offset, which this h-independent
    diagnostic neglects.

    A radius suggesting divergence logs at debug level; a Jacobian
    that cannot be evaluated logs at warning level.
    """
    jacobian = evaluator.jacobian(index_map, t0=t0)
    if not np.isfinite(jacobian).all():
        n_bad = int(np.count_nonzero(~np.isfinite(jacobian)))
        logger.warning(
            "Neumann preconditioner convergence not verified: the "
            "Jacobian at the initial state has %d non-finite entries "
            "(likely a genuine singularity at t0).",
            n_bad,
        )

    # A bare step size is the full series factor only when the stage
    # coefficient(s) are known; without them the factor is a_ij * h
    # and no exact verdict is possible.
    step_factor_value = (
        step_size if stage_coefficients is not None else None
    )

    result = neumann_spectral_radius(
        jacobian, beta=beta, gamma=gamma,
        stage_coefficients=stage_coefficients,
        step_factor_value=step_factor_value,
    )

    if (
        result["series_converges"] is None
        and result["rho_per_unit_step_factor"] >= 1.0
    ):
        rho_per_step = result["rho_per_unit_step_factor"]
        critical_step = result["critical_step_factor"]
        step_factor = result["step_factor"].replace(" ", "")
        message = (
            "Neumann-series convergence depends on the runtime step "
            f"factor. At the initial state, rho(T) / abs({step_factor}) "
            f"= {rho_per_step:.3g}; convergence requires "
            f"abs({step_factor}) < "
            f"{critical_step:.3g}. The complete runtime step factor "
            "was unavailable to this static check, so this is not a "
            "divergence verdict."
        )
        logger.debug(message)
    elif result["series_converges"] is False:
        rho_series = result["rho_series"]
        critical_step = result["critical_step_factor"]
        step_factor = result["step_factor"].replace(" ", "")
        message = (
            "The infinite Neumann series does not converge at the "
            f"initial-state Jacobian for abs({step_factor}) = "
            f"{abs(step_factor_value):.3g}: rho(T) = "
            f"{rho_series:.3g} >= 1. Convergence requires "
            f"abs({step_factor}) < {critical_step:.3g}."
        )
        logger.debug(message)

    return result
