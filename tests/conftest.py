import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests._utils import (
    _build_solver_instance,
    _build_cpu_step_controller,
    _get_algorithm_order,
    _get_algorithm_tableau,
    _build_enhanced_algorithm_settings,
    _get_evaluate_driver_at_t,
    _get_driver_del_t,
)
import attrs

from cubie.batchsolving.BatchInputHandler import BatchInputHandler
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.buffer_registry import buffer_registry
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import merge_kwargs_into_settings
from cubie.integrators.step_control import get_controller
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
)
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS

from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie.array_interpolator import ArrayInterpolator
from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
from cubie.vendored import cellmlmanip
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import (
    ALL_MEMORY_MANAGER_PARAMETERS,
    MemoryManager,
)
from cubie.outputhandling.output_functions import (
    OutputFunctions,
    ALL_OUTPUT_FUNCTION_PARAMETERS,
)
from tests.integrators.cpu_reference import (
    CPUODESystem,
    DriverEvaluator,
    run_reference_loop,
)

from tests._utils import (
    MockMemoryManager,
    _driver_sequence,
    run_controller_device_step,
    run_device_loop,
)
from tests.system_fixtures import (
    build_colliding_constants_system,
    build_coupled_oscillator_system,
    build_diagonally_dominant_system,
    build_gating_singularity_system,
    build_hostile_names_system,
    build_large_nonlinear_system,
    build_lorenz_julia_system,
    build_medium_nonlinear_system,
    build_off_diagonal_heavy_system,
    build_safe_names_system,
    build_singular_initial_state_system,
    build_status_staining_stiff_system,
    build_three_chamber_system,
    build_three_state_constant_deriv_system,
    build_three_state_linear_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
    build_time_array_driver_system,
    build_time_function_driver_system,
    build_two_driver_system,
)
from numpy.typing import NDArray

Array = NDArray[np.floating]

enable_tempdir = "1"
os.environ["CUBIE_GENERATED_DIR_REDIRECT"] = enable_tempdir
np.set_printoptions(linewidth=120, threshold=np.inf, precision=12)


# --------------------------------------------------------------------------- #
#                           Test ordering hook                                #
# --------------------------------------------------------------------------- #


def _canonical_param(value):
    """Return a stable repr for a param value, dict-order independent."""
    if isinstance(value, dict):
        return repr(
            sorted((k, _canonical_param(v)) for k, v in value.items())
        )
    return repr(value)


def _session_param_signature(item):
    """Signature of the session-scoped indirect params an item carries.

    Session-scoped parametrised fixtures (solver_settings_override and
    friends) tear down and rebuild the compiled fixture chain whenever
    consecutive tests carry different param sets, so tests sharing a
    signature must run contiguously on one xdist worker for each param
    set to compile once. Returns None for tests using pure defaults.
    """
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    name2fixturedefs = item._fixtureinfo.name2fixturedefs
    parts = []
    for name in sorted(callspec.params):
        fixturedefs = name2fixturedefs.get(name)
        if not fixturedefs or fixturedefs[-1].scope != "session":
            continue
        parts.append(f"{name}={_canonical_param(callspec.params[name])}")
    return "; ".join(parts) if parts else None


def pytest_configure(config):
    """Silence the vendored performance warning on the MLIR backend.

    pyproject's ``filterwarnings`` names numba's warning class, which
    is importable on every backend; the MLIR frontend raises its own
    vendored class, registered here only when that backend is active
    (the class path does not import under numba-cuda).
    numba-cuda workers also load the NVVM library here.
    """
    from cubie.cuda_backend import IS_MLIR
    from cubie.cuda_simsafe import CUDA_SIMULATION

    if not IS_MLIR and not CUDA_SIMULATION:
        try:
            from cuda.pathfinder import load_nvidia_dynamic_lib

            load_nvidia_dynamic_lib("nvvm")
        except Exception:
            pass
    if IS_MLIR:
        config.addinivalue_line(
            "filterwarnings",
            "ignore::numba_cuda_mlir.numba_cuda.core.errors."
            "NumbaPerformanceWarning",
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Group override-param tests for xdist and order the collection.

    Under ``--dist=loadgroup`` each non-default session param set
    becomes one xdist_group, so each set is compiled by exactly one
    worker. Grouped items run first (largest groups first, for
    balance); default-param tests follow as individually scheduled
    items, so every worker builds the default fixture chain exactly
    once, after its override groups are done. Tests which close the
    CUDA context stay at the very end, ungrouped, so the streams used
    in session-scoped fixtures don't disappear on them mid-run.
    """
    final_basenames = {"test_cupyemm.py", "test_memmgmt.py"}
    grouped = []
    default = []
    final = []
    group_sizes = {}
    for item in items:
        if item.fspath.basename in final_basenames:
            final.append(item)
            continue
        signature = _session_param_signature(item)
        if signature is None:
            default.append(item)
            continue
        digest = hashlib.sha1(signature.encode()).hexdigest()[:10]
        item.add_marker(pytest.mark.xdist_group(name=f"pg-{digest}"))
        grouped.append((digest, item))
        group_sizes[digest] = group_sizes.get(digest, 0) + 1

    group_order = {}
    for index, (digest, _) in enumerate(grouped):
        group_order.setdefault(digest, (-group_sizes[digest], index))
    grouped.sort(key=lambda pair: group_order[pair[0]])
    items[:] = [item for _, item in grouped] + default + final


# --------------------------------------------------------------------------- #
#                            Codegen Redirect                                 #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def codegen_dir():
    """Redirect every disk cache to a temporary session directory.

    Sets the shared cache root (:mod:`cubie.cache_root`), which the
    codegen, CellML parse, and compiled-kernel caches all resolve
    through, so no cache artefacts leak into or out of the session.
    Use tempfile.mkdtemp instead of pytest's tmp path so the directory
    isn't removed automatically between parameterized test cases.
    Remove the directory at session teardown.

    Toggle: set environment variable `CUBIE_GENERATED_DIR_REDIRECT` to
    `0` to disable the temporary redirect and keep the default
    ``<cwd>/generated`` cache root.
    """
    import tempfile
    import shutil
    import os
    from cubie import cache_root

    redirect_enabled = int(os.environ.get("CUBIE_GENERATED_DIR_REDIRECT", "1"))

    if not redirect_enabled:
        yield cache_root.get_cache_root()
        return

    gen_dir = Path(tempfile.mkdtemp(prefix="cubie_generated_"))
    previous = cache_root.get_cache_root_override()
    cache_root.set_cache_root(gen_dir)
    try:
        yield gen_dir
    finally:
        # restore the previous root and remove the temporary dir. Wrap
        # in try/except in case multiple workers attempt to delete the
        # same directory when running tests in parallel.
        try:
            cache_root.set_cache_root(previous)
            shutil.rmtree(gen_dir, ignore_errors=True)
        except PermissionError:
            pass


@pytest.fixture(scope="function")
def isolated_cache_root(tmp_path, monkeypatch):
    """Give cache tests a fresh root and no environment overrides."""
    from cubie import cache_root

    monkeypatch.delenv("CUBIE_KERNEL_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUBIE_MAX_CACHE_ENTRIES", raising=False)
    previous = cache_root.get_cache_root_override()
    root = tmp_path / "generated"
    cache_root.set_cache_root(root)
    yield root
    cache_root.set_cache_root(previous)


# ========================================
# SETTINGS DICTS (override -> fixture -> override -> fixture)
# ========================================


@pytest.fixture(scope="session")
def precision(solver_settings_override):
    """Return precision from the override, defaulting to float32.

    Usage:
    @pytest.mark.parametrize("solver_settings_override",
        [{"precision": np.float64}], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    if solver_settings_override and "precision" in solver_settings_override:
        return solver_settings_override["precision"]
    return np.float32


@pytest.fixture(scope="session")
def tolerance_override(request):
    if hasattr(request, "param"):
        return request.param
    return None


@pytest.fixture(scope="session")
def tolerance(tolerance_override, precision):
    if tolerance_override is not None:
        return tolerance_override

    if precision == np.float32:
        return SimpleNamespace(
            abs_loose=1e-5,
            abs_tight=1e-7,
            rel_loose=1e-5,
            rel_tight=1e-7,
        )

    if precision == np.float64:
        return SimpleNamespace(
            abs_loose=1e-9,
            abs_tight=1e-12,
            rel_loose=1e-9,
            rel_tight=1e-12,
        )

    raise ValueError("Unsupported precision for tolerance fixture")


@pytest.fixture(scope="session")
def system(request, solver_settings_override, precision):
    """Return the appropriate symbolic system, defaulting to nonlinear.

    Usage:
    @pytest.mark.parametrize("solver_settings_override",
        [{"system_type": "three_chamber"}], indirect=True)
    def test_something(system):
        # system will be the cardiovascular symbolic model here
    """
    model_type = "nonlinear"
    if solver_settings_override:
        model_type = solver_settings_override.get(
            "system_type", model_type
        )

    if model_type == "linear":
        return build_three_state_linear_system(precision)
    if model_type == "nonlinear":
        return build_three_state_nonlinear_system(precision)
    if model_type in ["three_chamber", "threecm"]:
        return build_three_chamber_system(precision)
    if model_type == "two_driver":
        return build_two_driver_system(precision)
    if model_type == "stiff":
        return build_three_state_very_stiff_system(precision)
    if model_type == "large":
        return build_large_nonlinear_system(precision)
    if model_type == "medium":
        return build_medium_nonlinear_system(precision)
    if model_type == "constant_deriv":
        return build_three_state_constant_deriv_system(precision)
    if model_type == "colliding_constants":
        return build_colliding_constants_system(precision)
    if model_type == "diagonally_dominant":
        return build_diagonally_dominant_system(precision)
    if model_type == "off_diagonal_heavy":
        return build_off_diagonal_heavy_system(precision)
    if model_type == "gating_singularity":
        return build_gating_singularity_system(precision)
    if model_type == "singular_initial_state":
        return build_singular_initial_state_system(precision)
    if model_type == "hostile_names":
        return build_hostile_names_system(precision)
    if model_type == "lorenz_julia":
        return build_lorenz_julia_system(precision)
    if model_type == "coupled_oscillator":
        return build_coupled_oscillator_system(precision)
    if model_type == "staining_stiff":
        return build_status_staining_stiff_system(precision)
    if model_type == "time_function_driver":
        return build_time_function_driver_system(precision)
    if model_type == "time_array_driver":
        return build_time_array_driver_system(precision)
    if not isinstance(model_type, str):
        # A prebuilt system object passed directly as system_type.
        return model_type

    raise ValueError(f"Unknown model type: {model_type}")


@pytest.fixture(scope="session")
def time_function_driver_system(precision):
    """Return the equation-driven twin of ``time_array_driver``.

    The interpolated twin arrives through the chain as ``system``;
    driver-interpolation tests solve both and compare.
    """
    return build_time_function_driver_system(precision)


@pytest.fixture(scope="session")
def hostile_names_system(precision):
    """Return the hostile-named system without a solver chain."""
    return build_hostile_names_system(precision)


@pytest.fixture(scope="session")
def safe_names_system(precision):
    """Return the safe-named twin of the ``hostile_names`` system.

    Collision tests solve both systems and compare; the dynamics are
    identical, only the constant names differ.
    """
    return build_safe_names_system(precision)


@pytest.fixture(scope="session")
def thread_mem_manager():
    """Instantiate a memory manager instance in each thread"""
    return MemoryManager()


# --------------------------------------------------------------------------- #
#                       Chunked-solve fixtures                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="function")
def forced_free_mem(request):
    if hasattr(request, "param"):
        return request.param
    return 950


@pytest.fixture(scope="session")
def _low_mem_mock_manager():
    """Session mock manager whose reported free memory is settable.

    Chunking is a solve-time decision: the manager's limit is queried
    during ``solve``, so one solver built against this manager serves
    every ``forced_free_mem`` value without rebuilding or recompiling.
    """
    return MockMemoryManager()


@pytest.fixture(scope="function")
def low_memory(_low_mem_mock_manager, forced_free_mem):
    """Set the session mock manager's limit for this test."""
    _low_mem_mock_manager._custom_limit = forced_free_mem
    return _low_mem_mock_manager


@pytest.fixture(scope="session")
def _low_mem_solver_base(
    system,
    solver_settings,
    driver_settings,
    _low_mem_mock_manager,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_settings=driver_settings,
        memory_manager=_low_mem_mock_manager,
    )


@pytest.fixture(scope="function")
def low_mem_solver(_low_mem_solver_base, low_memory):
    return _low_mem_solver_base


@pytest.fixture(scope="session")
def _second_low_mem_solver_base(
    system,
    solver_settings,
    driver_settings,
    _low_mem_mock_manager,
):
    """Second solver sharing the mock manager for cross-solver tests."""
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_settings=driver_settings,
        memory_manager=_low_mem_mock_manager,
    )


@pytest.fixture(scope="function")
def second_low_mem_solver(_second_low_mem_solver_base, low_memory):
    return _second_low_mem_solver_base


@pytest.fixture(scope="session")
def start_cuda_busy_work():
    """Return a launcher for a host-released busy kernel on a stream.

    The launcher returns ``(out, stream, done, release)``; ``done``
    is an event recorded after the kernel, so ``done.query()``
    reports whether the unrelated work has finished, and
    ``release()`` lets the kernel exit promptly from the host.
    Completion is gated on the host, so not-yet-done assertions
    cannot flake under GPU contention, and the kernel occupies the
    device only for the bracketed operation: a long-resident spin
    kernel under parallel test load draws driver-level preemption
    exceptions that can poison the whole worker's CUDA context. The
    spin cap bounds a regression in which the bracketed operation
    synchronizes the whole device — the test then fails its canary
    assertion instead of deadlocking against a kernel the host has
    not yet released. The kernel runs on a non-blocking stream:
    ambient legacy-default-stream traffic (allocator frees and pool
    growth land there) serializes every *blocking* stream against
    it, which would entangle the canary with the code under test
    through no fault of that code. A non-blocking stream is immune
    to legacy-stream ordering while still being awaited by any
    genuine device-wide synchronization.
    """
    from cubie.cuda_simsafe import CUDA_SIMULATION, cuda, cupy

    if CUDA_SIMULATION:
        pytest.skip("busy-kernel canary requires real CUDA")

    @cuda.jit
    def _busy_kernel(flag, out):
        spins = 0.0
        while cuda.atomic.add(flag, 0, 0) == 0 and spins < 1.0e9:
            spins += 1.0
        # The +1 keeps the completed-write sentinel nonzero even
        # when the release lands before the first poll.
        out[0] = spins + 1.0

    def _start():
        cupy_stream = cupy.cuda.Stream(non_blocking=True)
        stream = cuda.external_stream(cupy_stream.ptr)
        # Keep the owning cupy stream alive alongside the wrapper.
        stream._cubie_owner = cupy_stream
        out = cuda.device_array(1, dtype=np.float32)
        flag = cuda.to_device(np.zeros(1, dtype=np.int32), stream=stream)
        done = cuda.event()
        _busy_kernel[1, 1, stream](flag, out)
        done.record(stream)

        def release():
            # A copy-engine write lands while the spin kernel is
            # resident; a release kernel could not. The closure
            # keeps the polled flag alive until the kernel exits.
            release_stream = cuda.stream()
            flag.copy_to_device(
                np.ones(1, dtype=np.int32), stream=release_stream
            )

        return out, stream, done, release

    return _start


@pytest.fixture(scope="session")
def unchunking_solver(
    system,
    solver_settings,
    driver_settings,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_settings=driver_settings,
    )


@pytest.fixture(scope="function")
def chunked_solved_solver(
    system, precision, low_mem_solver, driver_settings
):
    solver = low_mem_solver

    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

    # With iteration counters inactive their device array is a
    # placeholder, so the request is smaller than it once was.
    # Measured chunk counts against forced free memory:
    #   <= 680        -> 5 chunks (one run per chunk)
    #   700  .. 830   -> 3 chunks (2-2-1)
    #   890  .. 1130  -> 2 chunks (3-2, then 4-1 near the top)
    #   >= 1190       -> unchunked
    result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    return solver, result


@pytest.fixture(scope="session")
def unchunked_solved_solver(
    system,
    precision,
    driver_settings,
    unchunking_solver,
):
    solver = unchunking_solver
    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

    # Run without chunking
    result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    return solver, result


@pytest.fixture(scope="session")
def solver_settings_override(request):
    """Override for solver settings, if provided."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def solver_settings(solver_settings_override, system, precision):
    """Create LoopStepConfig with default solver configuration."""
    # Clear the buffer_registry singleton when we set up a new system
    buffer_registry.reset()
    defaults = {
        "algorithm": "euler",
        "system_type": "nonlinear",
        "duration": np.float64(0.2),
        "warmup": np.float64(0.0),
        "t0": np.float64(0.0),
        "dt": precision(0.01),
        "dt_min": precision(1e-7),
        "dt_max": precision(1.0),
        "save_every": precision(0.02),
        "summarise_every": precision(0.04),
        "sample_summaries_every": precision(0.02),
        "atol": precision(1e-6),
        "rtol": precision(1e-6),
        "saved_state_indices": [0, 1],
        "saved_observable_indices": [0, 1],
        "summarised_state_indices": [0, 1],
        "summarised_observable_indices": [0, 1],
        "output_types": ["state", "time", "observables", "mean"],
        "blocksize": 32,
        "lineinfo": False,
        "memory_manager": default_memmgr,
        "stream_group": "test_group",
        "mem_proportion": None,
        "host_spill_threshold": None,
        "spill_directory": None,
        "step_controller": "fixed",
        "precision": precision,
        "driverspline_order": 3,
        "driverspline_wrap": False,
        "driverspline_boundary_condition": "clamped",
        "krylov_atol": precision(1e-7),
        "krylov_rtol": precision(1e-7),
        "krylov_residual_reduction": None,
        "krylov_residual_floor": None,
        "linear_correction_type": "minimal_residual",
        "newton_atol": precision(1e-7),
        "newton_rtol": precision(1e-7),
        "preconditioner_order": 2,
        "krylov_max_iters": 50,
        "newton_max_iters": 50,
        "newton_target_iters": 20,
        "min_gain": precision(0.1),
        "max_gain": precision(5.0),
        "safety": precision(0.9),
        "n": system.sizes.states,
        "kp": precision(0.7),
        "ki": precision(-0.4),
        "kd": precision(0.0),
        "deadband_min": precision(0.95),
        "deadband_max": precision(1.05),
        "fix_singularities": True,
        "voltage_variable": None,
        "auto_memory": True,
    }

    float_keys = {
        "duration",
        "warmup",
        "dt",
        "dt_min",
        "dt_max",
        "save_every",
        "summarise_every",
        "sample_summaries_every",
        "atol",
        "rtol",
        "krylov_atol",
        "krylov_rtol",
        "krylov_residual_reduction",
        "krylov_residual_floor",
        "newton_atol",
        "newton_rtol",
        "kp",
        "ki",
        "kd",
        "deadband_min",
        "deadband_max",
    }
    if solver_settings_override:
        # Update defaults with any overrides provided
        for key, value in solver_settings_override.items():
            if key in float_keys:
                # Handle None values for optional float parameters
                if value is None:
                    defaults[key] = None
                else:
                    defaults[key] = precision(value)
            else:
                defaults[key] = value

    # Add derived metadata
    defaults["algorithm_order"] = _get_algorithm_order(defaults["algorithm"])
    defaults["n_states"] = system.sizes.states
    defaults["n_parameters"] = system.sizes.parameters
    defaults["n_drivers"] = system.sizes.drivers
    defaults["n_observables"] = system.sizes.observables

    return defaults


@pytest.fixture(scope="session")
def driver_settings_override(request):
    """Optional override for driver array configuration."""

    return request.param if hasattr(request, "param") else None


@pytest.fixture(scope="session")
def driver_settings(
    driver_settings_override,
    solver_settings,
    system,
    precision,
):
    """Return default driver samples mapped to system driver symbols."""

    if system.num_drivers == 0:
        return None

    if solver_settings["save_every"] is None:
        dt_sample = solver_settings["duration"] / 10.0
    else:
        dt_sample = precision(solver_settings["save_every"]) / 2.0
    total_span = precision(solver_settings["duration"])
    t0 = precision(solver_settings["warmup"])

    order = int(solver_settings["driverspline_order"])

    samples = int(np.ceil(total_span / dt_sample)) + 1
    samples = max(samples, order + 1)
    total_time = precision(dt_sample) * max(samples - 1, 1)

    driver_matrix = _driver_sequence(
        samples=samples,
        total_time=total_time,
        n_drivers=system.num_drivers,
        precision=precision,
    )
    driver_names = list(system.indices.driver_names)
    drivers_dict = {
        name: np.array(driver_matrix[:, idx], dtype=precision, copy=True)
        for idx, name in enumerate(driver_names)
    }
    drivers_dict["driver_sample_period"] = precision(dt_sample)
    drivers_dict["wrap"] = solver_settings["driverspline_wrap"]
    drivers_dict["order"] = order
    drivers_dict["boundary_condition"] = solver_settings[
        "driverspline_boundary_condition"
    ]
    drivers_dict["t0"] = t0

    if driver_settings_override:
        for key, value in driver_settings_override.items():
            drivers_dict[key] = value

    return drivers_dict


@pytest.fixture(scope="session")
def driver_array(
    driver_settings,
    solver_settings,
    precision,
):
    """Instantiate :class:`ArrayInterpolator` for the configured system."""

    if driver_settings is None:
        return None

    return ArrayInterpolator(
        precision=precision,
        input_dict=driver_settings,
    )


@pytest.fixture(scope="session")
def cpu_driver_evaluator(
    driver_settings,
    driver_array,
    solver_settings,
    precision,
    system,
) -> DriverEvaluator:
    """Return a CPU evaluator configured from the driver fixtures."""

    width = system.num_drivers
    order = int(solver_settings["driverspline_order"])
    if driver_settings is None or width == 0 or driver_array is None:
        coeffs = np.zeros((1, width, order + 1), dtype=precision)
        dt_value = precision(solver_settings["save_every"]) / 2.0
        t0_value = 0.0
        wrap_value = bool(solver_settings["driverspline_wrap"])
    else:
        coeffs = np.array(
            driver_array.coefficients,
            dtype=precision,
            copy=True,
        )
        dt_value = precision(driver_array.driver_sample_period)
        t0_value = precision(driver_array.t0)
        wrap_value = bool(driver_array.wrap)

    return DriverEvaluator(
        coefficients=coeffs,
        dt=dt_value,
        t0=t0_value,
        wrap=wrap_value,
        precision=precision,
        boundary_condition=(
            None if driver_array is None else driver_array.boundary_condition
        ),
    )


@pytest.fixture(scope="session")
def algorithm_settings(solver_settings):
    """Filter algorithm configuration from solver_settings dict.

    Note: Functions (evaluate_f, evaluate_observables,
    get_solver_helper_fn, evaluate_driver_at_t, driver_del_t) are NOT
    included in settings. These are passed directly when building
    step objects, not stored in settings dict.
    """
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_ALGORITHM_STEP_PARAMETERS,
    )
    # n_drivers comes from solver_settings (added in Task Group 1)
    # Functions are NOT part of algorithm_settings
    return settings


@pytest.fixture(scope="session")
def loop_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_LOOP_SETTINGS,
    )
    return settings


@pytest.fixture(scope="session")
def step_controller_settings(solver_settings):
    """Base configuration used to instantiate loop step controllers.

    algorithm_order comes from solver_settings which was enriched with
    this metadata during fixture setup.
    """
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_STEP_CONTROLLER_PARAMETERS,
    )
    settings.update(algorithm_order=solver_settings["algorithm_order"])
    return settings


# ========================================
# OBJECT FIXTURES
# ========================================


@pytest.fixture(scope="session")
def output_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_OUTPUT_FUNCTION_PARAMETERS,
    )
    return settings


@pytest.fixture(scope="session")
def memory_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_MEMORY_MANAGER_PARAMETERS,
    )
    return settings


@pytest.fixture(scope="session")
def output_functions(output_settings, system, precision):
    settings = output_settings.copy()
    settings.pop("precision", None)
    outputfunctions = OutputFunctions(
        system.sizes.states,
        system.sizes.observables,
        precision=precision,
        **settings,
    )
    return outputfunctions


@pytest.fixture(scope="function")
def output_functions_mutable(output_settings, system, precision):
    """Return a fresh ``OutputFunctions`` for mutation-prone tests."""
    settings = output_settings.copy()
    settings.pop("precision", None)
    return OutputFunctions(
        system.sizes.states,
        system.sizes.observables,
        precision=precision,
        **settings,
    )


@pytest.fixture(scope="session")
def solverkernel(
    solver_settings,
    system,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    memory_settings,
    loop_settings,
):
    """Top-level composite fixture for BatchSolverKernel.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for BatchSolverKernel
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return BatchSolverKernel(
        system,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        lineinfo=solver_settings["lineinfo"],
        step_control_settings=dict(step_controller_settings),
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=dict(output_settings),
        memory_settings=dict(memory_settings),
        loop_settings=dict(loop_settings),
    )


@pytest.fixture(scope="function")
def solverkernel_mutable(
    solver_settings,
    system,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    memory_settings,
    loop_settings,
):
    """Function-scoped composite fixture for BatchSolverKernel.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for BatchSolverKernel
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return BatchSolverKernel(
        system,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        lineinfo=solver_settings["lineinfo"],
        step_control_settings=dict(step_controller_settings),
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=dict(output_settings),
        memory_settings=dict(memory_settings),
        loop_settings=dict(loop_settings),
    )


@pytest.fixture(scope="session")
def solver(system, solver_settings, driver_settings, thread_mem_manager):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_settings=driver_settings,
        memory_manager=thread_mem_manager,
    )


@pytest.fixture(scope="function")
def solver_mutable(
    system,
    solver_settings,
    driver_settings,
    thread_mem_manager,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_settings=driver_settings,
        memory_manager=thread_mem_manager,
    )


@pytest.fixture(scope="session")
def step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""
    controller = get_controller(precision, step_controller_settings)
    return controller


@pytest.fixture(scope="function")
def step_controller_mutable(precision, step_controller_settings):
    """Return a fresh step controller for mutation-focused tests."""

    return get_controller(precision, step_controller_settings)


@pytest.fixture(scope="function")
def step_setup(request, precision, system):
    """Inputs for a single controller device step, override via param."""
    n = system.sizes.states
    setup_dict = {
        "dt0": 0.05,
        "error": np.asarray(
            [0.01] * system.sizes.states, dtype=precision
        ),
        "state": np.ones(n, dtype=precision),
        "state_prev": np.ones(n, dtype=precision),
        "local_mem": np.zeros(2, dtype=precision),
    }
    if hasattr(request, "param"):
        for key, value in request.param.items():
            if key in setup_dict:
                setup_dict[key] = value
    return setup_dict


@pytest.fixture(scope="function")
def device_step_results(step_controller, precision, step_setup):
    """Run the session controller's device function one step."""
    return run_controller_device_step(
        step_controller.device_function,
        precision,
        step_setup["dt0"],
        step_setup["error"],
        state=step_setup["state"],
        state_prev=step_setup["state_prev"],
        local_mem=step_setup["local_mem"],
    )


@pytest.fixture(scope="session")
def loop(single_integrator_run):
    """Return the IVPLoop from single_integrator_run.

    SingleIntegratorRun builds all components internally, including the
    loop. Access the cached loop instance rather than rebuilding.
    """
    return single_integrator_run._loop


@pytest.fixture(scope="function")
def loop_mutable(single_integrator_run_mutable):
    """Return the IVPLoop from mutable single_integrator_run.

    Function-scoped variant for mutation-focused tests.
    """
    return single_integrator_run_mutable._loop


@pytest.fixture(scope="session")
def single_integrator_run(
    system,
    solver_settings,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """Top-level composite fixture for SingleIntegratorRun.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for SingleIntegratorRun
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return SingleIntegratorRun(
        system=system,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        step_control_settings=dict(step_controller_settings),
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=dict(output_settings),
        loop_settings=dict(loop_settings),
    )


@pytest.fixture(scope="function")
def single_integrator_run_mutable(
    system,
    solver_settings,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """Function-scoped composite fixture for SingleIntegratorRun.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for SingleIntegratorRun
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return SingleIntegratorRun(
        system=system,
        loop_settings=dict(loop_settings),
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        step_control_settings=dict(step_controller_settings),
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=dict(output_settings),
    )


@pytest.fixture(scope="session")
def time_function_driver_run(
    time_function_driver_system,
    solver_settings,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """Session run over the equation-driven twin of ``time_array_driver``.

    Built from the same chain settings as ``single_integrator_run`` so
    driver-interpolation tests can solve both twins and compare; the
    twin has no drivers, so no driver evaluators are wired in.
    """
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, time_function_driver_system, None
    )
    return SingleIntegratorRun(
        system=time_function_driver_system,
        evaluate_driver_at_t=None,
        driver_del_t=None,
        step_control_settings=dict(step_controller_settings),
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=dict(output_settings),
        loop_settings=dict(loop_settings),
    )


@pytest.fixture(scope="session")
def cpu_system(system):
    """Return a CPU-based system."""
    return CPUODESystem(system)


@pytest.fixture(scope="session")
def step_object(single_integrator_run):
    """Return the step object from single_integrator_run.

    This avoids double-building by extracting the already-built step object
    from the composite single_integrator_run fixture.
    """
    return single_integrator_run._algo_step


@pytest.fixture(scope="function")
def step_object_mutable(single_integrator_run_mutable):
    """Return the mutable step object from single_integrator_run_mutable.

    Function-scoped variant for mutation-focused tests.
    """
    return single_integrator_run_mutable._algo_step


@pytest.fixture(scope="session")
def cpu_step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""

    return _build_cpu_step_controller(
        precision=precision,
        step_controller_settings=step_controller_settings,
    )


# ========================================
# INPUT FIXTURES
# ========================================


@pytest.fixture(scope="session")
def initial_state(system, precision, request):
    """Return a copy of the system's initial state vector."""
    if hasattr(request, "param"):
        try:
            request_inits = np.asarray(
                request.param,
                dtype=precision,
            )
            if (
                request_inits.ndim != 1
                or request_inits.shape[0] != system.sizes.states
            ):
                raise ValueError(
                    "initial state override has incorrect shape",
                )
        except TypeError as error:
            raise TypeError(
                "initial state override could not be coerced into numpy array",
            ) from error
        return request_inits
    return system.initial_values.values_array.astype(precision, copy=True)


# ========================================
# COMPUTED OUTPUT FIXTURES
# ========================================
@pytest.fixture(scope="session")
def cpu_loop_runner(
    system,
    cpu_system,
    precision,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_driver_evaluator,
):
    """Return a callable for generating CPU reference loop outputs."""

    def _run_loop(
        *,
        initial_values=None,
        parameters=None,
        driver_coefficients=None,
    ):
        initial_vec = (
            np.array(initial_values, dtype=precision, copy=True)
            if initial_values is not None
            else np.array(
                system.initial_values.values_array,
                dtype=precision,
                copy=True,
            )
        )
        parameter_vec = (
            np.array(parameters, dtype=precision, copy=True)
            if parameters is not None
            else np.array(
                system.parameters.values_array,
                dtype=precision,
                copy=True,
            )
        )

        inputs = {
            "initial_values": initial_vec,
            "parameters": parameter_vec,
        }
        if driver_coefficients is not None:
            inputs["driver_coefficients"] = np.array(
                driver_coefficients, dtype=precision, copy=True
            )

        controller = _build_cpu_step_controller(
            precision=precision,
            step_controller_settings=step_controller_settings,
        )
        tableau = _get_algorithm_tableau(solver_settings["algorithm"])
        return run_reference_loop(
            evaluator=cpu_system,
            inputs=inputs,
            driver_evaluator=cpu_driver_evaluator,
            solver_settings=solver_settings,
            output_functions=output_functions,
            controller=controller,
            tableau=tableau,
        )

    return _run_loop


@pytest.fixture(scope="session")
def cpu_loop_outputs(
    system,
    cpu_system,
    precision,
    initial_state,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_driver_evaluator,
    driver_array,
    single_integrator_run,
) -> dict[str, Array]:
    """Execute the CPU reference loop with the provided configuration."""
    inputs = {
        "initial_values": initial_state.copy(),
        "parameters": system.parameters.values_array.copy(),
    }
    coefficients = (
        driver_array.coefficients if driver_array is not None else None
    )
    inputs["driver_coefficients"] = coefficients

    controller = _build_cpu_step_controller(
        precision=precision,
        step_controller_settings=step_controller_settings,
    )
    # Extract step_object from single_integrator_run
    step_object = single_integrator_run._algo_step
    tableau = getattr(step_object, "tableau", None)
    return run_reference_loop(
        evaluator=cpu_system,
        inputs=inputs,
        driver_evaluator=cpu_driver_evaluator,
        solver_settings=solver_settings,
        output_functions=output_functions,
        controller=controller,
        tableau=tableau,
    )


@pytest.fixture(scope="session")
def device_loop_outputs(
    system,
    single_integrator_run,
    initial_state,
    solver_settings,
    driver_array,
):
    """Execute the device loop with the provided configuration."""
    return run_device_loop(
        singleintegratorrun=single_integrator_run,
        system=system,
        initial_state=initial_state,
        solver_config=solver_settings,
        driver_array=driver_array,
    )


# ========================================
# BATCH INPUT FIXTURES
# ========================================
@pytest.fixture(scope="session")
def system_interface(system) -> SystemInterface:
    """Return a SystemInterface wrapping the configured system."""
    return SystemInterface.from_system(system)


@pytest.fixture(scope="function")
def system_interface_mutable(system) -> SystemInterface:
    """Return a fresh SystemInterface for mutation tests."""
    return SystemInterface(
        system.parameters.copy(),
        system.initial_values.copy(),
        system.observables.copy(),
    )


@pytest.fixture(scope="session")
def input_handler(system) -> BatchInputHandler:
    """Return a batch input handler for the configured system."""
    return BatchInputHandler.from_system(system)


@pytest.fixture(scope="session")
def batch_settings_override(request) -> dict:
    """Override values for batch grid settings when parametrised."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def batch_settings(batch_settings_override) -> dict:
    """Return default batch grid settings merged with overrides."""
    defaults = {
        "num_state_vals_0": 2,
        "num_state_vals_1": 0,
        "num_param_vals_0": 2,
        "num_param_vals_1": 0,
        "kind": "combinatorial",
    }
    defaults.update(
        {k: v for k, v in batch_settings_override.items() if k in defaults}
    )
    return defaults


@pytest.fixture(scope="session")
def batch_request(system, batch_settings, precision) -> dict[str, Array]:
    """Build a request dictionary describing the batch sweep."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_state_vals_0"],
                dtype=precision,
            ),
            [system.initial_values.values_dict[state_names[0]]],
        ]),
        state_names[1]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_state_vals_1"],
                dtype=precision,
            ),
            [system.initial_values.values_dict[state_names[1]]],
        ]),
        param_names[0]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_param_vals_0"],
                dtype=precision,
            ),
            [system.parameters.values_dict[param_names[0]]],
        ]),
        param_names[1]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_param_vals_1"],
                dtype=precision,
            ),
            [system.parameters.values_dict[param_names[1]]],
        ]),
    }


@pytest.fixture(scope="session")
def batch_input_arrays(
    batch_request,
    batch_settings,
    input_handler,
    system,
) -> tuple[Array, Array]:
    """Return the initial state and parameter arrays for the batch run."""
    state_names = set(system.initial_values.names)
    param_names = set(system.parameters.names)

    states_dict = {
        k: v for k, v in batch_request.items() if k in state_names
    }
    params_dict = {
        k: v for k, v in batch_request.items() if k in param_names
    }

    return input_handler(
        states=states_dict,
        params=params_dict,
        kind=batch_settings["kind"],
    )


@attrs.define
class BatchResult:
    """Container for CPU reference outputs for a single batch run."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int


@pytest.fixture(scope="session")
def cpu_batch_results(
    batch_input_arrays,
    cpu_loop_runner,
    system,
    solver_settings,
    precision,
    driver_array,
) -> BatchResult:
    """Compute CPU reference outputs for each run in the batch."""
    initial_sets, parameter_sets = batch_input_arrays
    results: list[BatchResult] = []
    coefficients = (
        driver_array.coefficients if driver_array is not None else None
    )
    n_runs = initial_sets.shape[1]
    for idx in range(n_runs):
        loop_result = cpu_loop_runner(
            initial_values=initial_sets[:, idx],
            parameters=parameter_sets[:, idx],
            driver_coefficients=coefficients,
        )
        results.append(
            BatchResult(
                state=loop_result["state"],
                observables=loop_result["observables"],
                state_summaries=loop_result["state_summaries"],
                observable_summaries=loop_result["observable_summaries"],
                status=int(loop_result["status"]),
            )
        )

    return BatchResult(
        state=np.stack([r.state for r in results], axis=2),
        observables=np.stack([r.observables for r in results], axis=2),
        state_summaries=np.stack(
            [r.state_summaries for r in results], axis=2,
        ),
        observable_summaries=np.stack(
            [r.observable_summaries for r in results], axis=2,
        ),
        status=0 if all(r.status == 0 for r in results) else 1,
    )


# ========================================
# CELLML FIXTURES
# ========================================


@pytest.fixture(scope="session")
def cellml_fixtures_dir():
    """Return the path to the CellML test-fixture directory."""
    return Path(__file__).parent / "fixtures" / "cellml"


@pytest.fixture(scope="session")
def basic_model(cellml_fixtures_dir):
    """Return the imported basic ODE CellML model.

    Pinned to ``fix_singularities=False``: basic_ode is a non-cardiac
    toy model with no membrane voltage, so the GHK rewrite is
    meaningless and would only emit a skip warning on every load.
    """
    return load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        fix_singularities=False,
    )


@pytest.fixture(scope="session")
def beeler_reuter_model(cellml_fixtures_dir, solver_settings):
    """Return the imported Beeler-Reuter CellML model.

    Declares two algebraic equations as observables so the
    observable-promotion path shares this parse instead of paying a
    second full cardiac-model load.
    """
    br_path = cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"
    return load_cellml_model(
        str(br_path),
        observables=[
            "sodium_current_i_Na",
            "sodium_current_m_gate_alpha_m",
        ],
        fix_singularities=solver_settings["fix_singularities"],
        voltage_variable=solver_settings["voltage_variable"],
    )


@pytest.fixture(scope="session")
def basic_model_custom(cellml_fixtures_dir):
    """basic_ode loaded with a caller-supplied name and precision."""
    return load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        name="custom_model",
        precision=np.float64,
        fix_singularities=False,
    )


@pytest.fixture(scope="session")
def basic_model_param_main_a(cellml_fixtures_dir):
    """basic_ode with its numeric constant promoted to a parameter."""
    return load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        parameters=["main_a"],
        fix_singularities=False,
    )


@pytest.fixture(scope="session")
def basic_model_parameters_dict(cellml_fixtures_dir):
    """basic_ode with a parameters dict naming one known and one new
    symbol."""
    return load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        parameters={"main_a": 1.0, "user_param": 1.5},
        fix_singularities=False,
    )


@pytest.fixture(scope="session")
def ghk_singularity_model(cellml_fixtures_dir, solver_settings):
    """Return the single-GHK-singularity model used to verify the fix."""
    path = cellml_fixtures_dir / "ghk_singularity.cellml"
    return load_cellml_model(
        str(path),
        fix_singularities=solver_settings["fix_singularities"],
        voltage_variable=solver_settings["voltage_variable"],
    )


@pytest.fixture(scope="session")
def beeler_reuter_raw(cellml_fixtures_dir):
    """Raw cellmlmanip Beeler-Reuter model (read-only detection tests)."""
    br_path = cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"
    return cellmlmanip.load_model(str(br_path))


@pytest.fixture(scope="session")
def basic_ode_raw(cellml_fixtures_dir):
    """Raw cellmlmanip basic_ode model (no membrane-voltage state)."""
    return cellmlmanip.load_model(
        str(cellml_fixtures_dir / "basic_ode.cellml")
    )
