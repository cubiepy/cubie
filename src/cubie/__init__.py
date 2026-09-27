"""
cubie: CUDA Batch Integration Engine
"""

from importlib.metadata import version
from importlib.util import find_spec

if find_spec("numba_cuda_mlir") is None:
    raise ImportError(
        "cubie needs its CUDA backend, numba-cuda-mlir. Install it "
        "with 'pip install cubie[mlir-cuda12]' or 'cubie[mlir-cuda13]' "
        "(the bare 'mlir' extra skips the toolkit wheels when a system "
        "CUDA install is present)."
    )

# Apply the numba-cuda-mlir patches before anything can compile a
# kernel: missing lowerings, frontend perf patches and the unroll_if
# pass.
import cubie.backend._mlir_compat  # noqa: F401,E402
import cubie.backend._mlir_cubie_extensions  # noqa: F401,E402

from cubie.result_codes import CUBIE_RESULT_CODES  # noqa: E402
from cubie.batchsolving import *  # noqa
from cubie.integrators import *  # noqa
from cubie.outputhandling import *  # noqa
from cubie.memory import *  # noqa
from cubie.odesystems import *  # noqa
from cubie._utils import *  # noqa
from cubie.batchsolving import (  # noqa: E402
    ArrayTypes,
    Solver,
    solve_ivp,
)
from cubie.memory import default_memmgr  # noqa: E402
from cubie.odesystems import (  # noqa: E402
    SymbolicODE,
    create_ODE_system,
    load_cellml_model,
)
from cubie.outputhandling import summary_metrics  # noqa: E402
from cubie.array_interpolator import DriverSamples  # noqa: E402
from cubie.time_logger import TimeLogger, default_timelogger  # noqa: E402

__all__ = [
    "summary_metrics",
    "DriverSamples",
    "default_memmgr",
    "ArrayTypes",
    "Solver",
    "solve_ivp",
    "SymbolicODE",
    "create_ODE_system",
    "TimeLogger",
    "default_timelogger",
    "load_cellml_model",
    "CUBIE_RESULT_CODES",
]

try:
    __version__ = version("cubie")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
