"""
cubie: CUDA Batch Integration Engine
"""

from importlib.metadata import version

# Suppress Numba performance warnings for library users. The warnings are
# emitted from Numba internals when kernels are dispatched with an
# inefficient batch size.
# These are not actionable for CuBIE users,
# so they are filtered at import time.
import os

os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

# Apply the active backend's compatibility patches before anything
# can compile a kernel. On numba-cuda these are compile-time
# performance patches (no-op on the cubie_patch fork, under CUDASIM,
# and for any patch already accepted upstream); on numba-cuda-mlir
# they register missing lowerings and carry the frontend perf
# patches.
from cubie.cuda_backend import IS_MLIR as _IS_MLIR  # noqa: E402

if _IS_MLIR:
    import cubie.backend._mlir_compat  # noqa: F401
    import cubie.backend._mlir_cubie_extensions  # noqa: F401
else:
    import cubie.backend._numba_cuda_compat  # noqa: F401

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
from cubie.time_logger import TimeLogger, default_timelogger  # noqa: E402

__all__ = [
    "summary_metrics",
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
