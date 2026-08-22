"""Coordinate GPU batch ODE solves and expose supporting infrastructure.

The package surfaces the :class:`Solver` class alongside :func:`solve_ivp`, a
convenience wrapper for configuring batch integrations. Supporting modules
provide grid construction, kernel compilation, system interfaces, and result
containers. The :mod:`cubie.batchsolving.arrays` subpackage hosts array
managers for host and device buffers used throughout the workflow.

See Also
--------
:class:`Solver`
    User-facing entry point for batch integrations.
:func:`solve_ivp`
    Convenience wrapper combining solver creation and execution.
:class:`SolveResult`
    Result container returned by solver runs.
:class:`BatchSolverKernel`
    Kernel factory compiling and launching the integration kernel.
"""

from typing import Optional, Union
from numpy.typing import NDArray
from cubie.cuda_simsafe import DeviceNDArrayBase, MappedNDArray

ArrayTypes = Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]

from cubie.batchsolving.BatchInputHandler import (  # noqa: E402
    BatchInputHandler,
)
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig, \
    ActiveOutputs  # noqa: E402
from cubie.batchsolving.BatchSolverKernel import (  # noqa: E402
    BatchSolverKernel,
)
from cubie.batchsolving.SystemInterface import SystemInterface  # noqa: E402
from cubie.batchsolving.arrays.BaseArrayManager import (  # noqa: E402
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving.arrays.BatchInputArrays import (  # noqa: E402
    InputArrayContainer,
    InputArrays,
)
from cubie.batchsolving.arrays.BatchOutputArrays import (  # noqa: E402
    OutputArrayContainer,
    OutputArrays,
)
from cubie.batchsolving.calibration import (  # noqa: E402
    CalibrationResult,
    CandidateResult,
    CandidateSpec,
)
from cubie.batchsolving.solver import Solver, solve_ivp  # noqa: E402
from cubie.batchsolving.solveresult import (  # noqa: E402
    DeviceSolveResult,
    SolveResult,
    SolveSpec,
)
from cubie.outputhandling import summary_metrics  # noqa: E402


__all__ = [
    "ActiveOutputs",
    "ArrayContainer",
    "ArrayTypes",
    "BatchInputHandler",
    "BatchSolverConfig",
    "BatchSolverKernel",
    "BaseArrayManager",
    "CalibrationResult",
    "CandidateResult",
    "CandidateSpec",
    "DeviceSolveResult",
    "InputArrayContainer",
    "InputArrays",
    "ManagedArray",
    "OutputArrayContainer",
    "OutputArrays",
    "Solver",
    "SolveResult",
    "SolveSpec",
    "SystemInterface",
    "solve_ivp",
    "summary_metrics",
]
