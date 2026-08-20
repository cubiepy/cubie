"""CUDA output management package for solver integrations.

The package centres on :class:`OutputFunctions`, which compiles CUDA callables
for saving time histories and summary metrics. Configuration helpers such as
:class:`OutputConfig` and the sizing utilities provide validated settings that
shape buffer allocation and host-visible array layouts. The
``summarymetrics`` subpackage exposes :data:`summary_metrics` and
:func:`register_metric` so metric implementations can register their CUDA
device callbacks during import.
"""

from cubie.outputhandling.output_config import OutputCompileFlags, OutputConfig
from cubie.outputhandling.output_functions import (
    OutputFunctionCache,
    OutputFunctions,
)
from cubie.outputhandling.output_sizes import (
    BatchInputSizes,
    BatchOutputSizes,
    OutputArrayHeights,
    SingleRunOutputSizes,
)
from cubie.outputhandling.summarymetrics import (
    register_metric,
    summary_metrics,
)


__all__ = [
    "OutputCompileFlags",
    "OutputConfig",
    "OutputFunctionCache",
    "OutputFunctions",
    "OutputArrayHeights",
    "SingleRunOutputSizes",
    "BatchInputSizes",
    "BatchOutputSizes",
    "summary_metrics",
    "register_metric",
]
