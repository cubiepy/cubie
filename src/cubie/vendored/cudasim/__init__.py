# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
"""CUDA simulator vendored from numba-cuda, running on numba-cuda-mlir.

Vendored from NVIDIA/numba-cuda 0.30.4 on 2026-09-27.
Source: numba_cuda/numba/cuda/simulator (package) and
numba_cuda/numba/cuda/simulator_init.py (``is_available``,
``cuda_error``).

Local modifications are listed in ``cubie/vendored/AGENTS.md``.
"""

import sys

from . import api, args
from .api import *
from .vector_types import vector_types
from .reduction import Reduce
from .cudadrv.devicearray import (
    device_array,
    device_array_like,
    pinned,
    pinned_array,
    pinned_array_like,
    mapped_array,
    to_device,
    auto_device,
)
from .cudadrv import devicearray
from .cudadrv.devices import require_context, gpus
from .cudadrv.devices import get_context as current_context
from .cudadrv.runtime import runtime
from .cudadrv.linkable_code import LinkableCode
from numba_cuda_mlir.numba_cuda.core import config

reduce = Reduce

# Register simulated vector types as module level variables
for name, svty in vector_types.items():
    setattr(sys.modules[__name__], name, svty)
    for alias in svty.aliases:
        setattr(sys.modules[__name__], alias, svty)
del vector_types, name, svty, alias

from . import cudadrv, dispatcher, experimental  # noqa: E402,F401
from . import bf16, compiler, _internal, memory_management  # noqa: E402,F401


def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU."""
    # Simulator is always available
    return True


def cuda_error():
    """Returns None or an exception if the CUDA driver fails to initialize."""
    # Simulator never fails to initialize
    return None
