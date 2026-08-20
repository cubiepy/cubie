"""Structured array allocation requests and responses for GPU memory.

This module defines lightweight data containers that describe array
allocation requirements and report allocation outcomes. Requests
capture shape, precision, and memory placement details, while
responses track allocated buffers and any chunking performed by the
memory manager.

Published Classes
-----------------
:class:`ArrayRequest`
    Specification for requesting array allocation.

    >>> from numpy import float64
    >>> req = ArrayRequest(dtype=float64, shape=(100, 4, 1))
    >>> req.memory
    'device'

:class:`ArrayResponse`
    Result of an array allocation containing buffers and chunking
    data.

    >>> resp = ArrayResponse(chunks=2, chunk_length=50)
    >>> resp.chunks
    2

See Also
--------
:class:`~cubie.memory.mem_manager.MemoryManager`
    Processes :class:`ArrayRequest` objects and produces
    :class:`ArrayResponse` instances.
"""

from typing import Optional

import attrs
import attrs.validators as val
import numpy as np

from cubie._utils import opt_getype_validator, getype_validator
from cubie.cuda_simsafe import DeviceNDArrayBase


@attrs.define
class ArrayRequest:
    """Specification for requesting array allocation.

    Parameters
    ----------
    shape
        Tuple describing the requested array shape. Defaults to ``(1, 1, 1)``.
    dtype
        NumPy precision constructor used to produce the allocation. Defaults to
        :func:`numpy.float64`. Integer status buffers use :func:`numpy.int32`.
    memory
        Memory placement option. Must be ``"device"`` or ``"pinned"``.
    unchunkable
        Whether the memory manager is allowed to chunk the allocation.
    total_runs
        Total number of runs for chunking calculations. Defaults to ``1`` for
        arrays not intended for run-axis chunking (e.g., driver_coefficients).
        Memory manager extracts this value to determine chunk parameters.
        Always >= 1.

    Attributes
    ----------
    shape
        Tuple describing the requested array shape.
    dtype
        NumPy precision constructor used to produce the allocation.
    memory
        Memory placement option.
    chunk_axis_index
        Axis index along which chunking may occur.
    unchunkable
        Flag indicating that chunking should be disabled.
    total_runs
        Total number of runs for chunking calculations. Always >= 1.
    """

    dtype = attrs.field(
        validator=val.in_([np.float64, np.float32, np.int32]),
    )
    shape: tuple[int, ...] = attrs.field(
        default=(1, 1, 1),
        validator=val.deep_iterable(
            val.instance_of(int), val.instance_of(tuple)
        ),
    )
    memory: str = attrs.field(
        default="device",
        validator=val.in_(["device", "pinned"]),
    )
    chunk_axis_index: Optional[int] = attrs.field(
        default=2,
        validator=opt_getype_validator(int, 0),
    )
    unchunkable: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    total_runs: int = attrs.field(
        default=1,
        validator=getype_validator(int, 1),
    )


@attrs.define
class ArrayResponse:
    """Result of an array allocation containing buffers and chunking data.

    Parameters
    ----------
    arr
        Dictionary mapping array labels to allocated device arrays.
    chunks
        Number of chunks the allocation was divided into.
    chunk_length
        Length of each chunk along the run axis (except possibly last).
    chunked_shapes
        Mapping from array labels to their per-chunk shapes. Empty dict when
        no chunking occurs.

    Attributes
    ----------
    arr
        Dictionary mapping array labels to allocated device arrays.
    chunks
        Number of chunks the allocation was divided into.
    chunk_length
        Length of each chunk along the run axis.
    chunked_shapes
        Mapping from array labels to their per-chunk shapes.
    """

    arr: dict[str, DeviceNDArrayBase] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )
    chunks: int = attrs.field(
        default=1,
    )
    chunk_length: int = attrs.field(
        default=1,
    )
    chunked_shapes: dict[str, tuple[int, ...]] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )
