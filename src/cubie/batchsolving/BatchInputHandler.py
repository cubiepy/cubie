"""Batch input handling for state and parameter processing.

This module processes user-supplied dictionaries or arrays into the 2D NumPy
arrays expected by the batch solver, classifying them to decide on fast 
paths and combining them into grids if requested. 
:class:`BatchInputHandler` is the primary
entry point and is usually accessed through :class:`cubie.batchsolving.solver.Solver`.

Notes
-----
``BatchInputHandler.__call__`` accepts three arguments:

``states``
    Mapping or array containing state values only. One-dimensional
    inputs override defaults for every run, while two-dimensional inputs
    are treated as pre-built grids in (variable, run) format.
``params``
    Mapping or array containing parameter values only. Interpretation matches
    ``states``.
``kind``
    Controls how inputs are combined. ``"combinatorial"`` builds the
    Cartesian product, while ``"verbatim"`` preserves column-wise groupings.

When arrays are supplied directly they are treated as fully specified grids
in (variable, run) format where each column represents a run configuration.
Dictionary inputs trigger combinatorial expansion before assembly so
that every value combination is represented in the resulting grid.

``BatchInputHandler.__call__`` processes states and params through
independent paths, materialising each at its aligned run count:

1. Each input is classified via ``_plan_single_input()``, which
   validates it and builds a compact grid without allocating any
   full-size array
2. ``_fill_aligned()`` computes the aligned run count for the
   specified ``kind`` strategy, allocates each category's final
   backing through the memory manager's host policy, and assembles
   defaults, swept values, and expansions directly into it
3. Results are cast to system precision

Handler-assembled grids therefore never hold a full-size
intermediate alongside the result.

Examples
--------
>>> import numpy as np
>>> import cubie as qb
>>> from cubie.batchsolving.BatchInputHandler import BatchInputHandler
>>> system = qb.create_ODE_system(
...    dxdt=["dx = p0 * p1 * y","dy = p1 * x"],
...    parameters = {'p0': 2.0, 'p1': 1.5}
... )
>>> handler = BatchInputHandler.from_system(system)
>>> params = {"p0": [0.1, 0.2], "p1": [10, 20]}
>>> states = {"x0": [1.0, 2.0], "x1": [0.5, 1.5]}
>>> inits, params = handler(
...     states=states, params=params, kind="combinatorial"
... )
>>> print(inits.shape)
(2, 16)
>>> print(inits)
[[1.  1.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  2.  2. ]
 [0.5 0.5 0.5 0.5 1.5 1.5 1.5 1.5 0.5 0.5 0.5 0.5 1.5 1.5 1.5 1.5]]
>>> print(params.shape)
(2, 16)
>>> print(params)
[[ 0.1  0.1  0.2  0.2  0.1  0.1  0.2  0.2  0.1  0.1  0.2  0.2  0.1  0.1  0.2  0.2]
 [10.  20.  10.  20.  10.  20.  10.  20.  10.  20.  10.  20.  10.  20.  10.  20. ]]

Example 2: verbatim arrays

>>> params = np.array([[0.1, 0.2], [10, 20]])
>>> states = np.array([[1.0, 2.0], [0.5, 1.5]])
>>> inits, params = handler(states=states, params=params, kind="verbatim")
>>> print(inits.shape)
(2, 2)
>>> print(inits)
[[1.  2. ]
 [0.5 1.5]]
>>> print(params.shape)
(2, 2)
>>> print(params)
[[ 0.1  0.2]
 [10.  20. ]]

>>> inits, params = handler(
...     states=states, params=params, kind="combinatorial"
... )
>>> print(inits.shape)
(2, 4)
>>> print(inits)
[[1.  1.  2.  2. ]
 [0.5 0.5 1.5 1.5]]
>>> print(params.shape)
(2, 4)
>>> print(params)
[[ 0.1  0.2  0.1  0.2]
 [10.  20.  10.  20. ]]

Example 3: single parameter sweep (unspecified filled with defaults)

>>> params = {"p0": [0.1, 0.2]}
>>> inits, params = handler(params=params, kind="combinatorial")
>>> print(inits.shape)
(2, 2)
>>> print(inits)  # unspecified variables are filled with defaults from system
[[1. 1.]
 [1. 1.]]
>>> print(params.shape)
(2, 2)
>>> print(params)
[[0.1 0.2]
 [2.  2. ]]

Published Classes
-----------------
:class:`BatchInputHandler`
    Process user-supplied dicts or arrays into solver-ready 2D arrays.

Module-Level Functions
----------------------
:func:`unique_cartesian_product`
    Deduplicated Cartesian product of input arrays.

:func:`combinatorial_grid`
    Build a grid of all unique parameter combinations.

:func:`verbatim_grid`
    Build a row-aligned grid without combinatorial expansion.

:func:`generate_grid`
    Dispatch to combinatorial or verbatim grid builders.

:func:`combine_grids`
    Combine two grids using combinatorial or verbatim strategy.

:func:`extend_grid_to_array`
    Join a partial grid with default values into a complete array.

See Also
--------
:class:`~cubie.batchsolving.SystemInterface.SystemInterface`
    Label-to-index resolution used by the handler.
:class:`~cubie.batchsolving.solver.Solver`
    Primary consumer that delegates input processing here.
"""
from itertools import product
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union
from warnings import warn

from numpy import (
    ndarray,
    array as np_array,
    dtype as np_dtype,
    memmap as np_memmap,
    tile as np_tile,
    repeat as np_repeat,
    newaxis as np_newaxis,
    asarray as np_asarray,
    empty as np_empty,
    vstack as np_vstack,
    atleast_1d as np_atleast_1d,
)

from numpy.typing import ArrayLike

from cubie.cuda_simsafe import is_device_array, is_pinned_array
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.memory import default_memmgr
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues

if TYPE_CHECKING:
    from cubie.memory.mem_manager import MemoryManager


def unique_cartesian_product(arrays: List[ndarray]) -> ndarray:
    """Return unique combinations of elements from input arrays.

    Each input array can contain duplicates, but the output omits duplicate
    rows while preserving the order of the input arrays.

    Parameters
    ----------
    arrays
        List of one-dimensional NumPy arrays containing elements to combine.

    Returns
    -------
    ndarray
        Two-dimensional array in (variable, run) format where each column
        is a unique combination of the supplied values.

    Notes
    -----
    Duplicate elements are removed by constructing an ordered dictionary per
    input array. ``itertools.product`` then generates the Cartesian product
    of the deduplicated inputs.

    Examples
    --------
    >>> unique_cartesian_product([np.array([1, 2, 2]), np.array([3, 4])])
    array([[1, 1, 2, 2],
           [3, 4, 3, 4]])
    """
    deduplicated_inputs = [
        list(dict.fromkeys(a)) for a in arrays
    ]  # preserve order, remove dups
    # Build array in (variable, run) format: rows are variables, columns runs
    return np_array([list(t) for t in product(*deduplicated_inputs)]).T


def combinatorial_grid(
    request: Dict[Union[str, int], Union[float, ArrayLike, ndarray]],
    values_instance: SystemValues,
    silent: bool = False,
) -> tuple[ndarray, ndarray]:
    """Build a grid of all unique combinations of requested values.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values. Value arrays may differ
        in length.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    silent
        When ``True`` suppresses warnings about unrecognised keys.

    Returns
    -------
    tuple of ndarray and ndarray
        Pair of index and value arrays describing the combinatorial grid.
        Value array is in (variable, run) format.

    Notes
    -----
    Unspecified parameters retain their defaults when the grid is later
    expanded. The number of runs equals the product of all supplied value
    counts.

    Examples
    --------
    >>> combinatorial_grid(
    ...     {"param1": [0.1, 0.2, 0.3], "param2": [10, 20]}, system.parameters
    ... )
    (array([0, 1]),
     array([[ 0.1,  0.1,  0.2,  0.2,  0.3,  0.3],
            [10. , 20. , 10. , 20. , 10. , 20. ]]))
    """
    cleaned_request = {
        k: v for k, v in request.items() if np_asarray(v).size > 0
    }
    indices = values_instance.get_indices(
        list(cleaned_request.keys()), silent=silent
    )
    combos = unique_cartesian_product(
        [np_asarray(v) for v in cleaned_request.values()],
    )
    return indices, combos


def verbatim_grid(
    request: Dict[Union[str, int], Union[float, ArrayLike, ndarray]],
    values_instance: SystemValues,
    silent: bool = False,
) -> tuple[ndarray, ndarray]:
    """Build a grid that aligns parameter rows without combinatorial expansion.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    silent
        When ``True`` suppresses warnings about unrecognised keys.

    Returns
    -------
    tuple of ndarray and ndarray
        Pair of index and value arrays describing the row-wise grid.
        Value array is in (variable, run) format.

    Notes
    -----
    All value arrays must share the same length so rows stay aligned.

    Examples
    --------
    >>> verbatim_grid(
    ...     {"param1": [0.1, 0.2, 0.3], "param2": [10, 20, 30]},
    ...     system.parameters,
    ... )
    (array([0, 1]),
     array([[ 0.1,  0.2,  0.3],
            [10. , 20. , 30. ]]))
    """
    cleaned_request = {
        k: v for k, v in request.items() if np_asarray(v).size > 0
    }
    indices = values_instance.get_indices(
        list(cleaned_request.keys()), silent=silent
    )
    # Build in (variable, run) format: rows are swept variables, columns runs
    combos = np_asarray([item for item in cleaned_request.values()])
    return indices, combos


def generate_grid(
    request: Dict[Union[str, int], Union[float, ArrayLike, ndarray]],
    values_instance: SystemValues,
    kind: str = "combinatorial",
    silent: bool = False,
) -> tuple[ndarray, ndarray]:
    """Generate a parameter grid for batch runs from a request dictionary.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    kind
        Strategy used to assemble the grid. ``"combinatorial"`` expands all
        combinations while ``"verbatim"`` preserves row groupings.
    silent
        When ``True`` suppresses warnings about unrecognised keys.

    Returns
    -------
    tuple of ndarray and ndarray
        Pair of index and value arrays describing the generated grid.
        Value array is in (variable, run) format.

    Notes
    -----
    ``kind`` selects between :func:`combinatorial_grid` and
    :func:`verbatim_grid`.
    """
    # When kind == 'combinatorial' use combinatorial expansion of values
    if kind == "combinatorial":
        return combinatorial_grid(request, values_instance, silent=silent)
    # When kind == 'verbatim' preserve row-wise groupings without expansion
    elif kind == "verbatim":
        return verbatim_grid(request, values_instance, silent=silent)
    # Any other kind is invalid
    else:
        raise ValueError(
            f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'."
        )


def combine_grids(
    grid1: ndarray, grid2: ndarray, kind: str = "combinatorial"
) -> tuple[ndarray, ndarray]:
    """Combine two grids according to the requested pairing strategy.

    Parameters
    ----------
    grid1
        First grid in (variable, run) format, typically parameters.
    grid2
        Second grid in (variable, run) format, typically initial states.
    kind
        ``"combinatorial"`` builds the Cartesian product and
        ``"verbatim"`` pairs columns directly.

    Returns
    -------
    tuple of ndarray and ndarray
        Extended versions of ``grid1`` and ``grid2`` in (variable, run)
        format aligned to the chosen strategy.

    Raises
    ------
    ValueError
        Raised when ``kind`` is ``"verbatim"`` and the inputs have different
        run counts or when ``kind`` is unknown.
    """
    # For 'combinatorial' return the Cartesian product of runs (columns)
    if kind == "combinatorial":
        # Cartesian product: all combinations of runs from each grid
        # Repeat each column of grid1 for each column in grid2
        g1_repeat = np_repeat(grid1, grid2.shape[1], axis=1)
        # Tile grid2 columns for each column in grid1
        g2_tile = np_tile(grid2, (1, grid1.shape[1]))
        return g1_repeat, g2_tile
    # For 'verbatim' pair columns directly and error if run counts differ
    elif kind == "verbatim":
        # Capture original sizes before any broadcast
        g1_runs = grid1.shape[1]
        g2_runs = grid2.shape[1]
        # Broadcast single-run grids to match the other grid's size
        if g1_runs == 1 and g2_runs > 1:
            grid1 = np_repeat(grid1, g2_runs, axis=1)
        elif g2_runs == 1 and g1_runs > 1:
            grid2 = np_repeat(grid2, g1_runs, axis=1)
        # After broadcasting, check dimensions match
        if grid1.shape[1] != grid2.shape[1]:
            raise ValueError(
                "For 'verbatim', both grids must have the same number "
                "of runs, or one grid must have exactly 1 run so it can be "
                "broadcast to match the other."
            )
        return grid1, grid2
    # Any other kind is invalid
    else:
        raise ValueError(
            f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'."
        )


def extend_grid_to_array(
    grid: ndarray,
    indices: ndarray,
    default_values: ndarray,
    out: Optional[ndarray] = None,
    repeats: int = 1,
    tiles: int = 1,
) -> ndarray:
    """Join a grid with defaults to create complete parameter arrays.

    Parameters
    ----------
    grid
        Two-dimensional array of gridded parameter values in (variable, run)
        format.
    indices
        One-dimensional array describing which parameter indices were swept.
    default_values
        One-dimensional array of default parameter values.
    out
        Destination array of shape
        ``(default_values.size, tiles * n_runs * repeats)``,
        assembled in place. A fresh array is allocated when omitted.
    repeats
        Consecutive copies of each grid column in the result, for
        combinatorial expansion against a faster-varying grid.
    tiles
        Whole-grid cycles in the result, for combinatorial expansion
        against a slower-varying grid.

    Returns
    -------
    ndarray
        Two-dimensional array in (variable, run) format containing complete
        parameter values for each run.

    Raises
    ------
    ValueError
        Raised when ``grid`` row count does not match ``indices`` length.
    """
    # A 1D grid or empty indices contributes no swept columns: the
    # result is default values for every run.
    scatter = indices.size > 0 and grid.ndim > 1
    if indices.size == 0:
        n_runs = grid.shape[1] if grid.ndim > 1 else 1
    elif grid.ndim == 1:
        n_runs = 1
    else:
        # When multidimensional ensure the grid row count matches indices
        if grid.shape[0] != indices.shape[0]:
            raise ValueError("Grid shape does not match indices shape.")
        n_runs = grid.shape[1]
    total_runs = tiles * n_runs * repeats
    if out is None:
        out = np_tile(default_values[:, np_newaxis], (1, total_runs))
    else:
        out[:] = default_values[:, np_newaxis]
    if scatter:
        # Scatter grid rows to their variable indices; grid rows follow
        # the caller's key order, which need not match declared order.
        view = out.reshape(out.shape[0], tiles, n_runs, repeats)
        view[indices] = grid[:, np_newaxis, :, np_newaxis]
    return out


def _views_user_input(array: ndarray, user_input: object) -> bool:
    """Return whether ``array`` is or views a caller-supplied array."""
    if not isinstance(user_input, ndarray):
        return False
    base = array
    # Walk views back to the original array they were taken from.
    while isinstance(base, ndarray):
        if base is user_input:
            return True
        base = base.base
    return False


class BatchInputHandler:
    """Process and validate solver inputs for batch runs.

    The handler converts dictionaries or arrays into solver-ready
    two-dimensional arrays, classifies input types for optimal
    processing paths, and validates array shapes and dtypes.

    Parameters
    ----------
    interface
        System interface containing parameter and state metadata.
    memory_manager
        Manager that allocates materialised input arrays.
    host_spill_threshold
        Disk-backing size in bytes; ``None`` = the RAM default.
    spill_directory
        Directory for disk-backed arrays; ``None`` = temp dir.

    Attributes
    ----------
    parameters
        Parameter metadata, read live from ``interface``.
    states
        State metadata, read live from ``interface``.
    precision
        Floating-point precision for returned arrays.
    memory_manager
        Manager that allocates materialised input arrays.
    host_spill_threshold
        Disk-backing size in bytes for assembled arrays.
    spill_directory
        Directory for disk-backed arrays.
    """

    def __init__(
        self,
        interface: SystemInterface,
        memory_manager: "MemoryManager" = default_memmgr,
        host_spill_threshold: Optional[int] = None,
        spill_directory: Optional[str] = None,
    ):
        """Initialise the handler with a system interface."""
        self.interface = interface
        self.precision = interface.parameters.precision
        self.memory_manager = memory_manager
        self.host_spill_threshold = host_spill_threshold
        self.spill_directory = spill_directory

    @property
    def parameters(self) -> SystemValues:
        """Parameter metadata, read live from the interface."""
        return self.interface.parameters

    @property
    def states(self) -> SystemValues:
        """State metadata, read live from the interface."""
        return self.interface.states

    @classmethod
    def from_system(
        cls,
        system: BaseODE,
        memory_manager: "MemoryManager" = default_memmgr,
        host_spill_threshold: Optional[int] = None,
        spill_directory: Optional[str] = None,
    ) -> "BatchInputHandler":
        """Create a handler from a system model.

        Parameters
        ----------
        system
            System model providing parameter and state metadata.
        memory_manager
            Manager that allocates materialised input arrays.
        host_spill_threshold
            Disk-backing size in bytes; ``None`` = the RAM default.
        spill_directory
            Directory for disk-backed arrays; ``None`` = temp dir.

        Returns
        -------
        BatchInputHandler
            Handler configured for ``system``.
        """
        interface = SystemInterface.from_system(system)
        return cls(
            interface,
            memory_manager=memory_manager,
            host_spill_threshold=host_spill_threshold,
            spill_directory=spill_directory,
        )

    def __call__(
        self,
        states: Optional[Union[Dict, ArrayLike]] = None,
        params: Optional[Union[Dict, ArrayLike]] = None,
        kind: str = "combinatorial",
    ) -> tuple[ndarray, ndarray]:
        """Process user input to generate state and parameter arrays.

        Parameters
        ----------
        states
            Optional dictionary or array describing initial state sweeps.
        params
            Optional dictionary or array describing parameter sweeps.
        kind
            Strategy for grid assembly. ``"combinatorial"`` expands
            all combinations while ``"verbatim"`` preserves pairings.

        Returns
        -------
        tuple[ndarray, ndarray]
            Initial state and parameter arrays aligned for batch execution.

        Notes
        -----
        Passing ``states`` and ``params`` as arrays treats each as a
        complete grid. ``kind="combinatorial"`` computes the Cartesian
        product of both grids. When arrays already describe paired runs,
        set ``kind`` to ``"verbatim"`` to keep them aligned.

        Device arrays are treated as prebuilt (variable, run) grids
        and returned with no processing; they must already match the
        system precision and variable count. A host-side counterpart
        is paired verbatim: ``None`` or a single-column input is
        broadcast to the device array's run count, and a multi-column
        input must match it exactly.

        Drivers never pass through the handler: driver samples flow
        through ``ArrayInterpolator`` (host coefficients), and a
        coefficient array handed to ``BatchSolverKernel.run`` directly
        is validated against the compiled layout by
        ``InputArrays._attach_device_inputs``.
        """
        # Update precision from current system state
        self.precision = self.states.precision

        device_result = self._process_device_inputs(states, params)
        if device_result is not None:
            return device_result

        fast_result = self._fast_return_arrays(states, params, kind)
        if fast_result is not None:
            return fast_result

        # Plan compactly so assembly writes into the final backing.
        backed = set()
        states_plan = self._plan_single_input(states, self.states, kind)
        params_plan = self._plan_single_input(params, self.parameters, kind)
        states_array, params_array = self._fill_aligned(
            states_plan, params_plan, kind, backed
        )

        # Cast to system precision
        return self._cast_to_precision(
            states_array, params_array, states, params, backed
        )

    def _validate_device_array(
        self,
        arr: object,
        values_object: SystemValues,
        label: str,
    ) -> None:
        """Validate a device array against system metadata.

        Parameters
        ----------
        arr
            Device array to validate.
        values_object
            System values object for the input category.
        label
            Human-readable input name used in error messages.

        Raises
        ------
        ValueError
            If the array is not 2D or its variable count differs from
            the system's.
        TypeError
            If the array dtype differs from the system precision.
        """
        shape = arr.shape
        if len(shape) != 2:
            raise ValueError(
                f"Device-array {label} must be 2D in (variable, run) "
                f"format, got a {len(shape)}D array."
            )
        if shape[0] != values_object.n:
            raise ValueError(
                f"Device-array {label} has {shape[0]} variables, but "
                f"the system has {values_object.n}. Device arrays "
                f"must be fully specified; they are not padded with "
                f"defaults."
            )
        if np_dtype(arr.dtype) != np_dtype(self.precision):
            raise TypeError(
                f"Device-array {label} has dtype {arr.dtype}, but the "
                f"system precision is {np_dtype(self.precision).name}. "
                f"Cast it on the device before solving."
            )

    def _process_device_inputs(
        self,
        states: Optional[Union[ArrayLike, Dict]],
        params: Optional[Union[ArrayLike, Dict]],
    ) -> Optional[Tuple[object, object]]:
        """Pass device arrays through, pairing any host counterpart.

        Parameters
        ----------
        states
            Initial-state input, possibly a device array.
        params
            Parameter input, possibly a device array.

        Returns
        -------
        Optional[tuple]
            ``(states, params)`` with device arrays untouched and any
            host counterpart expanded to a matching (variable, run)
            array, or ``None`` when neither input is a device array.

        Raises
        ------
        ValueError
            If two device arrays have different run counts, or a
            host-side counterpart cannot be broadcast to the device
            array's run count.
        TypeError
            If a non-empty dict grid accompanies a device array.

        Notes
        -----
        Device arrays are always treated as prebuilt verbatim grids;
        combinatorial expansion of device-resident values is not
        supported.
        """
        states_is_device = is_device_array(states)
        params_is_device = is_device_array(params)
        if not (states_is_device or params_is_device):
            return None

        if states_is_device:
            self._validate_device_array(states, self.states, "states")
        if params_is_device:
            self._validate_device_array(
                params, self.parameters, "params"
            )

        if states_is_device and params_is_device:
            if states.shape[1] != params.shape[1]:
                raise ValueError(
                    f"Device-array states and params have different "
                    f"run counts ({states.shape[1]} and "
                    f"{params.shape[1]}). Device arrays are paired "
                    f"verbatim and must match."
                )
            return states, params

        if states_is_device:
            device_arr = states
            other = params
            other_values = self.parameters
            other_label = "params"
        else:
            device_arr = params
            other = states
            other_values = self.states
            other_label = "states"
        n_runs = device_arr.shape[1]

        if isinstance(other, dict):
            if other:
                raise TypeError(
                    f"Dict {other_label} cannot be combined with a "
                    f"device array; build the host side into a "
                    f"(variable, run) array first, e.g. with "
                    f"Solver.build_grid."
                )
            other = None

        backed = set()
        if other is None:
            host_arr = self._fill_defaults(other_values, n_runs, backed)
        else:
            if not isinstance(other, (list, tuple, ndarray)):
                raise TypeError(
                    f"Input {other_label} must be None, dict, or "
                    f"array-like, got {type(other)}"
                )
            host_arr = self._sanitise_arraylike(other, other_values)
            if host_arr is None:
                host_arr = self._fill_defaults(
                    other_values, n_runs, backed
                )
            elif host_arr.shape[1] == 1 and n_runs > 1:
                column = host_arr
                host_arr = self._final_array(
                    column.shape[0], n_runs, backed
                )
                host_arr[:] = column
            elif host_arr.shape[1] != n_runs:
                raise ValueError(
                    f"Host-side {other_label} has "
                    f"{host_arr.shape[1]} runs but the device array "
                    f"has {n_runs}. Device arrays are paired "
                    f"verbatim: pass a single column to broadcast or "
                    f"a matching run count."
                )
        host_arr = self._materialise(np_asarray(host_arr), other, backed)

        if states_is_device:
            return device_arr, host_arr
        return host_arr, device_arr

    def _trim_or_extend(
        self, arr: ndarray, values_object: SystemValues
    ) -> ndarray:
        """Extend incomplete arrays with defaults or trim extra values.

        Parameters
        ----------
        arr
            Array in (variable, run) format requiring adjustment.
        values_object
            System values object containing defaults and dimension metadata.

        Returns
        -------
        ndarray
            Array in (variable, run) format whose row count matches
            ``values_object.n``.
        """
        # If the array has fewer rows than the number of values, extend it
        # with default values
        if arr.shape[0] < values_object.n:
            n_runs = arr.shape[1]
            # Create padding with default values for missing variables
            padding = np_tile(
                values_object.values_array[arr.shape[0]:, np_newaxis],
                (1, n_runs)
            )
            arr = np_vstack([arr, padding])
        # If the array has more rows than expected, trim the extras
        elif arr.shape[0] > values_object.n:
            arr = arr[:values_object.n, :]
        return arr

    def _sanitise_arraylike(
        self, arr: Optional[ArrayLike], values_object: SystemValues
    ) -> Optional[ndarray]:
        """Convert array-likes to 2D arrays in (variable, run) format.

        Parameters
        ----------
        arr
            Array-like data describing sweep values. If 2D, expected in
            (variable, run) format.
        values_object
            System values object containing defaults and dimension metadata.

        Returns
        -------
        Optional[ndarray]
            Two-dimensional array in (variable, run) format sized to
            ``values_object`` or ``None`` when no data remain after
            sanitisation.

        Raises
        ------
        ValueError
            Raised when the input has more than two dimensions.

        Warns
        -----
        UserWarning
            Warned when the number of provided rows differs from the
            expected dimension.
        """
        # If no array provided, pass through None
        if arr is None:
            return arr
        # If the input is not already an ndarray, coerce it to one
        elif not isinstance(arr, ndarray):
            arr = np_asarray(arr)
        # Reject inputs with more than two dimensions explicitly
        if arr.ndim > 2:
            raise ValueError(
                f"Input must be a 1D or 2D array, but got a {arr.ndim}D array."
            )
        # Convert 1D vectors to single-column 2D arrays (one run)
        elif arr.ndim == 1:
            arr = arr[:, np_newaxis]

        # Warn and adjust arrays whose row count differs from expected
        if arr.shape[0] != values_object.n:
            warn(
                f"Provided input data has {arr.shape[0]} variables, but there "
                f"are {values_object.n} settable values. Missing values "
                f"will be filled with default values, and extras ignored."
            )
            arr = self._trim_or_extend(arr, values_object)
        # Empty arrays collapse to None
        if arr.size == 0:
            return None

        return arr  # correctly sized array just falls through untouched

    def _plan_single_input(
        self,
        input_data: Optional[Union[Dict, ArrayLike]],
        values_object: SystemValues,
        kind: str,
    ) -> dict:
        """Classify a single input category without materialising runs.

        Parameters
        ----------
        input_data
            Input as None (use defaults), dict (expand to grid),
            or array-like (sanitize).
        values_object
            SystemValues instance for this category (params or states).
        kind
            Grid type: "combinatorial" or "verbatim".

        Returns
        -------
        dict
            Plan with ``mode`` (``"empty"``, ``"defaults"``,
            ``"grid"``, or ``"array"``), ``n_runs``, and the mode's
            payload (``indices``/``grid`` or ``array``).

        Raises
        ------
        TypeError
            Raised when input_data is not None, dict, or array-like.
        ValueError
            Raised when non-empty input_data is provided but
            values_object has no variables, or a dict grid's row
            count does not match its indices.
        """
        # Handle empty SystemValues (system has no variables of this type)
        if values_object.empty:
            if input_data is not None:
                # Check if input is truly empty or has actual data
                is_empty_input = False
                if isinstance(input_data, dict) and len(input_data) == 0:
                    is_empty_input = True
                elif isinstance(input_data, ndarray) and input_data.size == 0:
                    is_empty_input = True
                elif isinstance(input_data, (list, tuple)) and len(input_data) == 0:
                    is_empty_input = True

                if not is_empty_input:
                    raise ValueError(
                        f"Grid values were provided but the system has no "
                        f"settable variables of this type. Expected None or "
                        f"empty input, got {type(input_data).__name__}."
                    )
            return {"mode": "empty", "n_runs": 1}

        # None -> defaults for every run
        if input_data is None:
            return {"mode": "defaults", "n_runs": 1}

        # Dict -> compact grid of the swept variables only
        if isinstance(input_data, dict):
            # Ensure all values are iterable by wrapping scalars
            input_data = {k: np_atleast_1d(v) for k, v in input_data.items()}
            indices, grid = generate_grid(input_data, values_object, kind=kind)
            if indices.size == 0:
                n_runs = grid.shape[1] if grid.ndim > 1 else 1
                return {"mode": "defaults", "n_runs": n_runs}
            # A 1D grid carries no swept columns: defaults, one run.
            if grid.ndim == 1:
                return {"mode": "defaults", "n_runs": 1}
            if grid.shape[0] != indices.shape[0]:
                raise ValueError("Grid shape does not match indices shape.")
            return {
                "mode": "grid",
                "n_runs": grid.shape[1],
                "indices": indices,
                "grid": grid,
            }

        # Array-like -> sanitize to 2D
        if isinstance(input_data, (list, tuple, ndarray)):
            sanitised = self._sanitise_arraylike(input_data, values_object)
            if sanitised is None:
                # Treat empty inputs like None: defaults, one run
                return {"mode": "defaults", "n_runs": 1}
            return {
                "mode": "array",
                "n_runs": sanitised.shape[1],
                "array": sanitised,
            }

        # Unsupported type
        raise TypeError(
            f"Input must be None, dict, or array-like, got {type(input_data)}"
        )

    def _fill_aligned(
        self,
        states_plan: dict,
        params_plan: dict,
        kind: str,
        backed: set,
    ) -> tuple[ndarray, ndarray]:
        """Materialise both categories at their aligned run count.

        Combinatorial results hold every pairing: state columns vary
        slowly, parameter columns vary quickly. Verbatim columns pair
        directly, broadcasting a single-run category. Each category
        is written straight into a policy-backed destination.

        Parameters
        ----------
        states_plan
            Plan for the states category.
        params_plan
            Plan for the params category.
        kind
            Grid type: "combinatorial" or "verbatim".
        backed
            Identity set of destination arrays this call allocated.

        Returns
        -------
        tuple[ndarray, ndarray]
            Aligned (states, params) arrays with matching run counts.

        Raises
        ------
        ValueError
            Raised when ``kind`` is ``"verbatim"`` and the run counts
            cannot be paired, or ``kind`` is unknown.
        """
        n_states = states_plan["n_runs"]
        n_params = params_plan["n_runs"]
        if kind == "combinatorial":
            states_array = self._fill_category(
                states_plan, self.states, backed,
                repeats=n_params, tiles=1,
            )
            params_array = self._fill_category(
                params_plan, self.parameters, backed,
                repeats=1, tiles=n_states,
            )
            return states_array, params_array
        elif kind == "verbatim":
            if n_states == 1 and n_params > 1:
                states_reps, params_reps = n_params, 1
            elif n_params == 1 and n_states > 1:
                states_reps, params_reps = 1, n_states
            elif n_states != n_params:
                raise ValueError(
                    "For 'verbatim', both grids must have the same number "
                    "of runs, or one grid must have exactly 1 run so it "
                    "can be broadcast to match the other."
                )
            else:
                states_reps, params_reps = 1, 1
            # Unexpanded arrays pair verbatim through materialisation.
            if states_plan["mode"] == "array" and states_reps == 1:
                states_array = states_plan["array"]
            else:
                states_array = self._fill_category(
                    states_plan, self.states, backed,
                    repeats=states_reps, tiles=1,
                )
            if params_plan["mode"] == "array" and params_reps == 1:
                params_array = params_plan["array"]
            else:
                params_array = self._fill_category(
                    params_plan, self.parameters, backed,
                    repeats=params_reps, tiles=1,
                )
            return states_array, params_array
        raise ValueError(
            f"Unknown grid type '{kind}'. Use 'combinatorial' or "
            f"'verbatim'."
        )

    def _fill_category(
        self,
        plan: dict,
        values_object: SystemValues,
        backed: set,
        repeats: int,
        tiles: int,
    ) -> ndarray:
        """Materialise one category into its final backing.

        Parameters
        ----------
        plan
            Plan from :meth:`_plan_single_input`.
        values_object
            SystemValues instance for this category.
        backed
            Identity set recording destination arrays this call
            allocated.
        repeats
            Consecutive copies of each source column in the result.
        tiles
            Whole-source cycles in the result.

        Returns
        -------
        ndarray
            Array in (variable, run) format at
            ``tiles * plan["n_runs"] * repeats`` runs.
        """
        total_runs = tiles * plan["n_runs"] * repeats
        if plan["mode"] == "empty":
            return np_empty((0, total_runs), dtype=self.precision)
        defaults = values_object.values_array
        if plan["mode"] == "defaults":
            return self._fill_defaults(values_object, total_runs, backed)
        if plan["mode"] == "grid":
            out = self._final_array(defaults.size, total_runs, backed)
            return extend_grid_to_array(
                plan["grid"], plan["indices"], defaults,
                out=out, repeats=repeats, tiles=tiles,
            )
        source = plan["array"]
        out = self._final_array(source.shape[0], total_runs, backed)
        view = out.reshape(
            source.shape[0], tiles, plan["n_runs"], repeats
        )
        view[:] = source[:, np_newaxis, :, np_newaxis]
        return out

    def _choose_backing(self, nbytes: int) -> str:
        """Pick the backing for a handler-materialised array."""
        return self.memory_manager.choose_host_memory_type(
            nbytes, self.host_spill_threshold
        )

    def _final_array(
        self, n_rows: int, n_runs: int, backed: set
    ) -> ndarray:
        """Allocate the final backing for one assembled category."""
        if n_rows == 0 or n_runs == 0:
            return np_empty((n_rows, n_runs), dtype=self.precision)
        nbytes = n_rows * n_runs * np_dtype(self.precision).itemsize
        memory_type = self._choose_backing(nbytes)
        array = self.memory_manager.create_host_array(
            (n_rows, n_runs),
            self.precision,
            memory_type,
            spill_directory=self.spill_directory,
        )
        backed.add(id(array))
        return array

    def _cast_to_precision(
        self,
        states: ndarray,
        params: ndarray,
        states_input: object,
        params_input: object,
        backed: frozenset = frozenset(),
    ) -> tuple[ndarray, ndarray]:
        """Return arrays in system precision, pinned when handler-owned.

        Parameters
        ----------
        states
            Initial state array in (variable, run) format.
        params
            Parameter array in (variable, run) format.
        states_input
            The caller's original ``states`` argument.
        params_input
            The caller's original ``params`` argument.
        backed
            Identity set of destination arrays this call allocated.

        Returns
        -------
        tuple of ndarray and ndarray
            State and parameter arrays with ``dtype`` matching
            ``self.precision``.
        """
        return (
            self._materialise(states, states_input, backed),
            self._materialise(params, params_input, backed),
        )

    def _materialise(
        self,
        array: ndarray,
        user_input: object,
        backed: frozenset = frozenset(),
    ) -> ndarray:
        """Return ``array`` in precision, copying at most once."""
        # Nothing to transfer: empty arrays pass through.
        if array.size == 0:
            return array
        # Arrays this call assembled directly into policy backing.
        if id(array) in backed:
            return array
        aligned = (
            array.dtype == self.precision
            and array.flags["C_CONTIGUOUS"]
        )
        # The caller's own array is used as-is, whatever its backing.
        if aligned and _views_user_input(array, user_input):
            return array
        nbytes = int(array.size) * np_dtype(self.precision).itemsize
        memory_type = self._choose_backing(nbytes)
        # Already transfer-ready in an acceptable backing: no copy.
        if aligned and (
            memory_type == "host"
            or is_pinned_array(array)
            or (memory_type == "memmap" and isinstance(array, np_memmap))
        ):
            return array
        # Copy once into a buffer of the chosen backing.
        return self.memory_manager.create_host_array(
            array.shape,
            self.precision,
            memory_type,
            like=array,
            spill_directory=self.spill_directory,
        )

    def _is_right_sized_array(
        self,
        arr: Optional[Union[ArrayLike, Dict]],
        values_object: SystemValues,
    ) -> bool:
        """Check if input is a right-sized 2D array.

        Parameters
        ----------
        arr
            Input to check.
        values_object
            SystemValues instance for dimension comparison.

        Returns
        -------
        bool
            True if arr is a 2D ndarray with correct variable count,
            or True if values_object is empty and arr is None.
        """
        # If the SystemValues is empty (no variables), consider it right-sized
        # if the array is None or an empty 2D array with 0 rows
        if values_object.empty:
            if arr is None:
                return True
            if isinstance(arr, ndarray) and arr.ndim == 2 and arr.shape[0] == 0:
                return True
            return False
        if not isinstance(arr, ndarray):
            return False
        if arr.ndim != 2:
            return False
        return arr.shape[0] == values_object.n

    def _is_1d_or_none(
        self,
        arr: Optional[Union[ArrayLike, Dict]],
    ) -> bool:
        """Check if input is None or a 1D array-like.

        Parameters
        ----------
        arr
            Input to check.

        Returns
        -------
        bool
            True if arr is None or a 1D array-like (list, tuple, 1D ndarray).
        """
        if arr is None:
            return True
        if isinstance(arr, dict):
            return False
        if isinstance(arr, ndarray):
            return arr.ndim == 1
        if isinstance(arr, (list, tuple)):
            # Check if flat (1D) - no nested lists/tuples
            # Use hasattr('__len__') to check for iterables, excluding scalars
            return not any(
                isinstance(x, (list, tuple)) or
                (isinstance(x, ndarray) and x.ndim > 0)
                for x in arr
            )
        return False

    def _fill_defaults(
        self,
        values_object: SystemValues,
        n_runs: int,
        backed: set,
    ) -> ndarray:
        """Broadcast defaults into a policy-backed (variable, run) array.

        Parameters
        ----------
        values_object
            SystemValues instance containing default values.
        n_runs
            Number of run columns to create.
        backed
            Identity set recording destination arrays this call
            allocated.

        Returns
        -------
        ndarray
            2D array in (variable, run) format with defaults.
        """
        defaults = values_object.values_array
        if defaults.size == 0:
            return np_empty((0, n_runs), dtype=self.precision)
        out = self._final_array(defaults.size, n_runs, backed)
        out[:] = defaults[:, np_newaxis]
        return out

    def _fast_return_arrays(
        self,
        states: Optional[Union[ArrayLike, Dict]],
        params: Optional[Union[ArrayLike, Dict]],
        kind: str,
    ) -> Optional[Tuple[ndarray, ndarray]]:
        """Attempt fast returns for pre-sized host array inputs."""
        states_input = states
        params_input = params
        states_runs = self._get_run_count(states)
        params_runs = None
        if not (self.parameters.empty and params is None):
            params_runs = self._get_run_count(params)

        states_ok = self._is_right_sized_array(states, self.states)
        params_ok = self._is_right_sized_array(params, self.parameters)

        if self.parameters.empty and params is None:
            params_ok = True
            params_runs = states_runs or 1
            params = np_empty((0, params_runs), dtype=self.precision)

        if states_ok and params_ok:
            if states_runs is not None and params_runs is not None:
                if states_runs == params_runs:
                    return self._cast_to_precision(
                        states, params, states_input, params_input
                    )

        states_small = self._is_1d_or_none(states)
        params_small = self._is_1d_or_none(params)

        if states_ok and params_small:
            backed = set()
            n_runs = states_runs if states_runs is not None else 1
            column = None
            if params is not None:
                column = self._sanitise_arraylike(params, self.parameters)
            if kind == "combinatorial":
                # The column broadcasts to n_runs, squaring the pairing.
                total = n_runs * n_runs
                states_array = self._final_array(
                    states.shape[0], total, backed
                )
                view = states_array.reshape(
                    states.shape[0], n_runs, n_runs
                )
                view[:] = states[:, :, np_newaxis]
            else:
                total = n_runs
                states_array = states
            if column is None:
                params_array = self._fill_defaults(
                    self.parameters, total, backed
                )
            else:
                params_array = self._final_array(
                    column.shape[0], total, backed
                )
                params_array[:] = column
            return self._cast_to_precision(
                states_array, params_array, states_input, params_input,
                backed,
            )

        if params_ok and states_small:
            backed = set()
            n_runs = params_runs if params_runs is not None else 1
            column = None
            if states is not None:
                column = self._sanitise_arraylike(states, self.states)
            if kind == "combinatorial":
                # Mirror branch: the params grid cycles whole.
                total = n_runs * n_runs
                params_array = self._final_array(
                    params.shape[0], total, backed
                )
                view = params_array.reshape(
                    params.shape[0], n_runs, n_runs
                )
                view[:] = params[:, np_newaxis, :]
            else:
                total = n_runs
                params_array = params
            if column is None:
                states_array = self._fill_defaults(
                    self.states, total, backed
                )
            else:
                states_array = self._final_array(
                    column.shape[0], total, backed
                )
                states_array[:] = column
            return self._cast_to_precision(
                states_array, params_array, states_input, params_input,
                backed,
            )

        return None

    def _get_run_count(self, arr: Optional[Union[ArrayLike, Dict]]) -> Optional[int]:
        """Return run count (columns) for 2D host arrays."""
        if isinstance(arr, ndarray) and arr.ndim == 2:
            return arr.shape[1]
        return None
