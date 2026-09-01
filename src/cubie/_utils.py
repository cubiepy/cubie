"""Utility helpers used throughout :mod:`cubie`.

This module provides general-purpose helpers for array slicing,
dictionary updates, attrs validators/converters, and CUDA utilities
shared across the code base.

Published Types
---------------
:data:`PrecisionDType`
    Union type alias for supported floating-point precision dtypes.

Published Functions
-------------------
:func:`build_config`
    Construct an attrs config instance from required and optional kwargs.

    >>> import numpy as np
    >>> from cubie.CUDAFactory import CUDAFactoryConfig
    >>> cfg = build_config(CUDAFactoryConfig, {"precision": np.float64})
    >>> cfg.precision
    <class 'numpy.float64'>

:func:`merge_kwargs_into_settings`
    Filter kwargs against an allowlist and merge with user settings.

    >>> merged, unused = merge_kwargs_into_settings(
    ...     {"a": 1, "b": 2}, valid_keys={"a"}
    ... )
    >>> merged
    {'a': 1}

:func:`ensure_nonzero_size`
    Replace zero-size shapes with minimal placeholders for safe CUDA
    allocation.

    >>> ensure_nonzero_size((0, 5))
    (1, 1)

:func:`unpack_dict_values`
    Flatten nested dicts for update pipelines.

    >>> result, keys = unpack_dict_values({"g": {"x": 1}, "y": 2})
    >>> result
    {'x': 1, 'y': 2}

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Base config class that uses validators and converters from this
    module.
:mod:`cubie.buffer_registry`
    Buffer registry that uses ``getype_validator`` and
    ``buffer_dtype_validator``.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, Iterable, Set
from warnings import warn

from numpy import (
    all as np_all,
    array as np_array,
    array_equal,
    asarray,
    dtype as np_dtype,
    float16 as np_float16,
    float32 as np_float32,
    float64 as np_float64,
    floating as np_floating,
    full,
    int32 as np_int32,
    int64 as np_int64,
    integer as np_integer,
    isfinite as np_isfinite,
    isscalar,
    ndarray,
)
from numpy.typing import ArrayLike
from cubie.cuda_simsafe import cuda
from attrs import NOTHING, Attribute, Factory, fields, has, validators
from cubie.cuda_simsafe import compile_kwargs, fmax, fmin, is_devfunc

PrecisionDType = Union[
    type[np_float16],
    type[np_float32],
    type[np_float64],
    np_dtype[np_float16],
    np_dtype[np_float32],
    np_dtype[np_float64],
]

ALLOWED_PRECISIONS = {
    np_dtype(np_float16),
    np_dtype(np_float32),
    np_dtype(np_float64),
}

_package_source_hashes: Dict[str, str] = {}


def package_source_hash(package_dir: Optional[Path] = None) -> str:
    """Content hash of every Python source file under a package.

    Both cache layers fold this digest into their keys so that any
    edit to the installed cubie source invalidates compiled kernels
    and generated code, which are otherwise keyed only on
    configuration values and the system definition.

    Parameters
    ----------
    package_dir
        Directory whose ``**/*.py`` files are hashed. Defaults to the
        installed ``cubie`` package directory.

    Returns
    -------
    str
        64-character SHA256 hex digest over the sorted relative paths
        and file contents. Memoised per directory for the lifetime of
        the process.
    """
    if package_dir is None:
        package_dir = Path(__file__).parent
    package_dir = Path(package_dir).resolve()
    key = str(package_dir)
    cached = _package_source_hashes.get(key)
    if cached is not None:
        return cached
    digest = sha256()
    for source_path in sorted(package_dir.rglob("*.py")):
        relative = source_path.relative_to(package_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(source_path.read_bytes())
        digest.update(b"\x00")
    result = digest.hexdigest()
    _package_source_hashes[key] = result
    return result


ALLOWED_BUFFER_DTYPES = {
    np_dtype(np_float16),
    np_dtype(np_float32),
    np_dtype(np_float64),
    np_dtype(np_int32),
    np_dtype(np_int64),
}


def precision_converter(value: PrecisionDType) -> type[np_floating]:
    """Return a canonical NumPy scalar type for precision configuration."""

    dtype_ = np_dtype(value)
    if dtype_ not in ALLOWED_PRECISIONS:
        raise ValueError(
            "precision must be one of float16, float32, or float64",
        )
    return dtype_.type


def precision_validator(
    _: object,
    __: Attribute,
    value: PrecisionDType,
) -> None:
    """Validate that ``value`` resolves to a supported precision."""

    if np_dtype(value) not in ALLOWED_PRECISIONS:
        raise ValueError(
            "precision must be one of float16, float32, or float64",
        )


def buffer_dtype_validator(
    _: object,
    __: Attribute,
    value: type,
) -> None:
    """Validate that value is a supported buffer dtype (float or int)."""
    if np_dtype(value) not in ALLOWED_BUFFER_DTYPES:
        raise ValueError(
            "Buffer dtype must be one of float16, float32, float64, "
            "int32, or int64",
        )


def slice_variable_dimension(slices, indices, ndim):
    """Create a combined slice for selected dimensions.

    Parameters
    ----------
    slices : slice or list[slice]
        Slice to apply to each index in ``indices``.
    indices : int or list[int]
        Dimension indices corresponding to ``slices``.
    ndim : int
        Total number of dimensions of the target array.

    Returns
    -------
    tuple
        Tuple of slice objects with ``slices`` applied to ``indices``.

    Raises
    ------
    ValueError
        If ``slices`` and ``indices`` differ in length or indices exceed
        ``ndim``.
    """
    if isinstance(slices, slice):
        slices = [slices]
    if isinstance(indices, int):
        indices = [indices]
    if len(slices) != len(indices):
        raise ValueError("slices and indices must have the same length")
    if max(indices) >= ndim:
        raise ValueError("indices must be less than ndim")

    outslice = [slice(None)] * ndim
    for i, s in zip(indices, slices):
        outslice[i] = s

    return tuple(outslice)


def in_attr(name, attrs_class_instance):
    """Check whether a field exists on an attrs class instance.

    Parameters
    ----------
    name : str
        Field name to query.
    attrs_class_instance : attrs class
        Instance whose fields are inspected.

    Returns
    -------
    bool
        ``True`` if ``name`` or ``_name`` is a field of the instance.
    """
    field_names = {
        field.name for field in fields(attrs_class_instance.__class__)
    }
    return name in field_names or ("_" + name) in field_names


def merge_kwargs_into_settings(
    kwargs: dict[str, object],
    valid_keys: Iterable[str],
    user_settings: Optional[dict[str, object]] = None,
) -> Tuple[dict[str, object], set[str]]:
    """Merge component settings from ``kwargs`` and ``user_settings``.

    Parameters
    ----------
    kwargs
        Keyword arguments supplied directly to a component.
    valid_keys
        Iterable of keys recognised by the component. Only these keys are
        extracted from ``kwargs``.
    user_settings
        Explicit settings dictionary supplied by the caller. When provided,
        these values supply defaults that keyword arguments may override.
        ``None`` values are ignored in both inputs.

    Returns
    -------
    merged
        Dictionary containing recognised settings with keyword arguments
        overriding values from ``user_settings``.
    recognized
        Keys in ``kwargs`` accepted by the component.
    """

    allowed = set(valid_keys)
    recognized = {key for key in kwargs if key in allowed}
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in allowed and value is not None
    }
    if user_settings is None:
        user_settings = {}
    else:
        user_settings = {
            key: value
            for key, value in user_settings.items()
            if value is not None
        }
    duplicates = {key for key in filtered if key in user_settings}
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        warn(
            (
                "Duplicate settings were provided for keys "
                f"{{{joined}}}; values from keyword arguments take "
                "precedence over the explicit settings dictionary."
            ),
            UserWarning,
            stacklevel=2,
        )

    user_settings.update(filtered)
    return user_settings, recognized


def optional_tuple_converter(value: Optional[Iterable]) -> Optional[tuple]:
    """Return ``value`` as a tuple, passing ``None`` through."""
    if value is None:
        return None
    return tuple(value)


def clamp_factory(precision):
    """Compile a CUDA device function that clamps a value to a range.

    Parameters
    ----------
    precision
        NumPy dtype used for the clamp operation.

    Returns
    -------
    Callable
        CUDA device function ``clamp(value, minimum, maximum)``.
    """
    from cubie.cuda_simsafe import numba_from_dtype as from_dtype
    precision = from_dtype(precision)

    # no cover: start
    @cuda.jit(
        # precision(precision, precision, precision),
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def clamp(value, minimum, maximum):
        return fmax(minimum, fmin(value, maximum))

    # no cover: end
    return clamp


def is_device_validator(instance, attribute, value):
    """Validate that a value is a Numba CUDA device function."""
    if not is_devfunc(value):
        raise TypeError(
            f"{attribute} must be a Numba CUDA device function,"
            f"got {type(value)}."
        )


def float_array_validator(instance, attribute, value):
    """Validate that a value is a NumPy np_floating-point array with finite
    values.

    Raises a TypeError if the value is not a NumPy ndarray of floats, and a
    ValueError if any elements are NaN or infinite.
    """
    if not isinstance(value, ndarray):
        raise TypeError(
            f"{attribute} must be a numpy array of floats, got {type(value)}."
        )
    if value.dtype.kind != "f":
        raise TypeError(
            f"{attribute} must be a numpy array of floats, got "
            f"dtype {value.dtype}."
        )
    if not np_all(np_isfinite(value)):
        raise ValueError(f"{attribute} must not contain NaNs or infinities.")


def nonnegative_float_array_validator(instance, attribute, value):
    """Validate a finite float array with no negative elements.

    Applies :func:`float_array_validator`, then raises a ValueError if
    any element is negative. Used for tolerance arrays, where negative
    values would corrupt the scaled error norm.
    """
    float_array_validator(instance, attribute, value)
    if not np_all(value >= 0):
        raise ValueError(f"{attribute} must not contain negative values.")


def inrangetype_validator(dtype, min_, max_):
    """Return a composite attrs validator for type and range checks.

    Parameters
    ----------
    dtype
        Expected Python or NumPy type (``float`` and ``int`` are
        expanded to accept their NumPy equivalents).
    min_
        Minimum allowed value (inclusive).
    max_
        Maximum allowed value (inclusive).

    Returns
    -------
    attrs validator
        Composed validator checking ``instance_of``, ``ge``, and ``le``.
    """
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)),
        validators.ge(min_),
        validators.le(max_),
    )


# Helper: expand Python dtype to accept corresponding NumPy scalar hierarchy
# e.g. float -> (float, np_floating), int -> (int, np_integer)
# Unknown types are returned unchanged.


def _expand_dtype(data_type):
    if data_type is float:
        return (float, np_floating)
    if data_type is int:
        return (int, np_integer)
    return data_type


def gttype_validator(dtype, min_):
    """Return a composite attrs validator for type and lower-bound check.

    Parameters
    ----------
    dtype
        Expected type (``float``/``int`` expanded for NumPy scalars).
    min_
        Exclusive lower bound.

    Returns
    -------
    attrs validator
        Composed validator checking ``instance_of`` and ``gt``.
    """
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)), validators.gt(min_)
    )


def getype_validator(dtype, min_):
    """Return a composite attrs validator for type and lower-bound check.

    Parameters
    ----------
    dtype
        Expected type (``float``/``int`` expanded for NumPy scalars).
    min_
        Inclusive lower bound.

    Returns
    -------
    attrs validator
        Composed validator checking ``instance_of`` and ``ge``.
    """
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)), validators.ge(min_)
    )


def tol_converter(
    value: Union[float, ArrayLike],
    self_: Any,
) -> ndarray:
    """Convert tolerance input into an array with target precision.

    For use as an attrs Converter with takes_self=True, converting
    scalar or array-like tolerance specifications into arrays of
    shape (tol_length,) with dtype matching self_.precision.

    Parameters
    ----------
    value
        Scalar or array-like tolerance specification.
    self_
        Configuration instance providing precision and dimension
        information. Must have `tol_length` (int) and
        `precision` attributes.

    Returns
    -------
    numpy.ndarray
        Tolerance array with one value per state variable. Always an
        owned, read-only copy: neither the caller's array nor the
        stored one can mutate compile-critical tolerance state after
        the snapshot hashes.

    Raises
    ------
    ValueError
        Raised when ``value`` cannot be broadcast to shape
        (tol_length,).
    """
    if isscalar(value):
        tol = full(self_.tol_length, value, dtype=self_.precision)
    else:
        # Copy: asarray would alias a caller array that already has
        # the target dtype, leaving stored tolerances writable.
        tol = np_array(value, dtype=self_.precision)
        if tol.shape[0] != self_.tol_length:
            # A uniform array is a scalar specification: broadcast
            # it to the configured length.
            if all(tol == tol[0]):
                tol = full(
                    self_.tol_length, tol[0], dtype=self_.precision
                )
            else:
                raise ValueError(
                    "tol must have shape (tol_length,)."
                )
    tol.setflags(write=False)
    return tol


def opt_gttype_validator(dtype, min_):
    """Optional validator that accepts None or values greater than min."""
    return validators.optional(gttype_validator(dtype, min_))


def opt_getype_validator(dtype, min_):
    """Optional validator that accepts None or values greater than or equal to
    min.
    """
    return validators.optional(getype_validator(dtype, min_))


def ensure_nonzero_size(
    value: Union[int, Tuple[int, ...]],
) -> Union[int, Tuple[int, ...]]:
    """
    Replace zero-size shapes with minimal placeholder shapes for safe
    allocation.

    When creating CUDA local arrays, zero-sized dimensions cause errors. This
    function converts shapes containing any zero (or None) to minimal size-1
    placeholder shapes. If ANY dimension is zero, the entire shape becomes
    all 1s, creating a minimal memory footprint for inactive arrays.

    Parameters
    ----------
    value : Union[int, Tuple[int, ...]]
        Input value or tuple of values to process.

    Returns
    -------
    Union[int, Tuple[int, ...]]
        For integers: max(1, value).
        For tuples: if ANY element is 0 or None, returns tuple of all 1s
        with the same length. If no zeros/Nones, returns original tuple.
        Non-numeric values in tuples are treated as valid (non-zero).
        Other types are passed through unchanged.

    Examples
    --------
    >>> ensure_nonzero_size(0)
    1
    >>> ensure_nonzero_size(5)
    5
    >>> ensure_nonzero_size((0, 5))
    (1, 1)
    >>> ensure_nonzero_size((0, 2, 0))
    (1, 1, 1)
    >>> ensure_nonzero_size((2, 3, 4))
    (2, 3, 4)
    >>> ensure_nonzero_size((0, None))
    (1, 1)
    """
    if isinstance(value, int):
        return max(1, value)
    elif isinstance(value, tuple):
        # If ANY element is 0 or None, return all-ones tuple
        has_zero = any(
            (isinstance(v, (int, float)) and v == 0) or v is None
            for v in value
        )
        if has_zero:
            return tuple(1 for _ in value)
        return value
    else:
        return value


def mass_equal(left, right) -> bool:
    """Compare two ODE mass matrices for attrs equality.

    A mass matrix (``ODEData._mass``) may be ``None``, a
    :class:`numpy.ndarray`, or a :class:`sympy.Matrix`. Used as the ``eq``
    comparator so a mass change participates in the config hash and forces
    recompilation.

    Parameters
    ----------
    left, right
        Mass-matrix operands, each ``None``, an ``ndarray``, or a
        ``sympy.Matrix``.

    Returns
    -------
    bool
        ``True`` only when both operands are the same kind and hold equal
        entries.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, ndarray) or isinstance(right, ndarray):
        return array_equal(asarray(left), asarray(right))
    return bool(left == right)


def unpack_dict_values(updates_dict: dict) -> Tuple[dict, Set[str]]:
    """Unpack dict values into flat key-value pairs.

    When an update() method receives parameters grouped in dicts, this
    utility flattens them before distributing to sub-components. The
    original dict keys are tracked separately so they can be marked as
    recognized even though they don't correspond to actual parameters.

    Parameters
    ----------
    updates_dict
        Dictionary potentially containing dicts as values

    Returns
    -------
    Tuple[dict, Set[str]]
        - dict: Flattened dictionary with dict values unpacked
        - set: Set of original keys that were unpacked dicts

    Examples
    --------
    >>> import numpy as np
    >>> result, unpacked = unpack_dict_values(
    ...     {
    ...         "step_settings": {"dt_min": 0.01, "dt_max": 1.0},
    ...         "precision": np.float32,
    ...     }
    ... )
    >>> result
    {'dt_min': 0.01, 'dt_max': 1.0, 'precision': <class 'numpy.float32'>}
    >>> unpacked
    {'step_settings'}

    Notes
    -----
    If a value in the input dict is itself a dict, its key-value pairs
    are added to the result dict directly, and the original key is
    tracked in the unpacked set. Regular key-value pairs are preserved
    as-is.

    Only unpacks one level deep - nested dicts within dict values are
    not recursively unpacked. This allows each level of the update chain
    to handle its own unpacking.

    Raises
    ------
    ValueError
        If a key appears both as a regular entry and within an unpacked
        dict, indicating a collision that would lead to ambiguous behavior.
    """
    result = {}
    unpacked_keys = set()
    for key, value in updates_dict.items():
        if isinstance(value, dict):
            # Check for key collisions before unpacking
            collision_keys = set(value.keys()) & set(result.keys())
            if collision_keys:
                raise ValueError(
                    f"Key collision detected: the following keys appear "
                    f"both as regular entries and within an unpacked dict: "
                    f"{sorted(collision_keys)}"
                )
            # Unpack the dict value and track the original key
            result.update(value)
            unpacked_keys.add(key)
        else:
            # Check if key already exists in result
            if key in result:
                raise ValueError(
                    f"Key collision detected: the key '{key}' appears "
                    f"multiple times in updates_dict."
                )
            result[key] = value
    return result, unpacked_keys


def build_config(
    config_class: type, required: dict, instance_label: str = "", **optional
) -> Any:
    """Build attrs config instance from required and optional parameters.

    Merges required parameters with optional overrides and passes them to the
    attrs config class constructor. The config class itself defines defaults
    for optional fields - this function simply filters and routes kwargs.

    Parameters
    ----------
    config_class : type
        Attrs class to instantiate (e.g., DIRKStepConfig).
    required : dict
        Required parameters that must be provided. These are typically
        function parameters like precision, n, evaluate_f.
    instance_label : str, optional
        Instance label for MultipleInstanceCUDAFactoryConfig classes.
        When provided, prefixed keys (e.g., 'krylov_atol') are
        transformed to unprefixed keys ('atol') before field matching.
        Default is empty string (no prefix transformation).
    **optional
        Optional parameter overrides passed to the config constructor.
        Extra keys not in the config class signature are ignored.

    Returns
    -------
    config_class instance
        Configured attrs object.

    Raises
    ------
    TypeError
        If config_class is not an attrs class.

    Examples
    --------
    >>> import numpy as np
    >>> # Without instance_label
    >>> config = build_config(
    ...     DIRKStepConfig,
    ...     required={"precision": np.float32, "n": 3},
    ... )
    >>>
    >>> # With instance_label (prefix transformation)
    >>> config = build_config(
    ...     ScaledNormConfig,
    ...     required={"precision": np.float32, "n": 3},
    ...     instance_label="krylov",
    ...     krylov_atol=1e-6,  # Transformed to atol
    ... )

    Notes
    -----
    The helper:
    - Merges required and optional kwargs
    - Applies prefix transformation if instance_label is provided and
      config_class has get_prefixed_attributes
    - Converts field names to aliases for underscore-prefixed attrs fields
    - Filters to only valid fields (ignores extra keys)
    - Lets attrs handle defaults for unspecified optional parameters
    """
    if not has(config_class):
        raise TypeError(f"{config_class.__name__} is not an attrs class")

    # Merge all inputs; required/optional are user-facing distinctions.
    merged = {**required, **optional}

    field_to_external = {}
    prefixed_attrs = set()
    prefix = ""
    # Generate prefix if instance_label provided and applicable
    if instance_label:
        # Add instance_label to merged for config constructor
        merged["instance_label"] = instance_label

        if not hasattr(config_class, "instance_label"):
            raise ValueError(
                f"instance_label '{instance_label}' is not valid for "
                f"{config_class.__name__}. Use `instance_label` for "
                f"MultipleInstanceCUDAFactoryConfig classes whose attributes "
                f"are prefaced, i.e. a solver with max_iters might have "
                f"`instance_label='newton'` so that newton_max_iters is "
                f"recognised in init/updates."
            )
        else:
            prefixed_attrs = config_class.get_prefixed_attributes(aliases=True)
            prefix = f"{instance_label}_"

    # Get external handle (Cubie keyword argument) and init handle (
    # attrs init arg). Always use aliases, prefix external handle if
    # applicable.
    for field in fields(config_class):
        # Non-init fields take no constructor kwarg, but circulating
        # settings dicts may still carry their keys.
        if not field.init:
            continue
        name = field.name
        alias = field.alias
        handle = alias if alias is not None else name
        is_prefixed = handle in prefixed_attrs

        external_handle = f"{prefix}{handle}" if is_prefixed else handle
        field_to_external[external_handle] = handle

    # Keep the provided fields, keyed by init handle.
    final = {
        field_to_external[k]: v
        for k, v in merged.items()
        if k in field_to_external
    }

    # Loose keys of a nested attrs field derive that field from its default.
    for fld in fields(config_class):
        handle = fld.alias if fld.alias is not None else fld.name
        if not fld.init or handle in final or not _is_nested_config(fld):
            continue
        default = fld.default
        if isinstance(default, Factory):
            default = default.factory()
        if default is NOTHING or default is None:
            continue
        nested, _, changed = default.update(merged)
        if changed:
            final[handle] = nested

    return config_class(**final)


def _is_nested_config(fld: Attribute) -> bool:
    """Return whether ``fld`` holds an attrs class with ``update``."""
    from typing import Union, get_args, get_origin

    candidates = (fld.type,)
    if get_origin(fld.type) is Union:
        candidates = get_args(fld.type)
    return any(
        isinstance(candidate, type)
        and has(candidate)
        and callable(getattr(candidate, "update", None))
        for candidate in candidates
    )
