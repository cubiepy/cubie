"""Load CellML models into CuBIE's symbolic ODE framework.

Wraps the ``cellmlmanip`` library to parse CellML files and convert
them into :class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
instances. Inspired by :mod:`chaste_codegen.model_with_conversions`
(MIT licence); only the subset required for basic model loading is
implemented.

Published Functions
-------------------
:func:`load_cellml_model`
    Parse a CellML file and return a fully initialised
    :class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`.

    >>> from cubie.odesystems.symbolic.parsing.cellml import (
    ...     load_cellml_model,
    ... )
    >>> ode = load_cellml_model("cardiac_model.cellml")
    >>> ode.num_states  # doctest: +SKIP
    18

Notes
-----
``cellmlmanip`` is vendored under
:mod:`cubie.vendored.cellmlmanip`, so no external install is
required.

See Also
--------
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Object returned by :func:`load_cellml_model`.
:func:`~cubie.odesystems.symbolic.parsing.parser.parse_input`
    String-based alternative for hand-written equations.
"""

from cubie.vendored import cellmlmanip

import sympy as sp
from pathlib import Path
import numpy as np
from typing import Optional, List
import re
import logging
import warnings

from cubie._utils import PrecisionDType
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import from_sympy
from cubie.time_logger import default_timelogger
from .cellml_cache import CellMLCache

logger = logging.getLogger(__name__)

# Register timing events for cellml import functions
# Module-level registration required for proper event tracking
default_timelogger.register_event(
    "codegen_cellml_load_model", "codegen",
    "Codegen time for cellmlmanip.load_model()"
)
default_timelogger.register_event(
    "codegen_cellml_symbol_conversion", "codegen",
    "Codegen time for converting Dummy symbols to Symbols"
)
default_timelogger.register_event(
    "codegen_cellml_equation_processing", "codegen",
    "Codegen time for processing differential and algebraic equations"
)
default_timelogger.register_event(
    "codegen_cellml_sympy_preparation", "codegen",
    "Codegen time for preparing SymPy equations for parser"
)


def _sanitize_symbol_name(name: str) -> str:
    """Sanitize CellML symbol names for Python identifiers.
    
    CellML uses $ for namespacing and allows names starting with _
    followed by numbers. We need to convert these to valid Python
    identifiers.
    """
    # Replace $ with _
    name = name.replace('$', '_')
    
    # Replace . with _
    name = name.replace('.', '_')
    
    # If name starts with _, check if next char is a digit
    # If so, prepend with 'var_' to make it valid
    if name.startswith('_') and len(name) > 1 and name[1].isdigit():
        name = 'var' + name
    
    # Ensure name doesn't start with a digit
    if name and name[0].isdigit():
        name = 'var_' + name
    
    # Replace any remaining invalid characters with _
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    return name


def _find_membrane_voltage(model) -> Optional[sp.Symbol]:
    """Return the state variable identified as membrane voltage.

    Matches the first membrane-component state whose lowercased name
    ends with ``$v`` or contains ``$voltage`` (e.g. ``membrane$V``), so
    that other membrane states such as ``membrane$reversal_potential``
    are not misidentified.

    Parameters
    ----------
    model
        A ``cellmlmanip`` model instance.

    Returns
    -------
    sympy.Symbol or None
        The membrane-voltage state variable, or ``None`` when no state
        name matches.
    """
    for state in model.get_state_variables():
        name = str(state).lower()
        if "membrane" in name and (
            name.endswith("$v") or "$voltage" in name
        ):
            return state
    return None


def _remove_fixable_singularities(model, voltage_variable) -> None:
    """Rewrite removable GHK singularities in a CellML model in place.

    Goldman-Hodgkin-Katz current terms of the form ``U / (exp(U) - 1)``
    evaluate to ``0 / 0`` at ``U == 0`` and produce non-finite
    gradients that break float32 Newton-Krylov solves. ``cellmlmanip``
    replaces them with a piecewise bridge across the singular point,
    given the membrane voltage variable.

    When ``voltage_variable`` is ``None`` the voltage state is
    auto-detected by name: on success the detected name is logged at
    INFO level; when no membrane voltage can be found a
    :class:`UserWarning` is issued and the rewrite is skipped so that
    non-cardiac models still load.

    Parameters
    ----------
    model
        A ``cellmlmanip`` model instance, mutated in place.
    voltage_variable
        Name of the membrane voltage variable, or ``None`` to
        auto-detect it.

    Raises
    ------
    ValueError
        When an explicitly named variable is not found in the model.
    """
    if voltage_variable is None:
        voltage = _find_membrane_voltage(model)
        if voltage is None:
            warnings.warn(
                "fix_singularities is enabled but no membrane voltage "
                "state could be auto-detected; skipping singularity "
                "removal. Pass voltage_variable to apply it.",
                UserWarning,
                stacklevel=3,
            )
            return
        logger.info(
            "fix_singularities: auto-detected membrane voltage '%s'",
            voltage,
        )
    else:
        try:
            voltage = model.get_variable_by_name(voltage_variable)
        except KeyError:
            raise ValueError(
                "Could not resolve membrane voltage variable "
                f"{voltage_variable!r} for singularity removal."
            )
    model.remove_fixable_singularities(voltage)


def load_cellml_model(
    path: str,
    precision: PrecisionDType = np.float32,
    name: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    observables: Optional[List[str]] = None,
    fix_singularities: bool = True,
    voltage_variable: Optional[str] = None,
    show_gui: bool = False,
):
    """Load a CellML model and return an initialized SymbolicODE system.

    This function uses the cellmlmanip library to parse CellML files
    and converts them into a ready-to-use SymbolicODE system with all
    differential equations and algebraic constraints properly configured.

    Parameters
    ----------
    path : str
        Filesystem path to the CellML source file. Must have .cellml
        extension and be a valid CellML 1.0 or 1.1 model file.
    precision : numpy dtype, optional
        Target floating-point precision for compiled kernels.
        Default is np.float32.
    name : str, optional
        Identifier for the generated system. If None, uses the
        filename without extension.
    parameters : list of str, optional
        List of symbol names to assign as parameters. Otherwise,
        these symbols become constants or anonymous auxiliaries.
    observables : list of str, optional
        List of symbol names to assign as observables. Otherwise,
        these symbols become anonymous auxiliaries.
    fix_singularities : bool, optional
        If True, rewrite removable Goldman-Hodgkin-Katz singularities
        (``U / (exp(U) - 1)``) with cellmlmanip's piecewise
        replacement before parsing. Default is True.
    voltage_variable : str, optional
        Name of the membrane voltage variable used by the singularity
        rewrite. If None while ``fix_singularities`` is True, the
        voltage state is auto-detected by name; when none is found a
        UserWarning is issued and the rewrite is skipped. Ignored when
        ``fix_singularities`` is False.
    show_gui : bool, optional
        If True, launch the constants/parameters editor GUI after
        loading. Default is False.

    Returns
    -------
    SymbolicODE
        Initialized ODE system ready for use with solve_ivp.
        State variables are configured with initial values from the
        CellML model, and algebraic equations are set up according
        to the parameters and observables specifications.

    Raises
    ------
    ImportError
        If cellmlmanip is not installed. Install with:
        pip install cellmlmanip
    TypeError
        If path is not a string.
    FileNotFoundError
        If the specified CellML file does not exist.
    ValueError
        If the file does not have .cellml extension, or if an explicit
        ``voltage_variable`` cannot be resolved for singularity removal.

    Examples
    --------
    Load a CellML model and run a simulation:

    >>> from cubie import load_cellml_model, solve_ivp
    >>> import numpy as np
    >>> 
    >>> # Load the model
    >>> ode_system = load_cellml_model("beeler_reuter_model_1977.cellml")
    >>> 
    >>> # Set up simulation
    >>> t_span = (0.0, 100.0)
    >>> initial_states = np.ones(ode_system.num_states, dtype=np.float32)
    >>> 
    >>> # Run simulation
    >>> result = solve_ivp(ode_system, t_span, initial_states)

    Notes
    -----
    - Differential equations become state equations in the ODE system
    - Algebraic equations become observables or anonymous auxiliaries
    - State variables are converted from sympy.Dummy to sympy.Symbol
    - Initial values from CellML are preserved in the ODE system
    - Supports CellML 1.0 and 1.1 formats
    - CellML models from Physiome repository are compatible
    - The cellmlmanip library handles the complex CellML XML parsing
    """
    # Validate input type
    if not isinstance(path, str):
        raise TypeError(
            f"path must be a string, got {type(path).__name__}"
        )
    
    # Validate file existence
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CellML file not found: {path}")
    
    # Validate file extension
    if not path.endswith('.cellml'):
        raise ValueError(
            f"File must have .cellml extension, got: {path}"
        )
    
    # Use filename as default name if not provided
    if name is None:
        name = path_obj.stem
    
    # When no GUI is requested, check the cache early to skip the
    # expensive cellmlmanip parse entirely.  When the GUI is active
    # the cache check must wait until after user edits.
    if not show_gui:
        cache = CellMLCache(model_name=name, cellml_path=path)
        args_hash = cache.compute_cache_key(
            parameters, observables, precision, name,
            fix_singularities=fix_singularities,
            voltage_variable=voltage_variable,
        )
        if cache.cache_valid(args_hash):
            cached_data = cache.load_from_cache(args_hash)
            # Entries without a definition predate constant
            # specialisation; reparse so the checkpoint exists.
            if cached_data is not None and (
                cached_data.get('definition') is not None
            ):
                from cubie.odesystems.symbolic.symbolicODE import (
                    SymbolicODE,
                )

                ode = SymbolicODE(
                    equations=cached_data['parsed_equations'],
                    all_indexed_bases=cached_data['indexed_bases'],
                    all_symbols=cached_data['all_symbols'],
                    fn_hash=cached_data['fn_hash'],
                    user_functions=cached_data['user_functions'],
                    name=cached_data['name'],
                    precision=precision,
                    mass=cached_data.get('mass'),
                    definition=cached_data['definition'],
                )
                default_timelogger.print_message(
                    f"Loaded {name} from CellML cache "
                    f"(config: {args_hash[:8]})"
                )
                return ode

    default_timelogger.start_event("codegen_cellml_load_model")
    model = cellmlmanip.load_model(path)
    if fix_singularities:
        # Rewrite removable GHK singularities before extracting
        # equations so the fix flows through parsing and codegen.
        _remove_fixable_singularities(model, voltage_variable)
    raw_states = list(model.get_state_variables())
    raw_derivatives = list(model.get_derivatives())
    default_timelogger.stop_event("codegen_cellml_load_model")
    
    # Extract state defaults and units.
    initial_values = {}
    state_units = {}
    
    default_timelogger.start_event("codegen_cellml_symbol_conversion")
    # Map CellML symbols to valid SymPy names. Rebuilding the SymPy
    # expression exposes equivalent terms before IR conversion.
    dummy_to_symbol = {}
    for raw_state in raw_states:
        clean_name = _sanitize_symbol_name(raw_state.name)
        dummy_to_symbol[raw_state] = sp.Symbol(clean_name)

        # Read the optional initial value.
        if hasattr(raw_state, 'initial_value') and raw_state.initial_value is not None:
            initial_values[clean_name] = float(raw_state.initial_value)

        # cellmlmanip Variables always carry units
        state_units[clean_name] = str(raw_state.units)
    
    # Collect units for the remaining symbols.
    all_symbol_units = {}
    
    # Map the single derivative variable to ``t``.
    time_variable = None
    if raw_derivatives:
        independent_variables = set()
        for derivative in raw_derivatives:
            if hasattr(derivative, "args") and len(derivative.args) > 1:
                independent_variables.add(derivative.args[1][0])
        if len(independent_variables) > 1:
            raise ValueError(
                "CellML model uses multiple independent variables in "
                "derivatives; CuBIE requires a single shared time "
                "variable."
            )
        if independent_variables:
            time_variable = next(iter(independent_variables))
            dummy_to_symbol[time_variable] = sp.Symbol("t", real=True)
    
    # Convert remaining symbols, including numeric quantities.
    for eq in model.equations:
        for atom in eq.atoms(sp.Dummy):
            if atom not in dummy_to_symbol:
                clean_name = _sanitize_symbol_name(atom.name)

                if atom.name.startswith('_'):
                    try:
                        value = float(atom.name[1:])
                        if value == int(value):
                            dummy_to_symbol[atom] = sp.Integer(int(value))
                        else:
                            dummy_to_symbol[atom] = sp.Float(value)
                        continue
                    except (ValueError, IndexError):
                        pass

                dummy_to_symbol[atom] = sp.Symbol(clean_name)

                # cellmlmanip Variables and Quantities always carry
                # units
                all_symbol_units[clean_name] = str(atom.units)
    default_timelogger.stop_event("codegen_cellml_symbol_conversion")

    default_timelogger.start_event("codegen_cellml_equation_processing")
    # Replace symbols, then convert all equations with one shared memo.
    convert_memo = {}
    dxdt_equations = []
    algebraic_pairs = []
    for eq in model.equations:
        rhs_ir = from_sympy(
            eq.rhs.xreplace(dummy_to_symbol), convert_memo
        )
        if eq.lhs in raw_derivatives:
            state_name = str(dummy_to_symbol[eq.lhs.args[0]])
            dxdt_equations.append((ir.sym(f"d{state_name}"), rhs_ir))
        else:
            algebraic_pairs.append(
                (
                    from_sympy(
                        eq.lhs.xreplace(dummy_to_symbol),
                        convert_memo,
                    ),
                    rhs_ir,
                )
            )
    default_timelogger.stop_event("codegen_cellml_equation_processing")

    default_timelogger.start_event("codegen_cellml_sympy_preparation")

    constants_dict = {}
    parameters_dict = {}
    algebraic_equation_tuples = []
    observable_units = {}

    if parameters is None:
        parameters_set = set()
    elif isinstance(parameters, dict):
        parameters_set = set(parameters.keys())
    else:
        parameters_set = set(parameters)

    for lhs_ir, rhs_ir in algebraic_pairs:
        if isinstance(rhs_ir, ir.Num):
            var_name = str(lhs_ir.name)
            var_value = float(rhs_ir.value)

            if var_name in parameters_set:
                parameters_dict[var_name] = var_value
            else:
                constants_dict[var_name] = var_value
        else:
            algebraic_equation_tuples.append((lhs_ir, rhs_ir))

            lhs_name = lhs_ir.name
            if lhs_name in all_symbol_units:
                observable_units[lhs_name] = all_symbol_units[lhs_name]

    all_equations = dxdt_equations + algebraic_equation_tuples
    
    parameter_units = {}
    if parameters:
        for param in parameters:
            if param in all_symbol_units:
                parameter_units[param] = all_symbol_units[param]
    
    if observables:
        for obs in observables:
            if obs not in observable_units and obs in all_symbol_units:
                observable_units[obs] = all_symbol_units[obs]
    
    if parameters is not None and isinstance(parameters, dict):
        # CellML-extracted values take precedence; the user dict only
        # adds entries for parameters that lack a CellML numeric value.
        parameters_dict = {**parameters, **parameters_dict}

    default_timelogger.stop_event("codegen_cellml_sympy_preparation")

    # ---- Pre-parse GUI (before cache key) ----
    # The GUI operates on raw dicts so the user's constant/parameter
    # choices are reflected in the cache key and codegen output.
    if show_gui:
        # no cover: start
        from cubie.gui.constants_editor import edit_pre_parse_dicts

        constant_units = {
            k: all_symbol_units.get(k, "")
            for k in constants_dict
        }
        constants_dict, parameters_dict, initial_values = (
            edit_pre_parse_dicts(
                constants_dict,
                parameters_dict,
                initial_values,
                constant_units=constant_units,
                parameter_units=parameter_units,
                state_units=state_units,
            )
        )
        # no cover: end

    # ---- Cache check (incorporates GUI choices) ----
    # Initialize cache manager with argument-based cache keys
    cache = CellMLCache(model_name=name, cellml_path=path)
    # Build the parameters list from the (possibly GUI-modified) dict
    # so the cache key reflects actual categorisation.
    effective_params = list(parameters_dict.keys()) or None
    cache_parameters = effective_params if show_gui else parameters
    args_hash = cache.compute_cache_key(
        cache_parameters, observables, precision, name,
        fix_singularities=fix_singularities,
        voltage_variable=voltage_variable,
        constant_values=constants_dict if show_gui else None,
        parameter_values=parameters_dict if show_gui else None,
        initial_values=initial_values if show_gui else None,
    )

    if cache.cache_valid(args_hash):
        cached_data = cache.load_from_cache(args_hash)
        # Entries without a definition predate constant
        # specialisation; reparse so the checkpoint exists.
        if cached_data is not None and (
            cached_data.get('definition') is not None
        ):
            from cubie.odesystems.symbolic.symbolicODE import SymbolicODE

            ode = SymbolicODE(
                equations=cached_data['parsed_equations'],
                all_indexed_bases=cached_data['indexed_bases'],
                all_symbols=cached_data['all_symbols'],
                fn_hash=cached_data['fn_hash'],
                user_functions=cached_data['user_functions'],
                name=cached_data['name'],
                precision=precision,
                mass=cached_data.get('mass'),
                definition=cached_data['definition'],
            )
            default_timelogger.print_message(
                f"Loaded {name} from CellML cache "
                f"(config: {args_hash[:8]})"
            )
            return ode

    # ---- Cache miss: parse from source ----
    # Import required modules for direct parse_input call
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
    from cubie.odesystems.symbolic.parsing import parse_input

    # Register parsing event (same as SymbolicODE.create)
    default_timelogger.register_event(
        "symbolic_ode_parsing",
        "codegen",
        "Codegen time for symbolic ODE parsing",
    )

    # Parse equations into structured components
    default_timelogger.start_event("symbolic_ode_parsing")
    sys_components = parse_input(
        dxdt=all_equations,
        states=initial_values if initial_values else None,
        observables=observables,
        parameters=parameters_dict if parameters_dict else None,
        constants=constants_dict if constants_dict else None,
        drivers=None,
        user_functions=None,
        strict=False,
        state_units=state_units if state_units else None,
        parameter_units=parameter_units if parameter_units else None,
        constant_units=None,
        observable_units=observable_units if observable_units else None,
        driver_units=None,
    )
    (
        index_map,
        all_symbols,
        functions,
        equations,
        fn_hash,
        simplified,
        definition,
    ) = sys_components
    default_timelogger.stop_event("symbolic_ode_parsing")

    mass = None
    if simplified is not None and simplified.mass_matrix is not None:
        from numpy import asarray

        mass = asarray(simplified.mass_matrix, dtype=precision)

    # Save to cache
    cache.save_to_cache(
        args_hash=args_hash,
        parsed_equations=equations,
        indexed_bases=index_map,
        all_symbols=all_symbols,
        user_functions=functions,
        fn_hash=fn_hash,
        precision=precision,
        name=name,
        mass=mass,
        definition=definition,
    )

    # Construct SymbolicODE directly (not via .create())
    symbolic_ode = SymbolicODE(
        equations=equations,
        all_indexed_bases=index_map,
        all_symbols=all_symbols,
        name=name,
        fn_hash=fn_hash,
        user_functions=functions,
        precision=precision,
        mass=mass,
        definition=definition,
    )

    return symbolic_ode
