Default resolution
==================

.. currentmodule:: cubie.batchsolving.resolve_defaults

These functions fill in every setting you leave out of a
:class:`~cubie.batchsolving.solver.Solver`, from the settings you gave,
the system and the algorithm. :func:`resolve` returns the result as the
solver's ``effective`` settings.

.. autofunction:: resolve

.. autoclass:: VariableSelection
    :members:
    :exclude-members: observable_indices, state_indices, variables
    :show-inheritance:

.. autofunction:: given_or

.. autofunction:: resolve_controller

.. autofunction:: resolve_step_bounds

.. autofunction:: resolve_step_defaults

.. autofunction:: resolve_inner_tolerances

.. autofunction:: resolve_output_selection

.. autofunction:: output_flags

.. autofunction:: resolve_loop_timing

.. autofunction:: check_loop_timing

.. autofunction:: check_duration

.. autodata:: DEFAULT_TOLERANCE

.. autodata:: STEP_BOUND_DECADES

.. autodata:: DT_MIN_DECADES
