Default resolution
==================

.. currentmodule:: cubie.batchsolving.resolve_defaults

These functions work out the settings that you don't give a
:class:`~cubie.batchsolving.solver.Solver` from the ones that you do
give, the nature of your ODE system, and the defaults of the algorithm.
:func:`resolve` returns the full set of settings, which you can read as
``solver.effective``.

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
