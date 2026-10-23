Variables
=========

.. currentmodule:: cubie.odesystems.variables

When you parse a system of equations, it records the variables it
identifies as states, parameters and observables. Each set is ordered by
name and kept as a :class:`VariableSet`; states and parameters keep
their default values in a :class:`ValuedSet`. A :class:`ParameterSet`
also captures whether a parameter is swept as part of a batch solve or
set to a fixed value. A new :class:`ParameterSet` is created for each
combination of swept parameters you solve over.

.. autoclass:: Variable
    :members:

.. autoclass:: State
    :members:
    :show-inheritance:

.. autoclass:: Parameter
    :members:
    :show-inheritance:

.. autoclass:: CompiledParameter
    :members:
    :show-inheritance:

.. autoclass:: VariableSet
    :members:

.. autoclass:: ValuedSet
    :members:
    :show-inheritance:

.. autoclass:: ParameterSet
    :members:
    :show-inheritance:
