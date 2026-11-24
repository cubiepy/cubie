Variables
=========

.. currentmodule:: cubie.odesystems.variables

When you parse a system, you get its states, parameters and observables
as a :class:`VariableSet` ordered by name. States and parameters keep
their default values in a :class:`ValuedSet`. A :class:`ParameterSet`
also records whether you sweep each parameter in a batch solve or fix it
at a value, and you get a new one for each combination of swept
parameters and fixed values you solve with.

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
