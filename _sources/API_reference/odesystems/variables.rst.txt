Variables
=========

.. currentmodule:: cubie.odesystems.variables

A system declares its states, parameters and observables as frozen
records held in ordered sets. A set's order is the row order of every
array that holds one value per variable. A :class:`ParameterSet` holds
each parameter as the current batch compiles it: swept, so each run
reads it from the parameters array, or fixed at the value compiled into
the code.

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
