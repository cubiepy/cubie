Variables
=========

.. currentmodule:: cubie.odesystems.variables

The parser records the variables it identifies as states, parameters
and observables, each set ordered by name and kept as a
:class:`VariableSet`. States and parameters keep their default values
in a :class:`ValuedSet`. A :class:`ParameterSet` also records whether
you sweep a parameter in a batch solve or hold it at a fixed value, and
we build a new one for each combination of swept parameters you solve
over.

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
