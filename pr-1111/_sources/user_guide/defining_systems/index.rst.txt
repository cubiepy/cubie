Defining systems
================

The first step in solving a system of differential equations is to create a
system of differential equations.
:func:`create_ODE_system <cubie.odesystems.symbolic.symbolicODE.create_ODE_system>`
creates a cubie-native
:class:`SymbolicODE <cubie.odesystems.symbolic.symbolicODE.SymbolicODE>`
object out of your input, from either a Python function or strings of
maths. You can also pass the function or strings straight to ``solve_ivp``,
which builds the system for you; build it yourself when you want to set
default values or reuse the system.

You can build a system with time-dependent inputs (see :doc:`drivers`),
with your own helper functions (see :doc:`userfunctions`), or loaded from a
CellML file (see :doc:`cellml`).

Glossary
--------

States
   The variables that cubie solves for. You give each state a derivative
   equation, :math:`\frac{dx}{dt}`, and an initial value at which its run
   starts.

Parameters
   Inputs that cubie does not solve for, which stay constant within each
   run of a batch. You can give them default values, and give several
   values for a parameter when you build a batch, so that cubie integrates
   one run for each parameter set.

Observables
   Values that cubie does not solve for, but works out from the states and
   parameters. You usually compute them on the way to the derivative
   equations, often as physical quantities of interest in the system. List
   a variable as an observable when you want to record it.

Drivers
   Time-dependent inputs to the system, also called forcing terms. cubie
   gives every run in a batch the same driver samples. To vary a forcing term
   between runs, write it as a function of ``t`` with parameters; see
   :doc:`/user_guide/defining_systems/drivers`.

.. toctree::
   :maxdepth: 1

   systems
   drivers
   userfunctions
   cellml
