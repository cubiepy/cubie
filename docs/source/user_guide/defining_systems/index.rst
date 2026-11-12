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

You can build a system with time-dependent inputs (see drives), with your
own helper functions (see userfunctions), or loaded from a CellML file (cellml).

# Put the glossary here.

.. toctree::
   :maxdepth: 1

   systems
   glossary
   drivers
   userfunctions
   cellml
