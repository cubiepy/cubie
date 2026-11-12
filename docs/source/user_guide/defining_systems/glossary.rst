Glossary
========

States
   The variables that cubie solves for. You give each state a derivative
   equation, :math:`\frac{dx}{dt}`, and an initial value at which its run
   starts.

Parameters
   Inputs that cubie does not solve for, constant per individual run in a batch.
   You can give these a default value, and include them when building a batch
   to solve over to integrate one run under each parameter set.

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
