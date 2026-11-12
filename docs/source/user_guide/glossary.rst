Glossary
========

States
   The variables that cubie solves for. Each state has a derivative
   equation, :math:`\frac{dx}{dt}`, and an initial value, which sets the
   starting point of the initial value problem.

Parameters
   Inputs that cubie does not solve for. They set the behaviour of the
   system, and they are one of the two inputs you can batch over, so that
   cubie solves many IVPs with different parameter sets at once.

Observables
   Values that cubie does not solve for, but works out from the states and
   parameters. They usually appear on the way to the derivative equations,
   and might represent physical quantities of interest in the system. List
   a variable as an observable when you want to record it.

Drivers
   Time-dependent inputs to the system, also called forcing terms. Every
   run in a batch uses the same driver samples. To vary a forcing term
   between runs, write it as a function of ``t`` with parameters; see
   :doc:`drivers`.
