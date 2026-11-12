Creating a "System" of Differential Equations
=============================================
#BaseODe is long dead, do not mention it. This is implementation-focussed from the start, which doesn't bode well. We should be telling them that it understands them in the form of a Python function or in strings.
# It ooks like the only part of this intro I like is the opening line, which isn't reason enough to keep it. It could stay, then we would follow it with `create_ODE_system` creates a cubie-native SymbolicODE object out of your input, from either a function or from strings of math.

The first step in solving a system of differential equations is to create a
system of differential equations. Cubie understands ODEs in the form of a
:class:`GenericODE <cubie.odesystems.baseODE.BaseODE>` object, which holds the
\(\frac{dx}{dt}\) equations and descriptions and default values for all of the
variables that those equations depend on.

:func:`create_ODE_system <cubie.odesystems.symbolic.symbolicODE.create_ODE_system>`
builds one of these from either a plain Python function or a set of equation
strings. Both produce a
:class:`SymbolicODE <cubie.odesystems.symbolic.symbolicODE.SymbolicODE>`, which
does some extra tricks to produce extra helper functions for the integration
algorithms to use. ``BaseODE`` is the abstract interface, and ``SymbolicODE``
is currently its only concrete subclass.

From a Python function
----------------------

The default way to define a system is to write the right-hand side as a
Python function, in the same shape that ``scipy.integrate.solve_ivp`` uses:
time first, then the state, then a container of named values. Here is the
Lotka-Volterra predator-prey system, a common example of a nonlinear ODE
system:

.. code-block:: python
   :caption: Creating a system of ODEs from a Python function.

   import cubie as qb

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
       lotka_volterra,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )
   print(LV)

The values that you assign to each parameter and state when creating the 
system are set as the defaults. When you solve the system, you can omit 
values for the initial states or parameters that you're not varying across
a batch, and they will be set to these defaults instead.

Cubie never calls the function you provide. It reads the source, converts the
arithmetic into symbolic equations, and compiles those to CUDA, so the
function body must be plain arithmetic on the arguments (plus the maths
functions Cubie knows, such as ``np.sin``, ``exp``, and friends) rather
than any old arbitrary Python. 

You can read a state by attribute (``y.x``), by name (``y["x"]``), or by
index, where ``y[0]`` is the first state you list in ``states``. You can
read a parameter by attribute (``p.a``) or by name (``p["a"]``), but not by
index. The derivatives can be returned as a list or tuple in ``states``
order, or as a dict keyed by state name. 


From equation strings
---------------------

If your model already exists as equations on paper, you can hand Cubie the
equations directly as strings and let it parse them with SymPy:

.. code-block:: python
   :caption: Creating a system of ODEs from equation strings.

   import cubie as qb

   LV = qb.create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       name="LotkaVolterra",
   )
   print(LV)

In this example, Cubie has determined that x and y are state variables, as
they have \(\frac{dx}{dt}\) equations. \(\frac{dx}{dt}\) equations must
start with a "d" followed by the variable name. The variables a, b, c, and
d have been called "parameters", as they don't have \(\frac{dx}{dt}\)
equations, and they don't appear on the left-hand side of any equations.
The string form is happy to infer roles like this; you can also declare
``parameters`` and ``states`` explicitly, exactly as in
the function form, and provide default values:

.. code-block:: python
   :caption: Equation strings with explicit roles and defaults.

   import cubie as qb

   LV = qb.create_ODE_system(
       """
       predator_death_rate = c*y
       dx = a*x - b*x*y
       dy = -predator_death_rate + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       observables=["predator_death_rate"],
       name="LotkaVolterra",
   )
   print(LV)

Default values are optional (a list or tuple of names works too), but they
make it a little easier to specify which variables you're "batching" over
when you come to solve the system. Left-hand side assignments that do not
target known states or listed observables (like ``predator_death_rate``
above, had we not listed it) still participate in the symbolic
expressions, but they are stored as anonymous auxiliaries and their
trajectories are not saved.

#This is a new section, which I think should take the common points from strings and a function. There's still duplication, the string one talks about automatic identification and defaults, when that's common to both.
States, parameters, and observables

If you don't provide ``parameters`` or ``states``, cubie will try its best 
to identify them, finding any variable which is prefaced by a ``d`` as a state,
and any rhs terms that aren't states or derivatives as parameters.
If you don't provide default values, the defaults will be set to 0. 


You may want to record intermediate values that aren't states.To do this, you
can declare them ``observables``:

.. code-block:: python
   :caption: An observable computed on the way to the derivatives.
# get all the imports updated to import the function directly where the 
   import cubie as qb

   def lotka_volterra(t, y, p):
       predator_death_rate = p.c * y.y
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -predator_death_rate + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
       lotka_volterra,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       observables=["predator_death_rate"],
       name="LotkaVolterra",
   )
   print(LV)

Differences between the two forms
---------------------------------

The two forms produce identical systems and identical results for the same
equations. A few capabilities differ:
# this looks stale and dead
- **Undeclared variables.** The string form infers any undeclared
  right-hand-side symbol as a parameter with a default value of 0.0 and
  warns you. The function form does not infer: every name in the function
  body must resolve to an argument access, a local assignment, a declared
  variable, or the time argument, otherwise compilation fails with a
  ``NameError`` naming the stray symbol.
- **Custom helper functions.** The ``user_functions`` argument, which lets
  equation strings call your own callables
  (:doc:`/user_guide/userfunctions`), is only supported by the string
  form. Inside a Python-function system, only Cubie's built-in maths
  functions can be called.
- **State inference.** The string form can infer states from ``dx = ...``
  left-hand sides; the function form requires the ``states`` argument.

# This glossary should be somewhere else, not here. maybe it's own file, probably under systems still.
Cubie ODE System Glossary
-------------------------

- *States*: The variables that are being solved for. Each state variable must
  have a \(\frac{dx}{dt}\) equation. Each state variable must also have an
  initial value, which sets the starting point of the initial value problem.
- *Parameters*: Input variables that are not solved for. These set the behaviour
  of the system, and in Cubie, they are one of the two inputs that can be
  "batched", i.e. we can solve many IVPs with different parameter sets
  simultaneously.
- *Observables*: Also called auxiliary variables. These variables that are not
  solved for, but are derived from the state inputs and parameters. These
  typically pop up on the way to the \(\frac{dx}{dt}\) equations, and might
  represent physical quantities of interest in the system. Any state variables
  that don't have a \(\frac{dx}{dt}\) equation should be moved into observables.
- *Drivers*: Also called forcing terms. These are time-dependent inputs to the
  system. Cubie currently only supports one set of drivers per batch (i.e. all
  IVPs use the same driver), but this can be worked around by parameterising the
  driver function and passing a time vector as the driver function.

# No one cares about this here
Jacobians
---------

Implicit algorithms require the system's Jacobian for their internal
solvers.  CuBIE generates Jacobian helper functions symbolically and
caches them to disk.  For a full explanation of how this works and why,
see :doc:`/theory/jacobians`.
