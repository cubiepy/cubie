Creating a "System" of Differential Equations
=============================================

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
       constants={"a": 0.1, "c": 0.3},
       parameters={"b": 0.02, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )
   print(LV)

Cubie never calls this function. It reads the source, converts the
arithmetic into symbolic equations, and compiles those to CUDA, so the
function body must be plain arithmetic on the arguments (plus the maths
functions Cubie knows, such as ``np.sin``, ``exp``, and friends) rather
than arbitrary Python. Because the source is read with ``inspect``, the
function must live in a real file or a notebook cell; a function typed
into a bare REPL cannot be parsed.

States and named values can be accessed by attribute (``y.x``), by name
(``y["x"]``), or by index (``y[0]``). The derivatives can be returned as a
list or tuple in ``states`` order, or as a dict keyed by state name. The
keyword arguments sort your variables into roles:

- ``states`` are the variables being solved for. The dict values are
  default initial values, which the solver can override per run. The
  function form requires ``states``, since the returned derivatives are
  matched against it.
- ``parameters`` are inputs that can take a different value in every run
  of a batch. Alongside initial values, they form the inputs you can
  "batch" over.
- ``constants`` are inputs that hold one value for the whole batch. They
  are baked into the compiled GPU code, which typically speeds up the
  solve and lets you fit more IVPs in a batch, as Cubie doesn't need to
  find a place in memory for them. Any parameter that won't change within
  a batch should be a constant.

Intermediate local assignments in the function body work too. They are
treated as anonymous auxiliary variables unless you name them in
``observables``, in which case their trajectories can be recorded:

.. code-block:: python
   :caption: An observable computed on the way to the derivatives.

   import cubie as qb

   def lotka_volterra(t, y, p):
       predator_death_rate = p.c * y.y
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -predator_death_rate + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
       lotka_volterra,
       constants={"a": 0.1, "c": 0.3},
       parameters={"b": 0.02, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       observables=["predator_death_rate"],
       name="LotkaVolterra",
   )
   print(LV)

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
``constants``, ``parameters``, and ``states`` explicitly, exactly as in
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
       constants={"a": 0.1, "c": 0.3},
       parameters={"b": 0.02, "d": 0.01},
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

Differences between the two forms
---------------------------------

The two forms produce identical systems and identical results for the same
equations. A few capabilities differ:

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

Solver helpers
--------------

The integration algorithms retrieve system-specific device functions
through :meth:`SymbolicODE.get_solver_helper` — linear operators,
Jacobian preparation and Jacobian-vector products, Neumann and Jacobi
preconditioners, stage residuals for implicit methods, and the
explicit time-derivative helper.  You normally never call this
yourself; the full list of helper names is documented on the method
itself.

Cubie ODE System Glossary
-------------------------

- *States*: The variables that are being solved for. Each state variable must
  have a \(\frac{dx}{dt}\) equation. Each state variable must also have an
  initial value, which sets the starting point of the initial value problem.
- *Parameters*: Input variables that are not solved for. These set the behaviour
  of the system, and in Cubie, they are one of the two inputs that can be
  "batched", i.e. we can solve many IVPs with different parameter sets
  simultaneously.
- *Constants*: Input variables that are not solved for, and do not change
  between IVPs in a single batch. You can still change constants between
  batches, but it will add a little overhead as the CUDA machine recompiles the
  problem. Any parameters which will not change in a certain batch should be
  moved to constants, as this will speed up the solving process.
- *Observables*: Also called auxiliary variables. These variables that are not
  solved for, but are derived from the state inputs and parameters. These
  typically pop up on the way to the \(\frac{dx}{dt}\) equations, and might
  represent physical quantities of interest in the system. Any state variables
  that don't have a \(\frac{dx}{dt}\) equation should be moved into observables.
- *Drivers*: Also called forcing terms. These are time-dependent inputs to the
  system. Cubie currently only supports one set of drivers per batch (i.e. all
  IVPs use the same driver), but this can be worked around by parameterising the
  driver function and passing a time vector as the driver function.

Jacobians
---------

Implicit algorithms require the system's Jacobian for their internal
solvers.  CuBIE generates Jacobian helper functions symbolically and
caches them to disk.  For a full explanation of how this works and why,
see :doc:`/theory/jacobians`.
