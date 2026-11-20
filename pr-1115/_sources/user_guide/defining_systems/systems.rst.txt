Creating a "System" of Differential Equations
=============================================

From a Python function
----------------------

The default way to define a system is to write the right-hand side as a
Python function, in the same shape that ``scipy.integrate.solve_ivp`` uses:
time first, then the state, then a container of named values. Here is the
Lotka-Volterra predator-prey system, a common example of a nonlinear ODE
system:

.. code-block:: python
   :caption: Creating a system of ODEs from a Python function.

   from cubie import create_ODE_system

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   LV = create_ODE_system(
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
equations directly as strings and let it parse them with SymPy. Write each
derivative as ``d<name> = ...``; cubie reads ``x`` as a state because it
you wrote a ``dx`` equation for it:

.. code-block:: python
   :caption: Creating a system of ODEs from equation strings.

   from cubie import create_ODE_system

   LV = create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )
   print(LV)

In strings, you can also write implicit equations, which you can't express
in a Python function. Put an expression on the left-hand side, such as ``C*dx = ...``,
or write an algebraic constraint as ``0 = ...``. Solve these systems with
an implicit algorithm.

States, parameters, and observables
-----------------------------------

If you don't provide ``parameters`` or ``states``, cubie will try its best
to identify them. In equation strings, it takes any variable with a
``d<name>`` equation as a state, and any other right-hand-side variable as
a parameter. In a Python function, it takes the states from the keys of a
returned dict, or names them ``y0``, ``y1`` and so on when you write them
as ``y[0]``, ``y[1]``. If you use a parameter in your function, such as
``p.a``, without listing it in ``parameters``, cubie adds it as a
parameter. If you don't provide default values, either by leaving a
variable out or by listing names instead of a dict, the defaults will be
set to 0.

You may want to record intermediate values that aren't states. To do this,
you can declare them ``observables``:

.. code-block:: python
   :caption: An observable computed on the way to the derivatives.

   from cubie import create_ODE_system

   def lotka_volterra(t, y, p):
       predator_death_rate = p.c * y.y
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -predator_death_rate + p.d * y.x * y.y
       return [dx, dy]

   LV = create_ODE_system(
       lotka_volterra,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       observables=["predator_death_rate"],
       name="LotkaVolterra",
   )
   print(LV)

cubie still uses an intermediate value that you don't list in
``observables``, but it doesn't save that value's trajectory. You can do
the same in equation strings: write ``predator_death_rate = c*y`` as its own
line and list it in ``observables``.
