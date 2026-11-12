Coming from SciPy
=================

If you have a right-hand-side function written for
``scipy.integrate.solve_ivp``, cubie can usually run it unchanged. 

A direct port
-------------

The same function object works in both libraries. In SciPy you pass
parameter values through ``args=``; in cubie you declare them by name so
they can be swept in batches later.
# Is this by-name thing true? I think they now must be part of a params object, which makes them a closer match to args.

.. code-block:: python
   :caption: SciPy

   import numpy as np
   from scipy.integrate import solve_ivp

   def van_der_pol(t, y, mu):
       return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

   sol = solve_ivp(
       van_der_pol,
       (0.0, 20.0),
       [1.0, 0.0],
       args=(1.5,),
       method="RK45",
   )
   # sol.t, sol.y

.. code-block:: python
   :caption: cubie — the same function, unchanged

   import cubie as qb

   def van_der_pol(t, y, mu):
       return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

   result = qb.solve_ivp(
       van_der_pol,
       y0={"x": [1.0], "v": [0.0]},
       parameters={"mu": [1.5]},
       duration=20.0,
       method="ode45",
       save_every=0.1,
   )
   trajectories = result.as_numpy["time_domain_array"]

# Change this to match my voice and content from the matlab one, this is weird robot language that is written to a coder, not actual meaning communicated to an entry-level scipy user.
Line by line, the differences are:

* **Parameters are named.** SciPy's ``args=(1.5,)`` becomes
  ``parameters={"mu": 1.5}``. The name must match the argument name in
  the function signature. Extra arguments after ``(t, y)`` are matched
  to the parameters or drivers with the same name.
* **States are named.** The ``y0`` dict gives each state a label and an
  initial value; dict order matches the order of the returned
  derivatives. If you pass a plain array instead, cubie synthesises
  names (``y0``, ``y1``, ...) from positional access.
* **Time is a duration.** ``t_span=(0.0, 20.0)`` becomes
  ``duration=20.0`` (with an optional ``t0``). ``t_eval`` has no
  direct equivalent; ``save_every`` sets a regular output interval.
* **Method names differ.** See the table below.
* **Results come back as arrays per batch run**, not a single
  trajectory — see :doc:`/user_guide/results`.

# Who gives a solitary fuck about the below? We can list the things they can't use here, if it helps. can scipy run the things we exclude?
cubie never calls your function. It reads the source, converts the
mathematics to symbolic form, and compiles a CUDA kernel from it. This
means the body must be expressible mathematics — arithmetic, ``math``
or ``numpy`` scalar calls, ``if``/``else``, and constant-bound ``for``
loops — rather than arbitrary Python (no ``while``, comprehensions, or
I/O). Because your function is still an ordinary Python function, you
can keep testing it directly with SciPy or plain NumPy on the CPU.

Method names
------------

===================== ==============================================
SciPy ``method=``     cubie ``method=``
===================== ==============================================
``RK45``              ``"ode45"`` (also ``"rk45"``, ``"dopri54"``)
``RK23``              ``"ode23"`` (also ``"rk23"``)
``DOP853``            ``"dop853"``
``Radau``             ``"firk"`` (fully implicit Runge-Kutta)
``BDF``, ``LSODA``    no direct equivalent; for stiff systems use
                      ``"rosenbrock"``, ``"backwards_euler"``, or
                      ``"crank_nicolson"``
===================== ==============================================

#Bullshit robot wording about rtol and atol. How is that relevant here at all? 
``rtol`` and ``atol`` keep their names. See
:doc:`/user_guide/choosing_algorithms` for the full list and guidance.

Where cubie pays off: batches
-----------------------------
# Include the closer match to the parfor section. 
In SciPy, a parameter sweep is a Python loop around ``solve_ivp``. In
cubie, arrays of values describe the whole batch and one call
integrates every run in parallel on the GPU:

.. code-block:: python

   result = qb.solve_ivp(
       van_der_pol,
       y0={"x": [1.0], "v": [0.0]},
       parameters={"mu": np.linspace(0.5, 4.0, 1024)},
       duration=20.0,
       method="ode45",
       save_every=0.1,
   )

By default every combination of the supplied values is run
(``grid_type="combinatorial"``); pass ``grid_type="verbatim"`` to pair
inputs run-for-run. See :doc:`/user_guide/batching`.

#Nothing to do with scipy.
Repeated solves: build the system once
--------------------------------------

``qb.solve_ivp(function, ...)`` builds and compiles the system on every
call. If you solve the same system repeatedly — new parameters or
initial values each time — build it once with
:func:`~cubie.odesystems.symbolic.symbolicODE.create_ODE_system` and
keep a :class:`~cubie.batchsolving.solver.Solver`, so later calls skip
system construction and reuse the compiled kernel:

.. code-block:: python

   system = qb.create_ODE_system(
       van_der_pol,
       states={"x": 1.0, "v": 0.0},
       parameters={"mu": 1.5},
   )
   solver = qb.Solver(system, algorithm="ode45", save_every=0.1)

   for mu_batch in mu_batches:
       result = solver.solve(
           {"x": [1.0], "v": [0.0]},
           {"mu": mu_batch},
           duration=20.0,
       )

Note that ``Solver.solve`` defaults to ``grid_type="verbatim"`` where
``qb.solve_ivp`` defaults to ``"combinatorial"``.

Not supported
-------------

* ``dense_output`` and event functions have no cubie equivalent.
* Solutions are saved on the regular ``save_every`` grid, not at
  arbitrary ``t_eval`` points.
# How on earth would the scipy user know what a closure variable is or a global constant, why would they care what a "parse-time"error is.
* The right-hand side must be self-contained: closure variables and
  global constants raise a parse-time error naming the symbol —
  declare them in ``parameters`` instead.
