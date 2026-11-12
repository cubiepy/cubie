Coming from SciPy
=================

If you have a right-hand-side function written for
``scipy.integrate.solve_ivp``, cubie can usually run it unchanged. The main
differences are in how you pass the parameters and the time span, and in
what comes back.

A direct port
-------------

You can use the same function in both libraries. SciPy passes the values in
``args=`` to your function's extra arguments in order; cubie matches each
extra argument to the entry with the same name in ``parameters=``.

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
       t_eval=np.linspace(0.0, 20.0, 201),
   )
   t, y = sol.t, sol.y

.. code-block:: python
   :caption: cubie, with the same function

   from cubie import solve_ivp

   def van_der_pol(t, y, mu):
       return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

   result = solve_ivp(
       van_der_pol,
       y0={"y1": 1.0, "y2": 0.0},
       parameters={"mu": 1.5},
       method="ode45",
       duration=20.0,
       save_every=0.1,
       output_types=["state", "time"],
   )
   t, y = result.time[:, 0], result.time_domain_array[:, :, 0].T

Key changes:

- Parameters by name: give the parameter values in a dict in
  ``parameters=``, keyed by your function's argument names, instead of in
  ``args=``.
- Named starting values: give ``y0`` as a dict with an entry for each
  state. cubie sorts the state columns by name, so name the states ``y1``
  and ``y2`` to keep SciPy's order.
- Time: instead of ``t_span``, give the total integration time as
  ``duration``, and a start time as ``t0`` if it isn't 0. Instead of
  ``t_eval``, give the time between saved points as ``save_every``.
  Without ``save_every``, cubie saves only the first and last points.
- Tolerances: cubie's default ``rtol`` is ``1e-6``, where SciPy's is
  ``1e-3``, so you get different answers from the two unless you pass the
  same ``rtol`` to both.
- cubie uses different method names; see the table below.
- cubie returns the results of every run in the batch in one array, inside
  a ``SolveResult``; see :doc:`/user_guide/results`. Add ``"time"`` to
  ``output_types`` to get the time points.

What your function can contain
------------------------------

SciPy calls your function, so you can use anything Python allows in it.
cubie reads your function and turns it into GPU code instead, so you can
only use arithmetic, maths functions such as ``np.sin`` or ``math.exp``,
``if``/``else``, and ``for`` loops over a fixed ``range(...)``. You can't
use:

- ``while`` loops or list comprehensions.
- NumPy array functions such as ``np.array``. Return the derivatives as a
  list.
- Variables defined outside your function. Add them as arguments and give
  their values in ``parameters=``.
- Anything that isn't maths, such as printing or reading files.

Method names
------------

===================== ==============================================
SciPy ``method=``     cubie ``method=``
===================== ==============================================
``RK45``              ``"ode45"`` (also ``"rk45"``, ``"dopri54"``)
``RK23``              ``"ode23"`` (also ``"rk23"``)
``DOP853``            ``"dop853"``
``Radau``             ``"radau"``
``BDF``, ``LSODA``    no direct equivalent; pick a stiff method from
                      :doc:`/user_guide/choosing_algorithms`
===================== ==============================================

See :doc:`/user_guide/choosing_algorithms` for every method cubie offers.

Replacing a loop of solves
--------------------------

Cubie directly replaces a loop of ``solve_ivp`` calls if you're already
solving a batch in SciPy:

.. code-block:: python
   :caption: SciPy

   mus = np.linspace(0.5, 4.0, 10000)
   solutions = [
       solve_ivp(van_der_pol, (0.0, 20.0), [1.0, 0.0], args=(mu,))
       for mu in mus
   ]

In cubie, you pass all of the input values to one solve call instead:

.. code-block:: python
   :caption: cubie

   result = solve_ivp(
       van_der_pol,
       y0={"y1": 1.0, "y2": 0.0},
       parameters={"mu": np.linspace(0.5, 4.0, 10000)},
       method="ode45",
       duration=20.0,
       save_every=0.1,
   )

If you supply arrays for several parameters or initial values, cubie runs
every combination of the supplied values by default. See
:doc:`/user_guide/batching` for the other options.

Not supported
-------------

* cubie has no equivalent of ``dense_output`` or event functions.
* cubie saves solutions at a regular interval set by ``save_every``, not
  at arbitrary ``t_eval`` points.
