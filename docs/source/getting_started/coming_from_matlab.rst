Coming from MATLAB
==================

You can run most of a MATLAB script in cubie once you translate it into
Python. The main difference is that cubie has a single solve function,
``solve_ivp``, and takes the integration method as an argument, instead of
one function per algorithm.

Porting a parfor sweep
----------------------

Cubie directly replaces a ``parfor`` or ``for`` loop of ODE solves. You
pass all of the input values to one solve call, and cubie solves every run
in parallel on the GPU.

.. code-block:: matlab
   :caption: MATLAB

   mus = linspace(0.5, 4, 10000);
   opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
   results = cell(numel(mus), 1);
   parfor i = 1:numel(mus)
       f = @(t, y) [y(2); mus(i)*(1 - y(1)^2)*y(2) - y(1)];
       [t, y] = ode45(f, 0:0.1:20, [1; 0], opts);
       results{i} = y;
   end

.. code-block:: python
   :caption: cubie

   import numpy as np
   from cubie import solve_ivp

   def f(t, y, mu):
       return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

   result = solve_ivp(
       f,
       y0={"y1": 1.0, "y2": 0.0},
       parameters={"mu": np.linspace(0.5, 4.0, 10000)},
       method="ode45",
       duration=20.0,
       save_every=0.1,
       rtol=1e-6,
       atol=1e-8,
       output_types=["state", "time"],
   )
   t = result.time    # read run i's time points as t[:, i]
   y = result.state   # read run i as y[:, :, i], like results{i}

Key changes:

- Python indexing: use square brackets and count from 0 instead of 1, so
  you write ``y[0]`` where MATLAB has ``y(1)``.
- Parameters as arguments: cubie can't capture variables from your script
  the way a MATLAB anonymous function does. Add each parameter as an
  argument of your function, and give its values in ``parameters=`` when
  you call ``solve_ivp``. cubie runs once for each value in the array.
- Return a list: return the derivatives as a Python list, ``[dy1, dy2]``,
  or a tuple, instead of a column vector. cubie does not accept a NumPy
  array here.
- Time: instead of a time span, give the total integration time as
  ``duration``, a start time as ``t0`` if it isn't 0, and the time between
  saved points as ``save_every``. Without ``save_every``, cubie saves only
  the first and last points.
- Pass the ``odeset`` options as keyword arguments of ``solve_ivp``: write
  ``rtol`` and ``atol`` for ``RelTol`` and ``AbsTol``, and ``dt_max`` for
  ``MaxStep``. See :doc:`/user_guide/optional_arguments`.
- cubie returns every run's results in one array, inside a
  ``SolveResult``. Index the array by time first, then state, then run; see
  :doc:`/user_guide/results`. Add ``"time"`` to ``output_types`` to get
  the time points. cubie sorts the
  state columns by name, so name the states ``y1`` and ``y2`` to keep
  MATLAB's order.
- If you supply arrays for several parameters or initial values, cubie runs
  every combination of the supplied values by default. See
  :doc:`/user_guide/batching` for the other options.

Method names
------------

===================== ==============================================
MATLAB                cubie ``method=``
===================== ==============================================
``ode45``             ``"ode45"``
``ode23``             ``"ode23"``
``ode23s``            ``"ode23s"``
``ode23t``            ``"ode23t"``
``ode89``             ``"dop853"`` (closest high-order explicit)
``ode15s``            no direct equivalent; pick a stiff method from
                      :doc:`/user_guide/choosing_algorithms`
===================== ==============================================

See :doc:`/user_guide/choosing_algorithms` for every method cubie offers.
