Troubleshooting
===============

CUDA Not Found
--------------

If cubie cannot find a GPU, it raises an error such as
``NoCudaDeviceError`` the first time you build or run a solver.

1. Check that your computer has an NVIDIA GPU and driver by running
   ``nvidia-smi`` in a terminal. You should see your GPU and the newest
   CUDA version your driver supports in its output. If the command is
   missing, install the NVIDIA driver.
2. Then either install a CUDA toolkit yourself, or reinstall cubie with a
   toolkit included: ``pip install cubie[cuda13]``, or
   ``pip install cubie[cuda12]`` if ``nvidia-smi`` reports a CUDA version
   below 13.

#This section is still no good. This should be a list of possible error/status codes that cubie can return with a pointer to the appropriate page - each can have a list. we wouldn't list a default algorithm like this anywhere, we'd point to the choosing algorithm page.

Runs Fail on ff Problems
---------------------------

When a run fails, cubie fills its output with NaN and records why in
``result.status_messages``. Look up the failed run's index in that dict to
see flag names such as ``MAX_NEWTON_ITERATIONS_EXCEEDED``,
``NEWTON_DIVERGENCE`` or ``STEP_TOO_SMALL``. If many runs fail:

#get rid of the 
1. **Use an algorithm built for stiff problems.** Pass ``method="radau"``
   for very stiff systems, or pick a cheaper option from
   :doc:`/user_guide/choosing_algorithms`.
2. **Loosen the tolerances.** If you ask for tighter tolerances than your
   problem needs, cubie has to take tiny steps and struggles to converge
   each implicit solve.
3. **Use double precision for tight tolerances.** In single precision,
   cubie works with about seven significant digits, so it cannot meet an
   ``rtol`` or ``atol`` much below ``1e-6``. Pass
   ``precision=np.float64`` to ``create_ODE_system``, or to ``solve_ivp``
   when you give it the equations directly.
