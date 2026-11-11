Troubleshooting
===============

CUDA Not Found
--------------

If cubie cannot find a GPU, it raises an error such as
``NoCudaDeviceError`` the first time it builds or runs a solver.

1. Check that your computer has an NVIDIA GPU and driver by running
   ``nvidia-smi`` in a terminal. It lists your GPU and the newest CUDA
   version your driver supports. If the command is missing, install the
   NVIDIA driver.
2. Then either install a CUDA toolkit yourself, or reinstall cubie with a
   toolkit included: ``pip install cubie[cuda13]``, or
   ``pip install cubie[cuda12]`` if ``nvidia-smi`` reports a CUDA version
   below 13.

Runs Fail on Stiff Problems
---------------------------

When a run fails, cubie fills its output with NaN and records why in
``result.status_messages``, a dict that maps each failed run's index to
flag names such as ``MAX_NEWTON_ITERATIONS_EXCEEDED``,
``NEWTON_DIVERGENCE`` or ``STEP_TOO_SMALL``. If many runs fail:

1. **Use an algorithm built for stiff problems.** ``method="radau"``
   handles very stiff systems; :doc:`/user_guide/choosing_algorithms`
   lists the cheaper options.
2. **Loosen the tolerances.** Tolerances tighter than your problem needs
   force tiny steps and make each implicit solve harder to converge.
3. **Use double precision for tight tolerances.** Single precision holds
   about seven significant digits, so it cannot meet an ``rtol`` or
   ``atol`` much below ``1e-6``. Build the system with
   ``precision=np.float64``, which ``create_ODE_system`` accepts, and so
   does ``solve_ivp`` when you pass it the equations directly.
