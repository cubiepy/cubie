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
   ``pip install cubie[cuda12]`` if you see a CUDA version below 13 in the
   ``nvidia-smi`` output.

Failed Runs
-----------

When a run fails, cubie stops integrating it, fills its output with NaN,
and records why in ``result.status_messages``. Look up the failed run's
index in that dict to see the status codes cubie set for it; cubie can
set more than one for a run. In an adaptive solve, cubie retries a failed
implicit solve with a smaller step, so you will usually see the Newton and
linear-solver codes together with ``STEP_TOO_SMALL``. In a fixed-step
solve, cubie ends the run at the first failed step.

``STEP_TOO_SMALL``
~~~~~~~~~~~~~~~~~~

cubie needed a step smaller than ``dt_min`` to meet your tolerances.

- Lower ``dt_min`` if it is not much smaller than the fastest change in your system; see
  :doc:`/user_guide/timing`.
- Choose a method that suits how stiff your problem is; see
  :doc:`/user_guide/choosing_algorithms`.
- Loosen ``rtol`` and ``atol``; see :doc:`/user_guide/optional_arguments`.
- Use double precision for tolerances much below ``1e-6``. In single
  precision, cubie works with about seven significant digits, so it cannot
  meet tighter tolerances. Pass ``precision=np.float64`` to
  ``create_ODE_system``, or to ``solve_ivp`` when you give it the equations
  directly.

``STAGNATION``
~~~~~~~~~~~~~~

cubie took steps too small to move time forward. Check the same things as for
``STEP_TOO_SMALL``.

``MAX_NEWTON_ITERATIONS_EXCEEDED``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cubie could not converge the Newton iteration of an implicit method within
``newton_max_iters`` iterations.

- Choose a method that suits how stiff your problem is; see
  :doc:`/user_guide/choosing_algorithms`.
- In a fixed-step solve, lower ``dt``; see :doc:`/user_guide/timing`.
- Raise ``newton_max_iters``; see :doc:`/user_guide/optional_arguments`.

``NEWTON_DIVERGENCE``
~~~~~~~~~~~~~~~~~~~~~

cubie moved further from the solution with each Newton update of an
implicit method, instead of closer.

- Choose a method that suits how stiff your problem is; see
  :doc:`/user_guide/choosing_algorithms`.
- In a fixed-step solve, lower ``dt``; see :doc:`/user_guide/timing`.
- Check that your equations and starting values are what you intended.

``MAX_LINEAR_ITERATIONS_EXCEEDED``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cubie could not converge an iterative linear solve within
``krylov_max_iters`` iterations. See :doc:`/user_guide/optional_arguments` for each of these
settings.

- Use the direct solver, ``linear_correction_type="lu"``.
- Try a different ``preconditioner_type`` or ``preconditioner_order``.
- Raise ``krylov_max_iters``.

``BICGSTAB_BREAKDOWN``
~~~~~~~~~~~~~~~~~~~~~~

cubie stopped a BiCGSTAB linear solve before it converged, because of a
numerical breakdown in BiCGSTAB. Use a different
``linear_correction_type``, such as ``"lu"``; see
:doc:`/user_guide/optional_arguments`.

``DAE_INITIALISATION_FAILED``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cubie could not make the algebraic states of a DAE consistent with the
equations at the start of the run, so cubie ended the run at ``t0`` with
your starting values. See :doc:`/user_guide/optional_arguments`.

- Give starting values for the algebraic states that are closer to
  consistent.
- Set ``dae_initialisation="shampine"`` to use the other initialisation
  method, or ``"none"`` to skip it.
