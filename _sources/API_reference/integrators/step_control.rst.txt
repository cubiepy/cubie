Step control
============

``cubie.integrators.step_control``
----------------------------------

.. currentmodule:: cubie.integrators.step_control

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   step_control/get_controller
   step_control/base_step_controller
   step_control/base_adaptive_step_controller
   step_control/fixed_step_controller
   step_control/adaptive_i_controller
   step_control/adaptive_pi_controller
   step_control/adaptive_pid_controller
   step_control/gustafsson_controller
   step_control/base_step_controller_config
   step_control/adaptive_step_control_config
   step_control/fixed_step_control_config
   step_control/pi_step_control_config
   step_control/pid_step_control_config
   step_control/gustafsson_step_control_config

The step control package encapsulates the configuration and compilation
machinery used to build CUDA device functions that manage integrator time-step
selection. Fixed, integral, proportional–integral, proportional–integral–
derivative, and Gustafsson-style predictive controllers share a common
configuration interface that feeds the :class:`cubie.CUDAFactory` compilation
pipeline.

Controller interfaces
---------------------

* :doc:`BaseStepController <step_control/base_step_controller>` – abstract base for all controllers.
* :doc:`BaseAdaptiveStepController <step_control/base_adaptive_step_controller>` – adds adaptive gain handling.
* :doc:`FixedStepController <step_control/fixed_step_controller>` – returns constant time steps.
* :doc:`AdaptiveIController <step_control/adaptive_i_controller>` – integral-only adaptive controller.
* :doc:`AdaptivePIController <step_control/adaptive_pi_controller>` – proportional–integral controller.
* :doc:`AdaptivePIDController <step_control/adaptive_pid_controller>` – proportional–integral–derivative controller.
* :doc:`GustafssonController <step_control/gustafsson_controller>` – Gustafsson PI controller variant.
* :doc:`get_controller <step_control/get_controller>` – resolves controller implementations from settings.

Configuration objects
---------------------

* :doc:`BaseStepControllerConfig <step_control/base_step_controller_config>` – base attrs configuration shared by all
  controllers.
* :doc:`AdaptiveStepControlConfig <step_control/adaptive_step_control_config>` – configuration used by adaptive
  controllers.
* :doc:`FixedStepControlConfig <step_control/fixed_step_control_config>` – fixed-step configuration container.
* :doc:`PIStepControlConfig <step_control/pi_step_control_config>` – proportional–integral gain configuration.
* :doc:`PIDStepControlConfig <step_control/pid_step_control_config>` – proportional–integral–derivative configuration.
* :doc:`GustafssonStepControlConfig <step_control/gustafsson_step_control_config>` – Gustafsson controller configuration.

Dependencies
------------

* Relies on :mod:`cubie.CUDAFactory` for compile-time caching of CUDA device
  functions.
* Requires Numba CUDA support during runtime to JIT the device controllers.
* Pulls validators and clamp helpers from :mod:`cubie._utils` when building
  device functions.

Suggested controller parameters
-------------------------------

The default proportional, integral, and derivative gains mirror the
recommendations from Söderlind and Wang while matching the guidance in
`OrdinaryDiffEq.jl <https://github.com/SciML/OrdinaryDiffEq.jl>`_. Common
choices include:

.. list-table::
   :header-rows: 1

   * - Controller
     - ``beta1``
     - ``beta2``
     - ``beta3``
   * - basic
     - 1.00
     - 0.00
     - 0
   * - PI42
     - 0.60
     - -0.20
     - 0
   * - PI33
     - 2/3
     - -1/3
     - 0
   * - PI34
     - 0.70
     - -0.40
     - 0
   * - H211PI
     - 1/6
     - 1/6
     - 0
   * - H312PID
     - 1/18
     - 1/9
     - 1/18
