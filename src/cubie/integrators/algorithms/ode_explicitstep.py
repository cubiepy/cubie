"""Infrastructure for explicit integration step implementations.

Published Classes
-----------------
:class:`ExplicitStepConfig`
    Configuration container for explicit ODE integration algorithms.
    Inherits all fields from :class:`BaseStepConfig` without additions.

:class:`ODEExplicitStep`
    Abstract base for explicit algorithms. Provides :meth:`build` which
    unpacks configuration and delegates to :meth:`build_step`.

See Also
--------
:class:`~cubie.integrators.algorithms.base_algorithm_step.BaseAlgorithmStep`
    Parent factory class.
:class:`~cubie.integrators.algorithms.base_algorithm_step.BaseStepConfig`
    Parent configuration class.
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Implicit counterpart.
"""

from abc import abstractmethod
from typing import Callable, Optional

from attrs import frozen

from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)


@frozen
class ExplicitStepConfig(BaseStepConfig):
    """Configuration settings for explicit ODE integration algorithms."""
    pass


class ODEExplicitStep(BaseAlgorithmStep):
    """Base helper for explicit integration algorithms."""

    def build(self) -> StepCache:
        """Create and cache the device function for the explicit algorithm.

        Returns
        -------
        StepCache
            Container with the compiled step device function.
        """

        config = self.compile_settings
        dxdt_fn = config.dxdt_fn
        numba_precision = config.numba_precision
        n = config.n
        observables_fn = config.observables_fn
        drivers_fn = config.drivers_fn
        n_drivers = config.n_drivers
        return self.build_step(
            dxdt_fn,
            observables_fn,
            drivers_fn,
            numba_precision,
            n,
            n_drivers,
        )

    @abstractmethod
    def build_step(
        self,
        dxdt_fn: Callable,
        observables_fn: Callable,
        drivers_fn: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:
        """Build and return the explicit step device function.

        Parameters
        ----------
        dxdt_fn
            Device function for evaluating the ODE right-hand side f(t, y).
        observables_fn
            Device helper that computes observables for the system.
        drivers_fn
            Optional device function evaluating drivers at arbitrary times.
        numba_precision
            Numba precision for compiled device buffers.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the device step implementation.
        """
        raise NotImplementedError

    @property
    def is_implicit(self) -> bool:
        """Return ``False`` to indicate the algorithm is explicit."""
        return False
