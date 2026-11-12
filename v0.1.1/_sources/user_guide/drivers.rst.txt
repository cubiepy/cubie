Drivers (Time-dependent functions)
==================================

In some models, rates of change can be functions of time. This makes the ODE system "non-autonomous" - an "autonomous"
system is one where time isn't explicitly included in the equations. The trick that mathematicians use to handle this is
to add an extra state variable that changes at a constant rate - effectively, a clock. In Cubie, we skip the middle-state
and just allow you to use the symbol `t` in your equations, and it will be automatically evaluated with time as the
integration proceeds. If you can neatly express your variable as a function of time, this is the cleanest way to get a
forcing term into your problem. As an example, an experiment I was working on recently involved us shaking a tiny MEMS
cantilever using a piezoelectric actuator under its base, measuring how much the cantilever bent as a result, then feeding
that measured signal back into a heater on the cantilever that caused it to bend. That system could be modelled in Cubie
like this:

.. code-block::
    :linenos:

    import cubie as qb
    import numpy as np

    # Parameters
    k = 0.1      # cantilever spring constant
    c = 0.01     # cantilever damping coefficient (largely due to air)
    alpha = 0.5  # heater coupling coefficient (how much bend the heat caused)
    beta = 0.1   # heater dissipation coefficient (how quickly the heat dissipates)


    def cantilever(t, y, p):
        f = 1e4 * t + 1e5
        base_wiggle = np.sin(2 * p.pi * f * t)
        dx = y.v
        dv = -p.k * y.x - p.c * y.v + p.alpha * y.T + base_wiggle
        dT = (p.alpha * (p.feedback_strength * y.x + p.feedback_offset)
              - p.beta * y.T)
        di = p.feedback_strength * y.x + p.feedback_offset
        return {"x": dx, "v": dv, "T": dT, "i": di}

    constants = {
        "k": 0.1,
        "c": 0.01,
        "alpha": 0.5,
        "beta": 0.1,
        "pi": np.pi,
    }
    initial_conditions = {
        "x": 0,    # initial position
        "v": 0,    # initial velocity
        "T": 0,    # initial temperature
        "i": 0,    # initial current
    }
    parameters = {
        "feedback_strength": 0.5,
        "feedback_offset": 0.1,
    }

    sys = qb.create_ODE_system(
        cantilever,
        parameters=parameters,
        constants=constants,
        states=initial_conditions,
        name="MEMSCantilever",
    )

    result = qb.solve_ivp(
        sys,
        y0=initial_conditions,
        parameters={
            "feedback_strength": np.linspace(-1, 1, 200),
            "feedback_offset": np.linspace(-1, 1, 200),
        },
        method="crank_nicolson",
        duration=1e-3,
    )

This is an easy way to set up and parametrize a forcing term, and see how your system responds to different versions of
it.

Arbitrary Values
----------------
Sometimes, your forcing function might not be easily expressed as a function of time. You might have some measured data
that you want to test, you might want to use a random signal, or you might just want to throw some numbers in and see what
comes out. If your integrator is fixed-step and you can supply the value of the forcing term at each time step, this is
straightforward in theory - just pick a value for each time step, and use that. It's a bit limiting to only use fixed-step
algorithms, and if you're running a long-duration simulation at a high time resolution, you might need to store a lot of
time points. To give a bit more flexibility, Cubie can accept an array of forcing values and the time points to which they
correspond, and will interpolate between them to get the value at any time point. This allows you to use adaptive-step
algorithms, and also means you don't need to store the value at every time point if your forcing function is smooth. Here's
an example of how to do this:

.. code-block:: python

    import cubie as qb
    import numpy as np

    # Create a measured signal as a driver
    t_driver = np.linspace(0, 1.0, 1000)
    signal = np.sin(2 * np.pi * 5 * t_driver) * np.exp(-t_driver)

    def driven(t, y, p):
        dx = -p.k * y.x + p.amplitude * drive_signal
        return [dx]

    sys = qb.create_ODE_system(
        driven,
        constants={"k": 1.0},
        parameters={"amplitude": 1.0},
        states={"x": 0.0},
        drivers=["drive_signal"],
        name="DrivenSystem",
    )

    result = qb.solve_ivp(
        sys,
        y0={"x": np.array([0.0])},
        parameters={"amplitude": np.linspace(0.1, 2.0, 100)},
        drivers={"drive_signal": signal, "time": t_driver},
        method="dormand-prince-54",
        duration=1.0,
    )

Note how the driver appears: it is declared in ``drivers`` when the
system is created, and referenced by its bare name inside the function
body, since drivers are not part of the state or parameter containers.

The ``drivers`` dictionary maps each driver name to a 1-D array of
sampled values.  The sample times are supplied alongside them, either
as a ``"time"`` array of the same length, or as scalar ``"dt"`` (and
optionally ``"t0"``) keys when the samples are evenly spaced.

Interpolation Options
---------------------

The remaining keys of the ``drivers`` dictionary control how CuBIE
interpolates between your samples:

**order** (default ``3``)
   Polynomial degree of the spline fitted over each sample segment.
   The default cubic is smooth enough for most signals.

**wrap** (default ``True``)
   What happens outside the sampled time range.  With ``wrap=True``
   the signal repeats periodically — useful for periodic forcing.
   With ``wrap=False`` the signal is zero outside the sampled range;
   CuBIE adds a smooth spline segment from the last data point down to
   zero (and up from zero to the first point) so there is no
   discontinuous jump.

**boundary_condition**
   End-point condition for the spline fit: one of ``"natural"``,
   ``"periodic"``, ``"clamped"``, or ``"not-a-knot"`` (the scipy
   ``CubicSpline`` conventions).  If you don't set it, CuBIE picks
   ``"periodic"`` when ``wrap=True`` and ``"clamped"`` when
   ``wrap=False``.

.. code-block:: python

    drivers={
        "drive_signal": signal,
        "time": t_driver,
        "order": 3,
        "wrap": True,
    }

A worked example, including a sanity-check plot of the interpolated
driver, lives in ``docs/source/examples/array_interpolation_example.py``.