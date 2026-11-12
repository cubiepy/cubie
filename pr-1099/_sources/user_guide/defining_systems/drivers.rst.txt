Drivers (Time-dependent functions)
==================================

In some models, rates of change can be functions of time. This makes the ODE system "non-autonomous" - an "autonomous"
system is one where time isn't explicitly included in the equations. The trick that mathematicians use to handle this is
to add an extra state variable that changes at a constant rate - effectively, a clock. In Cubie, we skip the middle-state
and just allow you to use the symbol `t` in your equations, and it will be automatically evaluated with time as the
integration proceeds. If you can neatly express your variable as a function of time, this is the cleanest way to get a
forcing term into your problem. As an example, we'll follow a self-actuated atomic force microscopy cantilever being shaken
by a piezoelectric actuator under its base. In the real experiment, we measure how much the cantilever bends as a result,
and feed some function of that measured signal back into the cantilever's actuator, a heater which causes it to bend.
That system could be modelled in Cubie like this:

.. code-block:: python

    import numpy as np
    from cubie import solve_ivp

#Define this in string form, it will be clearer.
    def cantilever(t, y, p):
        reference = min(max(p.rhat * np.cos(p.omega_forcing * t), -p.clip), p.clip)
        control = min(max(p.a * y.h + p.b, -p.clip), p.clip)
        dx = y.v
        dv = -y.x - p.delta * y.v + p.alpha * y.T + p.pz_sign * reference
        dT = -p.beta * y.T + p.gamma * y.u * y.u
        du = p.omega_l * (control - y.u)
        dh = -p.omega_h * y.h + p.hpf_sign * y.v
        return [dx, dv, dT, du, dh]

    result = solve_ivp(
        cantilever,
        y0={"x": 0.0, "v": 0.0, "T": 0.0, "u": 0.0, "h": 0.0},
        parameters={
            "alpha": 0.52,
            "beta": 0.0133,
            "gamma": 0.0624,
            "delta": 0.012,
            "omega_h": 1 / 114,
            "omega_l": 2.0,
            "pz_sign": 1.0,
            "hpf_sign": -1.0,
            "rhat": 0.01,
            "clip": 1.0,
            "omega_forcing": 1.0,
            "a": np.linspace(0.0, 4.0, 200),
            "b": np.linspace(-1.0, 1.0, 200),
        },
        method="ode45",
        duration=500.0,
    )

Here ``x`` and ``v`` are the cantilever's deflection and velocity, ``T`` is
the heater's temperature, ``u`` is the filtered control signal sent to the
heater, and ``h`` is a high-pass-filtered copy of the deflection that
the controller feeds back. The piezo reference and the control signal are
both clipped to ``±clip``, like the hardware. We sweep the
controller's gain ``a`` and offset ``b`` over 200 values each, so cubie
solves all 40,000 combinations.

``t`` appears in the system the same way you'd write it in math, and cubie does the 
rest of the work. This is an easy way to set up and parametrize a forcing term, and see how your system responds to different versions of
it.

Measured/Arbitrary Values
-------------------------

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

    import numpy as np
    from cubie import DriverSamples, solve_ivp

    t_samples = np.linspace(0.0, 1.0, 1000)
    signal = np.sin(2 * np.pi * 5 * t_samples) * np.exp(-t_samples)
    drive = DriverSamples({"drive_signal": signal}, time=t_samples)

    def driven(t, y, p, d):
        return [-p.k * y.x + p.amplitude * d.drive_signal]

    result = solve_ivp(
        driven,
        y0={"x": 0.0},
        parameters={"k": 1.0, "amplitude": np.linspace(0.1, 2.0, 10000)},
        drivers=drive,
        method="ode45",
        duration=1.0,
    )

Read your driver by name from the argument after the parameters, here
``d``. Keep the parameter argument ``p`` in your function even if your
system has no parameters. Then pass your samples as ``drivers=`` when you
solve. If you build the system yourself with ``create_ODE_system``, list
the driver names there as well, with ``drivers=["drive_signal"]``.

Sampled forcing data is defined using a ``DriverSamples`` object.
It takes a ``dict`` of ``"[driver name]": [sampled data]`` pairs, with
either the sample time vector given as ``time``, or a starting time
and sample spacing given as ``t0`` and ``driver_sample_period``.

Interpolation Options
---------------------

Three further solver settings control how CuBIE interpolates between
your samples:

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

Pass these settings to ``solve_ivp`` along with ``drivers=``. In this solve
we reuse ``driven`` and ``drive`` from the example above, and run past the
end of the samples with wrapping turned off:

.. code-block:: python

    result = solve_ivp(
        driven,
        y0={"x": 0.0},
        parameters={"k": 1.0, "amplitude": np.linspace(0.1, 2.0, 10000)},
        drivers=drive,
        order=3,
        wrap=False,
        method="ode45",
        duration=2.0,
    )
