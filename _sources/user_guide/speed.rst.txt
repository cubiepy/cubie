Making it Faster (basic)
========================

TL/DR
-----
To get the best performance from Cubie, try to:

- Solve many problems at once (thousands if possible).
- Reduce the number of variables and samples you save or summarise.
- Set all parameters that you're not changing between solves to be `constants`.
- Reuse existing Solvers.

Parallelism
-----------

If we compare a like-for-like implementation of an IVP integration using Cubie
vs using an optimised CPU-utilising library like SciPy, we find that Cubie is
some linear factor slower. This is expected - GPU hardware isn't optimised
to process single tasks quickly. Instead, it's optimised to process many
tasks in parallel. Except for a penalty transferring data in and out of the
integration functions, increasing the number of problems being solved at
once by a factor \(n\) has little effect on the total time taken - you get \
(n-1\) problems solved _almost_ for free. The single best way to get a
performance gain from using Cubie is to solve more problems at once.

Memory
------
The big bottleneck in GPU computing is memory traffic. When you're completing
32,000 integrations at the same time, they all want to save a sample of
their state at the same time, and this puts a lot of pressure on the tubes
between the GPU and its memory. The more data you save, the slower it goes.
Cubie has three main levers you can pull to reduce memory traffic and speed
up your solves:

1. Reduce the number of variables you save. If you're solving a 10D system
   but only care about one variable, only save that one variable.
2. Reduce the number of samples you save. If you're solving a system for
   1000 time units but only care about the state at the end, only save the
   final state.
3. Use summary metrics. If you want to know the mean and standard
   deviation of a variable over the course of the solve, rather than save
   the whole history and process each dataset offline (slow!), use Cubie's
   built in summary metrics to calculate these on the GPU during the solve. You
   don't even need to save the state history at all!

Constants
---------
When you tell Cubie about your problem, you provide some symbols/variables
that are input-only - they don't change during the solve. If you're
brute-forcing a parameter study, you will want to be able to start an IVP from
a bunch of different values for some of these parameters. However, you may
have more parameters that you're not interested in changing between solves.
If you mark these as `constants` when defining your system of ODEs, Cubie
puts them in a different place in memory - rather than taking up space in
the scarce fast memory that needs to be able to change often, they go into
the compiled program itself. This means they require no memory traffic, and
they free up more space to run more runs at once!

Profiling with TimeLogger
-------------------------

CuBIE includes a built-in profiler that logs the time spent in each
phase of a solve.  Enable it by passing ``time_logging_level`` to
:func:`~cubie.solve_ivp` or the :class:`~cubie.Solver`:

.. code-block:: python

   result = qb.solve_ivp(
       system, y0=y0, parameters=params,
       method="dormand-prince-54", duration=10.0,
       time_logging_level="default",
   )

Levels: ``"default"`` (high-level phases), ``"verbose"`` (sub-phases),
``"debug"`` (everything).  After the solve, the logger prints a summary
table to stdout.

You can also access the logger programmatically:

.. code-block:: python

   from cubie.time_logger import default_timelogger
   default_timelogger.print_summary()

Precision Trade-offs
--------------------

CuBIE supports ``float32`` and ``float64``.  Single precision uses half
the memory per value, allowing larger batches and faster memory transfers.
However, it provides roughly 7 decimal digits of accuracy versus 15 for
double precision.  For problems where tolerances tighter than ~1e-6 are
needed, use ``float64``.

Buffer Location Tuning
----------------------

Working arrays can be placed in shared memory (fast, limited) or local
memory (slower, larger).  CuBIE assigns locations automatically, but you
can override them through ``optional_arguments`` for specific buffers.
See :doc:`optional_arguments` for details.

Reusing Solvers
---------------
Cubie uses Numba to just-in-time compile your system of ODEs and chosen
integration algorithm into a CUDA kernel. This compilation step takes time
(up to 30s!), so if you're only solving a few problems at once, it can
dominate the total time taken. The compiled kernel is cached, and as long as
you're running the same system with the same algorithm and saving at the
same cadence (more or less, there's some other variables you might change
that could force a recompile), Cubie will reuse the existing kernel and skip
compilation for the next run. For this reason, if you're going to do
multiple batches with the same system, instead of using the :func:`cubie.solve_ivp`
function, create a :class:`cubie.Solver` object and call its :meth:`solve`
method multiple times. Keeping a reference to the :class:`cubie.Solver`
object means that subsequent calls to :meth:`solve` will be much faster.