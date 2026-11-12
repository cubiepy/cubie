CUDA and GPU Computing
======================

CuBIE runs on NVIDIA GPUs via the CUDA programming model.  This page
explains why GPUs are a natural fit for batch ODE solving and where the
performance bottlenecks lie.

What is a GPU?
--------------

A Graphics Processing Unit (GPU) contains thousands of small processors
that can execute the same instruction in parallel.  A modern NVIDIA GPU
has tens of thousands of *CUDA cores* grouped into *Streaming
Multiprocessors* (SMs), each with a pool of fast on-chip *shared memory*
(typically 48--100 KB) and access to several gigabytes of slower
*global memory* (VRAM).

What is CUDA?
-------------

CUDA is NVIDIA's framework for running general-purpose code on their GPUs.
A program (called a *kernel*) is launched across a grid of *threads*.
Threads are grouped into *blocks*; threads within a block share the SM's
shared memory and can synchronize with each other.

CuBIE uses `Numba <https://numba.pydata.org/>`__ to JIT-compile Python
functions into CUDA kernels, so you never need to write CUDA or C++ code
directly.

Why Batch ODE Solving Maps to CUDA
------------------------------------

Each initial value problem in a batch is independent: it has its own
states, parameters, and time-stepping loop.  CuBIE assigns one CUDA
thread per IVP.  Because there is no data dependency between threads, the
GPU can run thousands of IVPs simultaneously with near-perfect parallel
efficiency.

A single IVP runs roughly 10x slower on a GPU thread than on a modern CPU
core.  However, adding another thread is essentially free, so as few as
a few dozen IVPs already break even.  At the scale CuBIE targets
(thousands to hundreds of thousands of IVPs), the GPU is orders of
magnitude faster than serial CPU execution.

Memory Hierarchy and Bottlenecks
---------------------------------

The dominant bottleneck in GPU ODE solving is *memory bandwidth*.  When
32,000 threads all try to write a state sample at the same time, the bus
between the SMs and VRAM becomes saturated.

CuBIE mitigates this in several ways:

**Constants vs Parameters.**
   Values that do not change between IVPs in a batch are declared as
   *constants*.  Constants are embedded in the compiled kernel and occupy
   no per-thread memory.

**Matrix-free Jacobians.**
   Storing an :math:`n \times n` Jacobian per thread would consume huge
   amounts of shared memory.  CuBIE uses symbolic code generation to
   produce a JVP function instead, trading memory for compute (see
   :doc:`jacobians`).

**Selective saving.**
   Only the state variables and observables you request are written back
   to global memory.  Everything else stays in registers and shared
   memory.

**On-GPU summaries.**
   Summary metrics (mean, max, peaks, etc.) are accumulated on-chip
   during the integration.  The full time-domain trajectory need never be
   written to VRAM.

Shared Memory Constraints
-------------------------

Each SM has a limited pool of shared memory (typically 48 KB by default).
CuBIE packs per-thread working arrays into this pool.  The more memory
each thread requires, the fewer threads can run concurrently on the same
SM, reducing *occupancy*.

You can inspect buffer allocations through the
:doc:`/developer_guide/buffer_registry` and tune buffer locations
(shared vs local) through ``optional_arguments``.  CuBIE also reports
occupancy and memory usage through the
:class:`~cubie.time_logger.TimeLogger`.
