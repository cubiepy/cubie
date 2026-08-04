# CuBIE

## CUDA Batch Integration Engine for Python

[![Docs](https://github.com/cubiepy/cubie/actions/workflows/documentation.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/documentation.yml)
[![CUDA tests](https://github.com/cubiepy/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/ci_cuda_tests.yml)
[![Python tests](https://github.com/cubiepy/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/ci_nocuda_tests.yml)
[![codecov](https://codecov.io/gh/cubiepy/cubie/graph/badge.svg?token=SKJNOT6061)](https://codecov.io/gh/cubiepy/cubie)
![PyPI version](https://img.shields.io/pypi/v/cubie)

CuBIE JIT-compiles CUDA kernels with Numba to integrate large batches of
ordinary differential equations (ODEs) and differential-algebraic equations
(DAEs) on NVIDIA GPUs. It provides a SciPy-like `solve_ivp` function and a
reusable `Solver` interface without requiring users to write CUDA code.

CuBIE is pre-1.0, so its public interface may still change.

## Capabilities

- Build combinatorial parameter and initial-condition sweeps, or solve
  verbatim batches from NumPy, CuPy, or Numba arrays.
- Define systems with Python callables, equation strings, symbolic
  expressions, or CellML 1.0/1.1 models.
- Use fixed- or adaptive-step explicit Runge-Kutta, diagonally implicit
  Runge-Kutta, fully implicit Runge-Kutta, and Rosenbrock-W methods.
- Structurally simplify DAEs with alias elimination, index reduction, and
  tearing before generating solver code.
- Supply time-dependent drivers as functions or sampled arrays.
- Save selected states and observables, or calculate summary metrics on the
  GPU without storing complete trajectories.
- Keep inputs and outputs on the GPU, automatically chunk batches that exceed
  available VRAM, and spill very large host results to disk.
- Cache generated source and compiled kernels between sessions.

## Installation

```console
pip install "cubie[mlir-cuda13]"
```

The extra in square brackets installs the required CUDA dependencies. There
are four options:

- `mlir-cuda12`
- `mlir-cuda13`
- `cuda12`
- `cuda13`

We recommend `mlir-cuda13` unless you have a specific reason to use the older
numba-cuda backend or the CUDA 12 toolkit.

CuBIE requires Python 3.11-3.14, an up-to-date NVIDIA driver, and an NVIDIA GPU
with compute capability 6.0 or later. Python 3.10 is supported only by the
numba-cuda backend. Pandas and Matplotlib support can be installed with
`pip install "cubie[optional]"`.

## Quick start

```python
from cubie import solve_ivp


with solve_ivp(
    ["dx = v", "dv = mu * (1 - x*x) * v - x"],
    y0={"x": [1.0, 2.0], "v": [0.0]},
    parameters={"mu": [1.0, 1.5, 2.0]},
    method="tsit5",
    duration=20.0,
    save_every=0.01,
) as result:
    trajectories = result.as_numpy["time_domain_array"]
```

This integrates every combination of the supplied initial conditions and
parameters. The first solve compiles and caches the CUDA kernels for reuse.

## Documentation

The [documentation](https://cubiepy.github.io/cubie/) covers system creation,
batching, solver configuration, outputs, and performance.

## Acknowledgements

- **[SciML/DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)**
  — No code is directly ported from DifferentialEquations.jl, but I treat its
  solver suite as the authority on numerical integration. I check CuBIE's
  methods against it, and when an implementation is unclear I first look at
  how DifferentialEquations.jl handles it. See
  [Rackauckas and Nie (2017)](https://doi.org/10.5334/jors.151).
- **[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)** —
  CuBIE's DAE tearing and structural-simplification implementation is a direct
  port of ModelingToolkit.jl's approach, adapted to CuBIE's symbolic IR and
  CUDA code generation. See
  [Ma et al. (2021)](https://doi.org/10.48550/arXiv.2103.05244).
- **[cellmlmanip](https://github.com/ModellingWebLab/cellmlmanip) and
  [chaste_codegen](https://github.com/ModellingWebLab/chaste-codegen)** — Their
  work is used to import CellML models and detect and repair removable
  singularities in Goldman-Hodgkin-Katz-style equations. See
  [Hendrix et al. (2022)](https://doi.org/10.12688/wellcomeopenres.17206.2).

## Contributing

Pull requests are welcome. Please open an issue before starting a major change
so that the design can be discussed first.
