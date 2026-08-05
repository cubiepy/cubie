# CuBIE

## CUDA Batch Integration Engine for Python

[![Docs](https://github.com/cubiepy/cubie/actions/workflows/documentation.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/documentation.yml)
[![CUDA tests](https://github.com/cubiepy/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/ci_cuda_tests.yml)
[![Python tests](https://github.com/cubiepy/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/ci_nocuda_tests.yml)
[![codecov](https://codecov.io/gh/cubiepy/cubie/graph/badge.svg?token=SKJNOT6061)](https://codecov.io/gh/cubiepy/cubie)
![PyPI version](https://img.shields.io/pypi/v/cubie)

CuBIE performs numerical integration in parallel on NVIDIA GPUs. It
JIT-compiles ordinary differential equation (ODE) and differential-algebraic
equation (DAE) systems into CUDA kernels, making large parameter and
initial-condition sweeps much faster than calling SciPy's
[`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
or MATLAB's [`ode45`](https://www.mathworks.com/help/matlab/ref/ode45.html)
once per system.

On an RTX 4070 SUPER, a cached run of the 1,048,576-system RK45 example below
takes about 20 ms. Serial SciPy 1.18 calls at the same tolerances extrapolate
to about 47 minutes: over 100,000 times slower. MATLAB `ode45` follows the same
one-problem-per-call pattern; exact timings depend on the system and hardware.

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
import numpy as np
from cubie import create_ODE_system, solve_ivp


system = create_ODE_system(
    ["dx = v", "dv = mu * (1 - x*x) * v - x"],
    states={"x": 1.0, "v": 0.0},
    parameters={"mu": 1.5},
)

result = solve_ivp(
    system,
    y0={"x": np.linspace(1.0, 2.0, 1024), "v": [0.0]},
    parameters={"mu": np.linspace(1.0, 3.0, 1024)},
    method="rk45",
    duration=20.0,
    atol=1e-6,
    rtol=1e-3,
)
```

This integrates all 1,048,576 combinations of the 1,024 initial values and
1,024 parameter values. The first solve compiles and caches the CUDA kernels;
later solves reuse them.

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
