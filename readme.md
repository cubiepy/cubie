# CuBIE

## CUDA Batch Integration Engine for Python

[![Docs](https://github.com/cubiepy/cubie/actions/workflows/documentation.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/documentation.yml)
[![CUDA tests](https://github.com/cubiepy/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/ci_cuda_tests.yml)
[![Python tests](https://github.com/cubiepy/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/cubiepy/cubie/actions/workflows/ci_nocuda_tests.yml)
[![codecov](https://codecov.io/gh/cubiepy/cubie/graph/badge.svg?token=SKJNOT6061)](https://codecov.io/gh/cubiepy/cubie)
![PyPI version](https://img.shields.io/pypi/v/cubie)

CuBIE performs numerical integration in parallel on NVIDIA GPUs. It provides a ~10000x* 
speedup over functions like MATLAB's `ode45` and SciPy's `solve_ivp` for parallel batch integrations, 
while offering a similar interface to make it easy to switch from those environments.

Under the hood, cubie uses [`numba-cuda`](https://nvidia.github.io/numba-cuda/) to compile
python integration algorithms and your provided ODE/DAE systems into GPU code
and ferry your data in between your computer and GPU. Python-side, it generates Jacobian-vector 
product (JVP), residual, and preconditioner functions from your system of equations and folds those into
iterative linear or nonlinear solvers (depending on the algorithm you choose) which are compiled
into the final kernel. By treating the core math as code instead of evaluating it per-step,
cubie achieves a low memory footprint on the GPU, allowing you to fit more integrations
onto it at once. 


## Capabilities

- Define systems of ODE/DAEs as either Python functions, strings, SymPy symbolic
  expressions, or BigModel 1.0/1.1 models.
- Use fixed- or adaptive-step explicit Runge-Kutta, diagonally implicit
  Runge-Kutta, fully implicit Runge-Kutta, and Rosenbrock-W methods.
- Structurally simplify DAEs with alias elimination, index reduction, and
  tearing before generating solver code (logic taken almost verbatim from
  [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)).
- Supply time-dependent forcing terms as functions or sampled (measured) arrays.
- Save selected states or algebraic variables (observables) to reduce result size
- Discard trajectories and calculate summary metrics on the GPU to keep only the 
relevant information and allow larger solves.
- Automatically divide large solves into chunks that can fit into your GPU, and arrays
that can fit into your computers RAM, to allow REALLY large solves.
- Cache solvers between sessions, so you only pay the compile time once per config.
- Build combinatorial grids of parameters/initial conditions to solve over.

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
1,024 parameter values. The first solve compiles and caches the CUDA kernels (~0.1s);
later solves reuse them (~0.025s).

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
- **bigmodelmanip** — vendored under `src/cubie/vendored/bigmodelmanip`
  (see its `LICENSE`) to import BigModel files and to detect and repair
  removable singularities.

## Contributing

Pull requests are welcome. Please open an issue before starting a major change
so that the design can be discussed first.


_____

\* One million runs of the example above, on an RTX 4070 SUPER with an i7-12700: 29 ms in cubie, 47 minutes in SciPy (98,000×) and 2.7 minutes in MATLAB (5,500×). Using multiprocessing/parfor to run the integrations in parallel on the CPU, SciPy drops to 6 minutes (12,000×) and MATLAB to 1 minute (2,200×). Rough numbers from one machine.