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

Under the hood, cubie uses [`numba-cuda-mlir`](https://github.com/NVIDIA/numba-cuda-mlir) to compile
python integration algorithms and your provided ODE/DAE systems into GPU code
and ferry your data in between your computer and GPU. Python-side, it generates Jacobian-vector 
product (JVP), residual, and preconditioner functions from your system of equations and folds those into
iterative linear or nonlinear solvers (depending on the algorithm you choose) which are compiled
into the final kernel. By treating the core math as code instead of evaluating it per-step,
cubie achieves a low memory footprint on the GPU, allowing you to fit more integrations
onto it at once. 


## Capabilities

- Define systems of ODE/DAEs as either Python functions, strings, SymPy symbolic
  expressions, or CellML 1.0/1.1 models.
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
pip install "cubie[cuda13]"
```

The extra in square brackets installs a CUDA toolkit: `cuda13` for CUDA 13,
`cuda12` for CUDA 12. Without it, `pip install cubie` uses a system CUDA install.

CuBIE requires Python 3.11-3.14, an up-to-date NVIDIA driver, and an NVIDIA GPU
with compute capability 6.0 or later. Pandas and Matplotlib support can be installed with
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

- **[SciML, DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)**
  — SciML's differential equations ecosystem was the main reference for every
  correctness, step-control, algorithm detail, or default constant question I
  ran into when building Cubie. Cubie's implementation differs a lot, mostly
  due to it's GPU-first nature, but almost every piece of math and many default
  settings values were either drawn from or rigorously checked against a SciML
  implementation.
  The DAE initialiser is ported directly from OrdinaryDiffEq.jl under an MIT
  license, as was some code in the DAE structural simplification pipeline.
  . Any solving of DAEs through Cubie should cite them directly: See
  [Rackauckas and Nie (2017)](https://doi.org/10.5334/jors.151).
  If you want to explore numerical integration further, head to their docs,
  where they have many accessibly-written explanations of which algorithms
  do what well.
- **[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)** —
  CuBIE's DAE tearing and structural-simplification implementation is largely
  a direct port of ModelingToolkit.jl's approach, adapted to CuBIE's symbolic
  IR and CUDA code generation. DAE users should cite them directly as well: See
  [Ma et al. (2021)](https://doi.org/10.48550/arXiv.2103.05244).
- **[cellmlmanip](https://github.com/ModellingWebLab/cellmlmanip) and
  [chaste_codegen](https://github.com/ModellingWebLab/chaste-codegen)** — Their
  work is used to import CellML models and detect and repair removable
  singularities in Goldman-Hodgkin-Katz-style equations. See
  [Hendrix et al. (2022)](https://doi.org/10.12688/wellcomeopenres.17206.2).

## License

MIT (`LICENSE`); third-party notices in `THIRD_PARTY_LICENSES`.

## Contributing

Pull requests are welcome. Please open an issue before starting a major change
so that the design can be discussed first.


_____

\* One million runs of the example above, on an RTX 4070 SUPER with an i7-12700: 29 ms in cubie, 47 minutes in SciPy (98,000×) and 2.7 minutes in MATLAB (5,500×). Using multiprocessing/parfor to run the integrations in parallel on the CPU, SciPy drops to 6 minutes (12,000×) and MATLAB to 1 minute (2,200×). Rough numbers from one machine.
