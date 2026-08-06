"""Time the readme example exactly as written, after a warm-up solve."""

import time

import numpy as np

from cubie import create_ODE_system, solve_ivp

system = create_ODE_system(
    ["dx = v", "dv = mu * (1 - x*x) * v - x"],
    states={"x": 1.0, "v": 0.0},
    parameters={"mu": 1.5},
)

t0 = time.perf_counter()
warm = solve_ivp(
    system,
    y0={"x": np.linspace(1.0, 2.0, 8), "v": [0.0]},
    parameters={"mu": np.linspace(1.0, 3.0, 8)},
    method="rk45",
    duration=20.0,
    atol=1e-6,
    rtol=1e-3,
)
print(f"warm-up (64 runs, includes compile): {time.perf_counter() - t0:.2f} s")

for rep in range(3):
    t0 = time.perf_counter()
    result = solve_ivp(
        system,
        y0={"x": np.linspace(1.0, 2.0, 1024), "v": [0.0]},
        parameters={"mu": np.linspace(1.0, 3.0, 1024)},
        method="rk45",
        duration=20.0,
        atol=1e-6,
        rtol=1e-3,
    )
    elapsed = time.perf_counter() - t0
    print(f"rep {rep}: 1,048,576 runs in {elapsed * 1e3:.1f} ms")
