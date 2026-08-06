"""Compare final-state error of per-system and stacked-block solves.

Both run at rtol=1e-3/atol=1e-6 against a per-system rtol=1e-11 reference.
"""

import numpy as np
from scipy.integrate import solve_ivp

from scipy_vdp_bench import ATOL, DURATION, RTOL, rhs_block, rhs_python, sample_grid

B = 64
x0s, mus = sample_grid(B, seed=1)

ref = np.array(
    [
        solve_ivp(
            rhs_python,
            (0.0, DURATION),
            [x0, 0.0],
            method="RK45",
            rtol=1e-11,
            atol=1e-13,
            args=(mu,),
        ).y[:, -1]
        for x0, mu in zip(x0s, mus)
    ]
)

indiv = np.array(
    [
        solve_ivp(
            rhs_python,
            (0.0, DURATION),
            [x0, 0.0],
            method="RK45",
            rtol=RTOL,
            atol=ATOL,
            args=(mu,),
        ).y[:, -1]
        for x0, mu in zip(x0s, mus)
    ]
)

sol = solve_ivp(
    rhs_block,
    (0.0, DURATION),
    np.concatenate([x0s, np.zeros_like(x0s)]),
    method="RK45",
    rtol=RTOL,
    atol=ATOL,
    args=(np.ascontiguousarray(mus),),
)
mega = np.column_stack([sol.y[:B, -1], sol.y[B:, -1]])

for name, arr in (("individual", indiv), (f"mega/{B}", mega)):
    err = np.abs(arr - ref).max(axis=1)
    print(f"{name:<12} max|err| {err.max():.3e}  median {np.median(err):.3e}")

print(f"\nsteps: mega block = {sol.t.size - 1}")
