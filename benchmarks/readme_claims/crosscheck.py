"""Score matlab_crosscheck.csv and scipy on the same runs and reference."""

import numpy as np
from scipy.integrate import solve_ivp

from scipy_vdp_bench import ATOL, DURATION, RTOL, rhs_python

data = np.loadtxt("matlab_crosscheck.csv", delimiter=",")
x0s, mus, m_steps, m_x, m_v = data.T
m_end = np.column_stack([m_x, m_v])

s_steps = np.empty(x0s.size)
s_end = np.empty((x0s.size, 2))
ref = np.empty((x0s.size, 2))
for i, (x0, mu) in enumerate(zip(x0s, mus)):
    sol = solve_ivp(
        rhs_python,
        (0.0, DURATION),
        [x0, 0.0],
        method="RK45",
        rtol=RTOL,
        atol=ATOL,
        args=(mu,),
    )
    s_steps[i] = sol.t.size - 1
    s_end[i] = sol.y[:, -1]
    tight = solve_ivp(
        rhs_python,
        (0.0, DURATION),
        [x0, 0.0],
        method="RK45",
        rtol=1e-11,
        atol=1e-13,
        args=(mu,),
    )
    ref[i] = tight.y[:, -1]

print(f"steps   scipy median {np.median(s_steps):7.1f}  mean {s_steps.mean():7.1f}")
print(f"steps   matlab median {np.median(m_steps):6.1f}  mean {m_steps.mean():7.1f}")
for name, arr in (("scipy", s_end), ("matlab", m_end)):
    err = np.abs(arr - ref).max(axis=1)
    print(
        f"{name:<7} max|err| {err.max():.3e}  median {np.median(err):.3e}"
    )
