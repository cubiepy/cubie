"""Matrix-free BiCGSTAB linear solver.

This module builds CUDA device functions that implement the
Bi-Conjugate Gradient Stabilized algorithm without forming
Jacobian matrices explicitly.

Published Classes
-----------------
:class:`BiCGSTABSolverConfig`
    Attrs configuration for the BiCGSTAB solver factory.

:class:`BiCGSTABSolver`
    CUDAFactory subclass that compiles a preconditioned BiCGSTAB
    solver for use inside Newton--Krylov iterations.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    Abstract parent providing shared infrastructure.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Newton--Krylov solver that wraps a linear solver.
"""

from math import sqrt as math_sqrt
from typing import Dict, Any

from attrs import frozen
from cubie.cuda_simsafe import cuda, int32, unroll_if
from numpy import float32 as np_float32, float64 as np_float64

from cubie._utils import PrecisionDType
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    IterativeLinearSolverConfig,
    IterativeLinearSolverBase,
    LinearSolverCache,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, fmin, selp
from cubie.result_codes import CUBIE_RESULT_CODES


@frozen
class BiCGSTABSolverConfig(IterativeLinearSolverConfig):
    """Configuration for BiCGSTABSolver compilation."""

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return the settings dictionary with the correction type."""
        settings = super().settings_dict
        settings["linear_correction_type"] = "bicgstab"
        return settings


class BiCGSTABSolver(IterativeLinearSolverBase):
    """Factory for BiCGSTAB linear solver device functions.

    Implements the Bi-Conjugate Gradient Stabilized algorithm
    for solving linear systems without forming Jacobian matrices.
    Uses 5 work vectors: r0_hat, p, v, tmp, s_hat.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    solver_width : int
        Length of residual and search-direction vectors.
    **kwargs
        Forwarded to :class:`BiCGSTABSolverConfig` and the norm
        factory.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        **kwargs,
    ) -> None:
        super().__init__(
            config_class=BiCGSTABSolverConfig,
            precision=precision,
            solver_width=solver_width,
            **kwargs,
        )

    def register_buffers(self) -> None:
        """Register 7 device buffers with buffer_registry."""
        config = self.compile_settings
        for name, loc in [
            ("bicg_r0_hat", config.r0_hat_location),
            ("bicg_p", config.p_location),
            ("bicg_v", config.v_location),
            ("bicg_tmp", config.tmp_location),
            ("bicg_s_hat", config.s_hat_location),
        ]:
            buffer_registry.register(
                name, self, config.solver_width, loc
            )

    @property
    def linear_correction_type(self) -> str:
        """Return 'bicgstab' as the correction strategy."""
        return "bicgstab"

    def build(self) -> LinearSolverCache:
        """Compile BiCGSTAB solver device function.

        Returns
        -------
        LinearSolverCache
            Container with compiled linear_solver device function.
        """
        config = self.compile_settings

        # Device Functions
        operator_apply = config.operator_apply
        preconditioner = config.preconditioner
        scaled_norm_fn = config.norm_device_function

        # Config parameters
        n = config.solver_width
        max_iters = config.max_iters
        precision = config.precision

        preconditioned = preconditioner is not None
        zero_initial_guess = config.zero_initial_guess
        reference_is_state = config.norm_reference == "state"
        jit_kwargs = self.jit_kwargs

        # Convert types for device function
        n_val = int32(n)
        unroll_solver_element = config.unroll.unroll_solver_element
        max_iters_val = int32(max_iters)
        precision_numba = config.numba_precision
        typed_zero = precision_numba(0.0)
        typed_reduction = config.residual_reduction
        typed_floor = config.residual_floor
        typed_largest = config.largest_finite
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_linear_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED
        )
        bicgstab_breakdown = int32(CUBIE_RESULT_CODES.BICGSTAB_BREAKDOWN)

        # Breakdown thresholds: absolute floors for rho and omega,
        # plus relative overflow guards on every recurrence quotient.
        # A quotient whose magnitude would exceed ``dot_clamp`` is a
        # breakdown: computing it would poison x, r, or p with values
        # the elementwise clamps cannot repair once they reach inf or
        # NaN (NaN passes through selp comparisons untouched). Keeping
        # every scalar and vector element within ``dot_clamp`` keeps
        # all products finite (clamp**2 is representable in both
        # precisions), so breakdown detection stays functional.
        if precision == np_float32:
            breakdown_tol_rho = precision_numba(1e-30)
            breakdown_tol_omega = precision_numba(1e-30)
            dot_clamp = precision_numba(1e16)
        elif precision == np_float64:
            breakdown_tol_rho = precision_numba(1e-200)
            breakdown_tol_omega = precision_numba(1e-200)
            dot_clamp = precision_numba(1e150)
        else:
            breakdown_tol_rho = precision_numba(1e-200)
            breakdown_tol_omega = precision_numba(1e-200)
            dot_clamp = precision_numba(1e150)

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_r0_hat = get_alloc("bicg_r0_hat", self)
        alloc_p = get_alloc("bicg_p", self)
        alloc_v = get_alloc("bicg_v", self)
        alloc_tmp = get_alloc("bicg_tmp", self)
        alloc_s_hat = get_alloc("bicg_s_hat", self)

        # no cover: start

        # Bind the norm's scaling reference at compile time.
        if reference_is_state:
            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def weighted_norm(values, state, base_state):
                return scaled_norm_fn(values, state)
        else:
            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def weighted_norm(values, state, base_state):
                return scaled_norm_fn(values, base_state)

        @cuda.jit(
            device=True,
            inline=True,
            **jit_kwargs,
        )
        def bicgstab_solver(
            state,
            parameters,
            drivers,
            base_state,
            cached_aux,
            t,
            h,
            a_ij,
            rhs,
            x,
            shared,
            persistent_local,
            krylov_iters_out,
        ):
            """Run one preconditioned BiCGSTAB solve.

            Parameters
            ----------
            state
                State vector forwarded to operator/preconditioner.
            parameters
                Model parameters forwarded to operator/preconditioner.
            drivers
                External drivers forwarded to operator/preconditioner.
            base_state
                Base state for n-stage operators.
            cached_aux
                Cached auxiliary values; may be zero-length.
            t
                Stage time.
            h
                Step size used by the operator evaluation.
            a_ij
                Stage coefficient.
            rhs
                Right-hand side; overwritten with running residual.
            x
                Initial guess; overwritten with final solution.
            shared
                Shared memory pool.
            persistent_local
                Persistent local memory pool.
            krylov_iters_out
                Single-element int32 array receiving iteration count.

            Returns
            -------
            int32
                ``0`` on convergence, ``4`` when iteration limit
                reached, ``128`` on BiCGSTAB breakdown.
            """

            # Allocate buffers
            r0_hat = alloc_r0_hat(shared, persistent_local)
            p = alloc_p(shared, persistent_local)
            v = alloc_v(shared, persistent_local)
            tmp = alloc_tmp(shared, persistent_local)
            s_hat = alloc_s_hat(shared, persistent_local)

            # ── INIT ────────────────────────────────────
            # Target: ||r|| <= floor + reduction * ||b||.
            rhs_norm2 = weighted_norm(rhs, state, base_state)
            tol = typed_floor + typed_reduction * precision_numba(
                math_sqrt(rhs_norm2)
            )
            tol2 = fmin(tol * tol, typed_largest)

            mask = activemask()
            if zero_initial_guess:
                # A zero guess leaves the residual equal to rhs.
                converged = rhs_norm2 <= tol2
                # Warp-uniform zero-iteration exit skips seeding.
                if all_sync(mask, converged):
                    krylov_iters_out[0] = int32(0)
                    return success
            else:
                converged = False
                operator_apply(
                    state, parameters, drivers, cached_aux, base_state,
                    t, h, a_ij, x, tmp,
                )

            # I1-I5 fused: seed r, r0_hat, p, rho_prev in one pass.
            rho_prev = typed_zero
            for i in unroll_if(range(n_val), unroll_solver_element):
                if zero_initial_guess:
                    residual_i = rhs[i]
                else:
                    ax = tmp[i]
                    ax = selp(ax > dot_clamp, dot_clamp, ax)
                    ax = selp(ax < -dot_clamp, -dot_clamp, ax)
                    residual_i = rhs[i] - ax
                rhs[i] = residual_i
                r0_hat[i] = residual_i
                pi = selp(
                    residual_i > dot_clamp, dot_clamp, residual_i
                )
                pi = selp(pi < -dot_clamp, -dot_clamp, pi)
                p[i] = pi
                sq = residual_i * residual_i
                sq = selp(sq > dot_clamp, dot_clamp, sq)
                rho_prev += sq

            # I6: initial convergence check on the seeded residual.
            if not zero_initial_guess:
                acc = weighted_norm(rhs, state, base_state)
                converged = acc <= tol2
            broken = False
            finished = converged

            iter_count = int32(0)

            for _ in range(max_iters_val):
                if all_sync(mask, finished):
                    break

                iter_count = selp(
                    not finished,
                    int32(iter_count + int32(1)),
                    iter_count,
                )

                # ── Step 1: tmp = P(p), jvp = v ─────
                # p is maintained within the clamp budget, so the
                # unpreconditioned copy needs no re-clamp.
                if preconditioned:
                    preconditioner(
                        state, parameters, drivers, cached_aux,
                        base_state, t, h, a_ij, p, tmp, v,
                    )
                    for i in unroll_if(range(n_val), unroll_solver_element):
                        tmp[i] = selp(
                            tmp[i] > dot_clamp, dot_clamp, tmp[i]
                        )
                        tmp[i] = selp(
                            tmp[i] < -dot_clamp, -dot_clamp, tmp[i]
                        )
                else:
                    for i in unroll_if(range(n_val), unroll_solver_element):
                        tmp[i] = p[i]

                # ── Step 2-3 fused: v = clamp(A(tmp)) and
                # dot_r0v = <r0_hat, v> in one pass.
                operator_apply(
                    state, parameters, drivers, cached_aux, base_state,
                    t, h, a_ij, tmp, v,
                )
                dot_r0v = typed_zero
                for i in unroll_if(range(n_val), unroll_solver_element):
                    vi = v[i]
                    vi = selp(vi > dot_clamp, dot_clamp, vi)
                    vi = selp(vi < -dot_clamp, -dot_clamp, vi)
                    v[i] = vi
                    prod = r0_hat[i] * vi
                    prod = selp(prod > dot_clamp, dot_clamp, prod)
                    prod = selp(prod < -dot_clamp, -dot_clamp, prod)
                    dot_r0v += prod
                # Pivot breakdown: <r0_hat, v> vanished relative to
                # rho, so the quotient would exceed the clamp budget.
                alpha_overflow = (
                    abs(rho_prev) > abs(dot_r0v) * dot_clamp
                )
                broken = broken or (
                    (not finished) and alpha_overflow
                )
                finished = converged or broken
                alpha = selp(
                    (dot_r0v != typed_zero) and (not alpha_overflow),
                    rho_prev / dot_r0v,
                    typed_zero,
                )

                # ── Step 4-5 fused: x += alpha*tmp and
                # s = r - alpha*v. Frozen lanes multiply by zero
                # instead of predicating each element.
                alpha_eff = selp(finished, typed_zero, alpha)
                for i in unroll_if(range(n_val), unroll_solver_element):
                    x[i] = x[i] + alpha_eff * tmp[i]
                    rhs[i] = rhs[i] - alpha_eff * v[i]

                # ── Step 6: half-step convergence check ─
                acc = weighted_norm(rhs, state, base_state)
                converged = converged or (acc <= tol2)
                finished = converged or broken

                # ── Step 7: s_hat = clamp(P(s)), jvp = tmp
                if preconditioned:
                    preconditioner(
                        state, parameters, drivers, cached_aux,
                        base_state, t, h, a_ij, rhs, s_hat, tmp,
                    )
                    for i in unroll_if(range(n_val), unroll_solver_element):
                        s_hat[i] = selp(
                            s_hat[i] > dot_clamp, dot_clamp, s_hat[i]
                        )
                        s_hat[i] = selp(
                            s_hat[i] < -dot_clamp, -dot_clamp, s_hat[i]
                        )
                else:
                    for i in unroll_if(range(n_val), unroll_solver_element):
                        si = rhs[i]
                        si = selp(si > dot_clamp, dot_clamp, si)
                        si = selp(si < -dot_clamp, -dot_clamp, si)
                        s_hat[i] = si

                # ── Step 8-9 fused: tmp = clamp(A(s_hat)),
                # omega = <tmp,s>/<tmp,tmp> in the same pass.
                operator_apply(
                    state, parameters, drivers, cached_aux, base_state,
                    t, h, a_ij, s_hat, tmp,
                )
                dot_ts = typed_zero
                dot_tt = typed_zero
                for i in unroll_if(range(n_val), unroll_solver_element):
                    ti = tmp[i]
                    ti = selp(ti > dot_clamp, dot_clamp, ti)
                    ti = selp(ti < -dot_clamp, -dot_clamp, ti)
                    tmp[i] = ti
                    prod = ti * rhs[i]
                    prod = selp(prod > dot_clamp, dot_clamp, prod)
                    prod = selp(prod < -dot_clamp, -dot_clamp, prod)
                    dot_ts += prod
                    sq = ti * ti
                    sq = selp(sq > dot_clamp, dot_clamp, sq)
                    dot_tt += sq
                # An overflowing quotient zeroes omega; the absolute
                # omega floor in Step 14 then labels the breakdown.
                omega_overflow = abs(dot_ts) > dot_tt * dot_clamp
                omega = selp(
                    (dot_tt != typed_zero) and (not omega_overflow),
                    dot_ts / dot_tt,
                    typed_zero,
                )

                # ── Step 10-11 fused: x += omega*s_hat and
                # r = s - omega*tmp, zero-multiplied when frozen.
                omega_eff = selp(finished, typed_zero, omega)
                for i in unroll_if(range(n_val), unroll_solver_element):
                    x[i] = x[i] + omega_eff * s_hat[i]
                    rhs[i] = rhs[i] - omega_eff * tmp[i]

                # ── Step 12: full-step convergence check ─
                acc = weighted_norm(rhs, state, base_state)
                converged = converged or (acc <= tol2)

                # ── Step 13: rho_new = <r0_hat, r> ──────
                rho_new = typed_zero
                for i in unroll_if(range(n_val), unroll_solver_element):
                    prod = r0_hat[i] * rhs[i]
                    prod = selp(prod > dot_clamp, dot_clamp, prod)
                    prod = selp(prod < -dot_clamp, -dot_clamp, prod)
                    rho_new += prod

                # ── Step 14-15: breakdown detection ──────
                rho_bad = abs(rho_new) < breakdown_tol_rho
                omega_bad = abs(omega) < breakdown_tol_omega
                broken = broken or (
                    (not converged) and (rho_bad or omega_bad)
                )
                finished = converged or broken

                # ── Step 16: beta ────────────────────────
                # A factor exceeding the clamp budget (omega tiny
                # relative to alpha, or rho rebounding relative to
                # rho_prev) is a breakdown; beta would poison p.
                beta_overflow = (
                    abs(rho_new) > abs(rho_prev) * dot_clamp
                ) or (
                    abs(alpha) > abs(omega) * dot_clamp
                )
                broken = broken or (
                    (not finished) and beta_overflow
                )
                finished = converged or broken
                beta = selp(
                    not finished,
                    (rho_new / rho_prev) * (alpha / omega),
                    typed_zero,
                )

                # ── Step 17: p = r + beta*(p - omega*v) ──
                for i in unroll_if(range(n_val), unroll_solver_element):
                    p[i] = selp(
                        not finished,
                        rhs[i] + beta * (p[i] - omega * v[i]),
                        p[i],
                    )
                    p[i] = selp(
                        p[i] > dot_clamp, dot_clamp, p[i]
                    )
                    p[i] = selp(
                        p[i] < -dot_clamp, -dot_clamp, p[i]
                    )

                # ── Step 18: rho_prev = rho_new ─────────
                rho_prev = selp(
                    not finished, rho_new, rho_prev
                )

            # ── Exit ────────────────────────────────────
            final_status = selp(
                converged, success,
                selp(broken, bicgstab_breakdown, max_linear_iters_exceeded),
            )
            krylov_iters_out[0] = iter_count
            return final_status

        # no cover: end
        return LinearSolverCache(linear_solver=bicgstab_solver)
