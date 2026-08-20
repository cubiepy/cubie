"""Concrete solver-helper roles binding source generation.

Each role is one :class:`~cubie.odesystems.solver_helpers.SolverHelperRole`
subclass declaring its capabilities and binding contract and
implementing ``generate``. Two identities per request:
:func:`helper_source_hash` (role, variant, system identity, stage
spec, cache selection) names the generated factory, and
:func:`helper_member_hash` (source hash plus the declared binding
arguments) keys the bound member. Bindings that share source reuse
one generated factory.

See Also
--------
:mod:`cubie.odesystems.solver_helpers`
    Declarative role base plus request and cache containers.
:mod:`cubie.odesystems.symbolic.codegen`
    Source generators the roles dispatch to.
"""

from cubie._serialize import canonical_digest
from cubie.odesystems.solver_helpers import (
    HelperVariant,
    ORDERED_FACTORY_ARGS,
    SCALAR_FACTORY_ARGS,
    SolverHelperRequest,
    SolverHelperRole,
)
from cubie.odesystems.symbolic.codegen import (
    generate_apply_mass_code,
    generate_jacobi_preconditioner_code,
    generate_linear_operator_code,
    generate_neumann_preconditioner_code,
    generate_no_preconditioner_code,
    generate_prepare_jac_code,
    generate_residual_code,
)
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_evaluate_inv_mass_f_code,
)
from cubie.odesystems.symbolic.codegen.lu_solver import (
    generate_lu_prepare_blocks_code,
    generate_lu_smoothing_solve_code,
    generate_lu_solve_code,
)
from cubie.odesystems.symbolic.codegen.time_derivative import (
    generate_time_derivative_fac_code,
)
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    check_neumann_convergence,
)

__all__ = [
    "LinearOperator",
    "NeumannPreconditioner",
    "JacobiPreconditioner",
    "NoPreconditioner",
    "LuSolve",
    "LuPrepareBlocks",
    "LuSmoothingSolve",
    "Residual",
    "ApplyMass",
    "EvaluateInvMassF",
    "TimeDerivativeRHS",
    "PrepareJac",
    "helper_source_hash",
    "helper_member_hash",
]


class LinearOperator(SolverHelperRole):
    """Linear operator applying ``beta*M@v - gamma*a_ij*h*J@v``."""

    name = "linear_operator"
    jacobian_carrying = True
    stacked_capable = True
    folded_args = ("beta", "gamma", "a_ij")

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_linear_operator_code(
            system.equations,
            system.indices,
            variant=request.variant,
            M=system.compile_settings.mass,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            jvp_equations=system._get_jvp_exprs(),
            operation_ordering=system.operation_ordering,
            beta=request.beta,
            gamma=request.gamma,
            a_ij=request.a_ij,
        )


class NeumannPreconditioner(SolverHelperRole):
    """Truncated Neumann series preconditioner."""

    name = "neumann_preconditioner"
    jacobian_carrying = True
    stacked_capable = True
    factory_args = ORDERED_FACTORY_ARGS
    folded_args = ("beta", "gamma", "a_ij")
    preconditioner_type_name = "neumann"
    default_preconditioner_order = 2

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_neumann_preconditioner_code(
            system.equations,
            system.indices,
            variant=request.variant,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            jvp_equations=system._get_jvp_exprs(),
            operation_ordering=system.operation_ordering,
            beta=request.beta,
            gamma=request.gamma,
            a_ij=request.a_ij,
        )

    @classmethod
    def validate(cls, system, request, cache_policy):
        """Reject mass-matrix systems, then run the convergence check.

        Raises
        ------
        ValueError
            If the system carries a mass matrix; Neumann assumes the
            identity.
        """
        if system.compile_settings.mass is not None:
            raise ValueError(
                "Neumann preconditioners assume an identity mass "
                "matrix and cannot precondition a system with torn "
                "algebraic rows. Use preconditioner_type='jacobi'."
            )
        check_neumann_convergence(
            system.indices,
            evaluator=system._get_neumann_evaluator(cache_policy),
            stage_coefficients=request.stage_coefficients,
            beta=request.beta,
            gamma=request.gamma,
        )


class JacobiPreconditioner(SolverHelperRole):
    """Diagonal Jacobi preconditioner with an optional series."""

    name = "jacobi_preconditioner"
    jacobian_carrying = True
    stacked_capable = True
    factory_args = ORDERED_FACTORY_ARGS
    folded_args = ("beta", "gamma", "a_ij")
    preconditioner_type_name = "jacobi"
    default_preconditioner_order = 0

    @classmethod
    def validate(cls, system, request, cache_policy):
        """Reject series orders on stacked multi-stage operators."""
        if (
            request.stacked
            and request.preconditioner_order > 0
            and len(request.stage_coefficients) > 1
        ):
            raise ValueError(
                "Jacobi series orders above zero diverge on stacked "
                "multi-stage operators; use preconditioner_order=0."
            )

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_jacobi_preconditioner_code(
            system.equations,
            system.indices,
            variant=request.variant,
            M=system.compile_settings.mass,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            jvp_equations=system._get_jvp_exprs(),
            operation_ordering=system.operation_ordering,
            beta=request.beta,
            gamma=request.gamma,
            a_ij=request.a_ij,
        )


class NoPreconditioner(SolverHelperRole):
    """``preconditioner_type='none'``: identity preconditioner."""

    name = "no_preconditioner"
    stacked_capable = True
    preconditioner_type_name = "none"

    @classmethod
    def legal_variants(cls):
        """Accept every variant; only the width follows the axes."""
        return frozenset(HelperVariant)

    @classmethod
    def generate(cls, system, request, func_name):
        n_out = system.sizes.states
        if request.stacked:
            n_out *= len(request.stage_coefficients)
        return generate_no_preconditioner_code(
            n_out=n_out, func_name=func_name
        )


class LuSolve(SolverHelperRole):
    """Direct sparse LU solve of ``beta*M - gamma*a_ij*h*J``."""

    name = "lu_solve"
    jacobian_carrying = True
    stacked_capable = True
    prefactor_capable = True
    factory_args = SCALAR_FACTORY_ARGS
    folded_args = ("beta", "gamma", "a_ij")

    @classmethod
    def uses_cache_selection(cls, variant):
        return variant is HelperVariant.CACHED

    @classmethod
    def prepare_request_kwargs(cls, request):
        if request.variant is HelperVariant.CACHED:
            return super().prepare_request_kwargs(request)
        return {
            "role": "lu_prepare_blocks",
            "jacobian_at": "step",
            "prefactored": True,
            "stacked": request.stacked,
            "beta": request.beta,
            "gamma": request.gamma,
            "stage_coefficients": request.stage_coefficients,
            "stage_nodes": request.stage_nodes,
        }

    @classmethod
    def generate(cls, system, request, func_name):
        code, _ = generate_lu_solve_code(
            system.equations,
            system.indices,
            variant=request.variant,
            M=system.compile_settings.mass,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            jvp_equations=system._get_jvp_exprs(),
            operation_ordering=system.operation_ordering,
            beta=request.beta,
            gamma=request.gamma,
            a_ij=request.a_ij,
        )
        return code


class LuPrepareBlocks(SolverHelperRole):
    """Step-start LU block factorisation companion for lu_solve.

    Factors land in ``cached_aux``; the stamped ``aux_count`` is the
    flat factor length in reals.
    """

    name = "lu_prepare_blocks"
    jacobian_carrying = True
    is_prepare_helper = True
    factory_args = SCALAR_FACTORY_ARGS
    folded_args = ("beta", "gamma")

    @classmethod
    def legal_variants(cls):
        return frozenset(
            {
                HelperVariant.PREFACTORED,
                HelperVariant.PREFACTORED_STACKED,
            }
        )

    @classmethod
    def uses_cache_selection(cls, variant):
        return False

    @classmethod
    def generate(cls, system, request, func_name):
        code, _ = generate_lu_prepare_blocks_code(
            system.equations,
            system.indices,
            variant=request.variant,
            M=system.compile_settings.mass,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            operation_ordering=system.operation_ordering,
            beta=request.beta,
            gamma=request.gamma,
        )
        return code


class LuSmoothingSolve(SolverHelperRole):
    """Smoothed-error solve on the block transform's real block."""

    name = "lu_smoothing_solve"
    jacobian_carrying = True
    factory_args = SCALAR_FACTORY_ARGS
    folded_args = ("gamma",)

    @classmethod
    def legal_variants(cls):
        return frozenset({HelperVariant.PREFACTORED_STACKED})

    @classmethod
    def uses_cache_selection(cls, variant):
        return False

    @classmethod
    def prepare_request_kwargs(cls, request):
        return {
            "role": "lu_prepare_blocks",
            "jacobian_at": "step",
            "prefactored": True,
            "stacked": True,
            "beta": request.beta,
            "gamma": request.gamma,
            "stage_coefficients": request.stage_coefficients,
            "stage_nodes": request.stage_nodes,
        }

    @classmethod
    def generate(cls, system, request, func_name):
        code, _ = generate_lu_smoothing_solve_code(
            system.equations,
            system.indices,
            M=system.compile_settings.mass,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            operation_ordering=system.operation_ordering,
            gamma=request.gamma,
        )
        return code


class Residual(SolverHelperRole):
    """Nonlinear stage-increment residual."""

    name = "residual"
    stacked_capable = True
    folded_args = ("beta", "gamma", "a_ij")

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_residual_code(
            system.equations,
            system.indices,
            variant=request.variant,
            M=system.compile_settings.mass,
            stage_coefficients=request.stage_coefficients,
            stage_nodes=request.stage_nodes,
            func_name=func_name,
            operation_ordering=system.operation_ordering,
            beta=request.beta,
            gamma=request.gamma,
            a_ij=request.a_ij,
        )


class ApplyMass(SolverHelperRole):
    """Mass-matrix product ``out = M @ v``."""

    name = "apply_mass"
    factory_args = SCALAR_FACTORY_ARGS

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_apply_mass_code(
            system.equations,
            system.indices,
            M=system.compile_settings.mass,
            func_name=func_name,
        )


class EvaluateInvMassF(SolverHelperRole):
    """Combined evaluation ``out = M**-1 @ f``."""

    name = "evaluate_inv_mass_f"
    factory_args = SCALAR_FACTORY_ARGS

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_evaluate_inv_mass_f_code(
            system.equations,
            system.indices,
            M=system.compile_settings.mass,
            func_name=func_name,
            operation_ordering=system.operation_ordering,
        )


class TimeDerivativeRHS(SolverHelperRole):
    """Partial time derivative of the right-hand side."""

    name = "time_derivative_rhs"
    factory_args = SCALAR_FACTORY_ARGS

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_time_derivative_fac_code(
            system.equations,
            system.indices,
            func_name=func_name,
            operation_ordering=system.operation_ordering,
        )


class PrepareJac(SolverHelperRole):
    """Auxiliary-cache preparation companion for cached helpers.

    Served on every cached Jacobian-carrying member as
    ``HelperResult.prepare_jac``; accepts only ``CACHED``.
    """

    name = "prepare_jac"
    jacobian_carrying = True
    is_prepare_helper = True
    factory_args = SCALAR_FACTORY_ARGS

    @classmethod
    def legal_variants(cls):
        return frozenset({HelperVariant.CACHED})

    @classmethod
    def generate(cls, system, request, func_name):
        code, _ = generate_prepare_jac_code(
            system.equations,
            system.indices,
            func_name=func_name,
            jvp_equations=system._get_jvp_exprs(),
            operation_ordering=system.operation_ordering,
        )
        return code


def helper_source_hash(system, request: SolverHelperRequest) -> str:
    """Return the generated-source identity for a request.

    Constants enter through ``fn_hash``; factory bindings key the
    member hash; ``folded_args`` bake into the source, so key here.
    """
    selection = None
    if request.role.uses_cache_selection(request.variant):
        plan = system._get_jvp_exprs().cache_selection
        selection = (
            tuple(repr(leaf) for leaf in plan.cached_leaf_order),
            tuple(repr(node) for node in plan.removal_nodes),
        )
    folded = tuple(
        (name, getattr(request, name))
        for name in request.role.folded_args
    )
    return canonical_digest(
        (
            "cubie-helper-source",
            request.role.name,
            request.variant.value,
            system.fn_hash,
            system.compile_settings.operation_ordering,
            request.stage_coefficients,
            request.stage_nodes,
            selection,
            folded,
        )
    )


def helper_member_hash(source_hash: str, canonical_args: tuple) -> str:
    """Return the bound-member identity for one factory binding."""
    return canonical_digest(
        ("cubie-helper-member", source_hash, canonical_args)
    )
