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
    generate_prepare_jac_code,
    generate_residual_code,
)
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_evaluate_inv_mass_f_code,
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
        )


class NeumannPreconditioner(SolverHelperRole):
    """Truncated Neumann series preconditioner."""

    name = "neumann_preconditioner"
    jacobian_carrying = True
    stacked_capable = True
    factory_args = ORDERED_FACTORY_ARGS
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
    preconditioner_type_name = "jacobi"
    default_preconditioner_order = 0

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
        )


class Residual(SolverHelperRole):
    """Nonlinear stage-increment residual."""

    name = "residual"
    stacked_capable = True

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
    returns_aux_count = True
    factory_args = SCALAR_FACTORY_ARGS

    @classmethod
    def legal_variants(cls):
        return frozenset({HelperVariant.CACHED})

    @classmethod
    def generate(cls, system, request, func_name):
        return generate_prepare_jac_code(
            system.equations,
            system.indices,
            func_name=func_name,
            jvp_equations=system._get_jvp_exprs(),
            operation_ordering=system.operation_ordering,
        )


def helper_source_hash(system, request: SolverHelperRequest) -> str:
    """Return the generated-source identity for a request.

    Contains only inputs that change the emitted source; binding
    values (beta, gamma, order, constants, precision, lineinfo) are
    absent.
    """
    selection = None
    if request.variant.cached:
        plan = system._get_jvp_exprs().cache_selection
        selection = (
            tuple(repr(leaf) for leaf in plan.cached_leaf_order),
            tuple(repr(node) for node in plan.removal_nodes),
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
        )
    )


def helper_member_hash(source_hash: str, canonical_args: tuple) -> str:
    """Return the bound-member identity for one factory binding."""
    return canonical_digest(
        ("cubie-helper-member", source_hash, canonical_args)
    )
