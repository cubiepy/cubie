"""
``remove_fixable_singularities`` specifies a method to remove fixable singularities from a model
"""
from functools import lru_cache
from math import isclose

import networkx as nx
from sympy import (
    Add,
    And,
    Eq,
    Float,
    Le,
    Mul,
    Or,
    Piecewise,
    Pow,
    S,
    Wild,
    exp,
    log,
    solveset,
)
from sympy.codegen.rewriting import ReplaceOptim, optimize
from sympy.sets import Intersection

from . import parser
from .model import FLOAT_PRECISION, Quantity


# For P^n make sure n is passed as int if it is actually a whole number
_POW_OPT = ReplaceOptim(lambda p: p.is_Pow and isinstance(p.exp, (Float, float)) and float(p.exp).is_integer(),
                        lambda p: Pow(p.base, int(p.exp)))


ONE = Quantity(1.0, 'dimensionless')

ONE = Quantity(1.0, 'dimensionless')


def subs_parsed_math_funcs(expr):
    """ Substitutes math functions that have been changed with sympy functions so we can calculate the value
    :param expr: sympy expression

    Example:
    >> str(expr)
    '2.0 * exp_(V)'
    >> subs_parsed_math_funcs(expr)
    '2.0 * exp(V)'

    :return: expr with all placeholder functions replaced by sympy functions.
    """
    for tag, sympy_func in parser.SIMPLE_MATHML_TO_SYMPY_CLASSES.items():
        func_to_replace_with = parser._SIMPLE_MATHML_TO_SYMPY_CLASSES.get(tag, sympy_func)
        if sympy_func != func_to_replace_with:  # skip replacing with the same function
            expr = expr.replace(sympy_func, func_to_replace_with)
    return expr


@lru_cache(maxsize=128)
def _generate_piecewise(expr, V, sp, Vmin, Vmax):
    """
    Generates a new (piecewise) expression based on expr with a linear interpolation when V is between Vmin and Vmax

    The returned expression between Vmin and Vmax is::
    f(Vmin) + (V - Vmin) / (Vmax - Vmin) * (f(Vmax) - f(Vmin))
    where ``f`` is ``expr``

    :param expr: The expression to turn into a piecewise
    :param V: The voltage variable.
    :param sp: The value of the singularity point: the value of V for which U == 0.
    :param Vmin: The value of the lower bound: the value of V for which U - U_offset == 0.
    :param Vmax: The value of the upper bound: the value of V for which U + U_offset == 0.
    :return: expr with a singularity exception around sp, if sp is not None else return the original expr.

    *Please note:* Despite their names Vmin and Vmax are not necessarily ordered, we do not guarantee that Vmin < Vmax

    U here refers to the expression in the singularity witin the patter found
    U_offset is the offset in U specified when finding Vmin / Vmax
    see :meth:`_fix_expr_parts`
    """
    try:
        if float(Vmax) < float(Vmin):
            Vmin, Vmax = Vmax, Vmin
        range_condition = And(Le(Vmin, V), Le(V, Vmax))
    except TypeError:  # Vmax / Vmin are not a number, but contain variables
        range_condition = Or(And(Le(Vmin, V), Le(V, Vmax)), And(Le(Vmax, V), Le(V, Vmin)))

    f_Vmin = expr.xreplace({V: Vmin})
    f_Vmax = expr.xreplace({V: Vmax})
    return Piecewise((f_Vmin + ((V - Vmin) / (Vmax - Vmin)) * (f_Vmax - f_Vmin),
                      range_condition), (expr, True))


def _float_dummies(expr):
    """Replaces ``sympy.Float`` objects in ``expr`` with dimensionless :class:`bigmodelmanip.model.Quantity` objects."""
    return expr.xreplace({f: Quantity(f, 'dimensionless') for f in expr.atoms(Float)})


def _is_negative_power(expr):
    """
    Checks if ``expr`` is a sympy ``Pow`` with a negative exponent.

    If the exponent contains variables the sign of the exponent can't be determined, and ``False`` is returned.
    """
    try:
        return isinstance(expr, Pow) and bool(expr.args[1].evalf() < 0)
    except TypeError:  # if this is a power with variables still in it we can't determine if it's negative
        pass
    return False


def _solve_real(u, V):
    """Gives the values of V for which u == 0 """
    u = optimize(u, (_POW_OPT, ))  # make sure powers of ints are represented as ints
    result = solveset(u, V, domain=S.Reals)
    # The result is usually a set.
    # However if a custom function is used sympy doesn't know if the custom function is Real,
    # in that case the result of solveset is a sympy Intersection between the actual result and Reals.
    if isinstance(result, Intersection) and result.args[0] == S.Reals:
        result = result.args[1]
    return result


@lru_cache(maxsize=128)
def _get_singularity(expr, V, U_offset, exp_function):
    """
    Finds singularities in equations of the form:
       - ``U / (exp(U) - 1.0)``
       - ``U / (1.0 - exp(U))``
       - ``(exp(U) - 1.0) / U``
       - ``(1.0 - exp(U)) / U``

    In addition to the singularity itself, this functions returns a lower and upper bounds as follows:
    - sp is the singularity point: the value of V for which U == 0
    - Vmin is the value of V for which U - U_offset == 0
    - Vmax is the value of V for which U + U_offset == 0:

    *Please Note:* If U contains any non-V variables, the analysis assumes these to be positive (e.g. concentrations)
                   it is theoretically possible such variables mean a singularity is found where there is none.
                   E.g. if such variables make U evaluate to 0.

    :return: (Vmin, Vmax, sp)
    """

    # Create "wildcards", that act as catch-all (*) when matching expressions
    # https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.match
    P_wildcard = Wild('P_wildcard', real=True, exclude=[V])
    Z_wildcard = Wild('Z_wildcard', real=True)
    U_wildcard = Wild('U_wildcard', real=True, include=[V])
    SP_wildcard = Wild('SP_wildcard', real=True)
    singularities = []

    def check_U_match(m, sp):
        """
        Checks that the match object resulting from pattern matching the expression is a valid match.
        :param m: The match object resulting from pattern matching.
        :param sp: The singularity point found.
        :return: (Vmin, Vmax, sp)
        """
        assert m is None or m[Z_wildcard] != 0
        return m is not None and \
            (sp == m[SP_wildcard] or
             (isinstance(sp, (Float, float)) and
             isinstance(m[SP_wildcard], (Float, float)) and
             isclose(m[SP_wildcard], sp)))

    # find all fractions and sperate numerator and denominator
    # the denominator is all args where a **-1
    numerator, denominator = [], []
    for a in expr.args:
        if _is_negative_power(a):
            denominator.append(Pow(a.args[0], - a.args[1]))
        else:
            numerator.append(a)

    # Second check: must be a fraction
    if len(denominator) == 0 or len(numerator) == 0 or not expr.has(exp_function):
        return []

    numerator.append(Mul(*numerator))  # pattern could match entire numerator as well as any of its parts
    denominator.append(Mul(*denominator))  # pattern could match entire denominator as well as any of its parts

    # U might be on top, try numerator / denominator and denominator / numerator
    for fraction_part_1, fraction_part_2 in ((numerator, denominator), (denominator, numerator)):
        found_on_top = False
        for fp2 in fraction_part_2:  # Check arguments in denominator (or numerator)
            (Vmin, Vmax, sp) = None, None, None
            if not fp2.has(exp_function):
                continue
            # look for exp(U) * -Z_wildcard + 1.0
            # Works as we later reuire Z_wildcard to be positive
            find_U = fp2.match(exp_function(U_wildcard) * -Z_wildcard + 1.0)
            if not find_U:
                find_U = fp2.match(exp_function(U_wildcard) * Z_wildcard - 1.0)  # look for exp(U) * Z_wildcard - 1.0

            # Z_wildcard should be positive, we replace variables by 1 to be able to evaluate the sign of Z_wildcard
            # Please note: a limitation of this approach is that any variables in Z_wildcard
            # could have values that mean the singularity does not occur, such as 0
            if find_U and subs_parsed_math_funcs(find_U[Z_wildcard])\
                    .xreplace({s: 1.0 for s in find_U[Z_wildcard].free_symbols}) > 0:
                # We found a match for exp(U) * Z_wildcard -1 or exp(U) * -Z_wildcard +1,
                # since exp(U) * Z_wildcard == exp(U + log(Z_wildcard)) we can bring Z_wildcard into the u expression
                # Note: the top check (check_U_match) will not require Z_wildcard to be positive (just not 0)
                u = (find_U[U_wildcard] + log(find_U[Z_wildcard]))

                # Find the singularity point by solving u for V==0, excluding irrational results
                singularity_points = _solve_real(u, V)

                # Find Vmin, Vmax for the fix range
                Vmin_points = _solve_real(u - U_offset, V)
                Vmax_points = _solve_real(u + U_offset, V)
                for sp in singularity_points:
                    if len(Vmin_points) == len(Vmax_points) == 1:
                        Vmin, Vmax = tuple(Vmin_points)[0], tuple(Vmax_points)[0]
                    else:  # multiple solutions for Vmin / Vmax we don't know which to use, so revert to fixed range
                        Vmin, Vmax = sp - U_offset, sp + U_offset

                    for fp1 in fraction_part_1:  # Check arguments in numerator (or denominator)
                        match = fp1.match(P_wildcard * u)  # search for multiple of U
                        found_on_top = match is not None and P_wildcard in match and match[P_wildcard] != 0
                        if not found_on_top:
                            match = fp1.match(Z_wildcard * V - Z_wildcard * SP_wildcard)  # look for multiple of V - sp
                            found_on_top = check_U_match(match, sp)
                            if not found_on_top:
                                # search for a exp(multiple of V - sp)
                                match = fp1.match(exp_function(Z_wildcard * V - Z_wildcard * SP_wildcard))
                                found_on_top = check_U_match(match, sp)
                        if found_on_top:  # We've found a match stop looking in the other numerator arguments
                            break
                    if found_on_top:  # found singularity
                        # If this isngularity was previously found, adjust Vmin/Vmax to widest range
                        for sing in singularities:
                            if sing[2] == sp:
                                # if Vmin & Vmax are the same it's the same singularity
                                if Vmin == sing[0] and Vmax == sing[1]:
                                    break
                                elif all(len(v.free_symbols) == 0 for v in (Vmin, Vmax, sing[0], sing[1])):
                                    # if Vmin/Vmax have no variables we can pick min/max
                                    sing[0] = min(sing[0], sing[1], Vmin, Vmax)
                                    sing[1] = max(sing[0], sing[1], Vmin, Vmax)
                                    break
                        else:  # if it's a new singularity add to the list
                            singularities.append([Vmin, Vmax, sp])
        if not found_on_top:
            (Vmin, Vmax, sp) = None, None, None

    # Put dummies back in and return the singularity point and range boundries
    return [(_float_dummies(Vmin), _float_dummies(Vmax), _float_dummies(sp)) for (Vmin, Vmax, sp) in singularities]


def _fix_expr_parts(expr, V, U_offset, exp_function):
    """
    Removes fixable singularities and replaces them with piecewise expressions.

    :return: either (Vmin, Vmax, singularity point, expression, True)
                    if we have identified a singularity needs to be made but it can't be done yet.
                    For eample if 2 singularities with the same singularity point are added up,
                    a single singularity is constructed instead.
             or ``(None, None, None, fixed expr, fixed_expr has piecewise)`` otherwise

    see :meth:`remove_fixable_singularities for more details` "
    """

    if not expr.has(exp_function):  # Expressions without exp don't have removable-like equations
        return (None, None, None, expr, False)

    if isinstance(expr, Mul):  # 1 * A --> A (remove unneeded 1 *)
        expr = Mul(*[a for a in expr.args if not str(a) in ('1.0', '1')])

    # Turn Quantity dummies into numbers, to enable analysis
    subs_dict = {d: d.evalf(FLOAT_PRECISION) for d in expr.atoms(Quantity)}
    check_U_expr = expr.xreplace(subs_dict)

    if isinstance(expr, Add):  # A + B + ..
        # The expression is an addition, find singularities in each argument
        new_expr_parts = []
        for a in expr.args:
            new_expr_parts.append(_fix_expr_parts(a, V, U_offset, exp_function))

        # If all arguments have the same singularity point, return 1 singularity with the widest range
        range = [item for (Vmin, Vmax, _, _, _) in new_expr_parts for item in (Vmin, Vmax)]
        if len(new_expr_parts) > 1 and len(set([str(sp) for (_, _, sp, _, _) in new_expr_parts])) == 1 \
                and all(isinstance(b, Quantity) for b in range):
            sp = new_expr_parts[0][2]
            Vmin, Vmax = min(range, key=lambda v: float(str(v))), max(range, key=lambda v: float(str(v)))
            return (Vmin, Vmax, sp, expr, True)
        else:  # Singularity points differ, so create each piecewise seperately and add them up
            expr_parts = []
            is_piecewise = False
            for Vmin, Vmax, sp, ex, has_piecewise in new_expr_parts:
                is_piecewise = is_piecewise or has_piecewise or Vmin is not None
                expr_parts.append(_generate_piecewise(ex, V, sp, Vmin, Vmax) if sp is not None else ex)
            return (None, None, None, Add(*expr_parts), is_piecewise)

    elif isinstance(expr, Pow) and expr.args[1] == -1.0 and len(expr.args) == 2:  # 1/A
        # Find singularities in A and adjust result to represent 1 / A
        Vmin, Vmax, sp, ex, has_piecewise = _fix_expr_parts(expr.args[0], V, U_offset, exp_function)
        has_piecewise = has_piecewise or Vmin is not None
        return (None, None, None, ONE / (_generate_piecewise(ex, V, sp, Vmin, Vmax) if sp is not None else ex),
                has_piecewise)

    elif isinstance(expr, Mul):  # A * B * ...
        singularities = _get_singularity(check_U_expr, V, U_offset, exp_function)  # Find the singularity point
        range = [item for (Vmin, Vmax, _) in singularities for item in (Vmin, Vmax)]
        # If all arguments have the same singularity point, return 1 singularity with the widest range
        if len(singularities) == 1:
            Vmin, Vmax, sp = singularities[0]
            return (Vmin, Vmax, sp, expr, True)
        elif len(singularities) > 0:  # Singularity points differ, so generate a nested piecewise
            for Vmin, Vmax, sp in singularities[1:]:
                expr = _generate_piecewise(expr, V, sp, Vmin, Vmax)
            Vmin, Vmax, sp = singularities[0]
            return (Vmin, Vmax, sp, expr, True)
        else:  # Couldn't find singularity, try the expression's arguments
            expr_parts = []
            is_piecewise = False
            for sub_ex in expr.args:
                Vmin, Vmax, sp, ex, has_piecewise = _fix_expr_parts(sub_ex, V, U_offset, exp_function)
                has_piecewise = is_piecewise = is_piecewise or has_piecewise or Vmin is not None
                expr_parts.append(_generate_piecewise(ex, V, sp, Vmin, Vmax) if sp is not None else ex)
            return (None, None, None, Mul(*expr_parts), is_piecewise)

    else:  # Different type of equation e.g a number
        return (None, None, None, expr, False)


def _remove_singularities(expr, V, U_offset=1e-7, exp_function=exp):
    """
    Removes suitable singularities and replaces it with a piecewise.
    :param expr: The expression to analyse ().
    :param V: The voltage variable.
    :param U_offset: Determins the offset either side of U (see get_singularity) for which the fix is used.
    :param exp_function: The function representing exp.
    :return: (bool, expr) as follows: (expr has changed, expr with singularities fixes if appropriate).

    see :meth:`remove_fixable_singularities for more details `."
    """
    if not expr.has(exp_function):
        return False, expr
    (Vmin, Vmax, sp, ex, changed) = _fix_expr_parts(expr, V, U_offset, exp_function)
    ex = _generate_piecewise(ex, V, sp, Vmin, Vmax) if sp is not None else ex
    return changed or Vmin is not None, ex


def remove_fixable_singularities(model, V, modifiable_parameters, U_offset=1e-7, exp_function=exp):
    """
    Finds singularities in the removable-like equations in the model and replaces them with a piecewise.
    It finds singularities in equations of the form:
       - ``U / (exp(U) - 1.0)``
       - ``U / (1.0 - exp(U))``
       - ``(exp(U) - 1.0) / U``
       - ``(1.0 - exp(U)) / U``

    :param model: The bigmodelmanip model the model to analyse.
    :param V: The voltage variable.
    :param modifiable_parameters: The variables which are modifiable in the model,
            their defining equations are excluded form the analysis.
    :param U_offset: Determins the offset either side of U for which the fix is used.
    :param exp_function: the function representing exp.

    *Please Note:* As much as possible varaibles are substituted for their values during the analysis.
                   However state variables and parameters will not be substituted (as these can change)
                   If U contains parameters or state variables, the analysis assumes these to be positive
                   it is theoretically possible such variables mean a singularity is found where there is none.
                   E.g. if such variables make U evaluate to 0.
    """
    unprocessed_eqs = {}
    # iterate over sorted variables in the model
    for variable in nx.lexicographical_topological_sort(model.graph, key=str):

        # Get equation
        eq = model.graph.nodes[variable]['equation']

        # Skip variables that have no equation or equations defining parameters or where rhs is a Piecewise
        if eq is not None and not isinstance(eq.rhs, Piecewise) and eq.lhs not in modifiable_parameters:
            unprocessed_eqs[eq.lhs] = eq.rhs.xreplace(unprocessed_eqs)  # store partially evaluated version of the rhs
            changed, new_ex = _remove_singularities(unprocessed_eqs[eq.lhs], V, U_offset=U_offset,
                                                    exp_function=exp_function)
            if changed:  # update equation if the rhs has a singularity that can be fixed
                model.remove_equation(eq)
                model.add_equation(Eq(eq.lhs, new_ex))
                unprocessed_eqs.pop(eq.lhs)
