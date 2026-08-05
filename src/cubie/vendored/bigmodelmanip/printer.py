#
# Printer class to convert Sympy equations to strings (e.g. Python code).
#
# ----------------------------------------------------------------------------
#
# Parts of this code were adapted from:
#
#  https://github.com/sympy/sympy/blob/master/sympy/printing/printer.py
#  https://github.com/sympy/sympy/blob/master/sympy/printing/str.py
#
# Which came with the following license:
#
# Copyright (c) 2006-2018 SymPy Development Team
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of SymPy nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#
import sympy
import sympy.printing
from sympy.codegen.rewriting import ReplaceOptim, optimize
from sympy.core.mul import _keep_coeff
from sympy.logic.boolalg import BooleanTrue
from sympy.printing.precedence import precedence


class Printer(sympy.printing.printer.Printer):
    """
    Converts Sympy expressions to strings of Python code.

    To use, create a :class:`Printer` instance, and call its method :meth:`doprint()` with a Sympy expression as
    argument.

    To extend this to other languages, create a :class:`Printer` subclass, and override ``_function_names``,
    ``_literal_names``, or any of the ``_print_X`` methods as necessary.

    :param symbol_function: A function that converts ``sympy.Symbols`` to strings (variable names).
    :param derivative_function: A function that converts derivatives to strings.
    """
    # Dictionary mapping Sympy function names to string function names to output.
    _function_names = {
        'Abs': 'abs',
        'acos': 'math.acos',
        'acosh': 'math.acosh',
        'asin': 'math.asin',
        'asinh': 'math.asinh',
        'atan': 'math.atan',
        'atan2': 'math.atan2',
        'atanh': 'math.atanh',
        'ceiling': 'math.ceil',
        'cos': 'math.cos',
        'cosh': 'math.cosh',
        'exp': 'math.exp',
        'expm1': 'math.expm1',
        'factorial': 'math.factorial',
        'floor': 'math.floor',
        'log': 'math.log',
        'log10': 'math.log10',
        'log1p': 'math.log1p',
        'log2': 'math.log2',
        'sin': 'math.sin',
        'sinh': 'math.sinh',
        'sqrt': 'math.sqrt',
        'tan': 'math.tan',
        'tanh': 'math.tanh'
    }

    # Extra trig functio to be rewritten as other trig functions
    W = sympy.Wild('W')
    _extra_trig = {
        sympy.sec(W): (1 / sympy.cos(W)),
        sympy.csc(W): (1 / sympy.sin(W)),
        sympy.cot(W): (1 / sympy.tan(W)),
        sympy.sech(W): (1 / sympy.cosh(W)),
        sympy.csch(W): (1 / sympy.sinh(W)),
        sympy.coth(W): (1 / sympy.tanh(W)),

        sympy.asec(W): sympy.acos(1 / W),
        sympy.acsc(W): sympy.asin(1 / W),
        sympy.acot(W): sympy.atan(1 / W),
        sympy.asech(W): sympy.acosh(1 / W),
        sympy.acsch(W): sympy.asinh(1 / W),
        sympy.acoth(W): sympy.atanh(1 / W),
    }

    # The optimisations to apply to an expr to rewrite extra trig functions
    _optims = [ReplaceOptim(k, v) for k, v in _extra_trig.items()]

    # Dictionary mapping Sympy literals to strings for output.
    _literal_names = {
        'e': 'math.e',
        'nan': 'float(\'nan\')',
        'pi': 'math.pi',
    }

    def __init__(self, symbol_function=None, derivative_function=None):
        super(Printer, self).__init__(None)

        # Symbol and derivative handling (default)
        if symbol_function is None:
            self._symbol_function = lambda x: str(x)
        else:
            self._symbol_function = symbol_function

        if derivative_function is None:
            self._derivative_function = lambda x: str(x)
        else:
            self._derivative_function = derivative_function

    def doprint(self, expr):
        """Returns printer's representation for expr (as a string)"""
        if isinstance(expr, sympy.Expr):
            expr = optimize(expr, self._optims)
        return super().doprint(expr)

    def _bracket(self, expr, parent_precedence):
        """
        Converts ``expr`` to string, and adds parentheses around the result, if and only if
        ``precedence(expr) < parent_precedence``.
        """
        expr_prec = precedence(expr)
        parent_prec = parent_precedence
        # Some equations are substituted for expr. of lower precedence
        # For example x**-1 is printed as 1/x. An example where this would give na issue is 2**cos(x)**-1
        # Which should print as 2**(1 / math.cos(x))
        # Adjust precedence to put brackets around 1/x if necessary
        if isinstance(expr, sympy.Pow) and expr.is_commutative and \
                (-expr.exp is sympy.S.Half or -expr.exp is sympy.S.One):
            expr_prec -= 1

        if expr_prec < parent_prec:
            return '(' + self._print(expr) + ')'
        return self._print(expr)

    def _bracket_args(self, args, parent_precedence):
        """
        Applies :meth:`_bracket()` to a list of expressions, and joins them with a comma.
        """
        return ', '.join([self._bracket(x, parent_precedence) for x in args])

    def emptyPrinter(self, expr):
        """
        Called by :class:`Printer` as a last resort for unknown expressions.
        """
        raise ValueError(
            'Unsupported expression type (' + str(type(expr)) + '): '
            + str(expr))

    def _print_Add(self, expr):
        """ Handles addition & subtraction, with n terms. """
        # This method is based on sympy.printing.Str

        parts = []
        my_prec = precedence(expr)
        for term in expr.args:
            # Don't use _bracket() here because we want to check the sign
            t = self._print(term)

            # Add sign
            s = '+'
            if t.startswith('-'):
                s = '-'
                t = t[1:]
            parts.append(s)

            # Add remaining term
            parts.append('(' + t + ')' if precedence(term) < my_prec else t)

        # Concatenate and return
        if parts[0] == '+':
            # Ignore leading plus
            return ' '.join(parts[1:])
        else:
            # No space after first minus
            return parts[0] + ' '.join(parts[1:])

    def _print_And(self, expr):
        """ Handles logical and. """
        my_prec = precedence(expr)
        return ' and '.join([self._bracket(x, my_prec) for x in expr.args])

    def _print_bool(self, expr):
        """ Handles Python ``bool``s. """
        if expr:
            return self._print_BooleanTrue(expr)
        else:
            return self._print_BooleanFalse(expr)

    def _print_BooleanFalse(self, expr):
        """ Handles Sympy ``False``. """
        return 'False'

    def _print_BooleanTrue(self, expr):
        """ Handles Sympy ``True``. """
        return 'True'

    def _print_Derivative(self, expr):
        """ Handles Derivative objects. """
        return self._derivative_function(expr)

    def _print_Exp1(self, expr):
        """ Handles the Sympy ``E`` object. """
        return self._literal_names['e']

    def _print_float(self, expr):
        """ Handles Python ``float``s. """
        return str(expr)

    def _print_Float(self, expr):
        """ Handles Sympy Float objects. """
        return self._print_float(float(expr))

    def _print_Function(self, expr):
        """ Handles function calls. """

        # Check if function is known to python math
        name = expr.func.__name__
        # Convert arguments
        args = self._bracket_args(expr.args, 0)

        # Normal function
        func = self._function_names.get(name, None)
        if func is not None:
            return func + '(' + args + ')'

        # Unknown function
        raise ValueError('Unsupported function: ' + str(name))

    def _print_int(self, expr):
        """ Handles python ``int``s. """
        return str(expr)

    def _print_Integer(self, expr):
        """
        Handles Sympy Integer objects, including special ones like Zero, One, and NegativeOne.
        """
        return str(expr.p)

    def _print_Mul(self, expr):
        """
        Handles multiplication & division, with n terms.

        Division is specified as a power: ``x / y --> x * y**-1``.
        Subtraction is specified as ``x - y --> x + (-1 * y)``.
        """
        # This method is mostly copied from sympy.printing.Str

        # Check overall sign of multiplication
        sign = ''
        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = '-'

        # Collect all pows with more than one base element and exp = -1
        pow_brackets = []

        # Gather terms for numerator and denominator
        a, b = [], []
        for item in sympy.Mul.make_args(expr):

            # Check if this is a negative power that we can write as a division
            negative_power = (
                item.is_commutative and item.is_Pow
                and item.exp.is_Rational and item.exp.is_negative)
            if negative_power:
                if item.exp != -1:
                    # E.g. x * y**(-2 / 3) --> x / y**(2 / 3)
                    # Add as power
                    b.append(sympy.Pow(item.base, -item.exp, evaluate=False))
                else:
                    # Add without power
                    b.append(sympy.Pow(item.base, -item.exp))

                    # Check if it's a negative power that needs brackets
                    # Sympy issue #14160
                    if (len(item.args[0].args) != 1
                            and isinstance(item.base, sympy.Mul)):
                        pow_brackets.append(item)

            # Split Rationals over a and b, ignoring any 1s
            elif item.is_Rational:
                if item.p != 1:
                    a.append(sympy.Rational(item.p))
                if item.q != 1:
                    b.append(sympy.Rational(item.q))

            else:
                a.append(item)

        # Replace empty numerator with one
        a = a or [sympy.S.One]

        # Convert terms to code
        my_prec = precedence(expr)
        a_str = [self._bracket(x, my_prec) for x in a]
        b_str = [self._bracket(x, my_prec) for x in b]

        # Fix brackets for Pow with exp -1 with more than one Symbol
        for item in pow_brackets:
            assert item.base in b, "item.base should be kept in b for powers"
            b_str[b.index(item.base)] = \
                '(' + b_str[b.index(item.base)] + ')'

        # Combine numerator and denomenator and return
        a_str = sign + ' * '.join(a_str)
        if len(b) == 0:
            return a_str
        b_str = ' * '.join(b_str)
        return a_str + ' / ' + (b_str if len(b) == 1 else '(' + b_str + ')')

    # def _print_NaN(self, expr):
    #    return 'float(\'nan\')'

    # def _print_NegativeInfinity(self, expr):
    #    return 'float(\'-inf\')'

    def _print_Or(self, expr):
        """ Handles logical Or. """
        my_prec = precedence(expr)
        return ' or '.join([self._bracket(x, my_prec) for x in expr.args])

    def _print_Pi(self, expr):
        """ Handles pi. """
        return self._literal_names['pi']

    def _print_ternary(self, cond, expr):
        parts = ''
        parts += '('
        parts += self._print(expr)
        parts += ') if ('
        parts += self._print(cond)
        parts += ') else ('
        return parts

    def _print_Piecewise(self, expr):
        """
        Handles Piecewise functions.

        Sympy's piecewise is defined as a list of tuples ``(expr, cond)`` and evaluated by returning the first ``expr``
        whose ``cond`` is true. If none of the conditions hold a value error is raised.
        """
        # Assign NaN if no conditions hold
        # If a condition `True` is found, use its expression instead
        other = self._literal_names['nan']

        parts = '('
        brackets = 1
        for e, c in expr.args:  # pragma: no branch
            # Note that Sympy filters BooleanFalse out as well as clauses after a true clause
            if isinstance(c, BooleanTrue):
                other = self._print(e)
                break
            # Add e-if-c-else-? statement
            parts += self._print_ternary(c, e)
            brackets += 1
        parts += other
        parts += ')' * brackets
        return parts

    def _print_ordinary_pow(self, expr):
        """ Handles Pow(), hanles just ordinary powers without division. """
        p = precedence(expr)
        return self._bracket(expr.base, p) + '**' + self._bracket(expr.exp, p)

    def _print_Pow(self, expr):
        """ Handles Pow(), which includes all division. """
        p = precedence(expr)

        # Handle square root
        if expr.exp is sympy.S.Half:
            return self._function_names['sqrt'] + '(' + self._print(expr.base) + ')'

        # Division, only if commutative (following sympy implementation)
        if expr.is_commutative:
            # 1 / sqrt()
            if -expr.exp is sympy.S.Half:
                return (
                    '1 / ' + self._function_names['sqrt'] + '(' + self._print(expr.base) + ')')

            # Ordinary division
            if -expr.exp is sympy.S.One:
                return '1 / ' + self._bracket(expr.base, p)

        # Ordinary power
        return self._print_ordinary_pow(expr)

    def _print_Rational(self, expr):
        """ Handles rationals (int divisions, stored symbollicaly). """
        return str(expr.p) + ' / ' + str(expr.q)

    def _print_Relational(self, expr):
        """ Handles equality and inequalities. """

        op = expr.rel_op
        ops = {'==', '!=', '<', '<=', '>', '>='}
        if op not in ops:   # pragma: no cover
            raise ValueError('Unsupported relational: "' + str(op) + '".')

        # Note: Nested relationals (x == (y == z)) should get brackets, so
        # using slightly increased parent precedence here
        my_prec = precedence(expr) + 1
        lhs = self._bracket(expr.lhs, my_prec)
        rhs = self._bracket(expr.rhs, my_prec)
        return lhs + ' ' + op + ' ' + rhs

    def _print_Symbol(self, expr):
        """ Handles Sympy Symbol objects. """
        return self._symbol_function(expr)
