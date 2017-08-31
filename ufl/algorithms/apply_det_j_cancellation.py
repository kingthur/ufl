# -*- coding: utf-8 -*-
"""This module contains the apply_jacobian_cancellation algorithm
which removes any redundant pairs of a Jacobian matrix and its
inverse."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.


from functools import partial
from ufl.algebra import Abs, Division, Product, ScalarTensorProduct, Sum
from ufl.algorithms.map_integrands import map_integrands
from ufl.conditional import Conditional, LT
from ufl.constantvalue import FloatValue, Zero
from ufl.core.terminal import Terminal
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.differentiation import ReferenceGrad
from ufl.geometry import JacobianDeterminant
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.log import error
from ufl.referencevalue import ReferenceValue
from ufl.tensoralgebra import Dot, Inner
from ufl.tensors import ComponentTensor, ListTensor


class JacobianDeterminantCancellation(MultiFunction):
    """Each handler returns a 4-tuple: the handled node with:
    - the node as it is, with any determinant cancellation having occurred.
    - the node with a factor of detJ removed.
    - the node with a factor of 1/detJ removed.
    For each of these, if there is no such thing, then None is used instead."""
    def __init__(self, domain):
        MultiFunction.__init__(self)
        self.domain = domain

    def expr(self, o):
        error("Missing Jacobian determinant cancellation handler for type {0}.".format(o._ufl_class_.__name__))

    def jacobian_determinant(self, o):
        if o.ufl_domain() != self.domain:
            error("Distinct Jacobian determinants in expression.")
        return (o, FloatValue(1.0), None)

    def division(self, o, num_tuple, denom_tuple):
        (num, num_sdj, num_s1odj) = num_tuple
        (denom, denom_sdj, denom_s1odj) = denom_tuple
        if num_sdj and denom_sdj:
            return (Division(num_sdj, denom_sdj),
                    None, None)
        elif num_s1odj and denom_s1odj:
            return (Division(num_s1odj, denom_s1odj),
                    None, None)
        else:
            def combo(num, num_sans, denom, denom_sans_opp):
                if num_sans:
                    return Division(num_sans, denom)
                elif denom_sans_opp:
                    return Division(num, denom_sans_opp)
                else:
                    return None
            return (Division(num, denom),
                    combo(num, num_sdj, denom, denom_s1odj),
                    combo(num, num_s1odj, denom, denom_sdj))

    def general_product(self, o, left_tuple, right_tuple, constructor):
        # "sdj" means "sans detJ".
        (left, left_sdj, left_s1odj) = left_tuple
        (right, right_sdj, right_s1odj) = right_tuple
        if left_sdj and right_s1odj:
            return (constructor(left_sdj, right_s1odj),
                    None, None)
        elif left_s1odj and right_sdj:
            return (constructor(left_s1odj, right_sdj),
                    None, None)
        else:
            def combo(left, left_sans, right, right_sans):
                if left_sans:
                    return constructor(left_sans, right)
                elif right_sans:
                    return constructor(left, right_sans)
                else:
                    return None
            return (constructor(left, right),
                    combo(left, left_sdj, right, right_sdj),
                    combo(left, left_s1odj, right, right_s1odj))

    def product(self, o, left_tuple, right_tuple):
        return self.general_product(o, left_tuple, right_tuple, Product)

    # def product(self, o, left_tuple, right_tuple):
    #     # Use Product instead of * to avoid unnecessary shape checks.
    #     # "sdj" means "sans detJ".
    #     (left, left_sdj, left_s1odj) = left_tuple
    #     (right, right_sdj, right_s1odj) = right_tuple
    #     if left_sdj and right_s1odj:
    #         return (Product(left_sdj, right_s1odj),
    #                 None, None)
    #     elif left_s1odj and right_sdj:
    #         return (Product(left_s1odj, right_sdj),
    #                 None, None)
    #     else:
    #         def combo(left, left_sans, right, right_sans):
    #             if left_sans:
    #                 return Product(left_sans, right)
    #             elif right_sans:
    #                 return Product(left, right_sans)
    #             else:
    #                 return None
    #         return (Product(left, right),
    #                 combo(left, left_sdj, right, right_sdj),
    #                 combo(left, left_s1odj, right, right_s1odj))

    def scalar_tensor_product(self, o, left_tuple, right_tuple):
        return self.general_product(o, left_tuple, right_tuple, ScalarTensorProduct)

    def dot(self, o, left_tuple, right_tuple):
        return self.general_product(o, left_tuple, right_tuple, Dot)

    def inner(self, o, left_tuple, right_tuple):
        return self.general_product(o, left_tuple, right_tuple, Inner)

    def sum(self, o, left_tuple, right_tuple):
        (left, left_sdj, left_s1odj) = left_tuple
        (right, right_sdj, right_s1odj) = right_tuple
        return (Sum(left, right),
                Sum(left_sdj, right_sdj) if left_sdj and right_sdj else None,
                Sum(left_s1odj, right_s1odj) if left_s1odj and right_s1odj else None)

    def power(self, o, base_tuple, exponent_tuple):
        return (base_tuple[0]**exponent_tuple[0],
                None, None)

    def abs(self, o, arg_tuple):
        (arg, arg_sdj, arg_s1odj) = arg_tuple
        s = Conditional(LT(JacobianDeterminant(self.domain),
                           Zero((), (), ())),
                        FloatValue(-1.0), FloatValue(1.0))
        return (Abs(arg),
                Abs(arg_sdj) * s if arg_sdj else None,
                Abs(arg_s1odj) * s if arg_s1odj else None)

    def jacobian(self, o):
        return (o, None, None)

    def jacobian_inverse(self, o):
        return (o, None, None)

    def terminal(self, o):
        return (o, None, None)

    def reference_value(self, o):
        f, = o.ufl_operands
        if not f._ufl_is_terminal_:
            error("ReferenceValue can only wrap a terminal")
        return (o, None, None)

    def reference_grad(self, o):
        if not isinstance(o.ufl_operands[0],
                          (ReferenceGrad, ReferenceValue, Terminal)):
            error("Expecting only grads applied to a terminal.")
        return (o, None, None)

    def reference_div(self, o):
        if not isinstance(o.ufl_operands[0],
                          (ReferenceGrad, ReferenceValue)):
            error("Expecting ReferenceDiv applied to ReferenceGrad or ReferenceValue.")
        return (o, None, None)

    def reference_curl(self, o):
        if not isinstance(o.ufl_operands[0], ReferenceValue):
            error("Expecting ReferenceCurl applied to ReferenceValue.")
        return (o, None, None)

    def list_tensor(self, o, *op_tuples):
        op_list = [op for (op, _, _) in op_tuples]
        sdj_list = [sdj for (_, sdj, _) in op_tuples]
        s1odj_list = [s1odj for (_, _, s1odj) in op_tuples]
        return (ListTensor(*op_list),
                ListTensor(*sdj_list) if all(sdj_list) else None,
                ListTensor(*s1odj_list) if all(s1odj_list) else None)

    def index_related(self, o, expression_tuple, multiindex_tuple, constructor):
        expr, expr_sdj, expr_s1odj = expression_tuple
        multiindex, _, _ = multiindex_tuple
        return (constructor(expr, multiindex),
                constructor(expr_sdj, multiindex) if expr_sdj else None,
                constructor(expr_s1odj, multiindex) if expr_s1odj else None)

    def component_tensor(self, o, expression_tuple, multiindex_tuple):
        return self.index_related(o, expression_tuple, multiindex_tuple, ComponentTensor)

    def indexed(self, o, expression_tuple, multiindex_tuple):
        return self.index_related(o, expression_tuple, multiindex_tuple, Indexed)

    def index_sum(self, o, expression_tuple, multiindex_tuple):
        return self.index_related(o, expression_tuple, multiindex_tuple, IndexSum)

    def restricted(self, o, operand_tuple):
        return (operand_tuple[0](o._side), None, None)


def apply_det_j_cancellation_in_expression(expression):
    """Here, the argument must be an expression."""
    rules = JacobianDeterminantCancellation(expression.ufl_domain())
    result, _, _ = map_expr_dag(rules, expression)
    return result


def apply_det_j_cancellation(expression):
    """ Here, the argument can be a form or integral too."""
    # GTODO: Assumes that the sign of the Jacobian does not change in the cell?
    return map_integrands(apply_det_j_cancellation_in_expression,
                          expression)
