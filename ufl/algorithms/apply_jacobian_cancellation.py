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


from ufl.algebra import ScalarTensorProduct
from ufl.algorithms.map_integrands import map_integrands
from ufl.constantvalue import Identity
from ufl.core.terminal import Terminal
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.differentiation import ReferenceGrad
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.log import error
from ufl.referencevalue import ReferenceValue
from ufl.tensoralgebra import Dot
from ufl.tensors import ComponentTensor, ListTensor


class JacobianCancellation(MultiFunction):
    """Each handler returns a 5-tuple: the handled node with:
    - the node as it is, with any Jacobian cancellation having occurred.
    - a Jacobian cancelled on the left,
    - a Jacobian cancelled on the right,
    - an inverse Jacobian cancelled on the left, and
    - an inverse Jacobian cancelled on the right.
    For each of these, if there is no such thing, then None is returned."""
    def __init__(self):
        MultiFunction.__init__(self)

    def expr(self, o):
        error("Missing Jacobian cancellation handler for type {0}.".format(o._ufl_class_.__name__))

    def jacobian(self, o):
        dim, _ = o.ufl_shape
        return (o, Identity(dim), Identity(dim), None, None)

    def jacobian_inverse(self, o):
        dim, _ = o.ufl_shape
        return (o, None, None, Identity(dim), Identity(dim))

    def terminal(self, o):
        return (o, None, None, None, None)

    def reference_value(self, o):
        f, = o.ufl_operands
        if not f._ufl_is_terminal_:
            error("ReferenceValue can only wrap a terminal")
        return (o, None, None, None, None)

    def reference_grad(self, o):
        if not isinstance(o.ufl_operands[0],
                          (ReferenceGrad, ReferenceValue, Terminal)):
            error("Expecting only grads applied to a terminal.")
        return (o, None, None, None, None)

    def scalar_tensor_product(self, o, scalar_tuple, tensor_tuple):
        scalar, _, _, _, _ = scalar_tuple
        tensor, sansJLeft, sansJRight, sansKLeft, sansKRight = tensor_tuple
        return (ScalarTensorProduct(scalar, tensor),
                ScalarTensorProduct(scalar, sansJLeft) if sansJLeft else None,
                ScalarTensorProduct(scalar, sansJRight) if sansJRight else None,
                ScalarTensorProduct(scalar, sansKLeft) if sansKLeft else None,
                ScalarTensorProduct(scalar, sansKRight) if sansKRight else None)

    def dot(self, o, left_tuple, right_tuple):
        # _sjl means "sans Jacobian on the left", etc.
        left, left_sjl, left_sjr, left_skl, left_skr = left_tuple
        right, right_sjl, right_sjr, right_skl, right_skr = right_tuple
        transpose_allowed = o.ufl_shape == ()
        if left_sjr and right_skl:
            return (Dot(left_sjr, right_skl),
                    None, None, None, None)
        elif left_skr and right_sjl:
            return (Dot(left_skr, right_sjl),
                    None, None, None, None)
        elif transpose_allowed and left_sjl and right_skr:
            return (Dot(left_sjl, right_skr),
                    None, None, None, None)
        elif transpose_allowed and left_skl and right_sjr:
            return (Dot(left_skl, right_sjr),
                    None, None, None, None)
        else:
            return (Dot(left, right),
                    Dot(left_sjl, right) if left_sjl else None,
                    Dot(left, right_sjr) if right_sjr else None,
                    Dot(left_skl, right) if left_skl else None,
                    Dot(left, right_skr) if right_skr else None)

    def division(self, o):
        # Only divison of scalars is allowed.
        return (o, None, None, None, None)

    def inner(self, o):
        return (o, None, None, None, None)

    def sum(self, o, left_tuple, right_tuple):
        left, left_sjl, left_sjr, left_skl, left_skr = left_tuple
        right, right_sjl, right_sjr, right_skl, right_skr = right_tuple
        return (left + right,
                left_sjl + right_sjl if left_sjl and right_sjl else None,
                left_sjr + right_sjr if left_sjr and right_sjr else None,
                left_skl + right_skl if left_skl and right_skl else None,
                left_skr + right_skr if left_skr and right_skr else None)

    def product(self, o):
        return (o, None, None, None, None)

    def power(self, o, base_tuple, exponent_tuple):
        return (base_tuple[0]**exponent_tuple[0],
                None, None, None, None)

    def list_tensor(self, o, *op_tuples):
        return (ListTensor(*[op for (op, _, _, _, _) in op_tuples]),
                None, None, None, None)

    def component_tensor(self, o, expression_tuple, multiindex_tuple):
        expression, _, _, _, _ = expression_tuple
        multiindex, _, _, _, _ = multiindex_tuple
        return (ComponentTensor(expression, multiindex),
                None, None, None, None)

    def indexed(self, o, expression_tuple, multiindex_tuple):
        expression, _, _, _, _ = expression_tuple
        multiindex, _, _, _, _ = multiindex_tuple
        return (Indexed(expression, multiindex),
                None, None, None, None)

    def index_sum(self, o, expression_tuple, multiindex_tuple):
        expression, _, _, _, _ = expression_tuple
        multiindex, _, _, _, _ = multiindex_tuple
        return (IndexSum(expression, multiindex),
                None, None, None, None)

    def restricted(self, o, operand_tuple):
        return (operand_tuple[0](o._side),
                None, None, None, None)


def apply_jacobian_cancellation_in_expression(expression):
    """Here, the argument must be an expression."""
    rules = JacobianCancellation()
    result, _, _, _, _ = map_expr_dag(rules, expression)
    return result


def apply_jacobian_cancellation(expression):
    """ Here, the argument can be a form or integral too."""
    return map_integrands(apply_jacobian_cancellation_in_expression,
                          expression)
