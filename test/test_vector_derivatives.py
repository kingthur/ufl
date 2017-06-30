# Tests for coefficient derivatives of expressions involving vector
# operators (grad, inner, etc) without algebra lowering having been
# applied first.

from ufl import *
from ufl.algebra import Product, Sum
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.core.multiindex import MultiIndex, FixedIndex
from ufl.core.operator import Operator
from ufl.corealg.traversal import post_traversal
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.tensors import ComponentTensor
from itertools import izip_longest
import pytest

def equal_up_to_index_relabelling(e1, e2):
    """Comparison function used in later tests."""
    def equal_indices(i1, i2, relabelling):
        if i1.count() in relabelling:
            return relabelling[i1.count()] == i2.count()
        else:
            relabelling[i1.count()] = i2.count()
            return True

    def equal_index_bases(ib1, ib2, relabelling):
        if isinstance(ib1, FixedIndex):
            return (isinstance(ib2, FixedIndex)
                    and ib1._value == ib2._value)
        elif isinstance(ib1, Index):
            return (isinstance(ib2, Index)
                    and equal_indices(ib1, ib2, relabelling))
        else:
            raise NotImplementedError

    def equal_multi_indices(mi1, mi2, relabelling):
        return (len(mi1) == len(mi2) and
                all([equal_index_bases(i1, i2, relabelling)
                         for (i1, i2) in zip(mi1, mi2)]))

    def equal_subexpressions(sub1, sub2, relabelling):
        if type(sub1) != type(sub2):
            return False
        if isinstance(sub1, MultiIndex):
            return equal_multi_indices(sub1, sub2, relabelling)
        elif isinstance(sub1, Operator):
            if len(sub1.ufl_operands) != len(sub2.ufl_operands):
                return False
            # Before we can return, we must remove from relabelling any
            # indices that are "swallowed" by this operator.
            for index_in_subtree in list(relabelling):
                if index_in_subtree not in sub1.ufl_free_indices:
                    del relabelling[index_in_subtree]
            return True
        else: # Terminals (other than MultiIndices), Objects, or anything else.
            return sub1 == sub2

    relabelling = {}
    return all([equal_subexpressions(sub1, sub2, relabelling) for (sub1, sub2)
                in izip_longest(post_traversal(e1), post_traversal(e2),
                                fillvalue=object())])

class TestEqualUpToIndexRelabelling:
    def test_basic(self):
        cell = triangle
        element = VectorElement("Lagrange", cell, degree=1, dim=2)
        f = Coefficient(element)
        g = Coefficient(element)
        w = Argument(element, 0)

        f1 = f
        g1 = g

        assert equal_up_to_index_relabelling(f, f)
        assert not equal_up_to_index_relabelling(f, g)
        assert equal_up_to_index_relabelling(f, f1)
        assert equal_up_to_index_relabelling(w, w)
        assert not equal_up_to_index_relabelling(f, w)
        assert equal_up_to_index_relabelling(grad(f), grad(f))
        assert not equal_up_to_index_relabelling(grad(f), grad(g))
        assert equal_up_to_index_relabelling(dot(grad(f), grad(g)), dot(grad(f), grad(g)))
        assert not equal_up_to_index_relabelling(dot(grad(f), grad(g)), dot(grad(f), grad(f)))

        exp = dot(grad(f), grad(g))
        low1 = apply_algebra_lowering(exp)
        low2 = apply_algebra_lowering(exp) # Will use different indices.
        assert not equal_up_to_index_relabelling(exp, low1)
        assert not equal_up_to_index_relabelling(exp, low2)
        assert not low1 == low2 # Shows only that the equality test is non-trivial.
        assert equal_up_to_index_relabelling(low1, low2)

        # The first expression is dot(grad(f), grad(g)). This verifies
        # that we actually check down to the level of the indices, not
        # just stopping at any MultiIndex.
        i1, i2, i3 = Index(1), Index(2), Index(3)
        exp1 = ComponentTensor(IndexSum(Product(Indexed(grad(f), MultiIndex((i1, i2))),
                                                Indexed(grad(g), MultiIndex((i2, i3)))),
                                        MultiIndex((i2,))),
                               MultiIndex((i1, i3)))
        exp2 = ComponentTensor(IndexSum(Product(Indexed(grad(f), MultiIndex((i2, i3))),
                                                Indexed(grad(g), MultiIndex((i3, i1)))),
                                        MultiIndex((i3,))),
                               MultiIndex((i2, i1)))
        exp3 = ComponentTensor(IndexSum(Product(Indexed(grad(f), MultiIndex((i1, i2))),
                                                Indexed(grad(g), MultiIndex((i3, i2)))),
                                        MultiIndex((i2,))),
                               MultiIndex((i1, i3)))
        assert equal_up_to_index_relabelling(exp1, exp2)
        assert not equal_up_to_index_relabelling(exp1, exp3)

    def test_distinct_relabellings_between_subtrees(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, degree=1)
        f = Coefficient(element)
        g = Coefficient(element)
        w = Argument(element, 0)
        i1, i2, i3 = Index(1), Index(2), Index(3)

        exp1 = Sum(dot(grad(f), ComponentTensor(Product(Indexed(grad(g), MultiIndex((i1,))), w),
                                                MultiIndex((i1,)))),
                   dot(grad(w), ComponentTensor(Product(Indexed(grad(g), MultiIndex((i1,))), f),
                                                MultiIndex((i1,)))))

        exp2 = Sum(dot(grad(f), ComponentTensor(Product(Indexed(grad(g), MultiIndex((i3,))), w),
                                                MultiIndex((i3,)))),
                   dot(grad(w), ComponentTensor(Product(Indexed(grad(g), MultiIndex((i2,))), f),
                                                MultiIndex((i2,)))))

        assert equal_up_to_index_relabelling(exp1, exp2)


@pytest.fixture
def context():
    class Context:
        def return_values(self, element):
            f = Coefficient(element)
            g = Coefficient(element)
            w = Argument(element, 0)
            return f, g, w, element
        def scalar(self):
            cell = triangle
            element = FiniteElement("Lagrange", cell, degree=1)
            return self.return_values(element)
        def vector(self, dim):
            cell = triangle
            element = VectorElement("Lagrange", cell, degree=1, dim=dim)
            return self.return_values(element)
    return Context()

class TestDotDerivative:
    def testLeftSimple(self, context):
        f, g, w, element = context.vector(dim=2)
        baseExpression = dot(f, g)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = dot(w, g)
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testRightSimple(self, context):
        f, g, w, element = context.vector(dim=3)
        baseExpression = dot(f, g)
        result = apply_derivatives(derivative(baseExpression, g, w))
        expectedResult = dot(f, w)
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testBothSimple(self, context):
        f, g, w, element = context.vector(dim=3)
        baseExpression = dot(f, f)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = dot(w, f) + dot(f, w)
        # Note that the expectedResult is not reduced to 2 * dot(w, f),
        # which it could be if the field of scalars is the reals.
        assert equal_up_to_index_relabelling(result, expectedResult)
