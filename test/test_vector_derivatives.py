# Tests for coefficient derivatives of expressions involving vector
# operators (grad, inner, etc) without algebra lowering having been
# applied first.

from ufl import *
from ufl.algebra import Product, Sum
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.algorithms.expand_indices import expand_indices
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
        def scalar(self, cell=triangle):
            element = FiniteElement("Lagrange", cell, degree=1)
            return self.return_values(element)
        def vector(self, dim, cell=triangle):
            element = VectorElement("Lagrange", cell, degree=1, dim=dim)
            return self.return_values(element)
        def tensor(self, dim1, dim2, cell=triangle):
            element = TensorElement("Lagrange", cell, degree=1, shape=(dim1,dim2))
            return self.return_values(element)
    return Context()

class TestCoefficientDerivativeOfDot:
    def test_left_simple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=3)
        base_expression = dot(f, g)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = dot(w, g)
        assert equal_up_to_index_relabelling(actual, expected)

    def test_right_simple(self, context):
        f, g, w, element = context.vector(dim=4)
        base_expression = dot(f, g)
        actual = apply_derivatives(derivative(base_expression, g, w))
        expected = dot(f, w)
        assert equal_up_to_index_relabelling(actual, expected)

    def test_both_simple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=3)
        base_expression = dot(f, f)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = dot(w, f) + dot(f, w) # NB: not 2*dot(w, f).
        assert equal_up_to_index_relabelling(actual, expected)

class TestCoefficientDerivativeOfInner:
    def test_left_simple(self, context):
        f, g, w, element = context.tensor(dim1=2, dim2=3)
        base_expression = inner(f, g)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = inner(w, g)
        assert equal_up_to_index_relabelling(actual, expected)

    def test_right_simple(self, context):
        f, g, w, element = context.vector(dim=4)
        base_expression = inner(f, g)
        actual = apply_derivatives(derivative(base_expression, g, w))
        expected = inner(f, w)
        assert equal_up_to_index_relabelling(actual, expected)

    def test_both_simple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=2)
        base_expression = inner(f, f)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = inner(w, f) + inner(f, w) # NB: not 2*inner(w, f).
        assert equal_up_to_index_relabelling(actual, expected)

class TestCoefficientDerivativeOfGrad:
    def test_simple(self, context):
        f, g, w, element = context.vector(dim=3)
        base_expression = grad(f)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = grad(w)
        assert equal_up_to_index_relabelling(actual, expected)

class TestCoefficientDerivativeOfDiv:
    def test_simple_2D(self, context):
        f, g, w, element = context.vector(dim=2)
        base_expression = div(f)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = div(w)
        assert equal_up_to_index_relabelling(actual, expected)

    def test_simple_3D(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        base_expression = div(f)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = div(w)
        assert equal_up_to_index_relabelling(actual, expected)

class TestCoefficientDerivativeOfCurl:
    def test_simple(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        base_expression = curl(f)
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = curl(w)
        assert equal_up_to_index_relabelling(actual, expected)

class TestGradientOfDot:
    def test_simple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=3, cell=triangle)
        base_expression = dot(f, g)
        actual = apply_derivatives(grad(base_expression))
        # We test only that this call actually succeeds.

class TestGradientOfInner:
    def test_simple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=2)
        base_expression = inner(f, g)
        actual = apply_derivatives(grad(base_expression))
        # We test only that this call actually succeeds.

class TestCombined:
    def test_dot_grad(self, context):
        f, g, w, element = context.scalar()
        base_expression = dot(grad(f), grad(g))
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = dot(grad(w), grad(g))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_dot_grad_multiply(self, context):
        f, g, w, element = context.scalar()
        base_expression = dot(grad(f), f * grad(g))
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = dot(grad(w), f * grad(g)) + dot(grad(f), w * grad(g))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_specified_coefficient_derivatives(self, context):
        f, g, w, element = context.scalar()
        base_expression = dot(grad(f), grad(g))
        h = Coefficient(element)
        df = Coefficient(element)
        dg = Coefficient(element)
        actual = apply_derivatives(derivative(base_expression, h, w, {f: df, g:dg}))
        # The expected result is really
        # dot(grad(w)*df + w*grad(df), grad(g))
        # + dot(grad(f), grad(w)*dg + w*grad(dg))
        # But, when expanded, this gives something involving sums of
        # component tensors, while the actual result has component
        # tensors of sums. Accordingly, we have to expand this
        # ourselves.
        expected = (
            dot(ComponentTensor(Indexed(grad(w), MultiIndex((i,)))*df
                                + w*Indexed(grad(df), MultiIndex((i,))),
                                MultiIndex((i,))),
                grad(g))
            + dot(grad(f),
                  ComponentTensor(Indexed(grad(w), MultiIndex((i,)))*dg
                                  + w*Indexed(grad(dg), MultiIndex((i,))),
                                  MultiIndex((i,))))
            )
        assert equal_up_to_index_relabelling(actual, expected)

    def test_inner_grad(self, context):
        f, g, w, element = context.tensor(dim1=2, dim2=3)
        base_expression = inner(grad(f), grad(g))
        actual = apply_derivatives(derivative(base_expression, f, w))
        expected = inner(grad(w), grad(g))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_component_derivative(self):
        # Simpler test of a use-case shown first by
        # test_derivative::test_segregated_derivative_of_convection.
        cell = triangle
        V = FiniteElement("CG", cell, 1)
        W = VectorElement("CG", cell, 1, dim=2)
        u = Coefficient(W)
        v = Coefficient(W)
        du = TrialFunction(V)

        L = inner(grad(u), grad(v))
        dL = derivative(L, u[0], du)
        form = dL * dx
        fd = compute_form_data(form)
        pf = fd.preprocessed_form
        a = expand_indices(pf) # This test exists to ensure that this call does not cause an exception.

    def test_derivative_wrt_tuple(self):
        cell = triangle
        scalar_element = FiniteElement("CG", cell, degree=1)
        f = Coefficient(scalar_element)
        g = Coefficient(scalar_element)
        vector_element = VectorElement("CG", cell, degree=1, dim=2)
        w = Argument(vector_element, 0)
        base_expression = dot(grad(f), grad(g))
        actual = apply_derivatives(derivative(base_expression, (f, g), w))
        expected = dot(grad(w[0]), grad(g)) + dot(grad(f), grad(w[1]))
        expected = apply_derivatives(expected) # Push grads inside indexing.
        assert equal_up_to_index_relabelling(actual, expected)

    def test_grad_spatial_coordinate_shape(self, context):
        # Checks that the geometric dimension assigned to GradRuleset
        # in DerivativeRuleDispatcher is correct.
        cell = tetrahedron
        element = VectorElement("CG", cell, degree=1, dim=2)
        x = SpatialCoordinate(element.cell()) # shape: (3,)
        y = Coefficient(element) # (2,)
        w = grad(grad(y)) # (2, 3, 3)
        z = dot(w, x) # (2, 3)
        expr = grad(z) # (2, 3, 3)
        assert expr.ufl_shape == (2, 3, 3)
        expr = apply_derivatives(expr)
        assert expr.ufl_shape == (2, 3, 3)

    def test_nabla_grad_spatial_coordinate_shape(self, context):
        # Checks that the geometric dimension assigned to GradRuleset
        # in DerivativeRuleDispatcher is correct.
        cell = tetrahedron
        element = VectorElement("CG", cell, degree=1, dim=2)
        x = SpatialCoordinate(element.cell()) # shape: (3,)
        y = Coefficient(element) # (2,)
        w = nabla_grad(nabla_grad(y)) # (3, 3, 2)
        z = dot(x, w) # (3, 2)
        expr = nabla_grad(z) # (3, 3, 2)
        assert expr.ufl_shape == (3, 3, 2)
        expr = apply_derivatives(expr)
        assert expr.ufl_shape == (3, 3, 2)
