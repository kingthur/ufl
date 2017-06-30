# Tests for coefficient derivatives of expressions involving vector
# operators (grad, inner, etc) without algebra lowering having been
# applied first.

from ufl import *
from ufl.algorithms.apply_derivatives import apply_derivatives
from equal_up_to_index_relabelling import equal_up_to_index_relabelling
import pytest


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
        def tensor(self, dim1, dim2):
            cell = triangle
            element = TensorElement("Lagrange", cell, degree=1, shape=(dim1,dim2))
            return self.return_values(element)
    return Context()

class TestDotDerivative:
    def testLeftSimple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=3)
        baseExpression = dot(f, g)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = dot(w, g)
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testRightSimple(self, context):
        f, g, w, element = context.vector(dim=4)
        baseExpression = dot(f, g)
        result = apply_derivatives(derivative(baseExpression, g, w))
        expectedResult = dot(f, w)
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testBothSimple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=3)
        baseExpression = dot(f, f)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = dot(w, f) + dot(f, w) # NB: not 2*dot(w, f).
        assert equal_up_to_index_relabelling(result, expectedResult)

class TestInnerDerivative:
    def testLeftSimple(self, context):
        f, g, w, element = context.tensor(dim1=2, dim2=3)
        baseExpression = inner(f, g)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = inner(w, g)
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testRightSimple(self, context):
        f, g, w, element = context.vector(dim=4)
        baseExpression = inner(f, g)
        result = apply_derivatives(derivative(baseExpression, g, w))
        expectedResult = inner(f, w)
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testBothSimple(self, context):
        f, g, w, element = context.tensor(dim1=3, dim2=2)
        baseExpression = inner(f, f)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = inner(w, f) + inner(f, w) # NB: not 2*inner(w, f).
        assert equal_up_to_index_relabelling(result, expectedResult)

class TestGradDerivative:
    def testSimple(self, context):
        f, g, w, element = context.vector(dim=3)
        baseExpression = grad(f)
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = grad(w)
        assert equal_up_to_index_relabelling(result, expectedResult)

class TestCombined:
    def testDotGrad(self, context):
        f, g, w, element = context.scalar()
        baseExpression = dot(grad(f), grad(g))
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = dot(grad(w), grad(g))
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testDotGradMultiply(self, context):
        f, g, w, element = context.scalar()
        baseExpression = dot(grad(f), f * grad(g))
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = dot(grad(w), f * grad(g)) + dot(grad(f), w * grad(g))
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testSpecifiedCoefficientDerivatives(self, context):
        f, g, w, element = context.scalar()
        baseExpression = dot(grad(f), grad(g))
        h = Coefficient(element)
        df = Coefficient(element)
        dg = Coefficient(element)
        result = apply_derivatives(derivative(baseExpression, h, w, {f: df, g:dg}))
        expectedResult = dot(grad(w*df), grad(g)) + dot(grad(f), grad(w*dg))
        assert equal_up_to_index_relabelling(result, expectedResult)

    def testInnerGrad(self, context):
        f, g, w, element = context.tensor(dim1=2, dim2=3)
        baseExpression = inner(grad(f), grad(g))
        result = apply_derivatives(derivative(baseExpression, f, w))
        expectedResult = inner(grad(w), grad(g))
        assert equal_up_to_index_relabelling(result, expectedResult)
