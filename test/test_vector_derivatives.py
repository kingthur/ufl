# Tests for coefficient derivatives of expressions involving vector
# operators (grad, inner, etc) without algebra lowering having been
# applied first.

from ufl import *
from ufl.algebra import Product, Sum, Division
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering, apply_minimal_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks, ref_div_ito_ref_grad, ref_curl_ito_ref_grad
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.apply_jacobian_cancellation import apply_jacobian_cancellation
from ufl.conditional import Conditional, LT
from ufl.constantvalue import FloatValue, IntValue, Zero
from ufl.core.multiindex import MultiIndex, FixedIndex
from ufl.core.operator import Operator
from ufl.corealg.traversal import post_traversal
from ufl.differentiation import ReferenceGrad
from ufl.geometry import QuadratureWeight
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.referencevalue import ReferenceValue
from ufl.tensors import ComponentTensor, ListTensor
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
        def rt(self):
            cell = tetrahedron
            element = FiniteElement("RT", cell, degree=1)
            return self.return_values(element)
        def n1curl(self):
            cell = tetrahedron
            element = FiniteElement("N1curl", cell, degree=1)
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

class TestDiv:
    def test_list_tensor_of_scalars(self, context):
        f, g, w, element = context.scalar()
        base_expression = as_tensor([f, g])
        actual = apply_derivatives(div(base_expression))
        expected = as_tensor([grad(f), grad(g)])[i, i]
        assert equal_up_to_index_relabelling(actual, expected)

    def test_list_tensor_of_vectors(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        base_expression = as_tensor([f, g])
        assert base_expression.ufl_shape == (2, 3)
        actual = apply_derivatives(div(base_expression))
        assert actual.ufl_shape == (2,)
        expected = as_tensor([div(f), div(g)])
        assert equal_up_to_index_relabelling(actual, expected)

    def test_vector_component_tensor(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        ii = MultiIndex((Index(),))
        base_expression = ComponentTensor(Indexed(f, ii), ii)
        # Check that no simplification has occurred.
        assert type(base_expression) == ComponentTensor
        actual = apply_derivatives(div(base_expression))
        expected = grad(f)[i, i]
        assert equal_up_to_index_relabelling(actual, expected)

    def test_tensor_component_tensor(self, context):
        # Note here that the Div might in principle pass through the
        # ComponentTensor, but it does not here for reasons commented
        # in the implementation.
        f, g, w, element = context.tensor(dim1=2, dim2=3, cell=tetrahedron)
        ii, jj = indices(2)
        iijj = MultiIndex((ii, jj))
        base_expression = ComponentTensor(Indexed(f, iijj), iijj)
        # Check that no simplification has occurred.
        assert type(base_expression) == ComponentTensor
        actual = apply_derivatives(div(base_expression))
        # Expected is difficult to express otherwise. GTODO
        expected = apply_derivatives(apply_algebra_lowering(div(base_expression)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_indexed(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        base_expression = as_tensor([f, g])[1]
        actual = apply_derivatives(div(base_expression))
        expected = div(base_expression)
        assert equal_up_to_index_relabelling(actual, expected)


class TestCurl:
    def test_list_tensor(self, context):
        f, g, w, element = context.scalar(cell=tetrahedron)
        h = Coefficient(element)
        base_expression = as_tensor([f, g, h])
        actual = apply_derivatives(curl(base_expression))
        expected = apply_derivatives(
            apply_algebra_lowering(curl(base_expression)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_component_tensor(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        ii = MultiIndex((Index(),))
        base_expression = ComponentTensor(Indexed(f, ii), ii)
        # Check that no simplification has occurred.
        assert type(base_expression) == ComponentTensor
        actual = apply_derivatives(curl(base_expression))
        expected = apply_derivatives(
            apply_algebra_lowering(curl(base_expression)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_indexed(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        base_expression = as_tensor([f, g])[1]
        actual = apply_derivatives(curl(base_expression))
        expected = curl(base_expression)
        assert equal_up_to_index_relabelling(actual, expected)


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

    def test_component_derivative_simple(self):
        cell = triangle
        vector_element = VectorElement("CG", cell, 1, dim=3)
        scalar_element = FiniteElement("CG", cell, 1)
        u = Coefficient(vector_element)
        w = Argument(scalar_element, 0)
        base_expression = derivative(grad(u), u[0], w)
        actual = apply_derivatives(base_expression)
        zero_2d = Zero((2,), (), ())
        expected = as_tensor([grad(w), zero_2d, zero_2d])
        assert equal_up_to_index_relabelling(actual, expected)

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

class TestCancellations:
    def compute_form_data_with_pullbacks(self, expr):
        integral = expr * dx
        form_data = compute_form_data(
            integral,
            do_apply_function_pullbacks=True)
        return form_data.preprocessed_form._integrals[0].integrand()

    def transform(self, expr):
        # GTODO: Check that this matches compute_form_data.
        # GTODO: Merge this with the functions above and below.
        expr = apply_minimal_algebra_lowering(expr)
        expr = apply_derivatives(expr)
        expr = apply_function_pullbacks(expr)
        expr = apply_derivatives(expr)
        expr = apply_jacobian_cancellation(expr)
        expr = apply_algebra_lowering(expr)
        expr = apply_derivatives(expr)
        return expr


    def test_div_div_nonconforming(self, context):
        f, g, w, element = context.vector(dim=3, cell=tetrahedron)
        base_expression = div(f)
        actual = self.compute_form_data_with_pullbacks(base_expression)
        expected = IndexSum(
            Indexed(ComponentTensor(
                IndexSum(Product(
                    Indexed(
                        JacobianInverse(f.ufl_domain()),
                        MultiIndex((Index(43), Index(42)))),
                    Indexed(ReferenceGrad(ReferenceValue(f)),
                            MultiIndex((Index(41), Index(43))))),
                         MultiIndex((Index(43),))),
                MultiIndex((Index(41), Index(42)))),
                    MultiIndex((Index(40), Index(40)))),
            MultiIndex((Index(40),)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_div_div_conforming(self, context):
        f, g, w, element = context.rt()
        base_expression = div(f)
        actual = self.compute_form_data_with_pullbacks(base_expression)
        mesh = f.ufl_domain()
        expected = apply_algebra_lowering((1.0)/JacobianDeterminant(mesh) * ref_div_ito_ref_grad(ReferenceValue(f)))
        explicit_expected = Product(Division(FloatValue(1.0), JacobianDeterminant(mesh)),
                                    IndexSum(Indexed(ReferenceGrad(ReferenceValue(f)), MultiIndex((Index(13), Index(13)))),
                                             MultiIndex((Index(13),))))
        assert equal_up_to_index_relabelling(actual, expected)
        assert equal_up_to_index_relabelling(actual, explicit_expected)

    def test_curl_covariant_Piola(self, context):
        f, g, w, element = context.n1curl()
        base_expression = curl(f)
        actual = self.transform(base_expression)
        domain = f.ufl_domain()
        J = Jacobian(domain)
        detJ = JacobianDeterminant(domain)
        expected = apply_derivatives(
            apply_algebra_lowering(
                1.0/detJ * dot(J, ref_curl_ito_ref_grad(ReferenceValue(f)))))
        assert equal_up_to_index_relabelling(actual, expected)


def transform(expr):
    form = expr * dx
    form_data = compute_form_data(form, do_apply_function_pullbacks=True)
    return form_data.preprocessed_form.integrals()[0]._integrand
    # # Transform as for compute_form_data with
    # # do_apply_function_pullbacks=True
    # expr = apply_minimal_algebra_lowering(expr)
    # expr = apply_derivatives(expr)
    # expr = apply_function_pullbacks(expr)
    # expr = apply_integral_scaling(expr)
    # expr = apply_algebra_lowering(expr)
    # expr = apply_derivatives(expr)
    # return expr
        
class TestPullbacks():
    def test_grad_pullback(self, context):
        f, g, w, element = context.vector(dim=3, cell=triangle)
        base_expression = inner(grad(f), grad(g))
        actual = transform(base_expression)
        K = JacobianInverse(f.ufl_domain()) # Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), -1))
        expected = inner(dot(ReferenceGrad(ReferenceValue(f)), K),
                         dot(ReferenceGrad(ReferenceValue(g)), K))
        expected = apply_algebra_lowering(expected)
        assert equal_up_to_index_relabelling(actual, expected)

    def test_div_conforming_identity(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        q = Coefficient(rt_element)
        v = Coefficient(cg_element)
        base_expression = dot(q, grad(v))
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q),
                ReferenceGrad(ReferenceValue(v))))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_div_conforming_identity_reversed(self):
        # Identical to test_div_conforming_identity except for the
        # order of the operands to the dot product.
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        q = Coefficient(rt_element)
        v = Coefficient(cg_element)
        base_expression = dot(grad(v), q)
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q),
                ReferenceGrad(ReferenceValue(v))))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_curl_conforming_identity(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        n_element = FiniteElement("N1curl", cell, degree=1)
        q = Coefficient(rt_element)
        chi = Coefficient(n_element)
        base_expression = dot(q, chi)
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q),
                ReferenceValue(chi)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_curl_conforming_identity_reversed(self):
        # Identical to test_curl_conforming_identity except for the
        # order of the operands to the dot product.
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        n_element = FiniteElement("N1curl", cell, degree=1)
        q = Coefficient(rt_element)
        chi = Coefficient(n_element)
        base_expression = dot(chi, q)
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q),
                ReferenceValue(chi)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_sum_as_cutoff_type_1(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        q1 = Coefficient(rt_element)
        q2 = Coefficient(rt_element)
        v1 = Coefficient(cg_element)
        v2 = Coefficient(cg_element)
        base_expression = dot(q1, grad(v1)) + dot(grad(v2), q2)
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q1.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q1),
                ReferenceGrad(ReferenceValue(v1)))
            + dot((1.0/detJ) * ReferenceValue(q2),
                ReferenceGrad(ReferenceValue(v2))))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_sum_as_cutoff_type_2(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        q1 = Coefficient(rt_element)
        q2 = Coefficient(rt_element)
        v1 = Coefficient(cg_element)
        v2 = Coefficient(cg_element)
        base_expression = dot(q1 + q2, grad(v1) + grad(v2))
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q1.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q1)
                + (1.0/detJ) * ReferenceValue(q2),
                ReferenceGrad(ReferenceValue(v1))
                + ReferenceGrad(ReferenceValue(v2))))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_sum_as_cutoff_type_3(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        q1 = Coefficient(rt_element)
        q2 = Coefficient(rt_element)
        v1 = Coefficient(cg_element)
        v2 = Coefficient(cg_element)
        base_expression = dot(grad(v1) + grad(v2), q1 + q2)
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q1.ufl_domain())
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ReferenceValue(q1)
                + (1.0/detJ) * ReferenceValue(q2),
                ReferenceGrad(ReferenceValue(v1))
                + ReferenceGrad(ReferenceValue(v2))))
        assert equal_up_to_index_relabelling(actual, expected)

    @pytest.mark.xfail
    def test_div_conforming_identity_with_split(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        element = MixedElement(rt_element, cg_element)
        coeff = Coefficient(element)
        q, v = split(coeff)
        base_expression = dot(q, grad(v))
        actual = transform(base_expression)
        detJ = JacobianDeterminant(q.ufl_domain())
        ref_q = as_tensor([ReferenceValue(coeff)[0],
                           ReferenceValue(coeff)[1]])
        ref_v = ReferenceValue(coeff)[2]
        expected = apply_algebra_lowering(
            dot((1.0/detJ) * ref_q,
                ReferenceGrad(ref_v)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_evaluation_form_1(self):
        cell = triangle
        rt_element = FiniteElement("RT", cell, degree=1)
        cg_element = FiniteElement("CG", cell, degree=1)
        q = Coefficient(rt_element)
        f = Coefficient(cg_element)
        v = Argument(cg_element, 0)
        a = dot(q, grad(f)) * dx
        a_prime = derivative(a, f, v)
        a_prime_data = compute_form_data(
            a_prime, do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True)
        a_prime = a_prime_data.preprocessed_form
        actual = a_prime.integrals()[0]._integrand
        domain = q.ufl_domain()
        expected = Product(
            Product(
                QuadratureWeight(domain),
                Conditional(
                    LT(
                        JacobianDeterminant(domain),
                        Zero((), (), ())),
                    FloatValue(-1.0),
                    FloatValue(1.0))),
            IndexSum(
                Product(
                    Indexed(
                        ReferenceGrad(
                            ReferenceValue(v)),
                        MultiIndex((Index(10),))),
                    Indexed(
                        ReferenceValue(q),
                        MultiIndex((Index(10),)))),
                MultiIndex((Index(10),))))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_evaluation_form_2(self):
        cell = triangle
        cg_element = FiniteElement("CG", cell, degree=1)
        u = Coefficient(cg_element)
        rt_element = FiniteElement("RT", cell, degree=1)
        f = Coefficient(rt_element)
        v = Argument(rt_element, 0)
        a = u * div(f) * dx
        a_prime = derivative(a, f, v)
        a_prime_data = compute_form_data(
            a_prime, do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True)
        a_prime = a_prime_data.preprocessed_form
        actual = a_prime.integrals()[0]._integrand
        domain = u.ufl_domain()
        expected = Product(
            Product(
                QuadratureWeight(domain),
                Conditional(
                    LT(
                        JacobianDeterminant(domain),
                        Zero((), (), ())),
                    FloatValue(-1.0),
                    FloatValue(1.0))),
            Product(
                IndexSum(
                    Indexed(
                        ReferenceGrad(
                            ReferenceValue(v)),
                        MultiIndex((Index(10), Index(10)))),
                    MultiIndex((Index(10),))),
                ReferenceValue(u)))
        assert equal_up_to_index_relabelling(actual, expected)

    def test_evaluation_form_3(self):
        cell = tetrahedron
        n_element = FiniteElement("N1curl", cell, degree=1)
        u = Coefficient(n_element)
        v = Coefficient(n_element)
        rt_element = FiniteElement("RT", cell, degree=1)
        q = Coefficient(rt_element)
        a = dot(u, q + curl(v)) * dx
        a_data = compute_form_data(
            a, do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True)
        a = a_data.preprocessed_form
        actual = a.integrals()[0]._integrand
        domain = u.ufl_domain()
        expected = Product(
            Product(
                QuadratureWeight(domain),
                Conditional(
                    LT(
                        JacobianDeterminant(domain),
                        Zero((), (), ())),
                    FloatValue(-1.0),
                    FloatValue(1.0))),
            IndexSum(
                Product(
                    Indexed(
                        Sum(
                            ListTensor(
                                Sum(
                                    Indexed(
                                        ReferenceGrad(
                                        ReferenceValue(v)),
                                        MultiIndex((FixedIndex(2), FixedIndex(1)))),
                                    Product(
                                        IntValue(-1),
                                        Indexed(
                                            ReferenceGrad(
                                                ReferenceValue(v)),
                                            MultiIndex((FixedIndex(1), FixedIndex(2)))))),
                                Sum(
                                    Indexed(
                                        ReferenceGrad(
                                            ReferenceValue(v)),
                                        MultiIndex((FixedIndex(0), FixedIndex(2)))),
                                    Product(
                                        IntValue(-1),
                                        Indexed(
                                            ReferenceGrad(
                                                ReferenceValue(v)),
                                            MultiIndex((FixedIndex(2), FixedIndex(0)))))),
                                Sum(
                                    Indexed(
                                        ReferenceGrad(
                                            ReferenceValue(v)),
                                        MultiIndex((FixedIndex(1), FixedIndex(0)))),
                                    Product(
                                        IntValue(-1),
                                        Indexed(
                                            ReferenceGrad(
                                                ReferenceValue(v)),
                                            MultiIndex((FixedIndex(0), FixedIndex(1))))))),
                            ReferenceValue(q)),
                        MultiIndex((Index(17),))),
                    Indexed(
                        ReferenceValue(u),
                        MultiIndex((Index(17),)))),
                MultiIndex((Index(17),))))
        assert equal_up_to_index_relabelling(actual, expected)
