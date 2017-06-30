from ufl import *
from ufl.algebra import Product, Sum
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.core.multiindex import MultiIndex
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.tensors import ComponentTensor
from equal_up_to_index_relabelling import equal_up_to_index_relabelling

def test_basic():
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
