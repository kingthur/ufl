#!/usr/bin/env py.test

import pytest

from ufl import *
#from ufl.indexutils import *
from ufl.algorithms import *
from ufl.classes import IndexSum

# TODO: add more expressions to test as many possible combinations of index notation as feasible...

def xtest_index_utils(self):
    ii = indices(3)
    self.assertEqual(ii, unique_indices(ii) )
    self.assertEqual(ii, unique_indices(ii+ii) )

    self.assertEqual((), repeated_indices(ii) )
    self.assertEqual(ii, repeated_indices(ii+ii) )

    self.assertEqual(ii, shared_indices(ii, ii) )
    self.assertEqual(ii, shared_indices(ii, ii+ii) )
    self.assertEqual(ii, shared_indices(ii+ii, ii) )
    self.assertEqual(ii, shared_indices(ii+ii, ii+ii) )

    self.assertEqual(ii, single_indices(ii) )
    self.assertEqual((), single_indices(ii+ii) )

def test_vector_indices(self):
    element = VectorElement("CG", "triangle", 1)
    u = Argument(element, 2)
    f = Coefficient(element)
    a = u[i]*f[i]*dx
    b = u[j]*f[j]*dx

def test_tensor_indices(self):
    element = TensorElement("CG", "triangle", 1)
    u = Argument(element, 2)
    f = Coefficient(element)
    a = u[i, j]*f[i, j]*dx
    b = u[j, i]*f[i, j]*dx
    c = u[j, i]*f[j, i]*dx
    with pytest.raises(UFLException):
        d = (u[i, i]+f[j, i])*dx

def test_indexed_sum1(self):
    element = VectorElement("CG", "triangle", 1)
    u = Argument(element, 2)
    f = Coefficient(element)
    a = u[i]+f[i]
    with pytest.raises(UFLException):
        a*dx

def test_indexed_sum2(self):
    element = VectorElement("CG", "triangle", 1)
    v = Argument(element, 2)
    u = Argument(element, 3)
    f = Coefficient(element)
    a = u[j]+f[j]+v[j]+2*v[j]+exp(u[i]*u[i])/2*f[j]
    with pytest.raises(UFLException):
        a*dx

def test_indexed_sum3(self):
    element = VectorElement("CG", "triangle", 1)
    u = Argument(element, 2)
    f = Coefficient(element)
    with pytest.raises(UFLException):
        a = u[i]+f[j]

def test_indexed_function1(self):
    element = VectorElement("CG", "triangle", 1)
    v = Argument(element, 2)
    u = Argument(element, 3)
    f = Coefficient(element)
    aarg = (u[i]+f[i])*v[i]
    a = exp(aarg)*dx

def test_indexed_function2(self):
    element = VectorElement("CG", "triangle", 1)
    v = Argument(element, 2)
    u = Argument(element, 3)
    f = Coefficient(element)
    bfun  = cos(f[0])
    left  = u[i] + f[i]
    right = v[i] * bfun
    self.assertEqual(len(left.free_indices()), 1)
    self.assertEqual(left.free_indices()[0], i)
    self.assertEqual(len(right.free_indices()), 1)
    self.assertEqual(right.free_indices()[0], i)
    b = left * right * dx

def test_indexed_function3(self):
    element = VectorElement("CG", "triangle", 1)
    v = Argument(element, 2)
    u = Argument(element, 3)
    f = Coefficient(element)
    with pytest.raises(UFLException):
        c = sin(u[i] + f[i])*dx

def test_vector_from_indices(self):
    element = VectorElement("CG", "triangle", 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)

    # legal
    vv = as_vector(u[i], i)
    uu = as_vector(v[j], j)
    w  = v + u
    ww = vv + uu
    self.assertEqual(vv.rank(), 1)
    self.assertEqual(uu.rank(), 1)
    self.assertEqual(w.rank(), 1)
    self.assertEqual(ww.rank(), 1)

def test_matrix_from_indices(self):
    element = VectorElement("CG", "triangle", 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)

    A  = as_matrix(u[i]*v[j], (i, j))
    B  = as_matrix(v[k]*v[k]*u[i]*v[j], (j, i))
    C  = A + A
    C  = B + B
    D  = A + B
    self.assertEqual(A.rank(), 2)
    self.assertEqual(B.rank(), 2)
    self.assertEqual(C.rank(), 2)
    self.assertEqual(D.rank(), 2)

def test_vector_from_list(self):
    element = VectorElement("CG", "triangle", 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)

    # create vector from list
    vv = as_vector([u[0], v[0]])
    ww = vv + vv
    self.assertEqual(vv.rank(), 1)
    self.assertEqual(ww.rank(), 1)

def test_matrix_from_list(self):
    element = VectorElement("CG", "triangle", 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)

    # create matrix from list
    A  = as_matrix( [ [u[0], u[1]], [v[0], v[1]] ] )
    # create matrix from indices
    B  = as_matrix( (v[k]*v[k]) * u[i]*v[j], (j, i) )
    # Test addition
    C  = A + A
    C  = B + B
    D  = A + B
    self.assertEqual(A.rank(), 2)
    self.assertEqual(B.rank(), 2)
    self.assertEqual(C.rank(), 2)
    self.assertEqual(D.rank(), 2)

def test_tensor(self):
    element = VectorElement("CG", "triangle", 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)
    f  = Coefficient(element)
    g  = Coefficient(element)

    # define the components of a fourth order tensor
    Cijkl = u[i]*v[j]*f[k]*g[l]
    self.assertEqual(Cijkl.rank(), 0)
    self.assertEqual(set(Cijkl.free_indices()), {i, j, k, l})

    # make it a tensor
    C = as_tensor(Cijkl, (i, j, k, l))
    self.assertEqual(C.rank(), 4)
    self.assertSameIndices(C, ())

    # get sub-matrix
    A = C[:,:, 0, 0]
    self.assertEqual(A.rank(), 2)
    self.assertSameIndices(A, ())
    A = C[:,:, i, j]
    self.assertEqual(A.rank(), 2)
    self.assertEqual(set(A.free_indices()), {i, j})

    # legal?
    vv = as_vector([u[i], v[i]])
    ww = f[i]*vv # this is well defined: ww = sum_i <f_i*u_i, f_i*v_i>

    # illegal
    with pytest.raises(UFLException):
        vv = as_vector([u[i], v[j]])

    # illegal
    with pytest.raises(UFLException):
        A = as_matrix( [ [u[0], u[1]], [v[0],] ] )

    # ...

def test_indexed(self):
    element = VectorElement("CG", "triangle", 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)
    f  = Coefficient(element)
    i, j, k, l = indices(4)

    a = v[i]
    self.assertSameIndices(a, (i,))

    a = outer(v, u)[i, j]
    self.assertSameIndices(a, (i, j))

    a = outer(v, u)[i, i]
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)

def test_spatial_derivative(self):
    cell = triangle
    element = VectorElement("CG", cell, 1)
    v  = TestFunction(element)
    u  = TrialFunction(element)
    i, j, k, l = indices(4)
    d = cell.geometric_dimension()

    a = v[i].dx(i)
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, ())

    a = v[i].dx(j)
    self.assertSameIndices(a, (i, j))
    self.assertNotIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, ())

    a = (v[i]*u[j]).dx(i, j)
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, ())

    a = v.dx(i, j)
    #self.assertSameIndices(a, (i,j))
    self.assertEqual(set(a.free_indices()), {j, i})
    self.assertNotIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, (d,))

    a = v[i].dx(0)
    self.assertSameIndices(a, (i,))
    self.assertNotIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, ())

    a = (v[i]*u[j]).dx(0, 1)
    # indices change place because of sorting, I guess this may be ok
    self.assertEqual(set(a.free_indices()), {i, j})
    self.assertNotIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, ())

    a = v.dx(i)[i]
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)
    self.assertEqual(a.ufl_shape, ())

def test_renumbering(self):
    pass