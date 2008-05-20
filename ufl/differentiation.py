#!/usr/bin/env python

"""
Differential operators. Needs work!
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"

from output import *
from base import *
from indexing import *


#--- Differentiation ---

# FIXME: Add SpatialDerivative and TimeDerivative?

class PartialDerivative(UFLObject):
    "Partial derivative of an expression w.r.t. a spatial direction given by an index."
    
    __slots__ = ("_expression", "_rank", "_indices", "_free_indices") #, "_fixed_indices", "_repeated_indices")
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr
            self._indices = indices
        else:
            self._indices = MultiIndex(indices, 1) # len(indices)) instead of 1 to support higher order derivatives.
        
        # Find free and repeated indices among the combined indices of the expression and dx((i,j,k))
        indices = expression.free_indices() + self._indices._indices
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices(indices)
        # FIXME: We don't need to store all these here, remove the ones we don't use after implementing summation expansion.
        #self._fixed_indices      = fixed_indices
        self._free_indices       = free_indices
        #self._repeated_indices   = repeated_indices
        self._rank = num_unassigned_indices
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._expression.rank()
    
    def __str__(self):
        # TODO: Pretty-print for higher order derivatives.
        return "(d[%s] / dx_%s)" % (str(self._expression), str(self._indices))
    
    def __repr__(self):
        return "PartialDerivative(%s, %s)" % (repr(self._expression), repr(self._indices))


# Extend UFLObject with spatial differentiation operator a.dx(i)
def _dx(self, *i):
    """Return the partial derivative with respect to spatial variable number i"""
    return PartialDerivative(self, i)
UFLObject.dx = _dx



# FIXME: Anders: Can't we just remove this?
#        Martin: Not necessarily, not unless PartialDerivative is made more general. The idea is that Diff represents df/ds where s is a Symbol and/or Variable.
# FIXME: This is the same mathematical operation as PartialDiff, should have very similar behaviour or even be the same class.
class Diff(UFLObject):
    __slots__ = ("_f", "_x", "_index", "_free_indices")
    
    def __init__(self, f, x):
        ufl_assert(isinstance(f, UFLObject), "Expecting an UFLObject in Diff.")
        ufl_assert(isinstance(x, (Symbol, Variable)), "Expecting a Symbol or Variable in Diff.")
        self._f = f
        self._x = x
        fi = f.free_indices()
        xi = x.free_indices()
        ufl_assert(len(set(fi) ^ set(xi)) == 0, "FIXME: Repeated indices in Diff not implemented.") # FIXME
        self._free_indices = tuple(fi + xi)
    
    def operands(self):
        return (self._f, self._x)
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._f.rank()
    
    def __str__(self):
        return "(d[%s] / d[%s])" % (str(self._f), str(self._x))

    def __repr__(self):
        return "Diff(%s, %s)" % (repr(self._f), repr(self._x))

def diff(f, x):
    return Diff(f, x)


class DifferentialOperator(UFLObject):
    """For the moment this is just a dummy class to enable "isinstance(o, DifferentialOperator)"."""
    __slots__ = ()


class Grad(DifferentialOperator):
    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking gradient of an expression with free indices, should this be a valid expression? Please provide examples!")
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return self.f.rank() + 1
    
    def __str__(self):
        return "grad(%s)" % str(self.f)

    def __repr__(self):
        return "Grad(%s)" % repr(self.f)


class Div(DifferentialOperator):
    __slots__ = ("f",)

    def __init__(self, f):
        ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking divergence of an expression with free indices, should this be a valid expression? Please provide examples!")
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return self.f.rank() - 1
    
    def __str__(self):
        return "div(%s)" % str(self.f)

    def __repr__(self):
        return "Div(%s)" % repr(self.f)


class Curl(DifferentialOperator):
    __slots__ = ("f",)

    def __init__(self, f):
        ufl_assert(f.rank()== 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return 1
    
    def __str__(self):
        return "curl(%s)" % str(self.f)
    
    def __repr__(self):
        return "Curl(%s)" % repr(self.f)


class Rot(DifferentialOperator):
    __slots__ = ("f",)

    def __init__(self, f):
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking rot of an expression with free indices, should this be a valid expression? Please provide examples!")
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "rot(%s)" % str(self.f)
    
    def __repr__(self):
        return "Rot(%s)" % repr(self.f)

