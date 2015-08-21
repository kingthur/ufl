# -*- coding: utf-8 -*-
"This module defines classes representing constant values."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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
#
# Modified by Anders Logg, 2011.

from six.moves import zip
from six.moves import xrange as range
from six import iteritems

from ufl.log import warning, error
from ufl.assertions import ufl_assert, expecting_python_scalar
from ufl.core.expr import Expr
from ufl.core.terminal import Terminal
from ufl.core.multiindex import Index, FixedIndex
from ufl.utils.dicts import EmptyDict
from ufl.core.ufl_type import ufl_type

#--- Helper functions imported here for compatibility---
from ufl.checks import is_python_scalar, is_ufl_scalar, is_true_ufl_scalar

# Precision for float formatting
precision = None
def format_float(x):
    "Format float value based on global UFL precision."
    if precision is None:
        return repr(x)
    else:
        return ("%%.%dg" % precision) % x


#--- Base classes for constant types ---

@ufl_type(is_abstract=True)
class ConstantValue(Terminal):
    __slots__ = ()
    def __init__(self):
        Terminal.__init__(self)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return True

    def domains(self):
        "Return tuple of domains related to this terminal object."
        return ()


#--- Class for representing abstract constant symbol only for use internally in form compilers
#@ufl_type()
#class AbstractSymbol(ConstantValue):
#    "UFL literal type: Representation of a constant valued symbol with unknown properties."
#    __slots__ = ("_name", "ufl_shape")
#    def __init__(self, name, shape):
#        ConstantValue.__init__(self)
#        self._name = name
#        self.ufl_shape = shape
#
#    def reconstruct(self, name=None):
#        if name is None:
#            name = self._name
#        return AbstractSymbol(name, self.ufl_shape)
#
#    def __str__(self):
#        return "<Abstract symbol named '%s' with shape %s>" % (self._name, self.ufl_shape)
#
#    def __repr__(self):
#        return "AbstractSymbol(%r, %r)" % (self._name, self.ufl_shape)
#
#    def __eq__(self, other):
#        return isinstance(other, AbstractSymbol) and self._name == other._name and self.ufl_shape == other.ufl_shape


#--- Class for representing zero tensors of different shapes ---

# TODO: Add geometric dimension/domain and Argument dependencies to Zero?
@ufl_type(is_literal=True)
class Zero(ConstantValue):
    "UFL literal type: Representation of a zero valued expression."
    __slots__ = ("ufl_shape", "ufl_free_indices", "ufl_index_dimensions")

    _cache = {}

    def __getnewargs__(self):
        return (self.ufl_shape, self.ufl_free_indices, self.ufl_index_dimensions)

    def __new__(cls, shape=(), free_indices=(), index_dimensions=None):
        if free_indices:
            self = ConstantValue.__new__(cls)
        else:
            self = Zero._cache.get(shape)
            if self is not None:
                return self
            self = ConstantValue.__new__(cls)
            Zero._cache[shape] = self
        self._init(shape, free_indices, index_dimensions)
        return self

    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        pass

    def _init(self, shape=(), free_indices=(), index_dimensions=None):
        ConstantValue.__init__(self)

        if not all(isinstance(i, int) for i in shape):
            error("Expecting tuple of int.")
        if not isinstance(free_indices, tuple):
            error("Expecting tuple for free_indices, not %s" % str(free_indices))

        self.ufl_shape = shape
        if not free_indices:
            self.ufl_free_indices = ()
            self.ufl_index_dimensions = ()
        elif all(isinstance(i, Index) for i in free_indices): # Handle old input format
            if not (isinstance(index_dimensions, dict)
                    and all(isinstance(i, Index) for i in index_dimensions.keys())):
                error("Expecting tuple of index dimensions, not %s" % str(index_dimensions))
            self.ufl_free_indices = tuple(sorted(i.count() for i in free_indices))
            self.ufl_index_dimensions = tuple(d for i, d in sorted(iteritems(index_dimensions), key=lambda x: x[0].count()))
        else: # Handle new input format
            if not all(isinstance(i, int) for i in free_indices):
                error("Expecting tuple of integer free index ids, not %s" % str(free_indices))
            if not (isinstance(index_dimensions, tuple)
                    and all(isinstance(i, int) for i in index_dimensions)):
                error("Expecting tuple of integer index dimensions, not %s" % str(index_dimensions))
            # TODO: Assume sorted and avoid this cost.
            ufl_assert(sorted(free_indices) == list(free_indices),
                       "Expecting sorted input. Remove this check later for efficiency.")
            self.ufl_free_indices = free_indices
            self.ufl_index_dimensions = index_dimensions

    def free_indices(self):
        "Intermediate helper property getter to transition from .free_indices() to .ufl_free_indices."
        return tuple(Index(count=i) for i in self.ufl_free_indices)

    def index_dimensions(self):
        "Intermediate helper property getter to transition from .index_dimensions() to .ufl_index_dimensions."
        return { Index(count=i): d for i, d in zip(self.ufl_free_indices, self.ufl_index_dimensions) }

    def reconstruct(self, free_indices=None):
        if not free_indices:
            return self
        ufl_assert(len(free_indices) == len(self.ufl_free_indices),
                   "Size mismatch between old and new indices.")
        fid = self.ufl_index_dimensions
        new_fi, new_fid = zip(*tuple(sorted((free_indices[pos], fid[pos]) for pos, a in enumerate(self.ufl_free_indices))))
        return Zero(self.ufl_shape, new_fi, new_fid)

    def evaluate(self, x, mapping, component, index_values):
        return 0.0

    def __str__(self):
        if self.ufl_shape == () and self.ufl_free_indices == ():
            return "0"
        return "(0<%r, %r>)" % (self.ufl_shape, self.ufl_free_indices)

    def __repr__(self):
        return "Zero(%r, %r, %r)" % (self.ufl_shape,
                self.ufl_free_indices, self.ufl_index_dimensions)

    def __eq__(self, other):
        if isinstance(other, Zero):
            if self is other:
                return True
            return (self.ufl_shape == other.ufl_shape and
                    self.ufl_free_indices == other.ufl_free_indices and
                    self.ufl_index_dimensions == other.ufl_index_dimensions)
        elif isinstance(other, (int, float)):
            return other == 0
        else:
            return False

    def __neg__(self):
        return self

    def __abs__(self):
        return self

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

def zero(*shape):
    "UFL literal constant: Return a zero tensor with the given shape."
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return Zero(shape[0])
    else:
        return Zero(shape)


#--- Scalar value types ---

@ufl_type(is_abstract=True, is_scalar=True)
class ScalarValue(ConstantValue):
    "A constant scalar value."
    __slots__ = ("_value",)

    def __init__(self, value):
        ConstantValue.__init__(self)
        self._value = value

    def value(self):
        return self._value

    def evaluate(self, x, mapping, component, index_values):
        return self._value

    def __eq__(self, other):
        """This is implemented to allow comparison with python scalars.

        Note that this will make IntValue(1) != FloatValue(1.0),
        but ufl-python comparisons like
            IntValue(1) == 1.0
            FloatValue(1.0) == 1
        can still succeed. These will however not have the same
        hash value and therefore not collide in a dict.
        """
        if isinstance(other, self._ufl_class_):
            return self._value == other._value
        elif isinstance(other, (int, float)):
            # FIXME: Disallow this, require explicit 'expr == IntValue(3)' instead to avoid ambiguities!
            return other == self._value
        else:
            return False

    def __str__(self):
        return str(self._value)

    def __float__(self):
        return float(self._value)

    def __int__(self):
        return int(self._value)

    def __neg__(self):
        return type(self)(-self._value)

    def __abs__(self):
        return type(self)(abs(self._value))


@ufl_type(wraps_type=float, is_literal=True)
class FloatValue(ScalarValue):
    "UFL literal type: Representation of a constant scalar floating point value."
    __slots__ = ()

    def __getnewargs__(self):
        return (self._value,)

    def __new__(cls, value):
        if value == 0.0:
            # Always represent zero with Zero
            return Zero()
        return ConstantValue.__new__(cls)

    def __init__(self, value):
        ScalarValue.__init__(self, float(value))

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, format_float(self._value))


@ufl_type(wraps_type=int, is_literal=True)
class IntValue(ScalarValue):
    "UFL literal type: Representation of a constant scalar integer value."
    __slots__ = ()

    _cache = {}

    def __getnewargs__(self):
        return (self._value,)

    def __new__(cls, value):
        if value == 0:
            # Always represent zero with Zero
            return Zero()
        elif abs(value) < 100:
            # Small numbers are cached to reduce memory usage (fly-weight pattern)
            self = IntValue._cache.get(value)
            if self is not None:
                return self
            self = ScalarValue.__new__(cls)
            IntValue._cache[value] = self
        else:
            self = ScalarValue.__new__(cls)
        self._init(value)
        return self

    def _init(self, value):
        ScalarValue.__init__(self, int(value))

    def __init__(self, value):
        pass

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, repr(self._value))


#--- Identity matrix ---

@ufl_type()
class Identity(ConstantValue):
    "UFL literal type: Representation of an identity matrix."
    __slots__ = ("_dim", "ufl_shape")

    def __init__(self, dim):
        ConstantValue.__init__(self)
        self._dim = dim
        self.ufl_shape = (dim, dim)

    def evaluate(self, x, mapping, component, index_values):
        a, b = component
        return 1 if a == b else 0

    def __getitem__(self, key):
        ufl_assert(len(key) == 2, "Size mismatch for Identity.")
        if all(isinstance(k, (int, FixedIndex)) for k in key):
            return IntValue(1) if (int(key[0]) == int(key[1])) else Zero()
        return Expr.__getitem__(self, key)

    def __str__(self):
        return "I"

    def __repr__(self):
        return "Identity(%d)" % self._dim

    def __eq__(self, other):
        return isinstance(other, Identity) and self._dim == other._dim

#--- Permutation symbol ---

@ufl_type()
class PermutationSymbol(ConstantValue):
    """UFL literal type: Representation of a permutation symbol.

    This is also known as the Levi-Civita symbol, antisymmetric symbol,
    or alternating symbol."""
    __slots__ = ("ufl_shape", "_dim")

    def __init__(self, dim):
        ConstantValue.__init__(self)
        self._dim = dim
        self.ufl_shape = (dim,)*dim

    def evaluate(self, x, mapping, component, index_values):
        return self.__eps(component)

    def __getitem__(self, key):
        ufl_assert(len(key) == self._dim, "Size mismatch for PermutationSymbol.")
        if all(isinstance(k, (int, FixedIndex)) for k in key):
            return self.__eps(key)
        return Expr.__getitem__(self, key)

    def __str__(self):
        return "eps"

    def __repr__(self):
        return "PermutationSymbol(%d)" % self._dim

    def __eq__(self, other):
        return isinstance(other, PermutationSymbol) and self._dim == other._dim

    def __eps(self, x):
        """This function body is taken from
        http://www.mathkb.com/Uwe/Forum.aspx/math/29865/N-integer-Levi-Civita"""
        result = IntValue(1)
        for i, x1 in enumerate(x):
            for j in range(i + 1, len(x)):
                x2 = x[j]
                if x1 > x2:
                    result = -result
                elif x1 == x2:
                    return Zero()
        return result

def as_ufl(expression):
    "Converts expression to an Expr if possible."
    if isinstance(expression, Expr):
        return expression
    if isinstance(expression, float):
        return FloatValue(expression)
    if isinstance(expression, int):
        return IntValue(expression)
    error(("Invalid type conversion: %s can not be converted to any UFL type.\n"+\
           "The representation of the object is:\n%r") % (type(expression), expression))
