# -*- coding: utf-8 -*-
# Copyright (C) 2014 Andrew T. T. McRae
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
# Modified by Massimiliano Leoni, 2016

# import six
from ufl.utils.py23 import as_native_str
from ufl.finiteelement.finiteelementbase import FiniteElementBase


# @six.python_2_unicode_compatible
class InteriorElement(FiniteElementBase):
    """A version of an existing Finite Element space in which only the dofs
    associated with the interior have been kept."""
    def __init__(self, element):
        self._element = element
        self._repr = as_native_str("InteriorElement(%s)" % repr(element))

        family = "InteriorElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = element.value_shape()
        reference_value_shape = element.reference_value_shape()
        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def mapping(self):
        return self._element.mapping()

    def reconstruct(self, **kwargs):
        return InteriorElement(self._element.reconstruct(**kwargs))

    def __str__(self):
        return "InteriorElement(%s)" % str(self._element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "InteriorElement(%s)" % str(self._element.shortstr())
