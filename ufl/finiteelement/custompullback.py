# -*- coding: utf-8 -*-
# Copyright (C) 2016 Imperial College London
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

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class CustomPullback(FiniteElementBase):
    """
    Provide a custom pullback for an existing finite element space.

    :arg element: The element to wrap.
    :arg pullback: a :class:`Pullback` object providing the pullback.
    """
    __slots__ = ("_pullback", "_element")

    def __init__(self, element, pullback):
        self._pullback = pullback
        self._element = element

    def pullback(self, domain, rv):
        """Pullback a form argument to reference space.

        :arg domain: The :class:`ufl.Domain` object used to construct
            geometric information.

        :arg rv: The :class:`ufl.FormArgument` to pull back (already
            wrapped in :class:`ufl.ReferenceValue`).

        :returns: a :class:`ufl.Expr` representing ``rv`` in reference
            space."""
        return self._pullback(domain, rv)

    def __getattr__(self, name):
        return getattr(self._element, name)

    def __repr__(self):
        return "CustomPullback(%r, %r)" % (self._element, self._pullback)

    def __str__(self):
        return "CustomPullback(%s, %s)" % (self._element, self._pullback)

    def shortstr(self):
        return "CustomPullback(%s, %s)" % (self._element.shortstr(),
                                           self._pullback.shortstr())

    def mapping(self):
        return "custom"


class Pullback(object):
    """A representation of a pullback.

    :arg name: A descriptive name.

    Subclass this object and implement :meth:`__call__` to provide a
    particular pullback type.
    """

    __slots__ = ("name")

    def __init__(self, name):
        self.name = name

    def __call__(self, domain, rv):
        """Pull back ``rv`` to reference space.

        :arg domain: The :class:`ufl.Domain` object used to construct
            geometric information.

        :arg g: The :class:`ufl.FormArgument` to pull back (already
            wrapped in :class:`ufl.ReferenceValue`).

        :returns: a :class:`ufl.Expr` representing ``rv`` in reference
            space."""
        raise NotImplementedError("Subclass should implement __call__")

    def __repr__(self):
        return "Pullback(%r)" % self.name

    __str__ = __repr__

    shortstr = __str__
