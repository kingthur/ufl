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

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class CustomPullback(FiniteElementBase):
    """
    Define a custom pullback for an existing finite element space
    """
    __slots__ = ("_pullback", "_element")
    def __init__(self, element, pullback):
        self._pullback = pullback
        self._element = element

    def __getattr__(self, name):
        return getattr(self._element, name)

    def repr(self):
        return "CustomPullback(%r, %r)" % (self._element, self._pullback)


class Pullback(object):
    """A class to hold the pullback to pass to the CustomPullback class"""
    __slots__ = ("callable", "name")
    def __init__(self, callable, name):
        self.callable = callable
        self.name = name

    def repr(self):
        return "Pullback(%r)" % self.name
