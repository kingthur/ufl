# -*- coding: utf-8 -*-
"""Algorithm for expanding compound expressions involving intractable
operators such as Cofactor into equivalent representations without
these operators."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
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
# Modified by Anders Logg, 2009-2010

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.compound_expressions import cofactor_expr
from ufl.corealg.multifunction import MultiFunction


class MinimalAlgebraLowering(MultiFunction):
    """Expands expressions involving intractable high level compound
    operators such as cofactor to equivalent representations without
    these operators, for the application of derivatives."""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def cofactor(self, o, A):
        return cofactor_expr(A)


def apply_minimal_algebra_lowering(expr):
    """Expands expressions involving intractable high level compound
    operators such as cofactor to equivalent representations without
    these operators, for the application of derivatives."""
    return map_integrand_dags(MinimalAlgebraLowering(), expr)
