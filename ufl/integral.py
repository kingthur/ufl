"""The Integral class."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008-2009
#
# First added:  2008-03-14
# Last changed: 2011-08-25

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.constantvalue import is_true_ufl_scalar, is_python_scalar



# TODO: Move these somewhere more suitable?
def is_globally_constant(expr):
    """Check if an expression is globally constant, which
    includes spatially independent constant coefficients that
    are not known before assembly time."""

    from ufl.algorithms.traversal import traverse_terminals
    from ufl.argument import Argument
    from ufl.coefficient import Coefficient

    for e in traverse_terminals(expr):
        if isinstance(e, Argument):
            return False
        if isinstance(e, Coefficient) and e.element().family() != "Real":
            return False
        if not e.is_cellwise_constant():
            return False

    # All terminals passed constant check
    return True

# TODO: Move these somewhere more suitable?
def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if expr.shape() != ():
        return False
    return is_globally_constant(expr)


# TODO: Define some defaults as to how metadata should represent integration data here?
# quadrature_degree, quadrature_rule, ...

def register_domain_type(domain_type, measure_name):
    domain_type = domain_type.replace(" ", "_")
    if domain_type in Measure._domain_types:
        ufl_assert(Measure._domain_types[domain_type] == measure_name,
                   "Domain type already added with different measure name!")
        # Nothing to do
    else:
        ufl_assert(measure_name not in Measure._domain_types.values(),
                   "Measure name already used for another domain type!")
        Measure._domain_types[domain_type] = measure_name

class Measure(object):
    """A measure for integration."""
    __slots__ = ("_domain_type", "_domain_id", "_metadata", "_domain_data", "_repr")
    def __init__(self, domain_type, domain_id=0, metadata=None, domain_data=None):
        # Allow long domain type names with ' ' or '_'
        self._domain_type = domain_type.replace(" ", "_")
        # Map short domain type name to long automatically
        if not self._domain_type in Measure._domain_types:
            for k, v in Measure._domain_types.iteritems():
                if v == domain_type:
                    self._domain_type = k
                    break
            # In the end, did we find a valid domain type?
            if not self._domain_type in Measure._domain_types:
                error("Invalid domain type %s." % domain_type)

        self._domain_id = domain_id
        self._metadata = metadata
        self._domain_data = domain_data

        # NB! Deliberately ignoring domain data in repr, since it causes a bug
        # in the cache mechanisms in PyDOLFIN. I don't believe this is the
        # last word in this case, but it should fix all known problems for now.
        self._repr = "Measure(%r, %r, %r)" % (self._domain_type, self._domain_id,
                                              self._metadata)
        #self._repr = "Measure(%r, %r, %r, %r)" % (self._domain_type, self._domain_id,
        #                                          self._metadata, self._domain_data)

    def reconstruct(self, domain_id=None, metadata=None, domain_data=None):
        """Construct a new Measure object with some properties replaced with new values.

        Example:
            <dm = Measure instance>
            b = dm.reconstruct(domain_id=2)
            c = dm.reconstruct(metadata={ "quadrature_degree": 3 })

        Used by the call operator, so this is equivalent:
            b = dm(2)
            c = dm(0, { "quadrature_degree": 3 })
        """
        if domain_id is None:
            domain_id = self._domain_id
        if metadata is None:
            metadata = self._metadata
        if domain_data is None:
            domain_data = self._domain_data
        return Measure(self._domain_type, domain_id, metadata, domain_data)

    # Enumeration of valid domain types
    CELL = "cell"
    EXTERIOR_FACET = "exterior_facet"
    INTERIOR_FACET = "interior_facet"
    MACRO_CELL = "macro_cell"
    SURFACE = "surface"
    _domain_types = { \
        CELL: "dx",
        EXTERIOR_FACET: "ds",
        INTERIOR_FACET: "dS",
        MACRO_CELL: "dE",
        SURFACE: "dc"
        }
    _domain_types_tuple = (CELL, EXTERIOR_FACET, INTERIOR_FACET, MACRO_CELL, SURFACE)

    # Constant for undefined domain id
    UNDEFINED_DOMAIN_ID = "undefined"

    def domain_type(self):
        'Return the domain type, one of "cell", "exterior_facet" or "interior_facet".'
        return self._domain_type

    def domain_id(self):
        "Return the domain id (integer)."
        return self._domain_id

    def metadata(self):
        """Return the integral metadata. This data is not interpreted by UFL.
        It is passed to the form compiler which can ignore it or use it to
        compile each integral of a form in a different way."""
        return self._metadata

    def domain_data(self):
        """Return the integral domain_data. This data is not interpreted by UFL.
        Its intension is to give a context in which the domain id is interpreted."""
        return self._domain_data

    def __getitem__(self, domain_data):
        """Return a new Measure for same integration type with an attached
        context for interpreting domain ids. The default ID of this new Measure
        is undefined, and thus it must be qualified with a domain id to use
        in an integral. Example: dx = dx[boundaries]; L = f*v*dx(0) + g*v+dx(1)."""
        return self.reconstruct(domain_id=Measure.UNDEFINED_DOMAIN_ID, domain_data=domain_data)

    def __call__(self, domain_id=None, metadata=None):
        """Return integral of same type on given sub domain,
        optionally with some metadata attached."""
        return self.reconstruct(domain_id=domain_id, metadata=metadata)

    def __mul__(self, other):
        error("Can't multiply Measure from the right (with %r)." % (other,))

    def __rmul__(self, integrand):
        if self._domain_id == Measure.UNDEFINED_DOMAIN_ID:
            error("Missing domain id. You need to select a subdomain, " +\
                  "e.g. M = f*dx(0) for subdomain 0.")

        if isinstance(integrand, tuple):
            error("Mixing tuple and integrand notation not allowed.")
        #    from ufl.expr import Expr
        #    from ufl import inner
        #    # Experimental syntax like a = (u,v)*dx + (f,v)*ds
        #    ufl_assert(len(integrand) == 2 \
        #        and isinstance(integrand[0], Expr) \
        #        and isinstance(integrand[1], Expr),
        #        "Invalid integrand %s." % repr(integrand))
        #    integrand = inner(integrand[0], integrand[1])

        ufl_assert(is_true_ufl_scalar(integrand),
            "Trying to integrate expression of rank %d with free indices %r." \
            % (integrand.rank(), integrand.free_indices()))

        from ufl.form import Form
        return Form( [Integral(integrand, self)] )

    def __str__(self):
        d = Measure._domain_types[self._domain_type]
        metastring = "" if self._metadata is None else ("<%s>" % repr(self._metadata))
        return "%s%s%s" % (d, self._domain_id, metastring)

    def __repr__(self):
        return self._repr

    def __hash__(self):
        return hash(self._repr)

    def __eq__(self, other):
        return repr(self) == repr(other)

class Integral(object):
    "An integral over a single domain."
    __slots__ = ("_integrand", "_measure", "_repr")
    def __init__(self, integrand, measure):
        from ufl.expr import Expr
        ufl_assert(isinstance(integrand, Expr), "Expecting integrand to be an Expr instance.")
        ufl_assert(isinstance(measure, Measure), "Expecting measure to be a Measure instance.")
        self._integrand = integrand
        self._measure   = measure
        self._repr = "Integral(%r, %r)" % (self._integrand, self._measure) # REPR don't cache here, do in form

    def reconstruct(self, integrand):
        """Construct a new Integral object with some properties replaced with new values.

        Example:
            <a = Integral instance>
            b = a.reconstruct(expand_compounds(a.integrand()))
        """
        return Integral(integrand, self._measure)

    def integrand(self):
        "Return the integrand expression, which is an Expr instance."
        return self._integrand

    def measure(self):
        "Return the measure associated with this integral."
        return self._measure

    def __neg__(self):
        return self.reconstruct(-self._integrand)

    def __mul__(self, scalar):
        ufl_assert(is_python_scalar(scalar), "Cannot multiply an integral with non-constant values.")
        return self.reconstruct(scalar*self._integrand)

    def __rmul__(self, scalar):
        ufl_assert(is_scalar_constant_expression(scalar),
                   "An integral can only be multiplied by a globally constant scalar expression.")
        return self.reconstruct(scalar*self._integrand)

    def __str__(self):
        return "{ %s } * %s" % (self._integrand, self._measure)

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr)
