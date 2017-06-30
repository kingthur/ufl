from ufl import *
from ufl.core.multiindex import MultiIndex, FixedIndex
from ufl.core.operator import Operator
from ufl.corealg.traversal import post_traversal
from itertools import izip_longest


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
