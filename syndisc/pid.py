"""
Information decomposition based on synergistic disclosure.

References:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE.txt for details.

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np
from .syndisc import disclosure

from dit.pid.pid import BasePID
from dit.multivariate import coinformation
from dit.utils import flatten
from dit.utils import powerset
from itertools import combinations, islice, permutations
import networkx as nx


def full_constraint_lattice(elements):
    """
    Return a lattice of constrained marginals, with the same partial order
    relationship as Ryan James' constraint lattice, but where the nodes are not
    restricted to cover the whole set of variables.

    Parameters
    ----------
    elements : iter of iters
        Input variables to the PID.

    Returns
    -------
    lattice : nx.DiGraph
        The lattice of antichains.
    """
    def comparable(a, b):
        return a < b or b < a

    def antichain(ss):
        if not ss:
            return False
        return all(not comparable(frozenset(a), frozenset(b)) for a, b in combinations(ss, 2))


    def less_than(sss1, sss2):
        return all(any(set(ss1) <= set(ss2) for ss2 in sss2) for ss1 in sss1)

    def normalize(sss):
        return tuple(sorted(tuple( tuple(sorted(ss)) for ss in sss ),
                            key=lambda s: (-len(s), s)))

    # Enumerate all nodes in the lattice (same nodes as in usual PID lattice)
    elements = set(elements)
    combos = (sum(s, tuple()) for s in powerset(elements))
    pps = [ss for ss in powerset(combos) if antichain(ss)]

    # Compute all order relationships using the constraint lattice ordering
    order = [(a, b) for a, b in permutations(pps, 2) if less_than(a, b)]

    # Build the DiGraph by removing redundant order relationships
    lattice = nx.DiGraph()
    for a, b in order:
        if not any(((a, c) in order) and ((c, b) in order) for c in pps):
            lattice.add_edge(normalize(b), normalize(a))
    lattice.root = next(iter(nx.topological_sort(lattice)))

    return lattice


def synergy(dist):
    """
    Computes simple synergy for the first n-1 variables in the distribution,
    using the last one as a target, and preserving the individual marginals.

    Acts as a simple wrapper around `syndisc.disclosure` for ease of use.

    Parameters
    ----------
    dist : dit.Distribution
        Distribution to compute synergy over. Last variable is used as target

    Returns
    -------
    S : float
        Synergistic information disclosure
    """
    return disclosure(dist)
       

class PID_SD(BasePID):
    """
    The disclosure information decomposition.
    """
    _name = "I_dis"

    def __init__(self, dist, inputs=None, output=None, reds=None, pis=None, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        inputs : iter of iters, None
            The set of input variables. If None, `dist.rvs` less indices
            in `output` is used.
        output : iter, None
            The output variable. If None, `dist.rvs[-1]` is used.
        reds : dict, None
            Redundancy values pre-assessed.
        pis : dict, None
            Partial information values pre-assessed.
        """
        self._dist = dist

        if output is None:
            output = dist.rvs[-1]
        if inputs is None:
            inputs = [var for var in dist.rvs if var[0] not in output]

        self._inputs = tuple(map(tuple, inputs))
        self._output = tuple(output)
        self._kwargs = kwargs

        self._lattice = full_constraint_lattice(self._inputs)

        # To compute the Mobius inversion reusing dit's code, we reverse the
        # lattice, compute Mobius, and reverse it back again.
        self._lattice = self._lattice.reverse()
        self._lattice.root = next(iter(nx.topological_sort(self._lattice)))

        self._total = coinformation(self._dist, [list(flatten(self._inputs)), self._output])
        self._compute(reds, pis)

        self._lattice = self._lattice.reverse()
        self._lattice.root = next(iter(nx.topological_sort(self._lattice)))


    @staticmethod
    def _measure(d, inputs, output):
        """
        Compute synergistic disclosure.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dis for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        disclosure : float
            The value of I_dis
        """
        return disclosure(d, cons=inputs, output=output)

