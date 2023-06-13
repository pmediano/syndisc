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
from lattices.lattices import dependency_lattice
from dit.pid.pid import _transform


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
    elements = set(elements)
    return _transform(dependency_lattice(elements, cover=False).inverse())


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

        self._total = coinformation(self._dist, [list(flatten(self._inputs)), self._output])
        self._reds = {} if reds is None else reds
        self._pis = {} if pis is None else pis


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

