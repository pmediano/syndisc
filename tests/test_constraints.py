"""
Unit tests for funky channel constraints.

Fernando Rosas and Pedro Mediano, 2019
"""
import numpy as np
import dit
from dit.multivariate import coinformation, entropy

from syndisc import disclosure, self_disclosure
from syndisc.syndisc import build_constraint_matrix

def test_no_constraints():
    """Test that unconstrained disclosure is MI."""
    nb_reps = 10
    for _ in range(nb_reps):
        d   = dit.random_distribution(3, 2)
        MI  = coinformation(d, [[0,1], [2]])
        syn = disclosure(d, cons=[])
        assert(np.isclose(MI, syn))


def test_all_constraints():
    """Test that fully constrained disclosure is zero."""
    nb_reps = 10
    for _ in range(nb_reps):
        d   = dit.random_distribution(3, 2)
        syn = disclosure(d, cons=[[0,1]])
        assert(np.isclose(syn, 0))


def test_no_constraints_self_disclosure():
    """Test that unconstrained self-disclosure is joint entropy."""
    nb_reps = 10
    for _ in range(nb_reps):
        d   = dit.random_distribution(3, 2)
        H   = entropy(d)
        syn = self_disclosure(d, cons=[])
        assert(np.isclose(H, syn))

def test_combined_variable():
    """Test that using a coalesced input gives the same result as constraining
    on a bivariate marginal."""
    d = dit.random_distribution(4, 2)
    r1 = disclosure(d, cons=[[0,1],[2]], output=[3])
    r2 = disclosure(d.coalesce([[0,1],[2],[3]]))
    assert(np.isclose(r1, r2))

    # Same, but on self-disclosure
    d = dit.random_distribution(3, 2)
    r1 = self_disclosure(d, cons=[[0,1],[2]])
    r2 = self_disclosure(d.coalesce([[0,1],[2]]))
    assert(np.isclose(r1, r2))
