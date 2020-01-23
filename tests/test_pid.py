"""
Unit tests for information decomposition based on information disclosure.

Fernando Rosas and Pedro Mediano, 2019
"""
import numpy as np
import dit

from syndisc.pid import PID_SD

def assert_only_atom(pid, node, val):
    """
    Asserts that there is only one node with non-zero partial information.
    Assertion is satisfied if `node` has partial information `val` and all
    other atoms have zero PI. Otherwise, an AssertionError is thrown.

    Parameters
    ----------
    pid : dit.BasePID
        PID to verify
    node : tuple(tuples)
        The lattice node to check for non-zero PI
    val : float
        Expected value of PI at `node`
    """
    for n in pid._lattice.nodes:
        if n == node:
            assert(np.isclose(pid.get_partial(n), val))
        else:
            assert(np.isclose(pid.get_partial(n), 0))


def test_xor():
    xor = dit.example_dists.Xor()
    pid = PID_SD(xor)
    assert_only_atom(pid, ((0,),(1,)), 1)

def test_4bit_xor():
    u = dit.distconst.uniform_distribution(3, 2)
    xorfun = lambda outcome: (np.mod(np.sum(outcome), 2),)
    dist = dit.insert_rvf(u, xorfun)
    pid = PID_SD(dist)
    assert_only_atom(pid, ((0,1),(0,2),(1,2)), 1)

def test_copy_x0():
    # Target is a copy of x0 -- can be disclosed while keeping x1 private
    u = dit.distconst.uniform_distribution(2, 2)
    dist = u.coalesce([[0,1,0]], extract=True)
    pid = PID_SD(dist)
    assert_only_atom(pid, ((1,),), 1)

def test_giant_bit():
    # All sources and target are copies of the same bit
    dist = dit.example_dists.giant_bit(4)
    pid = PID_SD(dist)
    assert_only_atom(pid, ((),), 1)

def test_uniform():
    # Everything is independent from everything
    dist = dit.distconst.uniform_distribution(4, 2)
    pid = PID_SD(dist)
    assert_only_atom(pid, ((0,),), 0)

def test_null():
    # Inputs are correlated between them but independent from output
    dist = dit.pid.distributions.trivariate.null
    pid = PID_SD(dist)
    assert_only_atom(pid, ((0,),), 0)

