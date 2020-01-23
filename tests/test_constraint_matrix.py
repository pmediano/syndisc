"""
Unit tests for channel constraint matrix.

Fernando Rosas and Pedro Mediano, 2019
"""
import numpy as np
import dit

from syndisc.syndisc import build_constraint_matrix


def test_constraint_matrix_2bit():
    """Test that constraint matrix looks as expected for 2 bits."""
    # Constrain only on X1
    u = dit.distconst.uniform_distribution(2,2)
    P = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1]])
    assert(np.allclose(P, build_constraint_matrix([[0]], u)))

    # Constrain only on X2
    P = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]])
    assert(np.allclose(P, build_constraint_matrix([[1]], u)))

    # Constrain on both marginals
    P = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1]])
    assert(np.allclose(P, build_constraint_matrix([[0],[1]], u)))

    # Constrain on the joint
    P = np.eye(4)
    assert(np.allclose(P, build_constraint_matrix([[0,1]], u)))


def test_constraint_matrix_3bit():
    """Test that constraint matrix looks as expected for 3 bits."""
    # Constrain only on X1
    u = dit.distconst.uniform_distribution(3,2)
    P = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1]])
    assert(np.allclose(P, build_constraint_matrix([[0]], u)))


def test_constraint_marginal():
    """Test that the marginals are actually preserved."""
    nb_reps = 10
    u = dit.distconst.uniform_distribution(2,2)
    for _ in range(nb_reps):
        Pjoint = dit.random_distribution(2, 2)
        marg = np.random.choice([[0], [1], [0,1]])
        P = build_constraint_matrix([marg], u)
        Pmarg = Pjoint.marginal(marg)
        Pmarg.make_dense()
        assert(np.allclose(Pmarg.pmf, P @ Pjoint.pmf))


def test_constraints_sets():
    """Test that constraint builder works with dit's FrozenSets as well as with
    lists."""
    pass

