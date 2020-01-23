"""
Basic unit tests for synergistic disclosure

Fernando Rosas and Pedro Mediano, 2019
"""
import numpy as np
import dit
from dit.multivariate import coinformation

from syndisc import disclosure, disclosure_channel, self_disclosure

def test_xor():
    xor = dit.example_dists.Xor()
    assert(np.isclose(disclosure(xor), 1))

def test_uniform():
    u = dit.distconst.uniform_distribution(3, 2)
    assert(np.isclose(disclosure(u), 0))

def test_nbit_xor():
    def nbit_xor(n):
        u = dit.distconst.uniform_distribution(n-1, 2)
        xorfun = lambda outcome: (np.mod(np.sum(outcome), 2),)
        return dit.insert_rvf(u, xorfun)

    for n in range(3, 6):
        xor = nbit_xor(n)
        assert(np.isclose(disclosure(xor), 1))

def test_self_disclosure():
    u = dit.distconst.uniform_distribution(2,2)
    assert(np.isclose(self_disclosure(u), 1))

def test_ternary_xor():
    # Ternary XOR: y = x1 + x2 (mod 3)
    dist = dit.distconst.uniform(['000','011','022','101','112',
                                  '120','202','210','221'])
    mi = np.log2(3)
    assert(np.isclose(coinformation(dist, [[0],[2]]), 0))
    assert(np.isclose(coinformation(dist, [[1],[2]]), 0))
    assert(np.isclose(coinformation(dist, [[0,1],[2]]), mi))
    assert(np.isclose(disclosure(dist), mi))

def test_channel():
    nb_samples = 5
    for _ in range(nb_samples):
        # Build random dist but with I(X1; X2) = 0
        r = dit.random_distribution(3,2)
        pX, pYgX = r.condition_on([0,1])
        u = dit.distconst.uniform_distribution(2,2)
        dist = dit.cdisthelpers.joint_from_factors(u, pYgX, strict=False)

        # Compute disclosure and extract optimal synergistic channel
        S, C = disclosure_channel(dist)

        # Optimal synergistic channel is XOR and disclosure is I(X1 xor X2; Y)
        xorfun = lambda outcome: (np.mod(outcome[0] + outcome[1], 2),)
        dist = dit.insert_rvf(dist, xorfun)
        assert(np.allclose(C['pYgX'], np.matrix([[1,0,0,1],[0,1,1,0]])))
        assert(np.isclose(S, coinformation(dist, [[2],[3]])))

