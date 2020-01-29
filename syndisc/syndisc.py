"""
Synergistic disclosure and self-disclosure in discrete random variables.

References:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE for details.

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np
import dit

from .solver import syn_solve

def disclosure(dist, cons=None, output=None):
    """
    Compute the synergistic disclosure of a given probability distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the decomposition on.
    cons : iter of iters, None
        List of constraints to preserve when computing disclosure. Defaults to
        preserving all individual marginals.
    output : iter, None
        The output variable. If None, `dist.rvs[-1]` is used.

    Returns
    -------
    I_s : float
        disclosable information under given constraints, in bits
    """
    return disclosure_channel(dist, cons, output)[0]


def disclosure_channel(dist, cons=None, output=None):
    """
    Compute the synergistic disclosure of a given probability distribution, and
    return a dictionary with the optimal synergistic channel.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the decomposition on.
    cons : iter of iters, None
        List of constraints to preserve when computing disclosure. Defaults to
        preserving all individual marginals. (Note: if `output` is given, first
        the distribution is reshaped to move `output` to the last position, and
        then the constraints are applied.)
    output : iter, None
        The output variable. If None, `dist.rvs[-1]` is used.

    Returns
    -------
    Is : float
        disclosable information under given constraints, in bits
    channel_dict : dictionary
        Dict with forward, backward, and marginal PDFs for the optimal
        synergistic channel
    """
    if output is not None:
        # Reshape dist to move `output` to the last position
        inputs = [var for var in dist.rvs if var[0] not in output]
        dist = dist.coalesce([list(flatten(inputs)) + list(output)], extract=True)

    output = dist.rvs[-1]
    inputs = dist.rvs[:-1]

    if cons is None:
        cons = dist.rvs[:-1]

    # With no constraints, shortcut and return full mutual info directly
    if len(cons) == 0:
        MI = dit.multivariate.coinformation(dist, 
                                        [flatten(inputs), output])
        input_alphabet  = np.prod([len(dist.alphabet[i[0]]) for i in inputs])
        channel = np.eye(input_alphabet)
        return MI, channel


    pX, pWgX = dist.condition_on(flatten(inputs))

    # Make all distributions dense before extracting the PMFs
    pX.make_dense()
    for p in pWgX:
        p.make_dense()

    output_alphabet = len(dist.alphabet[output[0]])
    input_alphabet  = np.prod([len(dist.alphabet[i[0]]) for i in inputs])
    pWgX_ndarray = np.zeros((output_alphabet, input_alphabet))

    count = 0
    for i,(_,p) in enumerate(pX.zipped()):
        if p > 0:
            pWgX_ndarray[:,i] = pWgX[count].pmf
            count = count + 1

    P = build_constraint_matrix(cons, dist.coalesce(inputs))

    return syn_solve(P, pX.pmf, pWgX_ndarray)


def self_disclosure(dist, cons=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution to compute the decomposition on.
    cons : iter of iters, None
        List of constraints to preserve when computing self-disclosure.
        Defaults to preserving all individual marginals.

    Returns
    -------
    Is : float
        disclosable information under given constraints, in bits
    """
    return self_disclosure_channel(dist, cons)[0]


def self_disclosure_channel(dist, cons=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution to compute the decomposition on.
    cons : iter of iters, None
        List of constraints to preserve when computing self-disclosure.
        Defaults to preserving all individual marginals.

    Returns
    -------
    Is : float
        disclosable information under given constraints, in bits
    channel_dict : dictionary
        Dict with forward, backward, and marginal PDFs for the optimal
        synergistic channel
    """
    if cons is None:
        cons = dist.rvs

    # Make new distribution with the concatenation of all variables appended
    idx = dist.rvs + [list(range(dist.outcome_length()))]
    concat = dist.coalesce(idx)

    return disclosure_channel(concat, cons)


def build_constraint_matrix(cons, d):
    """
    Build constraint matrix.

    The constraint matrix is a matrix P that is the vertical stack
    of all the preserved marginals.

    Parameters
    ----------
    cons : iter of iter
        List of variable indices to preserve.
    d : dit.Distribution
        Distribution for which to design the constraints

    Returns
    -------
    P : np.ndarray
        Constraint matrix

    """
    # Initialise a uniform distribution to make sure it has full support
    u = dit.distconst.uniform_like(d)
    n = len(u.rvs)
    l = u.rvs
    u = u.coalesce(l + l)

    # Generate one set of rows of P per constraint
    P_list = []
    for c in cons:
        pX123, pX1gX123 = u.condition_on(crvs=range(n, 2*n), rvs=c)

        pX123.make_dense()
        for p in pX1gX123:
          p.make_dense()

        P_list.append(np.hstack([p.pmf[:,np.newaxis] for p in pX1gX123]))

    # Stack rows and return
    P = np.vstack(P_list)

    return P


def flatten(l):
    """Utility function to flatten an iter of iters into an iter."""
    return [item for sublist in l for item in sublist]

