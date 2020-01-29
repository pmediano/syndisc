"""
Backend functions to solve algebraic and optimisation problems related to
synergistic channels.

References:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE for details.

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np

from numpy.linalg import svd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist 
from cvxopt import matrix, solvers
from scipy.stats import entropy
import pypoman as pm


def extreme_points(P, Px, SVDmethod='standard'):
    """
    Calculation of extreme points of the polytope S
    Parameters
    ----------
    P : np.ndarray
        binary matrix of transitions
    Px : np.ndarray
        distribution of the dataset
    SVDmethod : str
        'standard' does the full SVD 
        'fast' uses a trick described in Apendix XX of journal paper

    Returns
    -------
    sols : np.ndarray
        N-by-D array with N vertices of the polytope
    """
    if SVDmethod == 'standard':
        U, S, Vh = svd(P)

    elif SVDmethod == 'fast':
        PtimesPt = P*P.transpose()

        # multiplication by 1+epsilon for numerical stability
        U0, S0, Vh0 = svd((1+1e-20)*PtimesPt)
        S = np.sqrt(S0)

        # Faster than using U.transpose() as it avoids the transpose operation 
        Vh = Vh0 * P
        
    else:
        raise ValueError("SVDMethod not recognised. Must be either \
                'standard' or 'fast'.")


    # Extract reduced A matrix using a small threshold to avoid numerical
    # problems
    rankP = np.sum(S > 1e-6)
    A = Vh[:rankP,:]
    
    b = np.matmul(A,Px)
    
    # Turn np.matrix into np.array for polytope computations
    A = np.array(A)
    b = np.array(b).flatten()

    # Build polytope and find extremes
    A = np.vstack([A, -A, -np.eye(A.shape[1])])
    b = np.hstack([b, -b, np.zeros(A.shape[1])])
    V = pm.compute_polytope_vertices(A, b)  # <-- magic happens here
    V = np.vstack(V)
    
    # To avoid numerical problems, manually cast all small negative values to 0
    if np.any(V < -1e-6):
        raise RuntimeError("Polytope vertices computation failed \
                (found negative vertices).")
    V[V < 0] = 0
    
    # Normalise the rows of V to 1, since they are conditional PMFs
    V = V/(V.sum(axis=1)[:,np.newaxis])
    
    # Eliminate vertices that are too similar (distance less than 1e-10)
    SS = cdist(V,V)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    
    return V[idx]


def lp_sol(c, A, b, mode='cvx'):
    """
    Solve the linear programme given by 
    
    min cT*x subject to Ax = b and x > 0.
    
    Parameters
    ----------
    c : np.ndarray
    A : np.ndarray
    b : np.ndarray
    mode : str
        'cvx' is the standard. 'scipy' uses the scipy solver (not recommended).

    Returns
    -------
    u : np.ndarray
        D-by-1 array that achieves the minimum
    minH : float
        minimum value of the objective (in this case an entropy)
    """
    
    if mode=='cvx':
        c = matrix(c)
        G = matrix(-np.eye(len(c)))
        h = matrix(np.zeros(len(c))) 
        A = matrix(A)
        b = matrix(b)
    
        options={'glpk':{'msg_lev':'GLP_MSG_OFF'}}
        res = solvers.lp(c, G, h, A, b, solver='glpk', options=options)
        u = np.array(res['x']).flatten()
        minH = res['dual objective']
               
    elif mode=='scipy':
        res = linprog(c, A_eq=A, b_eq=b)
        u = res['x']
        minH = res['fun']

    return u, minH


def syn_solve(P, Px, PWgX=None, SVDmethod='standard', extremes='normal'):
    """
    Computes the synergistic self-disclosure capacity, and provides the optimal
    disclosure mapping.

    Parameters
    ----------
    P : np.ndarray
        binary (usually rectangular) matrix of transition
    Px : np.ndarray
        column vector with the pmf of X^n
    PWgX : np.ndarray
        conditional probability of W given X. If none, one computes the self-disclosure ("synergistic entropy" according to Quax)
    SVDmethod : str
        method for addressing the SVD of P (check help in extreme_points)
    extremes : str
        method for finding the extreme values of the polytope
            'normal' is an exaustive search,
            'fast' is just looking for a specific subset of directions (see help of function extreme_points, only valid for binary variables)
    
    Returns 
    -------
    Is : float
        synergistic disclosure capacity
    pYgX : np.ndarray
        optimal disclosure mapping
        
    """
    
    # this is for numerical stability (avoid singular matrices and so)
    Px = Px + 10**-40
    Px = Px / Px.sum()

    if np.any(PWgX) == None:
        PWgX = np.eye( len(Px) )
        
    Px = np.array( [Px] ).T # transform vector input into a nx1 array

    ## Find the extremes of the channel polytope
    ext = extreme_points(P, Px, SVDmethod)
    
    if len(ext)==0:
        print('Damn, the optimization didn\'t work well...')
        return np.nan
    
    # find the c's
    dist_for_ent = np.matmul(PWgX,ext.T)
    c = [entropy(dist_for_ent[:,i],base=2) for i in range(len(ext))]
    
    ## Given these extremal entropies, solve the LP to find which distribution
    # minimises the joint entropy
    u, minH = lp_sol(np.array(c), ext.T, Px)
    
    if minH is None:
        raise RuntimeError("Optimisation did not work well.")

    elif (np.isnan(u).any()):
        raise RuntimeError('Optimiser error: ' + res['message'])
        
    # compute I_s
    Is = entropy(np.matmul(PWgX,Px),base=2) - minH
    Is = Is[0]
    
    # compute mappings
    u = quantize(u,1e-12)
    ext_nonzero = ext[np.nonzero(u)]
    PXgY = quantize(ext_nonzero.transpose(),1e-4)

    u_nonzero = u[np.nonzero(u)]
    pYgX = np.diag(u_nonzero)@ext_nonzero@np.linalg.inv(np.diagflat(Px.T))

    # Return full channel dictionary
    # (Note: change between Y and V is due to a mismatch between old notation
    # in TIFS/WIFS papers and newer notation in PID paper.)
    channel_dict = {'pV': u, 'pXgV': PXgY, 'pVgX': pYgX}

    return Is, channel_dict


def quantize(array,pres=0.01):
    """
    Function to reduce the decimal precision of an input array.

    Parameters
    ----------
    array : np.array
        numerical array to be quantized
        
    pres : float
        desired precision

    Returns
    -------
    np.array with quantized values
    """
    return ((pres**(-1))*array).round()*pres


