"""
Compute and plot self-disclosure of pairs of corelated bits. Reproduces Figure
5 of:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

Fernando Rosas and Pedro Mediano, 2019
"""

import numpy as np
import dit
import pandas as pd
from syndisc import self_disclosure
import matplotlib.pyplot as plt

# Define relevant variables and start loop
nb_samples = 100
m_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # marginal pdf
for i,m in enumerate(m_vec):

    # Vector with possible correlations, given marginals (a,b)
    R = np.min([m, 1-m])
    r_vec = np.linspace(max(m*m, 2*m-1), m, nb_samples)

    # For every feasible correlation, instantiate dit.Distribution and compute
    Is = np.zeros_like(r_vec)
    MI = np.zeros_like(r_vec)
    for j,r in enumerate(r_vec):
        Px = dit.Distribution(['00','01','10','11'], [1-2*m+r, m-r, m-r, r])
        MI[j] = dit.multivariate.coinformation(Px, [[0],[1]])
        Is[j] = self_disclosure(Px)

    plt.plot(MI, Is, label='m = %.1f'%m)

# Customise and plot
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.xlabel('MI')
plt.ylabel('Self-disclosure')
plt.legend()
plt.show()

