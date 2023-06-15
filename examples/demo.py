"""
Basic demo code for syndisc package. Compute synergistic disclosure information
in a few probability distributions of interest. Reproduces Figure 1 and Table 1
of:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

Pedro Mediano and Fernando Rosas, 2019
"""

import numpy as np
import dit
import pandas as pd
from syndisc import pid
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

# XOR
xor = dit.example_dists.Xor()
print(pid.PID_SD(xor))

# AND
AND = dit.example_dists.And()
print(pid.PID_SD(AND))

# Two-bit copy
tbc = dit.pid.distributions.bivariate.cat
print(pid.PID_SD(tbc))

# Unique info from X1
unq1 = dit.pid.distributions.bivariate.unq1
print(pid.PID_SD(unq1))

# AND gate with correated inputs
a = 0.5
b = 0.5
def CorrelatedAND(r):
    Px = dit.Distribution(['000','010','100','111'], [1-a-b+r, b-r, a-r, r])
    lat = pid.PID_SD(Px)
    return np.array([r,
                     lat.get_pi(())/lat._total,
                     lat.get_pi(((0,),(1,)))/lat._total,
                     ])

nb_samples = 100
and_pid = np.zeros((nb_samples, 3))
R = np.min([a, 1-b])
r_vec = np.linspace(max(a*b, a+b-1), min(a,b), nb_samples)
for i,rr in enumerate(r_vec):
    and_pid[i,:] = CorrelatedAND(rr)
df = pd.DataFrame(and_pid, columns=['r', '{}', '{0}{1}'])

df.plot(x='r')
plt.show()

