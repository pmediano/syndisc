SYNDISC: SYNergistic information via data DISClosure
====================================================

This package provides complete functionality to compute synergistic information
via privacy-preserving data disclosure. This repository contains the companion
software for the paper:

F. Rosas\*, P. Mediano\*, B. Rassouli and A. Barrett (2019). An operational
information decomposition via synergistic disclosure. https://arxiv.org/abs/2001.10387

Please cite the paper (and give us a shout!) if you use this software. Please
contact Pedro Mediano for bug reports, pull requests, and feature requests.

Description and basic intuitions
--------------------------------

Under the hood, the computation of synergistic disclosure proceeds in two steps:

1. Compute the set of synergistic channels, p(V|X), that do not disclose
information about any of the individual variables X_i. That is: I(V; X_i) = 0
for all i.

2. Compute synergistic disclosure as the maximum information about Y that can
be obtained through a synergistic channel on X. That is, max I(V; Y) subject to
V being a synergistic observable of X. In other words: how much information can
X provide about Y without "compromising" the individual X_i's?

In practice, step 1 is computed using tools from computational geometry (since
the set of synergistic channels is related to the vertices of a polytope), and
step 2 can be optimised with a standard Linear Programming (LP) solver.

This idea can be readily generalised to arbitrary ``constraint sets'' -- sets
of variables the information of which cannot be disclosed. For example, for
n=3 source variables, C(X, {1}{2}{3}) is the set of channels that does not
provide any information about any individual variable; and C(X, {1,2}) is the
set of channels that provides no information about the joint distribution of
X_1 and X_2 (but may disclose information about X_3).

See the main paper and references below for a detailed description of the
measure.

Examples
--------

The package aims to provide a lean interface to the key functions that compute
the synergistic disclosure as described in the paper above. The main point of
access to the code is the `disclosure()` function, which takes a
`dit.Distribution` as argument:

```python
import dit
from syndisc import disclosure

XOR = dit.example_dists.Xor()
print('Synergy in XOR: ', disclosure(XOR))

COPY = dit.example_dists.giant_bit()
print('Synergy in COPY: ', disclosure(COPY))
```

By default, for a distribution of n+1 variables, `disclosure()` uses the
first n as sources, the last one as target, and uses the constraint set
`{1}{2}...{n}`. These can be changed via optional arguments:

```python
dblxor = dit.pid.distributions.trivariate.dblxor
disclosure(dblxor)
disclosure(dblxor, cons=[[0,1]])
```

The practical limit of this algorithm is on around n=7 binary source
variables, after which the computation of the set of synergistic channels
becomes infeasible.

Finally, the package also implements a full information decomposition similar
to Williams and Beer's Partial Information Decomposition (PID):

```python
from syndisc.pid import PID_SD

AND = dit.example_dists.And()
print(PID_SD(AND))
```

Further examples can be found in the `examples/` folder.


Download and installation
-------------------------

The package can be installed via standard `distutils` functionality.

```
$ python setup.py install
```

This may need root (i.e. `sudo`) permissions, depending on your Python
environment configuration.

Pytest-compatible tests are provided in the `tests/` subfolder.

Requirements
------------

This package depends heavily on two other packages:

* `dit`, a general-purpose information theory package used for general handling
of probability distributions.

* `pypoman`, a computational geometry package used to find the set of
synergistic channels for a set of sources.

In addition, it depends on other common packages (i.e. numpy, scipy, etc). All
of these are specified in the `setup.py` file.

Licence
-------

This software is distributed under the modified 3-clause BSD Licence.

Further reading
---------------

* B. Rassouli\*, F. Rosas\*, D. Gunduz (2019). Data disclosure under perfect
  sample privacy. TIFS.

* P. Williams and R. Beer (2010). Nonnegative decomposition of multivariate
  information.


\(C\) Pedro Mediano and Fernando Rosas, 2019-2020
