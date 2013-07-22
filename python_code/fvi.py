# Filename: fvi.py
# Author: John Stachurski
# Date: August 2009
# Corresponds to: Listing 6.5

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import linspace, mean, exp, randn
from scipy.optimize import fminbound

from lininterp import LinInterp     # From listing 6.4

theta, alpha, rho = 0.5, 0.8, 0.9   # Parameters
U = lambda c: 1 - exp(- theta * c)  # Utility
f = lambda k, z: (k**alpha) * z     # Production
W = exp(randn(1000))                # Draws of shock

gridmax, gridsize = 8, 150
grid = linspace(0, gridmax**1e-1, gridsize)**10


def maximum(h, a, b):
    return float(h(fminbound(lambda x: -h(x), a, b)))


def bellman(w):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp.
    """
    vals = []
    for y in grid:
        h = lambda k: U(y - k) + rho * mean(w(f(k, W)))
        vals.append(maximum(h, 0, y))
    return LinInterp(grid, np.array(vals))

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Exercise  6.2.2
# Figure 10


def iterates(initial_w, n_iters=30):
    """
    Apply operator T to value function and accumulate the results.

    Parameters
    ----------

    initial_w : ufunc over grid
    n_iters : int. Number of times to apply T to the previous value function.

    Returns
    -------

    iters : list of LinInterps
    """
    iters = [initial_w]
    for i in range(n_iters):
        Tv = bellman(iters[i])  # i is previous.
        iters.append(Tv)
    iters.pop(0)  # Leave only instances of LinInterp
    return iters


def plot_iterations(iters):
    """
    Visualze the results of iterates().

    Parameters
    ----------

    iters : list of LinInterps

    Returns
    -------

    axes : list of matplotlib axes.
    """
    axes = [plt.plot(lin.X, lin.Y, color='k', alpha=.15) for lin in iters]
    return axes

#-----------------------------------------------------------------------------
# Figure 11


def get_greedy(w):
    vals = []
    for y in grid:
        h = lambda k: U(y - k) + rho * mean(w(f(k, W)))
        vals.append(maximum(h, 0, y))
    return LinInterp(grid, np.array(vals))

#-----------------------------------------------------------------------------
# Figure 12


# def look_ahead(phi, n=1000):
#     """Take n draws of X_t-1 and let psi = 1 / n sum p (X_t-1^i y)"""
#     inside = alpha * np.log(sigma * exp(x))
#     (1 / n) * sum()


def main():
    # Fig 10
    iters = iterates(lambda x: np.sqrt(x))
    axes = plot_iterations(iters)
    # Fig 11
    w = iters[-1]
    greedy = get_greedy(w)
    plt.figure()
    plt.plot(w.X, w.Y, greedy.X, greedy.Y, w.X, w.X)


if __name__ == '__main__':
    main()
