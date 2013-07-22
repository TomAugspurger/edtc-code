# Filename: fpi.py
# Author: John Stachurski
# Date: August 2009
# Corresponds to: Listing 6.6
from scipy import absolute as abs

from fvi import *  # Import all definitions from listing 6.5


def maximizer(h, a, b):
    return float(fminbound(lambda x: -h(x), a, b))


def T(sigma, w):
    "Implements the operator L T_sigma."
    vals = []
    for y in grid:
        Tw_y = U(y - sigma(y)) + rho * mean(w(f(sigma(y), W)))
        vals.append(Tw_y)
    return LinInterp(grid, np.array(vals))


def get_greedy(w):
    "Computes a w-greedy policy."
    vals = []
    for y in grid:
        h = lambda k: U(y - k) + rho * mean(w(f(k, W)))
        vals.append(maximizer(h, 0, y))
    return LinInterp(grid, vals)


def get_value(sigma, v):
    """Computes an approximation to v_sigma, the value
    of following policy sigma. Function v is a guess.
    """
    tol = 1e-5         # Error tolerance
    while 1:
        new_v = T(sigma, v)
        err = max(abs(new_v(grid) - v(grid)))
        if err < tol:
            return new_v
        v = new_v

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# My implementation:

w = lambda x: np.sqrt(x)
sigma = get_greedy(w)


def iterates(sigma, w, n_iters=30):
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
    iters = [w]
    for i in range(n_iters):
        Tv = T(sigma, iters[i])  # i is previous.
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
