# Filename: srs.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 6.1

import matplotlib.pyplot as plt


class SRS:

    def __init__(self, F=None, phi=None, X=None):
        """Represents X_{t+1} = F(X_t, W_{t+1}); W ~ phi.
        Parameters: F and phi are functions, where phi()
        returns a draw from phi. X is a number representing
        the initial condition."""
        self.F, self.phi, self.X, self._init_X = F, phi, X, X

    def update(self):
        "Update the state according to X = F(X, W)."
        self.X = self.F(self.X, self.phi())

    def sample_path(self, n):
        "Generate path of length n from current state."
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        return path

    def sample_path_default(self, n):
        "Generate path of length n from current state, starting at X=1"
        self.X = self._init_X
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        return path

    def example(self, n_time=500, n_simul=100):
        "Call with no arguments to see the class in action."
        res = [self.sample_path_default(n_time) for x in range(n_simul)]
        axes = [plt.plot(x, alpha=.15, color='k') for x in res]
        return axes
