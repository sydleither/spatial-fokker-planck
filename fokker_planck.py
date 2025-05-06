"""
Holds the FokkerPlanck class to handle different versions of the function and fixed variables
"""

import numpy as np


param_names = ["N", "mu", "awm", "amw", "sm"]


class FokkerPlanck:
    """
    FokkerPlanck class to handle equation
    """

    def __init__(self, n, mu):
        self.n = n
        self.mu = mu

    def fokker_planck(self, x, awm, amw, sm):
        """
        The Fokker-Planck equation as defined in Barker-Clarke et al., 2024
        x = 0 or x = 1 will break the equation due to the log
        """
        n = self.n
        mu = self.mu
        if awm == 0:
            fx = sm * x
        else:
            fx = (((1 + sm) * awm + (1 + awm) * amw) / awm**2) * np.log(1 + awm * x) - ((awm + amw) / awm) * x
        phi = (1 - 2 * n * mu) * np.log(x * (1 - x)) - 2 * n * fx - np.log(2 * n)
        rho = np.exp(-phi)
        return rho

    def fokker_planck_verbose(self, x, awm, amw, sm):
        """
        Print the parameters and a sample of rho.
        """
        rho = self.fokker_planck(x, awm, amw, sm)
        print(f"({self.n}, {self.mu}, {awm}, {amw}, {sm})")
        print(list((rho)[0::10]))
        print()
        return rho

    def fokker_planck_normalized(self, x, awm, amw, sm):
        """
        Return the normalized rho.
        """
        rho = self.fokker_planck(x, awm, amw, sm)
        rho = rho / max(rho)
        rho[rho < 1e-31] = 0
        rho[rho > 1e31] = 1e31
        return rho

    def fokker_planck_density(self, x, awm, amw, sm):
        """
        Return the probability density of rho.
        """
        rho = self.fokker_planck(x, awm, amw, sm)
        rho = rho / (np.sum(rho) * x[0])
        return rho
