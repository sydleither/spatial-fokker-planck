"""
Holds the FokkerPlanck and SpatialSubsample classes to handle different versions.
"""

from random import choices

import numpy as np
from scipy import stats


param_names = ["n", "mu", "awm", "amw", "sm", "c"]


class FokkerPlanck:
    """
    FokkerPlanck class to handle equation.
    """
    def fokker_planck(self, x, n, mu, awm, amw, sm, c):
        """
        The Fokker-Planck equation as defined in Barker-Clarke et al., 2024
        x = 0 or x = 1 will break the equation due to the log
        """
        if awm == 0:
            fx = sm * x
        else:
            fx = ((((1+sm)*awm + (1+awm)*amw)/awm**2) * np.log(1+awm*x)) - ((awm+amw)/awm)*x
        phi = (1 - 2*n*mu) * np.log(x*(1-x)) - 2*n*fx
        phi = np.clip(phi, -700, 700)
        rho = 2*n*np.exp(-phi)
        return c*rho

    def fokker_planck_normalized(self, x, n, mu, awm, amw, sm, c):
        """
        Return the normalized rho.
        """
        rho = self.fokker_planck(x, n, mu, awm, amw, sm, c)
        rho = rho / max(rho)
        return rho

    def fokker_planck_density(self, x, n, mu, awm, amw, sm, c):
        """
        Return the probability density of rho.
        """
        rho = self.fokker_planck(x, n, mu, awm, amw, sm, c)
        rho = rho / (np.sum(rho) * x[0])
        return rho

    def fokker_planck_log(self, x, n, mu, awm, amw, sm, c):
        """
        Return the negative log-space phi.
        """
        if awm == 0:
            fx = sm * x
        else:
            fx = ((((1+sm)*awm + (1+awm)*amw)/awm**2) * np.log(1+awm*x)) - ((awm+amw)/awm)*x
        phi = (1 - 2*n*mu) * np.log(x*(1-x)) - 2*n*fx - np.log(2*n)
        neg_lnrho = -np.log(c) + phi
        return neg_lnrho

    def get_fokker_planck(self, transform):
        """
        Return the specified transform of the Fokker Planck equation.
        """
        if transform is None or transform == "none":
            return self.fokker_planck
        if transform == "log":
            return self.fokker_planck_log
        if transform == "norm":
            return self.fokker_planck_normalized
        if transform == "density":
            return self.fokker_planck_density
        raise ValueError("Unknown transformation of the Fokker-Plank equation")


class SpatialSubsample:
    """
    Spatial Subsample PDF.
    """
    def spatial_subsample(self, s_coords, r_coords, sample_length, num_samples=5000):
        """
        Create spatial subsample support and distribution.
        """
        dims = range(len(s_coords[0]))
        max_dims = [max(np.max(s_coords[:, i]), np.max(r_coords[:, i])) for i in dims]
        dim_vals = [choices(range(0, max_dims[i] - sample_length), k=num_samples) for i in dims]
        fr_counts = []
        for s in range(num_samples):
            ld = [dim_vals[i][s] for i in dims]
            ud = [ld[i] + sample_length for i in dims]
            subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
            subset_s = np.sum(np.all(subset_s, axis=0))
            subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
            subset_r = np.sum(np.all(subset_r, axis=0))
            subset_total = subset_s + subset_r
            if subset_total == 0:
                continue
            fr_counts.append(subset_r / subset_total)
        xdata = np.linspace(max(min(fr_counts), 0.001), min(max(fr_counts), 0.999), 100)
        kde = stats.gaussian_kde(fr_counts)
        pdf = kde(xdata)
        # pdf, bin_edges = np.histogram(fr_counts, bins=101, range=(max(min(fr_counts), 0.001), min(max(fr_counts), 0.999)), density=True)
        # xdata = (bin_edges[:-1] + bin_edges[1:]) / 2
        return xdata, pdf
    
    def spatial_subsample_log(self, s_coords, r_coords, sample_length, num_samples=5000):
        """
        Return the negative log-space pdf.
        """
        xdata, pdf = self.spatial_subsample(s_coords, r_coords, sample_length, num_samples)
        return xdata, -np.log(pdf)

    def spatial_subsample_normalized(self, s_coords, r_coords, sample_length, num_samples=5000):
        """
        Return the normalized pdf.
        """
        xdata, pdf = self.spatial_subsample(s_coords, r_coords, sample_length, num_samples)
        return xdata, pdf / max(pdf)

    def get_spatial_subsample(self, transform):
        """
        Return the specified transform of the Fokker Planck equation.
        """
        if transform is None or transform == "none":
            return self.spatial_subsample
        if transform == "log":
            return self.spatial_subsample_log
        if transform == "norm":
            return self.spatial_subsample_normalized
        raise ValueError("Unknown transformation of the Spatial Subsample PDF")
