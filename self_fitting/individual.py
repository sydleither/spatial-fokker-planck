"""
Run MCMC on input params
"""

import sys

import numpy as np

from common import get_data_path
from fokker_planck import FokkerPlanck, param_names
from individual_fitting_plots import plot_all
from mcmc import mcmc


def main(params):
    """
    Given a set of params (N, mu, awm, amw, sm)
    Generate a distribution using Fokker-Planck equation with those params
    Give the distribution and Fokker-Planck equation to MCMC
    """
    true_params = [float(x) for x in params] + [1]
    fp = FokkerPlanck().fokker_planck_log
    xdata = np.linspace(0.01, 0.99, 100)
    ydata = fp(xdata, *true_params)
    walker_ends = mcmc(fp, xdata, ydata)

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    plot_all(save_loc, fp, walker_ends, xdata, ydata, true_params)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
