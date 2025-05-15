"""
Run MCMC on input params
"""

import sys

import numpy as np

from common import get_data_path
from fokker_planck import FokkerPlanck, param_names
from mcmc_utils import mcmc, plot_walker_curves, plot_walker_curve_mse, plot_walker_gamespace, plot_walker_params


def main(params):
    """
    Test fitting when log-transforming FP vs not
    Given a set of params (N, mu, awm, amw, sm)
    Generate a distribution using Fokker-Planck equation with those params
    Give the distribution and Fokker-Planck equation to MCMC
    """
    params = [float(x) for x in params]
    n = int(params[0])
    mu = params[1]
    true_params = params[2:]

    # log U(x)
    fp = FokkerPlanck(n, mu).fokker_planck_log
    xdata = np.linspace(0.01, 0.99, n)
    ydata = fp(xdata, *true_params)
    walker_ends = mcmc(fp, xdata, ydata)
    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str+"/log")
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata, True, True)
    plot_walker_curve_mse(save_loc, fp, walker_ends, xdata, ydata, True, True)
    plot_walker_gamespace(save_loc, walker_ends, true_params)
    plot_walker_params(save_loc, walker_ends)

    # not log rho
    fp = FokkerPlanck(n, mu).fokker_planck
    xdata = np.linspace(0.01, 0.99, n)
    ydata = fp(xdata, *true_params)
    walker_ends = mcmc(fp, xdata, ydata)
    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str+"/notlog")
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata, False, True)
    plot_walker_curve_mse(save_loc, fp, walker_ends, xdata, ydata, False, True)
    plot_walker_gamespace(save_loc, walker_ends, true_params)
    plot_walker_params(save_loc, walker_ends)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
