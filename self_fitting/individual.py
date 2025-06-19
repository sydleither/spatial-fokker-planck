"""
Run MCMC on input params
"""

import sys

import numpy as np

from common import get_data_path
from fokker_planck import FokkerPlanck, param_names
from individual_fitting_plots import (
    plot_walker_curves,
    plot_walker_curve_mse,
    plot_walker_gamespace,
    plot_walker_pairplot,
    plot_walker_gameparams,
)
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
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata, True)
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata, False)
    plot_walker_curve_mse(save_loc, fp, walker_ends, xdata, ydata)
    plot_walker_gamespace(save_loc, walker_ends, true_params[2:5])
    plot_walker_pairplot(save_loc, walker_ends)
    plot_walker_gameparams(save_loc, walker_ends, true_params)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
