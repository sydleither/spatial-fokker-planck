'''
Run MCMC on input params
'''
import sys

import numpy as np

from fokker_planck import FokkerPlanck, param_names
from mcmc_utils import (mcmc, plot_walker_curves,
                        plot_walker_gamespace, plot_walker_params)


def main(params):
    '''
    Given a set of params (N, mu, awm, amw, sm)
    Generate a distribution using Fokker-Planck equation with those params
    Give the distribution and Fokker-Planck equation to MCMC
    '''
    params = [float(x) for x in params]
    n = int(params[0])
    mu = params[1]
    true_params = params[2:]

    save_loc = "."
    fp = FokkerPlanck(n, mu).fokker_planck
    xdata = np.linspace(0.01, 0.99, n)
    ydata = fp(xdata, *true_params)
    walker_ends = mcmc(fp, xdata, ydata)

    file_name = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    plot_walker_curves(save_loc, file_name, fp, walker_ends, xdata, ydata)
    plot_walker_gamespace(save_loc, file_name, walker_ends, true_params)
    plot_walker_params(save_loc, file_name, walker_ends)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
