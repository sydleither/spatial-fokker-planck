"""
Run MCMC on input params
"""

import argparse

import numpy as np

from common import get_data_path
from pdfs import FokkerPlanck
from individual_fitting_plots import plot_all, plot_trace
from mcmc import mcmc


def main():
    """
    Generate a distribution using Fokker-Planck equation with input params
    Give the distribution and Fokker-Planck equation to MCMC
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=100)
    parser.add_argument("-mu", "--mu", type=float, default=0.005)
    parser.add_argument("-awm", "--awm", type=float, default=0.02)
    parser.add_argument("-amw", "--amw", type=float, default=-0.02)
    parser.add_argument("-sm", "--sm", type=float, default=0.01)
    parser.add_argument("-c", "--c", type=float, default=1)
    parser.add_argument("-fit_n", "--fit_n", type=int, default=1)
    parser.add_argument("-fit_mu", "--fit_mu", type=int, default=1)
    parser.add_argument("-fit_awm", "--fit_awm", type=int, default=1)
    parser.add_argument("-fit_amw", "--fit_amw", type=int, default=1)
    parser.add_argument("-fit_sm", "--fit_sm", type=int, default=1)
    parser.add_argument("-fit_c", "--fit_c", type=int, default=1)
    parser.add_argument("-t", "--transform", type=str, default="none")
    args = parser.parse_args()

    true_params = [args.n, args.mu, args.awm, args.amw, args.sm, args.c]
    fp = FokkerPlanck().get_fokker_planck(args.transform)
    xdata = np.linspace(0.01, 0.99, 100)
    ydata = fp(xdata, *true_params)

    fit_params = [args.fit_n, args.fit_mu, args.fit_awm, args.fit_amw, args.fit_sm, args.fit_c]
    sampler, walker_ends = mcmc(fp, xdata, ydata, true_params, fit_params, return_sampler=True)

    params_str = "_".join([str(x) for x in true_params + fit_params])
    params_str = args.transform + "_" + params_str
    save_loc = get_data_path("self", params_str)
    plot_all(save_loc, fp, walker_ends, xdata, ydata, true_params)
    plot_trace(save_loc, sampler, true_params, fit_params)


if __name__ == "__main__":
    main()
