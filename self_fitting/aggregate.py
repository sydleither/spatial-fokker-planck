"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""

import argparse

import numpy as np
import pandas as pd

from aggregate_fitting_plots import plot_all
from common import get_data_path
from fitting_utils import evaluate_performance, game_parameter_sweep
from pdfs import FokkerPlanck
from mcmc import mcmc


def main():
    """
    Iterate over amw, amw, and sm. Calculate quality of MCMC fit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=100)
    parser.add_argument("-mu", "--mu", type=float, default=0.005)
    parser.add_argument("-c", "--c", type=int, default=1)
    parser.add_argument("-fit_n", "--fit_n", type=int, default=1)
    parser.add_argument("-fit_mu", "--fit_mu", type=int, default=1)
    parser.add_argument("-fit_c", "--fit_c", type=int, default=1)
    parser.add_argument("-t", "--transform", type=str, default="none")
    args = parser.parse_args()

    n = args.n
    mu = args.mu
    c = args.c
    fit_params = [args.fit_n, args.fit_mu, 1, 1, 1, args.fit_c]
    fp = FokkerPlanck().get_fokker_planck(args.transform)
    xdata = np.linspace(0.01, 0.99, 100)
    game_parameters = game_parameter_sweep()

    data = []
    for awm, amw, sm in game_parameters:
        true_params = [n, mu, awm, amw, sm, c]
        ydata = fp(xdata, *true_params)
        walker_ends = np.array(mcmc(fp, xdata, ydata, true_params, fit_params))
        d = evaluate_performance(fp, xdata, ydata, walker_ends, *true_params)
        data.append(d)

    params_str = "_".join([str(x) for x in [n, mu, c] + [args.fit_n, args.fit_mu, args.fit_c]])
    params_str = args.transform + "_" + params_str
    save_loc = get_data_path("self", params_str)
    df = pd.DataFrame(data)
    plot_all(save_loc, df)


if __name__ == "__main__":
    main()
