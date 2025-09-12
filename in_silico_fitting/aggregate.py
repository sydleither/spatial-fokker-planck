"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""

import argparse
import os

import pandas as pd

from abm_utils import read_sample
from aggregate_fitting_plots import plot_all
from common import calculate_fp_params, get_data_path
from fitting_utils import evaluate_performance
from pdfs import FokkerPlanck, SpatialSubsample
from mcmc import mcmc


def main():
    """
    Iterate over ABM spatial subsample distributions. Calculate quality of MCMC fit.
    """
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str, default="none")
    parser.add_argument("-d", "--data_type", type=str, default="in_silico")
    parser.add_argument("-src", "--source", type=str, default="5_5")
    parser.add_argument("-sub", "--subsample_length", type=int, default=5)
    parser.add_argument("-w", "--walkers", type=int, default=100)
    parser.add_argument("-i", "--iterations", type=int, default=50000)
    args = parser.parse_args()

    # Define functions and variables
    data_path = get_data_path(f"{args.data_type}/{args.source}", "raw")
    fp = FokkerPlanck().get_fokker_planck(args.transform)
    spsb = SpatialSubsample().get_spatial_subsample(args.transform)
    fit_params = [1, 1, 1, 1, 1, 1]
    n = 100
    mu = 0.01
    c = 1
    subsample_length = args.subsample_length
    walkers = args.walkers
    iterations = args.iterations

    # Iterate over samples and calculate fit quality
    data_path = get_data_path(f"{args.data_type}/{args.source}", "raw")
    data = []
    for sample in data_path:
        if not os.path.isdir({data_path} / {sample}):
            continue
        s_coords, r_coords, config = read_sample(data_path, sample)
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        params = [n, mu, awm, amw, sm, c]
        xdata, ydata = spsb(s_coords, r_coords, subsample_length)
        walker_ends = mcmc(fp, xdata, ydata, params, fit_params, walkers, iterations)
        d = evaluate_performance(fp, xdata, ydata, walker_ends, n, mu, awm, amw, sm, c)
        data.append(d)

    # Save results
    save_loc = get_data_path(f"{args.data_type}/{args.source}", "images")
    df = pd.DataFrame(data)
    plot_all(save_loc, df)


if __name__ == "__main__":
    main()
