"""
Run MCMC on spatial data
"""

import argparse

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import calculate_fp_params, game_colors, get_data_path, theme_colors
from pdfs import FokkerPlanck, param_names, SpatialSubsample
from in_silico_fitting.abm_utils import read_sample
from individual_fitting_plots import plot_all, plot_trace
from mcmc import mcmc


def fp_vs_spsb(save_loc, xdata, spbp_ydata, fp_ydata, title):
    """
    Visualize FP curve against spatial subsample curve with same parameters.
    """
    fig, ax = plt.subplots()
    ax.plot(xdata, spbp_ydata, color=theme_colors[0], label="Spatial Subsample")
    ax.plot(xdata, fp_ydata, color=theme_colors[1], label="Fokker-Planck")
    ax.set(xlabel="Fraction Mutant", ylabel="Probability Density")
    ax.set_title(title)
    fig.legend()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/spsb_vs_fp.png", bbox_inches="tight")
    plt.close()


def main():
    """
    Given data type, source, sample, and subsample length
    Generate a spatial subsample distribution using the spatial data
    Fit Fokker-Planck to the distribution using MCMC
    """
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str, default="none")
    parser.add_argument("-d", "--data_type", type=str, default="in_vitro")
    parser.add_argument("-src", "--source", type=str, default="pc9")
    parser.add_argument("-sam", "--sample", type=str, default="0")
    parser.add_argument("-sub", "--subsample_length", type=int, default=10)
    parser.add_argument("-w", "--walkers", type=int, default=100)
    parser.add_argument("-i", "--iterations", type=int, default=5000)
    args = parser.parse_args()

    # Define functions and variables
    data_path = get_data_path(f"{args.data_type}/{args.source}", "processed")
    config = pd.read_csv(f"{data_path}/labels.csv")
    config = config.set_index(["source", "sample"])
    df = pd.read_csv(f"{data_path}/{args.sample}.csv")
    s_coords = df.loc[df["type"] == "sensitive"][["x", "y"]].values
    r_coords = df.loc[df["type"] == "resistant"][["x", "y"]].values
    source, sample = args.sample.split(" ")
    awm, amw, sm = calculate_fp_params(
        config.loc[source, sample]["a"],
        config.loc[source, sample]["b"],
        config.loc[source, sample]["c"],
        config.loc[source, sample]["d"],
    )
    fp = FokkerPlanck().get_fokker_planck(args.transform)
    spsb = SpatialSubsample().get_spatial_subsample(args.transform)
    xdata, ydata = spsb(s_coords, r_coords, args.subsample_length)

    # Run MCMC
    params = [100, 0.01, round(awm, 3), round(amw, 3), round(sm, 3), 1]
    fit_params = [1, 1, 1, 1, 1, 1]
    sampler, walker_ends = mcmc(
        fp,
        xdata,
        ydata,
        params,
        fit_params,
        args.walkers,
        args.iterations,
        return_sampler=True,
    )

    # Save MCMC results
    save_loc = get_data_path(
        f"{args.data_type}/{args.source}", f"images/{args.sample}/{args.subsample_length}"
    )
    plot_all(save_loc, fp, walker_ends, xdata, ydata, params, fit_params)
    title = " ".join([f"{param_names[i]}={round(params[i], 3)}" for i in range(len(params))])
    fp_vs_spsb(save_loc, xdata, ydata, fp(xdata, *params), title)
    plot_trace(save_loc, sampler, params, fit_params)


if __name__ == "__main__":
    main()
