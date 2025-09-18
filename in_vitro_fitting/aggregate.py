"""
In vitro testing
"""

import argparse
import os

import pandas as pd

from aggregate_fitting_plots import get_confidence_interval_str
from common import calculate_fp_params, get_data_path
from fitting_utils import evaluate_performance
from pdfs import FokkerPlanck, param_names, SpatialSubsample
from mcmc import mcmc


def main():
    """
    In vitro testing
    """
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str, default="none")
    parser.add_argument("-d", "--data_type", type=str, default="in_vitro")
    parser.add_argument("-src", "--source", type=str, default="pc9")
    parser.add_argument("-sub", "--subsample_length", type=int, default=10)
    parser.add_argument("-w", "--walkers", type=int, default=100)
    parser.add_argument("-i", "--iterations", type=int, default=10000)
    args = parser.parse_args()

    # Define functions and variables
    data_path = get_data_path(f"{args.data_type}/{args.source}", "processed")
    fp = FokkerPlanck().get_fokker_planck(args.transform)
    spsb = SpatialSubsample().get_spatial_subsample(args.transform)
    fit_params = [1, 1, 1, 1, 1, 1]
    n = 100
    mu = 0.01
    c = 1
    subsample_length = args.subsample_length
    walkers = args.walkers
    iterations = args.iterations

    # Get game data
    config = pd.read_csv(f"{data_path}/labels.csv")
    config = config.set_index(["source", "sample"])

    # Iterate over samples and calculate fit quality
    data = []
    for file_name in os.listdir(data_path):
        if file_name == "labels.csv":
            continue
        df = pd.read_csv(f"{data_path}/{file_name}")
        source, sample = file_name[:-4].split(" ")
        s_coords = df.loc[df["type"] == "sensitive"][["x", "y"]].values
        r_coords = df.loc[df["type"] == "resistant"][["x", "y"]].values
        awm, amw, sm = calculate_fp_params(
            config.loc[source, sample]["a"],
            config.loc[source, sample]["b"],
            config.loc[source, sample]["c"],
            config.loc[source, sample]["d"],
        )
        awm = round(awm, 3)
        amw = round(amw, 3)
        sm = round(sm, 3)
        params = [n, mu, awm, amw, sm, c]
        xdata, ydata = spsb(s_coords, r_coords, subsample_length)
        walker_ends = mcmc(fp, xdata, ydata, params, fit_params, walkers, iterations)
        d = evaluate_performance(fp, xdata, ydata, walker_ends, n, mu, awm, amw, sm, c)
        d["source"] = source
        d["sample"] = sample
        data.append(d)

    # Save results
    save_loc = get_data_path(f"{args.data_type}/{args.source}", "images")
    df = pd.DataFrame(data)

    with open(f"{save_loc}/mean_param_diff_ci.txt", "w") as f:
        for param in param_names:
            param_name = f"Mean {param} Difference"
            f.write(get_confidence_interval_str(df, param_name))
        f.write(get_confidence_interval_str(df, "Correct Game Classifications"))
        f.write(get_confidence_interval_str(df, "Game Quadrant Distance"))


if __name__ == "__main__":
    main()
