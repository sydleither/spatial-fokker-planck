"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""

import json
import os
import sys

import pandas as pd

from aggregate_fitting_plots import plot_all
from common import calculate_fp_params, get_data_path, spatial_subsample
from fitting_utils import evaluate_performance
from pdfs import FokkerPlanck
from mcmc import mcmc


def main(data_type, source):
    """
    Iterate over ABM spatial subsample distributions. Calculate quality of MCMC fit.
    """
    n = 100
    mu = 0
    c = 1
    fp = FokkerPlanck().fokker_planck_log
    subsample_length = 5

    data_path = get_data_path(f"{data_type}/{source}", "raw")
    data = []
    for sample in data_path:
        if not os.path.isdir({data_path}/{sample}):
            continue
        df = pd.read_csv(f"{data_path}/{sample}/{sample}/2Dcoords.csv")
        df = df[df["time"] == df["time"].max()]
        s_coords = df.loc[df["type"] == 0][["x", "y"]].values
        r_coords = df.loc[df["type"] == 1][["x", "y"]].values
        config = json.loads(open(f"{data_path}/{source}/{sample}/{sample}.json").read())
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        xdata, ydata = spatial_subsample(s_coords, r_coords, subsample_length)
        walker_ends = mcmc(fp, xdata, ydata, 100, 1000)
        d = evaluate_performance(fp, xdata, ydata, walker_ends, n, mu, awm, amw, sm, c)
        data.append(d)

    save_loc = get_data_path(f"{data_type}/{source}", "images")
    df = pd.DataFrame(data)
    plot_all(save_loc, df)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide the data type and source.")
    else:
        main(*sys.argv[1:])
