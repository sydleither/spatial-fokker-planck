"""
Run MCMC on spatial data
"""

import json
import sys

import pandas as pd

from common import get_data_path, spatial_subsample
from fokker_planck import FokkerPlanck
from individual_fitting_plots import (
    plot_walker_curves,
    plot_walker_curve_mse,
    plot_walker_gamespace,
    plot_walker_pairplot,
    plot_walker_gameparams,
)
from mcmc import mcmc


def main(data_type, source, sample):
    """
    Given data type, source, and sample
    Generate a spatial subsample distribution using the spatial data
    Fit Fokker-Planck to the distribution using MCMC
    """
    data_path = get_data_path(data_type, "raw")
    df = pd.read_csv(f"{data_path}/{source}/{sample}/{sample}/2Dcoords.csv")
    df = df[df["time"] == df["time"].max()]
    s_coords = df.loc[df["type"] == 0][["x", "y"]].values
    r_coords = df.loc[df["type"] == 1][["x", "y"]].values
    config = json.loads(open(f"{data_path}/{source}/{sample}/{sample}.json").read())
    true_game_params = [config["awm"], config["amw"], config["sm"]]

    fp = FokkerPlanck().fokker_planck_log
    xdata, ydata = spatial_subsample(s_coords, r_coords, 10)
    walker_ends = mcmc(fp, xdata, ydata, 100, 1000)

    save_loc = get_data_path(data_type, f"images/{source} {sample}")
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata, True)
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata, False)
    plot_walker_curve_mse(save_loc, fp, walker_ends, xdata, ydata)
    plot_walker_gamespace(save_loc, walker_ends, true_game_params)
    plot_walker_pairplot(save_loc, walker_ends)
    plot_walker_gameparams(save_loc, walker_ends, true_game_params)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide the data type, source, and sample.")
    else:
        main(*sys.argv[1:])
