"""
Run MCMC on spatial data
"""

from random import choices
import sys

import numpy as np
import pandas as pd

from fokker_planck import FokkerPlanck
from mcmc_utils import mcmc, plot_walker_curves, plot_walker_gamespace, plot_walker_params
from common import get_data_path


def create_sfp_dist(df, bins, sample_length, num_samples=1000):
    """
    Create spatial fokker-planck distribution.
    """
    s_coords = df.loc[df["type"] == "sensitive"][["x", "y"]].values
    r_coords = df.loc[df["type"] == "resistant"][["x", "y"]].values

    dims = range(len(s_coords[0]))
    max_dims = [max(np.max(s_coords[:, i]), np.max(r_coords[:, i])) for i in dims]
    dim_vals = [choices(range(0, max_dims[i] - sample_length), k=num_samples) for i in dims]
    fs_counts = []
    for s in range(num_samples):
        ld = [dim_vals[i][s] for i in dims]
        ud = [ld[i] + sample_length for i in dims]
        subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
        subset_s = np.sum(np.all(subset_s, axis=0))
        subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
        subset_r = np.sum(np.all(subset_r, axis=0))
        subset_total = subset_s + subset_r
        if subset_total == 0:
            continue
        fs_counts.append(subset_r / subset_total)

    hist, _ = np.histogram(fs_counts, bins=bins)
    hist = hist / max(hist)
    return hist


def main(args):
    """
    Given N, mu, and spatial data
    Generate a spatial subsample distribution using the data
    Fit Fokker-Planck to the distribution using MCMC
    """
    n = int(args[0])
    mu = float(args[1])

    data_path = args[2]
    source, sample_id = data_path.split("/")[-1].split(" ")
    sample_id = sample_id[:-4]
    df = pd.read_csv(data_path)
    payoff_df = pd.read_csv("data/payoff.csv")
    payoff_df["sample"] = payoff_df["sample"].astype(str)
    payoff_df = payoff_df[(payoff_df["source"] == source) & (payoff_df["sample"] == sample_id)]
    true_game_params = payoff_df[["a", "b", "c", "d"]].values[0]

    save_loc = "."
    fp = FokkerPlanck(n, mu).fokker_planck_normalized
    bin_size = 10
    xdata = np.linspace(0.01, 0.99, bin_size)
    ydata = create_sfp_dist(df, np.linspace(0, 1, bin_size + 1), 3)
    walker_ends = mcmc(fp, xdata, ydata)

    file_name = f"{source} {sample_id}"
    save_loc = get_data_path("HAL", file_name)
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata)
    plot_walker_gamespace(save_loc, walker_ends, true_game_params)
    plot_walker_params(save_loc, walker_ends)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide N, mu, and the data path.")
    else:
        main(sys.argv[1:])
