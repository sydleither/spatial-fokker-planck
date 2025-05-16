"""
Run MCMC on spatial data
"""

import json
from random import choices
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import calculate_fp_params, get_data_path
from fokker_planck import FokkerPlanck
from mcmc_utils import mcmc, plot_walker_curves, plot_walker_curve_mse, plot_walker_gamespace, plot_walker_pairplot, plot_walker_gameparams


def get_data_pdf_and_support(df, sample_length, num_samples=5000):
    """
    Create spatial fokker-planck distribution.
    """
    s_coords = df.loc[df["type"] == 0][["x", "y"]].values
    r_coords = df.loc[df["type"] == 1][["x", "y"]].values

    dims = range(len(s_coords[0]))
    max_dims = [max(np.max(s_coords[:, i]), np.max(r_coords[:, i])) for i in dims]
    dim_vals = [choices(range(0, max_dims[i] - sample_length), k=num_samples) for i in dims]
    fr_counts = []
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
        fr_counts.append(subset_r / subset_total)

    xdata = np.linspace(min(fr_counts), max(fr_counts), 100)
    kde = stats.gaussian_kde(fr_counts)
    pdf = kde(xdata)
    neg_lnpdf = -np.log(pdf)
    return xdata, neg_lnpdf


def main(data_type, source, sample):
    """
    Given data type, source, and sample 
    Generate a spatial subsample distribution using the spatial data
    Fit Fokker-Planck to the distribution using MCMC
    """
    data_path = get_data_path(data_type, "raw")
    df = pd.read_csv(f"{data_path}/{source}/{sample}/{sample}/2Dcoords.csv")
    df = df[df["time"] == 100]
    config = json.loads(open(f"{data_path}/{source}/{sample}/{sample}.json").read())
    true_game_params = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])

    fp = FokkerPlanck().fokker_planck_log
    xdata, ydata = get_data_pdf_and_support(df, 10)
    walker_ends = mcmc(fp, xdata, ydata, 100, 10000)

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
