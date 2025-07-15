"""
Run MCMC on spatial data
"""

import sys

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from common import calculate_fp_params, game_colors, get_data_path, theme_colors
from pdfs import FokkerPlanck, param_names, SpatialSubsample
from in_silico_fitting.abm_utils import read_sample
from individual_fitting_plots import plot_all
from mcmc import mcmc


def visualize_sample(save_loc, s_coords, r_coords, grid_size):
    colors = ListedColormap(
        ["#000000", game_colors["Sensitive Wins"], game_colors["Resistant Wins"]]
    )
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in s_coords:
        grid[y, x] = 1
    for x, y in r_coords:
        grid[y, x] = 2
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, cmap=colors)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/visual.png", bbox_inches="tight", dpi=200)
    plt.close()


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


def main(data_type, source, sample, subsample_length):
    """
    Given data type, source, sample, and subsample length
    Generate a spatial subsample distribution using the spatial data
    Fit Fokker-Planck to the distribution using MCMC
    """
    data_path = get_data_path(f"{data_type}/{source}", "raw")
    s_coords, r_coords, config = read_sample(data_path, sample)
    awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
    n = 100
    mu = 0
    c = 1
    params = [n, mu, awm, amw, sm, c]

    fp = FokkerPlanck().get_fokker_planck("norm")
    xdata, ydata = SpatialSubsample().get_spatial_subsample("none")(s_coords, r_coords, int(subsample_length))
    walker_ends = mcmc(fp, xdata, ydata, params, [1,1,1,1,1,1], 100, 10000)

    save_loc = get_data_path(f"{data_type}/{source}", f"images/{sample}/{subsample_length}")
    plot_all(save_loc, fp, walker_ends, xdata, ydata, params)
    title = " ".join([f"{param_names[i]}={round(params[i], 3)}" for i in range(len(params))])
    fp_vs_spsb(save_loc, xdata, ydata, fp(xdata, *params), title)
    visualize_sample(save_loc, s_coords, r_coords, config["x"])


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Please provide the data type, source, sample, and subsample length.")
    else:
        main(*sys.argv[1:])
