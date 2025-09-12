"""
Permutation test for resemblance of spatial subsampling PDFs to Fokker-Planck PDFs.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from common import calculate_fp_params, classify_game, game_colors, get_data_path
from mcmc import mcmc
from pdfs import FokkerPlanck, SpatialSubsample
from in_silico_fitting.abm_utils import read_sample


def confusion_matrix(save_loc, save_info, df):
    textcolors = ["black", "white"]
    fig, ax = plt.subplots(figsize=(4, 4))
    rows = df.index.tolist()
    cols = df.columns.tolist()
    im = ax.imshow(df, cmap="Purples")
    ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), labels=rows)
    matrix = df.to_numpy()
    threshold = im.norm(matrix.max()) / 2
    for i in range(len(rows)):
        for j in range(len(cols)):
            color = textcolors[int(im.norm(matrix[i, j]) > threshold)]
            ax.text(j, i, f"{matrix[i, j]:4.2f}", ha="center", va="center", color=color)
    ax.set(xlabel="Spatial Subsample Game", ylabel="Fokker-Planck Game")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/spsb_{save_info}.png", bbox_inches="tight", transparent=True)
    plt.close()


def plot_curves(save_loc, save_info, game_order, supports, spsb_pdfs, fp_pdfs, games):
    game_ax = {game: i for i, game in enumerate(game_order)}
    fig, ax = plt.subplots(4, 2, figsize=(4, 8))
    for i in range(len(games)):
        game = games[i]
        ax[game_ax[game]][0].plot(supports[i], spsb_pdfs[i], c=game_colors[game], alpha=0.2)
        ax[game_ax[game]][1].plot(supports[i], fp_pdfs[i], c=game_colors[game], alpha=0.2)
    ax[0][0].set_title("Spatial Subsample")
    ax[0][1].set_title("Fokker Planck")
    fig.supxlabel("Fraction Mutant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/spsb_v_fp_{save_info}.png", bbox_inches="tight", dpi=200)
    plt.close()


def main():
    """
    Iterate over ABM spatial subsample distributions and conduct a permutation test with Fokker-Planck PDFs.
    """
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str, default="none")
    parser.add_argument("-d", "--data_type", type=str, default="in_silico")
    parser.add_argument("-src", "--source", type=str, default="5_5")
    parser.add_argument("-sub", "--subsample_length", type=int, default=5)
    args = parser.parse_args()

    if args.transform == "norm":
        fit_c = 0
        c = 1
    else:
        fit_c = 1
        c = 0

    # Define functions and variables
    fp = FokkerPlanck().get_fokker_planck(args.transform)
    spsb = SpatialSubsample().get_spatial_subsample(args.transform)
    data_type = args.data_type
    source = args.source
    subsample_length = args.subsample_length
    data_path = get_data_path(f"{data_type}/{source}", "raw")

    # Calculate PDFs
    supports = []
    spsb_pdfs = []
    fp_pdfs = []
    games = []
    for sample in os.listdir(data_path):
        if os.path.isfile(f"{data_path}/{sample}"):
            continue
        s_coords, r_coords, config = read_sample(data_path, sample)
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        xdata, spsb_ydata = spsb(s_coords, r_coords, subsample_length, 500)
        walker_ends = mcmc(fp, xdata, spsb_ydata, [0, 0, awm, amw, sm, c], [1, 1, 0, 0, 0, fit_c], niter=10000)
        mean_walker = np.mean(np.array(walker_ends), axis=0)
        fp_ydata = fp(xdata, mean_walker[0], mean_walker[1], awm, amw, sm, mean_walker[-1])
        supports.append(xdata)
        spsb_pdfs.append(spsb_ydata)
        fp_pdfs.append(fp_ydata)
        games.append(classify_game(awm, amw, sm))

    # Calculate the distance between curves
    distance_data = []
    for i in range(len(spsb_pdfs)):
        for j in range(len(fp_pdfs)):
            if i == j:  # Distributions generated with the same parameters
                continue
            euclid = euclidean(spsb_pdfs[i], fp_pdfs[j])
            distance_data.append({"Distance": euclid, "SpSb Game": games[i], "FP Game": games[j]})
    df = pd.DataFrame(distance_data)
    save_loc = get_data_path(f"{data_type}/{source}", "images")
    game_order = [x for x in game_colors if x != "Unknown"]

    # Plot curves
    plot_curves(save_loc, args.transform, game_order, supports, spsb_pdfs, fp_pdfs, games)

    # Confusion matrix of differences
    df_mat = df.groupby(["FP Game", "SpSb Game"]).mean().reset_index()
    df_mat = df_mat.pivot(index="FP Game", columns="SpSb Game", values="Distance")
    df_mat = df_mat.reindex(game_order, axis=0)
    df_mat = df_mat.reindex(game_order, axis=1)
    confusion_matrix(save_loc, args.transform, df_mat)


if __name__ == "__main__":
    main()
