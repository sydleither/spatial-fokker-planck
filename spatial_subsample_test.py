"""
Permutation test for resemblance of spatial subsampling PDFs to Fokker-Planck PDFs.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance

from common import calculate_fp_params, classify_game, game_colors, get_data_path, spatial_subsample
from fokker_planck import FokkerPlanck
from in_silico_fitting.abm_utils import read_sample


def confusion_matrix(save_loc, df):
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
    ax.set_title("Earth Mover's Distance")
    ax.set(xlabel="Spatial Subsample Game", ylabel="Fokker-Planck Game")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/spatial_subsample.png", bbox_inches="tight", transparent=True)
    plt.close()


def main(data_type, source, subsample_length):
    """
    Iterate over ABM spatial subsample distributions and conduct a permutation test with Fokker-Planck PDFs.
    """
    n = 100
    mu = 0.001
    c = 1
    fp = FokkerPlanck().fokker_planck_log

    # Calculate PDFs
    subsample_length = int(subsample_length)
    data_path = get_data_path(f"{data_type}/{source}", "raw")
    spsb_pdfs = []
    fp_pdfs = []
    games = []
    for sample in os.listdir(data_path):
        if os.path.isfile(f"{data_path}/{sample}"):
            continue
        s_coords, r_coords, config = read_sample(data_path, sample)
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        xdata, spsb_ydata = spatial_subsample(s_coords, r_coords, subsample_length, 1000)
        fp_ydata = fp(xdata, n, mu, awm, amw, sm, c)
        spsb_pdfs.append(spsb_ydata)
        fp_pdfs.append(fp_ydata)
        games.append(classify_game(awm, amw, sm))

    # Calculate the EMD from each SpSb curve to each FP curve
    emd_data = []
    for i in range(len(spsb_pdfs)):
        for j in range(len(fp_pdfs)):
            if i == j: #Distributions generated with the same parameters
                continue
            emd = wasserstein_distance(spsb_pdfs[i], fp_pdfs[j])
            emd_data.append({"EMD":emd, "SpSb Game": games[i], "FP Game": games[j]})

    # Confusion matrix of differences
    game_order = [x for x in game_colors.keys() if x != "Unknown"]
    df = pd.DataFrame(emd_data)
    df_mat = df.groupby(["FP Game", "SpSb Game"]).mean().reset_index()
    df_mat = df_mat.pivot(index="FP Game", columns="SpSb Game", values="EMD")
    df_mat = df_mat.reindex(game_order, axis=0)
    df_mat = df_mat.reindex(game_order, axis=1)
    save_loc = get_data_path(f"{data_type}/{source}", "images")
    confusion_matrix(save_loc, df_mat)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide the data type, source, and subsample length.")
    else:
        main(*sys.argv[1:])
