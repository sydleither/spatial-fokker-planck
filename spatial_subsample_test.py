"""
Permutation test for resemblance of spatial subsampling PDFs to Fokker-Planck PDFs.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from common import calculate_fp_params, classify_game, get_data_path, spatial_subsample
from fitting_utils import game_parameter_sweep
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
    ax.set(title="B")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/spatial_subsample.png", bbox_inches="tight", transparent=True)
    plt.close()


def distance_between_distributions(row):
    return wasserstein_distance(row["FP PDF"], row["SpSb PDF"])


def main(data_type, source, subsample_length):
    """
    Iterate over ABM spatial subsample distributions and conduct a permutation test with Fokker-Planck PDFs.
    """
    # Calculate Fokker-Planck PDFs
    n = 100
    mu = 0.001
    c = 1
    fp = FokkerPlanck().fokker_planck_log
    xdata = np.linspace(0.01, 0.99, 100)
    game_params = game_parameter_sweep()
    fp_data = []
    for awm, amw, sm in game_params:
        fp_data.append({
            "FP PDF": fp(xdata, n, mu, awm, amw, sm, c),
            "FP Game": classify_game(awm, amw, sm)
        })

    # Calculate Spatial Subsample PDFs
    subsample_length = int(subsample_length)
    data_path = get_data_path(f"{data_type}/{source}", "raw")
    spsb_data = []
    for sample in os.listdir(data_path):
        if os.path.isfile(f"{data_path}/{sample}"):
            continue
        s_coords, r_coords, config = read_sample(data_path, sample)
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        xdata, ydata = spatial_subsample(s_coords, r_coords, subsample_length, 500)
        spsb_data.append({
            "SpSb PDF": ydata,
            "SpSb Game": classify_game(awm, amw, sm)
        })

    # Make a dataframe with a row for each SpSb - FP sample combination
    df_fp = pd.DataFrame(fp_data)
    df_sbsp = pd.DataFrame(spsb_data)
    df = df_fp.merge(df_sbsp, how="cross")

    # Confusion matrix of differences
    df["EMD"] = df.apply(distance_between_distributions, axis=1)
    df = df.drop(["FP PDF", "SpSb PDF"], axis=1)
    df_mat = df.groupby(["FP Game", "SpSb Game"]).mean().reset_index()
    df_mat = df_mat.pivot(index="FP Game", columns="SpSb Game", values="EMD")
    save_loc = get_data_path(f"{data_type}/{source}", "images")
    confusion_matrix(save_loc, df_mat)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide the data type, source, and subsample length.")
    else:
        main(*sys.argv[1:])
