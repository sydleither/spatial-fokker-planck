"""
Plot the curves resulting from the awm, amw, sm parameter sweep
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from common import calculate_fp_params, classify_game, game_colors, get_data_path
from in_silico_fitting.abm_utils import read_sample
from pdfs import FokkerPlanck, SpatialSubsample


def fp_plot(transform, n, mu, c, sm):
    num_a = 10
    r = 1
    fp = FokkerPlanck().get_fokker_planck(transform)
    fig, ax = plt.subplots(num_a, num_a, figsize=(20, 20), constrained_layout=True)
    for i,awm in enumerate(np.round(np.linspace(-r+sm, r+sm, num_a), 3)):
        for j,amw in enumerate(np.round(np.linspace(-r-sm, r-sm, num_a), 3)):
            xdata = np.linspace(0.01, 0.99, 100)
            params = [n, mu, awm, amw, sm, c]
            ydata = fp(xdata, *params)
            classified_game = classify_game(awm, amw, sm)
            ax[num_a-1-i][j].plot(xdata, ydata, color=game_colors[classified_game], linewidth=5)
            ax[num_a-1-i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.patch.set_alpha(0)
    save_loc = get_data_path("self", "images")
    fig.savefig(f"{save_loc}/sm={sm}_N={n}_mu={mu}_c={c}.png", bbox_inches="tight")


def spsb_plot(transform, data_type, source, subsample_length, desired_sm):
    spsb = SpatialSubsample().get_spatial_subsample(transform)
    data_path = get_data_path(f"{data_type}/{source}", "raw")
    data = dict()
    awms = set()
    amws = set()
    for sample in os.listdir(data_path):
        if not os.path.isdir(f"{data_path}/{sample}"):
            continue
        s_coords, r_coords, config = read_sample(data_path, sample)
        awm, amw, sm = calculate_fp_params(config["A"], config["B"], config["C"], config["D"])
        if not np.isclose(sm, desired_sm):
            continue
        xdata, ydata = spsb(s_coords, r_coords, subsample_length, 1000)
        awm = round(awm, 3)
        amw = round(amw, 3)
        data[(awm, amw)] = (xdata, ydata)
        awms.add(awm)
        amws.add(amw)

    num_a = len(awms)
    fig, ax = plt.subplots(num_a, num_a, figsize=(20, 20), constrained_layout=True)
    for i,awm in enumerate(sorted(awms)):
        for j,amw in enumerate(sorted(amws)):
            xdata, ydata = data[(awm, amw)]
            classified_game = classify_game(awm, amw, desired_sm)
            ax[num_a-1-i][j].plot(xdata, ydata, color=game_colors[classified_game], linewidth=5)
            ax[num_a-1-i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.patch.set_alpha(0)
    save_loc = get_data_path(f"{data_type}/{source}", "images")
    fig.savefig(f"{save_loc}/sm={desired_sm}_sub={subsample_length}.png", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str, default="none")
    parser.add_argument("-d", "--data_type", type=str, default="self")
    parser.add_argument("-src", "--source", type=str, default=None)
    parser.add_argument("-sub", "--subsample_length", type=int, default=None)
    parser.add_argument("-sm", "--sm", type=float, default=0.05)
    parser.add_argument("-n", "--n", type=int, default=100)
    parser.add_argument("-mu", "--mu", type=float, default=0.05)
    parser.add_argument("-c", "--c", type=float, default=1)
    args = parser.parse_args()

    data_type = args.data_type
    if data_type == "self":
        fp_plot(args.transform, args.n, args.mu, args.c, args.sm)
    else:
        spsb_plot(args.transform, data_type, args.source, args.subsample_length, args.sm)


if __name__ == "__main__":
    main()
