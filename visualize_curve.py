"""
Visualize the curve created by the input arguments
Order of arguments: N, mu, awm, amw, sm
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from common import classify_game, game_colors, get_data_path
from fokker_planck import FokkerPlanck, param_names


def visualize_curve(save_loc, params, x, y, logspace=True, density=True):
    """
    Visualize the FP solution
    """
    title = [f"{param_names[i]}={params[i]}" for i in range(len(params))]
    fig, ax = plt.subplots(figsize=(5, 5))
    classified_game = classify_game(params[2], params[3], params[4])
    if logspace:
        y = np.exp(-y)
    if density:
        y = y / (np.sum(y) * x[0])
    ax.plot(x, y, color=game_colors[classified_game], linewidth=3)
    ax.set(title=" ".join(title))
    fig.supxlabel("Fraction Mutant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/curve.png", bbox_inches="tight")


def main(params):
    """
    Visualize FP based on input parameters.
    """
    params = [float(x) for x in params]
    n = int(params[0])
    mu = params[1]

    fp = FokkerPlanck(n, mu).fokker_planck_log
    x = np.linspace(0.01, 0.99, n)
    y = fp(x, *params[2:])

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    visualize_curve(save_loc, params, x, y, True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
