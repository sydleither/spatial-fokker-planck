'''
Visualize the curve created by the input arguments
Order of arguments: N, mu, awm, amw, sm
'''
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import classify_game, fokker_planck, game_colors


def visualize_curve(params, x, y):
    '''
    Visualize the FP solution
    '''
    param_labels = ["N", "mu", "awm", "amw", "sm"]
    title = [f"{param_labels[i]}={params[i]}" for i in range(len(params))]
    fig, ax = plt.subplots(figsize=(5, 5))
    classified_game = classify_game(params[2], params[3], params[4])
    ax.plot(x, y/max(y), color=game_colors[classified_game], linewidth=3)
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.set(title=" ".join(title))
    fig.supxlabel("Fraction Mutant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"fp_{'_'.join(title)}.png", bbox_inches="tight")


def main(params):
    '''
    Visualize FP based on input parameters.
    '''
    params = [float(x) for x in params]
    params[0] = int(params[0])
    x = np.linspace(0.01, 0.99, params[0])
    y = fokker_planck(x, *params)
    visualize_curve(params, x, y)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide FP parameters as arguments.")
    else:
        main(sys.argv[1:])
