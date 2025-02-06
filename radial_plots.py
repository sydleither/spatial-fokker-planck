'''
Visualize the shapes of distributions changing as we move along the radial lines
'''
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import classify_game, fokker_planck, game_colors


def main(params):
    '''
    Visualize parameters on the radial.
    '''
    n, mu, sm = params
    n = int(n)
    mu = float(mu)
    sm = float(sm)
    x = np.linspace(0.01, 0.99, n)
    inc = 0.02

    fp_data = [[], [], [], []]
    a_data = [[], [], [], []]
    colors = [[], [], [], []]
    i = 0
    for awm_mod in [inc, -inc]:
        for amw_mod in [inc, -inc]:
            awm = sm
            amw = -sm
            for _ in range(8):
                awm += awm_mod
                amw += amw_mod
                y = fokker_planck(x, n, mu, awm, amw, sm)
                fp_data[i].append(y/max(y))
                a_data[i].append([awm, amw])
                colors[i].append(game_colors[classify_game(awm, amw, sm)])
            i += 1
    
    num_cols = len(fp_data[0])
    fig, ax = plt.subplots(4, num_cols, figsize=(num_cols*3, 12))
    for i in range(len(fp_data)):
        for j in range(num_cols):
            ax[i][j].plot(x, fp_data[i][j], c=colors[i][j], linewidth=3)
            ax[i][j].set(title=f"awm={a_data[i][j][0]:4.2f}, amw={a_data[i][j][1]:4.2f}")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"radial_{n}_{mu}_{sm}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide N, mu, and sm as arguments.")
    else:
        main(sys.argv[1:])
