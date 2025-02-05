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
    inc = 0.01

    fp_data = [[], []]
    a_data = [[], []]
    colors = [[], []]
    for awm_mod in [inc, -inc]:
        for amw_mod in [inc, -inc]:
            awm = sm
            amw = -sm
            i = 0 if awm_mod == amw_mod else 1
            for _ in range(10):
                awm += awm_mod
                amw += amw_mod
                y = fokker_planck(x, n, mu, awm, amw, sm)
                fp_data[i].append(y/max(y))
                a_data[i].append([awm, amw])
                colors[i].append(game_colors[classify_game(awm, amw, sm)])
    
    num_cols = len(fp_data[0])
    fig, ax = plt.subplots(2, num_cols, figsize=(num_cols*4, 8))
    for i in range(len(fp_data)):
        for j in range(num_cols):
            ax[i][j].plot(x, fp_data[i][j], c=colors[i][j])
            ax[i][j].set(title=f"awm={a_data[i][j][0]:4.2f}, amw={a_data[i][j][1]:4.2f}")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"radial_{n}_{mu}_{sm}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide N, mu, and sm as arguments.")
    else:
        main(sys.argv[1:])
