'''
Replicate and modify an approximate Figure 3 from MMMM
'''
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import classify_game, game_colors


def main(params):
    '''
    Visualize the game space with a given N, mu, and sm
    '''
    n, mu, sm = params
    n = int(n)
    mu = float(mu)
    sm = float(sm)

    awms = []
    amws = []
    games = []
    for awm in np.linspace(-0.2, 0.2, 100):
        for amw in np.linspace(-0.2, 0.2, 100):
            awms.append(awm)
            amws.append(amw)
            games.append(game_colors[classify_game(awm, amw, sm)])
    
    fig, ax = plt.subplots()
    ax.scatter(amws, awms, c=games)
    ax.set(title=f"N={n}, mu={mu}, sm={sm}", xlabel="amw", ylabel="awm")
    ax.axhline(0, c="black")
    ax.axvline(0, c="black")
    ax.axhline(-sm, c="black", ls="--")
    ax.axvline(sm, c="black", ls="--")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"gamespace_{n}_{mu}_{sm}.png")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide N, mu, and sm parameters as arguments.")
    else:
        main(sys.argv[1:])
