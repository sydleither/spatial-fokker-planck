"""
Use MCMC to find fokker-planck curves that fit multiple parameter sets
"""

import sys

import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import pandas as pd

from fokker_planck import FokkerPlanck
from mcmc_utils import mcmc


def plot_variance(save_loc, file_name, data):
    """
    Plot the variance of mcmc walker endpoints across awm, amw, sm.
    """
    df = pd.DataFrame(data)
    sms = df["sm"].unique()
    min_var = df["var"].min()
    max_var = df["var"].max()
    fig, ax = plt.subplots(1, len(sms), figsize=(5 * len(sms), 4))
    cmap = plt.get_cmap("Greens")
    norm = BoundaryNorm(np.linspace(min_var, max_var, 10), cmap.N)
    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        ax[i].scatter(df_sm["amw"], df_sm["awm"], c=df_sm["var"], cmap=cmap, norm=norm)
        ax[i].set(title=f"sm={sm}")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax.ravel().tolist(), shrink=0.95)
    fig.supxlabel("amw")
    fig.supylabel("awm")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/mcmc_var_{file_name}.png", bbox_inches="tight")


def main(params):
    """
    Given N and mu, iterate over amw, amw, and sm
    To see if any resulting curves have multimodal param sets
    """
    n = int(params[0])
    mu = float(params[1])

    save_loc = "."
    fp = FokkerPlanck(n, mu).fokker_planck
    xdata = np.linspace(0.01, 0.99, n)

    a_vals = np.arange(-0.1, 0.11, 0.01)
    sm_vals = np.arange(-0.05, 0.051, 0.05)
    data = []
    for awm in a_vals:
        awm = round(awm, 2)
        for amw in a_vals:
            amw = round(amw, 2)
            for sm in sm_vals:
                sm = round(sm, 2)
                ydata = fp(xdata, awm, amw, sm)
                walker_ends = mcmc(fp, xdata, ydata)
                var = np.mean(np.var(np.array(walker_ends), axis=0))
                data.append({"awm": awm, "amw": amw, "sm": sm, "var": var})

    plot_variance(save_loc, f"N={n}_mu={mu}", data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
