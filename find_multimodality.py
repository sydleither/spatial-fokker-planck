"""
Use MCMC to find fokker-planck curves that fit multiple parameter sets
"""

import sys

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_data_path
from fokker_planck import FokkerPlanck, param_names
from mcmc_utils import mcmc


def plot_metric(save_loc, data, metric):
    """
    Plot the variance of mcmc walker endpoints across awm, amw, sm.
    """
    df = pd.DataFrame(data)
    sms = df["sm"].unique()
    fig, ax = plt.subplots(1, len(sms), figsize=(5 * len(sms), 6), constrained_layout=True)
    cmap = sns.color_palette("flare", as_cmap=True)
    norm = plt.Normalize(df[metric].min(), df[metric].max())
    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        ax[i].scatter(df_sm["amw"], df_sm["awm"], c=df_sm[metric], cmap=cmap, norm=norm)
        ax[i].set(title=f"sm={sm}")
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax.ravel().tolist(),
        orientation="horizontal",
        pad=0.1,
    )
    cbar.set_label(metric)
    fig.supxlabel("amw")
    fig.supylabel("awm")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/mcmc_gamespaces_{metric}.png", bbox_inches="tight")
    plt.close()


def main(params):
    """
    Given N and mu, iterate over amw, amw, and sm
    To see if any resulting curves have multimodal param sets
    """
    n = int(params[0])
    mu = float(params[1])

    fp = FokkerPlanck(n, mu).fokker_planck_log
    xdata = np.linspace(0.01, 0.99, n)
    len_data = len(xdata)

    a_vals = np.round(np.arange(-0.1, 0.11, 0.01), 2)
    sm_vals = [-0.075, -0.05, -0.025, 0.025, 0.05, 0.075]
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                ydata = fp(xdata, awm, amw, sm)
                walker_ends = np.array(mcmc(fp, xdata, ydata))
                var = np.var(walker_ends, axis=0)
                distances = np.linalg.norm(walker_ends - np.array([awm, amw, sm]), axis=1)
                mse = []
                for walker_params in walker_ends:
                    mse.append(np.sum((ydata - fp(xdata, *walker_params)) ** 2) / len_data)
                data.append(
                    {
                        "awm": awm,
                        "amw": amw,
                        "sm": sm,
                        "params_mean_distance": np.mean(distances),
                        "params_var_distance": np.var(distances),
                        "params_mean_var": np.mean(var),
                        "mean_mse": np.mean(mse),
                        "max_mse": np.max(mse),
                    }
                )

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    metrics = [x for x in data[0].keys() if x not in ["awm", "amw", "sm"]]
    for metric in metrics:
        plot_metric(save_loc, data, metric)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
