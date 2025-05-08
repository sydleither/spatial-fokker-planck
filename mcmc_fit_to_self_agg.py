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


def plot_metric(save_loc, df, metric):
    """
    Plot the aggregation of mcmc walker endpoints across awm, amw, sm.
    """
    sms = df["sm"].unique()
    fig, ax = plt.subplots(1, len(sms), figsize=(5 * len(sms), 5), constrained_layout=True)
    cmap = sns.color_palette("flare", as_cmap=True)
    norm = plt.Normalize(df[metric].min(), df[metric].max())
    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        ax[i].scatter(df_sm["amw"], df_sm["awm"], c=df_sm[metric], s=80, cmap=cmap, norm=norm)
        ax[i].set(title=f"sm={sm}")
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax[-1]
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
    """
    n = int(params[0])
    mu = float(params[1])

    fp = FokkerPlanck(n, mu).fokker_planck_log
    xdata = np.linspace(0.01, 0.99, n)
    len_data = len(xdata)

    a_vals = np.round(np.arange(-0.5, 0.51, 0.05), 2)
    sm_vals = [0, 0.05, 0.1]
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                ydata = fp(xdata, awm, amw, sm)
                walker_ends = np.array(mcmc(fp, xdata, ydata))
                distances = np.linalg.norm(walker_ends - np.array([awm, amw, sm]), axis=1)
                mse = []
                for walker_params in walker_ends:
                    mse.append(np.sum((ydata - fp(xdata, *walker_params)) ** 2) / len_data)
                data.append(
                    {
                        "awm": awm,
                        "amw": amw,
                        "sm": sm,
                        "mean_param_distance": np.mean(distances),
                        "var_param_distance": np.var(distances),
                        "min_param_distance": np.min(distances),
                        "mean_curve_mse": np.mean(mse),
                        "var_curve_mse": np.var(mse),
                        "min_curve_mse": np.min(mse),
                    }
                )

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    metrics = [x for x in data[0].keys() if x not in ["awm", "amw", "sm"]]
    df = pd.DataFrame(data)
    for metric in metrics:
        plot_metric(save_loc, df, metric)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
