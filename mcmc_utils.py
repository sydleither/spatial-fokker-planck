"""
Functions for running MCMC
Based on https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
"""

import emcee
import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import calculate_fp_params, classify_game, game_colors
from fokker_planck import param_names


def plot_paramsweep(save_loc, df, metric):
    """
    Plot a metric across awm, amw, and sm.
    """
    sms = df["sm"].unique()
    fig, ax = plt.subplots(1, len(sms), figsize=(5 * len(sms), 5), constrained_layout=True)
    min_distance = df[metric].min()
    max_distance = df[metric].max()
    if min_distance < 0:
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=min_distance, vmax=max_distance)
        cmap = plt.get_cmap("PuOr")
    else:
        norm = mcolors.Normalize(vmin=min_distance, vmax=max_distance)
        cmap = plt.get_cmap("Purples")
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)

    awms = df["awm"].unique()
    amws = df["amw"].unique()
    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        df_sm = df_sm.pivot(index="amw", columns="awm", values=metric)
        ax[i].imshow(df_sm, cmap=cmap, norm=norm)
        ax[i].set_xticks(range(0, len(amws), 5), labels=amws[0::5])
        ax[i].set_yticks(range(0, len(awms), 5), labels=awms[0::5])
        ax[i].set_xlabel("amw")
        ax[i].set_ylabel("awm")
        ax[i].set_title(f"sm={sm}")
    cbar = fig.colorbar(scalarmap, drawedges=False, ax=ax[-1])
    cbar.set_label(metric)
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{metric}.png", bbox_inches="tight")
    plt.close()


def plot_paramsweep_game(save_loc, df):
    """
    Plot game quadrant across awm, amw, and sm.
    """
    sms = df["sm"].unique()
    fig, ax = plt.subplots(1, len(sms), figsize=(5 * len(sms), 5), constrained_layout=True)
    for i, sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        ax[i].scatter(df_sm["amw"], df_sm["awm"], c=df_sm["game"], s=200)
        ax[i].set_xlabel("amw")
        ax[i].set_ylabel("awm")
        ax[i].set_title(f"sm={sm}")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/game_quadrant.png", bbox_inches="tight")
    plt.close()


def gamespace_plot(ax, df, x, y):
    """
    Make a gamespace plot at the given axis.
    """
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="game",
        ax=ax,
        legend=False,
        hue_order=game_colors.keys(),
        palette=game_colors.values(),
    )
    sns.kdeplot(data=df, x=x, y=y, color="gray", alpha=0.3, ax=ax)
    ax.set_xlabel(r"$s_m + \alpha_{mw}$")
    ax.set_ylabel(r"$\alpha_{wm} - s_m$")


def plot_walker_gamespace(save_loc, walker_ends, true_params):
    """
    Plots of the final walker params on the Fokker-Planck transformed and normal game spaces.
    """
    if len(true_params) == 3:
        _, a, b, c, d = classify_game(*true_params, return_params=True)
    elif len(true_params) == 4:
        a, b, c, d = true_params
    else:
        return

    df = pd.DataFrame(walker_ends, columns=["N", "mu", "awm", "amw", "sm", "c"])
    df[["game", "a", "b", "c", "d"]] = df.apply(
        lambda x: classify_game(x["awm"], x["amw"], x["sm"], True), axis=1, result_type="expand"
    )
    df["c-a"] = df["c"] - df["a"]
    df["b-d"] = df["b"] - df["d"]

    fig, ax = plt.subplots(figsize=(4, 4))
    gamespace_plot(ax, df, "c-a", "b-d")
    ax.scatter([c - a], [b - d], marker="*", color="black")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_gamespace.png", bbox_inches="tight")
    plt.close()


def plot_walker_pairplot(save_loc, walker_ends):
    """
    Pairplot of the parameters the walkers ended on.
    """
    color1 = "xkcd:pink"
    color2 = "xkcd:rose"
    df = pd.DataFrame(walker_ends, columns=["N", "mu", "awm", "amw", "sm", "c"])
    g = sns.pairplot(df, diag_kind="kde", plot_kws={"color": color1}, diag_kws={"color": color2})
    g.map_lower(sns.kdeplot, levels=4, color=color2)
    plt.savefig(f"{save_loc}/mcmc_pairplot.png", transparent=True)
    plt.close()


def plot_walker_curves(save_loc, func, walker_ends, xdata, true_ydata, logspace=True):
    """
    Visualize curves resulting from walker end parameters.
    """
    fig, ax = plt.subplots()
    for params in walker_ends:
        ydata = func(xdata, *params)
        if logspace:
            ydata = np.exp(-ydata + params[-1])
        game = classify_game(*params[2:5])
        ax.plot(xdata, ydata, alpha=0.5, color=game_colors[game])
    if logspace:
        true_ydata = np.exp(-true_ydata)
    ax.plot(xdata, true_ydata, color="black", ls="--")
    ax.set(xlabel="Fraction Mutant", ylabel="Probability Density")
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_curves_{logspace}.png", bbox_inches="tight")
    plt.close()


def plot_walker_curve_mse(save_loc, func, walker_ends, xdata, true_ydata, logspace=True):
    """
    Histogram of the MSEs between each walker curve and the true curve
    """
    if logspace:
        true_ydata = np.exp(-true_ydata)
    len_data = len(true_ydata)
    mse = []
    for params in walker_ends:
        ydata = func(xdata, *params)
        if logspace:
            ydata = np.exp(-ydata + params[-1])
        mse.append(np.sum((ydata - true_ydata) ** 2) / len_data)

    fig, ax = plt.subplots()
    ax.hist(mse, bins=10, density=False, color="gray")
    ax.set(xlabel="MSE", ylabel="Count", ylim=(0, len(walker_ends)))
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_curves_mse.png", bbox_inches="tight")
    plt.close()


def plot_walker_gameparams(save_loc, walker_ends, true_game_params):
    """
    Histograms of the game parameters the walkers ended on.
    """
    df = pd.DataFrame(walker_ends, columns=param_names)
    fig, ax = plt.subplots(
        1, len(param_names), figsize=(4 * len(param_names), 4), layout="constrained"
    )
    for i, payoff_param in enumerate(param_names):
        ax[i].hist(df[payoff_param], bins=10, color="gray")
        ax[i].axvline(true_game_params[i], color="black", linestyle="dashed")
        ax[i].set(title=payoff_param, ylim=(0, len(walker_ends)))
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_gameparams.png")
    plt.close()


def lnprob(params, func, x, y):
    """
    Return chi-squared log likelihood, if the priors are satisfied.
    """
    # N
    if params[0] < 1 or params[0] > 1000:
        return -np.inf
    # mu
    if params[1] < 0 or params[1] > 0.1:
        return -np.inf
    # awm
    if params[2] < -1 or params[2] > 1:
        return -np.inf
    # amw
    if params[3] < -1 or params[3] > 1:
        return -np.inf
    # sm
    if params[4] < 0 or params[4] > 1:
        return -np.inf
    # c
    if params[5] < 0:
        return -np.inf
    return -0.5 * np.sum((y - func(x, *params)) ** 2)


def mcmc(func, xdata, ydata, nwalkers=100, niter=1000):
    """
    Run MCMC on true xdata, ydata and return walker end locations.
    """
    initial = (100, 0.05, 0, 0, 0.5, 1)
    ndim = len(initial)
    p0 = []
    for _ in range(nwalkers):
        walker = []
        for val in initial:
            sd = val / 10
            if sd == 0:
                sd = 0.1
            walker.append(np.random.normal(loc=val, scale=sd))
        p0.append(walker)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(func, xdata, ydata))
    sampler.run_mcmc(p0, niter)
    walker_ends = sampler.get_chain(discard=niter - 1)[0, :, :]

    # fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    # for i in range(ndim):
    #     axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.5)
    #     axes[i].set_ylabel(param_names[i])
    # axes[-1].set_xlabel("Step number")
    # plt.tight_layout()
    # fig.savefig("trace.png")
    # plt.close()

    return walker_ends
