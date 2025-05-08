"""
Functions for running MCMC
Based on https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
"""

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import calculate_fp_params, classify_game, game_colors


def gamespace_plot(ax, df, x, y, truex, truey, xline=0, yline=0):
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
    ax.axvline(xline, color="black", lw=0.5)
    ax.axhline(yline, color="black", lw=0.5)
    ax.scatter([truex], [truey], marker="*", color="black")


def plot_walker_gamespace(save_loc, walker_ends, true_params):
    """
    Plots of the final walker params on the Fokker-Planck transformed and normal game spaces.
    """
    if len(true_params) == 3:
        awm, amw, sm = true_params
        _, a, b, c, d = classify_game(*true_params, return_params=True)
    elif len(true_params) == 4:
        a, b, c, d = true_params
        awm, amw, sm = calculate_fp_params(a, b, c, d)
    else:
        return

    df = pd.DataFrame(walker_ends, columns=["awm", "amw", "sm"])
    df["awm/sm"] = df["awm"] / df["sm"]
    df["amw/sm"] = df["amw"] / df["sm"]
    df[["game", "a", "b", "c", "d"]] = df.apply(
        lambda x: classify_game(x["awm"], x["amw"], x["sm"], True), axis=1, result_type="expand"
    )
    df["c-a"] = df["c"] - df["a"]
    df["b-d"] = df["b"] - df["d"]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    gamespace_plot(ax[0], df, "amw/sm", "awm/sm", amw / sm, awm / sm, -1, 1)
    gamespace_plot(ax[1], df, "c-a", "b-d", c - a, b - d, 0, 0)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_gamespace.png", bbox_inches="tight")
    plt.close()


def plot_walker_params(save_loc, walker_ends):
    """
    Pairplot of the parameters the walkers ended on.
    """
    color1 = "xkcd:pink"
    color2 = "xkcd:rose"
    df = pd.DataFrame(walker_ends, columns=["awm", "amw", "sm"])
    g = sns.pairplot(df, diag_kind="kde", plot_kws={"color": color1}, diag_kws={"color": color2})
    g.map_lower(sns.kdeplot, levels=4, color=color2)
    plt.savefig(f"{save_loc}/mcmc_params.png", transparent=True)
    plt.close()


def plot_walker_curves(save_loc, func, walker_ends, xdata, true_ydata):
    """
    Visualize curves resulting from walker end parameters.
    """
    fig, ax = plt.subplots()
    for params in walker_ends:
        ydata = func(xdata, *params)
        game = classify_game(*params)
        ax.plot(xdata, ydata, alpha=0.5, color=game_colors[game])
    ax.plot(xdata, true_ydata, color="black", ls="--")
    ax.set(xlabel="Fraction Mutant", ylabel="Probability Density")
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_curves.png", bbox_inches="tight")
    plt.close()


def lnprob(params, func, x, y, yerr):
    """
    Return chi-squared log likelihood, if the priors are satisfied.
    """
    params = np.array(params)
    if np.any(params > 1) or np.any(params < -1):
        return -np.inf
    if params[2] < 0:
        return -np.inf
    return -0.5 * np.sum(((y - func(x, *params)) / yerr)**2)


def mcmc(func, xdata, ydata, nwalkers=50, niter=500):
    """
    Run MCMC on true xdata, ydata and return walker end locations.
    """
    yerr = 0.05 * ydata
    initial = (0, 0, 0.5)
    ndim = 3
    p0 = [np.array(initial) + 0.1 * np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(func, xdata, ydata, yerr))
    sampler.run_mcmc(p0, niter)
    walker_ends = sampler.get_chain(discard=niter - 1)[0, :, :]

    return walker_ends
