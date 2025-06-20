import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import classify_game, game_colors, theme_colors
from fokker_planck import param_names


def gamespace_plot(ax, df, x, y):
    """
    Make a gamespace plot at the given axis.
    """
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="Game",
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
    df[["Game", "a", "b", "c", "d"]] = df.apply(
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
    color1 = theme_colors[0]
    color2 = theme_colors[0]
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
    ax.hist(mse, bins=10, density=False, color=theme_colors[0])
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
        ax[i].hist(df[payoff_param], bins=10, color=theme_colors[0])
        ax[i].axvline(true_game_params[i], color="black", linestyle="dashed")
        ax[i].set(title=payoff_param, ylim=(0, len(walker_ends)))
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_gameparams.png", bbox_inches="tight")
    plt.close()


# fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
# for i in range(ndim):
#     axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.5)
#     axes[i].set_ylabel(param_names[i])
# axes[-1].set_xlabel("Step number")
# plt.tight_layout()
# fig.savefig("trace.png")
# plt.close()
