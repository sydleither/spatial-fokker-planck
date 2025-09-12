import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import classify_game, game_colors, theme_colors
from pdfs import param_names


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
    Plots of the final walker params on the game space.
    """
    if len(true_params) == 3:
        _, a, b, c, d = classify_game(*true_params, return_params=True)
    elif len(true_params) == 4:
        a, b, c, d = true_params
    else:
        return

    df = pd.DataFrame(walker_ends, columns=param_names)
    df[["Game", "a", "b", "c", "d"]] = df.apply(
        lambda x: classify_game(x["awm"], x["amw"], x["sm"], return_params=True),
        axis=1,
        result_type="expand",
    )
    df["c-a"] = df["c"] - df["a"]
    df["b-d"] = df["b"] - df["d"]

    fig, ax = plt.subplots(figsize=(4, 4))
    gamespace_plot(ax, df, "c-a", "b-d")
    ax.scatter([c - a], [b - d], marker="*", color="black")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_gamespace.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_walker_pairplot(save_loc, walker_ends, fit_params):
    """
    Pairplot of the parameters the walkers ended on.
    """
    color1 = theme_colors[0]
    color2 = theme_colors[0]
    params_to_plot = [param_names[i] for i in range(len(fit_params)) if fit_params[i] == 1]
    df = pd.DataFrame(walker_ends, columns=param_names)
    df = df[params_to_plot]
    g = sns.pairplot(df, diag_kind="kde", plot_kws={"color": color1}, diag_kws={"color": color2})
    g.map_lower(sns.kdeplot, levels=4, color=color2)
    plt.savefig(f"{save_loc}/mcmc_pairplot.png", transparent=True)
    plt.close()


def plot_walker_curves(save_loc, func, walker_ends, xdata, true_ydata):
    """
    Visualize curves resulting from walker end parameters.
    """
    fig, ax = plt.subplots()
    for params in walker_ends:
        ydata = func(xdata, *params)
        game = classify_game(*params[2:5])
        ax.plot(xdata, ydata, alpha=0.5, color=game_colors[game])
    ax.plot(xdata, true_ydata, color="black", ls="--")
    ax.set(xlabel="Fraction Mutant", ylabel="Probability Density")
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_curves.png", bbox_inches="tight")
    plt.close()


def plot_walker_curve_mse(save_loc, func, walker_ends, xdata, true_ydata):
    """
    Histogram of the MSEs between each walker curve and the true curve
    """
    len_data = len(true_ydata)
    mse = []
    for params in walker_ends:
        ydata = func(xdata, *params)
        mse.append(np.sum((ydata - true_ydata) ** 2) / len_data)

    fig, ax = plt.subplots()
    ax.hist(mse, bins=10, density=False, color=theme_colors[0])
    ax.set(xlabel="MSE", ylabel="Count", ylim=(0, len(walker_ends)))
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_curves_mse.png", bbox_inches="tight")
    plt.close()


def plot_walker_gameparams(save_loc, walker_ends, true_game_params, fit_params):
    """
    Histograms of the game parameters the walkers ended on.
    """
    vals_to_plot = [true_game_params[i] for i in range(len(fit_params)) if fit_params[i] == 1]
    params_to_plot = [param_names[i] for i in range(len(fit_params)) if fit_params[i] == 1]
    df = pd.DataFrame(walker_ends, columns=param_names)
    fig, ax = plt.subplots(
        1, len(params_to_plot), figsize=(4 * len(params_to_plot), 4), layout="constrained"
    )
    for i, param in enumerate(params_to_plot):
        ax[i].hist(df[param], bins=10, color=theme_colors[0])
        ax[i].axvline(vals_to_plot[i], color="black", linestyle="dashed")
        ax[i].set(title=param, ylim=(0, len(walker_ends)))
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/mcmc_gameparams.png", bbox_inches="tight")
    plt.close()


def plot_true_curve(save_loc, params, xdata, ydata):
    """
    Visualize the true curve.
    """
    title = [f"{param_names[i]}={params[i]}" for i in range(len(params))]
    fig, ax = plt.subplots(figsize=(4, 4))
    classified_game = classify_game(params[2], params[3], params[4])
    ax.plot(xdata, ydata, color=game_colors[classified_game], linewidth=3)
    ax.set(title=" ".join(title))
    fig.supxlabel("Fraction Mutant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/curve.png", bbox_inches="tight")


def plot_all(save_loc, fp, walker_ends, xdata, ydata, params, fit_params):
    plot_true_curve(save_loc, params, xdata, ydata)
    plot_walker_curves(save_loc, fp, walker_ends, xdata, ydata)
    plot_walker_curve_mse(save_loc, fp, walker_ends, xdata, ydata)
    plot_walker_gamespace(save_loc, walker_ends, params[2:5])
    plot_walker_pairplot(save_loc, walker_ends, fit_params)
    plot_walker_gameparams(save_loc, walker_ends, params, fit_params)


def plot_trace(save_loc, sampler, true_params, fit_params):
    ndim = len([x for x in fit_params if x == 1])
    fig, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    i = 0
    a = 0
    for fit_param in fit_params:
        if fit_param == 1:
            ax[a].plot(sampler.get_chain()[:, :, a], alpha=0.5)
            ax[a].axhline(true_params[i], color="black", ls="--")
            ax[a].set_ylabel(param_names[i])
            a += 1
        i += 1
    ax[-1].set_xlabel("Step Number")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/trace.png")
    plt.close()
