'''
Run MCMC on input data
Based on https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
'''
from random import choices
import sys

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (calculate_fp_params, classify_game,
                    fokker_planck, game_colors)


n = 0
mu = 0
def fokker_planck_fixed(x, awm, amw, sm):
    '''
    The Fokker-Planck equation with a fixed N and mu
    '''
    return fokker_planck(x, n, mu, awm, amw, sm)


def lnprob(params, x, y, yerr):
    '''
    Return chi-squared log likelihood if the priors are satisfied
    '''
    params = np.array(params)
    if np.any(params > 1) or np.any(params < -1):
        return -np.inf
    return -0.5*np.sum(((y-fokker_planck_fixed(x, *params))/yerr)**2)


def plot_walker_gamespace(walker_ends, file_name, true_game_params=None):
    '''
    Plots of the final walker params on the Fokker-Planck transformed and normal game spaces.
    '''
    df = pd.DataFrame(walker_ends, columns=["awm", "amw", "sm"])
    df["awm/sm"] = df["awm"] / df["sm"]
    df["amw/sm"] = df["amw"] / df["sm"]
    df[["game", "a", "b", "c", "d"]] = df.apply(
        lambda x: classify_game(x["awm"], x["amw"], x["sm"], True), axis=1, result_type="expand"
    )
    df["c-a"] = df["c"] - df["a"]
    df["b-d"] = df["b"] - df["d"]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.scatterplot(data=df, x="amw/sm", y="awm/sm",
                    hue="game", legend=False, ax=ax[0],
                    hue_order=game_colors.keys(),
                    palette=game_colors.values())
    sns.kdeplot(data=df, x="amw/sm", y="awm/sm", color="gray", alpha=0.3, ax=ax[0])
    ax[0].axhline(1, color="black", lw=0.5)
    ax[0].axvline(-1, color="black", lw=0.5)

    sns.scatterplot(data=df, x="c-a", y="b-d",
                    hue="game", legend=False, ax=ax[1],
                    hue_order=game_colors.keys(),
                    palette=game_colors.values())
    sns.kdeplot(data=df, x="c-a", y="b-d", color="gray", alpha=0.3, ax=ax[1])
    ax[1].axhline(0, color="black", lw=0.5)
    ax[1].axvline(0, color="black", lw=0.5)

    if true_game_params is not None:
        a, b, c, d = true_game_params
        awm, amw, sm = calculate_fp_params(a, b, c, d)
        ax[0].scatter([amw/sm], [awm/sm], marker="*", color="black")
        ax[1].scatter([c-a], [b-d], marker="*", color="black")

    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"mcmc_gamespace_{file_name}.png", bbox_inches="tight")
    plt.close()


def plot_curves(walker_ends, xdata, true_ydata, file_name):
    '''
    Visualize curves resulting from walker end parameters.
    '''
    fig, ax = plt.subplots()
    for params in walker_ends:
        ydata = fokker_planck_fixed(xdata, *params)
        game = classify_game(*params)
        ax.plot(xdata, ydata, alpha=0.5, color=game_colors[game])
    ax.plot(xdata, true_ydata, color="black", ls="--")
    ax.set(xlabel="Fraction Mutant", ylabel="Probability Density")
    fig.patch.set_alpha(0)
    fig.savefig(f"mcmc_curves_{file_name}.png", bbox_inches="tight")


def create_sfp_dist(df, bins, sample_length, num_samples=1000):
    '''
    Read in data, create spatial fokker-planck distribution.
    '''
    s_coords = df.loc[df["type"] == "sensitive"][["x", "y"]].values
    r_coords = df.loc[df["type"] == "resistant"][["x", "y"]].values

    dims = range(len(s_coords[0]))
    max_dims = [max(np.max(s_coords[:, i]), np.max(r_coords[:, i])) for i in dims]
    dim_vals = [choices(range(0, max_dims[i]-sample_length), k=num_samples) for i in dims]
    fs_counts = []
    for s in range(num_samples):
        ld = [dim_vals[i][s] for i in dims]
        ud = [ld[i]+sample_length for i in dims]
        subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
        subset_s = np.sum(np.all(subset_s, axis=0))
        subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
        subset_r = np.sum(np.all(subset_r, axis=0))
        subset_total = subset_s + subset_r
        if subset_total == 0:
            continue
        fs_counts.append(subset_r/subset_total)

    hist, _ = np.histogram(fs_counts, bins=bins)
    hist = hist / max(hist)
    return hist


def main(args):
    '''
    Given a data path, N, and mu
    Generate a spatial subsample distribution using the data
    Fit Fokker-Planck to the distribution using MCMC
    '''
    global n, mu
    n = int(args[0])
    mu = float(args[1])

    data_path = args[2]
    source, sample_id = data_path.split("/")[-1].split(" ")
    sample_id = sample_id[:-4]
    df = pd.read_csv(data_path)
    payoff_df = pd.read_csv("data/payoff.csv")
    payoff_df["sample"] = payoff_df["sample"].astype(str)
    payoff_df = payoff_df[(payoff_df["source"] == source) & (payoff_df["sample"] == sample_id)]
    true_game_params = payoff_df[["a", "b", "c", "d"]].values[0]

    bin_size = 10
    xdata = np.linspace(0.01, 0.99, bin_size)
    ydata = create_sfp_dist(df, np.linspace(0, 1, bin_size+1), 3)
    yerr = 0.05*ydata

    nwalkers = 50
    niter = 500
    initial = (0, 0, 0)
    ndim = 3
    p0 = [np.array(initial) + 0.1*np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerr))
    sampler.run_mcmc(p0, niter)
    walker_ends = sampler.get_chain(discard=niter-1)[0,:,:]

    title = f"{source} {sample_id}"
    plot_walker_gamespace(walker_ends, title, true_game_params)
    plot_curves(walker_ends, xdata, ydata, title)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide N, mu, and the data path.")
    else:
        main(sys.argv[1:])
