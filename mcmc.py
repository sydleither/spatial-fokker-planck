'''
Run MCMC on input params
Based on https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
'''
import sys

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import fokker_planck, param_names


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


def plot_final_walker_params(walker_ends, file_name, color1="xkcd:pink", color2="xkcd:rose"):
    '''
    Pairplot of the parameters the walkers ended on.
    '''
    df = pd.DataFrame(walker_ends, columns=["awm", "amw", "sm"])
    g = sns.pairplot(df, diag_kind="kde", plot_kws={"color":color1}, diag_kws={"color":color2})
    g.map_lower(sns.kdeplot, levels=4, color=color2)
    plt.savefig(f"mcmc_{file_name}.png", transparent=True)
    plt.close()


def get_best_params(walker_ends, xdata, true_ydata):
    '''
    Filter the walker end params to only be ones that best fit the true curve.
    '''
    new_ends = []
    for params in walker_ends:
        ydata = fokker_planck_fixed(xdata, *params)
        res1 = np.sum((true_ydata/max(true_ydata) - ydata/max(ydata))**2)
        res2 = np.sum((true_ydata - ydata)**2)
        if res1 < 1e-5:
            new_ends.append(params)
        print(f"{[round(x,2) for x in params]}\t{res1}\t{res2}")
    return new_ends


def plot_curves(walker_ends, xdata, true_ydata, file_name):
    '''
    Visualize curves resulting from walker end parameters.
    '''
    fig, ax = plt.subplots()
    for params in walker_ends:
        ydata = fokker_planck_fixed(xdata, *params)
        ax.plot(xdata, ydata, alpha=0.5, label=[round(x, 2) for x in params])
    ax.plot(xdata, true_ydata, color="black", ls="--")
    fig.legend(bbox_to_anchor=(1.2, 1))
    fig.patch.set_alpha(0)
    fig.savefig(f"mcmc_{file_name}.png", bbox_inches="tight")


def main(params):
    '''
    Given a set of params (N, mu, awm, amw, sm)
    Generate a distribution using Fokker-Planck equation with those params
    Give the distribution and Fokker-Planck equation to MCMC
    '''
    global n, mu
    params = [float(x) for x in params]
    n = int(params[0])
    mu = params[1]
    true_params = params[2:]
    xdata = np.linspace(0.01, 0.99, n)
    ydata = fokker_planck_fixed(xdata, *true_params)
    yerr = 0.05*ydata

    nwalkers = 50
    niter = 200
    initial = (0, 0, 0)
    ndim = len(true_params)
    p0 = [np.array(initial) + 0.1*np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerr))
    sampler.run_mcmc(p0, niter)
    walker_ends = sampler.get_chain(discard=niter-1)[0,:,:]

    title = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    plot_final_walker_params(walker_ends, f"all_{title}")
    best_walker_ends = get_best_params(walker_ends, xdata, ydata)
    if len(best_walker_ends) > 0:
        plot_final_walker_params(best_walker_ends, f"best_{title}")
    plot_curves(walker_ends, xdata, ydata, f"curves_{title}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
