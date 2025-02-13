'''
Use MCMC to find fokker-planck curves that fit multiple parameter sets
'''
import sys

import emcee
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import pandas as pd

from common import fokker_planck


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


def main(params):
    '''
    Given N and mu, iterate over amw, amw, and sm
    To see if any resulting curves have multimodal param sets that fit
    '''
    global n, mu
    n = int(params[0])
    mu = float(params[1])
    xdata = np.linspace(0.01, 0.99, n)

    nwalkers = 50
    niter = 200
    initial = (0, 0, 0)
    ndim = 3
    p0 = [np.array(initial) + 0.1*np.random.randn(ndim) for _ in range(nwalkers)]

    a_vals = np.arange(-0.1, 0.11, 0.01)
    sm_vals = np.arange(-0.05, 0.051, 0.05)
    data = []
    for awm in a_vals:
        awm = round(awm, 2)
        for amw in a_vals:
            amw = round(amw, 2)
            for sm in sm_vals:
                sm = round(sm, 2)
                print(n, mu, awm, amw, sm)
                ydata = fokker_planck_fixed(xdata, awm, amw, sm)
                yerr = 0.05*ydata
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerr))
                sampler.run_mcmc(p0, niter)
                walker_ends = sampler.get_chain(discard=niter-1)[0,:,:]
                var = np.mean(np.var(np.array(walker_ends), axis=0))
                data.append({"awm":awm, "amw":amw, "sm":sm, "var":var})

    df = pd.DataFrame(data)
    sms = df["sm"].unique()
    min_var = df["var"].min()
    max_var = df["var"].max()
    fig, ax = plt.subplots(1, len(sms), figsize=(5*len(sms), 4))
    cmap = plt.get_cmap("Greens")
    norm = BoundaryNorm(np.linspace(min_var, max_var, 10), cmap.N)
    for i,sm in enumerate(sms):
        df_sm = df[df["sm"] == sm]
        ax[i].scatter(df_sm["amw"], df_sm["awm"], c=df_sm["var"], cmap=cmap, norm=norm)
        ax[i].set(title=f"sm={sm}")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax.ravel().tolist(), shrink=0.95)
    fig.supxlabel("amw")
    fig.supylabel("awm")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"mcmc_var_N={n}_mu={mu}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
