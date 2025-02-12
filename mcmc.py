'''
Run MCMC on input params
Based on https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
'''
import sys

import emcee
import numpy as np
import matplotlib.pyplot as plt

from common import fokker_planck


n = 0
mu = 0
def fokker_planck_fixed(x, awm, amw, s):
    '''
    The Fokker-Planck equation with a fixed N and mu
    '''
    return fokker_planck(x, n, mu, awm, amw, s)


def lnlike(params, x, y, yerr):
    '''
    chi-squared log likelihood
    '''
    return -0.5*np.sum(((y-fokker_planck_fixed(x, *params))/yerr)**2)


def lnprior(params):
    '''
    Enforce parameter boundaries (priors)
    '''
    params = np.array(params)
    if np.any(params > 1) or np.any(params < -1):
        return -np.inf
    return 0.0


def lnprob(params, x, y, yerr):
    '''
    Return likelihood if the priors are satisfied
    '''
    lp = lnprior(params)
    if lp != 0.0:
        return -np.inf
    return lnlike(params, x, y, yerr)


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
    niter = 100
    initial = (0, 0, 0)
    ndim = len(true_params)
    p0 = [np.array(initial) + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerr))
    sampler.run_mcmc(p0, niter)
    samples = sampler.flatchain
    probs = sampler.flatlnprobability

    order = np.flipud(probs.argsort())
    top_samples = samples[order]
    unique_samples = []

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata/max(ydata), color="black", ls="--")
    for i in range(100):
        sample = top_samples[i]
        is_unique = True
        for unique_sample in unique_samples:
            if np.allclose(sample, unique_sample, rtol=1e-3, atol=1e-5):
                is_unique = False
                break
        if not is_unique:
            continue
        y = fokker_planck_fixed(xdata, *sample)
        ax.plot(xdata, y/max(y), alpha=1, label=[round(x, 2) for x in sample])
        unique_samples.append(sample)
        if len(unique_samples) == 10:
            break
    fig.legend()
    fig.savefig("mcmc.png")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide N, mu, awm, amw, and sm as arguments.")
    else:
        main(sys.argv[1:])
