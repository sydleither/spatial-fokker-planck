import emcee
import numpy as np


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

    return walker_ends
