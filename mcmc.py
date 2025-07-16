import emcee
import numpy as np

from pdfs import param_names


INITIAL = {
    "n": 100,
    "mu": 0.001,
    "awm": 0.0,
    "amw": 0.0,
    "sm": 0.1,
    "c": 0.1
}

BOUNDS = {
    "n": (1, 1000),
    "mu": (0, 0.1),
    "awm": (-1, 1),
    "amw": (-1, 1),
    "sm": (0, 1),
    "c": (0, 1)
}


def lnprob(params, func, x, y, set_params):
    """
    Return chi-squared log likelihood, if the priors are satisfied.
    """
    i = 0
    full_params = []
    for param_name in INITIAL:
        if param_name in set_params:
            full_params.append(set_params[param_name])
        else:
            lower, upper = BOUNDS[param_name]
            param_val = params[i]
            if param_val < lower or param_val > upper:
                return -np.inf
            full_params.append(param_val)
            i += 1
    return -0.5 * np.sum((y - func(x, *full_params))**2)


def mcmc(func, xdata, ydata, true_params, fit_params, nwalkers=100, niter=1000, return_sampler=False):
    """
    Run MCMC over given params on true xdata, ydata and return walker end locations.
    """
    # Record which parameters should not vary during fitting
    set_params = {}
    for i in range(len(fit_params)):
        if fit_params[i] == 0:
            set_params[param_names[i]] = true_params[i]

    # Record which parameters should vary and set initial values for each walker
    params_to_fit = [x for x in param_names if x not in set_params]
    initial = [INITIAL[param_name] for param_name in params_to_fit]
    ndim = len(params_to_fit)
    p0 = []
    for _ in range(nwalkers):
        walker = []
        for val in initial:
            sd = val / 10
            if sd == 0:
                sd = 0.1
            walker.append(np.random.normal(loc=val, scale=sd))
        p0.append(walker)

    # Run MCMC
    lnprob_args = (func, xdata, ydata, set_params)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args)
    sampler.run_mcmc(p0, niter)
    walker_ends = sampler.get_chain(discard=niter - 1)[0, :, :]

    # Insert non-varying parameters into walkers
    for i in range(len(fit_params)):
        if fit_params[i] == 0:
            walker_ends = [np.insert(x, i, true_params[i]) for x in walker_ends]

    if return_sampler:
        return sampler, walker_ends
    return walker_ends
