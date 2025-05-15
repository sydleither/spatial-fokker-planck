"""
Use MCMC to find fokker-planck curves that fit multiple parameter sets
"""

import sys

import numpy as np
import pandas as pd

from common import calculate_sigma, get_data_path, get_regime_amws, get_regime_awms
from fokker_planck import FokkerPlanck, param_names
from mcmc_utils import mcmc, plot_paramsweep


def regime_distances(mu, awm, amw, sm):
    """
    Get the distance the parameter set is from each MMMM regime
    """
    sigma = calculate_sigma(mu, awm, amw, sm)
    if sigma in (-np.inf, 0):
        return {"maintenance": -1, "masking": -1, "mirroring": -1, "mimicry": -1}
    if sigma > 0:
        awm_regimes = get_regime_awms(mu, amw, sm, sigma)
        distances = {k:abs(v-awm) for k,v in awm_regimes.items()}
    else:
        amw_regimes = get_regime_amws(mu, awm, sm, sigma)
        distances = {k:abs(v-amw) for k,v in amw_regimes.items()}
    return distances


def main(params):
    """
    Given N and mu, iterate over amw, amw, and sm
    """
    n = int(params[0])
    mu = float(params[1])

    fp = FokkerPlanck(n, mu).fokker_planck_log
    xdata = np.linspace(0.01, 0.99, n)
    len_data = len(xdata)

    a_vals = np.round(np.arange(-0.5, 0.51, 0.01), 2)
    sm_vals = [0.025, 0.05, 0.075]
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                ydata = fp(xdata, awm, amw, sm)
                walker_ends = np.array(mcmc(fp, xdata, ydata))
                param_distances = np.linalg.norm(walker_ends - np.array([awm, amw, sm]), axis=1)
                mse = []
                for walker_params in walker_ends:
                    mse.append(np.sum((ydata - fp(xdata, *walker_params)) ** 2) / len_data)
                regimes = regime_distances(mu, awm, amw, sm)
                data.append(
                    {
                        "awm": awm,
                        "amw": amw,
                        "sm": sm,
                        "mean_param_distance": np.mean(param_distances),
                        "var_param_distance": np.var(param_distances),
                        "min_param_distance": np.min(param_distances),
                        "mean_curve_mse": np.mean(mse),
                        "var_curve_mse": np.var(mse),
                        "min_curve_mse": np.min(mse),
                    } | regimes
                )

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    metrics = [x for x in data[0].keys() if x not in ["awm", "amw", "sm"]]
    df = pd.DataFrame(data)
    for metric in metrics:
        plot_paramsweep(save_loc, df, metric)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
