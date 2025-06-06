"""
Use MCMC to self-fit fokker-planck curves across a range of true parameters
"""
import sys

import numpy as np
import pandas as pd

from common import calculate_sigma, classify_game, get_data_path, get_regime_amws, get_regime_awms
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
        distances = {k: abs(v - awm) for k, v in awm_regimes.items()}
    else:
        amw_regimes = get_regime_amws(mu, awm, sm, sigma)
        distances = {k: abs(v - amw) for k, v in amw_regimes.items()}
    return distances


def parameter_distances(true_ends, walker_ends):
    differences = {}
    for p, param_name in enumerate(param_names):
        walkers_param = walker_ends[:, p]
        walkers_param_distance = np.abs(walkers_param - true_ends[p])
        differences[f"mean_{param_name}_diff"] = np.mean(walkers_param_distance)
        differences[f"var_{param_name}_diff"] = np.var(walkers_param_distance)
    return differences


def quadrant_classification(true_params, walker_ends):
    game_matches = []
    true_game = classify_game(*true_params[2:5])
    for walker_end in walker_ends:
        walker_game = classify_game(*walker_end[2:5])
        game_matches.append(1 if walker_game == true_game else 0)
    game_matches = sum(game_matches) / len(game_matches)
    return {"correct_game_classifications": game_matches}


def main(params):
    """
    Iterate over amw, amw, and sm. Calculate quality of MCMC fit.
    """
    n = int(params[0])
    mu = float(params[1])

    fp = FokkerPlanck().fokker_planck_log
    xdata = np.linspace(0.01, 0.99, 100)
    len_data = len(xdata)

    a_vals = np.round(np.arange(-0.55, 0.551, 0.1), 2)
    sm_vals = np.round(np.arange(0.05, 0.5, 0.1), 2)
    data = []
    for awm in a_vals:
        for amw in a_vals:
            for sm in sm_vals:
                true_ydata = fp(xdata, n, mu, awm, amw, sm, 1)
                true_params = np.array([n, mu, awm, amw, sm, 1])
                walker_ends = np.array(mcmc(fp, xdata, true_ydata))
                #param_pct_errors = np.abs(walker_ends - true_params) / ((np.abs(true_params)+np.abs(walker_ends))/2)
                idv_param_distances = parameter_distances(true_params, walker_ends)
                mse = []
                for walker_params in walker_ends:
                    walker_ydata = fp(xdata, *walker_params)
                    mse.append(np.sum((true_ydata - walker_ydata) ** 2) / len_data)
                regimes = regime_distances(mu, awm, amw, sm)
                game_classifications = quadrant_classification(true_params, walker_ends)
                data.append(
                    {
                        "N": n,
                        "mu": mu,
                        "awm": awm,
                        "amw": amw,
                        "sm": sm,
                        #"mean_param_error": np.mean(param_pct_errors),
                        #"var_param_error": np.var(param_pct_errors),
                        "mean_curve_mse": np.mean(mse),
                        "var_curve_mse": np.var(mse),
                        "mean_probability_density": np.mean(true_ydata),
                    }
                    | regimes
                    | idv_param_distances
                    | game_classifications
                )

    params_str = "_".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
    save_loc = get_data_path("self", params_str)
    metrics = [x for x in data[0].keys() if x not in param_names]
    df = pd.DataFrame(data)
    for metric in metrics:
        plot_paramsweep(save_loc, df, metric)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide N and mu as arguments.")
    else:
        main(sys.argv[1:])
